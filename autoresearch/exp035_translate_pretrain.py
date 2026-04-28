#!/usr/bin/env python3
"""exp035 Stage 1: Continue pretrain mGPT-1.3B-kazakh on kk-ru parallel data.

Teaches the model bilingual translation mapping via causal LM on interleaved
[KK>RU] and [RU>KK] formatted pairs.

Setup:
    pip install transformers datasets accelerate huggingface_hub

Run:
    python exp035_translate_pretrain.py
"""
import logging
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# --- Config ---
MODEL_NAME = "ai-forever/mGPT-1.3B-kazakh"
DATASET_NAME = "stukenov/ekitil-parallel-kkru-v2"
DATASET_CONFIG = None  # default config, kk-ru parquet files
OUTPUT_DIR = "/root/exp035_translate_pretrain"
MAX_LENGTH = 256
BATCH_SIZE = 32
GRAD_ACCUM = 2  # effective batch = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.02
SAVE_STEPS = 2000
LOGGING_STEPS = 50
EVAL_STEPS = 2000
MAX_SAMPLES = 500_000  # 500K pairs -> 1M with both directions


def format_pairs(examples):
    """Create kk->ru direction."""
    texts = []
    for kk, ru in zip(examples["kk"], examples["ru"]):
        kk, ru = kk.strip(), ru.strip()
        if kk and ru:
            texts.append("[KK>RU] " + kk + " [SEP] " + ru + "</s>")
        else:
            texts.append("")
    return {"text": texts}


def format_pairs_reverse(examples):
    """Create ru->kk direction."""
    texts = []
    for kk, ru in zip(examples["kk"], examples["ru"]):
        kk, ru = kk.strip(), ru.strip()
        if kk and ru:
            texts.append("[RU>KK] " + ru + " [SEP] " + kk + "</s>")
        else:
            texts.append("")
    return {"text": texts}


def tokenize_fn(examples, tokenizer):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    labels = []
    for ids in result["input_ids"]:
        lab = [-100 if tok == tokenizer.pad_token_id else tok for tok in ids]
        labels.append(lab)
    result["labels"] = labels
    return result


def main():
    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    log.info("Model params: %.2fB", sum(p.numel() for p in model.parameters()) / 1e9)

    # Load dataset
    log.info("Loading dataset: %s (kk-ru only)", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, data_files="kk-ru/*.parquet", split="train")
    if MAX_SAMPLES:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    log.info("Dataset size: %d pairs", len(ds))

    # Create both directions
    log.info("Formatting kk->ru direction...")
    ds_kkru = ds.map(format_pairs, batched=True, num_proc=8, remove_columns=ds.column_names)
    ds_kkru = ds_kkru.filter(lambda x: len(x["text"]) > 0, num_proc=8)
    log.info("kk->ru: %d examples", len(ds_kkru))

    log.info("Formatting ru->kk direction...")
    ds_rukk = ds.map(format_pairs_reverse, batched=True, num_proc=8, remove_columns=ds.column_names)
    ds_rukk = ds_rukk.filter(lambda x: len(x["text"]) > 0, num_proc=8)
    log.info("ru->kk: %d examples", len(ds_rukk))

    # Concatenate and shuffle
    ds_combined = concatenate_datasets([ds_kkru, ds_rukk]).shuffle(seed=42)
    log.info("Combined: %d examples", len(ds_combined))

    for i in range(3):
        log.info("Sample %d: %s", i, ds_combined[i]["text"][:200])

    # Split
    ds_split = ds_combined.train_test_split(test_size=2000, seed=42)
    train_ds, val_ds = ds_split["train"], ds_split["test"]
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # Tokenize
    log.info("Tokenizing...")
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer), batched=True, num_proc=8,
        remove_columns=["text"],
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer), batched=True, num_proc=8,
        remove_columns=["text"],
    )
    log.info("Tokenization done.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator,
    )

    log.info("Starting Stage 1 training (continue pretrain)...")
    trainer.train()

    log.info("Saving to %s", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = trainer.evaluate()
    log.info("Final eval loss: %.4f", metrics["eval_loss"])

    # Quick test
    log.info("Quick translation test:")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    tests = [
        "[KK>RU] Қазақстан Республикасы — Орталық Азиядағы мемлекет. [SEP] ",
        "[RU>KK] Казахстан — государство в Центральной Азии. [SEP] ",
        "[KK>RU] Абай Құнанбайұлы — ұлы қазақ ақыны. [SEP] ",
        "[RU>KK] Алматы — крупнейший город Казахстана. [SEP] ",
    ]
    for prompt in tests:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=100, do_sample=False,
                repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info("  %s", text[:250])

    log.info("Stage 1 DONE")


if __name__ == "__main__":
    main()
