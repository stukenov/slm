#!/usr/bin/env python3
"""exp035 Stage 2: SFT LoRA on pretrained translation model.

Trains with structured instruction format so model follows direction tags cleanly.
Run AFTER exp035_translate_pretrain.py completes.

Setup:
    pip install transformers datasets accelerate peft huggingface_hub

Run:
    python exp035_translate_sft_lora.py
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
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# --- Config ---
BASE_MODEL = "/root/exp035_translate_pretrain"  # Stage 1 output
DATASET_NAME = "stukenov/ekitil-parallel-kkru-v2"
DATASET_CONFIG = None  # default config
OUTPUT_DIR = "/root/exp035_translate_sft"
HF_REPO = "stukenov/sozkz-mgpt-1.3b-translate-kkru-v1"
MAX_LENGTH = 256
BATCH_SIZE = 32
GRAD_ACCUM = 2  # effective batch = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
SAVE_STEPS = 5000
LOGGING_STEPS = 100
EVAL_STEPS = 5000
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_SAMPLES = 500_000  # 500K pairs -> 1M with both directions


def format_sft_kkru(examples):
    """Instruction format kk->ru."""
    texts = []
    for kk, ru in zip(examples["kk"], examples["ru"]):
        kk, ru = kk.strip(), ru.strip()
        if kk and ru:
            texts.append(
                "### Аудар [KK>RU]:\n" + kk + "\n### Аударма:\n" + ru + "</s>"
            )
        else:
            texts.append("")
    return {"text": texts}


def format_sft_rukk(examples):
    """Instruction format ru->kk."""
    texts = []
    for kk, ru in zip(examples["kk"], examples["ru"]):
        kk, ru = kk.strip(), ru.strip()
        if kk and ru:
            texts.append(
                "### Аудар [RU>KK]:\n" + ru + "\n### Аударма:\n" + kk + "</s>"
            )
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
    log.info("Loading Stage 1 model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    log.info("Loading dataset: %s [%s]", DATASET_NAME, DATASET_CONFIG)
    ds = load_dataset(DATASET_NAME, data_files="kk-ru/*.parquet", split="train")
    if MAX_SAMPLES:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    log.info("Dataset: %d pairs", len(ds))

    # Format both directions
    ds_kkru = ds.map(format_sft_kkru, batched=True, num_proc=8, remove_columns=ds.column_names)
    ds_kkru = ds_kkru.filter(lambda x: len(x["text"]) > 0, num_proc=8)

    ds_rukk = ds.map(format_sft_rukk, batched=True, num_proc=8, remove_columns=ds.column_names)
    ds_rukk = ds_rukk.filter(lambda x: len(x["text"]) > 0, num_proc=8)

    ds_combined = concatenate_datasets([ds_kkru, ds_rukk]).shuffle(seed=42)
    log.info("Combined: %d examples", len(ds_combined))

    for i in range(3):
        log.info("Sample %d: %s", i, ds_combined[i]["text"][:200])

    # Split
    ds_split = ds_combined.train_test_split(test_size=2000, seed=42)
    train_ds, val_ds = ds_split["train"], ds_split["test"]

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
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator,
    )

    log.info("Starting Stage 2 training (SFT LoRA)...")
    trainer.train()

    # Merge LoRA and save
    log.info("Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = trainer.evaluate()
    log.info("Final eval loss: %.4f", metrics["eval_loss"])

    # Upload to HF
    log.info("Uploading to %s", HF_REPO)
    merged.push_to_hub(HF_REPO, private=False)
    tokenizer.push_to_hub(HF_REPO)

    # Test translations
    log.info("Translation test:")
    merged.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(merged.parameters()).device

    tests = [
        "### Аудар [KK>RU]:\nҚазақстан Республикасы — Орталық Азиядағы мемлекет.\n### Аударма:\n",
        "### Аудар [RU>KK]:\nКазахстан — государство в Центральной Азии.\n### Аударма:\n",
        "### Аудар [KK>RU]:\nАбай Құнанбайұлы — ұлы қазақ ақыны, ағартушы, ойшыл.\n### Аударма:\n",
        "### Аудар [RU>KK]:\nАлматы — крупнейший город Казахстана и культурная столица.\n### Аударма:\n",
        "### Аудар [KK>RU]:\nНаурыз мейрамы — қазақ халқының көне мерекесі.\n### Аударма:\n",
        "### Аудар [RU>KK]:\nОбразование является важнейшим направлением государственной политики.\n### Аударма:\n",
    ]
    for prompt in tests:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = merged.generate(
                ids, max_new_tokens=100, do_sample=False,
                repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info("  %s", text[:300])

    log.info("DONE -- model at %s", HF_REPO)


if __name__ == "__main__":
    main()
