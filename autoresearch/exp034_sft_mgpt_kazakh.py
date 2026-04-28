#!/usr/bin/env python3
"""exp034: SFT fine-tune mGPT-1.3B-kazakh on AmanMussa/kazakh-instruction-v2.

Setup:
    pip install transformers datasets accelerate peft wandb

Run:
    python exp034_sft_mgpt_kazakh.py

Model: ai-forever/mGPT-1.3B-kazakh (1.4B, GPT-2 arch, PPL kk=2.0)
Dataset: AmanMussa/kazakh-instruction-v2 (52K instruction-output pairs)
"""
import os
import json
import logging
import torch
from datasets import load_dataset
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
DATASET_NAME = "AmanMussa/kazakh-instruction-v2"
OUTPUT_DIR = "/root/exp034_mgpt_sft"
HF_REPO = "stukenov/sozkz-mgpt-1.3b-kk-instruct-v1"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8  # effective batch = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
SAVE_STEPS = 500
LOGGING_STEPS = 50


def format_prompt(example):
    """Format instruction/input/output into a single training string."""
    instruction = example["instruction"].strip()
    inp = (example.get("input") or "").strip()
    output = example["output"].strip()

    if inp:
        text = "### Нұсқаулық:\n" + instruction + "\n\n### Кіріс:\n" + inp + "\n\n### Жауап:\n" + output + "</s>"
    else:
        text = "### Нұсқаулық:\n" + instruction + "\n\n### Жауап:\n" + output + "</s>"
    return {"text": text}


def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    # Mask padding tokens in labels with -100
    labels = result["input_ids"].copy()
    labels = [-100 if tok == tokenizer.pad_token_id else tok for tok in labels]
    result["labels"] = labels
    return result


def main():
    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    log.info("Model params: %.2fB", sum(p.numel() for p in model.parameters()) / 1e9)

    # Load and format dataset
    log.info("Loading dataset: %s", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split="train")
    log.info("Dataset size: %d examples", len(ds))

    # Format into text
    ds = ds.map(format_prompt, num_proc=4)

    # Show sample
    log.info("Sample:\n%s", ds[0]["text"][:300])

    # Split train/val
    ds_split = ds.train_test_split(test_size=0.02, seed=42)
    train_ds = ds_split["train"]
    val_ds = ds_split["test"]
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # Tokenize
    train_ds = train_ds.map(
        lambda x: tokenize(x, tokenizer),
        num_proc=4,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: tokenize(x, tokenizer),
        num_proc=4,
        remove_columns=val_ds.column_names,
    )

    # Token stats
    train_lengths = [len(x["input_ids"]) for x in train_ds]
    log.info("Token lengths -- mean: %.0f, max: %d, total: %.1fM",
             sum(train_lengths) / len(train_lengths), max(train_lengths),
             sum(train_lengths) / 1e6)

    # Data collator — padding already done in tokenize, just use default
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
    )

    # Training args
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
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    log.info("Starting training...")
    trainer.train()

    # Save final model
    log.info("Saving model to %s", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Final eval
    metrics = trainer.evaluate()
    log.info("Final eval loss: %.4f", metrics["eval_loss"])

    # Upload to HF
    log.info("Uploading to %s", HF_REPO)
    model.push_to_hub(HF_REPO, private=False)
    tokenizer.push_to_hub(HF_REPO)

    # Test generation
    log.info("Test generation:")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    test_prompts = [
        "### Нұсқаулық:\nҚазақстанның астанасы қай қала?\n\n### Жауап:\n",
        "### Нұсқаулық:\nАбай Құнанбайұлы кім?\n\n### Жауап:\n",
        "### Нұсқаулық:\nНаурыз мейрамы туралы айтып беріңіз.\n\n### Жауап:\n",
        "### Нұсқаулық:\nДені сау болу үшін не істеу керек?\n\n### Жауап:\n",
        "### Нұсқаулық:\nПрограммалау тілі Python не үшін қолданылады?\n\n### Жауап:\n",
    ]
    device = next(model.parameters()).device
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=150, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info("\n%s", text[:300])

    log.info("DONE -- model uploaded to %s", HF_REPO)


if __name__ == "__main__":
    main()
