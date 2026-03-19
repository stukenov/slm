"""Train dense Llama 500M on A100 80GB.

Single-file script. Downloads datasets from HuggingFace, creates model from scratch, trains, uploads.

Usage:
    # Smoke test (50 steps)
    python train_dense_500m_a100.py --smoke-test

    # Full training
    python train_dense_500m_a100.py

    # Resume from checkpoint
    python train_dense_500m_a100.py --resume
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time

import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
TOKENIZER = "stukenov/sozkz-core-gpt2-200k-kk-base-v1"
DATASETS = [
    "stukenov/sozkz-corpus-tokenized-kk-200k-v1",
    "stukenov/sozkz-corpus-tokenized-kk-multidomain-200k-v1",
    "stukenov/sozkz-corpus-tokenized-enkk-200k-v1",
]
HF_REPO = "stukenov/sozkz-llama-500m-kk-base-v1"
OUTPUT_DIR = "./outputs/exp018_llama_500m_200k"
BLOCK_SIZE = 2048

# Model architecture (~607M params)
MODEL_CONFIG = dict(
    vocab_size=200_019,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    bos_token_id=2,
    eos_token_id=0,
    pad_token_id=1,
)

# A100 80GB training params
BATCH_SIZE = 16
GRAD_ACCUM = 16         # effective batch = 256
LR = 3e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
EPOCHS = 1
EVAL_STEPS = 2000
SAVE_STEPS = 2000
LOGGING_STEPS = 50


def load_data(smoke_test=False):
    target_features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
    })
    train_parts, val_parts = [], []

    for repo in DATASETS:
        logger.info("Loading %s ...", repo)
        try:
            if smoke_test:
                tr = load_dataset(repo, split="train[:1%]")
                va = load_dataset(repo, split="validation[:1%]")
            else:
                ds = load_dataset(repo)
                tr, va = ds["train"], ds["validation"]

            keep = {"input_ids", "labels"}
            tr = tr.remove_columns([c for c in tr.column_names if c not in keep])
            va = va.remove_columns([c for c in va.column_names if c not in keep])
            tr = tr.cast(target_features)
            va = va.cast(target_features)

            logger.info("  train=%d  val=%d", len(tr), len(va))
            train_parts.append(tr)
            val_parts.append(va)
        except Exception as e:
            logger.warning("SKIP %s: %s", repo, e)

    train_ds = concatenate_datasets(train_parts)
    val_ds = concatenate_datasets(val_parts)
    logger.info("Combined: train=%d  val=%d  tokens=%.2fB",
                len(train_ds), len(val_ds), len(train_ds) * BLOCK_SIZE / 1e9)
    return train_ds, val_ds


def create_model():
    config = LlamaConfig(**MODEL_CONFIG)
    model = LlamaForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    unique = sum(p.numel() for p in set(model.parameters()))
    logger.info("Model: %.2fM params (unique %.2fM)", total / 1e6, unique / 1e6)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="50 steps only")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-push", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info("GPU %d: %s  %.1f GB", i, p.name, p.total_memory / 1e9)
    else:
        logger.warning("No CUDA!")

    # Data
    train_ds, val_ds = load_data(smoke_test=args.smoke_test)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = create_model()

    # Training config
    eff_batch = args.batch_size * args.grad_accum
    steps_epoch = math.ceil(len(train_ds) / eff_batch)
    logger.info("Effective batch: %d  Steps/epoch: %d", eff_batch, steps_epoch)

    max_steps = -1
    eval_strategy = "steps"
    save_strategy = "steps"
    log_steps = LOGGING_STEPS

    if args.smoke_test:
        max_steps = 50
        eval_strategy = "no"
        save_strategy = "no"
        log_steps = 1
        logger.info("SMOKE TEST: %d steps", max_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        gradient_checkpointing=True,
        torch_compile=(not args.no_compile),
        logging_steps=log_steps,
        eval_strategy=eval_strategy,
        eval_steps=EVAL_STEPS,
        save_strategy=save_strategy,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=(not args.smoke_test),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    # Resume
    checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            checkpoint = os.path.join(args.output_dir, max(ckpts, key=lambda x: int(x.split("-")[1])))
            logger.info("Resuming from %s", checkpoint)

    # Train
    logger.info("Starting training...")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - t0

    loss = result.metrics.get("train_loss", 0)
    logger.info("Training done in %.1fs (%.1f min). Loss: %.4f", elapsed, elapsed / 60, loss)

    if args.smoke_test:
        tokens = max_steps * eff_batch * BLOCK_SIZE
        logger.info("=== BENCHMARK ===")
        logger.info("Steps: %d  Time: %.1fs", max_steps, elapsed)
        logger.info("Speed: %.2f steps/sec  %.0f tokens/sec", max_steps / elapsed, tokens / elapsed)

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved to %s", final_dir)

    # Eval
    if not args.smoke_test:
        metrics = trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
        logger.info("Eval loss: %.4f  Perplexity: %.2f", metrics["eval_loss"], ppl)

    # Upload
    if not args.no_push and not args.smoke_test:
        logger.info("Uploading to %s ...", HF_REPO)
        model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)
        logger.info("Upload complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
