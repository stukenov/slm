"""Train dense Llama ~150M on 2x A10 (DDP) — Kazakh corpus.

Architecture (~153M params, Chinchilla-optimal for 3.28B tokens):
  - 200K vocab, hidden=512, intermediate=2048, 12 layers, 8 heads
  - Embedding = 102M (tied), Transformer = 50M

Usage:
    # Smoke test
    torchrun --nproc_per_node=2 train_dense_150m_2xa10.py --smoke-test

    # Full training
    torchrun --nproc_per_node=2 train_dense_150m_2xa10.py
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
]
HF_REPO = "stukenov/sozkz-llama-150m-200k-kk-base-v1"
OUTPUT_DIR = "./outputs/exp019_llama_150m_200k"
BLOCK_SIZE = 2048

# Model architecture (~153M params)
MODEL_CONFIG = dict(
    vocab_size=200_019,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=8,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    bos_token_id=2,
    eos_token_id=0,
    pad_token_id=1,
)

# 2x A10 22GB — 150M model fits easily, maximize batch size
BATCH_SIZE = 2           # per GPU — 200K vocab logits = 1.5GB/sample
GRAD_ACCUM = 32          # effective batch = 2 * 32 * 2 GPUs = 128
LR = 3e-4                # reduced for continued training
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
EPOCHS = 3
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 10


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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info("GPU %d: %s  %.1f GB", i, p.name, p.total_memory / 1e9)

    train_ds, val_ds = load_data(smoke_test=args.smoke_test)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = args.batch_size * args.grad_accum * world_size
    steps_epoch = math.ceil(len(train_ds) / eff_batch)
    logger.info("World size: %d  Effective batch: %d  Steps/epoch: %d", world_size, eff_batch, steps_epoch)

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
        dataloader_pin_memory=True,
        report_to="none",
        push_to_hub=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            checkpoint = os.path.join(args.output_dir, max(ckpts, key=lambda x: int(x.split("-")[1])))
            logger.info("Resuming from %s", checkpoint)

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
        logger.info("Estimated full training (%d steps): %.1f hours",
                     steps_epoch, steps_epoch * (elapsed / max_steps) / 3600)

    # Save
    if local_rank == 0:
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
    if not args.no_push and not args.smoke_test and local_rank == 0:
        logger.info("Uploading to %s ...", HF_REPO)
        model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)
        logger.info("Upload complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
