#!/usr/bin/env python3
"""Step 3: Train a 50M-parameter Llama model from scratch on Kazakh.

Uses pre-tokenized data from HuggingFace Hub — no local preprocessing needed.

Usage:
    # Single GPU
    python train.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py

    # Smoke test (50 steps)
    python train.py --max-steps 50
"""

from __future__ import annotations

import argparse
import logging

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

TOKENIZER = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
DATASET = "saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2"
OUTPUT_DIR = "./output"

MODEL_CONFIG = dict(
    model_type="llama",
    vocab_size=50257,
    hidden_size=512,
    intermediate_size=1344,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,
    max_position_embeddings=1024,
    tie_word_embeddings=True,
    bos_token_id=0,
    eos_token_id=0,
    pad_token_id=1,
)

TRAINING = dict(
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=6e-4,
    weight_decay=0.1,
    warmup_steps=500,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    dataloader_num_workers=4,
    seed=42,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1, help="Override max training steps (for smoke tests)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # ── Tokenizer ───────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", TOKENIZER)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (from scratch) ────────────────────────────────────────────────
    logger.info("Initializing Llama model from scratch")
    model_type = MODEL_CONFIG.pop("model_type")
    config = AutoConfig.for_model(model_type, **MODEL_CONFIG)
    MODEL_CONFIG["model_type"] = model_type  # restore for logging
    model = AutoModelForCausalLM.from_config(config)
    logger.info("Parameters: %.2fM", model.num_parameters() / 1e6)

    # ── Dataset (pre-tokenized) ─────────────────────────────────────────────
    logger.info("Loading pre-tokenized dataset: %s", DATASET)
    ds = load_dataset(DATASET)
    logger.info("Train: %d blocks, Val: %d blocks", len(ds["train"]), len(ds["validation"]))

    # ── Training ────────────────────────────────────────────────────────────
    training_kwargs = {**TRAINING, "output_dir": args.output_dir}
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
        if args.max_steps <= 50:
            training_kwargs["eval_strategy"] = "no"
            training_kwargs["save_strategy"] = "no"

    training_args = TrainingArguments(**training_kwargs)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ────────────────────────────────────────────────────────────────
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    logger.info("Model saved to %s/final", args.output_dir)

    # ── Evaluate ────────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    logger.info("Eval loss: %.4f, Perplexity: %.2f",
                eval_results["eval_loss"], 2 ** eval_results["eval_loss"])


if __name__ == "__main__":
    main()
