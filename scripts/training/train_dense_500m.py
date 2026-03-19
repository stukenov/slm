"""Train dense Llama 500M model from scratch on Kazakh corpus.

Usage:
    python train_dense_500m.py
    python train_dense_500m.py --batch-size 8 --grad-accum 32   # A100 80GB
    python train_dense_500m.py --batch-size 2 --grad-accum 128  # RTX 4090
    python train_dense_500m.py --smoke-test                      # quick 50-step test
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

TOKENIZER_NAME = "stukenov/sozkz-core-gpt2-200k-kk-base-v1"
DATASETS = [
    "stukenov/sozkz-corpus-tokenized-kk-200k-v1",
    "stukenov/sozkz-corpus-tokenized-kk-multidomain-200k-v1",
    "stukenov/sozkz-corpus-tokenized-enkk-200k-v1",
]
HF_PUSH_NAME = "stukenov/sozkz-llama-500m-kk-base-v1"
OUTPUT_DIR = "./outputs/exp018_llama_500m_200k"


def load_combined_datasets(smoke_test=False):
    """Load and merge pre-tokenized datasets."""
    train_parts, val_parts = [], []

    target_features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
    })

    for repo in DATASETS:
        logger.info("Loading %s...", repo)
        try:
            if smoke_test:
                ds_train = load_dataset(repo, split="train[:1%]")
                ds_val = load_dataset(repo, split="validation[:1%]")
            else:
                ds = load_dataset(repo)
                ds_train, ds_val = ds["train"], ds["validation"]

            logger.info("  Train: %d, Val: %d", len(ds_train), len(ds_val))

            # Keep only input_ids and labels
            cols_to_keep = {"input_ids", "labels"}
            ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols_to_keep])
            ds_val = ds_val.remove_columns([c for c in ds_val.column_names if c not in cols_to_keep])

            # Cast to uniform type
            ds_train = ds_train.cast(target_features)
            ds_val = ds_val.cast(target_features)

            train_parts.append(ds_train)
            val_parts.append(ds_val)
        except Exception as e:
            logger.warning("Failed to load %s: %s", repo, e)

    train_dataset = concatenate_datasets(train_parts)
    val_dataset = concatenate_datasets(val_parts)

    logger.info("Combined Train: %d, Val: %d", len(train_dataset), len(val_dataset))
    logger.info("Total tokens: ~%.2fB", len(train_dataset) * 2048 / 1e9)
    return train_dataset, val_dataset


def create_model():
    """Create Llama 500M from scratch."""
    config = LlamaConfig(
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
    model = LlamaForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    logger.info("Model params: %.2fM (unique: %.2fM)", total_params / 1e6, unique_params / 1e6)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Dense Llama 500M")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--push-to-hub", default=HF_PUSH_NAME)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info("GPU %d: %s, %.1f GB", i, props.name, props.total_memory / 1e9)
    else:
        logger.warning("No CUDA GPU found!")

    train_dataset, val_dataset = load_combined_datasets(smoke_test=args.smoke_test)

    logger.info("Loading tokenizer: %s", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Creating model from scratch...")
    model = create_model()

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    logger.info("Effective batch: %d, Steps/epoch: %d", effective_batch, steps_per_epoch)

    max_steps = -1
    eval_strategy = "steps"
    save_strategy = "steps"
    if args.smoke_test:
        max_steps = 50
        eval_strategy = "no"
        save_strategy = "no"
        logger.info("SMOKE TEST: %d steps only", max_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50 if not args.smoke_test else 1,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=(not args.smoke_test),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        torch_compile=(not args.no_compile),
        report_to="none",
        push_to_hub=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume from checkpoint
    checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(args.output_dir, latest)
            logger.info("Resuming from %s", checkpoint)

    logger.info("Starting training...")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - t0
    logger.info("Training done in %.1fs. Loss: %.4f", elapsed, train_result.metrics.get("train_loss", 0))

    if args.smoke_test:
        steps_done = train_result.metrics.get("train_steps", max_steps)
        samples = steps_done * args.batch_size * args.grad_accum
        tokens = samples * 2048
        logger.info("=== BENCHMARK RESULTS ===")
        logger.info("Steps: %d, Time: %.1fs", steps_done, elapsed)
        logger.info("Speed: %.2f steps/sec, %.0f tokens/sec", steps_done / elapsed, tokens / elapsed)

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)

    if not args.smoke_test:
        results = trainer.evaluate()
        logger.info("Eval loss: %.4f, Perplexity: %.2f", results["eval_loss"], 2 ** results["eval_loss"])

    if not args.no_push and not args.smoke_test:
        logger.info("Pushing to %s...", args.push_to_hub)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        logger.info("Upload complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
