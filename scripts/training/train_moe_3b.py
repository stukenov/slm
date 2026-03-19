"""Train MoE 3B model (128 experts, top-2, shared router) on Kazakh corpus.

Usage:
    python train_moe_3b.py
    python train_moe_3b.py --batch-size 8 --grad-accum 64   # A100 80GB
    python train_moe_3b.py --batch-size 2 --grad-accum 256  # if OOM
    python train_moe_3b.py --smoke-test                      # quick 50-step test
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
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "stukenov/sozkz-moe-mix-3b-kk-base-v1-init"
TOKENIZER_NAME = "stukenov/sozkz-core-gpt2-50k-kk-base-v1"
DATASET_1 = "stukenov/sozkz-corpus-tokenized-kk-llama50k-v3"
DATASET_2 = "stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1"
HF_PUSH_NAME = "stukenov/sozkz-moe-mix-3b-kk-base-v1"
OUTPUT_DIR = "./outputs/exp017_moe_shared_router_3b"


def load_combined_datasets(smoke_test=False):
    """Load and merge two pre-tokenized datasets."""
    if smoke_test:
        # Load only 1% to save disk and time
        logger.info("SMOKE TEST: loading only 1%% of each dataset")
        ds1_train = load_dataset(DATASET_1, split="train[:1%]")
        ds1_val = load_dataset(DATASET_1, split="validation[:1%]")
        ds2_train = load_dataset(DATASET_2, split="train[:1%]")
        ds2_val = load_dataset(DATASET_2, split="validation[:1%]")
    else:
        logger.info("Loading dataset 1: %s", DATASET_1)
        ds1 = load_dataset(DATASET_1)
        ds1_train, ds1_val = ds1["train"], ds1["validation"]
        logger.info("  Train: %d, Val: %d", len(ds1_train), len(ds1_val))

        logger.info("Loading dataset 2: %s", DATASET_2)
        ds2 = load_dataset(DATASET_2)
        ds2_train, ds2_val = ds2["train"], ds2["validation"]
        logger.info("  Train: %d, Val: %d", len(ds2_train), len(ds2_val))

    # Keep only input_ids and labels
    cols_to_keep = {"input_ids", "labels"}
    for ds in [ds1_train, ds1_val, ds2_train, ds2_val]:
        drop = [c for c in ds.column_names if c not in cols_to_keep]
        if drop:
            ds = ds.remove_columns(drop)

    ds1_train = ds1_train.remove_columns([c for c in ds1_train.column_names if c not in cols_to_keep])
    ds1_val = ds1_val.remove_columns([c for c in ds1_val.column_names if c not in cols_to_keep])
    ds2_train = ds2_train.remove_columns([c for c in ds2_train.column_names if c not in cols_to_keep])
    ds2_val = ds2_val.remove_columns([c for c in ds2_val.column_names if c not in cols_to_keep])

    # Cast ds2 labels int64 -> int32 to match ds1
    target_features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
    })
    ds2_train = ds2_train.cast(target_features)
    ds2_val = ds2_val.cast(target_features)

    train_dataset = concatenate_datasets([ds1_train, ds2_train])
    val_dataset = concatenate_datasets([ds1_val, ds2_val])

    logger.info("Combined Train: %d, Val: %d", len(train_dataset), len(val_dataset))
    logger.info("Total tokens: ~%.1fB", len(train_dataset) * 1024 / 1e9)
    return train_dataset, val_dataset


def load_model_and_tokenizer():
    """Load MoE model and link shared router."""
    logger.info("Loading tokenizer: %s", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    logger.info("Total params (with sharing): %.2fB", total_params / 1e9)
    logger.info("Unique params: %.2fB", unique_params / 1e9)

    # Link shared router across all layers
    shared_gate = model.model.layers[0].mlp.gate
    num_layers = len(model.model.layers)
    for layer_idx in range(1, num_layers):
        if hasattr(model.model.layers[layer_idx].mlp, "gate"):
            model.model.layers[layer_idx].mlp.gate = shared_gate
    logger.info("Shared router linked across %d layers", num_layers)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train MoE 3B")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4 for A100 40GB)")
    parser.add_argument("--grad-accum", type=int, default=128, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--push-to-hub", default=HF_PUSH_NAME, help="HF repo to push final model")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HF Hub")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 50-step test")
    args = parser.parse_args()

    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info("GPU %d: %s, %.1f GB", i, props.name, props.total_memory / 1e9)
    else:
        logger.warning("No CUDA GPU found!")

    # Load data
    train_dataset, val_dataset = load_combined_datasets(smoke_test=args.smoke_test)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Training args
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

    # Train
    logger.info("Starting training...")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - t0
    logger.info("Training done in %.1fs. Loss: %.4f", elapsed, train_result.metrics.get("train_loss", 0))

    if args.smoke_test:
        # Print benchmark results
        steps_done = train_result.metrics.get("train_steps", max_steps)
        samples = steps_done * args.batch_size * args.grad_accum
        tokens = samples * 1024
        logger.info("=== BENCHMARK RESULTS ===")
        logger.info("Steps: %d, Time: %.1fs", steps_done, elapsed)
        logger.info("Speed: %.2f steps/sec, %.0f tokens/sec", steps_done / elapsed, tokens / elapsed)
        logger.info("Estimated full epoch (%.1fM samples): %.1f hours",
                     17.36, 17.36e6 / (samples / elapsed) / 3600)

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)

    # Evaluate
    if not args.smoke_test:
        results = trainer.evaluate()
        logger.info("Eval loss: %.4f, Perplexity: %.2f", results["eval_loss"], 2 ** results["eval_loss"])

    # Push to Hub
    if not args.no_push and not args.smoke_test:
        logger.info("Pushing to %s...", args.push_to_hub)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        logger.info("Upload complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
