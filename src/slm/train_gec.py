"""Fine-tuning script for GEC (Grammatical Error Correction) task."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from slm.data_gec import GECDataCollator, SPECIAL_TOKENS, load_gec_dataset
from slm.data_gec_filtered import load_filtered_gec_dataset
from slm.utils import format_whitepaper_entry, load_config, set_seed

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train_gec(config: dict) -> None:
    """Run GEC fine-tuning from a config dict."""
    seed = config.get("seed", 42)
    set_seed(seed)

    experiment_name = config.get("experiment_name", "gec_finetune")
    output_dir = Path(config.get("output_dir", "./outputs")) / experiment_name

    logger.info("=== GEC Experiment: %s ===", experiment_name)

    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", config["model_name"])
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    logger.info("Added %d special tokens: %s", num_added, SPECIAL_TOKENS)

    # Load model
    init_from = config.get("init_from", config["model_name"])
    logger.info("Loading model from %s", init_from)
    model = AutoModelForCausalLM.from_pretrained(init_from)

    # Resize embeddings for new special tokens
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized embeddings to %d", len(tokenizer))

    logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

    # Load dataset
    pretokenized = config.get("pretokenized_dataset")
    if pretokenized:
        logger.info("Loading pre-tokenized dataset: %s", pretokenized)
        ds = hf_load_dataset(pretokenized)
        datasets = {split: ds[split] for split in ds}

        def pad_collator(features):
            max_len = max(len(f["input_ids"]) for f in features)
            pad_id = tokenizer.pad_token_id
            batch_ids, batch_mask, batch_labels = [], [], []
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                batch_ids.append(f["input_ids"] + [pad_id] * pad_len)
                batch_mask.append(f["attention_mask"] + [0] * pad_len)
                batch_labels.append(f["labels"] + [-100] * pad_len)
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

        data_collator = pad_collator
    else:
        dataset_name = config.get("dataset_name", "saken-tukenov/sozkz-corpus-synthetic-kk-gec-v1")
        error_type = config.get("error_type")
        if error_type:
            logger.info("Loading filtered GEC dataset: %s (type=%s)", dataset_name, error_type)
            datasets = load_filtered_gec_dataset(
                error_type=error_type,
                dataset_name=dataset_name,
                identity_ratio=config.get("identity_ratio", 0.2),
                seed=seed,
            )
        else:
            logger.info("Loading GEC dataset: %s", dataset_name)
            datasets = load_gec_dataset(
                dataset_name=dataset_name,
                identity_ratio=config.get("identity_ratio", 0.2),
                seed=seed,
                val_ratio=config.get("val_ratio", 0.05),
                test_ratio=config.get("test_ratio", 0.05),
            )

        max_seq_length = config.get("max_seq_length", 512)
        data_collator = GECDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # Optionally truncate
    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    if max_train and max_train < len(datasets["train"]):
        datasets["train"] = datasets["train"].select(range(max_train))
    if max_eval and max_eval < len(datasets["validation"]):
        datasets["validation"] = datasets["validation"].select(range(max_eval))

    logger.info("Train: %d, Validation: %d", len(datasets["train"]), len(datasets["validation"]))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)) if "warmup_steps" not in config else 0.0,
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        logging_dir=str(Path(config.get("logging_dir", "./logs")) / experiment_name),
        logging_steps=config.get("logging_steps", 50),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 1000),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 1000),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.get("report_to", "tensorboard"),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume from checkpoint
    checkpoint = None
    if (output_dir / "checkpoint-last").exists():
        checkpoint = str(output_dir / "checkpoint-last")
    elif config.get("resume_from_checkpoint"):
        checkpoint = config["resume_from_checkpoint"]

    logger.info("Starting GEC training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model + tokenizer
    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info("Eval results: %s", eval_results)

    # Log metrics
    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "eval_loss": eval_results.get("eval_loss", 0),
        "eval_perplexity": 2 ** eval_results.get("eval_loss", 0),
        "train_samples": len(datasets["train"]),
        "eval_samples": len(datasets["validation"]),
    }

    whitepaper_path = Path("WHITEPAPER.md")
    if whitepaper_path.exists():
        entry = format_whitepaper_entry(experiment_name, config, metrics)
        with open(whitepaper_path, "a") as f:
            f.write(f"\n\n{entry}\n")
        logger.info("Results appended to WHITEPAPER.md")

    logger.info("=== GEC Training complete: %s ===", experiment_name)


def main():
    parser = argparse.ArgumentParser(description="GEC Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        if args.max_steps <= 50:
            config["eval_strategy"] = "no"
            config["save_strategy"] = "no"

    train_gec(config)


if __name__ == "__main__":
    main()
