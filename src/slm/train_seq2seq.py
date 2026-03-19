"""T5 Seq2Seq training script for Kazakh language pretraining with span corruption."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from slm.data_seq2seq import T5SpanCorruptionCollator, add_sentinel_tokens, prepare_t5_datasets
from slm.utils import format_whitepaper_entry, load_config, set_seed

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(config: dict) -> None:
    """Run T5 seq2seq training from a config dict."""
    seed = config.get("seed", 42)
    set_seed(seed)

    experiment_name = config.get("experiment_name", "unnamed")
    model_name = config["model_name"]
    dataset_name = config.get("dataset_name", "unknown")
    output_dir = Path(config.get("output_dir", "./outputs")) / experiment_name

    logger.info("=== T5 Experiment: %s ===", experiment_name)
    logger.info("Model: %s", model_name)
    logger.info("Dataset: %s", dataset_name)

    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", model_name)
    logger.info("Tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add sentinel tokens
    tokenizer = add_sentinel_tokens(tokenizer)

    # Build T5 model from scratch
    model_config_dict = config.get("model_config", {})
    model_type = model_config_dict.pop("model_type", "t5")
    logger.info("Initializing %s model from scratch", model_type)
    model_config = AutoConfig.for_model(model_type, **model_config_dict)
    model = T5ForConditionalGeneration(model_config)

    # Resize embeddings to match tokenizer (with sentinels)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info(
            "Resizing embeddings: %d -> %d",
            model.get_input_embeddings().weight.shape[0],
            len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

    # Prepare data
    if config.get("pretokenized"):
        from datasets import load_dataset as _load_ds
        logger.info("Loading pre-tokenized dataset: %s", dataset_name)
        raw = _load_ds(dataset_name)
        datasets = {"train": raw["train"], "validation": raw["validation"]}
        logger.info("Train: %d blocks, Validation: %d blocks", len(datasets["train"]), len(datasets["validation"]))
    else:
        datasets = prepare_t5_datasets(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            block_size=config.get("block_size", 512),
            val_ratio=config.get("val_ratio", 0.05),
            seed=seed,
            num_proc=config.get("dataloader_num_workers", 4),
            cache_dir=config.get("data_cache_dir", "./data_cache"),
            dataset_split=config.get("dataset_split"),
        )

    # Truncate for smoke tests
    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    if max_train and max_train < len(datasets["train"]):
        datasets["train"] = datasets["train"].select(range(max_train))
    if max_eval and max_eval < len(datasets["validation"]):
        datasets["validation"] = datasets["validation"].select(range(max_eval))

    # Span corruption collator
    data_collator = T5SpanCorruptionCollator(
        tokenizer=tokenizer,
        mask_prob=float(config.get("mask_prob", 0.15)),
        mean_noise_span_length=float(config.get("mean_noise_span_length", 3.0)),
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 64),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)) if "warmup_steps" not in config else 0.0,
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "inverse_sqrt"),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        logging_dir=str(Path(config.get("logging_dir", "./logs")) / experiment_name),
        logging_steps=config.get("logging_steps", 25),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 2000),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 2000),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.get("report_to", "tensorboard"),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        seed=seed,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if available
    checkpoint = None
    if (output_dir / "checkpoint-last").exists():
        checkpoint = str(output_dir / "checkpoint-last")
    elif config.get("resume_from_checkpoint"):
        checkpoint = config["resume_from_checkpoint"]

    logger.info("Starting T5 training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info("Eval results: %s", eval_results)

    # Log metrics
    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "eval_loss": eval_results.get("eval_loss", 0),
        "train_samples": len(datasets["train"]),
        "eval_samples": len(datasets["validation"]),
    }

    # Append to whitepaper
    whitepaper_path = Path("WHITEPAPER.md")
    if whitepaper_path.exists():
        entry = format_whitepaper_entry(experiment_name, config, metrics)
        with open(whitepaper_path, "a") as f:
            f.write(f"\n\n{entry}\n")
        logger.info("Results appended to WHITEPAPER.md")

    logger.info("=== T5 Training complete: %s ===", experiment_name)


def main():
    parser = argparse.ArgumentParser(description="T5 Seq2Seq Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        if args.max_steps <= 50:
            config["eval_strategy"] = "no"
            config["save_strategy"] = "no"

    train(config)


if __name__ == "__main__":
    main()
