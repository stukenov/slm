"""Universal training script using HuggingFace Trainer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from slm.data import prepare_datasets
from slm.utils import format_whitepaper_entry, load_config, set_seed

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(config: dict) -> None:
    """Run training from a config dict."""
    seed = config.get("seed", 42)
    set_seed(seed)

    experiment_name = config.get("experiment_name", "unnamed")
    model_name = config["model_name"]
    dataset_name = config.get("dataset_name", config.get("pretokenized_dataset", "unknown"))
    output_dir = Path(config.get("output_dir", "./outputs")) / experiment_name

    logger.info("=== Experiment: %s ===", experiment_name)
    logger.info("Model: %s", model_name)
    logger.info("Dataset: %s", dataset_name)

    # Load or build tokenizer
    tokenizer_path = config.get("tokenizer_path", model_name)
    tokenizer_mode = config.get("tokenizer_mode", "pretrained")  # pretrained | train | extend

    if tokenizer_mode in ("train", "extend") and not Path(tokenizer_path).exists():
        from slm.tokenizer import extend_tokenizer, train_kazakh_bpe

        if tokenizer_mode == "train":
            logger.info("Training new BPE tokenizer -> %s", tokenizer_path)
            train_kazakh_bpe(
                dataset_name=dataset_name,
                vocab_size=config.get("tokenizer_vocab_size", 32000),
                output_dir=tokenizer_path,
            )
        else:  # extend
            logger.info("Extending tokenizer from %s -> %s", model_name, tokenizer_path)
            extend_tokenizer(
                base_model_name=model_name,
                dataset_name=dataset_name,
                num_new_tokens=config.get("tokenizer_num_new_tokens", 5000),
                output_dir=tokenizer_path,
            )

    logger.info("Tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    init_from = config.get("init_from")
    from_scratch = config.get("from_scratch", False)
    if init_from:
        logger.info("Loading pre-initialized model from %s", init_from)
        model = AutoModelForCausalLM.from_pretrained(init_from)
    elif from_scratch:
        custom_model_config = config.get("model_config")
        if custom_model_config:
            # Build model from custom architecture config
            model_type = custom_model_config.pop("model_type", "llama")
            logger.info("Initializing custom %s model from scratch", model_type)
            model_config = AutoConfig.for_model(model_type, **custom_model_config)
            model = AutoModelForCausalLM.from_config(model_config)
        else:
            logger.info("Initializing model from scratch with config from %s", model_name)
            model_config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(model_config)
    else:
        logger.info("Loading pretrained model %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize embeddings if tokenizer was extended
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing embeddings: %d -> %d", model.get_input_embeddings().weight.shape[0], len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

    # Shared router: re-link all MoE gates to layer 0's gate
    if config.get("shared_router"):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            shared_gate = model.model.layers[0].mlp.gate
            for layer_idx in range(1, len(model.model.layers)):
                if hasattr(model.model.layers[layer_idx].mlp, "gate"):
                    model.model.layers[layer_idx].mlp.gate = shared_gate
            logger.info("Shared router linked across %d layers", len(model.model.layers))

    # Prepare data (cached to disk for reuse across experiments)
    pretokenized = config.get("pretokenized_dataset")
    if pretokenized:
        import os
        import time as _time
        from datasets import load_dataset as _load_ds

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        lock_file = Path("/tmp/_dataset_download_done")

        if world_size > 1 and local_rank != 0:
            # Wait for rank 0 to finish downloading
            logger.info("Rank %d waiting for rank 0 to download dataset...", local_rank)
            while not lock_file.exists():
                _time.sleep(10)
            logger.info("Rank %d: lock file found, loading from cache", local_rank)

        logger.info("Loading pre-tokenized dataset from %s (rank %d)", pretokenized, local_rank)
        ds = _load_ds(pretokenized)

        if world_size > 1 and local_rank == 0:
            lock_file.touch()
            logger.info("Rank 0: dataset ready, created lock file")

        datasets = {"train": ds["train"], "validation": ds["validation"]}

        # Concatenate extra pretokenized datasets
        extra_datasets = config.get("extra_pretokenized_datasets", [])
        for extra_name in extra_datasets:
            logger.info("Loading extra dataset: %s", extra_name)
            extra_ds = _load_ds(extra_name)
            from datasets import concatenate_datasets
            datasets["train"] = concatenate_datasets([datasets["train"], extra_ds["train"]])
            if "validation" in extra_ds:
                datasets["validation"] = concatenate_datasets([datasets["validation"], extra_ds["validation"]])
        if extra_datasets:
            logger.info("Combined train: %d, validation: %d",
                        len(datasets["train"]), len(datasets["validation"]))
    else:
        datasets = prepare_datasets(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            block_size=config.get("block_size", 512),
            val_ratio=config.get("val_ratio", 0.05),
            seed=seed,
            num_proc=config.get("dataloader_num_workers", 4),
            cache_dir=config.get("data_cache_dir", "./data_cache"),
            dataset_split=config.get("dataset_split"),
        )

    # Optionally truncate datasets (for smoke tests)
    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    if max_train and max_train < len(datasets["train"]):
        logger.info("Truncating train: %d -> %d", len(datasets["train"]), max_train)
        datasets["train"] = datasets["train"].select(range(max_train))
    if max_eval and max_eval < len(datasets["validation"]):
        logger.info("Truncating eval: %d -> %d", len(datasets["validation"]), max_eval)
        datasets["validation"] = datasets["validation"].select(range(max_eval))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)) if "warmup_steps" not in config else 0.0,
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        adam_epsilon=float(config.get("adam_epsilon", 1e-8)),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        logging_dir=str(Path(config.get("logging_dir", "./logs")) / experiment_name),
        logging_steps=config.get("logging_steps", 50),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.get("report_to", "tensorboard"),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        seed=seed,
    )

    # Use domain-aware MoE trainer if configured
    if config.get("use_moe_trainer"):
        from slm.moe_trainer import MoEDomainTrainer
        logger.info("Using MoEDomainTrainer (domain_curriculum=%s)", config.get("domain_curriculum", True))
        trainer = MoEDomainTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            processing_class=tokenizer,
            domain_curriculum=config.get("domain_curriculum", True),
        )
    else:
        trainer = Trainer(
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

    logger.info("Starting training...")
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
        "eval_perplexity": 2 ** eval_results.get("eval_loss", 0),
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

    logger.info("=== Training complete: %s ===", experiment_name)


def main():
    parser = argparse.ArgumentParser(description="SLM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps (for smoke tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        # Disable eval during smoke test
        if args.max_steps <= 50:
            config["eval_strategy"] = "no"
            config["save_strategy"] = "no"

    train(config)


if __name__ == "__main__":
    main()
