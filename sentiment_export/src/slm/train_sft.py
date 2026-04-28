"""SFT (Supervised Fine-Tuning) training script for instruction-following.

Supports two formats:
- Alpaca: instruction/input/output fields (Kazakh chat template)
- ChatML: messages JSON field (multi-turn conversations)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from slm.utils import load_config, set_seed

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Alpaca format ---

PROMPT_TEMPLATE_WITH_INPUT = """\
### Нұсқаулық:
{instruction}

### Кіріс:
{input}

### Жауап:
"""

PROMPT_TEMPLATE_NO_INPUT = """\
### Нұсқаулық:
{instruction}

### Жауап:
"""


def format_example(instruction: str, input_text: str, output: str) -> tuple[str, str]:
    """Return (prompt, completion) pair."""
    if input_text:
        prompt = PROMPT_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    else:
        prompt = PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)
    return prompt, output


# --- ChatML format ---

CHATML_ROLE_TAGS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
CHATML_END = "<|end|>"
CHATML_SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def format_chatml(messages: list[dict]) -> str:
    """Format a list of message dicts into a ChatML string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        tag = CHATML_ROLE_TAGS.get(role, f"<|{role}|>")
        parts.append(f"{tag}\n{content}\n{CHATML_END}")
    return "\n".join(parts)


def parse_messages(raw: str | list) -> list[dict]:
    """Parse messages from string (JSON) or list."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


@dataclass
class SFTDataCollator:
    """Tokenize on-the-fly and mask prompt tokens in labels.

    Supports Alpaca, ChatML, and Classify formats, detected per-example.
    """

    tokenizer: AutoTokenizer
    max_length: int = 512
    sft_format: str = "auto"
    classify_tag: str = "sentiment"

    def _tokenize_alpaca(self, f: dict) -> tuple[list[int], list[int], list[int]]:
        prompt, completion = format_example(
            f["instruction"], f.get("input", ""), f["output"]
        )
        full_text = prompt + completion + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_encoded = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        prompt_len = len(prompt_encoded["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return input_ids, attention_mask, labels

    def _tokenize_classify(self, f: dict) -> tuple[list[int], list[int], list[int]]:
        tag = self.classify_tag
        prompt = f"<{tag}>{f['text']}</{tag}>\n"
        full_text = prompt + f["label"] + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_encoded = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        prompt_len = len(prompt_encoded["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return input_ids, attention_mask, labels

    def _tokenize_chatml(self, f: dict) -> tuple[list[int], list[int], list[int]]:
        messages = parse_messages(f["messages"])
        full_text = format_chatml(messages) + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Build labels: only train on assistant content (between <|assistant|>\n and \n<|end|>)
        labels = [-100] * len(input_ids)

        # Re-encode per segment to find assistant boundaries
        pos = 0
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            tag = CHATML_ROLE_TAGS.get(role, f"<|{role}|>")

            # Encode the non-assistant prefix: "{tag}\n"
            prefix = f"{tag}\n"
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            pos += len(prefix_ids)

            # Encode content + "\n<|end|>"
            suffix = f"{content}\n{CHATML_END}"
            suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]

            if role == "assistant":
                # Unmask assistant content tokens
                for j in range(len(suffix_ids)):
                    if pos + j < len(labels):
                        labels[pos + j] = input_ids[pos + j]

            pos += len(suffix_ids)

            # Account for "\n" between messages
            sep_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
            pos += len(sep_ids)

        # Unmask final eos token
        if pos < len(labels):
            labels[-1] = input_ids[-1]

        return input_ids, attention_mask, labels

    def _detect_format(self, f: dict) -> str:
        if self.sft_format != "auto":
            return self.sft_format
        if "text" in f and "label" in f:
            return "classify"
        if "messages" in f and f["messages"]:
            return "chatml"
        return "alpaca"

    def __call__(self, features: list[dict]) -> dict:
        batch_ids, batch_mask, batch_labels = [], [], []

        for f in features:
            fmt = self._detect_format(f)
            if fmt == "classify":
                input_ids, attention_mask, labels = self._tokenize_classify(f)
            elif fmt == "chatml":
                input_ids, attention_mask, labels = self._tokenize_chatml(f)
            else:
                input_ids, attention_mask, labels = self._tokenize_alpaca(f)

            batch_ids.append(input_ids)
            batch_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        for i in range(len(batch_ids)):
            pad_len = max_len - len(batch_ids[i])
            batch_ids[i] = batch_ids[i] + [pad_id] * pad_len
            batch_mask[i] = batch_mask[i] + [0] * pad_len
            batch_labels[i] = batch_labels[i] + [-100] * pad_len

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def ensure_chatml_tokens(tokenizer: AutoTokenizer) -> int:
    """Add ChatML special tokens if missing. Returns number of tokens added."""
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in CHATML_SPECIAL_TOKENS if t not in existing]
    if new_tokens:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info("Added %d ChatML tokens: %s", num_added, new_tokens)
        return num_added
    return 0


def train_sft(config: dict) -> None:
    """Run SFT training from a config dict."""
    seed = config.get("seed", 42)
    set_seed(seed)

    experiment_name = config.get("experiment_name", "sft_finetune")
    output_dir = Path(config.get("output_dir", "./outputs")) / experiment_name

    logger.info("=== SFT Experiment: %s ===", experiment_name)

    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", config.get("init_from"))
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    init_from = config["init_from"]
    logger.info("Loading model from %s", init_from)
    model = AutoModelForCausalLM.from_pretrained(init_from)

    # Add ChatML tokens if needed
    sft_format = config.get("sft_format", "auto")
    if sft_format in ("chatml", "auto"):
        num_added = ensure_chatml_tokens(tokenizer)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized embeddings to %d", len(tokenizer))

    logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

    # Load SFT dataset (local path or HuggingFace hub)
    sft_dataset = config["sft_dataset"]
    logger.info("Loading SFT dataset: %s", sft_dataset)
    if Path(sft_dataset).exists():
        from datasets import load_from_disk
        ds = load_from_disk(sft_dataset)
    else:
        ds = load_dataset(sft_dataset)

    train_ds = ds["train"]
    eval_ds = ds.get("validation") or ds.get("test")

    # Optionally truncate
    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    if max_train and max_train < len(train_ds):
        train_ds = train_ds.select(range(max_train))
    if eval_ds and max_eval and max_eval < len(eval_ds):
        eval_ds = eval_ds.select(range(max_eval))

    logger.info("Train: %d, Eval: %s", len(train_ds), len(eval_ds) if eval_ds else "none")

    max_length = config.get("max_length", 512)
    classify_tag = config.get("classify_tag", "sentiment")
    data_collator = SFTDataCollator(
        tokenizer=tokenizer, max_length=max_length, sft_format=sft_format,
        classify_tag=classify_tag,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)) if "warmup_steps" not in config else 0.0,
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        logging_dir=str(Path(config.get("logging_dir", "./logs")) / experiment_name),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        eval_strategy=config.get("eval_strategy", "steps") if eval_ds else "no",
        eval_steps=config.get("eval_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        report_to=config.get("report_to", "tensorboard"),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=config.get("dataloader_pin_memory", False),
        torch_compile=config.get("torch_compile", False),
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume from checkpoint
    checkpoint = None
    if (output_dir / "checkpoint-last").exists():
        checkpoint = str(output_dir / "checkpoint-last")
    elif config.get("resume_from_checkpoint"):
        checkpoint = config["resume_from_checkpoint"]

    logger.info("Starting SFT training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model + tokenizer
    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)

    # Evaluate
    if eval_ds:
        eval_results = trainer.evaluate()
        logger.info("Eval results: %s", eval_results)

    logger.info("=== SFT Training complete: %s ===", experiment_name)


def main():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        if args.max_steps <= 50:
            config["eval_strategy"] = "no"
            config["save_strategy"] = "no"

    train_sft(config)


if __name__ == "__main__":
    main()
