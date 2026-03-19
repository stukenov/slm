"""Tokenize dataset with T5 SP tokenizer and upload to HuggingFace.

Usage:
    python scripts/tokenize_and_upload.py \
        --tokenizer saken-tukenov/sozkz-vocab-sp-32k-kk-t5-v1 \
        --dataset saken-tukenov/sozkz-corpus-clean-kk-text-v2 \
        --block_size 512 \
        --hf_repo saken-tukenov/sozkz-seq-t5-50m-kk-base-v1-tokenized \
        --num_proc 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def tokenize_and_group(dataset, tokenizer, block_size: int, num_proc: int):
    """Tokenize text and group into fixed-length blocks."""

    def tokenize_fn(examples):
        out = tokenizer(examples["text"], add_special_tokens=False)
        return {"input_ids": out["input_ids"]}

    log.info("Tokenizing...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_fn(examples):
        # Concatenate all input_ids
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        # Split into blocks
        total = (len(concatenated) // block_size) * block_size
        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total, block_size)
            ],
        }
        return result

    log.info("Grouping into blocks of %d...", block_size)
    grouped = tokenized.map(
        group_fn,
        batched=True,
        num_proc=num_proc,
        desc="Grouping",
    )

    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="saken-tukenov/sozkz-vocab-sp-32k-kk-t5-v1")
    parser.add_argument("--dataset", default="saken-tukenov/sozkz-corpus-clean-kk-text-v2")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--hf_repo", default="saken-tukenov/sozkz-seq-t5-50m-kk-base-v1-tokenized")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load tokenizer
    log.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    log.info("Vocab size: %d", len(tokenizer))

    # Load dataset
    log.info("Loading dataset: %s", args.dataset)
    ds = load_dataset(args.dataset)

    if "validation" not in ds:
        log.info("Splitting train into train/validation (val_ratio=%.2f)", args.val_ratio)
        split = ds["train"].train_test_split(test_size=args.val_ratio, seed=args.seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    log.info("Train: %d, Validation: %d", len(ds["train"]), len(ds["validation"]))

    # Tokenize and group
    result = {}
    for split_name in ["train", "validation"]:
        log.info("Processing %s split...", split_name)
        result[split_name] = tokenize_and_group(
            ds[split_name], tokenizer, args.block_size, args.num_proc,
        )
        log.info("%s: %d blocks of %d tokens", split_name, len(result[split_name]), args.block_size)

    total_tokens = (len(result["train"]) + len(result["validation"])) * args.block_size
    log.info("Total tokens: %d (%.1fM)", total_tokens, total_tokens / 1e6)

    # Upload to HuggingFace
    log.info("Uploading to %s ...", args.hf_repo)
    dd = DatasetDict(result)
    dd.push_to_hub(args.hf_repo, commit_message=f"Tokenized with {args.tokenizer}, block_size={args.block_size}")
    log.info("Done! Dataset at: https://huggingface.co/datasets/%s", args.hf_repo)


if __name__ == "__main__":
    main()
