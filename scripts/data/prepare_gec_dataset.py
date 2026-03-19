"""Pre-tokenize and upload filtered GEC dataset to HuggingFace.

Usage:
    python scripts/prepare_gec_dataset.py --error-type morphology --repo saken-tukenov/sozkz-corpus-tokenized-kk-morph-v1
    python scripts/prepare_gec_dataset.py --error-type orthography --repo saken-tukenov/sozkz-corpus-tokenized-kk-morph-v1
"""

from __future__ import annotations

import argparse
import logging
import sys

from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from slm.data_gec import SPECIAL_TOKENS, SEP, TASK_FIX, SRC, format_gec_example, load_gec_dataset
from slm.data_gec_filtered import classify_error

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_example(
    input_text: str,
    target_text: str,
    tokenizer,
    max_length: int,
    sep_id: int,
) -> dict | None:
    text = format_gec_example(input_text, target_text) + tokenizer.eos_token
    encoded = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(input_ids)
    sep_found = False
    for i, tid in enumerate(input_ids):
        if tid == sep_id:
            sep_found = True
            continue
        if sep_found:
            labels[i] = input_ids[i]

    if not sep_found:
        return None

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-type", required=True, choices=["morphology", "word_order", "orthography"])
    parser.add_argument("--repo", required=True, help="HF repo to upload to")
    parser.add_argument("--tokenizer", default="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--identity-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load tokenizer + add special tokens
    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    sep_id = tokenizer.convert_tokens_to_ids(SEP)
    logger.info("SEP token id: %d, vocab size: %d", sep_id, len(tokenizer))

    # Load raw dataset
    logger.info("Loading raw GEC dataset...")
    ds = load_gec_dataset(identity_ratio=0.0)

    for split_name, split_ds in ds.items():
        logger.info("Processing %s (%d examples)...", split_name, len(split_ds))

        # Filter by error type
        logger.info("  Classifying errors...")
        labels = [classify_error(inp, tgt) for inp, tgt in zip(split_ds["input"], split_ds["target"])]
        keep = [i for i, l in enumerate(labels) if l == args.error_type]
        filtered = split_ds.select(keep)
        logger.info("  Filtered: %d -> %d (%s)", len(split_ds), len(filtered), args.error_type)

        # Add identity examples for train
        if split_name == "train" and args.identity_ratio > 0:
            from datasets import concatenate_datasets
            n_id = int(len(filtered) * args.identity_ratio)
            id_examples = filtered.select(range(min(n_id, len(filtered))))
            id_ds = Dataset.from_dict({"input": id_examples["target"], "target": id_examples["target"]})
            filtered = concatenate_datasets([filtered, id_ds]).shuffle(seed=args.seed)
            logger.info("  + %d identity -> %d total", n_id, len(filtered))

        # Tokenize
        logger.info("  Tokenizing %d examples...", len(filtered))
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        skipped = 0

        for i in range(len(filtered)):
            result = tokenize_example(
                filtered[i]["input"], filtered[i]["target"],
                tokenizer, args.max_length, sep_id,
            )
            if result is None:
                skipped += 1
                continue
            all_input_ids.append(result["input_ids"])
            all_attention_mask.append(result["attention_mask"])
            all_labels.append(result["labels"])

            if (i + 1) % 100000 == 0:
                logger.info("    %d/%d done", i + 1, len(filtered))

        logger.info("  Done: %d tokenized, %d skipped", len(all_input_ids), skipped)

        tokenized = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        })

        # Upload split
        logger.info("  Uploading %s to %s...", split_name, args.repo)
        tokenized.push_to_hub(args.repo, split=split_name, private=False)
        logger.info("  Uploaded %s (%d examples)", split_name, len(tokenized))

    logger.info("All done! Dataset at: https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
