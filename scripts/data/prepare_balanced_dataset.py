#!/usr/bin/env python3
"""Prepare a balanced GPT-2-style Kazakh dataset from multidomain CSV files.

Loads each domain CSV separately, filters for Kazakh text, normalizes,
deduplicates, rebalances to target proportions, tokenizes into 1024-token
blocks, and pushes to HuggingFace Hub.

Usage:
    python scripts/prepare_balanced_dataset.py \
        --tokenizer ./tokenizers/kazakh-bpe-32k \
        --hub-repo saken-tukenov/sozkz-corpus-balanced-kk-gpt2-v1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import unicodedata
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/root/slm/logs/prepare_balanced.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------- constants ----------

HF_DATASET = "kz-transformers/multidomain-kazakh-dataset"
HF_RAW_URL = "https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main"

# CSV filenames in the HF repo → domain labels
DOMAIN_FILES = {
    "oscar.csv": "oscar",
    "kazakhNews.csv": "kazakhNews",
    "kazakhBooks.csv": "kazakhBooks",
    "leipzig.csv": "leipzig",
    "cc100-monolingual-crawled-data.csv": "cc100",
}

# Target proportions (must sum to 1.0)
TARGET_PROPORTIONS = {
    "oscar": 0.47,
    "kazakhNews": 0.25,
    "kazakhBooks": 0.20,
    "leipzig": 0.05,
    "cc100": 0.03,
}

KAZAKH_PATTERN = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
MULTI_SPACE = re.compile(r"[ \t]+")
MIN_TEXT_LEN = 30
BLOCK_SIZE = 1024
SEED = 42


# ---------- helpers ----------


def normalize_text(text: str) -> str:
    """Normalize unicode and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def filter_and_clean(ds: Dataset) -> Dataset:
    """Filter for Kazakh text and clean."""
    col_names = ds.column_names

    def process(example):
        text = example.get("text", "") or ""
        text = normalize_text(text)

        # Basic filters
        if len(text) < MIN_TEXT_LEN:
            return {"text": "", "_keep": False}

        # Kazakh language filters (use columns if available, else heuristic)
        if "predicted_language" in col_names:
            if example.get("predicted_language") != "kaz":
                return {"text": "", "_keep": False}
        if "contains_kaz_symbols" in col_names:
            if not example.get("contains_kaz_symbols"):
                return {"text": "", "_keep": False}

        # Fallback: must contain at least one Kazakh-specific char
        if not KAZAKH_PATTERN.search(text):
            return {"text": "", "_keep": False}

        return {"text": text, "_keep": True}

    ds = ds.map(process, num_proc=8, desc="Filtering")
    ds = ds.filter(lambda x: x["_keep"], num_proc=8)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds


def deduplicate(ds: Dataset) -> Dataset:
    """Remove exact duplicates via MD5 hash."""
    seen = set()
    keep_indices = []
    for i, text in enumerate(ds["text"]):
        h = text_hash(text)
        if h not in seen:
            seen.add(h)
            keep_indices.append(i)
    logger.info("Dedup: %d -> %d texts", len(ds), len(keep_indices))
    return ds.select(keep_indices)


# ---------- main pipeline ----------


def load_domain(filename: str) -> Dataset:
    """Load a single CSV domain file from the HF dataset."""
    url = f"{HF_RAW_URL}/{filename}"
    logger.info("Loading %s from %s ...", filename, url)
    ds = load_dataset("csv", data_files=url, split="train")
    logger.info("  %d rows loaded", len(ds))
    return ds


def count_tokens(ds: Dataset, tokenizer) -> int:
    """Estimate total tokens in a dataset."""
    sample_size = min(5000, len(ds))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(ds), size=sample_size, replace=False)
    sample_texts = ds.select(indices.tolist())["text"]
    total_sample_tokens = sum(len(tokenizer.encode(t)) for t in sample_texts)
    avg_tokens = total_sample_tokens / sample_size
    return int(avg_tokens * len(ds))


def resample_domains(
    domain_datasets: dict[str, Dataset],
    tokenizer,
) -> dict[str, Dataset]:
    """Resample domains to match target token proportions."""
    # Count tokens per domain
    token_counts = {}
    for domain, ds in domain_datasets.items():
        tc = count_tokens(ds, tokenizer)
        token_counts[domain] = tc
        logger.info("Domain %-15s: %8d texts, ~%12d tokens", domain, len(ds), tc)

    total_tokens = sum(token_counts.values())
    logger.info("Total tokens before resampling: %d", total_tokens)

    # Use the total available tokens as budget.
    # For each domain, compute how many texts to sample.
    # We'll use the *smallest* domain's constraint to set the scale.
    # scale = min over domains of (available_tokens[d] / target_proportion[d])
    scales = []
    for domain in domain_datasets:
        target = TARGET_PROPORTIONS[domain]
        available = token_counts[domain]
        scales.append(available / target)

    budget_tokens = min(scales)  # constrained by smallest relative domain
    logger.info("Budget tokens (constrained): %d", int(budget_tokens))

    resampled = {}
    for domain, ds in domain_datasets.items():
        target_tokens = int(budget_tokens * TARGET_PROPORTIONS[domain])
        current_tokens = token_counts[domain]
        ratio = target_tokens / current_tokens
        n_samples = max(1, int(len(ds) * ratio))
        n_samples = min(n_samples, len(ds))  # no upsampling, only downsample

        if n_samples < len(ds):
            rng = np.random.default_rng(SEED)
            indices = rng.choice(len(ds), size=n_samples, replace=False)
            indices.sort()
            resampled[domain] = ds.select(indices.tolist())
            logger.info("Domain %-15s: downsampled %d -> %d texts", domain, len(ds), n_samples)
        else:
            resampled[domain] = ds
            logger.info("Domain %-15s: kept all %d texts", domain, len(ds))

    return resampled


def tokenize_and_group(
    ds: Dataset,
    tokenizer,
    block_size: int = BLOCK_SIZE,
    domain: str = "",
) -> Dataset:
    """Tokenize and group into fixed-length blocks."""

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {domain}",
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]

        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
            "domain": [domain] * (total_length // block_size),
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping {domain}",
    )

    logger.info("Domain %-15s: %d blocks of %d tokens", domain, len(grouped), block_size)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced Kazakh dataset")
    parser.add_argument(
        "--tokenizer", default="./tokenizers/kazakh-bpe-32k",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--hub-repo", default="saken-tukenov/sozkz-corpus-balanced-kk-gpt2-v1",
        help="HuggingFace Hub repo to push to",
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--no-push", action="store_true", help="Skip push to hub")
    args = parser.parse_args()

    logger.info("=== Starting balanced dataset preparation ===")
    logger.info("Tokenizer: %s", args.tokenizer)
    logger.info("Hub repo: %s", args.hub_repo)
    logger.info("Block size: %d", args.block_size)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer loaded, vocab size: %d", tokenizer.vocab_size)

    # Step 1: Load, filter, deduplicate each domain
    domain_datasets: dict[str, Dataset] = {}
    for filename, domain in DOMAIN_FILES.items():
        ds = load_domain(filename)
        ds = filter_and_clean(ds)
        ds = deduplicate(ds)
        domain_datasets[domain] = ds
        logger.info("Domain %-15s: %d clean texts", domain, len(ds))

    # Step 2: Resample to target proportions
    resampled = resample_domains(domain_datasets, tokenizer)

    # Step 3: Tokenize and group each domain
    grouped_datasets = []
    for domain, ds in resampled.items():
        grouped = tokenize_and_group(ds, tokenizer, block_size=args.block_size, domain=domain)
        grouped_datasets.append(grouped)

    # Step 4: Concatenate and shuffle
    combined = concatenate_datasets(grouped_datasets)
    combined = combined.shuffle(seed=SEED)
    logger.info("Combined dataset: %d blocks", len(combined))

    # Log domain distribution
    domain_counts = defaultdict(int)
    for d in combined["domain"]:
        domain_counts[d] += 1
    total = len(combined)
    logger.info("=== Final domain distribution ===")
    for domain, count in sorted(domain_counts.items()):
        logger.info("  %-15s: %8d blocks (%5.1f%%)", domain, count, 100 * count / total)

    # Step 5: Train/val split 99/1
    split = combined.train_test_split(test_size=0.01, seed=SEED)
    ds_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    logger.info("Train: %d blocks, Validation: %d blocks", len(ds_dict["train"]), len(ds_dict["validation"]))

    # Step 6: Push to hub
    if not args.no_push:
        logger.info("Pushing to %s ...", args.hub_repo)
        ds_dict.push_to_hub(args.hub_repo, private=False)
        logger.info("Done! Dataset published at https://huggingface.co/datasets/%s", args.hub_repo)
    else:
        logger.info("Skipping push to hub (--no-push)")

    logger.info("=== Finished ===")


if __name__ == "__main__":
    main()
