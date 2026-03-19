"""Prepare domain-labeled tokenized dataset for MoE domain-aware training.

Loads individual CSV files from kz-transformers/multidomain-kazakh-dataset:
  - cc100-monolingual-crawled-data.csv (wiki, 6.1GB)
  - kazakhNews.csv (news, 11.5GB)
  - kazakhBooks.csv (books, 3.97GB)
  - oscar.csv (web, 2.76GB)
  - leipzig.csv (academic, 389MB)

Domain mapping for 16 experts:
  0-1: shared (always active, bias in router)
  2-3: news (kazakhNews)
  4-5: web (oscar)
  6-7: encyclopedic (cc100)
  8-9: literary (kazakhBooks)
  10-11: academic (leipzig)
  12-13: reserved (future QA/instruction data)
  14-15: cultural (dastur-mc)
"""

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CSV file -> domain_id mapping
DOMAIN_CSV_FILES = {
    2: "kazakhNews.csv",
    4: "oscar.csv",
    6: "cc100-monolingual-crawled-data.csv",
    8: "kazakhBooks.csv",
    10: "leipzig.csv",
}

DOMAIN_NAMES = {
    2: "news", 4: "web/oscar", 6: "wiki/cc100",
    8: "books", 10: "academic/leipzig", 14: "cultural/dastur",
}

MIN_TEXT_LENGTH = 20
REPO_ID = "kz-transformers/multidomain-kazakh-dataset"


def load_csv_domain(domain_id: int, csv_file: str, max_samples: int, num_proc: int) -> Dataset:
    """Load a single CSV file as a domain dataset."""
    logger.info("Loading domain %d (%s): %s", domain_id, DOMAIN_NAMES.get(domain_id, "?"), csv_file)

    url = f"hf://datasets/{REPO_ID}/{csv_file}"
    ds = load_dataset("csv", data_files=url, split="train")
    logger.info("  Raw: %d samples", len(ds))

    # Find text column
    text_col = None
    for col in ["text", "content", "sentence"]:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = ds.column_names[0]
        logger.info("  Using first column as text: %s", text_col)

    # Keep only text, filter
    ds = ds.select_columns([text_col])
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    ds = ds.filter(
        lambda x: x["text"] is not None and len(x["text"]) >= MIN_TEXT_LENGTH,
        num_proc=num_proc, desc=f"Filtering domain {domain_id}",
    )
    logger.info("  After filter: %d samples", len(ds))

    # Downsample if needed
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logger.info("  Downsampled to %d", max_samples)

    ds = ds.add_column("domain_id", [domain_id] * len(ds))
    return ds


def tokenize_dataset(ds: Dataset, tokenizer, block_size: int, num_proc: int) -> Dataset:
    """Tokenize and group into fixed-length blocks, preserving domain_id."""

    def tokenize_fn(examples):
        tokens = tokenizer(examples["text"], return_attention_mask=False)
        tokens["domain_id"] = examples["domain_id"]
        return tokens

    tokenized = ds.map(
        tokenize_fn, batched=True, num_proc=num_proc,
        remove_columns=["text"], desc="Tokenizing",
    )

    def group_texts(examples):
        all_ids = []
        all_labels = []
        all_domains = []

        current_ids = []
        current_domain = examples["domain_id"][0] if examples["domain_id"] else 0

        for ids, dom in zip(examples["input_ids"], examples["domain_id"]):
            current_ids.extend(ids)
            current_domain = dom

            while len(current_ids) >= block_size:
                block = current_ids[:block_size]
                current_ids = current_ids[block_size:]
                all_ids.append(block)
                all_labels.append(block)
                all_domains.append(current_domain)

        return {"input_ids": all_ids, "labels": all_labels, "domain_id": all_domains}

    grouped = tokenized.map(
        group_texts, batched=True, num_proc=num_proc,
        remove_columns=tokenized.column_names, desc="Grouping",
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Prepare domain-labeled MoE dataset")
    parser.add_argument("--tokenizer", default="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--output", default="outputs/sozkz-corpus-balanced-kk-moe-domain-v1")
    parser.add_argument("--push-to-hub", default=None, help="HF repo to push to")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-samples-per-domain", type=int, default=2_000_000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_datasets = []

    # Load each CSV domain
    for domain_id, csv_file in DOMAIN_CSV_FILES.items():
        try:
            ds = load_csv_domain(domain_id, csv_file, args.max_samples_per_domain, args.num_proc)
            all_datasets.append(ds)
        except Exception as e:
            logger.warning("Failed to load domain %d (%s): %s", domain_id, csv_file, e)

    # Load kazakh-dastur-mc (cultural domain 14)
    try:
        logger.info("Loading kz-transformers/kazakh-dastur-mc (domain 14)...")
        dastur = load_dataset("kz-transformers/kazakh-dastur-mc", split="test")
        text_col = "question" if "question" in dastur.column_names else dastur.column_names[0]
        dastur = dastur.select_columns([text_col])
        if text_col != "text":
            dastur = dastur.rename_column(text_col, "text")
        dastur = dastur.filter(lambda x: len(x["text"]) >= MIN_TEXT_LENGTH, num_proc=2)
        dastur = dastur.add_column("domain_id", [14] * len(dastur))
        # Upsample small dataset
        if len(dastur) < 50_000:
            repeat = min(50_000 // max(len(dastur), 1), 50)
            if repeat > 1:
                dastur = concatenate_datasets([dastur] * repeat)
                logger.info("  Upsampled dastur-mc to %d (x%d)", len(dastur), repeat)
        all_datasets.append(dastur)
        logger.info("  Domain 14 (cultural): %d samples", len(dastur))
    except Exception as e:
        logger.warning("Failed to load dastur-mc: %s", e)

    if not all_datasets:
        raise ValueError("No datasets loaded!")

    # Concatenate
    logger.info("Concatenating %d domain datasets...", len(all_datasets))
    full_ds = concatenate_datasets(all_datasets)
    full_ds = full_ds.shuffle(seed=42)
    logger.info("Total samples: %d", len(full_ds))

    # Domain distribution
    from collections import Counter
    domain_counts = Counter(full_ds["domain_id"])
    for did, cnt in sorted(domain_counts.items()):
        logger.info("  Domain %2d (%s): %8d samples (%.1f%%)",
                     did, DOMAIN_NAMES.get(did, "?"), cnt, 100 * cnt / len(full_ds))

    # Tokenize
    logger.info("Tokenizing with block_size=%d...", args.block_size)
    tokenized = tokenize_dataset(full_ds, tokenizer, args.block_size, args.num_proc)
    logger.info("Tokenized blocks: %d", len(tokenized))

    # Train/val split
    split = tokenized.train_test_split(test_size=args.val_ratio, seed=42)
    ds_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
    logger.info("Train: %d, Validation: %d", len(ds_dict["train"]), len(ds_dict["validation"]))

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_path))
    logger.info("Saved to %s", output_path)

    if args.push_to_hub:
        logger.info("Pushing to HF Hub: %s", args.push_to_hub)
        ds_dict.push_to_hub(args.push_to_hub)
        logger.info("Done!")


if __name__ == "__main__":
    main()
