"""Tokenize a HuggingFace dataset and upload the result.

Handles long texts by splitting into chunks BEFORE tokenization to avoid
straggler workers. All text content is preserved — nothing is truncated.

Usage:
    python scripts/tokenize_dataset.py \
        --dataset saken-tukenov/sozkz-corpus-clean-kk-text-v2 \
        --tokenizer stukenov/sozkz-core-gpt2-200k-kk-base-v1 \
        --output stukenov/sozkz-corpus-tokenized-kk-200k-v1 \
        --block-size 2048

    # Multi-column dataset (FineWeb-Edu):
    python scripts/tokenize_dataset.py \
        --dataset stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1 \
        --tokenizer stukenov/sozkz-core-gpt2-200k-kk-base-v1 \
        --output stukenov/sozkz-corpus-tokenized-enkk-200k-v1 \
        --text-column "text_en,text_kk" \
        --block-size 2048
"""
from __future__ import annotations

import argparse
import logging
import time
from itertools import chain

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Long texts are split into chunks of this size (chars) before tokenization.
# This prevents straggler workers while preserving ALL content.
# 20K chars ≈ 3-5K tokens — safely fits in memory, no data loss.
CHUNK_CHARS = 20_000


def tokenize_and_upload(
    dataset_name: str,
    tokenizer_name: str,
    output_repo: str,
    block_size: int = 2048,
    text_column: str = "text",
    num_proc: int = 16,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    t0 = time.time()

    logger.info("Loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, verification_mode="no_checks")

    if isinstance(ds, dict):
        if "train" in ds:
            raw = ds["train"]
        else:
            raw = next(iter(ds.values()))
    else:
        raw = ds

    logger.info("Dataset size: %d rows", len(raw))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Support multiple text columns (comma-separated) — merge into single "text" column
    text_columns = [c.strip() for c in text_column.split(",")]
    missing = [c for c in text_columns if c not in raw.column_names]
    if missing:
        logger.warning("Columns %s not found. Available: %s", missing, raw.column_names)
        text_columns = [c for c in raw.column_names if c.startswith("text")]
        logger.info("Auto-detected text columns: %s", text_columns)

    if len(text_columns) > 1:
        logger.info("Merging %d text columns into flat dataset...", len(text_columns))
        from datasets import concatenate_datasets
        parts = []
        for col in text_columns:
            part = raw.select_columns([col]).rename_column(col, "text")
            parts.append(part)
        raw = concatenate_datasets(parts)
        text_column = "text"
        logger.info("Merged dataset size: %d rows", len(raw))
    elif len(text_columns) == 1:
        text_column = text_columns[0]

    # Step 1: Split long texts into chunks to avoid straggler workers.
    # Each chunk ≤ CHUNK_CHARS. We split on newlines/spaces to keep words intact.
    def split_long_texts(examples):
        result = []
        for text in examples[text_column]:
            if not text or len(text.strip()) == 0:
                continue
            text = text.strip()
            if len(text) <= CHUNK_CHARS:
                result.append(text)
            else:
                # Split into chunks at paragraph/newline boundaries
                pos = 0
                while pos < len(text):
                    end = min(pos + CHUNK_CHARS, len(text))
                    if end < len(text):
                        # Try to break at newline
                        nl = text.rfind("\n", pos + CHUNK_CHARS // 2, end)
                        if nl > pos:
                            end = nl + 1
                        else:
                            # Try to break at space
                            sp = text.rfind(" ", pos + CHUNK_CHARS // 2, end)
                            if sp > pos:
                                end = sp + 1
                    chunk = text[pos:end].strip()
                    if chunk:
                        result.append(chunk)
                    pos = end
        return {"text": result}

    logger.info("Splitting long texts (chunk_size=%d chars)...", CHUNK_CHARS)
    # Rename column to "text" if needed
    if text_column != "text":
        raw = raw.rename_column(text_column, "text")
        text_column = "text"
    # Remove all columns except text
    drop_cols = [c for c in raw.column_names if c != "text"]
    if drop_cols:
        raw = raw.remove_columns(drop_cols)

    chunked = raw.map(
        split_long_texts,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Chunking",
    )
    logger.info("After chunking: %d rows (was %d)", len(chunked), len(raw))

    # Step 2: Tokenize (no truncation needed — all texts are ≤ CHUNK_CHARS)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    logger.info("Tokenizing with %d workers...", num_proc)
    tokenized = chunked.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        writer_batch_size=1000,
        num_proc=num_proc,
        remove_columns=chunked.column_names,
        desc="Tokenizing",
    )

    # Step 3: Concatenate and group into fixed-size blocks
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Grouping into blocks of %d...", block_size)
    blocked = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        writer_batch_size=500,
        num_proc=num_proc,
        desc="Grouping",
    )

    total_tokens = len(blocked) * block_size
    logger.info("Total blocks: %d (%.2fB tokens)", len(blocked), total_tokens / 1e9)

    split = blocked.train_test_split(test_size=val_ratio, seed=seed)
    final = DatasetDict({"train": split["train"], "validation": split["test"]})
    logger.info("Train: %d, Validation: %d", len(final["train"]), len(final["validation"]))

    logger.info("Uploading to %s...", output_repo)
    final.push_to_hub(output_repo, private=False)
    elapsed = time.time() - t0
    logger.info("Done! Total time: %.0fs (%.1f min)", elapsed, elapsed / 60)


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset and upload to HF Hub")
    parser.add_argument("--dataset", required=True, help="HF dataset to tokenize")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer to use")
    parser.add_argument("--output", required=True, help="HF repo to upload tokenized dataset")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    tokenize_and_upload(
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        output_repo=args.output,
        block_size=args.block_size,
        text_column=args.text_column,
        num_proc=args.num_proc,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
