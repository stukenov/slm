#!/usr/bin/env python3
"""Tokenize sozkz-corpus-clean-v3 — fastest version.

Uses Rust tokenizer's internal parallelism (TOKENIZERS_PARALLELISM=true)
with large batches. Single process, no datasets.map() overhead.
Writes tokens directly to binary file on disk.

Usage:
    .venv/bin/python scripts/tokenize_v3.py
    .venv/bin/python scripts/tokenize_v3.py --dry-run
    .venv/bin/python scripts/tokenize_v3.py --skip-tokenize
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

BLOCK_SIZE = 1024
SEED = 42
SHARD_BLOCKS = 200_000
BATCH_SIZE = 500_000  # large batches for Rust tokenizer

DATASET_REPO = "saken-tukenov/sozkz-corpus-clean-v3"
TOKENIZER_REPO = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
OUTPUT_REPO = "saken-tukenov/sozkz-corpus-tokenized-kk-llama50k-v3"
OUTPUT_DIR = "/root/slm/data/tokenized_v3"


def tokenize_to_disk(split, tokenizer, eos_id, out_path, source_counts):
    """Tokenize split and write flat int32 binary. Returns total_tokens."""
    n = len(split)
    total_tokens = 0
    t0 = time.time()

    with open(out_path, "wb") as f:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = split[start:end]

            # Count sources
            if "source" in batch:
                for s in batch["source"]:
                    source_counts[s] += 1

            # Rust tokenizer parallelizes internally across all cores
            encoded = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]

            # Build numpy array with EOS separators
            chunks = []
            for ids in encoded:
                chunks.append(np.array(ids, dtype=np.int32))
                chunks.append(np.array([eos_id], dtype=np.int32))

            arr = np.concatenate(chunks)
            arr.tofile(f)
            total_tokens += len(arr)
            del chunks, arr, encoded

            elapsed = time.time() - t0
            speed = end / elapsed
            eta = (n - end) / speed if speed > 0 else 0
            print(f"  {end:,}/{n:,} ({total_tokens:,} tok) "
                  f"[{elapsed:.0f}s, {speed:.0f} docs/s, ETA {eta:.0f}s]")

    return total_tokens


def create_shards(bin_path, total_tokens, block_size, out_dir, seed):
    """Memmap → shuffle → parquet shards. Returns n_blocks."""
    n_blocks = total_tokens // block_size
    usable = n_blocks * block_size
    print(f"  Tokens: {total_tokens:,}, Blocks: {n_blocks:,} x {block_size}")

    tokens = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(usable,))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    attn_row = [1] * block_size

    written = 0
    shard_idx = 0
    t0 = time.time()

    while written < n_blocks:
        shard_end = min(written + SHARD_BLOCKS, n_blocks)
        shard_size = shard_end - written
        indices = perm[written:shard_end]

        # Vectorized block gathering
        offsets = indices.astype(np.int64) * block_size
        block_data = np.empty((shard_size, block_size), dtype=np.int32)
        for i, off in enumerate(offsets):
            block_data[i] = tokens[off:off + block_size]

        rows = block_data.tolist()

        table = pa.table({
            "input_ids": pa.array(rows, type=pa.list_(pa.int32())),
            "labels": pa.array(rows, type=pa.list_(pa.int32())),
            "attention_mask": pa.array([attn_row] * shard_size, type=pa.list_(pa.int32())),
        })

        path = os.path.join(out_dir, f"shard-{shard_idx:05d}.parquet")
        pq.write_table(table, path, compression="zstd")
        mb = os.path.getsize(path) / 1e6
        elapsed = time.time() - t0
        print(f"  Shard {shard_idx}: {shard_size:,} blocks ({mb:.0f}MB) [{elapsed:.0f}s]")

        del table, rows, block_data
        written = shard_end
        shard_idx += 1

    del tokens
    os.remove(bin_path)
    return n_blocks


def push_shards(split_name, out_dir, repo_id, api):
    shards = sorted(glob.glob(os.path.join(out_dir, "shard-*.parquet")))
    print(f"  Uploading {len(shards)} shards for {split_name}...")
    for i, path in enumerate(shards):
        fname = os.path.basename(path)
        mb = os.path.getsize(path) / 1e6
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"data/{split_name}/{fname}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"    [{i+1}/{len(shards)}] {fname} ({mb:.0f}MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--tokenizer", default=TOKENIZER_REPO)
    parser.add_argument("--dataset", default=DATASET_REPO)
    parser.add_argument("--output-repo", default=OUTPUT_REPO)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Enable Rust tokenizer's internal multi-threading
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    api = HfApi()
    source_counts = Counter()
    total_tokens_all = 0
    split_blocks = {}

    if not args.skip_tokenize:
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        eos_id = tokenizer.eos_token_id
        print(f"  Vocab: {tokenizer.vocab_size}, EOS ID: {eos_id}")

        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset(args.dataset)
        print(f"  Train: {len(ds['train']):,}, Val: {len(ds['validation']):,}")

        for split_name in ["train", "validation"]:
            print(f"\n{'='*60}")
            print(f"Processing {split_name}...")
            t0 = time.time()

            split_dir = os.path.join(args.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            bin_path = os.path.join(split_dir, "_tokens.bin")

            # Phase 1: tokenize → binary
            print(f"  Phase 1: Tokenizing (batch={BATCH_SIZE:,}, Rust parallel)...")
            total_tokens = tokenize_to_disk(
                ds[split_name], tokenizer, eos_id, bin_path, source_counts
            )
            total_tokens_all += total_tokens
            print(f"  Phase 1 done in {time.time()-t0:.0f}s")

            # Phase 2: binary → parquet shards
            t1 = time.time()
            print(f"  Phase 2: Creating parquet shards...")
            n_blocks = create_shards(
                bin_path, total_tokens, args.block_size, split_dir,
                SEED if split_name == "train" else SEED + 1,
            )
            split_blocks[split_name] = n_blocks
            print(f"  Phase 2 done in {time.time()-t1:.0f}s")
            print(f"  Total: {time.time()-t0:.0f}s")

    else:
        for split_name in ["train", "validation"]:
            split_dir = os.path.join(args.output_dir, split_name)
            shards = sorted(glob.glob(os.path.join(split_dir, "shard-*.parquet")))
            n = sum(pq.read_metadata(s).num_rows for s in shards)
            split_blocks[split_name] = n
            print(f"  {split_name}: {n:,} blocks in {len(shards)} shards")

    n_train = split_blocks.get("train", 0)
    n_val = split_blocks.get("validation", 0)

    print(f"\n{'='*60}")
    if total_tokens_all:
        print(f"TOTAL TOKENS: {total_tokens_all:,}")
        print(f"Train blocks: {n_train:,} ({n_train * args.block_size:,} tokens)")
        print(f"Val blocks:   {n_val:,} ({n_val * args.block_size:,} tokens)")
        chinchilla = total_tokens_all / 20
        print(f"Chinchilla-optimal model size: ~{chinchilla / 1e6:.0f}M params")

    if source_counts:
        print("\nSource distribution:")
        total_docs = sum(source_counts.values())
        for src, cnt in source_counts.most_common():
            print(f"  {src:20s}: {cnt:>10,d} ({100*cnt/total_docs:.1f}%)")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run — not pushing.")
        return

    print(f"\nCreating repo {args.output_repo}...")
    create_repo(args.output_repo, repo_type="dataset", exist_ok=True)

    for split_name in ["train", "validation"]:
        split_dir = os.path.join(args.output_dir, split_name)
        push_shards(split_name, split_dir, args.output_repo, api)

    # README
    if source_counts:
        total_docs = sum(source_counts.values())
        source_table = "| Source | Documents | Share |\n|--------|-----------|-------|\n"
        for src, cnt in source_counts.most_common():
            source_table += f"| {src} | {cnt:,} | {100*cnt/total_docs:.1f}% |\n"
    else:
        source_table = "_N/A_"

    tok_str = f"~{total_tokens_all/1e9:.2f}B" if total_tokens_all else "N/A"
    card = f"""---
language:
- kk
license: apache-2.0
task_categories:
- text-generation
tags:
- kazakh
- pretrain
- tokenized
size_categories:
- 1M<n<10M
dataset_info:
  features:
  - name: input_ids
    sequence: int32
  - name: labels
    sequence: int32
  - name: attention_mask
    sequence: int32
  splits:
  - name: train
    num_examples: {n_train}
  - name: validation
    num_examples: {n_val}
---

# SozKZ Corpus Tokenized v3 (50K)

Pre-tokenized Kazakh corpus. Each sample = {args.block_size} packed tokens.

Source: [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})

| Property | Value |
|----------|-------|
| **Train blocks** | {n_train:,} |
| **Val blocks** | {n_val:,} |
| **Block size** | {args.block_size} |
| **Total tokens** | {tok_str} |
| **Tokenizer** | [{args.tokenizer}](https://huggingface.co/{args.tokenizer}) |

## Source distribution

{source_table}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{args.output_repo}")
sample = ds["train"][0]
print(len(sample["input_ids"]))  # {args.block_size}
```
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.output_repo,
        repo_type="dataset",
    )
    print(f"\nPublished: https://huggingface.co/datasets/{args.output_repo}")

    print("Cleaning up local shards...")
    shutil.rmtree(args.output_dir, ignore_errors=True)
    print("Done!")


if __name__ == "__main__":
    main()
