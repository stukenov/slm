#!/usr/bin/env python3
"""Tokenize clean_corpus_v2 with sozkz-core-gpt2-50k-kk-base-v1 and push to HF Hub.

Uses batched map with multiprocessing for speed.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import HfApi

BLOCK_SIZE = 1024
SEED = 42

# Tokenizer loaded per-worker
_tokenizer = None
_tokenizer_name = None


def _init_worker(tokenizer_name: str):
    global _tokenizer, _tokenizer_name
    _tokenizer_name = tokenizer_name
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def _tokenize_batch(batch):
    """Tokenize a batch of texts, return flat token list."""
    all_ids = []
    eos_id = _tokenizer.eos_token_id
    for text in batch["text"]:
        ids = _tokenizer.encode(text)
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return {"token_ids": [all_ids], "n_tokens": [len(all_ids)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="/root/slm/data/clean_corpus_v2/dataset")
    parser.add_argument("--tokenizer", default="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")
    parser.add_argument("--output-repo", default="saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only count tokens, don't push")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    print(f"  Vocab: {tokenizer.vocab_size}, EOS ID: {eos_id}")

    print(f"Loading dataset from {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    print(f"  Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

    rng = np.random.default_rng(SEED)
    splits = {}
    total_tokens_all = 0

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for split_name in ["train", "validation"]:
        split = ds[split_name]
        n = len(split)
        print(f"\nTokenizing {split_name} ({n} texts) with {args.num_proc} workers...")

        # Batch tokenize: each batch returns a flat list of token IDs
        # Use large batch size to reduce overhead
        batch_size = 5000

        # Batch tokenize using Rust backend for speed
        token_stream = []
        chunk_size = 100_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_texts = split[start:end]["text"]
            # Batch encode — uses Rust tokenizer, much faster than per-text
            encoded = tokenizer(chunk_texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                token_stream.extend(ids)
                token_stream.append(eos_id)
            print(f"  {end}/{n} ({len(token_stream):,} tokens)")

        total_tokens = len(token_stream)
        total_tokens_all += total_tokens
        print(f"  Total tokens in {split_name}: {total_tokens:,}")

        n_blocks = total_tokens // args.block_size
        print(f"  Packing into {n_blocks:,} blocks of {args.block_size}")

        # Pack into blocks using numpy for speed
        arr = np.array(token_stream[:n_blocks * args.block_size], dtype=np.int32)
        arr = arr.reshape(n_blocks, args.block_size)

        # Shuffle
        perm = rng.permutation(n_blocks)
        arr = arr[perm]

        splits[split_name] = Dataset.from_dict({
            "input_ids": arr.tolist(),
            "labels": arr.tolist(),
            "attention_mask": [[1] * args.block_size] * n_blocks,
        })

    ds_dict = DatasetDict(splits)
    n_train = len(ds_dict["train"])
    n_val = len(ds_dict["validation"])

    print(f"\n{'='*60}")
    print(f"TOTAL TOKENS: {total_tokens_all:,}")
    print(f"Train blocks: {n_train:,} ({n_train * args.block_size:,} tokens)")
    print(f"Val blocks:   {n_val:,} ({n_val * args.block_size:,} tokens)")
    chinchilla = total_tokens_all / 20
    print(f"Chinchilla-optimal model size: ~{chinchilla / 1e6:.0f}M params")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run — not pushing to Hub.")
        return

    # Domain stats
    domain_counts = defaultdict(int)
    for d in ds["train"]["domain"]:
        domain_counts[d] += 1
    total_docs = sum(domain_counts.values())
    domain_table = "| Domain | Documents | Share |\n|--------|-----------|-------|\n"
    for domain in sorted(domain_counts, key=domain_counts.get, reverse=True):
        c = domain_counts[domain]
        domain_table += f"| {domain} | {c:,} | {100*c/total_docs:.1f}% |\n"

    print(f"\nPushing to {args.output_repo}...")
    ds_dict.push_to_hub(args.output_repo, private=False)

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
---

# Kazakh Clean Pretrain v2 (Tokenized)

Pre-tokenized Kazakh corpus for LLM training. Each sample is a packed block of {args.block_size} tokens.

| Property | Value |
|----------|-------|
| **Train blocks** | {n_train:,} |
| **Val blocks** | {n_val:,} |
| **Block size** | {args.block_size} tokens |
| **Total tokens** | ~{total_tokens_all/1e9:.2f}B |
| **Tokenizer** | [{args.tokenizer}](https://huggingface.co/{args.tokenizer}) |
| **Vocab size** | {tokenizer.vocab_size:,} |

## Domain distribution

{domain_table}
"""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.output_repo,
        repo_type="dataset",
    )

    print(f"\nPublished: https://huggingface.co/datasets/{args.output_repo}")
    print("Done!")


if __name__ == "__main__":
    main()
