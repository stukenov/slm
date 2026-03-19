#!/usr/bin/env python3
"""Step 2: Tokenize clean text into packed 1024-token blocks.

Loads the clean text dataset, tokenizes with sozkz-core-gpt2-50k-kk-base-v1,
packs into fixed-length blocks separated by <|endoftext|>, and
publishes the result to HuggingFace Hub.

Usage:
    python tokenize_data.py [--push-to-hub saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2]

Input:  saken-tukenov/sozkz-corpus-clean-kk-text-v2  (text)
Output: saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2       (tokenized blocks)
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

TEXT_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
TOKENIZER_REPO = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
BLOCK_SIZE = 1024
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", default=None, help="HF repo to push")
    parser.add_argument("--save-dir", default="./data/tokenized", help="Local save path")
    args = parser.parse_args()

    print(f"Loading tokenizer: {TOKENIZER_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    eos_id = tokenizer.eos_token_id
    print(f"  Vocab: {tokenizer.vocab_size}, EOS ID: {eos_id}")

    print(f"Loading text dataset: {TEXT_REPO}")
    ds = load_dataset(TEXT_REPO)
    print(f"  Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

    rng = np.random.default_rng(SEED)
    splits = {}

    for split_name in ["train", "validation"]:
        texts = ds[split_name]["text"]
        print(f"\nTokenizing {split_name} ({len(texts)} texts)...")

        # Tokenize all texts and concatenate with EOS separators
        token_stream = []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)
            token_stream.extend(ids)
            token_stream.append(eos_id)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(texts)} texts tokenized...")

        print(f"  Total tokens: {len(token_stream):,}")

        # Pack into fixed-length blocks
        n_blocks = len(token_stream) // BLOCK_SIZE
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for b in range(n_blocks):
            block = token_stream[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            all_input_ids.append(block)
            all_labels.append(block)
            all_attention_mask.append([1] * BLOCK_SIZE)

        # Shuffle blocks
        perm = rng.permutation(n_blocks).tolist()
        all_input_ids = [all_input_ids[i] for i in perm]
        all_labels = [all_labels[i] for i in perm]
        all_attention_mask = [all_attention_mask[i] for i in perm]

        splits[split_name] = Dataset.from_dict({
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        })
        print(f"  {n_blocks} blocks of {BLOCK_SIZE} tokens")

    ds_dict = DatasetDict(splits)
    print(f"\nFinal: {len(ds_dict['train'])} train, {len(ds_dict['validation'])} val blocks")
    print(f"Total train tokens: ~{len(ds_dict['train']) * BLOCK_SIZE / 1e6:.1f}M")

    # Save locally
    ds_dict.save_to_disk(args.save_dir)
    print(f"Saved to {args.save_dir}")

    if args.push_to_hub:
        print(f"Pushing to {args.push_to_hub}...")
        ds_dict.push_to_hub(args.push_to_hub, private=False)
        print(f"Published: https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
