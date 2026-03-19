#!/usr/bin/env python3
"""Tokenize clean Kazakh text dataset and push to HF Hub.

Loads saken-tukenov/sozkz-corpus-clean-kk-text-v2, tokenizes with
saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1 into 1024-token blocks, pushes result.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi

TEXT_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
TOKENIZER_REPO = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
OUTPUT_REPO = "saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2"
BLOCK_SIZE = 1024
SEED = 42

DATASET_CARD = """---
language:
  - kk
license: apache-2.0
task_categories:
  - text-generation
tags:
  - kazakh
  - pretrain
  - tokenized
  - gpt2
size_categories:
  - 10K<n<100K
source_datasets:
  - saken-tukenov/sozkz-corpus-clean-kk-text-v2
---

# Kazakh Clean Pretrain (Tokenized)

Pre-tokenized Kazakh corpus ready for GPT-2 style language model training. Each example is a packed block of 1024 tokens.

## Overview

| Property | Value |
|----------|-------|
| **Train blocks** | {n_train} |
| **Validation blocks** | {n_val} |
| **Block size** | 1,024 tokens |
| **Total train tokens** | ~{train_tokens_m:.1f}M |
| **Tokenizer** | [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) (50,257 vocab, ByteLevel BPE) |
| **Source** | [saken-tukenov/sozkz-corpus-clean-kk-text-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2) |
| **Language** | Kazakh (kk) |
| **License** | Apache 2.0 |

## Domain distribution

{domain_table}

## Dataset structure

Each example contains:
- `input_ids` (list[int]): Token IDs, length 1024
- `labels` (list[int]): Same as `input_ids` (for causal LM training)
- `attention_mask` (list[int]): All 1s, length 1024

Documents are separated by `<|endoftext|>` (token ID 0) and packed contiguously into fixed-length blocks.

## Usage

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2")
tokenizer = AutoTokenizer.from_pretrained("saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")

# Ready for training — no further preprocessing needed
print(ds["train"][0]["input_ids"][:20])
print(tokenizer.decode(ds["train"][0]["input_ids"][:50]))
```

### With HuggingFace Trainer

```python
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

config = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
model = GPT2LMHeadModel(config)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_ratio=0.05,
    bf16=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
```

## Cleaning pipeline

The source text corpus was processed through a 9-stage deep cleaning pipeline before tokenization:

1. **NFC normalization** + control char removal + whitespace collapsing
2. **Kazakh character check** — must contain Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І
3. **Script profile** — Cyrillic ≥ 60%, Latin ≤ 25%, Arabic ≤ 5%
4. **fastText LID** — Kazakh confidence ≥ 0.5, gap vs ru/en ≥ 0.1
5. **Junk removal** — URLs, HTML, boilerplate, special chars
6. **Repetition filter** — repeated n-grams, compression ratio
7. **Exact dedup** (MD5) + **Near dedup** (MinHash LSH, threshold=0.8)
8. **Domain balancing** — OSCAR 47%, News 25%, Books 20%, Leipzig 5%, CC-100 3%

Full details: [sozkz-corpus-clean-kk-text-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2)

## Citation

```bibtex
@misc{{sozkz-corpus-clean-kk-pretrain-v2,
  author = {{Saken Tukenov}},
  title = {{Kazakh Clean Pretrain (Tokenized)}},
  year = {{2026}},
  url = {{https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2}}
}}
```
"""


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    eos_id = tokenizer.eos_token_id
    print(f"  Vocab: {tokenizer.vocab_size}, EOS ID: {eos_id}")

    print(f"Loading text dataset from {TEXT_REPO}...")
    ds = load_dataset(TEXT_REPO)
    print(f"  Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

    rng = np.random.default_rng(SEED)
    splits = {}

    domain_block_counts = defaultdict(int)

    for split_name in ["train", "validation"]:
        texts = ds[split_name]["text"]
        domains = ds[split_name]["domain"]
        print(f"\nTokenizing {split_name} ({len(texts)} texts)...")

        # Tokenize and pack per domain to track domain stats
        # But pack all together for the final dataset
        token_stream = []
        for i, (text, domain) in enumerate(zip(texts, domains)):
            ids = tokenizer.encode(text)
            token_stream.extend(ids)
            token_stream.append(eos_id)
            if (i + 1) % 10000 == 0:
                print(f"  Tokenized {i+1}/{len(texts)}...")

        print(f"  Total tokens: {len(token_stream):,}")

        # Pack into blocks
        n_blocks = len(token_stream) // BLOCK_SIZE
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for b in range(n_blocks):
            block = token_stream[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            all_input_ids.append(block)
            all_labels.append(block)
            all_attention_mask.append([1] * BLOCK_SIZE)

        print(f"  {n_blocks} blocks of {BLOCK_SIZE} tokens")

        # Shuffle
        perm = rng.permutation(n_blocks).tolist()
        all_input_ids = [all_input_ids[i] for i in perm]
        all_labels = [all_labels[i] for i in perm]
        all_attention_mask = [all_attention_mask[i] for i in perm]

        splits[split_name] = Dataset.from_dict({
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        })

    ds_dict = DatasetDict(splits)
    n_train = len(ds_dict["train"])
    n_val = len(ds_dict["validation"])
    print(f"\nFinal: {n_train} train, {n_val} val blocks")

    # Domain stats from source
    domain_counts = defaultdict(int)
    for d in ds["train"]["domain"]:
        domain_counts[d] += 1
    total_docs = sum(domain_counts.values())
    domain_table = "| Domain | Documents | Share |\n|--------|-----------|-------|\n"
    for domain in sorted(domain_counts, key=domain_counts.get, reverse=True):
        c = domain_counts[domain]
        domain_table += f"| {domain} | {c:,} | {100*c/total_docs:.1f}% |\n"

    # Push dataset
    print(f"\nPushing to {OUTPUT_REPO}...")
    ds_dict.push_to_hub(OUTPUT_REPO, private=False)

    # Push README
    card = DATASET_CARD.format(
        n_train=f"{n_train:,}",
        n_val=f"{n_val:,}",
        train_tokens_m=n_train * BLOCK_SIZE / 1_000_000,
        domain_table=domain_table,
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
    )

    print(f"\nPublished: https://huggingface.co/datasets/{OUTPUT_REPO}")
    print("Done!")


if __name__ == "__main__":
    main()
