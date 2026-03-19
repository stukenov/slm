---
language:
- kk
license: apache-2.0
task_categories:
- text-generation
tags:
- kazakh
- language-modeling
- gpt2
- balanced
- pretokenized
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: input_ids
    sequence: int32
  - name: labels
    sequence: int32
  - name: domain
    dtype: string
  splits:
  - name: train
    num_examples: 475190
  - name: validation
    num_examples: 4800
---

# Kazakh Balanced GPT-2-Style Dataset

A balanced, pre-tokenized dataset for training causal language models on Kazakh text. Built from the [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) with careful domain rebalancing and quality filtering.

## Overview

| | |
|---|---|
| **Total blocks** | 479,990 |
| **Block size** | 1,024 tokens |
| **Total tokens** | ~491M |
| **Train split** | 475,190 blocks (99%) |
| **Validation split** | 4,800 blocks (1%) |
| **Tokenizer** | [saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1) |
| **License** | Apache 2.0 |

## Domain Distribution

The original multidomain dataset is heavily skewed toward web crawl data (cc100 alone accounts for ~80% of rows). This dataset rebalances across five domains to produce a more diverse training mix:

| Domain | Source | Target % | Actual % | Blocks |
|--------|--------|----------|----------|--------|
| **oscar** | OSCAR web corpus | 47% | 46.4% | 222,497 |
| **kazakhNews** | Kazakh news articles | 25% | 25.6% | 122,985 |
| **kazakhBooks** | Kazakh literary texts | 20% | 20.2% | 96,933 |
| **leipzig** | Leipzig corpora collection | 5% | 4.9% | 23,425 |
| **cc100** | CC-100 / Wikipedia crawl | 3% | 2.9% | 14,150 |

The rationale behind these proportions:
- **oscar (47%)**: High-quality web text provides broad vocabulary and topic coverage
- **kazakhNews (25%)**: News articles contribute formal, well-edited Kazakh prose
- **kazakhBooks (20%)**: Literary texts add stylistic diversity and long-form coherence
- **leipzig (5%)**: Sentence-level corpora add linguistic variety from multiple sources
- **cc100 (3%)**: Wikipedia-sourced content adds factual knowledge (kept small due to mixed quality)

## Data Processing Pipeline

### 1. Domain-wise Loading
Each CSV file was loaded independently from the source repository to preserve domain labels.

### 2. Quality Filtering
Each text was filtered through multiple stages:
- **Language filter**: Only texts with `predicted_language == "kaz"` retained
- **Script filter**: Only texts with `contains_kaz_symbols == 1` retained
- **Kazakh character check**: Must contain at least one Kazakh-specific character (e.g., `Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І`)
- **Length filter**: Minimum 30 characters after normalization
- **Unicode normalization**: NFC normalization applied
- **Whitespace normalization**: Collapsed multiple spaces/tabs

### 3. Deduplication
Exact deduplication via MD5 hash of normalized text.

| Domain | Raw rows | After filtering |
|--------|----------|-----------------|
| oscar | 269,047 | 269,030 |
| kazakhNews | 3,264,273 | 333,214 |
| kazakhBooks | 8,423 | 4,454 |
| leipzig | 1,706,485 | 1,653,675 |
| cc100 | 19,635,580 | 17,664,550 |

Notable: kazakhNews retains only ~10% of rows after language filtering, indicating the source contains significant non-Kazakh content.

### 4. Token-Proportional Resampling
Token counts were estimated by sampling 5,000 texts per domain, computing average tokens per text, and extrapolating. Domains were then downsampled (never upsampled) so that the token budget matches target proportions:

| Domain | Texts before | Texts after |
|--------|-------------|-------------|
| oscar | 269,030 | 175,610 |
| kazakhNews | 333,214 | 333,214 (all) |
| kazakhBooks | 4,454 | 1,551 |
| leipzig | 1,653,675 | 1,004,860 |
| cc100 | 17,664,550 | 420,864 |

### 5. Tokenization and Grouping
Texts were tokenized using [saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1) (ByteLevel BPE, 32K vocab). Token sequences were concatenated per domain and split into fixed blocks of 1,024 tokens (GPT-2 style). Remainder tokens that don't fill a complete block are discarded.

### 6. Shuffling and Splitting
All blocks across domains were concatenated, shuffled with seed 42, and split 99/1 into train and validation sets.

## Schema

Each example contains:

```python
{
    "input_ids": [int] * 1024,   # Token IDs
    "labels": [int] * 1024,      # Same as input_ids (for CLM)
    "domain": str                 # Source domain label
}
```

## Usage

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("saken-tukenov/sozkz-corpus-balanced-kk-gpt2-v1")
tokenizer = AutoTokenizer.from_pretrained("saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1")

# Ready for causal language model training
train_ds = dataset["train"]
val_ds = dataset["validation"]

# Inspect a sample
sample = train_ds[0]
print(f"Domain: {sample['domain']}")
print(f"Tokens: {len(sample['input_ids'])}")
print(f"Text: {tokenizer.decode(sample['input_ids'][:50])}")
```

```python
# Domain distribution in the dataset
from collections import Counter
counts = Counter(train_ds["domain"])
total = sum(counts.values())
for domain, count in counts.most_common():
    print(f"{domain:15s}: {count:>8d} ({100*count/total:.1f}%)")
```

## Intended Use

This dataset is designed for pre-training small-to-medium causal language models (GPT-2 architecture) on Kazakh text. The balanced domain mix aims to produce models with:
- Broad vocabulary coverage (web text)
- Formal language proficiency (news)
- Stylistic range (books)
- Linguistic diversity (Leipzig, CC-100)

## Limitations

- **No upsampling**: Smaller domains (kazakhBooks, cc100) are not upsampled, so the model may still underperform on literary or encyclopedic text
- **Quality variance**: Web-crawled domains (oscar, cc100) have variable text quality despite filtering
- **Language filtering**: The `predicted_language` field may have some misclassifications
- **No paragraph/document boundaries**: GPT-2-style concatenation means block boundaries do not align with document boundaries

## Citation

If you use this dataset, please cite the source:

```bibtex
@dataset{kazakh_balanced_gpt2_2026,
  title={Kazakh Balanced GPT-2-Style Dataset},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-balanced-kk-gpt2-v1},
  note={Balanced and pre-tokenized from kz-transformers/multidomain-kazakh-dataset}
}
```
