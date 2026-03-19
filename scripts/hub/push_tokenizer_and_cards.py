#!/usr/bin/env python3
"""Push tokenizer + cards to HF Hub."""

from __future__ import annotations
import os
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi, upload_file

TOKENIZER_DIR = "./tokenizers/sozkz-core-gpt2-50k-kk-base-v1"
TOKENIZER_REPO = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
DATASET_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"

TOKENIZER_CARD = """---
language:
  - kk
license: apache-2.0
library_name: transformers
tags:
  - tokenizer
  - bpe
  - gpt2
  - kazakh
pipeline_tag: text-generation
---

# Kazakh GPT-2 BPE Tokenizer (50,257 vocab)

A ByteLevel BPE tokenizer trained on a cleaned Kazakh corpus, following the GPT-2 tokenization scheme.

## Overview

| Property | Value |
|----------|-------|
| **Vocab size** | 50,257 |
| **Algorithm** | ByteLevel BPE (GPT-2 style) |
| **Training data** | [sozkz-corpus-clean-kk-text-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2) (~78K cleaned documents) |
| **Language** | Kazakh (kk) |
| **License** | Apache 2.0 |

## Vocabulary composition

- **256** byte-level base tokens
- **~49,634** BPE merges learned from Kazakh text
- **3** special tokens: `<|endoftext|>`, `<|padding|>`, `<|startoftext|>`
- **360** Unicode digit characters (Arabic-Indic, Devanagari, etc.)
- **Total: 50,257**

## Special tokens

| Token | Role |
|-------|------|
| `<|endoftext|>` | EOS / document separator |
| `<|startoftext|>` | BOS |
| `<|padding|>` | Padding |

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")

text = "Қазақстан — Орталық Азиядағы мемлекет."
tokens = tokenizer.encode(text)
print(f"{len(tokens)} tokens: {tokens}")
print(tokenizer.decode(tokens))
```

## Tokenization examples

| Text | Tokens |
|------|--------|
| Қазақстан — Орталық Азиядағы мемлекет. | 6 |
| Бүгін ауа райы жақсы болады. | 6 |
| 2024 жылы халықаралық конференция өтеді. | 7 |

## Training details

- **Source corpus**: 5 domains — OSCAR, Kazakh News, Kazakh Books, Leipzig, CC-100
- **Cleaning pipeline**: NFC normalization → script profile filter → fastText LID → junk removal → repetition filter → exact + MinHash near-dedup → domain balancing
- **min_frequency**: 2 (tokens appearing less than twice are excluded from merges)

## Intended use

Pre-training Kazakh language models (GPT-2, LLaMA-style, etc.). Optimized for Kazakh Cyrillic script with compact encoding.

## Citation

```bibtex
@misc{sozkz-core-gpt2-50k-kk-base-v1,
  author = {Saken Tukenov},
  title = {Kazakh GPT-2 BPE Tokenizer (50K vocab)},
  year = {2026},
  url = {https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1}
}
```
"""

DATASET_CARD = """---
language:
  - kk
license: apache-2.0
task_categories:
  - text-generation
tags:
  - kazakh
  - pretrain
  - cleaned
  - deduplicated
size_categories:
  - 10K<n<100K
source_datasets:
  - kz-transformers/multidomain-kazakh-dataset
---

# Kazakh Clean Pretrain Text

A deeply cleaned and deduplicated Kazakh text corpus for language model pre-training.

## Overview

| Property | Value |
|----------|-------|
| **Train texts** | 77,879 |
| **Validation texts** | 786 |
| **Split ratio** | 99/1 |
| **Language** | Kazakh (kk) |
| **Format** | Clean text (`text` + `domain` fields) |
| **License** | Apache 2.0 |

## Source domains

| Domain | Source | Target proportion |
|--------|--------|-------------------|
| OSCAR | Web crawl (OSCAR corpus) | 47% |
| Kazakh News | Kazakh news articles | 25% |
| Kazakh Books | Digitized Kazakh books | 20% |
| Leipzig | Leipzig corpora collection | 5% |
| CC-100 | Common Crawl monolingual | 3% |

Raw source: [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) (~24.9M raw documents).

## Cleaning pipeline

The corpus was processed through a 9-stage cleaning pipeline:

1. **Normalization** — NFC unicode normalization, control character removal, whitespace collapsing, truncation at 100K chars
2. **Kazakh character check** — must contain Kazakh-specific Cyrillic characters (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І)
3. **Script profile filter** — Cyrillic ≥ 60%, Latin ≤ 25%, Arabic ≤ 5%, Other ≤ 10%
4. **Language ID** — fastText `lid.176.bin`: Kazakh confidence ≥ 0.5, gap vs Russian/English ≥ 0.1
5. **Technical junk removal** — URL density, HTML tags, boilerplate patterns ("Теги:", "kz.kz", "©"), special character ratio > 15%
6. **Repetition filter** — repeated 10-gram ratio ≥ 30% or gzip compression ratio < 0.2
7. **Exact dedup** — MD5 hash deduplication within each domain
8. **Near dedup** — MinHash LSH across all domains (threshold=0.8, 128 permutations, 5-gram shingles)
9. **Domain balancing** — downsampling to target proportions

## Dataset structure

```python
from datasets import load_dataset

ds = load_dataset("saken-tukenov/sozkz-corpus-clean-kk-text-v2")
print(ds["train"][0])
# {'text': '...', 'domain': 'oscar'}
```

Each example has:
- `text` (str): Clean Kazakh text
- `domain` (str): Source domain label

## Usage

This dataset is intended for:
- Pre-training Kazakh language models
- Fine-tuning multilingual models on Kazakh
- Kazakh NLP research

Recommended tokenizer: [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1)

## Citation

```bibtex
@misc{sozkz-corpus-clean-kk-pretrain-v2,
  author = {Saken Tukenov},
  title = {Kazakh Clean Pretrain Text},
  year = {2026},
  url = {https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2}
}
```
"""


def main():
    # 1. Push tokenizer
    print("Loading tokenizer...")
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"  Vocab size: {tok.vocab_size}")

    print(f"Pushing tokenizer to {TOKENIZER_REPO}...")
    tok.push_to_hub(TOKENIZER_REPO, private=False)

    # Push tokenizer README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=TOKENIZER_CARD.encode(),
        path_in_repo="README.md",
        repo_id=TOKENIZER_REPO,
        repo_type="model",
    )
    print(f"  Tokenizer published: https://huggingface.co/{TOKENIZER_REPO}")

    # 2. Push dataset card
    print(f"Pushing dataset card to {DATASET_REPO}...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  Dataset card published: https://huggingface.co/datasets/{DATASET_REPO}")

    print("Done!")


if __name__ == "__main__":
    main()
