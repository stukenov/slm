---
language:
- kk
license: apache-2.0
task_categories:
- text-generation
tags:
- kazakh
- language-modeling
- cleaned
- deduplicated
- corpus
size_categories:
- 10M<n<100M
dataset_info:
  features:
  - name: text
    dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_examples: 13563018
  - name: validation
    num_examples: 137000
---

# SozKZ Corpus Clean v3 — Cleaned Kazakh Text Corpus

A large-scale cleaned and deduplicated Kazakh text corpus assembled from 18 public sources. Designed for pre-training causal language models on Kazakh text.

## Overview

| | |
|---|---|
| **Total texts** | 13,700,018 |
| **Train split** | ~13,563,018 (99%) |
| **Validation split** | ~137,000 (1%) |
| **Raw input** | 28,431,116 texts |
| **Pass rate** | 48.2% |
| **Dedup removed** | 3,170,330 |
| **License** | Apache 2.0 |

## Sources

| Source | Raw | Clean | Pass % | Type |
|--------|----:|------:|-------:|------|
| culturax | 2,731,934 | 2,707,214 | 99.1% | Web crawl (CulturaX) |
| madlad400 | 1,807,996 | 1,794,308 | 99.2% | Web crawl (MADLAD-400) |
| hplt_new | 2,637,330 | 2,204,165 | 83.6% | Web crawl (HPLT) |
| mc4 | 2,371,528 | 1,906,763 | 80.4% | Web crawl (mC4) |
| cc100 | 1,721,481 | 1,365,739 | 79.3% | Web crawl (CC-100) |
| kazparc_sync | 9,632,030 | 1,370,243 | 14.2% | Parallel corpus (KazParC) |
| md_leipzig | 1,706,485 | 1,128,122 | 66.1% | Leipzig corpora collection |
| md_kazakhNews | 3,264,273 | 288,247 | 8.8% | Kazakh news articles |
| md_oscar | 269,047 | 239,807 | 89.1% | OSCAR web corpus |
| moscar | 245,869 | 231,693 | 94.2% | OSCAR (secondary) |
| kazparc | 1,647,560 | 165,754 | 10.1% | Parallel corpus (KazParC) |
| kazsandra | 146,253 | 40,236 | 27.5% | Kazsandra corpus |
| md_kazakhBooks | 8,423 | 20,482 | — | Kazakh literary texts* |
| wikipedia | ~20,000 | ~19,988 | ~99% | Kazakh Wikipedia |
| belebele | 900 | 488 | 54.2% | Belebele benchmark |
| sib200 | 701 | 648 | 92.4% | SIB-200 benchmark |
| wikiann | ~1,000 | ~966 | — | WikiANN NER |

*kazakhBooks raw count < clean count because long texts are split into chunks during processing.

### Data Sources

- **[kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset)**: oscar, kazakhNews, kazakhBooks, leipzig CSVs
- **Collected parquets (wave 1)**: culturax, hplt_new, madlad400, mc4, cc100, kazparc, kazparc_sync, moscar, wikipedia, belebele, sib200, wikiann, kazsandra
- **Collected parquets (wave 2)**: additional sources

## Cleaning Pipeline

Nine-stage filter pipeline, ordered from fast to slow:

| # | Filter | Description | Threshold |
|---|--------|-------------|-----------|
| 1 | OSCAR dict fix | Unwrap `{'text': '...'}` wrapper | — |
| 2 | NFC normalize | Unicode NFC + control chars + whitespace normalization | — |
| 3 | Min length | Minimum text length and word count | ≥50 chars, ≥10 words |
| 4 | Kazakh chars | Must contain ≥1 Kazakh-specific character (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І) | ≥1 char |
| 5 | Script profile | Cyrillic ≥60%, Latin ≤25% | cyr≥0.60, lat≤0.25 |
| 6 | Junk filter | URL density, HTML tags, special char ratio, boilerplate | URL≤5/1K, HTML≤5, special≤40% |
| 7 | Gzip repetition | Compression ratio to detect repetitive text | ratio≥0.20 |
| 8 | FastText LID | Language identification (kk≥0.5, gap to rivals ≥0.1) | kk≥0.50, gap≥0.10 |
| 9 | Exact dedup | MD5 hash deduplication across all sources | — |

Long texts (>50K chars) are split into chunks at paragraph/sentence boundaries before filtering.

### Top rejection reasons

| Reason | Count | % of rejected |
|--------|------:|---:|
| no_kaz_chars | 7,713,488 | 52.3% |
| dedup | 3,170,330 | 21.5% |
| too_few_words | 1,672,400 | 11.3% |
| too_short | 1,627,752 | 11.0% |
| lid_rejected | 246,943 | 1.7% |
| script_profile | 195,774 | 1.3% |
| gzip_repetition | 127,404 | 0.9% |
| junk | 114,098 | 0.8% |

## Schema

```python
{
    "text": str,    # Cleaned text
    "source": str   # Source identifier (e.g., "culturax", "md_oscar", "cc100")
}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("saken-tukenov/sozkz-corpus-clean-v3")

train = ds["train"]
val = ds["validation"]

# Check source distribution
from collections import Counter
counts = Counter(train["source"])
for src, n in counts.most_common():
    print(f"{src:20s}: {n:>10,d}")

# Sample texts
for i in range(3):
    sample = train[i]
    print(f"[{sample['source']}] {sample['text'][:200]}")
```

## Intended Use

Pre-training small-to-medium language models on Kazakh text. The corpus provides broad coverage across web text, news, literature, encyclopedic content, and parallel corpora.

## Limitations

- **No quality scoring**: All texts that pass filters are treated equally; no quality-based weighting
- **Domain imbalance**: Web crawl sources (culturax, madlad, hplt) dominate the corpus
- **LID errors**: FastText LID may misclassify some Kazakh texts as related Turkic languages (Kyrgyz, Bashkir)
- **Parallel corpus residue**: kazparc/kazparc_sync texts are Kazakh-side extracts from parallel data; some may lack natural flow

## Citation

```bibtex
@dataset{sozkz_corpus_clean_v3_2026,
  title={SozKZ Corpus Clean v3: Cleaned Kazakh Text Corpus},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-v3}
}
```
