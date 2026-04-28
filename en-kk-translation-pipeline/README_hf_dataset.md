---
language:
  - kk
  - en
license: cc-by-4.0
task_categories:
  - translation
tags:
  - kazakh
  - english
  - machine-translation
  - fineweb-edu
  - parallel-corpus
  - sozkz
size_categories:
  - 10M<n<100M
source_datasets:
  - HuggingFaceFW/fineweb-edu-score-2
dataset_info:
  features:
    - name: text_en
      dtype: string
    - name: text_kk
      dtype: string
    - name: id
      dtype: string
    - name: num_sentences
      dtype: int64
  splits:
    - name: train
      num_examples: 18000000
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# SozKZ Corpus Clean EN→KK (FineWeb-Edu) v1

Machine-translated parallel corpus of educational web texts (English → Kazakh), derived from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2).

## Overview

| | |
|---|---|
| **Source** | FineWeb-Edu Score-2 (first 20M rows) |
| **Languages** | English (en) → Kazakh (kk) |
| **Translation** | CTranslate2, NLLB-200 (en→kk), greedy decoding |
| **Filtering** | Length (50–10K chars), exact dedup (xxhash), language detection (fasttext) |
| **Rows** | ~18M (after filtering ~90% pass rate) |
| **Format** | Expandable Parquet shards (`train-XXXXX.parquet`) |

## Dataset Structure

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `text_en` | string | Original English text from FineWeb-Edu |
| `text_kk` | string | Machine-translated Kazakh text |
| `id` | string | Original document ID from FineWeb-Edu |
| `num_sentences` | int | Number of sentences in the document |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1", split="train")
print(ds[0])
# {'text_en': '...', 'text_kk': '...', 'id': '...', 'num_sentences': 5}
```

## Processing Pipeline

1. **Stream** rows from `HuggingFaceFW/fineweb-edu-score-2`
2. **Filter** with cascade: length bounds → exact dedup (xxhash64) → language detection (fasttext lid.176)
3. **Split** text into sentences (regex-based)
4. **Translate** via CTranslate2 with NLLB-200 (en→kk), float16, greedy (beam=1), max 128 input / 200 output tokens
5. **Upload** each 1M-row chunk as a Parquet shard

Translation runs on 2× NVIDIA A10 GPUs in parallel.

## Shard Layout

Shards are numbered sequentially (`train-00000.parquet`, `train-00001.parquet`, ...) and can be extended without renaming existing files.

| Shards | Source |
|--------|--------|
| `train-00000` | Existing 902K pre-translated rows |
| `train-00001` – `train-00009` | FineWeb-Edu rows 1M–10M (filtered) |
| `train-00010` – `train-00019` | FineWeb-Edu rows 10M–20M (filtered) |

## Limitations

- Machine translation quality (NLLB-200) — not human-verified
- Sentence splitting is regex-based and may introduce segmentation errors
- Educational domain bias inherited from FineWeb-Edu
- Some truncation at 128 input tokens for very long sentences

## License

CC-BY-4.0 (following FineWeb-Edu licensing)

## Citation

```bibtex
@dataset{sozkz_corpus_clean_enkk_fineweb_edu_v1,
  title={SozKZ Corpus Clean EN-KK (FineWeb-Edu) v1},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1},
  note={Machine-translated from FineWeb-Edu using NLLB-200}
}
```
