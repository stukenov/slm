# Translation Pipeline v2 — FineWeb-Edu EN→KK

Modular pipeline for translating FineWeb-Edu from English to Kazakh with sentence-level confidence scoring and quality filtering.

## Architecture

```
FineWeb-Edu (streaming) → sentence_splitter → translator → postprocessor → parquet → HF Hub
```

### Modules

| Module | Purpose |
|--------|---------|
| `config.py` | All constants and thresholds |
| `sentence_splitter.py` | Split docs into sentences, pre-filter noise |
| `translator.py` | CTranslate2 translation with per-sentence confidence |
| `postprocessor.py` | Post-translation quality checks, document reassembly |
| `filters.py` | Sentence-level pre/post filter functions |
| `pipeline.py` | Orchestrator: chunking, multi-GPU, resume, HF upload |
| `test_pipeline.py` | E2E test on random rows with detailed report |

## Translation Model

- **HPLT/translate-en-kk-v2.0-hplt_opus** (Marian NMT)
- CTranslate2 runtime, FP16, greedy decoding
- Model files: `model_ct2/` (CTranslate2 format), `model_cache/model.en-kk.spm`

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `original_id` | str | Original document ID from FineWeb-Edu |
| `content_hash` | str | xxhash of `text_en` (for incremental dedup) |
| `text_en` | str | Original English text |
| `text_kk` | str | Kazakh translation (empty if all sentences failed) |
| `confidence_mean` | float | Mean per-sentence confidence (0-1) |
| `confidence_min` | float | Min per-sentence confidence (0-1) |
| `sentences_total` | int | Total sentences in document |
| `sentences_translated` | int | Successfully translated sentences |
| `sentences_skipped` | int | Skipped sentences (pre + post filter) |

## Filtering Scheme

### Pre-translation (per sentence)

Sentence is skipped if:
- **>30% non-alpha characters** (math formulas, code, special chars)
- **<3 words** (too short to be meaningful)
- **>512 characters** (exceeds CTranslate2 input limit)

### Post-translation (per sentence)

Translation is discarded if:
- **>90% character similarity** with original (copy-through, not translated)
- **3+ n-gram repeats** (model looping: "және және және...")
- **Length ratio >3x or <0.3x** vs original (hallucination or truncation)
- **Disproportionate special characters** vs original (garbage output)

### Document level

- Rows are NEVER deleted — every document stays with `text_en` and `original_id`
- `text_kk` may be empty (all sentences skipped) or partial
- Users filter by `confidence_mean` / `confidence_min` threshold

## Incremental Translation

Designed for incremental translation across FineWeb-Edu splits:

1. Translate `sample-10BT` → upload as split `sample-10BT`
2. Later: load `content_hash` set from sample-10BT
3. Stream `sample-100BT`, skip already-translated rows
4. Translate remaining ~90BT, upload as `sample-100BT`

## Quick Start

```bash
# Install dependencies
pip install ctranslate2 sentencepiece datasets huggingface_hub xxhash

# Download and convert model
bash setup_model.sh

# Run unit tests
python -m pytest test_filters.py test_sentence_splitter.py test_translator.py test_postprocessor.py -v

# E2E test (CPU, 10 rows)
python test_pipeline.py --num-rows 10 --cpu --sequential

# E2E test (GPU, 100 random rows)
python test_pipeline.py --num-rows 100

# Full pipeline (1 GPU, first chunk)
python pipeline.py --num-gpus 1 --start-chunk 0 --end-chunk 1

# Full pipeline (2 GPUs, auto-resume)
python pipeline.py --num-gpus 2 --start-chunk auto
```

## HuggingFace

- **Repo:** `stukenov/sozkz-fineweb-edu-kk-v2`
- **Split:** `sample-10BT` (mirrors FineWeb-Edu split naming)
