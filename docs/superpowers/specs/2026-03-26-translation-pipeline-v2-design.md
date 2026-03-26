# Translation Pipeline v2 — Design Spec

## Overview

New modular EN→KK translation pipeline for FineWeb-Edu, designed for incremental translation (sample-10BT now, sample-100BT later without re-translating).

## Requirements

1. Translate FineWeb-Edu `sample-10BT` split from English to Kazakh
2. Sentence-level translation with document reassembly
3. Incremental: when sample-100BT arrives, only translate the remaining ~90BT
4. Pre-translation noise filtering (math, code, formulas)
5. Post-translation quality checks (duplicates, artifacts, repetition)
6. Per-sentence confidence scores aggregated to document level (mean + min)
7. Failed/skipped translations leave `text_kk` empty, row preserved with `text_en`
8. Test end-to-end on cheap RunPod GPU before full run
9. All code and docs in `translation-pipeline-v2/`

## Project Structure

```
translation-pipeline-v2/
├── README.md                # Architecture, filtering scheme, requirements
├── config.py                # Constants: thresholds, HF repo names, model params
├── sentence_splitter.py     # Split docs into sentences + pre-filter noise
├── translator.py            # CTranslate2 translation + confidence scores
├── postprocessor.py         # Quality checks, dedup, document reassembly
├── pipeline.py              # Orchestrator: 1M chunks, resume, HF upload
├── filters.py               # Filtering utilities (noise, formulas, artifacts)
└── test_pipeline.py         # E2E test on random rows + manual inspection
```

## Translation Model

- **Model:** HPLT/translate-en-kk-v2.0-hplt_opus (Marian NMT)
- **Runtime:** CTranslate2, FP16, greedy decoding (beam_size=1)
- **Tokenizer:** SentencePiece (model.en-kk.spm)

## Data Flow

```
FineWeb-Edu (sample-10BT, streaming)
    │
    ▼
LOAD (pipeline.py)
    - Stream by 1M-doc chunks
    - Save original_id + content_hash (xxhash)
    │
    ▼
SPLIT (sentence_splitter.py)
    - Split into sentences
    - Pre-filter: >30% non-alpha chars → skip sentence
    - <3 words → skip
    - >512 chars → skip
    - Preserve paragraph boundaries
    │
    ▼
TRANSLATE (translator.py)
    - CTranslate2, FP16, greedy
    - Batch sorted by length (minimal padding)
    - return_scores=True → log-prob per token
    - Compute confidence per sentence (mean log-prob)
    │
    ▼
POSTPROCESS (postprocessor.py)
    - Translation ≈ original (>90% similarity) → skip
    - Repeated n-grams (3+ repeats) → skip (model looping)
    - Garbage chars (disproportionate special chars) → skip
    - Length ratio >3x or <0.3x vs original → skip
    - Reassemble into document preserving paragraph breaks
    - Compute confidence_mean and confidence_min per doc
    - Partial translations OK — confidence reflects quality
    │
    ▼
OUTPUT (parquet → HF Hub)
    Columns:
    - original_id       (str, from FineWeb-Edu)
    - content_hash      (str, xxhash of text_en)
    - text_en           (str, original English)
    - text_kk           (str, translation or empty)
    - confidence_mean   (float, mean sentence confidence)
    - confidence_min    (float, min sentence confidence)
    - sentences_total   (int)
    - sentences_translated (int)
    - sentences_skipped (int)
```

## HuggingFace Output

- **Repo:** `stukenov/sozkz-fineweb-edu-kk-v2`
- **Split naming mirrors FineWeb-Edu:** `sample-10BT`
- Future: `sample-100BT` (translate only rows not in sample-10BT, matched by content_hash)

## Incremental Translation Strategy

When translating sample-100BT:
1. Load set of `content_hash` values from existing sample-10BT dataset
2. Stream sample-100BT, skip any row whose content_hash is already translated
3. Translate remaining ~90BT
4. Upload as `sample-100BT` split (full 100BT, merging old + new)

## Filtering Rules

### Pre-translation (per sentence)

| Rule | Threshold | Action |
|------|-----------|--------|
| Non-alpha ratio | >30% of chars are not letters | Skip sentence |
| Too short | <3 words | Skip sentence |
| Too long | >512 chars | Skip sentence |

### Post-translation (per sentence)

| Rule | Threshold | Action |
|------|-----------|--------|
| Translation ≈ original | >90% char similarity | Skip sentence |
| N-gram repetition | 3+ repeats of same n-gram | Skip sentence |
| Length ratio | >3x or <0.3x vs original | Skip sentence |
| Garbage chars | Disproportionate special chars vs original | Skip sentence |

### Document level

- Nothing is deleted from the dataset — every row stays with `text_en` and `original_id`
- `text_kk` = reassembled translated sentences (may be partial)
- If 0 sentences translated → `text_kk` = empty string
- `confidence_mean` and `confidence_min` let the user filter by quality threshold later

## Configuration

```python
# Model
CT2_MODEL = "HPLT/translate-en-kk-v2.0-hplt_opus"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 4096
BEAM_SIZE = 1
MAX_INPUT_LENGTH = 128
MAX_DECODING_LENGTH = 200

# Source
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"
ROWS_PER_CHUNK = 1_000_000

# HF output
HF_REPO = "stukenov/sozkz-fineweb-edu-kk-v2"

# Pre-translation filters
MIN_WORDS_PER_SENTENCE = 3
MAX_SENTENCE_LENGTH = 512
NON_ALPHA_THRESHOLD = 0.3

# Post-translation filters
DUPLICATE_SIMILARITY_THRESHOLD = 0.9
LENGTH_RATIO_MAX = 3.0
LENGTH_RATIO_MIN = 0.3
NGRAM_REPEAT_THRESHOLD = 3

# Testing
TEST_SAMPLE_SIZE = 100
VALIDATION_SAMPLE_SIZE = 1000
```

## Testing Strategy

1. **Cheap GPU on RunPod** (RTX 3060/3070, ~$0.15-0.25/hr)
2. `test_pipeline.py` takes 100 random rows from sample-10BT
3. Runs full pipeline end-to-end
4. Outputs detailed report:
   - Sentences skipped per stage with reasons
   - Examples of skipped sentences (EN + reason)
   - Examples of translated sentences (EN → KK + confidence)
   - Borderline cases (low confidence)
   - Summary stats: % translated, avg confidence, distribution
5. **Iterative tuning:** inspect results, adjust thresholds, re-run on fresh 100 rows
6. **Final validation:** 1000 rows to confirm thresholds are stable
7. Fix all thresholds in `config.py`
8. Full run on sample-10BT on powerful GPU
