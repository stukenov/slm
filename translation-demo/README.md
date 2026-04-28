# EN→KK Translation Pipeline

Mass translation of English texts to Kazakh using **HPLT/translate-en-kk-v2.0-hplt_opus** (Marian NMT) via CTranslate2.

## What's inside

| File | Description |
|------|-------------|
| `translate.py` | Demo: download model, convert to CT2, translate 10 test sentences |
| `download_translation_model.py` | Standalone model download + CT2 conversion |
| `pipeline.py` | FineWeb-Edu EN→KK pipeline (100K–1M rows, multi-GPU) |
| `pipeline_bulk.py` | Bulk pipeline: 1M-row shards, auto-resume, HF upload |
| `pipeline_10bt.py` | 10B-token variant: documents + sentence pairs output |
| `translate_instruct.py` | Translate instruct/chat datasets (ChatML format) |
| `filters.py` | Pre-translation filters: length, exact dedup, lang detect, MinHash |
| `benchmark.py` | CPU vs GPU speed benchmark |
| `upload_hf.py` | Upload parquet to HuggingFace Hub |
| `upload_incremental.py` | Watch for checkpoints and upload incrementally |
| `report_stats.py` | Generate stats report for uploaded dataset |
| `README_hf_dataset.md` | HuggingFace dataset card template |

## Translation model

- **Model:** [HPLT/translate-en-kk-v2.0-hplt_opus](https://huggingface.co/HPLT/translate-en-kk-v2.0-hplt_opus) (Marian NMT)
- **Runtime:** CTranslate2 (FP16 on GPU, greedy decoding)
- **Tokenizer:** SentencePiece (`model.en-kk.spm`)

## Quick start

```bash
# Install dependencies
pip install ctranslate2 sentencepiece datasets huggingface_hub

# Download and convert model
python download_translation_model.py

# Test translation (CPU)
python translate.py

# Benchmark GPU speed
python benchmark.py

# Translate 1000 rows from FineWeb-Edu (1 GPU)
python pipeline.py --num-rows 1000 --num-gpus 1

# Smoke test
python pipeline.py --smoke-test
```

## Multi-GPU bulk translation

```bash
# Translate with pre-filtering, 2 GPUs, auto-resume
python pipeline_bulk.py --num-gpus 2 --filter --start-chunk 0 --end-chunk 10

# Resume interrupted run
python pipeline_bulk.py --num-gpus 2 --filter --start-chunk auto
```

## Translate instruct dataset

```bash
# Smoke test (100 rows, CPU)
python translate_instruct.py --smoke-test

# Full run (2 GPUs, resume-capable)
python translate_instruct.py --num-gpus 2 --resume

# Translate and upload to HF
python translate_instruct.py --num-gpus 2 --upload --repo your-org/your-dataset
```

## Pipeline architecture

1. **Stream/load** source dataset (FineWeb-Edu, instruct, etc.)
2. **Filter** (optional): length bounds → exact dedup (xxhash) → fasttext lang detect → MinHash near-dedup
3. **Split** text into sentences (regex-based)
4. **Batch translate** via CTranslate2 (sorted by length for minimal padding)
5. **Reassemble** translated sentences back into documents
6. **Checkpoint** periodically to parquet (resume-capable)
7. **Upload** to HuggingFace Hub

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 4096 | Sentences per CT2 batch |
| `--beam-size` | 1 | Beam size (1 = greedy, fastest) |
| `--max-input-length` | 128 | Truncate input beyond this |
| `--max-decoding-length` | 200 | Max output tokens |
| `--compute-type` | float16 | CT2 compute type |
| `--checkpoint-every` | 100000 | Rows between checkpoints |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for fast inference)
- `ctranslate2`, `sentencepiece`, `datasets`, `huggingface_hub`
- Optional: `fasttext`, `xxhash`, `datasketch` (for filtering)
