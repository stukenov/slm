# Translation Pipeline v2 — Benchmark Results

## Test Environment
- **Pod:** RunPod RTX A4000 (16GB), $0.17/hr
- **Image:** runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- **Data:** FineWeb-Edu sample-10BT, first 100-200 rows

## Optimization History

| Version | Change | E2E sents/sec | Speedup |
|---------|--------|---------------|---------|
| v0 | Original baseline | 717 | 1.0x |
| v1 | Fast char_similarity (early-exit) | 1,070 | 1.49x |
| v2 | + batch SP encode + global sort + int8_float16 | 1,301 | 1.81x |

### Breakdown (v2, 200 docs / 6896 sentences)

| Stage | Time | % |
|-------|------|---|
| Split (sentence_splitter) | 0.079s | 3% |
| Translate (GPU) | 2.850s | 93% |
| Postprocess | 0.149s | 5% |

GPU is the bottleneck at 93%. Further speedup requires faster/more GPUs.

## Optimizations Applied

1. **int8_float16 quantization** — same quality, slight speed gain on Marian NMT
2. **SentencePiece batch encode** — 3.4x faster tokenization vs per-sentence loop
3. **Global sort by token length** — optimal GPU batch packing, reduces padding waste
4. **Fast char_similarity** — early exit on length ratio + set overlap before expensive SequenceMatcher (8.6x postproc speedup)

## Optimizations Tested but NOT Applied

| Idea | Result | Why |
|------|--------|-----|
| `batch_type="tokens"` | 666-1065 sents/sec | Slower than default on this model |
| Dynamic `max_decoding_length` | 1,773 in isolation, 1,227 in E2E | Helps on sorted batches, hurts combined |
| `asynchronous=True` | API different in this CT2 version | Not compatible |
| `int8` (pure) | 4,540 sents/sec (benchmark) | Slightly slower than int8_float16 |

## E2E Test Results (100 docs)

- **100% documents translated** (none fully empty)
- **90.3% sentences translated**, 9.7% skipped
- **Confidence mean:** avg 0.925, range [0.80, 0.97]
- **Confidence min:** avg 0.753, range [0.38, 0.96]
- **Skip reasons:** pre-filter (noisy/short/long) + post-filter (copy-through, repetition)

## Cost Estimates for Full sample-10BT (~319M sentences)

Assumes linear GPU scaling. Prices from RunPod as of 2026-03-26.

| Config | sents/sec | Hours | Cost |
|--------|-----------|-------|------|
| **2x RTX 3090** | 4,380 | 20h | **$8.9** |
| **4x RTX 4090** | 12,160 | 7.3h | **$10.8** |
| 2x RTX 4090 | 6,080 | 14.6h | $10.8 |
| 1x RTX A4000 | 1,301 | 68h | $11.6 |
| 1x RTX 4090 | 3,040 | 29h | $12.8 |
| 1x A100 80GB | 4,870 | 18h | $29.8 |

**Best value:** 2x RTX 3090 ($8.9)
**Best speed/value:** 4x RTX 4090 ($10.8, 7.3h)

## RunPod Notes

- RunPod API key stored in `puh/ansible/group_vars/all.yml`
- `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` works, `pytorch/pytorch:*` did NOT boot
- HPLT model needs vocab fix before CT2 conversion (see `setup_model.sh`)
- Random sampling from streaming dataset is very slow (skip millions of rows); use `--sequential` for testing
