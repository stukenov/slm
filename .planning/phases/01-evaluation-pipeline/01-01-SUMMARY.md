---
phase: 01-evaluation-pipeline
plan: 01
subsystem: evaluation
tags: [model-registry, mc-qa, bpb, bits-per-byte, likelihood-scoring, kazakh-nlp]

# Dependency graph
requires: []
provides:
  - "MODEL_REGISTRY dict with 14 models (4 own + 9 HF competitors + 1 API)"
  - "load_model() with bf16/4-bit quantization and device fallback"
  - "score_choices() full-completion likelihood scoring for MC QA"
  - "compute_bpb() bits-per-byte on external Kazakh text with sliding window"
affects: [01-evaluation-pipeline]

# Tech tracking
tech-stack:
  added: [bitsandbytes, requests]
  patterns: [full-completion-likelihood-scoring, sliding-window-bpb, utf8-byte-counting]

key-files:
  created:
    - scripts/eval/model_registry.py
    - scripts/eval/eval_bpb.py
  modified:
    - scripts/eval/eval_mc_bench.py

key-decisions:
  - "Full answer text likelihood scoring replaces single-token logit comparison for MC QA"
  - "UTF-8 byte counting (not char counting) for BPB -- Kazakh Cyrillic is 2 bytes per char"
  - "Plain-text prompt format for base model evaluation (no chat templates)"
  - "Length-normalized log-prob scoring for fair comparison across varying choice lengths"

patterns-established:
  - "Registry pattern: MODEL_REGISTRY dict + load_model() for all eval scripts"
  - "CLI pattern: --model, --output, --limit, --quantize flags on all eval scripts"
  - "Output pattern: JSON to paper/results/{task}/{model_short}.json with ISO 8601 timestamps"
  - "API pattern: separate score_*_api() functions for GPT-OSS-120B"

requirements-completed: [EVAL-07, EVAL-02, EVAL-01]

# Metrics
duration: 6min
completed: 2026-03-20
---

# Phase 01 Plan 01: Evaluation Foundation Summary

**Model registry with 14 models, fixed MC QA scoring via full-completion likelihood, and BPB computation with UTF-8 byte counting on FLORES/Wikipedia**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-20T17:21:48Z
- **Completed:** 2026-03-20T17:27:42Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Model registry with all 14 models (4 own SozKZ + 9 HF competitors + 1 GPT-OSS API), load helpers with bf16 and optional 4-bit quantization
- Fixed critical MC QA scoring bug: replaced single-token logit comparison with full answer text likelihood scoring (log_softmax over choice tokens, length-normalized)
- BPB evaluation script with correct UTF-8 byte counting, sliding window for long texts, FLORES-200 and Wikipedia corpus loaders

## Task Commits

Each task was committed atomically:

1. **Task 1: Create model registry with load helpers** - `b95a332` (feat)
2. **Task 2: Fix MC QA scoring bug and rewrite eval_mc_bench.py** - `39efc63` (fix)
3. **Task 3: Create BPB evaluation script** - `0e95848` (feat)

## Files Created/Modified
- `scripts/eval/model_registry.py` - Central registry of 14 models with load_model(), list_models(), device fallback
- `scripts/eval/eval_mc_bench.py` - MC QA evaluation with score_choices() full-completion likelihood scoring
- `scripts/eval/eval_bpb.py` - BPB computation with compute_bpb() sliding window, FLORES/Wikipedia loaders

## Decisions Made
- Used full answer text likelihood scoring (sum log-probs, length-normalized) instead of single-token logit comparison -- fixes the 10% accuracy bug
- UTF-8 byte counting for BPB (Kazakh Cyrillic = 2 bytes per char) -- critical for correct bits-per-byte metric
- Plain-text prompt format for all base model evaluations (no chatml/alpaca templates)
- score_choices() exported at module level for reuse by sentiment, Belebele, SIB-200 scripts in Plan 02

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Project has multiple Python venvs (.venv lacks torch, .venv-cloud has both torch and transformers) -- used .venv-cloud for verification

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Model registry ready for consumption by all downstream eval scripts (Plan 02: sentiment, Belebele, SIB-200, NER)
- score_choices() pattern established for reuse in Plan 02 MC-format benchmarks
- BPB script ready for batch execution across all 14 models

## Self-Check: PASSED

All 3 created files exist. All 3 task commits verified (b95a332, 39efc63, 0e95848).

---
*Phase: 01-evaluation-pipeline*
*Completed: 2026-03-20*
