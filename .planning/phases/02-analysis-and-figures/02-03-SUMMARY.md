---
phase: 02-analysis-and-figures
plan: 03
subsystem: analysis
tags: [contamination, n-gram, benchmark, data-quality]

requires:
  - phase: 01-project-setup
    provides: eval pipeline and benchmark dataset definitions
provides:
  - contamination.py script with CLI for n-gram overlap detection
  - paper/results/contamination.json output for generate_all.py
affects: [02-analysis-and-figures, paper-writing]

tech-stack:
  added: [datasets (streaming mode)]
  patterns: [character-level 13-gram overlap (GPT-3 methodology), streaming sampling for large corpora]

key-files:
  created:
    - scripts/analysis/contamination.py
  modified: []

key-decisions:
  - "Named functions instead of lambdas for benchmark text extraction (serialization-safe)"
  - "Streaming + shuffle + take for memory-efficient training corpus sampling"
  - "Plan test assertion was wrong (expected 11 n-grams for 16-char string, correct is 12); implementation follows correct math"

patterns-established:
  - "Benchmark configs as dict with text_fn/filter_fn callables for uniform processing"

requirements-completed: [ANLS-04]

duration: 2min
completed: 2026-03-20
---

# Phase 02 Plan 03: Contamination Check Summary

**Character-level 13-gram overlap checker (GPT-3 methodology) with streaming sampling and per-benchmark contamination reporting**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T18:24:34Z
- **Completed:** 2026-03-20T18:26:30Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Built contamination.py with extract_char_ngrams, compute_ngram_overlap, and run_contamination_check
- Configured 5 benchmark datasets (MC QA, Belebele, Sentiment, NER, SIB-200) with named text extraction functions
- CLI with configurable training dataset, sample size, n-gram size, and output path
- Results saved as JSON with per-benchmark contamination rate and overlap distribution (mean/median/p95/max)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create contamination.py with n-gram overlap computation and CLI** - `b29100f` (feat)

## Files Created/Modified
- `scripts/analysis/contamination.py` - N-gram overlap contamination checker with streaming dataset sampling, 5 benchmark configs, and CLI

## Decisions Made
- Used named functions (_mc_qa_text, _belebele_text, etc.) instead of lambdas for benchmark text extraction -- lambdas cannot be serialized
- Streaming mode with shuffle+take for training corpus sampling -- avoids loading full 23.6M dataset into memory
- Added skipped_short counter to overlap results for transparency on texts shorter than n-gram size

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect test assertion in plan verification**
- **Found during:** Task 1 verification
- **Issue:** Plan's automated test asserted len(ngrams)==11 for "hello world test" with n=5, but correct count is 12 (16 chars - 5 + 1 = 12)
- **Fix:** Ran corrected assertion (==12) which passes; implementation is mathematically correct
- **Verification:** python3 -c confirms 12 n-grams for 16-char string

---

**Total deviations:** 1 auto-fixed (1 bug in test spec)
**Impact on plan:** Trivial -- plan's test math was off by one, implementation is correct.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- contamination.py ready to run on server (needs 16GB+ RAM for 500K sample)
- Output JSON at paper/results/contamination.json will be consumed by generate_all.py
- All 5 benchmark datasets configured with proper splits and filters

---
*Phase: 02-analysis-and-figures*
*Completed: 2026-03-20*
