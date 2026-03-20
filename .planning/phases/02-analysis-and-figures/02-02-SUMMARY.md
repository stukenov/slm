---
phase: 02-analysis-and-figures
plan: 02
subsystem: analysis
tags: [matplotlib, scipy, curve-fit, latex-macros, power-law, scaling]

requires:
  - phase: 02-analysis-and-figures/01
    provides: "config.py (shared constants), figures.py, tables.py, fertility.py"
provides:
  - "scaling.py with own-model power-law fit and all-models scatter plot"
  - "macros.py generating LaTeX providecommand for all model-task metrics"
  - "generate_all.py single-entry orchestrator for all analysis outputs"
affects: [02-analysis-and-figures/03, 02-analysis-and-figures/04, paper-writing]

tech-stack:
  added: [scipy]
  patterns: [power-law curve fitting with confidence bands, LaTeX macro generation]

key-files:
  created:
    - scripts/analysis/scaling.py
    - scripts/analysis/macros.py
    - scripts/analysis/generate_all.py
  modified: []

key-decisions:
  - "Used providecommand instead of newcommand to avoid LaTeX redefinition errors"
  - "Excluded gpt-oss-120b from all-models scatter to avoid compressing x-axis"
  - "Skip flags for network/GPU-dependent steps (fertility, efficiency, contamination)"

patterns-established:
  - "generate_* function pattern: takes summary dict, returns output path"
  - "generate_all.py as single orchestrator with --skip-* flags"

requirements-completed: [ANLS-02, ANLS-05]

duration: 3min
completed: 2026-03-20
---

# Phase 02 Plan 02: Scaling Curves, Macros, and Orchestrator Summary

**Scaling curve plots with scipy power-law fit, LaTeX macro generator for all 14x6 model-task metrics, and generate_all.py single-pass orchestrator**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-20T18:31:17Z
- **Completed:** 2026-03-20T18:34:43Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Own-model scaling plot with power-law curve_fit (y = a * x^b) and 95% confidence band from covariance matrix
- All-models scatter plot colored by family with log x-axis (excludes gpt-oss-120b)
- macros.py maps all 14 model keys to CamelCase LaTeX names and generates providecommand for every metric
- generate_all.py orchestrates all 6 analysis modules with skip flags for network-dependent steps

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scaling.py** - `b8d5b5b` (feat)
2. **Task 2: Create macros.py and generate_all.py** - `06f8b9e` (feat)

## Files Created/Modified
- `scripts/analysis/scaling.py` - Power-law fitted own-model curve + all-models scatter plot
- `scripts/analysis/macros.py` - LaTeX macro generation with model_key_to_latex mapping
- `scripts/analysis/generate_all.py` - Single entry point orchestrating all analysis outputs

## Decisions Made
- Used `\providecommand` instead of `\newcommand` to avoid LaTeX redefinition errors when macros.tex is included multiple times
- Excluded gpt-oss-120b from all-models scatter plot per research recommendation (compresses x-axis too much)
- Added `--skip-fertility`, `--skip-efficiency`, `--skip-contamination` flags for offline/no-GPU execution

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed scipy dependency**
- **Found during:** Task 1 (scaling.py verification)
- **Issue:** scipy not installed in project venv, curve_fit import failed
- **Fix:** Ran ensurepip + pip install scipy in .venv
- **Files modified:** .venv (runtime dependency only)
- **Verification:** Import succeeds after install
- **Committed in:** Not committed (venv is gitignored)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary runtime dependency. No scope creep.

## Issues Encountered
- No pip in .venv initially; used ensurepip to bootstrap pip, then installed scipy

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All analysis modules now have a single entry point via generate_all.py
- Ready for efficiency and contamination analysis (Plans 03/04)
- macros.tex will be generated alongside figures and tables

---
*Phase: 02-analysis-and-figures*
*Completed: 2026-03-20*
