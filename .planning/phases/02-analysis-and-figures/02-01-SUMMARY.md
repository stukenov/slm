---
phase: 02-analysis-and-figures
plan: 01
subsystem: analysis
tags: [matplotlib, pandas, latex, tokenizer-fertility, bar-chart, booktabs]

requires:
  - phase: 01-project-setup
    provides: "eval pipeline with model_registry.py and aggregate_results.py producing summary.json"
provides:
  - "scripts/analysis/config.py with FAMILY_COLORS, MODEL_ORDER, setup_academic_style()"
  - "scripts/analysis/fertility.py with compute_fertility(), generate_fertility_chart(), generate_fertility_table()"
  - "scripts/analysis/figures.py with generate_comparison_bar()"
  - "scripts/analysis/tables.py with generate_comparison_table()"
affects: [02-02, 02-03, 02-04]

tech-stack:
  added: [matplotlib, pandas]
  patterns: [academic-matplotlib-style, family-based-coloring, booktabs-latex]

key-files:
  created:
    - scripts/analysis/__init__.py
    - scripts/analysis/config.py
    - scripts/analysis/fertility.py
    - scripts/analysis/figures.py
    - scripts/analysis/tables.py
  modified: []

key-decisions:
  - "Comparison bar chart uses x-axis=tasks, bars=models (cleaner for many models)"
  - "BPB excluded from bar chart (inverse scale), shown only in LaTeX table"
  - "matplotlib Agg backend at module level for headless rendering"

patterns-established:
  - "All analysis modules import shared constants from config.py"
  - "setup_academic_style() called before every figure generation"
  - "Output to paper/figures/ (PDF+PNG) and paper/tables/ (TeX)"

requirements-completed: [ANLS-01, ANLS-05]

duration: 4min
completed: 2026-03-20
---

# Phase 02 Plan 01: Analysis Config & Core Figures Summary

**Shared analysis config with family colors/ordering, tokenizer fertility analysis (chart + table), grouped bar comparison chart, and LaTeX comparison table with booktabs**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T18:18:33Z
- **Completed:** 2026-03-20T18:22:50Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Created shared config.py with 6 family colors, 14-model ordering, academic matplotlib style (Type 1 fonts)
- Built fertility.py computing chars/token for 5 tokenizer families using FLORES-200 Kazakh devtest
- Built figures.py with grouped bar chart (models x accuracy tasks, family coloring)
- Built tables.py with LaTeX comparison table (bold best values, booktabs formatting)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create config.py with shared constants** - `afc715e` (feat)
2. **Task 2: Create fertility.py with tokenizer fertility analysis** - `2bfa52e` (feat)
3. **Task 3: Create figures.py and tables.py** - `87de59b` (feat)

## Files Created/Modified
- `scripts/analysis/__init__.py` - Package init
- `scripts/analysis/config.py` - Family colors, model ordering, academic matplotlib style, shared constants
- `scripts/analysis/fertility.py` - Tokenizer fertility computation, horizontal bar chart, LaTeX table
- `scripts/analysis/figures.py` - Grouped bar comparison chart across accuracy tasks
- `scripts/analysis/tables.py` - LaTeX comparison table with bold best values

## Decisions Made
- Comparison bar chart uses x-axis=tasks with bars grouped by model (cleaner for 14 models across 5 tasks)
- BPB excluded from bar chart due to inverse scale; shown only in LaTeX table with proper formatting
- matplotlib Agg backend set at module level to support headless rendering

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed matplotlib, pandas, datasets, transformers in .venv**
- **Found during:** Task 1 (config.py verification)
- **Issue:** No venv had matplotlib/pandas installed; datasets/transformers also missing
- **Fix:** Used `uv pip install` to add matplotlib, pandas, datasets, transformers to .venv
- **Files modified:** .venv (packages only, no committed files)
- **Verification:** All imports succeed

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Dependency installation was necessary to verify modules. No scope creep.

## Issues Encountered
- PyTorch not installed in .venv (warning shown), but not needed for analysis modules (tokenizer-only usage works fine)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 analysis modules ready for use by downstream plans (02-02, 02-03, 02-04)
- config.py provides shared constants for all future figures
- Modules import cleanly and are ready to generate outputs when summary.json data is available

---
*Phase: 02-analysis-and-figures*
*Completed: 2026-03-20*
