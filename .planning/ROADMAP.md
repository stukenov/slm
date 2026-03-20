# Roadmap: SozKZ arXiv Paper

## Overview

This roadmap delivers an arXiv paper proving that small, dedicated Kazakh language models (50M-600M params) can approach larger multilingual models at a fraction of the cost. The critical path flows through evaluation (numbers first), analysis (figures and tables from numbers), paper writing (prose consuming figures), and submission (final PDF to arXiv). Every claim in the paper depends on benchmark results, so the evaluation pipeline is Phase 1.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Evaluation Pipeline** - Build benchmark infrastructure and run all models on all tasks (completed 2026-03-20)
- [ ] **Phase 2: Analysis and Figures** - Transform raw results into publication-ready tables, charts, and LaTeX macros
- [ ] **Phase 3: Paper Writing** - Write complete LaTeX draft consuming upstream assets
- [ ] **Phase 4: Polish and Submission** - Final revision, abstract rewrite, arXiv submission

## Phase Details

### Phase 1: Evaluation Pipeline
**Goal**: All benchmark numbers exist as structured JSON/CSV files, covering every own model and every competitor on every task
**Depends on**: Nothing (first phase)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07
**Success Criteria** (what must be TRUE):
  1. A model registry config lists all own models (50M, 150M, 300M, 600M) and all competitors (Gemma-2B, Gemma-7B, Llama-3-1B, Llama-3-3B, Llama-3-8B, Qwen-2.5-0.5B, Qwen-2.5-1.5B, Qwen-2.5-7B, GPT-OSS-120B, Mistral-7B) with HF model IDs
  2. Running a single command produces BPB scores for any model on held-out Kazakh text
  3. MC QA benchmark returns accuracy above the 25% random baseline for models with known Kazakh capability (MC scoring bug is fixed)
  4. All six benchmark tasks (BPB, MC QA, sentiment, Belebele, NER, topic classification) produce JSON results for every model in the registry
  5. A contamination check report exists showing n-gram overlap between training data and each benchmark test set
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Model registry, MC QA bug fix, BPB computation
- [x] 01-02-PLAN.md — Sentiment, Belebele, SIB-200 evaluation scripts
- [x] 01-03-PLAN.md — NER evaluation, results aggregation, run_all.sh

### Phase 2: Analysis and Figures
**Goal**: All numbers are transformed into publication-ready assets -- comparison tables, scaling curves, efficiency metrics, and auto-generated LaTeX macros with no hardcoded values
**Depends on**: Phase 1
**Requirements**: ANLS-01, ANLS-02, ANLS-03, ANLS-04, ANLS-05
**Success Criteria** (what must be TRUE):
  1. A tokenizer fertility comparison table shows chars/token for every model's tokenizer on the same Kazakh text sample
  2. Scaling curve plots (PDF/PNG) show performance vs parameter count for own models across all benchmarks
  3. An efficiency table reports inference latency, throughput (tok/s), and peak GPU memory for every model measured on the same hardware
  4. Running a single script regenerates all figures and a macros.tex file from the raw results JSON -- no manual number entry
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — Config, tokenizer fertility analysis, comparison bar chart, comparison table
- [ ] 02-02-PLAN.md — Scaling curves, LaTeX macros, generate_all.py orchestrator
- [ ] 02-03-PLAN.md — Contamination check (n-gram overlap)
- [ ] 02-04-PLAN.md — Efficiency benchmarking (inference latency/throughput/memory on A10)

### Phase 3: Paper Writing
**Goal**: A complete LaTeX draft exists with all sections written, all figures included, and all numbers pulled from auto-generated macros
**Depends on**: Phase 2
**Requirements**: PAPR-01, PAPR-02, PAPR-03, PAPR-04, PAPR-05, PAPR-06
**Success Criteria** (what must be TRUE):
  1. The paper compiles to a valid PDF with sections: abstract, introduction, related work, methodology, experiments, results, conclusion
  2. The methodology section documents the full training pipeline (data sources, tokenizer design decisions, architecture choices, hyperparameters) with enough detail to reproduce
  3. Results section contains comparison tables showing own models vs all competitors across all benchmarks, with wins and losses both reported
  4. Tokenizer analysis section demonstrates fertility advantage of dedicated Kazakh BPE over multilingual tokenizers with supporting figures
  5. Scaling analysis section includes fitted curves showing performance trends across 50M/150M/300M/600M with discussion of diminishing returns
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Paper scaffold, Phase 2 asset generation, introduction, related work, conclusion, references.bib
- [ ] 03-02-PLAN.md — Methodology section (data, tokenizer, architecture, training) and experiments section
- [ ] 03-03-PLAN.md — Results section with comparison tables, tokenizer analysis, scaling analysis, efficiency

### Phase 4: Polish and Submission
**Goal**: The paper is finalized and published on arXiv as a preprint
**Depends on**: Phase 3
**Requirements**: PAPR-07
**Success Criteria** (what must be TRUE):
  1. Abstract is rewritten to reflect final results (not placeholder claims)
  2. The paper PDF meets arXiv formatting requirements and compiles without errors
  3. Paper is submitted to arXiv and a preprint URL exists
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Evaluation Pipeline | 3/3 | Complete   | 2026-03-20 |
| 2. Analysis and Figures | 0/4 | Not started | - |
| 3. Paper Writing | 1/3 | In progress | - |
| 4. Polish and Submission | 0/1 | Not started | - |
