# Architecture: arXiv Paper Project for Efficient Kazakh SLM

**Domain:** ML research paper (arXiv preprint) with evaluation pipeline
**Researched:** 2026-03-20
**Overall confidence:** MEDIUM (web search unavailable; based on strong training-data knowledge of ML paper conventions and the author's existing codebase)

## System Overview

The project has three major components that produce the final artifact (an arXiv PDF). They form a pipeline with clear data-flow direction:

```
[1. Evaluation Pipeline] --> benchmark results (JSON/CSV)
        |
        v
[2. Analysis & Figures]  --> tables, charts, diagrams (PDF/PNG)
        |
        v
[3. LaTeX Paper]          --> arXiv submission (PDF + source tarball)
```

Each component is independently testable but has strict upstream dependencies: you cannot write the Results section without benchmark numbers, and you cannot make scaling-curve figures without evaluation results from multiple model sizes.

---

## Component 1: Evaluation Pipeline

**Responsibility:** Run every model (own + competitor) on every benchmark task, produce structured JSON results.

**Boundary:** This is pure infrastructure code. It knows nothing about the paper -- it outputs machine-readable results that downstream components consume.

### Sub-components

| Sub-component | What It Does | Input | Output |
|---------------|-------------|-------|--------|
| **Benchmark Registry** | Defines tasks, datasets, metrics, few-shot configs | YAML/Python task definitions | Task specs |
| **Model Runner** | Loads a model, runs inference on benchmark inputs | Model path + task spec | Raw predictions |
| **Scorer** | Computes metrics (perplexity, accuracy, F1, BLEU, etc.) | Predictions + gold labels | Metric JSON per model per task |
| **Result Aggregator** | Merges all per-model-per-task JSONs into a single results table | Directory of JSON files | `results/all_results.csv` |

### Benchmark Tasks (recommended set)

Based on the project goals (prove small specialized model matches large multilingual ones on Kazakh), these task categories are needed:

| Category | Task Type | Metric | Why Needed |
|----------|-----------|--------|------------|
| **Perplexity** | Language modeling on held-out Kazakh text | PPL | Core LM quality; your existing `evaluate.py` already does this |
| **Text Classification** | Sentiment analysis (exp025 fine-tune exists) | Accuracy, F1 | Shows understanding capability |
| **Generation Quality** | Open-ended generation + human/auto eval | Kazakh char ratio, coherence score | Shows generative quality |
| **NLU / QA** | Question answering or reading comprehension in Kazakh | EM, F1 | Shows comprehension (if dataset exists) |
| **Translation** | kk->en or en->kk | BLEU, chrF++ | Shows cross-lingual capability (optional) |
| **Tokenizer Efficiency** | Fertility (tokens per word), coverage | Tokens/word ratio | Supports the "efficiency" narrative |

**Key architectural decision:** Use `lm-evaluation-harness` (EleutherAI) as the runner backbone where possible. It already supports custom tasks via YAML configs and handles batching, few-shot prompting, and metric computation. For tasks not in harness (e.g., tokenizer fertility, Kazakh-specific char ratio), write standalone scripts that output the same JSON schema.

### Data Flow Detail

```
configs/benchmarks/
  task_perplexity.yaml
  task_sentiment.yaml
  task_generation.yaml
  ...

models.yaml          # List of all models to evaluate
  - name: sozkz-core-llama-50m-kk-base-v1
    path: saken-tukenov/sozkz-core-llama-50m-kk-base-v1
    params: 50M
    type: own
  - name: gemma-2b
    path: google/gemma-2b
    params: 2B
    type: competitor
  ...

run_all_benchmarks.sh  # Iterates models x tasks, writes to results/

results/
  perplexity/
    sozkz-core-llama-50m-kk-base-v1.json
    gemma-2b.json
    ...
  sentiment/
    ...
  all_results.csv      # Aggregated by result_aggregator.py
```

### What Already Exists vs What Needs Building

| Component | Status | Gap |
|-----------|--------|-----|
| Perplexity eval | EXISTS (`src/slm/evaluate.py`) | Needs batching, competitor model support |
| Generation eval | EXISTS (basic, in `evaluate.py`) | Needs automated quality metrics (not just char ratio) |
| Sentiment eval | PARTIAL (exp025 trained a sentiment model) | Need benchmark dataset + eval script |
| Benchmark registry | MISSING | Need task YAML definitions |
| Competitor model runner | MISSING | Need to handle different tokenizers, model sizes, quantization |
| Result aggregator | MISSING | Need script to merge JSONs into CSV |
| lm-eval-harness integration | MISSING | Need custom task configs for Kazakh |

---

## Component 2: Analysis and Figures

**Responsibility:** Transform raw benchmark results into publication-quality tables and figures.

**Boundary:** Reads only from `results/` directory. Produces only visual artifacts in `paper/figures/` and `paper/tables/`. No model inference happens here.

### Sub-components

| Sub-component | What It Does | Output |
|---------------|-------------|--------|
| **Table Generator** | Reads `all_results.csv`, formats LaTeX tables | `.tex` table files |
| **Scaling Curve Plotter** | Plots metric vs. model size (50M, 150M, 300M, 600M) | `scaling_curves.pdf` |
| **Comparison Chart** | Bar charts: own models vs competitors on each task | `comparison_*.pdf` |
| **Architecture Diagram** | Model architecture visualization | `architecture.pdf` |
| **Tokenizer Analysis** | Fertility comparison charts | `tokenizer_*.pdf` |
| **Training Curves** | Loss curves from training logs (already have data) | `training_curves.pdf` |

### Recommended Tooling

- **matplotlib + seaborn** for all plots (do NOT use plotly -- rasterized output is wrong for papers)
- **pgfplots** (LaTeX-native) as an alternative if you want font-consistent figures, but matplotlib is faster to iterate
- Use a shared `paper/plot_style.py` that sets consistent fonts, colors, figure sizes across all plots
- Target figure sizes: single-column (3.25in) or double-column (6.75in) for NeurIPS/ACL-style templates

### Key Figures for the Paper

| Figure | Section | Purpose |
|--------|---------|---------|
| Scaling curves (params vs perplexity) | Results | Core result -- shows diminishing returns, sweet spot |
| Comparison bars (own vs competitors) | Results | The "hero chart" -- small model punches above weight |
| Training loss curves | Methodology | Shows training convergence, stability |
| Architecture diagram | Methodology | Llama architecture with your specific configs |
| Tokenizer fertility comparison | Methodology/Analysis | Shows custom BPE is more efficient for Kazakh |
| Efficiency plot (params vs metric, with cost overlay) | Analysis | The money chart -- cost per quality point |

---

## Component 3: LaTeX Paper

**Responsibility:** The actual written document. Consumes figures and tables from Component 2.

**Boundary:** Pure writing. All data-driven content (numbers, figures) comes from upstream components via `\input{}` and `\includegraphics{}`. No hardcoded numbers in the text -- use LaTeX macros (`\newcommand{\bestPPL}{24.3}`) so results auto-update when benchmarks re-run.

### Recommended Paper Structure

Based on standard ML/NLP paper conventions (ACL/EMNLP style, 8-12 pages):

| Section | Pages | Content |
|---------|-------|---------|
| **Title + Abstract** | 0.5 | "SozKZ: Efficient Small Language Models for Kazakh" -- 150-word abstract |
| **1. Introduction** | 1.5 | Problem (Kazakh underserved), contribution (full pipeline + benchmarks), key result (X-param model matches Y-param on Z tasks) |
| **2. Related Work** | 1.0 | Small LMs (SmolLM, MobileLLM, Phi), low-resource LMs (AfroLM, AceGPT, SEA-LION), Kazakh NLP (KazNLP if exists), efficient training |
| **3. Methodology** | 2.5 | 3.1 Data (corpus stats, cleaning), 3.2 Tokenizer (BPE training, fertility analysis), 3.3 Architecture (Llama configs per size), 3.4 Training (hyperparams, compute budget, scaling decisions) |
| **4. Experimental Setup** | 1.0 | 4.1 Benchmark suite, 4.2 Competitor models, 4.3 Evaluation protocol (few-shot, decoding params) |
| **5. Results** | 2.0 | 5.1 Main results table, 5.2 Scaling analysis, 5.3 Comparison with large models, 5.4 Efficiency analysis |
| **6. Analysis** | 1.0 | 6.1 Tokenizer efficiency, 6.2 Where small models fail, 6.3 MoE experiment findings |
| **7. Conclusion** | 0.5 | Summary, limitations, future work |
| **References** | 1.0 | ~30-50 references |
| **Appendix** | 1-2 | Generation examples, full hyperparameter tables, additional benchmark results |

### Directory Structure

```
paper/
  main.tex              # Root document, \input{} everything
  sections/
    abstract.tex
    introduction.tex
    related_work.tex
    methodology.tex
    experiments.tex
    results.tex
    analysis.tex
    conclusion.tex
  figures/              # Generated by Component 2
    scaling_curves.pdf
    comparison_bars.pdf
    ...
  tables/               # Generated by Component 2
    main_results.tex
    model_configs.tex
    ...
  macros.tex            # \newcommand for all numbers
  references.bib
  style/                # ACL/NeurIPS .sty files
```

### LaTeX Template Decision

Use **ACL 2024 template** (acl.sty). Rationale:
- This is an NLP paper, ACL format signals "this is serious NLP work"
- Well-supported, widely recognized
- Clean two-column layout, good for comparison tables
- Alternative: NeurIPS (single column, more ML-focused) -- use this if framing as ML efficiency paper rather than NLP paper

---

## Build Order and Dependencies

This is the critical part for roadmap planning. Components have strict ordering:

```
Phase 1: Evaluation Pipeline (MUST be first)
  |-- No paper content can be written without numbers
  |-- Competitor evaluation may require GPU time (vast.ai)
  |-- This is the highest-risk phase (benchmark selection, harness setup)

Phase 2: Analysis & Figures (after Phase 1 complete)
  |-- Mechanical: run plotting scripts on results
  |-- Can iterate quickly once data exists
  |-- Some figures (architecture diagram, training curves) can start in parallel with Phase 1

Phase 3: Paper Writing (after Phase 2 mostly complete)
  |-- Intro and Related Work can start in parallel with Phase 1
  |-- Methodology can start in parallel with Phase 1 (describes what was already done)
  |-- Results/Analysis sections BLOCKED on Phase 1+2
  |-- Conclusion BLOCKED on Results

Phase 4: Polish & Submit (after Phase 3 draft complete)
  |-- Proofreading, formatting, arXiv compliance
  |-- Abstract rewrite (always last -- summarizes final results)
```

### Parallelizable Work

Some work can happen in parallel with the evaluation pipeline:

| Work Item | Can Start When | Blocked By |
|-----------|---------------|------------|
| Related Work section | Immediately | Nothing |
| Methodology section (data, tokenizer, architecture) | Immediately | Nothing |
| Architecture diagram | Immediately | Nothing |
| Training curves figure | Immediately (have training logs) | Nothing |
| LaTeX template setup | Immediately | Nothing |
| Benchmark selection & task definition | Immediately | Nothing |
| Running own model benchmarks | After benchmark code ready | Eval pipeline |
| Running competitor benchmarks | After benchmark code ready | Eval pipeline + GPU access |
| Results tables | After benchmarks complete | Eval pipeline |
| Scaling curves | After benchmarks complete | Eval pipeline |
| Results section text | After figures/tables ready | Analysis & Figures |
| Abstract | After full draft | Everything |

---

## Data Flow: End to End

```
Training logs (existing)  ──┐
                             ├──> [Figure Generator] ──> paper/figures/training_*.pdf
Model checkpoints (HF) ─────┤
                             ├──> [Eval Pipeline] ──> results/*.json
Benchmark datasets ─────────┘         |
                                      v
                              [Result Aggregator] ──> results/all_results.csv
                                      |
                                      v
                              [Table Generator] ──> paper/tables/*.tex
                              [Chart Generator] ──> paper/figures/*.pdf
                                      |
                                      v
                              [LaTeX Compiler] ──> paper/main.pdf
                                      |
                                      v
                              [arXiv Packager] ──> submission.tar.gz
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Hardcoded Numbers in LaTeX
**What:** Writing `Our model achieves 24.3 perplexity` directly in .tex files.
**Why bad:** When you re-run benchmarks (and you will), every number must be manually updated. Guaranteed to have stale numbers in final paper.
**Instead:** Use `macros.tex` with `\newcommand{\bestPPL}{24.3}` and a script that generates macros.tex from results CSV.

### Anti-Pattern 2: Running Competitors on Wrong Settings
**What:** Evaluating Gemma/Llama-3 with different decoding params, context lengths, or prompting than your own models.
**Why bad:** Reviewers will catch this. Unfair comparison invalidates results.
**Instead:** Define evaluation protocol once (temperature, top-p, max tokens, few-shot count) in a config file. All models use the same config.

### Anti-Pattern 3: Cherry-Picking Benchmarks
**What:** Only reporting tasks where your model wins.
**Why bad:** Dishonest. Also, showing where you lose strengthens the paper (honest analysis, efficiency framing).
**Instead:** Report all benchmarks. Frame losses as expected given 100x fewer parameters. The story is efficiency, not dominance.

### Anti-Pattern 4: Monolithic Eval Script
**What:** One giant script that loads models, runs benchmarks, computes metrics, and generates figures.
**Why bad:** Cannot re-run one benchmark without re-running everything. Cannot debug one task. Cannot parallelize.
**Instead:** Separate scripts per stage. Each reads from and writes to well-defined directories.

---

## Scalability Considerations

| Concern | Current Scale | At Paper Deadline |
|---------|--------------|-------------------|
| Models to evaluate | ~4 own + ~5 competitor = ~9 | Same (fixed set) |
| Benchmark tasks | ~4-6 tasks | Same (fixed set) |
| Total eval runs | ~54 (9 models x 6 tasks) | Same |
| GPU hours for eval | ~10-20h on 2xA10 | May need vast.ai for large competitors |
| Storage | ~10GB results | Negligible |

The scale is manageable. The main constraint is GPU memory for running large competitor models (e.g., GPT-OSS-120B requires quantization or API access).

---

## Key Architectural Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Eval framework | lm-evaluation-harness + custom scripts | Standardized, reproducible, community-recognized |
| Results format | JSON per run, CSV aggregate | Machine-readable, easy to plot |
| Figure generation | matplotlib + seaborn | Publication quality, scriptable, fast iteration |
| LaTeX template | ACL 2024 | NLP-community standard, signals domain expertise |
| Number management | Auto-generated macros.tex from results | Prevents stale numbers, enables re-runs |
| Competitor eval | Same config for all models | Fair comparison, reviewer-proof |
| Paper length | 8-10 pages + appendix | Standard for arXiv NLP preprint |

## Sources

- Based on training-data knowledge of ML paper conventions (SmolLM, MobileLLM, Phi-series, AfroLM paper structures)
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- ACL 2024 style files: https://github.com/acl-org/acl-style-files
- Project codebase analysis: `src/slm/evaluate.py`, `eval/prompts_kk.txt`, experiment configs

**Confidence note:** Web search was unavailable during research. Kazakh-specific benchmark availability (KazNLP, TurkicBench) could not be verified and is flagged as LOW confidence. The overall architecture pattern is HIGH confidence as it follows well-established ML paper conventions.
