# Project Research Summary

**Project:** SozKZ arXiv Paper — Efficient Small Language Models for Kazakh
**Domain:** Academic ML paper (arXiv preprint) with evaluation pipeline
**Researched:** 2026-03-20
**Confidence:** MEDIUM

## Executive Summary

This project is an arXiv paper demonstrating that small, dedicated language models (50M-600M parameters) trained from scratch on Kazakh data can approach or match the Kazakh performance of much larger multilingual models (2B-8B) at a fraction of the compute and inference cost. The established pattern for this type of paper is: (1) build a rigorous evaluation pipeline covering 4-6 diverse benchmarks, (2) run all models -- own and competitor -- under identical conditions, (3) present results with efficiency analysis (not just accuracy), and (4) release everything openly. The models and training infrastructure already exist; the remaining work is evaluation, analysis, and writing.

The recommended approach is to treat this as a three-component pipeline: Evaluation Pipeline produces benchmark JSONs, Analysis and Figures transforms them into publication-ready assets, and LaTeX Paper consumes those assets. This strict data-flow separation prevents the most common paper-writing failure mode -- hardcoded numbers that go stale when benchmarks are re-run. The most critical immediate task is fixing the MC benchmark scoring bug (10% on 4-choice MC is below the 25% random baseline), followed by implementing bits-per-byte evaluation for fair cross-tokenizer comparison.

The key risks are (1) unfair comparison framing -- claiming "efficiency" without controlling for the multilingual tax or reporting proper efficiency metrics beyond parameter count, (2) benchmark contamination in the 23.6M-sample training corpus overlapping with test sets, and (3) the "why not fine-tune?" question that reviewers will inevitably ask. All three are addressable: the first through honest framing and a performance-per-FLOP analysis, the second through n-gram overlap checks, and the third either by including a fine-tuned Gemma-2B baseline or by providing a strong justification in the Discussion section.

## Key Findings

### Recommended Stack

The evaluation infrastructure centers on EleutherAI's lm-evaluation-harness (v0.4.11, verified active with 11.7K stars) as the benchmark runner, extended with custom YAML task definitions for Kazakh-specific benchmarks. Competitor models (7B+) should be loaded via vLLM for fast batch inference, with bitsandbytes for quantization on A10 GPUs. The paper itself uses standard LaTeX (ACL 2024 template recommended) with matplotlib/seaborn for figures. Results flow through JSON files aggregated by pandas into auto-generated LaTeX macros -- no hardcoded numbers.

**Core technologies:**
- **lm-evaluation-harness v0.4.11**: benchmark runner -- industry standard, supports custom YAML tasks, has built-in Belebele Kazakh
- **vLLM + bitsandbytes**: competitor model inference -- fast batched inference for 7B+ models on A10 GPUs
- **matplotlib + seaborn**: figure generation -- publication-quality, scriptable, PDF export for LaTeX
- **LaTeX (ACL 2024 template)**: paper writing -- signals NLP domain expertise, clean two-column layout
- **pandas + scipy.stats**: results aggregation and statistical significance testing

**Critical version pins:** `lm-eval==0.4.11`, `vllm>=0.6.0`, `bitsandbytes>=0.44`, `matplotlib>=3.9`, `seaborn>=0.13`

### Expected Features

**Must have (table stakes):**
- Standard paper sections: abstract, intro, related work, data, tokenizer, architecture, training, evaluation, results, conclusion, limitations
- 4+ benchmark evaluations: perplexity/BPB, MC knowledge benchmark, sentiment classification, reading comprehension
- Comparison against 5+ baselines (Qwen-0.5B/1.5B, Gemma-2B, Llama-3.1-8B, Mistral-7B, KazLLM-8B)
- Results tables with all benchmarks reported (wins AND losses)
- Scaling curves across 50M/150M/300M/600M model sizes
- Tokenizer fertility analysis (tokens/word vs multilingual tokenizers)

**Should have (differentiators):**
- Bits-per-byte instead of raw perplexity (essential for fair cross-tokenizer comparison -- borderline table stakes)
- Efficiency analysis: performance-per-FLOP, inference tokens/sec, memory footprint
- Scaling law fit for Kazakh (compare exponent to Chinchilla for English)
- Compute-matched comparison (not just parameter-matched)
- Qualitative generation examples with error analysis (bilingual KZ+EN)

**Defer (future work):**
- Morphological probing of model representations
- Cross-lingual transfer to other Turkic languages
- Human evaluation by native speakers
- Fine-tuning competitor models as baselines (address in Discussion if not run)

### Architecture Approach

The project is a three-stage pipeline with strict upstream dependencies. The Evaluation Pipeline produces structured JSON results per model per task. The Analysis and Figures component reads those JSONs and generates LaTeX tables and PDF charts. The LaTeX Paper consumes figures and tables via `\input{}` and `\includegraphics{}`, with all numbers referenced through auto-generated macros. This separation ensures reproducibility and prevents stale numbers.

**Major components:**
1. **Evaluation Pipeline** -- runs all models on all benchmarks, outputs JSON results. Sub-components: benchmark registry (YAML task definitions), model runner (HF transformers for own models, vLLM for competitors), scorer, result aggregator
2. **Analysis and Figures** -- transforms `results/all_results.csv` into scaling curves, comparison bar charts, tokenizer fertility plots, training loss curves, efficiency tables, and auto-generated LaTeX macros
3. **LaTeX Paper** -- 8-10 page document (ACL 2024 format) consuming upstream artifacts, structured as intro/related-work/methodology/experiments/results/analysis/conclusion

### Critical Pitfalls

1. **Unfair comparison framing** -- do not only compare 600M vs 7B+ models. Include same-scale comparisons (Qwen-0.5B, Gemma-2B). Report performance-per-FLOP, not just performance-per-parameter. Acknowledge the multilingual tax explicitly.
2. **Benchmark contamination** -- the 23.6M-sample training corpus likely overlaps with Kazakh benchmark test sets (small language, recycled web sources). Run 13-gram overlap detection and report contamination rates in the paper.
3. **Missing efficiency metrics** -- "efficient" cannot mean only "fewer parameters." Dedicate a table to inference tokens/sec, GPU memory, training FLOPs, and USD cost on the same hardware.
4. **MC benchmark scoring bug** -- current 10% accuracy on 4-choice MC is below 25% random. This is almost certainly a prompt format or token-matching bug, not model quality. Must debug before drawing any conclusions.
5. **"Why not fine-tune?" question** -- reviewers will ask why not fine-tune Gemma-2B on Kazakh data instead. Either run this baseline or have a strong justification ready in Discussion/Limitations.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Evaluation Infrastructure Setup
**Rationale:** Nothing can be written without benchmark numbers. The evaluation pipeline is the critical path and highest-risk component. All four research files converge on this: STACK identifies the tools, FEATURES lists the required benchmarks, ARCHITECTURE defines the pipeline, and PITFALLS warns against inconsistent evaluation settings.
**Delivers:** Working benchmark suite with YAML task definitions, model registry (`models.yaml`), reproducible eval scripts, result aggregation pipeline, directory structure (`configs/benchmarks/`, `results/`, `paper/`).
**Addresses:** Benchmark registry (MISSING), competitor model runner (MISSING), result aggregator (MISSING), lm-eval-harness integration (MISSING).
**Avoids:** Inconsistent evaluation settings across models (Pitfall 12), monolithic eval script anti-pattern.

### Phase 2: Benchmark Execution and Bug Fixes
**Rationale:** With infrastructure ready, run all evaluations. Must fix the MC scoring bug first, then run all models on all tasks under identical conditions. Contamination check must complete before any numbers are considered final.
**Delivers:** Complete `results/` directory with JSON outputs for ~54 model-task combinations (9 models x 6 tasks), contamination check report, BPB calculations.
**Addresses:** Fix MC benchmark scoring, BPB evaluation, perplexity, sentiment, Belebele, KazLLM benchmarks, tokenizer fertility data collection.
**Avoids:** Benchmark contamination (Pitfall 3), single-run variance (Pitfall 6), cherry-picking results.

### Phase 3: Analysis and Figure Generation
**Rationale:** Mechanical once data exists. Transform raw results into publication assets. Can iterate quickly.
**Delivers:** Scaling curves, comparison bar charts, tokenizer fertility plots, efficiency tables, training curves, `macros.tex` with all auto-generated numbers.
**Addresses:** All figures and tables identified in FEATURES.md and ARCHITECTURE.md.
**Avoids:** Hardcoded numbers in LaTeX (Architecture anti-pattern 1).

### Phase 4: Paper Writing
**Rationale:** Intro, Related Work, and Methodology can start in parallel with Phases 1-2 (they describe what was already done). Results and Analysis sections are blocked on Phases 2-3. Conclusion is blocked on Results.
**Delivers:** Complete LaTeX draft (8-10 pages + appendix), bibliography (~30-50 references), qualitative examples with bilingual translations, limitations section.
**Addresses:** All paper sections from FEATURES.md table stakes, qualitative error analysis, "why not fine-tune?" answer.
**Avoids:** Overclaiming efficiency (Pitfall 4/7), poor related work coverage (Pitfall 11), missing "why not fine-tune?" answer (Pitfall 10).

### Phase 5: Polish and Submission
**Rationale:** Final pass after full draft is complete. Abstract is always written last because it summarizes final results.
**Delivers:** arXiv-ready PDF + source tarball, model/code/tokenizer release on HuggingFace + GitHub.
**Addresses:** Abstract rewrite, arXiv compliance, open-source contribution framing, proofreading.
**Avoids:** Translationese acknowledgment gaps (Pitfall 5), missing training data statistics (Pitfall 13).

### Phase Ordering Rationale

- Phases 1-2 must precede Phase 3 because figures depend on benchmark results
- Phase 3 must precede the Results/Analysis sections of Phase 4
- Intro, Related Work, and Methodology (Phase 4) can start during Phases 1-2 as parallelizable work -- this is the key optimization
- The MC scoring bug fix (Phase 2) is the single highest-priority item -- all downstream analysis depends on correct benchmark numbers
- Contamination check (Phase 2) must complete before any numbers are considered final
- Architecture diagrams and training curves can be generated immediately (no dependency on new benchmarks)

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Kazakh benchmark dataset availability needs direct verification on HuggingFace (KazLLM Benchmark Dataset column/split structure, KazNERD format, Belebele kk subset). Custom YAML task authoring for lm-eval-harness needs investigation for non-standard Kazakh datasets.
- **Phase 2:** The MC scoring bug requires debugging -- could be prompt format, tokenization mismatch, or log-probability extraction issue. Contamination detection tooling for Cyrillic/agglutinative text is less mature than for English.

Phases with standard patterns (skip research-phase):
- **Phase 3:** matplotlib figure generation and LaTeX table generation are well-documented workflows.
- **Phase 4:** Paper structure follows established ML/NLP conventions. Reference papers identified: AfriBERTa, Jais, SEA-LION, Chinchilla.
- **Phase 5:** arXiv submission is a standard, well-documented process.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | lm-eval-harness and Kazakh benchmarks verified via GitHub API; version pins confirmed |
| Features | MEDIUM | Paper conventions are stable; specific Kazakh benchmark availability needs HF verification |
| Architecture | MEDIUM-HIGH | Pipeline pattern is standard for ML papers; existing codebase gaps clearly identified |
| Pitfalls | MEDIUM | Based on published paper patterns and evaluation methodology critiques; web verification unavailable |

**Overall confidence:** MEDIUM

### Gaps to Address

- **MC benchmark bug root cause:** The 10% accuracy on 4-choice MC needs debugging before any evaluation work proceeds. Could invalidate benchmark selection if it is a dataset issue rather than a code bug.
- **Kazakh benchmark dataset formats:** `issai/KazLLM_Benchmark_Dataset` column structure and splits need direct inspection on HuggingFace before writing lm-eval YAML configs.
- **Fine-tuning baseline decision:** Must decide whether to run a Gemma-2B fine-tuned-on-Kazakh baseline (strong but time-consuming) or write a convincing justification for omitting it. Affects Phase 2 scope significantly.
- **Contamination detection tooling:** n-gram overlap detection for Kazakh/Cyrillic text needs investigation -- standard tools may not handle agglutinative morphology well.
- **Qwen small model Kazakh support:** Whether Qwen2.5-0.5B/1.5B genuinely support Kazakh or just have trace amounts in training data -- affects baseline comparison validity.
- **XCOPA Kazakh inclusion:** LOW confidence on whether XCOPA includes Kazakh. Verify before considering.

## Sources

### Primary (HIGH confidence)
- EleutherAI/lm-evaluation-harness: GitHub API verified, v0.4.11, 11,770 stars, `belebele_kaz_Cyrl` task confirmed
- IS2AI/KazQAD: GitHub API verified, CC BY-SA 4.0, published at ACL 2024
- IS2AI/KazSAnDRA: GitHub API verified, sentiment analysis dataset
- IS2AI/KazNERD: GitHub API verified, 112K sentences, 25 entity classes
- IS2AI/KazLLM_Benchmark: GitHub API verified, MMLU/ARC/HellaSwag/Winogrande/GSM8K/DROP in Kazakh
- kz-transformers/kk-socio-cultural-bench-mc: Used in project (7,111 questions)
- Project codebase: existing eval scripts, experiment configs, training logs

### Secondary (MEDIUM confidence)
- ML paper conventions from ACL/EMNLP/NeurIPS proceedings (training data through May 2025)
- Comparable papers: AfriBERTa, Jais, SEA-LION, Chinchilla methodology patterns
- ACL 2024 style files on GitHub

### Tertiary (LOW confidence)
- XCOPA Kazakh inclusion -- needs verification
- KazNLP benchmark suite availability -- could not verify via web
- Contamination detection tooling maturity for Cyrillic text -- assumed from English-language tool limitations

---
*Research completed: 2026-03-20*
*Ready for roadmap: yes*
