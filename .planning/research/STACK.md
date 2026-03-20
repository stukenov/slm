# Technology Stack: arXiv Paper on Efficient Small Language Model for Kazakh

**Project:** SozKZ arXiv Paper
**Researched:** 2026-03-20
**Overall confidence:** MEDIUM-HIGH (lm-eval-harness and LaTeX stack verified via GitHub API; Kazakh benchmark datasets verified via IS2AI repos; some version details from training data)

## Recommended Stack

### 1. Evaluation Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| lm-evaluation-harness | v0.4.11 | Core benchmark runner | Industry standard (11.7K stars, actively maintained as of 2026-03-20). Supports custom tasks via YAML configs. Has built-in `belebele_kaz_Cyrl` task for Kazakh reading comprehension. Used by every major LLM benchmark paper. |
| Custom eval scripts | N/A | Kazakh-specific benchmarks not in lm-eval | Some IS2AI benchmarks (KazQAD, sentiment) need custom evaluation code. The project already has `scripts/eval/eval_mc_bench.py` as a pattern. |

**Why not IS2AI/KazLLM_Benchmark:** Their framework is Docker-heavy (docker-compose + nvidia-docker), designed for large instruct models with text generation (not logit-based scoring). Overkill for this paper. Better to use lm-eval-harness with custom YAML tasks that point to the same underlying datasets.

**Why not OpenAI Evals or HELM:** OpenAI Evals requires API calls (not local models). HELM is heavyweight and harder to extend with custom tasks. lm-eval-harness is the community standard for open-weight model evaluation.

### 2. Kazakh Benchmark Datasets

| Dataset | HuggingFace ID | Task Type | Metric | Confidence |
|---------|---------------|-----------|--------|------------|
| Belebele (Kazakh) | `facebook/belebele` (kaz_Cyrl split) | Reading comprehension (4-choice MC) | Accuracy | HIGH -- already in lm-eval-harness |
| KK Socio-Cultural Bench | `kz-transformers/kk-socio-cultural-bench-mc` | Knowledge MC (7K+ questions, 15 categories) | Accuracy | HIGH -- already evaluated with existing script |
| KazQAD | `issai/kazqad` | Open-domain QA (extractive) | F1 / EM | MEDIUM -- needs custom eval script; CC BY-SA 4.0; published at ACL 2024 |
| KazSAnDRA | `issai/kazsandra` | Sentiment analysis (polarity classification) | Accuracy / F1 | MEDIUM -- needs custom eval; good for showing downstream task capability |
| KazNERD | (IS2AI/KazNERD, CoNLL format) | Named Entity Recognition (25 classes) | F1 (entity-level) | LOW -- NER is a token-classification task, not directly applicable to causal LM evaluation without fine-tuning. Consider omitting. |
| FLORES-200 | `facebook/flores` (kaz_Cyrl) | Translation quality proxy (perplexity on parallel text) | Perplexity | MEDIUM -- useful for cross-lingual comparison but not a standard LLM benchmark |
| KazLLM Benchmark Dataset | `issai/KazLLM_Benchmark_Dataset` | Kazakh translations of MMLU, ARC, HellaSwag, Winogrande, GSM8K, DROP | Accuracy | HIGH -- this is the most comprehensive Kazakh LLM benchmark; covers standard English benchmarks translated to Kazakh |

**Recommended evaluation matrix (4 core benchmarks):**

1. **Belebele (kaz_Cyrl)** -- reading comprehension, runs directly in lm-eval-harness
2. **KK Socio-Cultural Bench MC** -- Kazakh cultural knowledge, existing eval script
3. **KazLLM MMLU-kk** -- general knowledge (Kazakh translation), from issai dataset
4. **KazLLM ARC-kk** -- reasoning (Kazakh translation), from issai dataset

Optional additions: KazQAD (QA), KazSAnDRA (sentiment), perplexity on held-out Kazakh text.

### 3. Competitor Models to Evaluate

| Model | Parameters | Why Include | Access |
|-------|-----------|-------------|--------|
| Qwen2.5-0.5B / 1.5B / 7B | 0.5B-7B | Strong multilingual, good Kazakh support reported | HF Hub (open weights) |
| Gemma-2-2B / 9B | 2B-9B | Known decent Kazakh performance (per project notes) | HF Hub (open weights) |
| Llama-3.1-8B | 8B | Industry baseline, weak on Kazakh expected | HF Hub (open weights, gated) |
| Mistral-7B-v0.3 | 7B | Popular baseline | HF Hub (open weights) |
| issai/LLama-3-KazLLM-1.0-8B | 8B | Kazakh-specific fine-tune by IS2AI, direct competitor | HF Hub |

**Why these:** The paper's thesis is "small specialized beats large general." You need large general models (Llama, Gemma, Qwen, Mistral at 7-9B) as upper bounds, and Qwen-0.5B/Gemma-2B as same-scale comparisons. KazLLM-8B is the only Kazakh-specific competitor.

### 4. LaTeX / Paper Writing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| LaTeX (pdflatex/xelatex) | TeX Live 2024+ | Paper compilation | Required for arXiv submission |
| arxiv-style or plain article class | N/A | Document class | Use `\documentclass{article}` with standard packages. arXiv accepts any valid LaTeX. For a preprint, plain article class is fine. If targeting a venue later, switch to their style. |
| booktabs | (standard) | Professional tables | Clean tables without vertical lines -- standard in ML papers |
| pgfplots / matplotlib (exported) | N/A | Figures and charts | pgfplots for native LaTeX charts; or generate with matplotlib and include as PDF |
| hyperref | (standard) | Clickable references | Standard for arXiv papers |
| cleveref | (standard) | Smart cross-references | `\cref{fig:scaling}` auto-formats as "Figure 1" |
| natbib or biblatex | (standard) | Bibliography | natbib with `.bst` style is most compatible with arXiv |
| algorithm2e or algorithmicx | (standard) | Pseudocode | If describing evaluation pipeline or training procedure |

**Template recommendation:** Start with a clean `article` class. If you want a polished preprint look, use the `arxiv` LaTeX style package (available on CTAN). Do NOT use NeurIPS/ICML templates unless submitting to those venues -- it implies peer review that hasn't happened.

### 5. Figures and Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| matplotlib | 3.9+ | Primary chart generation | Already in project dependencies. Export as PDF for LaTeX inclusion. |
| seaborn | 0.13+ | Statistical plots | Nicer defaults than raw matplotlib for heatmaps, distribution plots |
| tikz/pgfplots | (LaTeX package) | Architecture diagrams | Native LaTeX quality; good for model architecture diagrams |
| draw.io / Excalidraw | N/A | Architecture diagrams (alternative) | Quick iteration, export as PDF/SVG |

### 6. Experiment Tracking and Results

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| JSON result files | N/A | Structured benchmark outputs | Simple, version-controllable, already used in project (`results/eval_mc_bench.json`) |
| pandas | 2.2+ | Results aggregation and table generation | Already in project. Load JSON results, compute means, generate LaTeX tables |
| scipy.stats | (with scipy) | Statistical significance tests | Paired bootstrap or McNemar's test for comparing model accuracy |

**Why not W&B/MLflow:** This is paper writing, not experiment tracking. Results are already collected. JSON files + pandas is simpler and fully reproducible.

### 7. Infrastructure (Evaluation Runs)

| Technology | Purpose | Why |
|------------|---------|-----|
| vast.ai (existing) | Run evaluation on GPU | Already set up. A single A10 (23GB) is sufficient for inference on models up to ~7B with 4-bit quantization |
| vLLM or HF transformers | Model loading for inference | vLLM for fast batch inference on competitor models (7B+). transformers for own small models (already works). |
| bitsandbytes | 4-bit quantization | Load 7-9B competitor models on a single A10 (23GB VRAM). AWQ/GPTQ also work. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Eval framework | lm-evaluation-harness | IS2AI/KazLLM_Benchmark | Docker-heavy, designed for large instruct models, not as flexible for custom tasks |
| Eval framework | lm-evaluation-harness | HELM (Stanford) | Heavyweight, harder to extend with custom Kazakh tasks |
| Eval framework | lm-evaluation-harness | OpenAI Evals | Requires API calls, not for local open-weight models |
| LaTeX template | Plain article class | NeurIPS/ICML style | Implies peer review; misleading for a preprint |
| LaTeX template | Plain article class | Overleaf | Fine for editing but adds dependency; local LaTeX is simpler for version control |
| Figures | matplotlib + PDF export | pgfplots only | pgfplots has steep learning curve for complex charts; matplotlib is faster to iterate |
| Inference | vLLM | TGI (HuggingFace) | vLLM is simpler to set up for batch evaluation, better throughput |
| NER benchmark | Omit | Include KazNERD | NER requires token classification head, not directly testable with causal LM without fine-tuning |

## Key Version Pins

```bash
# Evaluation
pip install lm-eval==0.4.11
pip install vllm>=0.6.0  # for fast inference on large competitor models

# Already in project
# torch, transformers, datasets, pandas -- use existing versions

# Paper figures
pip install matplotlib>=3.9 seaborn>=0.13

# Statistical testing
pip install scipy>=1.13

# Quantization (for loading large competitor models on A10)
pip install bitsandbytes>=0.44
```

```bash
# LaTeX (system packages, not pip)
# macOS:
brew install --cask mactex-no-gui
# or minimal:
brew install texlive

# Server (Ubuntu):
apt-get install texlive-full  # or texlive-latex-extra for lighter install
```

## Critical Integration Notes

### lm-eval-harness with Custom Kazakh Tasks

lm-eval-harness supports custom tasks via YAML config files. For benchmarks not built in (KK Socio-Cultural, KazLLM translations), create YAML task definitions pointing to HuggingFace datasets:

```yaml
# Example: custom task for kk-socio-cultural-bench-mc
task: kk_sociocultural_mc
dataset_path: kz-transformers/kk-socio-cultural-bench-mc
output_type: multiple_choice
doc_to_text: "{{question}}\nA) {{A}}\nB) {{B}}\nC) {{C}}\nD) {{D}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: acc
```

### Evaluation Strategy for Small vs Large Models

- **Own models (50M-600M):** Load directly with `transformers`, evaluate via lm-eval-harness `hf` backend. Fast, fits on any GPU.
- **Competitor models (7B+):** Use `vllm` backend in lm-eval-harness (`--model vllm`) for fast batch inference. Quantize with AWQ/GPTQ if needed for A10.
- **Perplexity:** Compute on held-out Kazakh text (a split of the training corpus) as a basic language modeling metric.

### Paper Reproducibility Requirement

All evaluation commands should be captured in a Makefile or shell scripts under `scripts/eval/` so that results can be reproduced. Include exact model revision hashes from HuggingFace.

## Sources

- EleutherAI/lm-evaluation-harness: GitHub API verified -- v0.4.11 (released 2026-02-13), 11,770 stars, actively maintained. `belebele_kaz_Cyrl` task confirmed present. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- IS2AI/KazLLM_Benchmark: GitHub API verified -- benchmarks MMLU/ARC/HellaSwag/Winogrande/GSM8K/DROP/HumanEval in Kazakh. Uses `issai/KazLLM_Benchmark_Dataset`. [GitHub](https://github.com/IS2AI/KazLLM_Benchmark)
- IS2AI/KazQAD: GitHub API verified -- Kazakh open-domain QA, CC BY-SA 4.0, published at ACL 2024. [arXiv](https://arxiv.org/abs/2404.04487), [HuggingFace](https://huggingface.co/datasets/issai/kazqad)
- IS2AI/KazSAnDRA: GitHub API verified -- Kazakh sentiment analysis. [arXiv](https://arxiv.org/abs/2403.19335), [HuggingFace](https://huggingface.co/datasets/issai/kazsandra)
- IS2AI/KazNERD: GitHub API verified -- 112K sentences, 25 entity classes, CoNLL format. [GitHub](https://github.com/IS2AI/KazNERD)
- Belebele: Confirmed `belebele_kaz_Cyrl.yaml` exists in lm-eval-harness task configs (verified via GitHub API directory listing)
- Existing project eval: `scripts/eval/eval_mc_bench.py` evaluates on `kz-transformers/kk-socio-cultural-bench-mc` (7,111 questions)
- KazLLM reference model: `issai/LLama-3-KazLLM-1.0-8B-AWQ4` (from benchmark config)
