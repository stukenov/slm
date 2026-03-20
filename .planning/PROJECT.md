# SozKZ arXiv Paper: Efficient Small Language Model for Kazakh

## What This Is

An arXiv paper demonstrating that a small, purpose-built language model (≤600M parameters) trained from scratch on Kazakh can match or approach the performance of general-purpose models 10-100x larger on core Kazakh language tasks. The paper serves both as a technical contribution (full pipeline for low-resource language LM training) and as a portfolio piece showcasing end-to-end ML engineering capabilities.

## Core Value

Prove with benchmarks that a small specialized model is a viable and efficient alternative to large multilingual models for Kazakh language tasks.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Identify and select Kazakh language benchmarks covering multiple task types (generation, understanding, classification, translation)
- [ ] Build an evaluation pipeline that runs reproducibly across all models
- [ ] Evaluate own models (50M, 150M, 300M, 600M) on selected benchmarks
- [ ] Evaluate competitor models (Gemma, GPT-OSS-120B, Llama-3, Mistral, Qwen) on same benchmarks
- [ ] Produce comparison tables and analysis (params vs performance, cost efficiency)
- [ ] Determine the "hero model" based on evaluation results
- [ ] Write arXiv paper in LaTeX: abstract, intro, related work, methodology, experiments, results, conclusion
- [ ] Create figures: scaling curves, comparison charts, architecture diagrams
- [ ] Document full training pipeline (data, tokenizer, architecture, hyperparams) in the paper
- [ ] Publish paper to arXiv

### Out of Scope

- Training new models for the paper — use existing trained models
- Fine-tuning competitors — compare base/instruct checkpoints as-is
- Peer review / journal submission — arXiv preprint is the goal
- GEC-specific paper — GEC is one benchmark, not the focus

## Context

- **Existing models**: 26 experiments (exp001–exp026), models from 50M to 600M, including MoE 3B, GEC fine-tunes, SFT variants
- **Tokenizer**: Custom kazakh-bpe-32k (ByteLevel BPE, 32K vocab)
- **Training data**: kz-transformers/multidomain-kazakh-dataset (23.6M samples), plus tokenized variants
- **Known results**: GEC fine-tune was weak; generation quality subjectively good; no formal benchmarks yet
- **Competitors observed**: Gemma = decent on Kazakh, GPT-OSS-120B = best, most large models = weak
- **Infrastructure**: 2xA10 (kaznu server), vast.ai for larger runs (up to 8xH100)
- **Naming**: All HF repos follow SozKZ naming convention under saken-tukenov/

## Constraints

- **Models**: Use already-trained checkpoints — no new training runs for the paper
- **Compute**: Evaluation must run on available hardware (2xA10 or vast.ai)
- **Format**: Standard arXiv ML paper (LaTeX, 8-12 pages)
- **Honesty**: Report results truthfully — if the model loses on some tasks, acknowledge it and frame as efficiency tradeoff

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Efficiency angle (option 1) | "Small model matches big ones" is compelling narrative + honest framing | — Pending |
| Hero model TBD | Let benchmarks decide which model to highlight | — Pending |
| arXiv preprint, not journal | Speed to publish, broader reach, portfolio value | — Pending |

---
*Last updated: 2026-03-20 after initialization*
