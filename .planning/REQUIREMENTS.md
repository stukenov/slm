# Requirements: SozKZ arXiv Paper

**Defined:** 2026-03-20
**Core Value:** Prove with benchmarks that a small specialized Kazakh model is a viable and efficient alternative to large multilingual models.

## v1 Requirements

Requirements for the paper. Each maps to roadmap phases.

### Evaluation Infrastructure

- [x] **EVAL-01**: BPB (bits-per-byte) computation pipeline for any HF model on held-out Kazakh text
- [x] **EVAL-02**: MC QA scoring on kk-socio-cultural-bench-mc (fix existing 10% accuracy bug)
- [x] **EVAL-03**: Sentiment classification evaluation on Kazakh dataset
- [x] **EVAL-04**: Belebele reading comprehension evaluation on Kazakh subset (kaz_Cyrl)
- [x] **EVAL-05**: NER evaluation on KazNERD dataset
- [x] **EVAL-06**: Topic classification evaluation on SIB-200 Kazakh subset
- [x] **EVAL-07**: Model registry covering own models (50M, 150M, 300M, 600M) and competitors (Gemma-2B, Gemma-7B, Llama-3-1B, Llama-3-3B, Llama-3-8B, Qwen-2.5-0.5B, Qwen-2.5-1.5B, Qwen-2.5-7B, GPT-OSS-120B, Mistral-7B)

### Analysis

- [x] **ANLS-01**: Tokenizer fertility comparison (chars/token) across all model tokenizers on Kazakh text
- [ ] **ANLS-02**: Scaling curves -- performance vs parameter count for own models (50M/150M/300M/600M)
- [x] **ANLS-03**: Efficiency metrics -- inference latency, throughput (tok/s), peak memory per model
- [x] **ANLS-04**: Contamination check -- n-gram overlap between training data and all benchmark datasets
- [x] **ANLS-05**: Auto-generated comparison tables and charts (matplotlib PDF/PNG export)

### Paper

- [ ] **PAPR-01**: LaTeX paper with sections: abstract, introduction, related work, methodology, experiments, results, conclusion
- [ ] **PAPR-02**: Training methodology section documenting data pipeline, tokenizer design, architecture choices, hyperparameters
- [ ] **PAPR-03**: Results section with comparison tables (own models vs competitors across all benchmarks)
- [ ] **PAPR-04**: Tokenizer analysis section showing fertility advantage of dedicated Kazakh tokenizer
- [ ] **PAPR-05**: Scaling analysis section with curves showing diminishing returns / sweet spot
- [ ] **PAPR-06**: Figures -- scaling curves, radar/bar comparison charts, architecture diagram, tokenizer fertility plot
- [ ] **PAPR-07**: Submit final PDF to arXiv

## v2 Requirements

Deferred to revision or follow-up paper.

### Additional Baselines

- **V2-01**: Fine-tuning baseline (Gemma-2B fine-tuned on Kazakh data) to counter "why not fine-tune?" question
- **V2-02**: IS2AI KazLLM-8B comparison (Kazakh-specific competitor)

### Extended Evaluation

- **V2-03**: Machine translation evaluation (FLORES-200 Kazakh)
- **V2-04**: GEC evaluation (grammatical error correction)
- **V2-05**: Statistical significance tests (paired bootstrap)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training new models | Use existing checkpoints, no compute budget for new training |
| Fine-tuning competitors | May add as v2 if reviewers demand it |
| Journal submission | arXiv preprint is the goal for speed and reach |
| Closed-source models (GPT-4, Claude) | Not reproducible, can't run locally |
| Kazakh-specific pretraining data release | Licensing unclear, separate effort |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | Phase 1 | Complete |
| EVAL-02 | Phase 1 | Complete |
| EVAL-03 | Phase 1 | Complete |
| EVAL-04 | Phase 1 | Complete |
| EVAL-05 | Phase 1 | Complete |
| EVAL-06 | Phase 1 | Complete |
| EVAL-07 | Phase 1 | Complete |
| ANLS-01 | Phase 2 | Complete |
| ANLS-02 | Phase 2 | Pending |
| ANLS-03 | Phase 2 | Complete |
| ANLS-04 | Phase 2 | Complete |
| ANLS-05 | Phase 2 | Complete |
| PAPR-01 | Phase 3 | Pending |
| PAPR-02 | Phase 3 | Pending |
| PAPR-03 | Phase 3 | Pending |
| PAPR-04 | Phase 3 | Pending |
| PAPR-05 | Phase 3 | Pending |
| PAPR-06 | Phase 3 | Pending |
| PAPR-07 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after roadmap creation*
