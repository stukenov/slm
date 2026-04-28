# Grant Preparation Brief — SozKZ / SLM

Date: 2026-03-27
Author: Codex analysis pass

## 1. Project Snapshot

SLM / SozKZ is an open-source effort to build small and mid-size language models for Kazakh from scratch, with reproducible training configs, custom tokenizers, data cleaning pipelines, evaluation scripts, and cloud training automation.

Core grant-worthy claim:

Dedicated Kazakh models and tokenizers can achieve competitive results against much larger multilingual models while using far less compute, lower cost, and fully open artifacts.

## 2. Strongest Evidence Already Present in Repo

### Scientific / technical

- Custom Kazakh tokenizer shows materially better efficiency than multilingual tokenizers.
- Large cleaned Kazakh corpus exists, with documented multi-stage filtering and deduplication.
- Multiple model scales already trained from scratch: 50M, 150M, 300M, 600M.
- Benchmarks are already packaged into a paper-ready evaluation suite.
- A next-stage bilingual kk-ru pipeline is underway.

### Quantitative facts to reuse in grant applications

- Raw collection: 28.4M documents.
- Cleaned corpus: 13.7M documents after a 9-stage pipeline.
- Training corpus size: about 9B Kazakh tokens for core SozKZ models.
- Tokenizer efficiency: SozKZ tokenizer reaches 5.82 chars/token vs 2.18-2.53 for major multilingual tokenizers.
- SozKZ-600M:
  - MC QA: 30.3%
  - Belebele: 27.0%
  - SIB-200: 25.5%
- On SIB-200, SozKZ models outperform several much larger multilingual baselines up to 2B parameters.
- 150M training is extremely cheap relative to outcome: about $17 in the logged run.
- 1B-scale training has already been attempted, so the team has operational experience beyond paper scale.

### Infrastructure / execution

- Reproducible experiment configs under `configs/experiments/`.
- Modular Python package under `src/slm/`: `train`, `data`, `tokenizer`, `evaluate`, `publish`, `cloud`.
- Cloud launcher for vast.ai already exists.
- Paper assets are structured and mostly reproducible from scripts and JSON outputs.
- Hugging Face publication workflow is already part of the repo.

## 3. Best Grant Positioning Angles

### Angle A: AI for underrepresented languages

SozKZ addresses a clear market and public-good gap: Kazakh is under-served by mainstream LLMs despite large cultural and civic importance.

### Angle B: Open digital infrastructure

The project is not just a model release. It is a reproducible infrastructure stack:

- corpus collection
- cleaning
- tokenization
- training
- evaluation
- publication

This is attractive for grants focused on open infrastructure or digital public goods.

### Angle C: Cost-efficient sovereign AI

The repo demonstrates that useful Kazakh models can be trained without frontier-model budgets. This supports narratives around regional AI capability, compute efficiency, and local-language sovereignty.

### Angle D: Research platform, not a one-off demo

The experiment log, paper, and config discipline show a sustained research program rather than a single prototype.

### Angle E: Bilingual and regional expansion

The ongoing EkiTil work opens a second narrative:

- Kazakh-Russian bilingual infrastructure
- translation and cross-lingual transfer
- future extension to other Turkic languages

## 4. Likely Grant Categories to Search

- Low-resource language technology
- Digital public goods / open-source AI infrastructure
- AI for education and culture
- Language preservation and linguistic diversity
- Central Asia innovation and research capacity
- Responsible / inclusive AI
- Open research infrastructure
- Translation and multilingual access
- Public-interest AI

## 5. Search Keywords

Use these combinations when searching later:

- "low-resource language AI grant"
- "NLP grant underrepresented languages"
- "language technology grant open source"
- "digital public goods AI grant"
- "AI for linguistic diversity grant"
- "Kazakh language technology funding"
- "Central Asia research grant AI"
- "open-source LLM grant"
- "machine translation low-resource languages grant"
- "Turkic language NLP grant"
- "AI sovereignty local language models funding"

## 6. Narrative To Lead With

Recommended short narrative:

SozKZ is building open, reproducible language AI infrastructure for Kazakh, an underrepresented Turkic language. The project has already produced custom tokenizers, large cleaned corpora, and a family of from-scratch language models up to 600M parameters, with evidence that dedicated Kazakh models can match or exceed much larger multilingual systems on some Kazakh benchmarks. Funding would be used to expand data quality, scale training, complete bilingual Kazakh-Russian models, strengthen evaluations, and convert the research stack into durable public infrastructure for education, research, and local AI applications.

## 7. Risks And Gaps To Fix Before Outreach

These should be cleaned up before sending grants or funder emails.

### Messaging inconsistencies

- Hugging Face namespace is inconsistent across files: `stukenov/` vs `saken-tukenov/`.
- Kazakh speaker count is inconsistent: some docs say about 13M, paper LaTeX says over 22M.
- `docs/paper/paper.md` reflects an earlier 50M/150M narrative, while LaTeX `paper/` reflects the later 50M-600M story.

### Portfolio focus

The repo contains adjacent projects (`kz-calm`, `translation`, `parameter-golf`, `omniaudio`) that are useful internally but dilute the main SozKZ story for outsiders. Grant-facing materials should foreground one primary program.

### Evidence hierarchy

The 600M results are strong grant material.
The 1B run is technically impressive but currently should be framed as an operational lesson, not the headline achievement, because the exported HF artifact is degraded.

### Benchmark limitations

The repo already documents that some tasks remain weak or near baseline. This is acceptable, but grant language should position funding as the mechanism to close those gaps via better data, SFT, and larger-scale training.

## 8. What To Prepare Next For Grant Search

Before searching specific grants, prepare these reusable artifacts:

1. One-paragraph project summary.
2. 1-page concept note.
3. 6-month and 12-month milestone plan.
4. Budget tiers:
   - small grant: $10k-$25k
   - medium grant: $25k-$100k
   - research-scale grant: $100k+
5. Clear use-of-funds breakdown:
   - compute
   - data work
   - annotation / evaluation
   - engineering
   - publication / dissemination
6. Impact metrics:
   - number of released models
   - corpus size and quality
   - benchmark improvements
   - downstream adoption
7. Public-good commitment:
   - open weights where possible
   - open datasets / cards / evaluation scripts
   - open model cards and reproducibility

## 9. Suggested Work Packages For Funding

### WP1: Data quality and corpus expansion

- Improve filtering, deduplication, and domain balance
- Add Latin-script Kazakh support
- Expand higher-quality educational and literary text

### WP2: Next-generation base models

- Retrain 1B-class model with HF-compatible architecture
- Improve tokens-to-parameter ratio
- Continue scaling study toward 1B-3B

### WP3: Bilingual kk-ru platform

- Finish EkiTil tokenized corpus
- Train kk-ru bilingual model
- Add translation-focused evaluation

### WP4: Instruction tuning and applied use cases

- Higher-quality SFT data
- Better QA / education / government-style task evaluation
- Practical demo endpoints for local adoption

### WP5: Benchmark and ecosystem building

- Standardized Kazakh benchmark suite
- Public leaderboards / reproducible evaluation
- Documentation for researchers and integrators

## 10. Recommended Grant Search Filter

Prioritize funders that value:

- open-source outputs
- public-interest AI
- language access
- research infrastructure
- regional capacity building
- education or culture impact

Deprioritize grants that require:

- exclusive IP transfer
- closed commercial deliverables only
- mature revenue traction as the main criterion
- biomedical or unrelated vertical framing

## 11. Immediate Next Step

Use this brief as the basis for live grant search. The search should target:

1. open-source / public-interest AI funders
2. language and cultural preservation programs
3. academic or applied NLP research grants
4. Central Asia or emerging-market digital innovation funds

When searching, rank opportunities by:

- fit with open research
- grant size
- eligibility for an independent researcher
- tolerance for open licensing
- support for compute and data work
