# Features Research: ML Model Platform

## Reference Platforms Studied

- **mistral.ai** — marketing + docs + playground + API
- **cohere.com** — model catalog + playground + docs
- **replicate.com** — model catalog + playground + API
- **huggingface.co** — model hub + spaces + leaderboard
- **lmsys.org** (Chatbot Arena) — leaderboard + arena
- **openrouter.ai** — model catalog + unified API + pricing

## Table Stakes (must have or users leave)

### Landing Page
- **Hero section** with clear value proposition
- **Model highlights** — key numbers (params, languages, benchmarks)
- **Call to action** — try playground, read docs, download models
- **Social proof** — who uses it (Tilqazyna etc.)
- Complexity: LOW | Dependencies: none

### Model Catalog
- **List of all models** with filters (size, type, task)
- **Model card** per model — architecture, training data, metrics, download links
- **HuggingFace links** — direct to HF for download
- **Code snippets** — pip install + 3 lines to run
- Complexity: LOW | Dependencies: model data JSON

### Documentation
- **Quickstart** — from zero to inference in 5 minutes
- **API reference** — endpoints, parameters, response format
- **Fine-tuning guide** — how to fine-tune SozKZ models
- **SDK reference** — Python package API docs
- Complexity: MEDIUM | Dependencies: SDK must exist first

### Playground
- **Text input → model output** for GEC and generation
- **Model selector** — choose which model to try
- **Response time indicator** — users expect <3s
- Complexity: MEDIUM | Dependencies: inference API availability

## Differentiators (competitive advantage)

### Kazakh NLP Leaderboard
- **Benchmark table** — perplexity, GEC accuracy, generation quality across models
- **Third-party models included** — not just SozKZ, also KZ-Transformers, multilingual models
- **Methodology transparency** — eval scripts, prompts, scoring rubric published
- Complexity: MEDIUM | Dependencies: eval framework must exist
- **Why differentiating:** No Kazakh-specific leaderboard exists. LMSYS doesn't cover Kazakh.

### People Directory
- **Curated list** of KZ NLP/ML researchers
- **Profile cards** — name, affiliation, focus area, links (HF, GitHub, Twitter)
- **"Following" framing** — people Saken follows/respects, not ranking by skill
- Complexity: LOW | Dependencies: none

### Company Directory
- **Who uses Kazakh NLP** — companies, products, use cases
- **Integration stories** — how they use models
- Complexity: LOW | Dependencies: none

### SEO Blog
- **AI-generated articles** — tutorials, comparisons, news, use cases
- **Trilingual** — same topic in kk/ru/en
- **10 posts/day** — long-tail keyword coverage
- **Categories** — tutorials, benchmarks, news, case studies
- Complexity: MEDIUM | Dependencies: content generation pipeline

### Journey/Whitepaper
- **Interactive story** — from 14M to 600M, with charts and lessons
- **Experiment timeline** — visual progression
- Complexity: LOW | Dependencies: WHITEPAPER.md content exists

### Success Stories
- **Tilqazyna GEC integration** — how it works, results
- **Template for future stories**
- Complexity: LOW | Dependencies: none

## Anti-Features (deliberately NOT building)

| Feature | Why Not |
|---------|---------|
| User accounts / login | Adds complexity, no need for curator-driven platform |
| Model hosting / serving | Use HuggingFace for distribution, kaznu for playground only |
| Community forum / comments | Defer to Telegram/Discord — lower maintenance |
| Model training UI | This is a showcase, not a training platform |
| Payment / monetization | Free platform for ecosystem growth in v1 |
| Real-time collaboration | Single curator, no multi-user editing needed |
| Model comparison arena | Would need significant infra; leaderboard covers this |

## Feature Dependencies

```
Landing Page (independent)
  ↓
Model Catalog → Model Cards → Code Snippets
  ↓                              ↓
Playground (needs API)     Documentation (needs SDK)
  ↓
Leaderboard (needs eval data)
  ↓
Blog (independent, can start early)
People/Companies (independent)
Journey page (independent, content exists)
```
