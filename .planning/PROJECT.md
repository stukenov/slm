# SozKZ Platform — saken.tukenov.kz

## What This Is

A marketing-oriented platform and media hub for Kazakh NLP, built around the SozKZ model family. The site serves as a one-stop destination for anyone who wants to use, evaluate, or contribute to Kazakh language models — from ML engineers who need an SDK to journalists who want to understand what's happening in Kazakh AI. Hosted at saken.tukenov.kz (Cloudflare).

## Core Value

Become the authoritative center of the Kazakh NLP ecosystem — the place people go to find models, try them, learn from them, and see who's building what.

## Requirements

### Validated

- ✓ 7 trained models published on HuggingFace (50M–600M) — existing
- ✓ GEC inference API running on kaznu server — existing
- ✓ Qazgramma-turbo playground (Next.js) — existing
- ✓ Experiment documentation in WHITEPAPER.md — existing
- ✓ GitHub repo stukenov/slm published — existing

### Active

- [ ] Marketing landing page with Linear/Stripe-style minimalism
- [ ] Model catalog page — all SozKZ models with sizes, metrics, HF links, download/try
- [ ] Playground — GEC correction and text generation in-browser (connects to kaznu API)
- [ ] Documentation — quickstart, SDK usage, fine-tuning guide, API reference
- [ ] Python SDK (pip install) for local model inference
- [ ] Hosted REST API (OpenAI-compatible) for quick integration
- [ ] Whitepaper/journey page — the story from 14M to 600M, experiments, lessons
- [ ] Success stories — Tilqazyna GEC integration and others
- [ ] Leaderboard — benchmark results for Kazakh LLMs (own and third-party)
- [ ] People rating — curated directory of KZ NLP/ML researchers and contributors
- [ ] Company rating — who uses NLP for Kazakh, what they build, comparisons
- [ ] SEO blog engine — automated AI-generated content, ~10 posts/day on Kazakh NLP topics
- [ ] Trilingual (kk/ru/en) — full i18n across the site

### Out of Scope

- User accounts / login system — not needed for v1, curator-driven content
- Community forum / comments — defer to Telegram/Discord
- Model hosting / inference marketplace — use HuggingFace for distribution
- Paid features / monetization — free platform for ecosystem growth

## Context

The SLM project has produced 26 experiments and 7+ published models for Kazakh, from 14M Pythia DAPT pilots to a 600M Llama trained on 9B tokens. The results are scattered across HuggingFace repos, WHITEPAPER.md, and private server logs. There's no single place where someone can discover this work, try the models, or start using them.

The Kazakh NLP ecosystem is small — KZ-Transformers, Tilqazyna, a few university groups. By creating the definitive platform with ratings, content, and tools, SozKZ can become the center of gravity that attracts researchers, developers, and organizations.

Existing infrastructure:
- GPU server (kaznu) at 164.138.46.36 — runs inference API (GEC models, POS tagger)
- Web server at 91.224.74.233 — hosts qazgramma-turbo, 1tv-wp
- Cloudflare — DNS for saken.tukenov.kz (already configured)
- qazgramma-turbo — existing Next.js playground that can be integrated/reused

The SEO content strategy targets long-tail Kazakh NLP queries — tutorials, comparisons, use cases — to build organic traffic and establish domain authority. AI-generated posts in three languages maximize coverage.

## Constraints

- **Hosting**: Cloudflare (Vercel for Next.js, or Cloudflare Pages)
- **Design**: Linear/Stripe-style minimalism — generous whitespace, strong typography, no clutter
- **Backend API**: Must proxy to kaznu inference server (164.138.46.36:15127)
- **Languages**: Trilingual kk/ru/en from day one
- **Content volume**: SEO blog needs CMS-like infrastructure for ~10 AI-generated posts/day
- **Ratings**: Curator-driven (Saken maintains), not crowd-sourced

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| saken.tukenov.kz (not s.tukenov.kz) | Already in Cloudflare, personal brand | — Pending |
| Linear/Stripe minimalism (not brutalist) | Accessible to non-technical audience while staying clean | — Pending |
| Cloudflare/Vercel hosting | Managed, CDN, free tier, no server maintenance | — Pending |
| AI-generated SEO content | Scale to 10 posts/day without manual writing | — Pending |
| Curator-driven ratings (not crowd-sourced) | Small ecosystem, quality over quantity | — Pending |
| SDK + hosted API (both) | Low barrier (API) + full control (SDK) for different users | — Pending |

---
*Last updated: 2026-03-19 after initialization*
