# Requirements: SozKZ Platform

**Defined:** 2026-03-19
**Core Value:** Become the authoritative center of the Kazakh NLP ecosystem

## v1 Requirements

### Landing

- [x] **LAND-01**: User sees hero section with clear value proposition and key model numbers
- [x] **LAND-02**: User sees highlighted models (600M, 300M, 150M) with params and task type
- [x] **LAND-03**: User sees social proof section (Tilqazyna integration, partners)
- [x] **LAND-04**: User can navigate to Playground, Docs, Models, GitHub via CTA buttons
- [x] **LAND-05**: Landing is trilingual (kk/ru/en) with language switcher

### Models

- [ ] **MODL-01**: User can browse model catalog with all SozKZ models
- [ ] **MODL-02**: User can filter models by size, type (base/instruct/GEC), and task
- [ ] **MODL-03**: User can view model card with architecture, training data, metrics, download links
- [ ] **MODL-04**: User sees code snippet (pip install + 3 lines to run) on each model card
- [ ] **MODL-05**: User can click through to HuggingFace for model download

### Playground

- [ ] **PLAY-01**: User can input text and get GEC correction from sozkz-gec models
- [ ] **PLAY-02**: User can input prompt and get text generation from base models
- [ ] **PLAY-03**: User can select which model to use
- [ ] **PLAY-04**: User sees streaming response for text generation
- [ ] **PLAY-05**: User sees graceful error message when inference server is unavailable
- [ ] **PLAY-06**: Playground proxies through Vercel/Cloudflare edge (GPU server IP not exposed)

### Documentation

- [ ] **DOCS-01**: User can read quickstart guide (from zero to inference in 5 minutes)
- [ ] **DOCS-02**: User can read API reference (endpoints, parameters, response format)
- [ ] **DOCS-03**: User can read fine-tuning guide (how to fine-tune SozKZ models)
- [ ] **DOCS-04**: User can read SDK reference (Python package API docs)
- [ ] **DOCS-05**: Documentation is trilingual (kk/ru/en)

### SDK

- [ ] **SDK-01**: User can pip install sozkz package and run local inference
- [ ] **SDK-02**: User can load any SozKZ model with 3 lines of code
- [ ] **SDK-03**: User can run GEC correction via SDK
- [ ] **SDK-04**: User can run text generation via SDK
- [ ] **SDK-05**: SDK published on PyPI

### API

- [ ] **API-01**: User can send text to hosted REST API and get GEC correction
- [ ] **API-02**: User can send prompt to hosted REST API and get text generation
- [ ] **API-03**: API is OpenAI-compatible format (chat completions)
- [ ] **API-04**: API has rate limiting

### Leaderboard

- [ ] **LEAD-01**: User can view benchmark results for Kazakh LLMs in a ranked table
- [ ] **LEAD-02**: Leaderboard includes third-party models (not just SozKZ)
- [ ] **LEAD-03**: User can see methodology (eval scripts, prompts, scoring) for transparency
- [ ] **LEAD-04**: Leaderboard shows multiple benchmarks per model (perplexity, GEC, generation)

### Ratings

- [ ] **RATE-01**: User can browse curated directory of KZ NLP/ML researchers with profiles
- [ ] **RATE-02**: User can browse curated directory of companies using Kazakh NLP
- [ ] **RATE-03**: Each person/company card has name, affiliation, focus area, links

### Blog & SEO

- [ ] **BLOG-01**: User can read AI-generated blog posts about Kazakh NLP
- [ ] **BLOG-02**: Blog posts exist in all three languages (kk/ru/en)
- [ ] **BLOG-03**: Blog has categories (tutorials, benchmarks, news, case studies)
- [ ] **BLOG-04**: Blog posts are SEO-optimized (meta tags, structured data, sitemaps)
- [ ] **BLOG-05**: Content generation pipeline produces ~10 posts/day automatically

### Content Pages

- [ ] **CONT-01**: User can read journey page (story from 14M to 600M with experiment timeline)
- [ ] **CONT-02**: User can read success stories (Tilqazyna GEC integration)

### Infrastructure

- [x] **INFR-01**: Site deployed on Cloudflare Pages via @opennextjs/cloudflare
- [x] **INFR-02**: Cloudflare DNS configured for saken.tukenov.kz
- [x] **INFR-03**: i18n routing with next-intl (/kk/, /ru/, /en/)
- [x] **INFR-04**: SEO fundamentals (sitemap.xml, robots.txt, meta tags, OG images)

## v2 Requirements

### Community

- **COMM-01**: Telegram/Discord community integration links
- **COMM-02**: Newsletter signup
- **COMM-03**: Contribution guide for researchers

### Analytics

- **ANLY-01**: Visitor analytics dashboard
- **ANLY-02**: Playground usage metrics
- **ANLY-03**: API usage tracking

### Advanced Playground

- **APLY-01**: Model comparison side-by-side
- **APLY-02**: Batch processing mode
- **APLY-03**: Shareable playground links

## Out of Scope

| Feature | Reason |
|---------|--------|
| User accounts / authentication | Curator-driven platform, no user data needed |
| Model hosting / serving marketplace | Use HuggingFace for distribution |
| Community forum / comments | Defer to Telegram/Discord — lower maintenance |
| Model training UI | Showcase platform, not training platform |
| Payment / monetization | Free ecosystem platform in v1 |
| Real-time chat/collaboration | Single curator, no multi-user editing |
| Model comparison arena (Chatbot Arena-style) | Requires significant infra, leaderboard sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| LAND-01 | Phase 2 | Complete |
| LAND-02 | Phase 2 | Complete |
| LAND-03 | Phase 2 | Complete |
| LAND-04 | Phase 2 | Complete |
| LAND-05 | Phase 2 | Complete |
| MODL-01 | Phase 3 | Pending |
| MODL-02 | Phase 3 | Pending |
| MODL-03 | Phase 3 | Pending |
| MODL-04 | Phase 3 | Pending |
| MODL-05 | Phase 3 | Pending |
| PLAY-01 | Phase 4 | Pending |
| PLAY-02 | Phase 4 | Pending |
| PLAY-03 | Phase 4 | Pending |
| PLAY-04 | Phase 4 | Pending |
| PLAY-05 | Phase 4 | Pending |
| PLAY-06 | Phase 4 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |
| DOCS-04 | Phase 5 | Pending |
| DOCS-05 | Phase 5 | Pending |
| SDK-01 | Phase 6 | Pending |
| SDK-02 | Phase 6 | Pending |
| SDK-03 | Phase 6 | Pending |
| SDK-04 | Phase 6 | Pending |
| SDK-05 | Phase 6 | Pending |
| API-01 | Phase 7 | Pending |
| API-02 | Phase 7 | Pending |
| API-03 | Phase 7 | Pending |
| API-04 | Phase 7 | Pending |
| LEAD-01 | Phase 8 | Pending |
| LEAD-02 | Phase 8 | Pending |
| LEAD-03 | Phase 8 | Pending |
| LEAD-04 | Phase 8 | Pending |
| RATE-01 | Phase 9 | Pending |
| RATE-02 | Phase 9 | Pending |
| RATE-03 | Phase 9 | Pending |
| BLOG-01 | Phase 10 | Pending |
| BLOG-02 | Phase 10 | Pending |
| BLOG-03 | Phase 10 | Pending |
| BLOG-04 | Phase 10 | Pending |
| BLOG-05 | Phase 11 | Pending |
| CONT-01 | Phase 12 | Pending |
| CONT-02 | Phase 12 | Pending |
| INFR-01 | Phase 1 | Complete |
| INFR-02 | Phase 1 | Complete |
| INFR-03 | Phase 1 | Complete |
| INFR-04 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 44 total
- Mapped to phases: 44
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-19*
*Last updated: 2026-03-19 after roadmap creation*
