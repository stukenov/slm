# Roadmap: SozKZ Platform

**Created:** 2026-03-19
**Granularity:** Fine (8-12 phases)
**Core Value:** Become the authoritative center of the Kazakh NLP ecosystem

## Phases

### Phase 1: Project Setup & Infrastructure
**Goal:** Next.js app deployed on Cloudflare Pages with i18n routing, ready for content.

**Requirements:** INFR-01, INFR-02, INFR-03, INFR-04

**Plans:** 2/3 plans executed

Plans:
- [ ] 01-01-PLAN.md — Scaffold Next.js project with i18n routing and layout shell
- [ ] 01-02-PLAN.md — SEO fundamentals (sitemap, robots, meta tags) and Playwright e2e tests
- [ ] 01-03-PLAN.md — Deploy to Cloudflare Pages and configure DNS

**Success Criteria:**
1. `saken.tukenov.kz` resolves and shows a placeholder page
2. Routes work in all three locales (`/kk/`, `/ru/`, `/en/`)
3. Language switcher toggles between kk/ru/en
4. sitemap.xml, robots.txt, and OG meta tags present
5. Cloudflare Pages deploys automatically on git push

---

### Phase 2: Landing Page
**Goal:** Marketing landing page that communicates SozKZ value proposition and drives visitors to key sections.

**Requirements:** LAND-01, LAND-02, LAND-03, LAND-04, LAND-05

**Success Criteria:**
1. Hero section visible with mission text and key model numbers (50M->600M)
2. Model highlight cards show top 3 models with params and type
3. Social proof section mentions Tilqazyna and use cases
4. CTA buttons navigate to Playground, Models, Docs, GitHub
5. Landing renders correctly in kk, ru, and en

---

### Phase 3: Model Catalog & Cards
**Goal:** Users can browse all SozKZ models and see detailed info per model.

**Requirements:** MODL-01, MODL-02, MODL-03, MODL-04, MODL-05

**Success Criteria:**
1. `/models` page lists all SozKZ models with key info
2. User can filter by size, type (base/instruct/GEC), task
3. Each model has a dedicated card page with architecture, metrics, training details
4. Code snippet on each card shows pip install + 3-line inference example
5. HuggingFace download link works on each model card

---

### Phase 4: Playground
**Goal:** Users can try GEC and text generation directly in browser, connected to kaznu inference API.

**Requirements:** PLAY-01, PLAY-02, PLAY-03, PLAY-04, PLAY-05, PLAY-06

**Success Criteria:**
1. User inputs text and gets GEC correction back
2. User inputs prompt and gets generated text with streaming tokens
3. Model selector allows choosing between available models
4. When kaznu is down, user sees friendly error with HF download fallback
5. Browser devtools shows requests go to saken.tukenov.kz (not kaznu IP directly)

---

### Phase 5: Documentation
**Goal:** Developers can go from zero to inference in 5 minutes following documentation.

**Requirements:** DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05

**Success Criteria:**
1. Quickstart guide walks user from pip install to first inference result
2. API reference documents all endpoints with request/response examples
3. Fine-tuning guide shows how to fine-tune a SozKZ model on custom data
4. SDK reference covers all public methods with type signatures
5. All docs pages render in kk, ru, and en

---

### Phase 6: Python SDK
**Goal:** Developers can pip install sozkz and run inference with 3 lines of code.

**Requirements:** SDK-01, SDK-02, SDK-03, SDK-04, SDK-05

**Success Criteria:**
1. `pip install sozkz` succeeds from PyPI
2. User loads a model and runs inference in ≤3 lines of code
3. GEC correction works via SDK
4. Text generation works via SDK
5. Package has README with quickstart on PyPI page

---

### Phase 7: Hosted API
**Goal:** Users can integrate Kazakh NLP via REST API without running models locally.

**Requirements:** API-01, API-02, API-03, API-04

**Success Criteria:**
1. POST to `/v1/chat/completions` with GEC text returns corrected text
2. POST to `/v1/chat/completions` with generation prompt returns generated text
3. Response format matches OpenAI chat completions schema
4. Requests beyond rate limit return 429 with retry-after header

---

### Phase 8: Leaderboard
**Goal:** Users can compare Kazakh LLMs across benchmarks in a transparent ranked table.

**Requirements:** LEAD-01, LEAD-02, LEAD-03, LEAD-04

**Success Criteria:**
1. `/leaderboard` shows ranked table of Kazakh LLMs
2. Table includes at least 2 third-party models alongside SozKZ models
3. Methodology page explains eval scripts, prompts, and scoring
4. Multiple benchmark columns visible (perplexity, GEC accuracy, generation quality)

---

### Phase 9: People & Company Ratings
**Goal:** Users can discover who works on Kazakh NLP — researchers and companies.

**Requirements:** RATE-01, RATE-02, RATE-03

**Success Criteria:**
1. `/people` shows curated directory of KZ NLP researchers
2. `/companies` shows curated directory of companies using Kazakh NLP
3. Each card has name, affiliation/product, focus area, and external links

---

### Phase 10: Blog Engine
**Goal:** Blog infrastructure ready for content — manual and automated posts.

**Requirements:** BLOG-01, BLOG-02, BLOG-03, BLOG-04

**Success Criteria:**
1. `/blog` lists published posts with pagination
2. Blog posts render correctly in kk, ru, and en
3. Posts have categories (tutorials, benchmarks, news, case studies)
4. Each post has meta tags, OG image, and structured data for SEO

---

### Phase 11: SEO Content Pipeline
**Goal:** Automated generation of ~10 blog posts/day across 3 languages.

**Requirements:** BLOG-05

**Success Criteria:**
1. Script generates MDX blog posts via AI API (Claude/GPT)
2. Generated posts commit to repo and trigger Cloudflare rebuild
3. Pipeline produces posts in kk, ru, and en for each topic
4. Posts contain real SozKZ data (model names, benchmark numbers, code examples)
5. Pipeline can run as cron job or GitHub Action

---

### Phase 12: Content Pages (Journey & Stories)
**Goal:** Narrative pages that tell the SozKZ story and showcase real-world use.

**Requirements:** CONT-01, CONT-02

**Success Criteria:**
1. Journey page tells the story from 14M to 600M with experiment timeline
2. Success stories page features Tilqazyna GEC integration with details
3. Both pages are trilingual

---

## Summary

| # | Phase | Requirements | Success Criteria |
|---|-------|-------------|-----------------|
| 1 | 2/3 | In Progress|  |
| 2 | Landing Page | LAND-01–05 | 5 |
| 3 | Model Catalog & Cards | MODL-01–05 | 5 |
| 4 | Playground | PLAY-01–06 | 5 |
| 5 | Documentation | DOCS-01–05 | 5 |
| 6 | Python SDK | SDK-01–05 | 5 |
| 7 | Hosted API | API-01–04 | 4 |
| 8 | Leaderboard | LEAD-01–04 | 4 |
| 9 | People & Company Ratings | RATE-01–03 | 3 |
| 10 | Blog Engine | BLOG-01–04 | 4 |
| 11 | SEO Content Pipeline | BLOG-05 | 5 |
| 12 | Content Pages | CONT-01–02 | 3 |

**12 phases** | **44 requirements mapped** | **53 success criteria** | All v1 requirements covered

---
*Created: 2026-03-19*
