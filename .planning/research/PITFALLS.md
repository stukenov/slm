# Pitfalls Research: ML Model Platform

## 1. SEO Spam Detection

**Risk: HIGH**

10 AI-generated posts/day across 3 languages = 300 posts/month. Google's SpamBrain actively penalizes "scaled content abuse" (March 2024 update).

**Warning signs:**
- Sudden traffic drop in Search Console
- Pages excluded as "Discovered – currently not indexed"
- Manual action notification

**Prevention:**
- Each post must have **unique value** — not just rephrased versions of the same topic
- Include real data from experiments (benchmark numbers, code snippets, specific model comparisons)
- Interlink with model cards and playground (signals genuine content)
- Start with 2-3 posts/day, scale up gradually
- Monitor Search Console indexing rate — if <50% indexed, quality is too low

**Phase:** SEO content pipeline (late phase)

## 2. Single Point of Failure: kaznu Server

**Risk: HIGH**

Playground and API depend on one GPU server (kaznu, 164.138.46.36). If it goes down, playground shows errors to every visitor.

**Warning signs:**
- Playground timeout errors
- SSH connection failures (already known to be flaky)

**Prevention:**
- **Graceful degradation** — playground shows "Model temporarily unavailable" with HuggingFace download link as fallback
- **Health check** — API route pings kaznu, caches status, shows indicator on playground
- **Static fallback** — pre-recorded example outputs shown when server is down
- Consider HuggingFace Inference Endpoints as backup (paid but reliable)

**Phase:** Playground implementation

## 3. Trilingual Content Quality

**Risk: MEDIUM**

AI-generated Kazakh content may have grammatical errors — ironic for a platform showcasing Kazakh language models.

**Warning signs:**
- Native speakers flag errors in blog posts
- Kazakh text looks machine-translated from Russian/English

**Prevention:**
- Use SozKZ GEC model to proofread AI-generated Kazakh content (dogfooding!)
- Have a few native speakers review initial batch
- Kazakh as primary language, not translated afterthought
- Blog generation pipeline: generate in target language directly, don't translate

**Phase:** Blog engine + content pipeline

## 4. Leaderboard Bias Perception

**Risk: MEDIUM**

If SozKZ models top the leaderboard, it looks self-serving. If they don't, it undermines the platform.

**Warning signs:**
- Community skepticism about results
- Accusations of cherry-picked benchmarks

**Prevention:**
- **Methodology transparency** — publish eval scripts, prompts, datasets
- **Include ALL models** — even those that beat SozKZ
- **Multiple benchmarks** — no single score, show strengths/weaknesses per task
- **Reproducibility** — anyone can re-run evals
- Frame as "Kazakh NLP ecosystem benchmark" not "our models are best"

**Phase:** Leaderboard implementation

## 5. Content Maintenance Burden

**Risk: MEDIUM**

300 posts/month + model updates + benchmark updates + people/company ratings = significant maintenance for a solo operator.

**Warning signs:**
- Outdated model cards (new model released but card not updated)
- Stale blog posts referencing deprecated features
- Dead links to moved HuggingFace repos

**Prevention:**
- **Automate everything** — model cards generated from HF metadata, benchmarks from eval scripts
- **Content as code** — changes are PRs, easy to review and batch
- **Timestamps on everything** — "Last updated: date" visible to users
- **Prioritize automation over comprehensiveness** — better to have fewer accurate pages than many stale ones

**Phase:** Throughout, but especially content pipeline design

## 6. Playground Latency

**Risk: MEDIUM**

kaznu server inference + network latency (proxy through Vercel) could make playground feel slow.

**Warning signs:**
- Response times >5s
- Users abandon playground before seeing results

**Prevention:**
- **Streaming responses** — show tokens as they generate (text generation)
- **Loading indicators** — clear progress feedback
- **Pre-warming** — keep models loaded in memory on kaznu
- **Response caching** — cache common inputs (especially for GEC)
- **Set expectations** — show estimated time before submission

**Phase:** Playground implementation

## 7. Overbuilding Before Validating

**Risk: MEDIUM**

Building SDK, API, docs, blog engine, leaderboard, ratings all at once before anyone visits the site.

**Warning signs:**
- Months of development without a live URL
- Features nobody asked for

**Prevention:**
- **Launch landing + model catalog first** — validate that people find and visit the site
- **Add playground second** — the "try it" moment is what converts visitors
- **Blog + SEO third** — drives traffic
- **SDK/API/leaderboard after traffic exists** — build what's demanded

**Phase:** Phase ordering in roadmap (landing first, SDK later)

## 8. Domain Authority Cold Start

**Risk: LOW-MEDIUM**

saken.tukenov.kz is a subdomain of a personal domain. Google treats subdomains as separate sites. Zero domain authority initially.

**Prevention:**
- Link from GitHub repo README, HuggingFace model cards, social media profiles
- Submit sitemap to Search Console immediately
- Blog content targets low-competition long-tail queries ("казахский тілде NLP модель", "kazakh language model tutorial")
- Cross-link from existing properties (adapto.kz, HF repos)

**Phase:** SEO setup (early)
