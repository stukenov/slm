# Architecture Research: ML Model Platform

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  saken.tukenov.kz                │
│              (Vercel / Next.js App)              │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Marketing│ │  Docs    │ │    Blog (MDX)    │ │
│  │ Landing  │ │  Pages   │ │  10 posts/day    │ │
│  │  (SSG)   │ │  (SSG)   │ │     (SSG)        │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Model   │ │Playground│ │   Leaderboard    │ │
│  │ Catalog  │ │  (CSR)   │ │   + Ratings      │ │
│  │  (SSG)   │ │          │ │     (SSG)        │ │
│  └──────────┘ └────┬─────┘ └──────────────────┘ │
│                     │                            │
│  ┌──────────────────┴────────────────────┐       │
│  │         API Routes (Edge)             │       │
│  │  /api/check  /api/generate  /api/pos  │       │
│  └──────────────────┬────────────────────┘       │
└─────────────────────┼───────────────────────────┘
                      │ HTTPS proxy
                      ▼
            ┌─────────────────┐
            │  kaznu:15127    │
            │  (GPU Server)   │
            │  FastAPI + GEC  │
            │  + POS tagger   │
            └─────────────────┘
```

## Rendering Strategy

| Page Type | Rendering | Rationale |
|-----------|-----------|-----------|
| Landing, About | SSG | Static marketing content, max SEO |
| Model catalog | SSG | Model data changes infrequently |
| Model cards | SSG | One page per model, generated at build |
| Docs | SSG | MDX documentation pages |
| Blog posts | SSG | AI-generated MDX, rebuilt on push |
| Blog index | SSG with ISR | New posts added daily, revalidate hourly |
| Playground | CSR | Interactive, client-side state |
| Leaderboard | SSG | Benchmark data updated manually |
| People/Companies | SSG | Curator-driven, updated rarely |
| API routes | Edge | Proxy requests to kaznu server |

## Directory Structure (Proposed)

```
saken.tukenov.kz/
├── src/
│   ├── app/
│   │   ├── [locale]/              # i18n routing (kk, ru, en)
│   │   │   ├── page.tsx           # Landing
│   │   │   ├── models/
│   │   │   │   ├── page.tsx       # Catalog
│   │   │   │   └── [slug]/page.tsx # Model card
│   │   │   ├── playground/page.tsx
│   │   │   ├── docs/
│   │   │   │   ├── page.tsx       # Docs index
│   │   │   │   └── [...slug]/page.tsx # Doc pages
│   │   │   ├── blog/
│   │   │   │   ├── page.tsx       # Blog index
│   │   │   │   └── [slug]/page.tsx # Blog post
│   │   │   ├── leaderboard/page.tsx
│   │   │   ├── people/page.tsx
│   │   │   ├── companies/page.tsx
│   │   │   └── journey/page.tsx
│   │   └── api/
│   │       ├── check/route.ts     # GEC proxy
│   │       ├── generate/route.ts  # Text gen proxy
│   │       └── pos/route.ts       # POS tagger proxy
│   ├── components/
│   ├── lib/
│   └── styles/
├── content/
│   ├── models/                    # Model data (JSON or MDX)
│   ├── docs/{locale}/             # Documentation MDX
│   ├── blog/{locale}/             # Blog posts MDX
│   └── data/
│       ├── benchmarks.json        # Leaderboard data
│       ├── people.json            # People directory
│       └── companies.json         # Company directory
├── messages/                      # i18n UI strings
│   ├── kk.json
│   ├── ru.json
│   └── en.json
├── scripts/
│   └── generate-blog-posts.ts     # AI content generation
└── public/
```

## Data Flow

### Playground Request
```
User types text → Client JS → POST /api/check → Edge Route → kaznu:15127/v1/chat/completions → response → render diff
```

### Blog Generation
```
Cron/GitHub Action → scripts/generate-blog-posts.ts → Claude API → MDX files → git push → Vercel rebuild → live pages
```

### Content Updates
```
Edit JSON/MDX → git push → Vercel auto-deploy → SSG regeneration
```

## Build Order (Dependencies)

1. **Project setup** — Next.js, Tailwind, i18n, basic layout
2. **Landing page** — hero, model highlights, CTAs (no dependencies)
3. **Model catalog + cards** — needs model data JSON
4. **Playground** — needs API proxy routes, connects to kaznu
5. **Documentation** — needs content written/adapted from existing docs
6. **Blog engine** — needs MDX pipeline, can start with manual posts
7. **Leaderboard** — needs benchmark data JSON
8. **Ratings (people/companies)** — needs data JSON
9. **SEO content pipeline** — needs blog engine working first
10. **SDK + API docs** — can be parallel with site development

## Key Architectural Decisions

### i18n Approach
Use folder-based routing (`/[locale]/...`) with `next-intl`. Each blog post exists as separate MDX files per locale. UI strings in JSON message catalogs.

### Content as Code
All content lives in git (MDX, JSON). No database, no CMS. Changes go through git push → auto-deploy. This works because:
- Single curator (Saken)
- AI-generated content can be committed programmatically
- Full version history via git

### API Proxy Pattern
Never expose kaznu server directly. All inference goes through Vercel Edge routes:
- Rate limiting at edge
- CORS handled by Vercel
- GPU server IP stays private
- Can add caching at proxy layer
