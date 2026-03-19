# Stack Research: ML Model Platform

## Recommended Stack

### Framework: Next.js 15+ (App Router)
**Confidence: HIGH**

- SSR/SSG hybrid — marketing pages static, playground dynamic
- App Router with i18n via `next-intl` or folder-based routing (`/kk/`, `/ru/`, `/en/`)
- React 19 Server Components for SEO blog pages (zero JS shipped for static content)
- Already familiar stack (qazgramma-turbo, 1tv_design)

**Why not Astro:** Good for pure static, but playground needs React interactivity. Mixing frameworks adds complexity for one developer.

**Why not plain HTML:** SEO blog at 10 posts/day needs templating, i18n routing, and build-time generation. Manual HTML doesn't scale.

### Styling: Tailwind CSS 4
**Confidence: HIGH**

- Utility-first, minimal footprint for Linear/Stripe aesthetic
- `@tailwindcss/typography` for blog/docs prose rendering
- Custom design tokens for consistent spacing/typography

### Content/Blog: MDX + Content Layer
**Confidence: HIGH**

- Blog posts as MDX files (AI generates markdown, committed to repo or stored in /content)
- `@next/mdx` or `contentlayer2` for type-safe content loading
- Static generation at build time — 10 posts/day = ~300/month, well within SSG limits
- Alternative: headless CMS (Sanity, Keystatic) if manual editing needed later

### i18n: next-intl
**Confidence: HIGH**

- Best Next.js App Router i18n library (2025)
- Folder-based routing: `/kk/models`, `/ru/models`, `/en/models`
- Message catalogs for UI strings, MDX for long-form content per locale

### Hosting: Vercel
**Confidence: HIGH**

- Native Next.js deployment, zero config
- Edge functions for API proxy to kaznu inference server
- Free tier covers this scale easily
- Cloudflare DNS already configured → point saken.tukenov.kz to Vercel

**Why not Cloudflare Pages:** Next.js support via `@cloudflare/next-on-pages` is less mature than Vercel. Possible but adds friction.

### Playground API Proxy
**Confidence: HIGH**

- Next.js API routes (`/api/check`, `/api/generate`) proxy to kaznu:15127
- Server-side only — never expose GPU server IP to client
- Streaming support via ReadableStream for text generation

### Data: JSON/MDX files (no database)
**Confidence: MEDIUM**

- Model catalog: `data/models.json` or MDX per model
- Leaderboard: `data/benchmarks.json` — manually updated after evals
- People/companies: `data/people.json`, `data/companies.json`
- Blog: MDX files in `content/blog/{locale}/`
- No database needed for v1 — curator-driven, git-based content

**When to add a database:** If user-generated content, comments, or dynamic ratings are needed (v2+)

### SEO Content Pipeline
**Confidence: MEDIUM**

- Script: generates MDX posts via Claude/GPT API → commits to repo → triggers Vercel rebuild
- Cron job or GitHub Action: runs daily, generates 10 posts in 3 languages (30 MDX files)
- Each post: frontmatter (title, date, locale, tags, description) + body
- ISR (Incremental Static Regeneration) or full rebuild on push

## NOT Recommended

| Technology | Why Not |
|-----------|---------|
| WordPress | Overkill, slow, doesn't fit minimalist vision |
| Docusaurus | Good for docs but weak for marketing + blog + playground combo |
| Database (Postgres/Supabase) | No dynamic user data in v1 — JSON files suffice |
| Authentication | No user accounts in v1 |
| CMS (Sanity/Strapi) | Adds complexity; AI-generated MDX + git is simpler for automated content |
| React Native / mobile | Web-first, responsive design covers mobile |
