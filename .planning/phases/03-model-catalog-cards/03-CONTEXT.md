# Phase 3: Model Catalog & Cards - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can browse all SozKZ models on a `/models` catalog page and see detailed info on dedicated model card pages (`/models/[slug]`). Includes filtering, code snippets, benchmark data, and HuggingFace download links. No user accounts, no model hosting — display and discovery only.

</domain>

<decisions>
## Implementation Decisions

### Model data source
- Static per-model files in the repo (JSON or MDX), not fetched from HuggingFace API
- Each model gets its own file (e.g., `models/sozkz-core-llama-600m.json`) — CMS-like pattern
- Each file is the single source of truth: name, params, architecture, type (base/instruct/GEC), task, training data size, HF link, benchmark scores, training details (epochs, LR, hardware), code snippet, description markdown
- Pre-populate model data by extracting from WHITEPAPER.md and agents.md during implementation — user reviews and adjusts

### Catalog page layout
- Card grid layout (3 cols desktop, 2 tablet, 1 mobile)
- Glass/blur card style consistent with landing page (frosted glass effect)
- Each card shows: name, parameter count, type, task, HF link, "View" action
- Default sort: by size descending (600M first)

### Filtering
- Pill/chip filters in horizontal row above the grid
- Filter dimensions: type (All | Base | Instruct | GEC) and size ranges (< 100M | 100-300M | 300M+)
- Filters reflected in URL query params (shareable, bookmarkable, SEO-indexable)

### Model detail page
- Comprehensive single-page scroll with sections:
  1. Overview (name, description, key stats)
  2. Architecture table (layers, hidden size, heads, vocab)
  3. Training details (data, tokens, hardware, epochs, learning rate)
  4. Benchmark results table (pre-populated from WHITEPAPER.md — perplexity, GEC accuracy, etc.)
  5. Training loss chart (interactive, using a JS charting library like recharts)
  6. Code snippet (tabbed: Python / CLI)
  7. HuggingFace download button
  8. Related models section (2-3 similar models at bottom)

### Code snippet & download
- Tabbed code blocks: Python (transformers) and CLI (huggingface-cli)
- API tab hidden until Phase 7 (Hosted API) is built
- Model-specific code (uses actual model HF ID in examples)
- Syntax highlighting with a lightweight library (shiki or prism)
- Copy-to-clipboard button on each code block
- HuggingFace link: prominent button with HF icon ("View on HuggingFace"), opens in new tab

### Claude's Discretion
- Exact card grid spacing and responsive breakpoints
- Charting library choice (recharts, chart.js, or similar)
- Syntax highlighting library choice (shiki vs prism)
- Related models algorithm (same architecture different size, or same task)
- Exact pill/chip filter styling
- Empty state when no models match filters
- Model slug format for URLs

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope & constraints
- `.planning/PROJECT.md` — Core value, hosting constraint (Cloudflare), design direction (Linear/Stripe minimalism)
- `.planning/REQUIREMENTS.md` — MODL-01 through MODL-05 define this phase's requirements
- `.planning/ROADMAP.md` — Phase 3 success criteria

### Prior phase context
- `.planning/phases/01-project-setup/1-CONTEXT.md` — Stack decisions: Next.js 15, Tailwind 4, next-intl, layout shell
- `.planning/phases/02-landing-page/02-CONTEXT.md` — Glass/blur card style, model highlight cards (600M/300M/150M), "View all 7 models" link, light mode only

### Model data sources
- `WHITEPAPER.md` — Experiment results, training details, benchmark scores to pre-populate model files
- `agents.md` — SozKZ naming standard for HuggingFace model references

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Glass/blur card component from landing page (Phase 2) — reuse for catalog cards
- next-intl i18n routing with /kk/, /ru/, /en/ locales
- Tailwind CSS 4 theme and design tokens
- Landing page model highlight cards — similar structure, can share card component

### Established Patterns
- Next.js 15 App Router (Phase 1)
- Light mode only, Linear/Stripe minimalism
- Cloudflare Pages deployment pipeline
- i18n message files structure

### Integration Points
- Landing page "View all 7 models" link targets `/models`
- Landing page model cards link to `/models/[slug]` (currently disabled/non-interactive)
- Nav already has "Models" link (from Phase 1 layout shell)
- Phase 4 (Playground) will link from model cards ("Try this model")
- Phase 7 (Hosted API) will add API tab to code snippets

</code_context>

<specifics>
## Specific Ideas

- Card style must match landing page glass/blur frosted effect — visual consistency
- Model progression (50M to 600M) is a visual story — size-descending sort reinforces this
- Per-model files enable rich content per model (markdown descriptions, per-model code examples)
- Interactive loss chart adds a "research credibility" feel — shows real training data
- Related models section keeps users exploring the catalog
- Pre-populating from WHITEPAPER.md ensures real data from day one, not placeholders

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-model-catalog-cards*
*Context gathered: 2026-03-20*
