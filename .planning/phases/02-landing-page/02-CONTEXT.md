# Phase 2: Landing Page - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Marketing landing page for saken.tukenov.kz — a personal site showcasing Saken Tukenov's Kazakh NLP work. The page communicates the value proposition, highlights the SozKZ model family, shows real-world impact, and drives visitors to key sections (Playground, Models, Docs). SozKZ is the model family name, not the site name.

</domain>

<decisions>
## Implementation Decisions

### Hero Section
- Full-width gradient background with accent teal/cyan color (referencing Kazakh flag blue)
- Animated gradient, floating model size badges: 50M, 150M, 300M, 600M
- On mobile: badges replaced with inline text row "50M · 150M · 300M · 600M"
- Mixed branding: "Saken Tukenov" as the name, with subtitle like "Building the foundation for Kazakh language AI"
- Headline translates per locale (English on /en/, Kazakh on /kk/, Russian on /ru/)
- Two CTA buttons in hero: "Try Playground" (primary) + "Browse Models" (secondary)
- Light mode only — no dark mode toggle

### Model Highlight Cards
- Feature top 3 models: 600M, 300M, 150M (largest three)
- Full mini-card info: model name, parameter count, architecture, training data, HuggingFace link
- Glass/blur card style (backdrop-blur) — frosted glass effect on gradient background
- Cards link to own model detail page (/models/X) as primary action, HF icon as secondary link
- Section heading: "Models" with subtitle like "Open-source Kazakh language models, from 50M to 600M parameters"
- "View all 7 models →" link below the 3 featured cards

### Social Proof / Impact Section
- Dedicated section with heading ("Impact" or similar)
- Tilqazyna partnership shown as a case study card: "Tilqazyna uses SozKZ GEC models for Kazakh grammar correction"
- Research credentials: "26 experiments, 7 published models, trained on 9B tokens"
- No live GitHub/HF stats — static credibility numbers

### Page Flow (top to bottom)
1. Hero (gradient + badges + CTAs)
2. Models section (3 highlight cards + "View all" link)
3. Impact section (Tilqazyna case study + research stats)
4. Docs/SDK teaser section (text + link to /docs, no code snippet)
5. Bottom CTA banner (full-width, single CTA: "Ready to build with Kazakh NLP?")

### CTAs & Navigation
- Buttons linking to unbuilt pages (Playground, Docs, Models catalog) are disabled/grayed out with "Coming soon" label
- Buttons are enabled as phases deliver those pages
- Hero CTAs: "Try Playground" + "Browse Models"
- Bottom CTA: single prominent button

### Animations
- Scroll-triggered fade-in animations per section
- Animated gradient in hero background

### Content & Tone
- Warm and personal tone — first person, researcher's journey feel
- Trilingual content culturally adapted per language (not literal translations)

### Mobile
- Model cards in horizontal swipe carousel on mobile
- Hero badges become inline text row on mobile
- Sections stack vertically

### Performance
- Font: Inter via next/font (system fonts, no custom font loading)
- Model data hardcoded in JSON for now, migrate to HF API later (Phase 3)
- No external API calls at runtime for landing page

### Claude's Discretion
- Exact gradient colors and animation speed
- Section spacing and typography sizing
- Exact copy/messaging text (within the warm & personal tone)
- Scroll animation timing and easing
- Bottom CTA button text and destination
- Docs/SDK teaser section wording

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope & constraints
- `.planning/PROJECT.md` — Core value, hosting constraint (Cloudflare), design direction (Linear/Stripe minimalism), existing infrastructure
- `.planning/REQUIREMENTS.md` — LAND-01 through LAND-05 define this phase's requirements
- `.planning/ROADMAP.md` — Phase 2 success criteria and overall site structure

### Prior phase context
- `.planning/phases/01-project-setup/1-CONTEXT.md` — Phase 1 decisions: Next.js 15, Tailwind 4, next-intl, layout shell structure, nav items, footer

### Naming conventions
- `agents.md` — SozKZ naming standard for HuggingFace model references

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1 layout shell (nav, footer, language switcher) — landing page content fills this shell
- next-intl i18n routing already configured with /kk/, /ru/, /en/ locales
- Tailwind CSS 4 theme and design tokens from Phase 1

### Established Patterns
- Next.js 15 App Router patterns from Phase 1
- i18n message files structure (next-intl)
- Cloudflare Pages deployment pipeline

### Integration Points
- Landing page is the index route (/) within the existing layout shell
- Nav links to Models, Playground, Docs, Leaderboard, Blog (from Phase 1 nav)
- Language switcher in nav/footer (from Phase 1)

</code_context>

<specifics>
## Specific Ideas

- Site is saken.tukenov.kz — personal site, not a "SozKZ platform". SozKZ is the model family name
- Design aesthetic: teal/cyan gradient with frosted glass cards — modern but not dark-mode-heavy
- Tilqazyna case study is the main social proof anchor
- Model progression (50M → 600M) is a key visual story told through floating badges
- "View all models →" keeps users engaged even though catalog (Phase 3) isn't built yet

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-landing-page*
*Context gathered: 2026-03-20*
