---
phase: 02-landing-page
plan: 01
subsystem: ui
tags: [nextjs, tailwind, i18n, next-intl, landing-page, inter-font]

requires:
  - phase: 01-project-setup
    provides: Next.js app with i18n routing, Cloudflare deployment, navbar/footer layout
provides:
  - Landing page hero section with animated gradient and floating badges
  - Model cards section with 3 frosted-glass cards (600M, 300M, 150M)
  - Typed model data layer (Model interface, featuredModels array)
  - ScrollReveal and CtaButton reusable components
  - Trilingual i18n messages for landing page (en/kk/ru)
  - Landing page theme tokens and animation keyframes
affects: [03-model-catalog, 04-playground, 02-landing-page-plan-02]

tech-stack:
  added: [next/font/google Inter with Cyrillic]
  patterns: [frosted-glass cards, animated gradient hero, scroll-reveal animations, disabled CTA with coming-soon state, mobile swipe carousel]

key-files:
  created:
    - src/data/models.ts
    - src/components/landing/hero-section.tsx
    - src/components/landing/model-card.tsx
    - src/components/landing/model-cards-section.tsx
    - src/components/landing/scroll-reveal.tsx
    - src/components/landing/cta-button.tsx
  modified:
    - src/app/globals.css
    - src/app/[locale]/layout.tsx
    - src/app/[locale]/page.tsx
    - src/messages/en.json
    - src/messages/kk.json
    - src/messages/ru.json

key-decisions:
  - "Both CTAs disabled with coming-soon since Playground and Models pages not built yet"
  - "Model cards render as div (not link) since /models/:slug routes do not exist yet"
  - "View all models link rendered as muted span with coming-soon label"

patterns-established:
  - "Landing section components: full-bleed wrapper with max-w constrained inner content"
  - "CtaButton with variant system and disabled/coming-soon state pattern"
  - "ScrollReveal wrapper for staggered entrance animations"
  - "Model data layer in src/data/models.ts with typed exports"

requirements-completed: [LAND-01, LAND-02, LAND-04, LAND-05]

duration: 3min
completed: 2026-03-20
---

# Phase 2 Plan 1: Hero + Model Cards Summary

**Landing page hero with animated teal gradient, floating model badges, and 3 frosted-glass model cards (600M/300M/150M) with full trilingual i18n (en/kk/ru)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-19T21:02:50Z
- **Completed:** 2026-03-19T21:06:04Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments
- Inter font loaded via next/font/google with Cyrillic subset, layout main tag freed for full-bleed sections
- Hero section with animated gradient background, floating model size badges (desktop), inline text (mobile), and two disabled CTA buttons
- Model cards section with 3 frosted-glass cards in responsive grid (desktop) / horizontal swipe carousel (mobile)
- Complete trilingual i18n messages for hero and models sections in Kazakh (Cyrillic), Russian, and English
- Reusable ScrollReveal and CtaButton components established as landing page primitives

## Task Commits

Each task was committed atomically:

1. **Task 1: Foundation - theme tokens, Inter font, layout fix, data layer, shared components** - `ade607d` (feat)
2. **Task 2: Hero section, model cards section, page assembly, and i18n messages** - `7026de4` (feat)

## Files Created/Modified
- `src/app/globals.css` - Landing page color tokens, animation keyframes, reduced-motion support
- `src/app/[locale]/layout.tsx` - Inter font loading with Cyrillic, removed main width constraint
- `src/data/models.ts` - Typed Model interface and featuredModels array with 3 models
- `src/components/landing/scroll-reveal.tsx` - IntersectionObserver scroll animation wrapper
- `src/components/landing/cta-button.tsx` - CTA button with 4 variants and disabled/coming-soon state
- `src/components/landing/hero-section.tsx` - Animated gradient hero with badges, headline, CTAs
- `src/components/landing/model-card.tsx` - Frosted glass model card with HuggingFace link
- `src/components/landing/model-cards-section.tsx` - 3-column grid / mobile carousel for featured models
- `src/app/[locale]/page.tsx` - Assembled landing page with HeroSection + ModelCardsSection
- `src/messages/en.json` - English landing page translations
- `src/messages/kk.json` - Kazakh (Cyrillic) landing page translations
- `src/messages/ru.json` - Russian landing page translations

## Decisions Made
- Both CTA buttons rendered as disabled with "Coming soon" label since Playground and Models pages are not yet built
- Model cards render as non-interactive divs rather than links since model detail routes do not exist yet
- "View all models" link rendered as muted text with "Coming soon" rather than a dead link

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Hero and model cards sections complete, ready for Plan 02 (Impact, Docs teaser, Bottom CTA sections)
- CtaButton and ScrollReveal components available for reuse in remaining landing page sections
- Model data layer ready for expansion when model catalog phase begins

## Self-Check: PASSED

All 9 created/modified files verified on disk. Both task commits (ade607d, 7026de4) confirmed in git log.

---
*Phase: 02-landing-page*
*Completed: 2026-03-20*
