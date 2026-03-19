---
phase: 01-project-setup
plan: 01
subsystem: infra
tags: [nextjs, tailwind4, next-intl, cloudflare, i18n, typescript]

# Dependency graph
requires: []
provides:
  - "Next.js 15.5.14 project scaffold at ~/saken-tukenov-kz"
  - "i18n routing with en/kk/ru locales via next-intl middleware"
  - "Production layout shell: navbar, footer, language switcher"
  - "Cloudflare deployment config (wrangler, open-next)"
  - "Tailwind 4 CSS-first theme with Linear/Stripe design tokens"
affects: [landing-page, model-catalog, playground, docs, leaderboard, blog, seo, deployment]

# Tech tracking
tech-stack:
  added: [next@15.5.14, next-intl@4.8.3, "@opennextjs/cloudflare@1.17.1", wrangler@4.75.0, lucide-react, clsx, tailwind-merge, "@playwright/test"]
  patterns: [next-intl-middleware-routing, locale-dynamic-segment, css-first-tailwind4-theme, cn-utility]

key-files:
  created:
    - src/middleware.ts
    - src/i18n/routing.ts
    - src/i18n/request.ts
    - src/i18n/navigation.ts
    - src/messages/en.json
    - src/messages/kk.json
    - src/messages/ru.json
    - src/app/[locale]/layout.tsx
    - src/app/[locale]/page.tsx
    - src/components/layout/navbar.tsx
    - src/components/layout/footer.tsx
    - src/components/layout/language-switcher.tsx
    - src/lib/utils.ts
    - src/app/globals.css
    - open-next.config.ts
    - wrangler.jsonc
  modified:
    - package.json
    - next.config.ts
    - postcss.config.mjs
    - .gitignore

key-decisions:
  - "Pinned next to exact 15.5.14 (not ^) for @opennextjs/cloudflare compatibility"
  - "Used (typeof routing.locales)[number] cast instead of any to satisfy ESLint strict mode"

patterns-established:
  - "i18n routing: all pages under src/app/[locale]/, use Link from @/i18n/navigation"
  - "Client components: use 'use client' directive for components using useTranslations/useLocale"
  - "CSS theme: Tailwind 4 @theme directive in globals.css, no tailwind.config.js"
  - "cn() utility: clsx + tailwind-merge for conditional class merging"

requirements-completed: [INFR-03]

# Metrics
duration: 7min
completed: 2026-03-19
---

# Phase 1 Plan 01: Scaffold & Layout Shell Summary

**Next.js 15.5.14 with next-intl trilingual routing (en/kk/ru), Tailwind 4 theme, and production layout shell (navbar, footer, language switcher) deployed to ~/saken-tukenov-kz**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-19T19:06:01Z
- **Completed:** 2026-03-19T19:13:29Z
- **Tasks:** 2
- **Files modified:** 20

## Accomplishments
- Scaffolded new Next.js 15.5.14 project at ~/saken-tukenov-kz with all dependencies pinned
- Configured next-intl middleware routing for en/kk/ru with en as default (/ redirects to /en/)
- Built production navbar with 5 section links (Models, Playground, Docs, Leaderboard, Blog) and language switcher
- Built footer with GitHub and HuggingFace links
- Configured Cloudflare deployment (wrangler.jsonc, open-next.config.ts)
- Established Tailwind 4 CSS-first theme with Linear/Stripe design tokens

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold Next.js project with all dependencies and config files** - `b726e54` (feat)
2. **Task 2: Build production layout shell** - `b557c69` (feat)

## Files Created/Modified
- `package.json` - Dependencies pinned, scripts for dev/build/preview/deploy
- `next.config.ts` - next-intl plugin integration
- `postcss.config.mjs` - Tailwind 4 PostCSS plugin (object syntax)
- `open-next.config.ts` - @opennextjs/cloudflare adapter config
- `wrangler.jsonc` - Cloudflare Workers config with nodejs_compat
- `.gitignore` - Added .open-next/ and .wrangler/
- `src/app/globals.css` - Tailwind 4 @theme with design tokens
- `src/lib/utils.ts` - cn() utility (clsx + tailwind-merge)
- `src/i18n/routing.ts` - Locale definitions (en, kk, ru)
- `src/i18n/request.ts` - Server-side locale loading
- `src/i18n/navigation.ts` - Typed Link, usePathname, useRouter exports
- `src/middleware.ts` - next-intl locale routing middleware
- `src/messages/en.json` - English UI strings
- `src/messages/kk.json` - Kazakh UI strings
- `src/messages/ru.json` - Russian UI strings
- `src/app/[locale]/layout.tsx` - Root layout with Navbar, Footer, NextIntlClientProvider
- `src/app/[locale]/page.tsx` - Home page with translated title/subtitle
- `src/components/layout/navbar.tsx` - Sticky nav with locale-aware links
- `src/components/layout/footer.tsx` - Footer with external links
- `src/components/layout/language-switcher.tsx` - EN/QZ/RU toggle

## Decisions Made
- Pinned `next` to exact `15.5.14` (removed `^` prefix) to ensure @opennextjs/cloudflare compatibility
- Used `(typeof routing.locales)[number]` type cast instead of `any` to satisfy ESLint strict mode in request.ts and layout.tsx
- Removed `--turbopack` from build script (only in dev) since turbopack is dev-only optimization

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ESLint no-explicit-any error in request.ts**
- **Found during:** Task 1 (build verification)
- **Issue:** `locale as any` cast triggered @typescript-eslint/no-explicit-any rule, failing build
- **Fix:** Changed to `locale as (typeof routing.locales)[number]` for type-safe cast
- **Files modified:** src/i18n/request.ts
- **Verification:** npm run build succeeds
- **Committed in:** b726e54 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor type fix necessary for build to pass. No scope creep.

## Issues Encountered
- create-next-app prompted interactively for Turbopack -- resolved by passing `--turbopack` CLI flag
- Project directory had no .git (parent home dir git was picking up changes) -- initialized git in project dir

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Layout shell complete, ready for Plan 01-02 (SEO fundamentals and Playwright e2e tests)
- All three locales render correctly with navbar, footer, and language switcher
- Build succeeds, Cloudflare config in place for Plan 01-03 (deployment)

## Self-Check: PASSED

- All 16 created files verified present
- Commit b726e54 (Task 1) verified
- Commit b557c69 (Task 2) verified
- npm run build exits 0

---
*Phase: 01-project-setup*
*Completed: 2026-03-19*
