---
phase: 01-project-setup
plan: 02
subsystem: seo, testing
tags: [next.js, playwright, sitemap, robots, e2e, i18n, seo]

# Dependency graph
requires:
  - phase: 01-project-setup/01
    provides: "Next.js app with i18n routing, layout shell, locale messages"
provides:
  - "Dynamic sitemap.xml (18 URLs: 6 routes x 3 locales)"
  - "Dynamic robots.txt with Allow: / and sitemap reference"
  - "Playwright e2e test suite (14 tests: 8 i18n routing + 6 SEO)"
affects: [01-project-setup/03, all-future-phases]

# Tech tracking
tech-stack:
  added: ["@playwright/test"]
  patterns: ["Next.js Metadata API for sitemap/robots", "Playwright e2e with dev server"]

key-files:
  created:
    - "src/app/sitemap.ts"
    - "src/app/robots.ts"
    - "playwright.config.ts"
    - "tests/i18n-routing.spec.ts"
    - "tests/seo.spec.ts"
  modified:
    - ".gitignore"

key-decisions:
  - "Used Next.js Metadata API (sitemap.ts/robots.ts) instead of static files for dynamic generation"
  - "Playwright webServer config starts npm run dev automatically for tests"

patterns-established:
  - "E2e tests in tests/ directory with Playwright chromium project"
  - "Root-level metadata files (sitemap.ts, robots.ts) outside [locale]/ directory"

requirements-completed: [INFR-04]

# Metrics
duration: 3min
completed: 2026-03-19
---

# Phase 1 Plan 2: SEO & Playwright Tests Summary

**Dynamic sitemap/robots via Next.js Metadata API and 14 Playwright e2e tests covering i18n routing and SEO fundamentals**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-19T19:16:06Z
- **Completed:** 2026-03-19T19:18:49Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- sitemap.xml auto-generates 18 URLs (6 routes x 3 locales) with correct priorities
- robots.txt allows all crawlers and references sitemap at saken.tukenov.kz
- 8 i18n routing tests: redirect, 3 locale layouts, lang attr, nav links, language switcher, footer links
- 6 SEO tests: sitemap.xml content, robots.txt content, title, og:type, og:site_name, alternates hreflang
- All 14 tests pass green

## Task Commits

Each task was committed atomically:

1. **Task 1: Create sitemap.ts and robots.ts** - `aecb396` (feat)
2. **Task 2: Set up Playwright and write e2e tests** - `0752538` (feat)
3. **Gitignore update for Playwright outputs** - `a25cac4` (chore)

## Files Created/Modified
- `src/app/sitemap.ts` - Dynamic sitemap generation for all locale+route combos
- `src/app/robots.ts` - Dynamic robots.txt with Allow: / and sitemap URL
- `playwright.config.ts` - Playwright config with chromium project and dev server
- `tests/i18n-routing.spec.ts` - 8 e2e tests for i18n routing behavior
- `tests/seo.spec.ts` - 6 e2e tests for SEO elements
- `.gitignore` - Added test-results/ and playwright-report/

## Decisions Made
- Used Next.js Metadata API (sitemap.ts/robots.ts) for dynamic generation rather than static files
- Playwright webServer config auto-starts `npm run dev` so tests are self-contained

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed User-Agent casing in robots.txt test**
- **Found during:** Task 2 (Playwright tests)
- **Issue:** Plan used `User-agent: *` but Next.js outputs `User-Agent: *` (capital A)
- **Fix:** Changed test expectation to match actual Next.js output
- **Files modified:** tests/seo.spec.ts
- **Verification:** Test passes
- **Committed in:** 0752538 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed language switcher button selector ambiguity**
- **Found during:** Task 2 (Playwright tests)
- **Issue:** `getByRole("button", { name: "EN" })` matched both the language switcher button and Next.js Dev Tools button in dev mode
- **Fix:** Added `exact: true` to all language switcher button selectors
- **Files modified:** tests/i18n-routing.spec.ts
- **Verification:** Test passes without ambiguity
- **Committed in:** 0752538 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for test correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SEO fundamentals complete, ready for Cloudflare Pages deployment (Plan 03)
- Playwright test suite available as regression guard for all future changes
- Run `npx playwright test --project=chromium` to verify everything works

---
*Phase: 01-project-setup*
*Completed: 2026-03-19*
