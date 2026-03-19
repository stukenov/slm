---
phase: 2
slug: landing-page
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Playwright ^1.58.2 |
| **Config file** | `playwright.config.ts` (exists from Phase 1) |
| **Quick run command** | `npx playwright test --project=chromium tests/landing-*.spec.ts` |
| **Full suite command** | `npx playwright test` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `npx playwright test --project=chromium tests/landing-*.spec.ts`
- **After every plan wave:** Run `npx playwright test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | LAND-01 | e2e | `npx playwright test tests/landing-hero.spec.ts` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 0 | LAND-02 | e2e | `npx playwright test tests/landing-models.spec.ts` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 0 | LAND-03 | e2e | `npx playwright test tests/landing-impact.spec.ts` | ❌ W0 | ⬜ pending |
| 2-01-04 | 01 | 0 | LAND-04 | e2e | `npx playwright test tests/landing-ctas.spec.ts` | ❌ W0 | ⬜ pending |
| 2-01-05 | 01 | 0 | LAND-05 | e2e | `npx playwright test tests/landing-i18n.spec.ts` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/landing-hero.spec.ts` — hero section elements verification (LAND-01)
- [ ] `tests/landing-models.spec.ts` — model cards content verification (LAND-02)
- [ ] `tests/landing-impact.spec.ts` — impact section content verification (LAND-03)
- [ ] `tests/landing-ctas.spec.ts` — CTA buttons state verification (LAND-04)
- [ ] `tests/landing-i18n.spec.ts` — trilingual content rendering (LAND-05)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual design quality | ALL | Subjective aesthetic judgment | Review screenshots against UI-SPEC mockups |
| Mobile swipe UX | LAND-02 | Touch interaction on real device | Swipe model cards on mobile viewport |
| Animation smoothness | ALL | Performance perception | Scroll through page, verify 60fps transitions |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
