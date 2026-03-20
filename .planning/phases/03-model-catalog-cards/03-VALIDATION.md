---
phase: 3
slug: model-catalog-cards
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Playwright (e2e, already configured from Phase 1) |
| **Config file** | `playwright.config.ts` |
| **Quick run command** | `npx playwright test tests/models --reporter=line` |
| **Full suite command** | `npx playwright test --reporter=html` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `npx playwright test tests/models --reporter=line`
- **After every plan wave:** Run `npx playwright test --reporter=html`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | MODL-01 | e2e | `npx playwright test tests/models/catalog.spec.ts` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | MODL-02 | e2e | `npx playwright test tests/models/filter.spec.ts` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | MODL-03 | e2e | `npx playwright test tests/models/detail.spec.ts` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | MODL-04 | e2e | `npx playwright test tests/models/code-snippet.spec.ts` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | MODL-05 | e2e | `npx playwright test tests/models/hf-link.spec.ts` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/models/catalog.spec.ts` — stubs for MODL-01 (catalog page lists models)
- [ ] `tests/models/filter.spec.ts` — stubs for MODL-02 (filter by size, type, task)
- [ ] `tests/models/detail.spec.ts` — stubs for MODL-03 (model card page)
- [ ] `tests/models/code-snippet.spec.ts` — stubs for MODL-04 (code snippet display)
- [ ] `tests/models/hf-link.spec.ts` — stubs for MODL-05 (HuggingFace link)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Glass/blur card visual style | MODL-01 | Visual design check | Verify frosted glass effect matches landing page cards |
| Interactive loss chart usability | MODL-03 | Interactive UX | Hover over chart points, verify tooltip shows values |
| Syntax highlighting renders correctly | MODL-04 | Visual rendering | Check Python code block has colored syntax |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
