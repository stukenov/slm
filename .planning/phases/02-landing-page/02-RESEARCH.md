# Phase 2: Landing Page - Research

**Researched:** 2026-03-20
**Domain:** Next.js 15 landing page with Tailwind CSS 4, i18n, scroll animations
**Confidence:** HIGH

## Summary

Phase 2 fills the existing layout shell (navbar, footer, language switcher from Phase 1) with a marketing landing page for saken.tukenov.kz. The page has five sections: hero with animated gradient and floating model badges, model highlight cards with frosted glass effect, impact/social proof section, docs teaser, and bottom CTA banner. All content is trilingual (kk/ru/en) via the existing next-intl setup.

The existing codebase at `/Users/sakentukenov/saken-tukenov-kz/` already has: Next.js 15.5.14 with App Router, Tailwind CSS 4, next-intl 4.8.3, `@/` path alias, `cn()` utility (clsx + tailwind-merge), layout shell with sticky navbar and footer, three locale message files, and Playwright for testing. The landing page is the index route at `src/app/[locale]/page.tsx` which currently has a minimal placeholder.

**Primary recommendation:** Build the landing page as pure React Server Components where possible, with a small client component for scroll-triggered animations using Intersection Observer. No animation libraries needed -- CSS animations for the gradient, CSS transitions for scroll fade-ins, and Tailwind utilities for the frosted glass cards. Use `next/font` for Inter. Model data hardcoded in a JSON file.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full-width animated gradient hero with teal/cyan accent, floating model size badges (50M, 150M, 300M, 600M)
- On mobile: badges become inline text row "50M . 150M . 300M . 600M"
- Mixed branding: "Saken Tukenov" name + subtitle "Building the foundation for Kazakh language AI"
- Hero CTAs: "Try Playground" (primary, disabled/coming soon) + "Browse Models" (secondary, disabled/coming soon)
- Model highlight cards: 600M, 300M, 150M with name, params, architecture, training data, HF link
- Glass/blur card style (backdrop-blur) on gradient background
- Cards link to /models/X (primary) and HF (secondary icon)
- "View all 7 models" link below cards
- Impact section: Tilqazyna case study card + research stats (26 experiments, 7 models, 9B tokens)
- Static credibility numbers (no live API stats)
- Page flow: Hero -> Models -> Impact -> Docs/SDK teaser -> Bottom CTA banner
- Unbuilt page links disabled with "Coming soon" label
- Scroll-triggered fade-in animations per section
- Animated gradient in hero background
- Light mode only
- Inter via next/font
- Model data hardcoded in JSON
- Warm, personal, first-person tone
- Trilingual content culturally adapted (not literal translations)
- Mobile: horizontal swipe carousel for model cards

### Claude's Discretion
- Exact gradient colors and animation speed
- Section spacing and typography sizing
- Exact copy/messaging text (within warm & personal tone)
- Scroll animation timing and easing
- Bottom CTA button text and destination
- Docs/SDK teaser section wording

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LAND-01 | User sees hero section with clear value proposition and key model numbers | Hero section with animated gradient, floating badges (50M-600M), headline + subtitle, two CTAs |
| LAND-02 | User sees highlighted models (600M, 300M, 150M) with params and task type | Three frosted-glass model cards with full info, hardcoded JSON data source |
| LAND-03 | User sees social proof section (Tilqazyna integration, partners) | Impact section with Tilqazyna case study card + research stat counters |
| LAND-04 | User can navigate to Playground, Docs, Models, GitHub via CTA buttons | CTAs throughout page; unbuilt pages get disabled state with "Coming soon" |
| LAND-05 | Landing is trilingual (kk/ru/en) with language switcher | Extend existing next-intl message files with landing page content keys |
</phase_requirements>

## Standard Stack

### Core (already installed in Phase 1)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| next | 15.5.14 | Framework | Pinned for @opennextjs/cloudflare compat |
| react | 19.1.0 | UI library | Bundled with Next.js 15 |
| tailwindcss | ^4 | Styling | Tailwind CSS 4 with CSS-first config |
| next-intl | ^4.8.3 | i18n | Already configured with 3 locales |
| clsx + tailwind-merge | ^2.1.1 / ^3.5.0 | Class merging | `cn()` utility already in `src/lib/utils.ts` |
| lucide-react | ^0.577.0 | Icons | Already installed, use for HF icon, arrow icons |

### Supporting (no new dependencies needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| next/font | built-in | Font loading | Inter font via `next/font/google` |
| Intersection Observer API | browser-native | Scroll animations | Fade-in sections on scroll, no library needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw Intersection Observer | framer-motion | Framer adds ~30KB; raw IO + CSS transitions sufficient for fade-ins |
| CSS @keyframes gradient | GSAP | Overkill for a simple gradient animation |
| CSS scroll-snap carousel | Swiper/Embla | User wants simple horizontal swipe; CSS scroll-snap is zero-dependency |

**No new packages to install.** Everything needed is already in the project.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── app/[locale]/
│   └── page.tsx                    # Landing page (imports section components)
├── components/
│   ├── landing/
│   │   ├── hero-section.tsx        # Hero with gradient + badges + CTAs
│   │   ├── model-cards-section.tsx # 3 featured model cards
│   │   ├── impact-section.tsx      # Tilqazyna + stats
│   │   ├── docs-teaser-section.tsx # Docs/SDK teaser
│   │   ├── bottom-cta-section.tsx  # Full-width CTA banner
│   │   ├── scroll-reveal.tsx       # Client component: Intersection Observer wrapper
│   │   └── model-card.tsx          # Single model card component
│   └── layout/                     # Existing: navbar, footer, language-switcher
├── data/
│   └── models.ts                   # Hardcoded model data (typed, importable)
├── messages/
│   ├── en.json                     # Extended with landing.* keys
│   ├── kk.json                     # Extended with landing.* keys
│   └── ru.json                     # Extended with landing.* keys
└── lib/
    └── utils.ts                    # Existing cn() utility
```

### Pattern 1: Section Components as Server Components
**What:** Each landing page section is a separate Server Component that receives translations via `useTranslations`.
**When to use:** All five sections. Only `scroll-reveal.tsx` needs `"use client"`.
**Example:**
```typescript
// src/components/landing/hero-section.tsx
import { useTranslations } from "next-intl";

export function HeroSection() {
  const t = useTranslations("landing.hero");
  return (
    <section className="relative overflow-hidden">
      {/* gradient background, badges, CTAs */}
      <h1>{t("headline")}</h1>
    </section>
  );
}
```

### Pattern 2: Client-Side Scroll Reveal Wrapper
**What:** A thin client component that uses Intersection Observer to add a CSS class when element enters viewport.
**When to use:** Wrap each section for fade-in-on-scroll effect.
**Example:**
```typescript
// src/components/landing/scroll-reveal.tsx
"use client";
import { useRef, useEffect, useState, type ReactNode } from "react";

export function ScrollReveal({ children, className }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setIsVisible(true); observer.unobserve(el); } },
      { threshold: 0.1 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"} ${className ?? ""}`}
    >
      {children}
    </div>
  );
}
```

### Pattern 3: CSS Animated Gradient (no JS)
**What:** Pure CSS @keyframes for the hero gradient background animation.
**When to use:** Hero section background.
**Example:**
```css
/* In globals.css or as a Tailwind @layer */
@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.animate-gradient {
  background-size: 200% 200%;
  animation: gradient-shift 8s ease infinite;
}
```

### Pattern 4: CSS Scroll-Snap Carousel (Mobile Model Cards)
**What:** Native CSS scroll-snap for horizontal card swiping on mobile.
**When to use:** Model cards section on mobile breakpoints.
**Example:**
```typescript
// Container
<div className="flex gap-4 overflow-x-auto snap-x snap-mandatory md:grid md:grid-cols-3 md:overflow-visible">
  {models.map(m => (
    <div key={m.id} className="min-w-[280px] snap-center md:min-w-0">
      <ModelCard model={m} />
    </div>
  ))}
</div>
```

### Pattern 5: Hardcoded Model Data with TypeScript Types
**What:** Model data as a typed constant array, not fetched at runtime.
**When to use:** Model cards section and anywhere model info is referenced.
**Example:**
```typescript
// src/data/models.ts
export interface Model {
  id: string;
  name: string;
  slug: string;
  params: string;
  architecture: string;
  trainingData: string;
  huggingfaceUrl: string;
  featured: boolean;
}

export const models: Model[] = [
  {
    id: "600m",
    name: "sozkz-core-llama-600m-kk-base-v1",
    slug: "sozkz-core-llama-600m",
    params: "600M",
    architecture: "Llama",
    trainingData: "9B tokens",
    huggingfaceUrl: "https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1",
    featured: true,
  },
  // ... 300M, 150M
];

export const featuredModels = models.filter(m => m.featured);
```

### Pattern 6: Disabled CTA Links
**What:** Buttons for unbuilt pages rendered as disabled with "Coming soon" tooltip/label.
**When to use:** "Try Playground", "Browse Models", model detail page links until those phases ship.
**Example:**
```typescript
function CtaButton({ href, disabled, children }: { href: string; disabled?: boolean; children: ReactNode }) {
  if (disabled) {
    return (
      <span className="inline-flex items-center gap-2 rounded-lg bg-muted/20 px-6 py-3 text-muted cursor-not-allowed">
        {children}
        <span className="text-xs">(Coming soon)</span>
      </span>
    );
  }
  return <Link href={href} className="inline-flex items-center gap-2 rounded-lg bg-accent px-6 py-3 text-white">{children}</Link>;
}
```

### Pattern 7: i18n Message Structure
**What:** Nested message keys under `landing.*` namespace.
**Example:**
```json
{
  "landing": {
    "hero": {
      "headline": "Building the foundation for Kazakh language AI",
      "subtitle": "Open-source models from 50M to 600M parameters",
      "cta_playground": "Try Playground",
      "cta_models": "Browse Models",
      "coming_soon": "Coming soon"
    },
    "models": {
      "heading": "Models",
      "subtitle": "Open-source Kazakh language models, from 50M to 600M parameters",
      "view_all": "View all 7 models",
      "params": "Parameters",
      "architecture": "Architecture",
      "training_data": "Training data"
    },
    "impact": {
      "heading": "Impact",
      "tilqazyna_title": "Tilqazyna",
      "tilqazyna_description": "Uses SozKZ GEC models for Kazakh grammar correction",
      "stat_experiments": "26 experiments",
      "stat_models": "7 published models",
      "stat_tokens": "9B tokens trained"
    },
    "docs_teaser": {
      "heading": "Documentation",
      "description": "...",
      "cta": "Read the docs"
    },
    "bottom_cta": {
      "heading": "Ready to build with Kazakh NLP?",
      "cta": "Get started"
    }
  }
}
```

### Anti-Patterns to Avoid
- **Putting all landing page code in page.tsx:** Split into section components for maintainability and parallel development.
- **Using framer-motion for simple fades:** Adds unnecessary bundle size. CSS transitions + Intersection Observer is sufficient.
- **Fetching model data at runtime:** User explicitly decided on hardcoded JSON. No API calls on the landing page.
- **Literal word-for-word translations:** User wants culturally adapted content per language. Kazakh and Russian messages should feel natural, not translated.
- **Full-width sections inside the existing max-w-6xl main container:** The hero and bottom CTA need to break out of the `max-w-6xl` constraint in the layout. Either modify the layout's `<main>` to not constrain the landing page, or use negative margins/full-bleed technique.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scroll animations | Custom scroll event listener with debounce | Intersection Observer API | Native, performant, no scroll jank |
| Horizontal carousel | Custom drag/swipe JS | CSS `scroll-snap-type: x mandatory` | Zero JS, native touch support, smooth |
| Class merging | String concatenation | `cn()` from `src/lib/utils.ts` | Already exists, handles Tailwind conflicts |
| Icon components | Custom SVGs | `lucide-react` | Already installed, tree-shakeable |
| i18n message access | Manual locale switching | `useTranslations()` from next-intl | Already configured |

## Common Pitfalls

### Pitfall 1: Layout Constraint Breaks Full-Width Sections
**What goes wrong:** The existing layout wraps `<main>` in `max-w-6xl px-6`, so hero gradient and bottom CTA cannot go full-width.
**Why it happens:** Phase 1 layout was built for content pages, not full-bleed sections.
**How to avoid:** For the landing page specifically, either: (a) remove the `max-w-6xl px-6` from `<main>` in layout and let each page handle its own width, or (b) use a full-bleed pattern where sections use negative margins and `100vw` width. Option (a) is cleaner -- make `<main>` a simple pass-through and let page components manage width.
**Warning signs:** Hero gradient has visible left/right margins on desktop.

### Pitfall 2: Backdrop-Blur Not Rendering on Some Mobile Browsers
**What goes wrong:** `backdrop-filter: blur()` may not render on older Android WebView.
**Why it happens:** Limited `backdrop-filter` support in older browsers.
**How to avoid:** Always provide a fallback semi-transparent background color (e.g., `bg-white/80 backdrop-blur-lg`). The card is still readable without blur.
**Warning signs:** Cards look transparent with no frosted effect.

### Pitfall 3: Gradient Animation Causing High GPU Usage
**What goes wrong:** Constantly animating large gradient backgrounds can cause battery drain on mobile.
**Why it happens:** CSS animations on background-position trigger compositing.
**How to avoid:** Use `will-change: background-position` sparingly, keep animation duration long (8-12s), and consider `prefers-reduced-motion` media query to disable for users who prefer it.
**Warning signs:** High paint times in DevTools performance tab.

### Pitfall 4: next/font Import Breaking Existing Layout
**What goes wrong:** Adding Inter via `next/font/google` requires modifying the root layout to apply the font class.
**Why it happens:** The current globals.css references `"Inter"` in `--font-sans` but never loads it -- it falls back to system fonts.
**How to avoid:** Import Inter in `layout.tsx` via `next/font/google`, add its `.className` to the `<body>` or `<html>` tag. Update `--font-sans` in globals.css to use the CSS variable that `next/font` generates.
**Warning signs:** Font flash (FOUT) or Inter not loading at all.

### Pitfall 5: Scroll Reveal Causing Layout Shift
**What goes wrong:** Elements starting at `opacity-0 translate-y-8` cause CLS (Cumulative Layout Shift) if not sized properly.
**Why it happens:** The translated elements occupy different visual space.
**How to avoid:** Use `translate-y` only (not height changes), and ensure the invisible state still occupies its full height. The `translate` transform doesn't affect layout flow.
**Warning signs:** CLS score flagged in Lighthouse.

### Pitfall 6: Tailwind CSS 4 Theme Variables vs. Arbitrary Values
**What goes wrong:** Using Tailwind v3 syntax like `bg-[#0ea5e9]` everywhere instead of extending the theme.
**Why it happens:** Tailwind 4 uses CSS-first config in `@theme {}` block, different from v3's JS config.
**How to avoid:** Define gradient colors in `@theme {}` block in `globals.css` (e.g., `--color-accent-teal: #0ea5e9;`), then use `bg-accent-teal` class.
**Warning signs:** Lots of arbitrary value brackets in component code.

## Code Examples

### next/font Inter Setup
```typescript
// src/app/[locale]/layout.tsx
import { Inter } from "next/font/google";

const inter = Inter({
  subsets: ["latin", "cyrillic"],  // cyrillic for Russian text
  variable: "--font-inter",
  display: "swap",
});

// In the component:
<html lang={locale} className={inter.variable}>
  <body className="min-h-screen bg-background text-foreground antialiased font-sans">
```

```css
/* globals.css -- update --font-sans to use the variable */
@theme {
  --font-sans: var(--font-inter), ui-sans-serif, system-ui, sans-serif;
}
```

### Tailwind 4 Theme Extensions for Landing Page
```css
/* globals.css additions */
@theme {
  --color-accent-teal: #0ea5e9;
  --color-accent-cyan: #06b6d4;
  --color-accent-teal-dark: #0284c7;
  --color-glass: rgba(255, 255, 255, 0.7);
  --color-glass-border: rgba(255, 255, 255, 0.3);
}
```

### Full-Bleed Section Pattern
```typescript
// Hero section breaks out of any parent constraint
<section className="relative w-screen -mx-[calc((100vw-100%)/2)] overflow-hidden">
  <div className="mx-auto max-w-6xl px-6">
    {/* Content constrained to max-w-6xl */}
  </div>
</section>
```

Alternatively (recommended): modify layout.tsx `<main>` to remove `max-w-6xl px-6` and let each page handle its own constraints.

### Prefers-Reduced-Motion
```css
@media (prefers-reduced-motion: reduce) {
  .animate-gradient {
    animation: none;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tailwind config in JS (tailwind.config.js) | CSS-first config via `@theme {}` in globals.css | Tailwind v4 (2025) | All theme customization goes in CSS, not JS |
| framer-motion for scroll animations | Intersection Observer + CSS transitions | 2024-2025 trend | Zero-dependency, smaller bundle |
| Custom carousel libraries (Swiper) | CSS scroll-snap | Widely supported since 2023 | No JS needed for basic horizontal scroll |
| next/font with pages directory | next/font with App Router layout.tsx | Next.js 13+ | Font loaded once in root layout |

**Deprecated/outdated:**
- Tailwind v3 `tailwind.config.js` approach -- this project uses v4 CSS-first config
- `@apply` overuse -- Tailwind 4 encourages utility-first with `@theme` for tokens

## Open Questions

1. **Layout `<main>` constraint modification**
   - What we know: Current layout has `<main className="mx-auto max-w-6xl px-6">`, hero needs full-width
   - What's unclear: Whether modifying this will break Phase 1's deployed pages
   - Recommendation: Modify `<main>` to remove width constraint; currently only page.tsx uses it and it's being replaced. Add width constraints per-section inside the landing page instead.

2. **Model data accuracy**
   - What we know: User mentioned 7 models total, top 3 featured (600M, 300M, 150M)
   - What's unclear: Exact model names, architectures, and training data for all 7
   - Recommendation: Use the SozKZ naming standard from agents.md for model names. Hardcode the 3 featured ones first; the remaining 4 are shown on the catalog page (Phase 3).

3. **Exact Kazakh/Russian copy**
   - What we know: Content should be culturally adapted, not literal translations
   - What's unclear: Whether user wants to review translations before deploy
   - Recommendation: Write natural-sounding translations using knowledge of Kazakh and Russian ML terminology. Flag for user review during verification.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Playwright ^1.58.2 |
| Config file | `playwright.config.ts` (exists from Phase 1) |
| Quick run command | `npx playwright test --project=chromium` |
| Full suite command | `npx playwright test` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LAND-01 | Hero section with headline, subtitle, model numbers, CTAs | e2e | `npx playwright test tests/landing-hero.spec.ts` | No -- Wave 0 |
| LAND-02 | 3 model cards with params, architecture, HF link | e2e | `npx playwright test tests/landing-models.spec.ts` | No -- Wave 0 |
| LAND-03 | Impact section with Tilqazyna mention, stat numbers | e2e | `npx playwright test tests/landing-impact.spec.ts` | No -- Wave 0 |
| LAND-04 | CTA buttons present (disabled ones show "Coming soon") | e2e | `npx playwright test tests/landing-ctas.spec.ts` | No -- Wave 0 |
| LAND-05 | Content renders in kk, ru, en locales correctly | e2e | `npx playwright test tests/landing-i18n.spec.ts` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `npx playwright test --project=chromium tests/landing-*.spec.ts`
- **Per wave merge:** `npx playwright test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/landing-hero.spec.ts` -- hero section elements verification
- [ ] `tests/landing-models.spec.ts` -- model cards content verification
- [ ] `tests/landing-impact.spec.ts` -- impact section content verification
- [ ] `tests/landing-ctas.spec.ts` -- CTA buttons state verification
- [ ] `tests/landing-i18n.spec.ts` -- trilingual content rendering

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `/Users/sakentukenov/saken-tukenov-kz/` -- all source files read directly
- Phase 1 context: `.planning/phases/01-project-setup/1-CONTEXT.md`
- Phase 2 context: `.planning/phases/02-landing-page/02-CONTEXT.md`

### Secondary (MEDIUM confidence)
- Tailwind CSS 4 `@theme` pattern -- verified from existing `globals.css` in codebase
- next-intl usage patterns -- verified from existing components in codebase
- next/font/google Inter setup -- standard Next.js 15 pattern, well-documented

### Tertiary (LOW confidence)
- Web search unavailable (rate limited) -- animation patterns based on training data knowledge
- Specific Tailwind v4 animation utilities -- need to verify exact class names during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages already installed and verified in codebase
- Architecture: HIGH -- follows established patterns from Phase 1, standard Next.js patterns
- Pitfalls: MEDIUM -- layout constraint issue identified from direct code reading; animation pitfalls from training data
- Scroll animation approach: MEDIUM -- Intersection Observer is well-established but web search was unavailable to verify latest patterns

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable stack, no fast-moving dependencies)
