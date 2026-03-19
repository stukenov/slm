# Phase 1: Project Setup & Infrastructure - Research

**Researched:** 2026-03-19
**Domain:** Next.js App Router + Cloudflare Pages + i18n
**Confidence:** HIGH

## Summary

Phase 1 creates a new Next.js application deployed to Cloudflare Pages with trilingual routing (kk/ru/en) and a production layout shell (nav, footer, language switcher). The stack is Next.js 15.5.x on App Router, Tailwind CSS 4, next-intl for i18n, and @opennextjs/cloudflare as the deployment adapter. This is a new separate repository for saken.tukenov.kz.

The critical integration point is @opennextjs/cloudflare, which adapts Next.js server-side rendering to run on Cloudflare Workers. This requires specific Next.js version pinning (~15.5.10 range), wrangler configuration, and awareness of Cloudflare Workers limitations (no Node.js fs, limited runtime). next-intl provides the i18n routing middleware that prefixes all routes with locale segments.

**Primary recommendation:** Pin Next.js to 15.5.14 (latest in the ~15.5.10 compatibility range for @opennextjs/cloudflare 1.17.1), use next-intl's middleware-based routing with `[locale]` dynamic segment in the App Router layout, and deploy via wrangler with the opennextjs adapter.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New separate repo (not inside slm/)
- Repo will be the saken.tukenov.kz website project
- No placeholder pages -- build the real layout shell: navigation, footer, language switcher, page structure
- Default language is English (`/en/`), root `/` redirects to `/en/`
- Three locales: `/kk/`, `/ru/`, `/en/`
- Next.js 15+ App Router
- Tailwind CSS 4
- next-intl for i18n routing
- Cloudflare Pages via @opennextjs/cloudflare
- Domain: saken.tukenov.kz (already in Cloudflare DNS)
- Linear/Stripe-style minimalism -- generous whitespace, strong typography, no clutter

### Claude's Discretion
- Nav items and footer content (informed by the full site map from ROADMAP.md)
- Exact Tailwind theme/design tokens
- SEO meta tag content for the initial shell
- Project structure and folder layout
- CI/CD pipeline details for Cloudflare Pages

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFR-01 | Site deployed on Cloudflare Pages via @opennextjs/cloudflare | Standard Stack (@opennextjs/cloudflare + wrangler), Architecture Patterns (deployment config) |
| INFR-02 | Cloudflare DNS configured for saken.tukenov.kz | Architecture Patterns (DNS setup), Code Examples (wrangler config) |
| INFR-03 | i18n routing with next-intl (/kk/, /ru/, /en/) | Standard Stack (next-intl), Architecture Patterns (middleware + routing), Code Examples (routing setup) |
| INFR-04 | SEO fundamentals (sitemap.xml, robots.txt, meta tags, OG images) | Architecture Patterns (Next.js metadata API), Code Examples (sitemap/robots config) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| next | 15.5.14 | React framework with App Router, SSR/SSG | Latest stable in @opennextjs/cloudflare compat range (~15.5.10) |
| react / react-dom | 19.1.0 | UI library | Peer dep of Next.js 15.5.x |
| next-intl | 4.8.3 | i18n routing, message catalogs, locale switching | Best App Router i18n library, supports middleware-based routing |
| tailwindcss | 4.2.2 | Utility-first CSS framework | Locked decision, v4 uses CSS-first config |
| @opennextjs/cloudflare | 1.17.1 | Adapts Next.js for Cloudflare Workers/Pages | Required for Cloudflare Pages deployment |
| wrangler | 4.75.0 | Cloudflare CLI for local dev and deployment | Peer dep of @opennextjs/cloudflare (^4.65.0) |
| typescript | 5.x | Type safety | Peer dep of next-intl, standard for Next.js projects |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @tailwindcss/postcss | 4.2.2 | PostCSS plugin for Tailwind v4 | Required for Tailwind 4 with Next.js |
| clsx | 2.1.1 | Conditional className utility | Combining dynamic classes |
| tailwind-merge | 3.5.0 | Merge Tailwind classes without conflicts | When composing component variants |
| lucide-react | 0.577.0 | Icon library | SVG icons for nav, footer, language switcher |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| next-intl | next built-in i18n | next-intl has better App Router integration, message catalogs, and type safety |
| lucide-react | heroicons | Both fine; lucide has more icons and smaller tree-shaken bundles |
| Tailwind 4 | Tailwind 3 | v4 is locked decision; uses CSS-first config instead of tailwind.config.js |

**Installation:**
```bash
npx create-next-app@15.5.14 saken-tukenov-kz --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
cd saken-tukenov-kz
npm install next@15.5.14 next-intl@4.8.3 @opennextjs/cloudflare@1.17.1 wrangler@4.75.0
npm install lucide-react clsx tailwind-merge
```

**Note on create-next-app:** The `create-next-app@15.5.14` scaffolds with Tailwind 4 and App Router by default. Verify the generated `package.json` pins `next` to `15.5.14`.

## Architecture Patterns

### Recommended Project Structure
```
src/
  app/
    [locale]/
      layout.tsx          # Root layout with nav, footer (per-locale)
      page.tsx            # Home page (redirects or shows landing)
      models/
        page.tsx          # Placeholder route for future phase
      playground/
        page.tsx          # Placeholder route
      docs/
        page.tsx          # Placeholder route
      leaderboard/
        page.tsx          # Placeholder route
      blog/
        page.tsx          # Placeholder route
    favicon.ico
    robots.ts             # Dynamic robots.txt via Next.js Metadata API
    sitemap.ts            # Dynamic sitemap.xml via Next.js Metadata API
  components/
    layout/
      navbar.tsx          # Top navigation bar
      footer.tsx          # Site footer
      language-switcher.tsx  # kk/ru/en toggle
    ui/
      button.tsx          # Base button component
      container.tsx       # Max-width content wrapper
  i18n/
    routing.ts            # next-intl routing config (locales, defaultLocale)
    request.ts            # next-intl server request config
    navigation.ts         # createNavigation() for Link, redirect, usePathname
  messages/
    en.json               # English UI strings
    kk.json               # Kazakh UI strings
    ru.json               # Russian UI strings
  lib/
    utils.ts              # cn() helper (clsx + tailwind-merge)
middleware.ts             # next-intl middleware at src root
open-next.config.ts       # @opennextjs/cloudflare config at project root
wrangler.jsonc            # Cloudflare Workers config at project root
```

### Pattern 1: next-intl App Router Routing
**What:** Middleware-based locale detection and routing with `[locale]` dynamic segment
**When to use:** All page routes must be under `[locale]/`

The next-intl setup requires these configuration files:

**`src/i18n/routing.ts`** -- defines available locales and default:
```typescript
import { defineRouting } from "next-intl/routing";

export const routing = defineRouting({
  locales: ["en", "kk", "ru"],
  defaultLocale: "en",
  localePrefix: "always", // Always show /en/, /kk/, /ru/ in URL
});
```

**`src/i18n/request.ts`** -- server-side locale loading:
```typescript
import { getRequestConfig } from "next-intl/server";
import { routing } from "./routing";

export default getRequestConfig(async ({ requestLocale }) => {
  let locale = await requestLocale;
  if (!locale || !routing.locales.includes(locale as any)) {
    locale = routing.defaultLocale;
  }
  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
```

**`src/i18n/navigation.ts`** -- typed navigation helpers:
```typescript
import { createNavigation } from "next-intl/navigation";
import { routing } from "./routing";

export const { Link, redirect, usePathname, useRouter, getPathname } =
  createNavigation(routing);
```

**`src/middleware.ts`** -- must be at `src/` root:
```typescript
import createMiddleware from "next-intl/middleware";
import { routing } from "./i18n/routing";

export default createMiddleware(routing);

export const config = {
  matcher: ["/", "/(kk|ru|en)/:path*"],
};
```

**`next.config.ts`** -- must include next-intl plugin:
```typescript
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

/** @type {import('next').NextConfig} */
const nextConfig = {};

export default withNextIntl(nextConfig);
```

### Pattern 2: @opennextjs/cloudflare Deployment
**What:** Adapter that converts Next.js output to Cloudflare Workers-compatible format
**When to use:** Every deployment to Cloudflare Pages

**`open-next.config.ts`** at project root:
```typescript
import { defineCloudflareConfig } from "@opennextjs/cloudflare";

export default defineCloudflareConfig();
```

**`wrangler.jsonc`** at project root:
```jsonc
{
  "name": "saken-tukenov-kz",
  "main": ".open-next/worker.js",
  "compatibility_date": "2025-04-01",
  "compatibility_flags": ["nodejs_compat"],
  "assets": {
    "directory": ".open-next/assets",
    "binding": "ASSETS"
  }
}
```

**Build and deploy commands:**
```bash
# Build
npx opennextjs-cloudflare

# Local preview
npx wrangler dev

# Deploy
npx wrangler deploy
```

**`package.json` scripts:**
```json
{
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "next build",
    "preview": "opennextjs-cloudflare && wrangler dev",
    "deploy": "opennextjs-cloudflare && wrangler deploy"
  }
}
```

### Pattern 3: Tailwind CSS 4 Configuration
**What:** CSS-first configuration (no tailwind.config.js)
**When to use:** All styling

Tailwind 4 uses CSS-based configuration via `@theme` directive in the main CSS file:

**`src/app/globals.css`:**
```css
@import "tailwindcss";

@theme {
  /* Linear/Stripe minimalism: generous spacing, neutral palette */
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  /* Neutral color palette */
  --color-background: #ffffff;
  --color-foreground: #0a0a0a;
  --color-muted: #737373;
  --color-border: #e5e5e5;
  --color-accent: #2563eb;

  /* Spacing scale for generous whitespace */
  --spacing-section: 6rem;
  --spacing-content: 3rem;
}
```

**PostCSS config (`postcss.config.mjs`):**
```javascript
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
```

### Pattern 4: Next.js Metadata API for SEO
**What:** Built-in metadata generation for sitemap, robots, OG tags
**When to use:** SEO requirements (INFR-04)

**`src/app/sitemap.ts`:**
```typescript
import type { MetadataRoute } from "next";

const locales = ["en", "kk", "ru"];
const baseUrl = "https://saken.tukenov.kz";

export default function sitemap(): MetadataRoute.Sitemap {
  const routes = ["", "/models", "/playground", "/docs", "/leaderboard", "/blog"];

  return routes.flatMap((route) =>
    locales.map((locale) => ({
      url: `${baseUrl}/${locale}${route}`,
      lastModified: new Date(),
      changeFrequency: "weekly" as const,
      priority: route === "" ? 1 : 0.8,
    }))
  );
}
```

**`src/app/robots.ts`:**
```typescript
import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/" },
    sitemap: "https://saken.tukenov.kz/sitemap.xml",
  };
}
```

**`src/app/[locale]/layout.tsx`** metadata:
```typescript
import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;

  const titles: Record<string, string> = {
    en: "SozKZ - Kazakh NLP Platform",
    kk: "SozKZ - Qazaqsha NLP platforma",
    ru: "SozKZ - Kazahskaya NLP platforma",
  };

  return {
    title: {
      default: titles[locale] || titles.en,
      template: `%s | SozKZ`,
    },
    description: "The authoritative center of the Kazakh NLP ecosystem",
    openGraph: {
      type: "website",
      locale: locale,
      siteName: "SozKZ",
    },
    alternates: {
      languages: {
        en: "/en",
        kk: "/kk",
        ru: "/ru",
      },
    },
  };
}
```

### Pattern 5: cn() Utility
**What:** Merge class names with conflict resolution
**When to use:** Every component that accepts className prop

**`src/lib/utils.ts`:**
```typescript
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

### Anti-Patterns to Avoid
- **Using `tailwind.config.js` with Tailwind 4:** v4 uses CSS-first `@theme` directive. Do NOT create a JS config file.
- **Putting pages outside `[locale]/`:** All routed pages must be under `src/app/[locale]/` for i18n to work. Only `sitemap.ts`, `robots.ts`, and `favicon.ico` go directly under `src/app/`.
- **Using `next/link` directly:** Use `Link` from `src/i18n/navigation.ts` which auto-prefixes the current locale.
- **Running `next build` for Cloudflare:** Must run `opennextjs-cloudflare` (which calls next build internally) to get Cloudflare-compatible output.
- **Using Node.js APIs in server components:** Cloudflare Workers don't have `fs`, `path`, etc. Use the `nodejs_compat` compatibility flag and avoid direct fs operations.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Locale routing | Custom middleware with regex | next-intl middleware | Handles detection, redirects, cookie persistence, alternateLinks |
| Language switcher | Manual `<a>` tags with locale prefix | next-intl `usePathname` + `Link` | Preserves current path when switching locale |
| Sitemap generation | Manual XML file | Next.js `sitemap.ts` Metadata API | Auto-generates, type-safe, includes all locales |
| Class name merging | String concatenation | `cn()` with clsx + tailwind-merge | Handles conditional classes and Tailwind conflicts |
| Deployment adapter | Custom Cloudflare Worker script | @opennextjs/cloudflare | Handles SSR, static assets, routing in Workers |
| Icon system | Custom SVGs or icon font | lucide-react | Tree-shakeable, consistent, accessible by default |

**Key insight:** The next-intl + @opennextjs/cloudflare integration is the most complex part. Both use middleware (next-intl) and build adapters (opennextjs). Getting these to work together correctly is the core challenge -- don't try to implement locale routing manually.

## Common Pitfalls

### Pitfall 1: Next.js Version Incompatibility with @opennextjs/cloudflare
**What goes wrong:** `npm install` fails or build produces broken output because Next.js version isn't in the supported range.
**Why it happens:** @opennextjs/cloudflare pins exact minor version ranges (e.g., `~15.5.10` means `>=15.5.10 <15.6.0`). Installing latest Next.js (16.2.0) may break.
**How to avoid:** Pin `next` to `15.5.14` in package.json. Do NOT use `^15` or `latest`.
**Warning signs:** Build errors mentioning "unsupported Next.js version" or runtime errors on Cloudflare.

### Pitfall 2: Middleware Conflicts Between next-intl and Cloudflare
**What goes wrong:** next-intl middleware doesn't run, or locale detection fails on Cloudflare.
**Why it happens:** @opennextjs/cloudflare transforms middleware. The middleware matcher pattern must be precise.
**How to avoid:** Keep middleware matcher simple: `["/", "/(kk|ru|en)/:path*"]`. Test with `wrangler dev` locally before deploying.
**Warning signs:** Root `/` not redirecting to `/en/`, or locale prefix missing from URLs.

### Pitfall 3: Tailwind 4 PostCSS Misconfiguration
**What goes wrong:** Tailwind classes don't apply, CSS is empty.
**Why it happens:** Tailwind 4 requires `@tailwindcss/postcss` plugin (not `tailwindcss` in postcss config). The `@import "tailwindcss"` directive replaces the old `@tailwind base/components/utilities`.
**How to avoid:** Use `postcss.config.mjs` with `{"@tailwindcss/postcss": {}}`. CSS file must have `@import "tailwindcss"`.
**Warning signs:** No styles applied, browser devtools shows no Tailwind utility classes.

### Pitfall 4: `params` as Promise in Next.js 15
**What goes wrong:** Type errors or runtime errors accessing `params.locale`.
**Why it happens:** Next.js 15 changed dynamic route `params` to be a Promise. Must `await params` in server components and `generateMetadata`.
**How to avoid:** Always destructure as `const { locale } = await params;`
**Warning signs:** TypeScript errors about `params` type, or `locale` being undefined.

### Pitfall 5: `.open-next/` and `.wrangler/` Not in .gitignore
**What goes wrong:** Large build artifacts committed to git.
**Why it happens:** These directories are generated by `opennextjs-cloudflare` and `wrangler dev`.
**How to avoid:** Add `.open-next/` and `.wrangler/` to `.gitignore` immediately.
**Warning signs:** Git status showing hundreds of new files after first build.

### Pitfall 6: Missing `nodejs_compat` Flag
**What goes wrong:** Runtime errors on Cloudflare Workers for Node.js built-in modules.
**Why it happens:** Cloudflare Workers don't natively support Node.js APIs. The `nodejs_compat` flag enables a compatibility layer.
**How to avoid:** Always include `"compatibility_flags": ["nodejs_compat"]` in `wrangler.jsonc`.
**Warning signs:** Errors like "module 'crypto' not found" or "Buffer is not defined" on deployed site.

## Code Examples

### Language Switcher Component
```typescript
// src/components/layout/language-switcher.tsx
"use client";

import { useLocale } from "next-intl";
import { usePathname, useRouter } from "@/i18n/navigation";
import { routing } from "@/i18n/routing";

const localeNames: Record<string, string> = {
  en: "EN",
  kk: "QZ",
  ru: "RU",
};

export function LanguageSwitcher() {
  const locale = useLocale();
  const pathname = usePathname();
  const router = useRouter();

  function onChange(newLocale: string) {
    router.replace(pathname, { locale: newLocale });
  }

  return (
    <div className="flex gap-1">
      {routing.locales.map((loc) => (
        <button
          key={loc}
          onClick={() => onChange(loc)}
          className={cn(
            "px-2 py-1 text-sm rounded transition-colors",
            loc === locale
              ? "bg-foreground text-background"
              : "text-muted hover:text-foreground"
          )}
        >
          {localeNames[loc]}
        </button>
      ))}
    </div>
  );
}
```

### Navbar with Locale-Aware Links
```typescript
// src/components/layout/navbar.tsx
import { Link } from "@/i18n/navigation";
import { useTranslations } from "next-intl";
import { LanguageSwitcher } from "./language-switcher";

export function Navbar() {
  const t = useTranslations("nav");

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-sm">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-xl font-bold tracking-tight">
          SozKZ
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/models" className="text-sm text-muted hover:text-foreground transition-colors">
            {t("models")}
          </Link>
          <Link href="/playground" className="text-sm text-muted hover:text-foreground transition-colors">
            {t("playground")}
          </Link>
          <Link href="/docs" className="text-sm text-muted hover:text-foreground transition-colors">
            {t("docs")}
          </Link>
          <Link href="/leaderboard" className="text-sm text-muted hover:text-foreground transition-colors">
            {t("leaderboard")}
          </Link>
          <Link href="/blog" className="text-sm text-muted hover:text-foreground transition-colors">
            {t("blog")}
          </Link>
          <LanguageSwitcher />
        </div>
      </nav>
    </header>
  );
}
```

### Message Catalog Structure
```json
// messages/en.json
{
  "nav": {
    "models": "Models",
    "playground": "Playground",
    "docs": "Docs",
    "leaderboard": "Leaderboard",
    "blog": "Blog"
  },
  "footer": {
    "description": "The authoritative center of the Kazakh NLP ecosystem",
    "github": "GitHub",
    "huggingface": "HuggingFace",
    "rights": "All rights reserved"
  },
  "meta": {
    "title": "SozKZ - Kazakh NLP Platform",
    "description": "Models, tools, and research for the Kazakh language"
  }
}
```

### Layout with Locale Provider
```typescript
// src/app/[locale]/layout.tsx
import { NextIntlClientProvider, useMessages } from "next-intl";
import { notFound } from "next/navigation";
import { routing } from "@/i18n/routing";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import "../globals.css";

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;

  if (!routing.locales.includes(locale as any)) {
    notFound();
  }

  const messages = (await import(`@/messages/${locale}.json`)).default;

  return (
    <html lang={locale}>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <NextIntlClientProvider locale={locale} messages={messages}>
          <Navbar />
          <main className="mx-auto max-w-6xl px-6">{children}</main>
          <Footer />
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tailwind.config.js` | `@theme` in CSS file | Tailwind 4 (2025) | No JS config, CSS-first theming |
| `@tailwind base; @tailwind components; @tailwind utilities;` | `@import "tailwindcss";` | Tailwind 4 (2025) | Single import replaces three directives |
| `params` as plain object | `params` as Promise | Next.js 15 (2024) | Must `await params` in server components |
| `@cloudflare/next-on-pages` | `@opennextjs/cloudflare` | 2025 | OpenNext is the official Cloudflare-recommended adapter |
| next-intl `createSharedPathnamesNavigation` | next-intl `createNavigation` | next-intl 4.x (2025) | Unified API, simpler setup |
| `getStaticProps` / `getServerSideProps` | App Router Server Components | Next.js 13+ | Data fetching in component body, no separate functions |

**Deprecated/outdated:**
- `@cloudflare/next-on-pages`: Replaced by `@opennextjs/cloudflare` as the recommended adapter
- `tailwind.config.js`: Tailwind 4 uses CSS-first config via `@theme`
- `next-intl/server` `getTranslator`: Use `getTranslations` instead
- `createSharedPathnamesNavigation` / `createLocalizedPathnamesNavigation`: Replaced by unified `createNavigation`

## Open Questions

1. **Cloudflare Pages CI/CD Integration**
   - What we know: `wrangler deploy` pushes to Cloudflare. Cloudflare Pages can also auto-deploy from GitHub.
   - What's unclear: Whether to use Cloudflare Pages GitHub integration (auto-deploy on push) or GitHub Actions with `wrangler deploy`.
   - Recommendation: Use Cloudflare Pages GitHub integration for simplicity -- it auto-detects the build command and deploys. Set build command to `npx opennextjs-cloudflare` in Cloudflare dashboard.

2. **Font Loading Strategy**
   - What we know: Linear/Stripe aesthetic needs Inter or similar clean sans-serif.
   - What's unclear: Whether to use `next/font` (which may have Cloudflare Workers limitations) or self-host via CSS `@font-face`.
   - Recommendation: Try `next/font/google` first with Inter. If it fails on Cloudflare, fall back to system fonts or self-hosted woff2 in `public/fonts/`.

3. **Existing Cloudflare DNS Configuration**
   - What we know: saken.tukenov.kz DNS is already in Cloudflare.
   - What's unclear: Current DNS records and whether they point elsewhere.
   - Recommendation: Check current records before deployment. Cloudflare Pages projects get a `*.pages.dev` domain; add a CNAME record pointing `saken.tukenov.kz` to the Pages project.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Playwright (for e2e route/i18n testing) |
| Config file | `playwright.config.ts` -- Wave 0 |
| Quick run command | `npx playwright test --project=chromium` |
| Full suite command | `npx playwright test` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFR-01 | Site deployed and accessible | smoke/manual | `curl -s https://saken.tukenov.kz` | manual-only (requires deployment) |
| INFR-02 | DNS resolves correctly | smoke/manual | `dig saken.tukenov.kz` | manual-only (DNS) |
| INFR-03 | i18n routing: /kk/, /ru/, /en/ work, / redirects to /en/ | e2e | `npx playwright test tests/i18n-routing.spec.ts` | Wave 0 |
| INFR-04 | sitemap.xml, robots.txt, meta tags present | e2e | `npx playwright test tests/seo.spec.ts` | Wave 0 |

### Sampling Rate
- **Per task commit:** `npx playwright test --project=chromium` (quick, Chromium only)
- **Per wave merge:** `npx playwright test` (all browsers)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `playwright.config.ts` -- Playwright configuration
- [ ] `tests/i18n-routing.spec.ts` -- locale routing, redirect, language switcher
- [ ] `tests/seo.spec.ts` -- sitemap.xml, robots.txt, meta tags, OG tags
- [ ] Framework install: `npm install -D @playwright/test && npx playwright install chromium`

## Sources

### Primary (HIGH confidence)
- npm registry -- verified versions: next@15.5.14, next-intl@4.8.3, @opennextjs/cloudflare@1.17.1, tailwindcss@4.2.2, wrangler@4.75.0
- npm registry -- verified peer dependencies for @opennextjs/cloudflare (next ~15.5.10, wrangler ^4.65.0)
- npm registry -- verified peer dependencies for next-intl (next ^15.0.0 || ^16.0.0)

### Secondary (MEDIUM confidence)
- next-intl documentation patterns (routing.ts, request.ts, navigation.ts, middleware.ts) -- based on training data for next-intl 3.x/4.x, API names verified via peer deps
- @opennextjs/cloudflare configuration (open-next.config.ts, wrangler.jsonc) -- based on training data, core patterns stable
- Tailwind CSS 4 `@theme` directive and `@import "tailwindcss"` -- based on training data for v4 GA

### Tertiary (LOW confidence)
- `next/font` compatibility with Cloudflare Workers -- flagged as needing validation during implementation
- Exact Cloudflare Pages GitHub integration build command -- may need dashboard verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified against npm registry, peer deps confirmed compatible
- Architecture: HIGH -- patterns are well-established for this stack combination
- Pitfalls: HIGH -- version pinning and middleware conflicts are well-documented issues
- Cloudflare deployment specifics: MEDIUM -- core pattern known, edge cases may surface during implementation

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (30 days -- stable ecosystem)
