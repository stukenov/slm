# Phase 3: Model Catalog & Cards - Research

**Researched:** 2026-03-20
**Domain:** Next.js 15 dynamic routes, model data architecture, charting, syntax highlighting
**Confidence:** HIGH

## Summary

Phase 3 adds a `/models` catalog page with filterable card grid and `/models/[slug]` detail pages for each SozKZ model. The existing codebase (Next.js 15.5.14, Tailwind CSS 4, next-intl 4.8.3) at `/Users/sakentukenov/saken-tukenov-kz/` already has a `Model` interface in `src/data/models.ts` with 3 featured models, a `ModelCard` component with glass/blur styling, and the landing page linking to `/models`. The navbar already has a "Models" link. No `/models` route exists yet.

The core work is: (1) expand the `Model` type to include architecture details, benchmark data, training loss history, and code snippets, (2) create per-model JSON data files as the source of truth, (3) build the catalog page with pill filters and URL query params, (4) build dynamic `[slug]` detail pages with sections for architecture, training, benchmarks, loss chart, and code snippets, (5) add recharts for the training loss chart and shiki for syntax-highlighted code blocks.

Three new dependencies are needed: `recharts` (charting), `shiki` (syntax highlighting), and no others. The existing stack handles everything else: next-intl for i18n, lucide-react for icons, Tailwind CSS 4 for styling, Next.js App Router for dynamic routes.

**Primary recommendation:** Use static per-model JSON files in `src/data/models/` as the single source of truth. Build catalog with client-side filtering (URL query params via `useSearchParams`). Use Next.js `generateStaticParams` for model detail pages. Use recharts (client component) for the interactive loss chart and shiki (server-side highlighting at build time) for code snippets.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Static per-model files in the repo (JSON or MDX), not fetched from HuggingFace API
- Each model gets its own file -- CMS-like pattern, single source of truth
- Card grid layout (3 cols desktop, 2 tablet, 1 mobile)
- Glass/blur card style consistent with landing page (frosted glass effect)
- Each card shows: name, parameter count, type, task, HF link, "View" action
- Default sort: by size descending (600M first)
- Pill/chip filters: type (All | Base | Instruct | GEC) and size ranges (< 100M | 100-300M | 300M+)
- Filters reflected in URL query params (shareable, bookmarkable)
- Model detail page sections: Overview, Architecture table, Training details, Benchmark results, Training loss chart (interactive), Code snippet (tabbed: Python / CLI), HuggingFace download button, Related models
- Tabbed code blocks: Python (transformers) and CLI (huggingface-cli)
- API tab hidden until Phase 7
- Model-specific code using actual model HF ID
- Syntax highlighting with lightweight library
- Copy-to-clipboard button on code blocks
- HuggingFace link: prominent button with HF icon, opens in new tab
- Pre-populate model data from WHITEPAPER.md and agents.md

### Claude's Discretion
- Exact card grid spacing and responsive breakpoints
- Charting library choice (recharts, chart.js, or similar)
- Syntax highlighting library choice (shiki vs prism)
- Related models algorithm (same architecture different size, or same task)
- Exact pill/chip filter styling
- Empty state when no models match filters
- Model slug format for URLs

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | User can browse model catalog with all SozKZ models | Catalog page at `/models` with card grid, per-model JSON data files, glass/blur cards |
| MODL-02 | User can filter models by size, type, and task | Client-side pill filters with URL query params via `useSearchParams`, filter dimensions: type + size |
| MODL-03 | User can view model card with architecture, training data, metrics, download links | Dynamic route `/models/[slug]` with sections: overview, architecture table, training details, benchmarks, loss chart, related models |
| MODL-04 | User sees code snippet (pip install + 3 lines to run) on each model card | Tabbed code blocks (Python/CLI) with shiki syntax highlighting, copy-to-clipboard |
| MODL-05 | User can click through to HuggingFace for model download | Prominent HF button with icon on both catalog cards and detail pages, opens in new tab |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| next | 15.5.14 | Framework, dynamic routes | Pinned for @opennextjs/cloudflare compat |
| react | 19.1.0 | UI library | Bundled with Next.js 15 |
| tailwindcss | ^4 | Styling | CSS-first config, existing theme tokens |
| next-intl | ^4.8.3 | i18n | Already configured with 3 locales |
| clsx + tailwind-merge | ^2.1.1 / ^3.5.0 | Class merging | `cn()` utility in `src/lib/utils.ts` |
| lucide-react | ^0.577.0 | Icons | Already installed, use for HF icon, copy, external link |

### New Dependencies
| Library | Version | Purpose | Why This One |
|---------|---------|---------|--------------|
| recharts | 3.8.0 | Interactive training loss chart | React-native charting, composable, tree-shakeable, SVG-based. Works well as client component in Next.js |
| shiki | 4.0.2 | Syntax highlighting for code snippets | Server-side rendering (no client JS for highlighting), 603KB unpacked, supports all languages, theme-able. Can run at build time in Server Components |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| recharts | chart.js + react-chartjs-2 | Chart.js is canvas-based (not SSR-friendly), recharts is SVG-based React components -- better for Next.js |
| shiki | prismjs | Prism requires client-side JS and manual language imports; shiki renders at build time in Server Components with zero client JS. Shiki 4.x is the successor to Prism in the ecosystem |
| Per-model JSON files | Single models.ts array | Per-model files scale better for rich content (markdown descriptions, per-model loss data arrays), easier to maintain CMS-like |

**Installation:**
```bash
npm install recharts shiki
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  app/[locale]/
    models/
      page.tsx                     # Catalog page (Server Component shell + client filter)
      [slug]/
        page.tsx                   # Model detail page (Server Component)
  components/
    models/
      model-catalog-grid.tsx       # Client component: filterable grid
      model-catalog-card.tsx       # Single catalog card (reuses glass style)
      model-filter-pills.tsx       # Client component: pill/chip filters
      model-detail-header.tsx      # Model name, params, type badge, HF button
      model-architecture-table.tsx # Architecture specs table
      model-training-details.tsx   # Training hyperparameters table
      model-benchmark-table.tsx    # Benchmark results table
      model-loss-chart.tsx         # Client component: recharts loss chart
      model-code-snippet.tsx       # Tabbed code blocks with copy button
      model-related-cards.tsx      # Related models section
    landing/
      model-card.tsx               # Existing (landing page cards)
  data/
    models.ts                      # Existing: Model type + index (import all per-model files)
    models/
      types.ts                     # ModelData interface
      index.ts                     # Aggregates all model JSONs
      sozkz-core-llama-600m.json   # Per-model data file
      sozkz-core-llama-150m.json   # Per-model data file
      sozkz-core-llama-50m.json    # Per-model data file
      ...                          # One file per model
  messages/
    en.json                        # Extended with models.* keys
    kk.json                        # Extended with models.* keys
    ru.json                        # Extended with models.* keys
```

### Pattern 1: Extended Model Data Type
**What:** Rich per-model data schema covering all detail page sections.
**When to use:** Every model JSON file follows this schema.
```typescript
// src/data/models/types.ts
export interface ModelData {
  // Identity
  slug: string;
  name: string;            // Full HF name: "sozkz-core-llama-600m-kk-base-v1"
  displayName: string;     // Short: "Llama 600M Base"
  description: string;     // 1-2 paragraph description

  // Classification
  type: "base" | "instruct" | "gec" | "sentiment";
  task: string;            // "text-generation" | "text-classification" | etc.
  paramCount: number;      // 587000000 (numeric for filtering/sorting)
  paramLabel: string;      // "587M" (display)
  sizeCategory: "small" | "medium" | "large"; // <100M | 100-300M | 300M+

  // Architecture
  architecture: {
    type: string;          // "LlamaForCausalLM"
    layers: number;
    hiddenSize: number;
    attentionHeads: number;
    intermediateSize: number;
    vocabSize: number;
    contextLength: number;
    tiedEmbeddings: boolean;
    activation: string;    // "SwiGLU"
  };

  // Training
  training: {
    dataset: string;
    datasetUrl: string;
    tokensCount: string;   // "9B tokens"
    hardware: string;      // "8x H100 80GB"
    epochs: number;
    steps: number;
    learningRate: string;
    batchSize: string;
    precision: string;     // "bf16"
    trainingTime: string;
    optimizer: string;
  };

  // Benchmarks
  benchmarks: {
    metric: string;
    value: number | string;
    description?: string;
  }[];

  // Training loss curve data
  lossHistory: {
    step: number;
    loss: number;
    perplexity?: number;
  }[];

  // Code snippets (model-specific)
  codeSnippets: {
    python: string;
    cli: string;
  };

  // Links
  huggingfaceUrl: string;
  huggingfaceId: string;  // "stukenov/sozkz-core-llama-600m-kk-base-v1"
  gated: boolean;         // 300M+ models require approval

  // Relationships
  relatedSlugs: string[];

  // Metadata
  publishDate: string;    // ISO date
  experiment: string;     // "exp023"
}
```

### Pattern 2: Per-Model JSON Files with Index
**What:** Each model has its own JSON file. An index module imports and exports them all.
**When to use:** Data layer for both catalog and detail pages.
```typescript
// src/data/models/index.ts
import type { ModelData } from "./types";

import llama600m from "./sozkz-core-llama-600m.json";
import llama150m from "./sozkz-core-llama-150m.json";
import llama50m from "./sozkz-core-llama-50m.json";
// ... all models

export const allModels: ModelData[] = [
  llama600m,
  llama150m,
  llama50m,
] as ModelData[];

// Sorted by param count descending (default)
export const modelsBySize = [...allModels].sort(
  (a, b) => b.paramCount - a.paramCount
);

export function getModelBySlug(slug: string): ModelData | undefined {
  return allModels.find((m) => m.slug === slug);
}

export function getRelatedModels(model: ModelData): ModelData[] {
  return model.relatedSlugs
    .map((slug) => getModelBySlug(slug))
    .filter(Boolean) as ModelData[];
}
```

### Pattern 3: Static Generation with generateStaticParams
**What:** Pre-render all model detail pages at build time using Next.js static generation.
**When to use:** `/models/[slug]` route.
```typescript
// src/app/[locale]/models/[slug]/page.tsx
import { allModels, getModelBySlug } from "@/data/models";
import { notFound } from "next/navigation";
import { routing } from "@/i18n/routing";

export function generateStaticParams() {
  return allModels.flatMap((model) =>
    routing.locales.map((locale) => ({
      locale,
      slug: model.slug,
    }))
  );
}

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ locale: string; slug: string }>;
}) {
  const { slug } = await params;
  const model = getModelBySlug(slug);
  if (!model) notFound();

  return (
    <div className="mx-auto max-w-4xl px-6 py-section">
      <ModelDetailHeader model={model} />
      <ModelArchitectureTable model={model} />
      <ModelTrainingDetails model={model} />
      <ModelBenchmarkTable model={model} />
      <ModelLossChart model={model} />
      <ModelCodeSnippet model={model} />
      <ModelRelatedCards model={model} />
    </div>
  );
}
```

### Pattern 4: Client-Side Filtering with URL Query Params
**What:** Filters update URL search params for shareability. Client component reads params and filters the model list.
**When to use:** Catalog page filter interaction.
```typescript
// src/components/models/model-catalog-grid.tsx
"use client";

import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useMemo } from "react";
import type { ModelData } from "@/data/models/types";

export function ModelCatalogGrid({ models }: { models: ModelData[] }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const activeType = searchParams.get("type") ?? "all";
  const activeSize = searchParams.get("size") ?? "all";

  const filtered = useMemo(() => {
    return models.filter((m) => {
      if (activeType !== "all" && m.type !== activeType) return false;
      if (activeSize === "small" && m.paramCount >= 100_000_000) return false;
      if (activeSize === "medium" && (m.paramCount < 100_000_000 || m.paramCount >= 300_000_000)) return false;
      if (activeSize === "large" && m.paramCount < 300_000_000) return false;
      return true;
    });
  }, [models, activeType, activeSize]);

  function setFilter(key: string, value: string) {
    const params = new URLSearchParams(searchParams.toString());
    if (value === "all") params.delete(key);
    else params.set(key, value);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }

  return (
    <>
      <ModelFilterPills
        activeType={activeType}
        activeSize={activeSize}
        onTypeChange={(v) => setFilter("type", v)}
        onSizeChange={(v) => setFilter("size", v)}
      />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filtered.map((model) => (
          <ModelCatalogCard key={model.slug} model={model} />
        ))}
      </div>
      {filtered.length === 0 && <EmptyState />}
    </>
  );
}
```

### Pattern 5: Shiki Server-Side Syntax Highlighting
**What:** Highlight code at build time in Server Components. Zero client JS for code blocks.
**When to use:** Code snippets on model detail pages.
```typescript
// src/lib/highlight.ts
import { codeToHtml } from "shiki";

export async function highlightCode(code: string, lang: "python" | "bash") {
  return codeToHtml(code, {
    lang,
    theme: "github-light",  // Light mode only, matches project design
  });
}
```

Note: The highlighted HTML output from shiki is safe to render because the input is developer-controlled code from per-model JSON files, not user input. Use a sanitized rendering approach in the component.

The copy button is a separate small client component:
```typescript
// src/components/models/copy-button.tsx
"use client";
import { useState } from "react";
import { Check, Copy } from "lucide-react";

export function CopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button onClick={handleCopy} className="absolute top-3 right-3 p-1.5 rounded-md hover:bg-slate-200/50 transition-colors">
      {copied ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4 text-slate-400" />}
    </button>
  );
}
```

### Pattern 6: Recharts Loss Chart (Client Component)
**What:** Interactive training loss chart using recharts, wrapped in a client component.
**When to use:** Model detail page, training loss section.
```typescript
// src/components/models/model-loss-chart.tsx
"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { ModelData } from "@/data/models/types";

export function ModelLossChart({ lossHistory }: { lossHistory: ModelData["lossHistory"] }) {
  if (lossHistory.length === 0) return null;

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={lossHistory}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis dataKey="step" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Line type="monotone" dataKey="loss" stroke="#0EA5E9" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

### Pattern 7: Sitemap Extension for Model Pages
**What:** Extend the existing sitemap.ts to include `/models/[slug]` pages.
**When to use:** SEO -- all model pages should be indexed.
```typescript
// Update src/app/sitemap.ts
import { allModels } from "@/data/models";

const modelRoutes = allModels.map((m) => `/models/${m.slug}`);
const routes = ["", "/models", ...modelRoutes, "/playground", "/docs", "/leaderboard", "/blog"];
```

### Anti-Patterns to Avoid
- **Fetching model data from HuggingFace API at runtime:** User explicitly decided on static per-model files. No API calls.
- **Single large models.ts with all data inline:** Per-model JSON files are the decided pattern. Keeps each model's data (including loss arrays) in its own file.
- **Server-side filtering with searchParams in page.tsx:** Use client-side filtering for instant response. The full model list is small (7-10 models), no need for server-side filtering.
- **Using Prism with client-side highlighting:** Shiki can run in Server Components at build time, producing zero client JS.
- **Canvas-based charts (Chart.js):** Not SSR-friendly. Recharts is SVG-based and works as a client component in Next.js.
- **Hardcoding code snippets in components:** Put them in the per-model JSON files so each model has its own specific snippet with the correct HF ID.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Syntax highlighting | Custom regex-based highlighter | shiki 4.0.2 | Hundreds of edge cases in language grammars; shiki uses TextMate grammars (same as VS Code) |
| Interactive charts | Custom SVG/Canvas chart | recharts 3.8.0 | Tooltips, responsiveness, axes, grid -- deceptively complex to build correctly |
| Copy to clipboard | Manual `document.execCommand` | `navigator.clipboard.writeText()` | Modern API, works in all current browsers, simpler |
| URL query param management | Manual `window.location` manipulation | Next.js `useSearchParams` + `useRouter` | Handles encoding, history, SSR hydration |
| Class merging | String concatenation | `cn()` from `src/lib/utils.ts` | Already exists, handles Tailwind conflicts |

## Common Pitfalls

### Pitfall 1: Shiki Bundle Size in Client Components
**What goes wrong:** Importing shiki in a client component bundles all grammars/themes (~2MB+) into client JS.
**Why it happens:** Shiki loads grammars dynamically; in client components, the bundler includes everything.
**How to avoid:** Use shiki only in Server Components. Call `codeToHtml()` at build/render time, pass the HTML string to the page. The copy button is a separate small client component that receives the raw code string.
**Warning signs:** Large client bundle size, slow page load on model detail pages.

### Pitfall 2: useSearchParams Requires Suspense Boundary
**What goes wrong:** Using `useSearchParams()` in Next.js 15 without a Suspense boundary causes a build error or full-page client rendering.
**Why it happens:** Next.js 15 requires client components using `useSearchParams` to be wrapped in `<Suspense>`.
**How to avoid:** Wrap the `ModelCatalogGrid` client component in a `<Suspense fallback={...}>` in the catalog page.tsx (Server Component).
**Warning signs:** Build warnings about missing Suspense boundary, entire page becoming client-rendered.
```typescript
// src/app/[locale]/models/page.tsx
import { Suspense } from "react";
export default function ModelsPage() {
  return (
    <Suspense fallback={<CatalogSkeleton />}>
      <ModelCatalogGrid models={modelsBySize} />
    </Suspense>
  );
}
```

### Pitfall 3: JSON Import Types in Next.js
**What goes wrong:** Importing `.json` files without `resolveJsonModule` in tsconfig produces type errors.
**Why it happens:** TypeScript needs explicit configuration to import JSON as typed modules.
**How to avoid:** Ensure `tsconfig.json` has `"resolveJsonModule": true` (usually default in Next.js). Type the JSON imports with `as ModelData` or use a validation layer.
**Warning signs:** TypeScript errors on JSON imports, `any` types leaking through.

### Pitfall 4: Recharts SSR Hydration Mismatch
**What goes wrong:** Recharts renders differently on server vs client, causing hydration errors.
**Why it happens:** Recharts uses browser APIs for sizing (ResponsiveContainer measures DOM).
**How to avoid:** Mark the chart component with `"use client"` directive. If hydration issues persist, use dynamic import with `ssr: false`:
```typescript
const ModelLossChart = dynamic(() => import("./model-loss-chart").then(m => m.ModelLossChart), { ssr: false });
```
**Warning signs:** Console hydration mismatch warnings, chart rendering at wrong size on initial load.

### Pitfall 5: Glass Card Styling Inconsistency with Landing Page
**What goes wrong:** Catalog cards look different from landing page model cards despite intended visual consistency.
**Why it happens:** Different component, might use different classes or miss the gradient background context.
**How to avoid:** The existing `ModelCard` uses `bg-glass backdrop-blur-lg border border-glass-border rounded-xl p-6` -- reuse these exact Tailwind classes for catalog cards. Note: glass effect requires a colored/gradient background behind it to be visible. On a white catalog page, consider a subtle gradient section background or adjust to `bg-white border border-slate-200` with matching rounded-xl.
**Warning signs:** Visual inconsistency between landing page cards and catalog cards.

### Pitfall 6: generateStaticParams with next-intl Locales
**What goes wrong:** `generateStaticParams` doesn't generate all locale + slug combinations.
**Why it happens:** Need to cross-product locales with model slugs.
**How to avoid:** Return `locales.flatMap(locale => models.map(model => ({ locale, slug: model.slug })))`.
**Warning signs:** 404 errors on `/kk/models/sozkz-core-llama-600m` while `/en/` works.

### Pitfall 7: Landing Page Model Cards Need to Link to Detail Pages
**What goes wrong:** Landing page model cards are currently non-interactive divs (Phase 2 decision: "Model cards as non-interactive divs").
**Why it happens:** Phase 2 built cards as `<div>` not `<Link>` because detail routes did not exist yet.
**How to avoid:** Phase 3 must update the existing `ModelCard` component and the CTA button for "Browse Models" to make them interactive now that the routes will exist. Also enable the "View all 7 models" link.
**Warning signs:** Users cannot click landing page model cards to reach detail pages.

## Code Examples

### Per-Model JSON File Example
```json
{
  "slug": "sozkz-core-llama-600m",
  "name": "sozkz-core-llama-600m-kk-base-v1",
  "displayName": "Llama 600M Base",
  "description": "A 587M parameter Llama model trained from scratch on 9B Kazakh tokens. The largest dense model in the SozKZ family, designed for Kazakh text generation.",
  "type": "base",
  "task": "text-generation",
  "paramCount": 587000000,
  "paramLabel": "587M",
  "sizeCategory": "large",
  "architecture": {
    "type": "LlamaForCausalLM",
    "layers": 22,
    "hiddenSize": 1280,
    "attentionHeads": 20,
    "intermediateSize": 4480,
    "vocabSize": 50257,
    "contextLength": 2048,
    "tiedEmbeddings": true,
    "activation": "SwiGLU"
  },
  "training": {
    "dataset": "sozkz-corpus-tokenized-kk-llama50k-v3",
    "datasetUrl": "https://huggingface.co/datasets/stukenov/sozkz-corpus-tokenized-kk-llama50k-v3",
    "tokensCount": "9B tokens",
    "hardware": "8x H100 80GB SXM",
    "epochs": 1,
    "steps": 68672,
    "learningRate": "3e-4",
    "batchSize": "128 (16 x 8 GPUs)",
    "precision": "bf16",
    "trainingTime": "~3.3 hours",
    "optimizer": "AdamW"
  },
  "benchmarks": [
    { "metric": "Perplexity (eval)", "value": "TBD" }
  ],
  "lossHistory": [],
  "codeSnippets": {
    "python": "from transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel = AutoModelForCausalLM.from_pretrained(\"stukenov/sozkz-core-llama-600m-kk-base-v1\")\ntokenizer = AutoTokenizer.from_pretrained(\"stukenov/sozkz-core-llama-600m-kk-base-v1\")\n\ninputs = tokenizer(\"Kazakh text here\", return_tensors=\"pt\")\noutputs = model.generate(**inputs, max_new_tokens=100)\nprint(tokenizer.decode(outputs[0], skip_special_tokens=True))",
    "cli": "pip install transformers torch\nhuggingface-cli download stukenov/sozkz-core-llama-600m-kk-base-v1"
  },
  "huggingfaceUrl": "https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1",
  "huggingfaceId": "stukenov/sozkz-core-llama-600m-kk-base-v1",
  "gated": true,
  "relatedSlugs": ["sozkz-core-llama-150m", "sozkz-core-llama-50m"],
  "publishDate": "2026-03-18",
  "experiment": "exp023"
}
```

### Known Models to Pre-Populate (from WHITEPAPER.md)

| Slug | HuggingFace ID | Type | Params | Experiment | Status |
|------|----------------|------|--------|------------|--------|
| sozkz-core-llama-600m | stukenov/sozkz-core-llama-600m-kk-base-v1 | base | 587M | exp023 | Complete, gated |
| sozkz-core-llama-600m-sentiment | stukenov/sozkz-core-llama-600m-kk-sentiment-v1 | sentiment | 587M | exp025 | Complete, gated |
| sozkz-core-llama-150m | saken-tukenov/sozkz-core-llama-150m-kk-base-v1 | base | 152M | exp014 | Complete |
| sozkz-core-llama-150m-instruct | saken-tukenov/sozkz-core-llama-150m-kk-instruct-v1 | instruct | 152M | exp015 | Complete |
| sozkz-core-llama-150m-instruct-v2 | stukenov/sozkz-core-llama-150m-kk-instruct-v2 | instruct | 152M | exp016 | Complete |
| sozkz-core-llama-50m | saken-tukenov/sozkz-core-llama-50m-kk-base-v4 | base | 50.3M | exp013 | Complete |
| sozkz-core-pythia-14m | saken-tukenov/sozkz-core-pythia-14m-kk-dapt-v1 | base | 14M | exp001 | Paused |

Note: The MoE 3B model (exp017) and 500M 200K-vocab model (exp018) are not yet complete per WHITEPAPER. Include only completed/published models.

### Architecture Data Available from WHITEPAPER

**EXP-013 (50M):** 8 layers, 576 hidden, 8 heads, 1536 intermediate, 50000 vocab, 2048 ctx, SwiGLU. Final loss 3.184, eval PPL 24.2. Training: 2x RTX 4090, 36616 steps, LR 6e-4, ~12.5h. Full loss table available (500-36616 steps).

**EXP-014 (150M):** 16 layers, 768 hidden, 12 heads, 2048 intermediate, 50257 vocab, 1024 ctx, SwiGLU. Final eval PPL 19.78. Training: 2x RTX 4090, 36616 steps, LR 3e-4, ~26h. Full loss table available.

**EXP-015 (150M instruct):** SFT on base-v1, 1152 steps, LR 2e-5, 2x A5000, ~12min.

**EXP-016 (150M instruct v2):** SFT ChatML on base-v1, 714 steps, LR 2e-5, 16x RTX 4090, ~13min. Loss ~1.5.

**EXP-023 (600M):** 22 layers, 1280 hidden, 20 heads, 4480 intermediate, 50257 vocab, 2048 ctx, SwiGLU, tied embeddings. Training: 8x H100, LR 3e-4, 1 epoch. Loss data needs extraction from logs.

**EXP-025 (600M sentiment):** SFT on 600M base, 2688 steps, LR 2e-5, 4x RTX 4090, ~1.9h. 10/10 on manual tests.

### HF Account Note
Two HuggingFace accounts are used: `saken-tukenov` (older) and `stukenov` (current). Both should be supported in HF URLs.

### Gated Models Note (from CLAUDE.md)
Models 300M+ (including 300M) use gated access on HuggingFace (manual approval), but still MIT license. Display a "Gated -- requires approval" badge on these models.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Prism.js client-side highlighting | Shiki server-side rendering | Shiki 1.0 (2024) | Zero client JS for syntax highlighting |
| Chart.js with react-chartjs-2 | Recharts 3.x | Recharts 3.0 (2025) | Better SSR support, tree-shakeable, SVG-based |
| Manual filter state management | URL searchParams as source of truth | Next.js 13+ App Router | Shareable, bookmarkable filter states |
| getStaticPaths (Pages Router) | generateStaticParams (App Router) | Next.js 13+ | Simpler API, works with nested layouts |

## Open Questions

1. **Exact model count for initial catalog**
   - What we know: WHITEPAPER shows 7 published models (see table above), landing page says "7 models"
   - What's unclear: Whether to include paused experiments (exp001 Pythia-14m) or only fully completed ones
   - Recommendation: Include all published-to-HuggingFace models regardless of training status. Mark paused ones with a status indicator. User reviews model list during implementation.

2. **Training loss data availability**
   - What we know: WHITEPAPER has loss tables for exp013 (50M) and exp014 (150M). exp023 (600M) loss data needs to be extracted from logs.
   - What's unclear: Whether all models have loss history available
   - Recommendation: Include lossHistory as optional field. Show chart only when data exists. Pre-populate from WHITEPAPER tables. Missing data can be added later.

3. **Landing page card component refactoring**
   - What we know: Landing page `ModelCard` uses simplified `Model` interface. Phase 3 introduces richer `ModelData` type.
   - What's unclear: Whether to refactor landing page cards to use the new type or keep them separate.
   - Recommendation: Keep landing page `ModelCard` separate but make it link to detail pages. The landing page only needs a subset of fields. Create a new `ModelCatalogCard` for the catalog page.

4. **Shiki compatibility with Cloudflare Workers**
   - What we know: Shiki loads grammars dynamically. Cloudflare Workers have bundle size limits.
   - What's unclear: Whether shiki's dynamic grammar loading works in the Cloudflare Workers environment.
   - Recommendation: Test during implementation. If issues arise, pre-render highlighted HTML at build time using `generateStaticParams` (static pages), which avoids runtime shiki execution on the edge.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Playwright ^1.58.2 |
| Config file | `playwright.config.ts` (exists from Phase 1) |
| Quick run command | `npx playwright test --project=chromium tests/models*.spec.ts` |
| Full suite command | `npx playwright test` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-01 | Catalog page lists all models with card grid | e2e | `npx playwright test tests/models-catalog.spec.ts` | No -- Wave 0 |
| MODL-02 | Filter pills work, URL params update, results filter correctly | e2e | `npx playwright test tests/models-filter.spec.ts` | No -- Wave 0 |
| MODL-03 | Model detail page shows architecture, training, benchmarks | e2e | `npx playwright test tests/models-detail.spec.ts` | No -- Wave 0 |
| MODL-04 | Code snippet tabs work, copy button present, syntax highlighted | e2e | `npx playwright test tests/models-code.spec.ts` | No -- Wave 0 |
| MODL-05 | HuggingFace link present and opens in new tab | e2e | `npx playwright test tests/models-catalog.spec.ts` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `npx playwright test --project=chromium tests/models*.spec.ts`
- **Per wave merge:** `npx playwright test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/models-catalog.spec.ts` -- catalog page rendering, card count, card content, HF links
- [ ] `tests/models-filter.spec.ts` -- filter pills, URL params, filtered results, empty state
- [ ] `tests/models-detail.spec.ts` -- detail page sections, architecture table, training details
- [ ] `tests/models-code.spec.ts` -- code snippet tabs, syntax highlighting present, copy button

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `/Users/sakentukenov/saken-tukenov-kz/` -- all source files read directly (package.json, components, data, layout, routing, globals.css, tests, sitemap)
- WHITEPAPER.md -- model architecture, training details, benchmark data for all experiments
- Phase 1 & 2 CONTEXT.md -- established patterns, stack decisions, design direction
- npm registry -- recharts 3.8.0, shiki 4.0.2 versions verified via `npm view`

### Secondary (MEDIUM confidence)
- Next.js 15 `generateStaticParams` with nested dynamic routes -- standard App Router pattern
- Shiki 4.x `codeToHtml` server-side API -- standard usage pattern for Server Components
- Recharts 3.x with Next.js -- SVG-based, works as client component
- `useSearchParams` requiring Suspense -- Next.js 15 documented behavior

### Tertiary (LOW confidence)
- Recharts 3.x SSR behavior specifics -- may need `ssr: false` dynamic import, needs testing
- Shiki compatibility with Cloudflare Workers runtime -- needs verification during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all existing packages verified from codebase, new packages verified from npm registry
- Architecture: HIGH -- follows established Next.js App Router patterns, per-model JSON is straightforward
- Data model: HIGH -- all model details extracted from WHITEPAPER.md and experiment configs
- Pitfalls: HIGH -- identified from direct codebase analysis (Suspense boundary, glass card reuse, landing page linking)
- New dependencies (recharts/shiki): MEDIUM -- versions verified, patterns standard, but Cloudflare Worker compatibility needs testing

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable stack, no fast-moving dependencies)
