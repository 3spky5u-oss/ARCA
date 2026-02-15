# frontend/

Next.js 16 / React 19 frontend for ARCA. Dark-themed chat interface with comprehensive admin panel. Uses Tailwind CSS, Radix UI primitives, and WebSocket streaming. Built as standalone Docker image (node:20-alpine).

| File | Purpose | Key Exports |
|------|---------|-------------|
| package.json | Dependencies and scripts | -- |
| next.config.js | Standalone output, API proxy rewrites to backend:8000 | -- |
| tsconfig.json | TypeScript strict mode, `@/*` path alias | -- |
| eslint.config.mjs | Flat ESLint config (next/core-web-vitals + TypeScript) | -- |
| tailwind.config.ts | Tailwind theme with CSS variable colors, animate plugin | -- |
| postcss.config.js | PostCSS with Tailwind and autoprefixer | -- |
| Dockerfile | 3-stage build: deps, builder, runner (non-root nextjs user) | -- |
| next-env.d.ts | Next.js TypeScript declarations | -- |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| app/ | Next.js App Router pages (main chat, admin panel, 3D viz, layout, error boundary) |
| app/admin/components/ | Admin panel tab components (20+ tabs, lazy-loaded via next/dynamic) |
| components/ | Shared React components (chat panel, error display, domain branding, 3D viz) |
| lib/ | Client utilities (API client, domain context/config, utils) |
| types/ | TypeScript declarations (plotly.d.ts, react-force-graph-2d.d.ts) |
| public/ | Static assets |

## Key Pages

| Route | File | Purpose |
|-------|------|---------|
| `/` | app/page.tsx | Main chat (~3100 lines): WebSocket, auth, file upload, streaming, modes |
| `/admin` | app/admin/page.tsx | Admin panel: 8 main tabs, sub-tabs, lazy loading |
| `/viz/3d/[id]` | app/viz/3d/[id]/page.tsx | Full-page 3D Plotly viewer |

## Key Components

| File | Purpose |
|------|---------|
| components/domain-branding.tsx | Side-effect component: updates title + favicon from domain config |
| components/error-display.tsx | ErrorDisplay + InlineError with 20+ error code mappings |
| components/Visualization3D.tsx | Lazy-loading 3D viz with IntersectionObserver |
| lib/domain-context.tsx | DomainProvider context, `useDomain()` hook |
| lib/api.ts | `getApiBase()`, `ChatClient` class, API utilities |
