# ForgeSight UI Modernization & Hugging Face Spaces Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform ForgeSight's web interface into a clean, executive-grade, dark-themed workbench using Tailwind CSS and shadcn-style component primitives, eliminate all verbose subtitles and academic clutter, and deploy the full stack to a live Hugging Face Space for public access.

**Architecture:** The Vite/React 19 frontend in `web/` will be updated with Tailwind CSS and modular shadcn-style primitives with Lucide icons. The backend FastAPI service with dual workers (PyTorch & ONNX Runtime) and ledger reaper will be packaged into a standalone container managed by a multi-process entrypoint script and deployed to Hugging Face Spaces (`piika919/forgesight`) with pre-seeded demo data on SQLite.

**Tech Stack:** React 19, Vite 6, Tailwind CSS, Lucide React, clsx, tailwind-merge, FastAPI, SQLite, PyTorch, ONNX Runtime, Docker, Hugging Face Spaces.

## Global Constraints

- Retain full functionality of all existing API integrations in `web/src/api/client.ts`.
- Deep dark theme palette using Zinc/Slate (`#09090b` base, `#18181b` card, `#27272a` border).
- Completely remove verbose subtitles, banners, and explanatory academic copy.
- Zero TypeScript or lint errors during `bun run build` and `bun run typecheck`.
- Container must run on CPU in Hugging Face Spaces free tier (port 7860, SQLite ledger).

---

### Task 1: Tailwind CSS, PostCSS & Design Tokens Setup

**Files:**
- Modify: `web/package.json`
- Create: `web/tailwind.config.js`
- Create: `web/postcss.config.js`
- Create: `web/src/lib/utils.ts`
- Modify: `web/src/styles.css`

**Interfaces:**
- Produces: `cn(...inputs)` helper in `web/src/lib/utils.ts`
- Produces: Tailwind utility classes and CSS variables for dark theme

- [ ] **Step 1: Install frontend dependencies**

Run:
```bash
cd /Users/pika/Documents/Projects/forgesight/web && bun add -d tailwindcss@^3.4.17 postcss@^8.4.49 autoprefixer@^10.4.20 && bun add lucide-react clsx tailwind-merge class-variance-authority
```

- [ ] **Step 2: Create PostCSS and Tailwind configurations**

Create `web/postcss.config.js`:
```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

Create `web/tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#09090b",
        foreground: "#fafafa",
        card: {
          DEFAULT: "#121215",
          foreground: "#fafafa",
        },
        popover: {
          DEFAULT: "#18181b",
          foreground: "#fafafa",
        },
        primary: {
          DEFAULT: "#38bdf8",
          foreground: "#09090b",
        },
        secondary: {
          DEFAULT: "#27272a",
          foreground: "#fafafa",
        },
        muted: {
          DEFAULT: "#27272a",
          foreground: "#a1a1aa",
        },
        accent: {
          DEFAULT: "#27272a",
          foreground: "#fafafa",
        },
        destructive: {
          DEFAULT: "#ef4444",
          foreground: "#fafafa",
        },
        border: "#27272a",
        input: "#27272a",
        ring: "#38bdf8",
      },
      borderRadius: {
        lg: "0.5rem",
        md: "0.375rem",
        sm: "0.25rem",
      },
    },
  },
  plugins: [],
};
```

- [ ] **Step 3: Create cn helper in `web/src/lib/utils.ts`**

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 4: Update `web/src/styles.css` with Tailwind directives and dark theme baseline**

Add `@tailwind base; @tailwind components; @tailwind utilities;` and preserve detection canvas color variables.

- [ ] **Step 5: Verify build**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run build`
Expected: Build succeeds with 0 errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/package.json web/bun.lock web/tailwind.config.js web/postcss.config.js web/src/lib/utils.ts web/src/styles.css && git commit -m "feat(ui): configure tailwind and shadcn design tokens"
```

---

### Task 2: Build Shadcn Primitives (Button, Card, Badge, Tabs, Slider)

**Files:**
- Create: `web/src/components/ui/button.tsx`
- Create: `web/src/components/ui/card.tsx`
- Create: `web/src/components/ui/badge.tsx`
- Create: `web/src/components/ui/tabs.tsx`

**Interfaces:**
- Produces: `<Button>`, `<Card>`, `<CardHeader>`, `<CardTitle>`, `<CardContent>`, `<Badge>`, `<Tabs>`, `<TabsList>`, `<TabsTrigger>`, `<TabsContent>`

- [ ] **Step 1: Create `button.tsx`**

Implement Button with variants (`default`, `destructive`, `outline`, `secondary`, `ghost`, `link`) and sizes (`default`, `sm`, `lg`, `icon`) using `class-variance-authority`.

- [ ] **Step 2: Create `card.tsx`**

Implement Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter using clean Tailwind utility classes.

- [ ] **Step 3: Create `badge.tsx`**

Implement Badge with variants (`default`, `secondary`, `destructive`, `outline`, `success`, `warning`, `info`).

- [ ] **Step 4: Create `tabs.tsx`**

Implement Tabs wrapper for clean tab navigation with pills and high-contrast indicators.

- [ ] **Step 5: Verify typecheck**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run typecheck`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/src/components/ui/ && git commit -m "feat(ui): add shadcn primitives (button, card, badge, tabs)"
```

---

### Task 3: Modernize Topbar, App Shell & Prune Verbose Text

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/primitives.tsx`

**Interfaces:**
- Consumes: `<Button>`, `<Badge>`, `<Tabs>` from Task 2
- Produces: Polished dark App layout with zero subtitles, zero banners, and sleek status badges

- [ ] **Step 1: Refactor `App.tsx`**
  - Remove `<span className="sub">document-layout inference workbench</span>`.
  - Remove `<div className="banner">...</div>`.
  - Remove verbose error paragraph (`The API did not answer. Start it with make api...`).
  - Add ForgeSight brand icon (`Sparkles` or `Layers` from `lucide-react`).
  - Add workspace pill, active mode badge (`Live` or `Local`), and compact `Reset` button.
  - Wire tabs with modern shadcn style.

- [ ] **Step 2: Refactor `StatusPill` and `TimingPanel` in `primitives.tsx`**
  - Upgrade `StatusPill` to use shadcn `<Badge>` with subtle indicator dots.
  - Remove instructional paragraph from `TimingPanel` (`queue-inclusive = received → persisted...`).
  - Upgrade progress bars to sleek high-contrast zinc/sky meters.

- [ ] **Step 3: Verify build and typecheck**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run build && bun run typecheck`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/src/App.tsx web/src/components/primitives.tsx && git commit -m "feat(ui): modernize topbar and eliminate text clutter"
```

---

### Task 4: Overhaul Batches View & Upload Dropzone

**Files:**
- Modify: `web/src/components/BatchesView.tsx`

**Interfaces:**
- Consumes: `<Card>`, `<Button>`, `<Badge>`, Lucide icons (`UploadCloud`, `FileText`, `CheckCircle2`, `Clock`, `XCircle`, `Trash2`)
- Produces: High-tech document batch management and drag-and-drop ingestion

- [ ] **Step 1: Redesign Upload Dropzone in `BatchesView.tsx`**
  - Replace text-dense drop area with a sleek Card featuring `UploadCloud` icon, subtle drag hover animation, and format chips (`PDF`, `PNG`, `JPEG`, `WEBP`).
  - Style candidate select dropdown with dark zinc border and subtle focus rings.

- [ ] **Step 2: Redesign Batches List & Batch Header**
  - Turn raw button list into interactive dark cards displaying status badges, page count, and timestamp.
  - Provide a clean batch actions bar (`Cancel`, `View Details`).

- [ ] **Step 3: Verify build and typecheck**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run build && bun run typecheck`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/src/components/BatchesView.tsx && git commit -m "feat(ui): redesign upload dropzone and batch inspection cards"
```

---

### Task 5: Overhaul Page Viewer (Layout Canvas & Floating Toolbar)

**Files:**
- Modify: `web/src/components/PageViewer.tsx`

**Interfaces:**
- Consumes: `<Button>`, `<Badge>`, Lucide icons (`Sliders`, `Eye`, `Check`, `ZoomIn`, `ZoomOut`)
- Produces: Polished document layout canvas with floating controls and detection class chips

- [ ] **Step 1: Replace raw slider with floating toolbar**
  - Add floating control bar with confidence threshold slider, live score badge (`Score ≥ 0.30`), and toggle chips (`Active`, `Shadow`, `Diff`).

- [ ] **Step 2: Redesign class legend**
  - Display detection class chips with class colors (`title`, `text`, `table`, `picture`, `code`) and item counts.

- [ ] **Step 3: Verify build and typecheck**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run build && bun run typecheck`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/src/components/PageViewer.tsx && git commit -m "feat(ui): modernize page viewer with floating toolbar and class chips"
```

---

### Task 6: Overhaul Releases View & System View

**Files:**
- Modify: `web/src/components/ReleasesView.tsx`
- Modify: `web/src/components/SystemView.tsx`

**Interfaces:**
- Consumes: `<Card>`, `<Badge>`, `<Button>`, Lucide icons (`Cpu`, `Activity`, `GitCommit`, `HardDrive`, `ShieldCheck`, `AlertCircle`)
- Produces: Executive dashboard cards for release channels, gates evaluation matrix, and system queue metrics

- [ ] **Step 1: Redesign `ReleasesView.tsx`**
  - Candidate cards showing model architecture (`RT-DETRv2`, `D-FINE`), runtime (`PyTorch`, `ONNX`), and provenance status.
  - Gate verdicts matrix (G1-G6) displayed with pass/fail badges.
  - Action buttons (`Evaluate`, `Promote`, `Rollback`) with confirmation dialogs.

- [ ] **Step 2: Redesign `SystemView.tsx`**
  - Replace raw HTML table with 4 metric cards: Runtime Mode & Dialect, Memory Budget & Safety Margin, Queue Depths by Pool, Host Telemetry & Git SHA.

- [ ] **Step 3: Verify build and typecheck**

Run: `cd /Users/pika/Documents/Projects/forgesight/web && bun run build && bun run typecheck`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add web/src/components/ReleasesView.tsx web/src/components/SystemView.tsx && git commit -m "feat(ui): redesign releases and system dashboard"
```

---

### Task 7: Hugging Face Spaces Packaging & Multi-Process Entrypoint

**Files:**
- Create: `deploy/hf_entrypoint.sh`
- Create: `Dockerfile` (root Dockerfile configured for HF Spaces)
- Modify: `README.md` (add HF Spaces frontmatter)

**Interfaces:**
- Produces: Self-contained CPU Docker image that launches on port 7860, seeds demo data, and runs workers + api.

- [ ] **Step 1: Create `deploy/hf_entrypoint.sh`**

```bash
#!/usr/bin/env bash
set -e

export FORGESIGHT_DATABASE_URL="sqlite:////data/forgesight.db"
export FORGESIGHT_DATA_DIR="/data"
export FORGESIGHT_MODELS_DIR="/models"
export FORGESIGHT_ARTIFACTS_DIR="/artifacts"
export FORGESIGHT_OBJECT_STORE="file"

echo "[HF Entrypoint] Running database migrations..."
python -m forgesight.db.migrate

echo "[HF Entrypoint] Checking and seeding demo dataset..."
python scripts/seed_demo.py || true

echo "[HF Entrypoint] Starting PyTorch worker pool..."
python -m forgesight.worker --pool torch &
PID_WORKER_TORCH=$!

echo "[HF Entrypoint] Starting ONNX Runtime worker pool..."
python -m forgesight.worker --pool onnxruntime &
PID_WORKER_ORT=$!

echo "[HF Entrypoint] Starting ledger reaper..."
python -m forgesight.ledger &
PID_REAPER=$!

echo "[HF Entrypoint] Starting FastAPI on port 7860..."
uvicorn forgesight.api.app:app --host 0.0.0.0 --port 7860 &
PID_API=$!

cleanup() {
    echo "[HF Entrypoint] Stopping all background processes..."
    kill -TERM $PID_API $PID_REAPER $PID_WORKER_ORT $PID_WORKER_TORCH 2>/dev/null || true
    wait
}

trap cleanup SIGINT SIGTERM
wait -n $PID_API $PID_REAPER $PID_WORKER_ORT $PID_WORKER_TORCH
```

- [ ] **Step 2: Create root `Dockerfile` targeting HF Spaces (port 7860)**
- [ ] **Step 3: Add HF Spaces YAML header to `README.md`**

```yaml
---
title: ForgeSight
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

- [ ] **Step 4: Commit**

```bash
cd /Users/pika/Documents/Projects/forgesight && git add deploy/hf_entrypoint.sh Dockerfile README.md && git commit -m "feat(deploy): add Hugging Face Spaces entrypoint and metadata"
```

---

### Task 8: Deploy to Hugging Face Spaces & Verify Live Access

**Files:**
- Repository remote: `huggingface`

- [ ] **Step 1: Create Space using `hf` CLI**

Run: `hf space create piika919/forgesight --sdk docker --private=false` (or check if space exists).

- [ ] **Step 2: Add git remote and push to Hugging Face Space**

Run:
```bash
cd /Users/pika/Documents/Projects/forgesight
git remote add huggingface https://huggingface.co/spaces/piika919/forgesight || git remote set-url huggingface https://huggingface.co/spaces/piika919/forgesight
git push -f huggingface main
```

- [ ] **Step 3: Monitor deployment and verify public URL**

Verify deployment status and open `https://huggingface.co/spaces/piika919/forgesight`.
