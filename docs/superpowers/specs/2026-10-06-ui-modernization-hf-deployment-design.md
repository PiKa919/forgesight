# ForgeSight UI Modernization & Hugging Face Spaces Deployment Design

## 1. Overview & Objective
ForgeSight is a document-layout inference workbench serving layout models (`docling-layout-heron` and `docling-layout-egret-medium`) on CPU with dual runtimes (PyTorch eager and ONNX Runtime). The core backend logic, database ledger, and acceptance walkthrough are 100% complete and passing 5/5 CI jobs on GitHub Actions.

This design covers two remaining goals:
1. **Frontend Overhaul:** Transform the UI from a text-heavy, academic interface into a sleek, executive-grade, dark-themed workbench using Tailwind CSS and shadcn-style component primitives with Lucide icons. Strip all verbose banners, explanatory subtitles, and cluttered notes.
2. **Public Deployment:** Deploy ForgeSight to a public Hugging Face Space (`piika919/forgesight`) running in a Docker container with local SQLite, automated worker supervision, and initial demo data seeding so anyone on the web can use it immediately.

---

## 2. UI Modernization & Visual Design

### 2.1 Text & Subtitle Pruning
- **Header (`App.tsx`):**
  - Remove `<span className="sub">document-layout inference workbench</span>`.
  - Remove top warning banner (`synthetic demo data · CPU · results valid only...`).
  - Introduce a sleek brand identity: `ForgeSight` with a custom icon, a live status badge (`Live` / `Connected`), session ID badge, and a compact reset button.
- **Batches Tab (`BatchesView.tsx`):**
  - Dropzone: Replace text-dense drop area with a sleek drag-and-drop card containing an `UploadCloud` icon and subtle accepted badge chips (`PDF`, `PNG`, `JPEG`, `WEBP`).
  - Remove verbose explanatory paragraphs (`The API did not answer...`, `queue-inclusive = received → persisted...`).
  - Empty states: Minimalist, clean card layouts with intuitive icons.
- **Page Viewer (`PageViewer.tsx`):**
  - Floating/compact toolbar replacing raw HTML input sliders.
  - Quick toggles for `Active` (blue/indigo), `Shadow` (purple), and `Diff` (green/rose/amber) bounding boxes.
  - Interactive confidence threshold slider with live numeric badge.
  - Legend displayed as clean badge chips with count indicators for added/missing/relabelled detections.
- **Releases & System Views (`ReleasesView.tsx`, `SystemView.tsx`):**
  - Replace raw HTML tables with shadcn-style card grids, badge indicators, and Lucide icons for hardware telemetry, memory, and queue depths.

### 2.2 Design System & Tokens
- **Framework:** Tailwind CSS v3/v4 + `lucide-react` + `clsx` + `tailwind-merge` + `@radix-ui/react-slot`.
- **Palette:**
  - Background: `bg-zinc-950` (`#09090b`)
  - Elevated Cards/Panels: `bg-zinc-900/60` with `border-zinc-800` and backdrop blur
  - Text: `text-zinc-100` (primary), `text-zinc-400` (secondary), `text-zinc-500` (muted)
  - Active Model: Sky/Indigo (`#38bdf8` / `#6366f1`)
  - Shadow Candidate: Violet (`#a855f7`)
  - Detections: Emerald (`#10b981` added), Rose (`#f43f5e` missing), Amber (`#f59e0b` relabelled)

---

## 3. Hugging Face Spaces Deployment

### 3.1 Space Configuration
- **Hardware:** HF Spaces Free CPU Tier (2 vCPU, 16 GB RAM, 50 GB disk).
- **Metadata:** Frontmatter in `README.md`:
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

### 3.2 Container Multi-Process Supervisor
A dedicated entrypoint `deploy/hf_entrypoint.sh` coordinates container services:
1. Configures environment variables:
   - `FORGESIGHT_DATABASE_URL=sqlite:////data/forgesight.db`
   - `FORGESIGHT_OBJECT_STORE=file`
   - `FORGESIGHT_DATA_DIR=/data`
   - `FORGESIGHT_MODELS_DIR=/models`
   - `FORGESIGHT_ARTIFACTS_DIR=/artifacts`
2. Runs database migrations: `python -m forgesight.db.migrate`.
3. Seeds initial demo data if database is empty: `python scripts/seed_demo.py`.
4. Spawns background worker pools:
   - PyTorch worker: `python -m forgesight.worker --pool torch`
   - ONNX worker: `python -m forgesight.worker --pool onnxruntime`
   - Ledger reaper: `python -m forgesight.ledger`
5. Starts the FastAPI server on port 7860:
   - `uvicorn forgesight.api.app:app --host 0.0.0.0 --port 7860`
6. Manages graceful shutdown on `SIGTERM` / `SIGINT`.

### 3.3 Deployment Execution
- Create HF Space using `hf space create piika919/forgesight --sdk docker --private=false`.
- Configure git remote `huggingface` pointing to `https://huggingface.co/spaces/piika919/forgesight`.
- Push the repository to trigger the cloud Docker build.
- Monitor build logs via `hf` CLI until active.

---

## 4. Verification & Testing
1. **Frontend Build & Typecheck:** Run `bun run build` and `bun run typecheck` inside `web/` to guarantee zero TypeScript or CSS errors.
2. **Local Preview:** Test the UI locally via Vite dev server and ensure all interactive controls, sliders, upload zone, and canvas drawing function seamlessly with the dark theme.
3. **CI Pipeline:** Run tests via `pytest tests/unit` and verify all tests remain green.
4. **HF Space Live Check:** Verify HTTP 200 response and functional UI on `https://huggingface.co/spaces/piika919/forgesight`.
