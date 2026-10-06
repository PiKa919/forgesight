/**
 * Page viewer: the page image with detection boxes drawn over it.
 *
 * Two things this component is careful about:
 *
 * 1. **It draws in page pixels, not scaled pixels.** The API returns xyxy in the
 *    page's own pixel space, which may be a 1240x1754 render shown inside a
 *    600px-wide box. Every box is transformed by the same scale as the image, so
 *    a box can never drift away from the ink it describes.
 *
 * 2. **It never crops.** A detection that extends past the page edge is clipped
 *    by the canvas, not by the code, so what is drawn is what was returned.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import {
  Sliders,
  Eye,
  EyeOff,
  Check,
  ZoomIn,
  ZoomOut,
  Sparkles,
  Layers,
  PlusCircle,
  MinusCircle,
  RefreshCw,
  Tag,
  AlertCircle,
} from "lucide-react";
import { authHeaders, resolveApiUrl } from "../api/client";
import type { BoxDiff, Detection, WorkItem } from "../api/client";
import { Slider } from "./ui/slider";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

export const CLASS_COLORS: Record<string, string> = {
  title: "#6aa9ff",
  section_header: "#7fd1c4",
  text: "#9aa4b2",
  list_item: "#b3a1e0",
  table: "#e5b567",
  picture: "#4ec9a0",
  caption: "#8f9aa8",
  page_header: "#6f7a88",
  page_footer: "#6f7a88",
  footnote: "#7b8492",
  code: "#d59bf0",
  formula: "#c98a8a",
};

const ACTIVE = "#6aa9ff";
const SHADOW = "#d59bf0";
const ADDED = "#4ec9a0";
const MISSING = "#ef7a7a";
const RELABELLED = "#e5b567";

export function colorFor(label: string): string {
  return CLASS_COLORS[label] ?? "#8a94a3";
}

export interface PageViewerProps {
  item: WorkItem;
  imageUrl: string | null;
  showActive?: boolean;
  showShadow?: boolean;
  showDiff?: boolean;
  threshold: number;
  maxWidth: number;
  onThresholdChange: (v: number) => void;
  onToggleActive?: (v: boolean) => void;
  onToggleShadow?: (v: boolean) => void;
  onToggleDiff?: (v: boolean) => void;
}

export function PageViewer({
  item,
  imageUrl,
  showActive = true,
  showShadow = true,
  showDiff = true,
  threshold,
  maxWidth,
  onThresholdChange,
  onToggleActive,
  onToggleShadow,
  onToggleDiff,
}: PageViewerProps): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const [activeLocal, setActiveLocal] = useState(showActive);
  const [shadowLocal, setShadowLocal] = useState(showShadow);
  const [diffLocal, setDiffLocal] = useState(showDiff);

  useEffect(() => {
    setActiveLocal(showActive);
  }, [showActive]);

  useEffect(() => {
    setShadowLocal(showShadow);
  }, [showShadow]);

  useEffect(() => {
    setDiffLocal(showDiff);
  }, [showDiff]);

  const toggleActive = () => {
    const next = !activeLocal;
    setActiveLocal(next);
    onToggleActive?.(next);
  };

  const toggleShadow = () => {
    const next = !shadowLocal;
    setShadowLocal(next);
    onToggleShadow?.(next);
  };

  const toggleDiff = () => {
    const next = !diffLocal;
    setDiffLocal(next);
    onToggleDiff?.(next);
  };

  useEffect(() => {
    setImg(null);
    setLoadError(null);
    if (!imageUrl) return;
    // The image endpoint is authenticated, and `new Image()` cannot send an
    // Authorization header -- so the bytes are fetched with the bearer token
    // and handed to the element as an object URL. A plain <img src> would get a
    // 401 and silently render nothing.
    let revoked: string | null = null;
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(resolveApiUrl(imageUrl), { headers: authHeaders() });
        if (!res.ok) {
          setLoadError(`The page image could not be loaded (HTTP ${res.status})`);
          return;
        }
        const blob = await res.blob();
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        revoked = url;
        const el = new Image();
        el.onload = () => !cancelled && setImg(el);
        el.onerror = () =>
          !cancelled && setLoadError("The page image could not be decoded");
        el.src = url;
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [imageUrl, item.id]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const baseScale = Math.min(1, maxWidth / img.width);
    const scale = baseScale * zoom;
    const w = Math.max(1, Math.round(img.width * scale));
    const h = Math.max(1, Math.round(img.height * scale));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);

    const sx = w / img.width;
    const sy = h / img.height;

    if (diffLocal && item.diff) drawDiff(ctx, item.diff, sx, sy);
    if (activeLocal) {
      drawBoxes(
        ctx,
        item.detections.filter((d) => d.score >= threshold),
        sx,
        sy,
        ACTIVE,
        false,
      );
    }
    if (shadowLocal) {
      drawBoxes(
        ctx,
        item.shadow_detections.filter((d) => d.score >= threshold),
        sx,
        sy,
        SHADOW,
        true,
      );
    }
  }, [img, item, activeLocal, shadowLocal, diffLocal, threshold, maxWidth, zoom]);

  const activeCount = item.detections.filter((d) => d.score >= threshold).length;
  const shadowCount = item.shadow_detections.filter((d) => d.score >= threshold).length;
  const hasShadowData = item.shadow_detections.length > 0;
  const hasDiffData = Boolean(item.diff);

  const presentClasses = useMemo(() => {
    const counts = new Map<string, number>();
    const pool: Detection[] = [];
    if (activeLocal) {
      pool.push(...item.detections.filter((d) => d.score >= threshold));
    }
    if (shadowLocal) {
      pool.push(...item.shadow_detections.filter((d) => d.score >= threshold));
    }
    const source = (activeLocal || shadowLocal) ? pool : [];
    for (const d of source) {
      counts.set(d.label, (counts.get(d.label) ?? 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [item, threshold, activeLocal, shadowLocal]);

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Floating / Top Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 bg-zinc-900/90 backdrop-blur-md border border-zinc-800 rounded-xl shadow-sm">
        {/* Threshold Slider + Live Score Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-zinc-400">
            <Sliders className="size-4 text-zinc-400 shrink-0" />
            <span className="text-xs font-medium text-zinc-400 hidden sm:inline">Threshold</span>
          </div>
          <Badge
            variant="outline"
            className="font-mono text-xs px-2 py-0.5 bg-zinc-950/70 border-zinc-800 text-sky-400 font-semibold tabular-nums"
          >
            Score ≥ {threshold.toFixed(2)}
          </Badge>
          <Slider
            min={0}
            max={0.95}
            step={0.05}
            value={threshold}
            onValueChange={(val) => onThresholdChange(val)}
            className="w-24 sm:w-32 cursor-pointer"
            aria-label="Minimum detection score"
          />
        </div>

        {/* Quick Toggles & Zoom */}
        <div className="flex items-center flex-wrap gap-1.5">
          {/* Active Toggle */}
          <button
            type="button"
            onClick={toggleActive}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-all cursor-pointer select-none",
              activeLocal
                ? "border-sky-500/40 bg-sky-500/15 text-sky-300 shadow-xs ring-1 ring-sky-500/20"
                : "border-zinc-800 bg-zinc-900/60 text-zinc-500 hover:text-zinc-400 hover:border-zinc-700"
            )}
            title={activeLocal ? "Hide active detections" : "Show active detections"}
          >
            <span
              className={cn(
                "size-2 rounded-full transition-opacity",
                activeLocal ? "opacity-100" : "opacity-30"
              )}
              style={{ backgroundColor: ACTIVE }}
            />
            {activeLocal ? (
              <Eye className="size-3.5 shrink-0" />
            ) : (
              <EyeOff className="size-3.5 shrink-0" />
            )}
            <span>Active</span>
            <span className="font-mono text-[11px] opacity-80">({activeCount})</span>
          </button>

          {/* Shadow Toggle */}
          <button
            type="button"
            onClick={toggleShadow}
            disabled={!hasShadowData}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-all select-none",
              !hasShadowData
                ? "border-zinc-800/40 bg-zinc-900/20 text-zinc-600 opacity-50 cursor-not-allowed"
                : shadowLocal
                  ? "border-purple-500/40 bg-purple-500/15 text-purple-300 shadow-xs ring-1 ring-purple-500/20 cursor-pointer"
                  : "border-zinc-800 bg-zinc-900/60 text-zinc-500 hover:text-zinc-400 hover:border-zinc-700 cursor-pointer"
            )}
            title={
              !hasShadowData
                ? "No shadow detections for this item"
                : shadowLocal
                  ? "Hide shadow detections"
                  : "Show shadow detections"
            }
          >
            <span
              className={cn(
                "size-2 rounded-full transition-opacity",
                shadowLocal && hasShadowData ? "opacity-100" : "opacity-30"
              )}
              style={{ backgroundColor: SHADOW }}
            />
            {shadowLocal && hasShadowData ? (
              <Eye className="size-3.5 shrink-0" />
            ) : (
              <EyeOff className="size-3.5 shrink-0" />
            )}
            <span>Shadow</span>
            {hasShadowData ? (
              <span className="font-mono text-[11px] opacity-80">({shadowCount})</span>
            ) : null}
          </button>

          {/* Diff Toggle */}
          <button
            type="button"
            onClick={toggleDiff}
            disabled={!hasDiffData}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-all select-none",
              !hasDiffData
                ? "border-zinc-800/40 bg-zinc-900/20 text-zinc-600 opacity-50 cursor-not-allowed"
                : diffLocal
                  ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300 shadow-xs ring-1 ring-emerald-500/20 cursor-pointer"
                  : "border-zinc-800 bg-zinc-900/60 text-zinc-500 hover:text-zinc-400 hover:border-zinc-700 cursor-pointer"
            )}
            title={
              !hasDiffData
                ? "No diff available for this item"
                : diffLocal
                  ? "Hide diff overlay"
                  : "Show diff overlay"
            }
          >
            <span
              className={cn(
                "size-2 rounded-full transition-opacity",
                diffLocal && hasDiffData ? "opacity-100" : "opacity-30"
              )}
              style={{ backgroundColor: ADDED }}
            />
            {diffLocal && hasDiffData ? (
              <Eye className="size-3.5 shrink-0" />
            ) : (
              <EyeOff className="size-3.5 shrink-0" />
            )}
            <span>Diff</span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center ml-1 pl-2 border-l border-zinc-800 gap-1">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
              disabled={zoom <= 0.5}
              className="h-7 w-7 p-0 text-zinc-400 hover:text-zinc-200 cursor-pointer"
              title="Zoom out"
            >
              <ZoomOut className="size-3.5" />
            </Button>
            <button
              type="button"
              onClick={() => setZoom(1)}
              className="text-[11px] font-mono text-zinc-400 hover:text-zinc-200 px-1 py-0.5 rounded hover:bg-zinc-800 cursor-pointer"
              title="Reset zoom to 100%"
            >
              {Math.round(zoom * 100)}%
            </button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setZoom((z) => Math.min(2.5, Number((z + 0.25).toFixed(2))))}
              disabled={zoom >= 2.5}
              className="h-7 w-7 p-0 text-zinc-400 hover:text-zinc-200 cursor-pointer"
              title="Zoom in"
            >
              <ZoomIn className="size-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Class Legend & Diff Stats */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 p-2.5 bg-zinc-900/50 border border-zinc-800/80 rounded-lg text-xs">
        {/* Classes present on page */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mr-1 flex items-center gap-1.5 shrink-0">
            <Tag className="size-3 text-zinc-400" />
            <span>Classes:</span>
          </span>
          {presentClasses.length > 0 ? (
            presentClasses.map(([label, count]) => {
              const color = colorFor(label);
              return (
                <div
                  key={label}
                  className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border border-zinc-800 bg-zinc-900/80 text-zinc-300 shadow-2xs"
                >
                  <span
                    className="size-2 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span>{label.replace(/_/g, " ")}</span>
                  <span className="ml-0.5 px-1.5 py-0.2 rounded bg-zinc-800 text-[10px] font-mono text-zinc-400">
                    {count}
                  </span>
                </div>
              );
            })
          ) : (
            <span className="text-xs text-zinc-500 italic">
              No classes detected (score ≥ {threshold.toFixed(2)})
            </span>
          )}
        </div>

        {/* Diff Stats */}
        {diffLocal && item.diff ? (
          <div className="flex flex-wrap items-center gap-1.5 pt-1.5 sm:pt-0 border-t sm:border-t-0 border-zinc-800 sm:border-l sm:pl-3 shrink-0">
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border border-emerald-500/20 bg-emerald-500/10 text-emerald-300">
              <PlusCircle className="size-3 text-emerald-400 shrink-0" />
              <span>Added</span>
              <span className="ml-0.5 px-1.5 py-0.2 rounded bg-emerald-950/80 text-[10px] font-mono text-emerald-200">
                {item.diff.added.length}
              </span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border border-red-500/20 bg-red-500/10 text-red-300">
              <MinusCircle className="size-3 text-red-400 shrink-0" />
              <span>Missing</span>
              <span className="ml-0.5 px-1.5 py-0.2 rounded bg-red-950/80 text-[10px] font-mono text-red-200">
                {item.diff.missing.length}
              </span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border border-amber-500/20 bg-amber-500/10 text-amber-300">
              <RefreshCw className="size-3 text-amber-400 shrink-0" />
              <span>Relabelled</span>
              <span className="ml-0.5 px-1.5 py-0.2 rounded bg-amber-950/80 text-[10px] font-mono text-amber-200">
                {item.diff.relabelled.length}
              </span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border border-sky-500/20 bg-sky-500/10 text-sky-300">
              <Check className="size-3 text-sky-400 shrink-0" />
              <span>Agreement</span>
              <span className="font-mono text-xs font-semibold text-sky-200">
                {(item.diff.agreement * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ) : null}
      </div>

      {/* Canvas Container Frame */}
      <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3 overflow-auto flex items-center justify-center min-h-[420px] max-h-[70vh] shadow-inner relative">
        {loadError ? (
          <div className="flex flex-col items-center justify-center gap-2 py-16 px-4 text-center max-w-md">
            <div className="rounded-full bg-red-500/10 p-2.5 text-red-400 mb-1">
              <AlertCircle className="size-6" />
            </div>
            <span className="text-xs font-medium text-red-300">{loadError}</span>
          </div>
        ) : !img ? (
          <div className="flex flex-col items-center justify-center gap-3 py-20 text-zinc-500 animate-pulse">
            <div className="relative flex items-center justify-center">
              <Layers className="size-9 text-sky-500/70 animate-bounce" />
              <Sparkles className="size-4 text-sky-400 absolute -top-1 -right-1" />
            </div>
            <div className="flex flex-col items-center gap-1 text-center">
              <span className="text-xs font-medium text-zinc-300">Rendering document layout…</span>
              <span className="text-[11px] text-zinc-600 font-mono">
                Fetching page ink & detection layers
              </span>
            </div>
          </div>
        ) : null}

        <canvas
          ref={canvasRef}
          className="rounded shadow-md max-w-none m-auto shrink-0"
          style={{ display: img ? "block" : "none" }}
        />
      </div>

      {/* Metadata Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500 px-1 pt-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-zinc-400">
            {item.page_size ? `${item.page_size[0]} × ${item.page_size[1]} px` : "Unknown dimensions"}
          </span>
          <span className="text-zinc-700">•</span>
          <span>
            <strong className="text-zinc-300 font-medium">{item.detections.length}</strong> active
            {item.shadow_detections.length ? (
              <>
                {" · "}
                <strong className="text-zinc-300 font-medium">{item.shadow_detections.length}</strong> shadow
              </>
            ) : ""}
          </span>
          {item.synthetic && (
            <>
              <span className="text-zinc-700">•</span>
              <Badge
                variant="warning"
                className="text-[10px] font-mono uppercase tracking-wider py-0 px-1.5"
              >
                Synthetic
              </Badge>
            </>
          )}
        </div>
        {item.provenance && (
          <span
            className="font-mono text-[11px] text-zinc-500 truncate max-w-xs"
            title={item.provenance}
          >
            {item.provenance}
          </span>
        )}
      </div>
    </div>
  );
}

function drawBoxes(
  ctx: CanvasRenderingContext2D,
  dets: Detection[],
  sx: number,
  sy: number,
  color: string,
  dashed: boolean,
): void {
  ctx.lineWidth = 1.5;
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "top";
  if (dashed) ctx.setLineDash([4, 3]);
  else ctx.setLineDash([]);

  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box;
    const px = x1 * sx;
    const py = y1 * sy;
    const pw = (x2 - x1) * sx;
    const ph = (y2 - y1) * sy;
    // A class-coloured fill with a colour-coded stroke: the stroke answers
    // "which run produced this" and the fill answers "which class".
    ctx.strokeStyle = color;
    ctx.fillStyle = hexAlpha(colorFor(d.label), 0.14);
    ctx.fillRect(px, py, pw, ph);
    ctx.strokeRect(px, py, pw, ph);
    const label = `${d.label} ${d.score.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 4;
    // Flip the label inside the box when it would fall off the page, so it is
    // never silently invisible on a box at the right or bottom edge.
    const ty = py + ph + 10 > ctx.canvas.height ? py - 12 : py;
    ctx.fillStyle = color;
    ctx.fillRect(px, ty, tw, 12);
    ctx.fillStyle = "#0b0e14";
    ctx.fillText(label, px + 2, ty + 1.5);
  }
  ctx.setLineDash([]);
}

function drawDiff(
  ctx: CanvasRenderingContext2D,
  diff: { added: BoxDiff[]; missing: BoxDiff[]; relabelled: BoxDiff[] },
  sx: number,
  sy: number,
): void {
  const stroke = (b: BoxDiff, color: string, dash: number[]) => {
    const [x1, y1, x2, y2] = b.box;
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  };
  // Missing first: they are the finding that matters, so they draw on top.
  for (const b of diff.missing) stroke(b, MISSING, []);
  for (const b of diff.added) stroke(b, ADDED, [5, 3]);
  for (const b of diff.relabelled) stroke(b, RELABELLED, [2, 2]);
  ctx.setLineDash([]);
  ctx.lineWidth = 1.5;
}

function hexAlpha(hex: string, alpha: number): string {
  const v = hex.replace("#", "");
  const r = parseInt(v.slice(0, 2), 16);
  const g = parseInt(v.slice(2, 4), 16);
  const b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export { hexAlpha };
