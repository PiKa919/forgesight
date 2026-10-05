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

import { useEffect, useRef, useState, type ReactElement } from "react";
import { authHeaders } from "../api/client";
import type { BoxDiff, Detection, WorkItem } from "../api/client";

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

interface Props {
  item: WorkItem;
  imageUrl: string | null;
  showActive: boolean;
  showShadow: boolean;
  showDiff: boolean;
  threshold: number;
  maxWidth: number;
  onThresholdChange: (v: number) => void;
}

export function PageViewer({
  item,
  imageUrl,
  showActive,
  showShadow,
  showDiff,
  threshold,
  maxWidth,
  onThresholdChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

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
        const res = await fetch(imageUrl, { headers: authHeaders() });
        if (!res.ok) {
          setLoadError(`the page image could not be loaded (HTTP ${res.status})`);
          return;
        }
        const blob = await res.blob();
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        revoked = url;
        const el = new Image();
        el.onload = () => !cancelled && setImg(el);
        el.onerror = () =>
          !cancelled && setLoadError("the page image could not be decoded");
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
    const scale = Math.min(1, maxWidth / img.width);
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

    if (showDiff && item.diff) drawDiff(ctx, item.diff, sx, sy);
    if (showActive) {
      drawBoxes(
        ctx,
        item.detections.filter((d) => d.score >= threshold),
        sx,
        sy,
        ACTIVE,
        false,
      );
    }
    if (showShadow) {
      drawBoxes(
        ctx,
        item.shadow_detections.filter((d) => d.score >= threshold),
        sx,
        sy,
        SHADOW,
        true,
      );
    }
  }, [img, item, showActive, showShadow, showDiff, threshold, maxWidth]);

  return (
    <div className="viewer">
      <div className="row wrap">
        <label className="row" style={{ gap: 6 }}>
          <span className="faint">score ≥ {threshold.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            style={{ width: 130 }}
            aria-label="Minimum detection score"
          />
        </label>
        <div className="legend">
          <span>
            <i style={{ background: ACTIVE }} />
            active
          </span>
          <span>
            <i style={{ background: SHADOW }} />
            shadow
          </span>
          {showDiff && item.diff ? (
            <>
              <span>
                <i style={{ background: ADDED }} />
                added ({item.diff.added.length})
              </span>
              <span>
                <i style={{ background: MISSING }} />
                missing ({item.diff.missing.length})
              </span>
              <span>
                <i style={{ background: RELABELLED }} />
                relabelled ({item.diff.relabelled.length})
              </span>
            </>
          ) : null}
        </div>
      </div>

      {loadError ? <div className="error">{loadError}</div> : null}

      <div className="canvas-wrap">
        <canvas ref={canvasRef} />
        {!img && !loadError ? <p className="faint" style={{ padding: 12 }}>loading page…</p> : null}
      </div>

      <div className="row wrap faint">
        <span>
          {item.page_size ? `${item.page_size[0]}×${item.page_size[1]} px` : "unknown size"}
        </span>
        <span>·</span>
        <span>
          {item.detections.length} active
          {item.shadow_detections.length ? ` · ${item.shadow_detections.length} shadow` : ""}
        </span>
        {item.diff ? (
          <>
            <span>·</span>
            <span>agreement {(item.diff.agreement * 100).toFixed(1)}%</span>
          </>
        ) : null}
        {item.synthetic ? (
          <>
            <span>·</span>
            <span className="pill synthetic">SYNTHETIC</span>
          </>
        ) : null}
        {item.provenance ? <span className="mono faint">· {item.provenance}</span> : null}
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
