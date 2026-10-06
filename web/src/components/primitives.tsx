/** Small shared display components. Kept together so the formatting rules for
 *  durations, bytes and status pills are stated once. */

import type { ReactNode } from "react";
import type { Batch, TimingSplit } from "../api/client";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";

export function ms(v: number | null | undefined, nd = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(nd)} ms`;
}

export function bytes(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let n = v;
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function pct(v: number | null | undefined, nd = 1): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(nd)}%`;
}

export function StatusPill({
  status,
  className,
}: {
  status: string;
  className?: string;
}): ReactNode {
  const s = status.toLowerCase();
  let variant: "success" | "destructive" | "warning" | "info" | "secondary" = "info";
  let dotColor = "bg-sky-400";

  if (["succeeded", "stable", "ok", "promote", "promoted"].includes(s)) {
    variant = "success";
    dotColor = "bg-emerald-400";
  } else if (["failed", "unstable", "error", "rollback", "demote", "demoted"].includes(s)) {
    variant = "destructive";
    dotColor = "bg-red-400";
  } else if (["partial", "cancelled", "paused"].includes(s)) {
    variant = "warning";
    dotColor = "bg-amber-400";
  } else if (["running", "queued", "pending"].includes(s)) {
    variant = "warning";
    dotColor = "bg-amber-400";
  } else if (["synthetic", "shadow"].includes(s)) {
    variant = "secondary";
    dotColor = "bg-purple-400";
  }

  return (
    <Badge
      variant={variant}
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium tracking-tight shadow-none",
        className
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full shrink-0",
          dotColor,
          s === "running" && "animate-pulse"
        )}
      />
      <span>{status}</span>
    </Badge>
  );
}

/**
 * The timing split.
 *
 * Queue-inclusive and service time are both shown, always labelled, and never
 * collapsed into one number.
 */
export function TimingPanel({ t, title }: { t: TimingSplit | null; title?: string }): ReactNode {
  if (!t) return <p className="text-xs text-zinc-500 italic">no timings recorded</p>;
  const rows: Array<[string, number | null | undefined]> = [
    ["queue-inclusive", t.queue_inclusive_ms],
    ["service (excl. queue)", t.service_ms],
    ["preprocess", t.preprocess_ms],
    ["inference", t.infer_ms],
    ["postprocess", t.postprocess_ms],
    ["persist", t.persist_ms],
    ["batch wait", t.batch_wait_ms],
  ];
  const slowest = Math.max(
    1,
    ...rows.map(([, v]) => (typeof v === "number" ? v : 0)),
  );
  return (
    <div className="space-y-2">
      {title ? (
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
          {title}
        </h3>
      ) : null}
      <div className="timing grid grid-cols-[auto_1fr_auto] gap-x-3 gap-y-1.5 items-center text-xs">
        {rows.map(([label, v]) => (
          <Fragmented key={label}>
            <span className="label text-zinc-400">{label}</span>
            <div className="bar h-1.5 w-full min-w-[70px] bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-400 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, ((v ?? 0) / slowest) * 100)}%` }}
              />
            </div>
            <span className="value font-mono text-right tabular-nums text-zinc-200">
              {ms(v)}
            </span>
          </Fragmented>
        ))}
        {t.peak_rss_bytes ? (
          <>
            <span className="label text-zinc-400">peak RSS</span>
            <span />
            <span className="value font-mono text-right tabular-nums text-zinc-200">
              {bytes(t.peak_rss_bytes)}
            </span>
          </>
        ) : null}
      </div>
    </div>
  );
}

function Fragmented({ children }: { children: ReactNode }): ReactNode {
  return <>{children}</>;
}

export function BatchCounts({ batch }: { batch: Batch }): ReactNode {
  const total = Math.max(1, batch.total_items);
  const order = ["succeeded", "failed", "cancelled", "running", "queued"] as const;
  const present = order.filter((k) => (batch.counts[k] ?? 0) > 0);
  if (present.length === 0) return <span className="text-xs text-zinc-500 font-mono">queued</span>;
  return (
    <div className="stack flex flex-col gap-1.5 min-w-[130px]">
      <div className="row flex items-center gap-1.5">
        <StatusPill status={batch.status} />
        <span className="text-xs text-zinc-400 font-mono">
          {present.map((k) => `${batch.counts[k] ?? 0} ${k}`).join(" · ")}
        </span>
      </div>
      <div className="bar flex h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
        {present.map((k) => (
          <span
            key={k}
            className={cn(
              "h-full transition-all",
              k === "succeeded" && "bg-emerald-400",
              k === "failed" && "bg-red-400",
              k === "cancelled" && "bg-amber-400",
              k === "running" && "bg-sky-400 animate-pulse",
              k === "queued" && "bg-zinc-600"
            )}
            style={{
              width: `${((batch.counts[k] ?? 0) / total) * 100}%`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }): ReactNode {
  if (!error) return null;
  const text = error instanceof Error ? error.message : String(error);
  return (
    <div className="error rounded-md border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs font-mono text-red-400 whitespace-pre-wrap">
      {text}
    </div>
  );
}

