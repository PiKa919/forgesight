/** Small shared display components. Kept together so the formatting rules for
 *  durations, bytes and status pills are stated once. */

import type { ReactNode } from "react";
import type { Batch, TimingSplit } from "../api/client";

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

export function StatusPill({ status }: { status: string }): ReactNode {
  const good = ["succeeded", "stable", "ok"].includes(status);
  const bad = ["failed", "unstable", "error"].includes(status);
  const warn = ["partial", "cancelled", "running", "queued"].includes(status);
  return (
    <span
      className={`pill ${good ? "ok" : bad ? "bad" : warn ? "warn" : "info"}`}
    >
      {status}
    </span>
  );
}

/**
 * The timing split.
 *
 * Queue-inclusive and service time are both shown, always labelled, and never
 * collapsed into one number. A viewer reading "120 ms" cannot tell whether that
 * includes the wait for a free worker, and the whole point of the system is that
 * it can.
 */
export function TimingPanel({ t, title }: { t: TimingSplit | null; title?: string }): ReactNode {
  if (!t) return <p className="faint">no timings recorded</p>;
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
    <div>
      {title ? <h3>{title}</h3> : null}
      <div className="timing">
        {rows.map(([label, v]) => (
          <Fragmented key={label}>
            <span className="label">{label}</span>
            <span className="bar">
              <span style={{ width: `${Math.min(100, ((v ?? 0) / slowest) * 100)}%` }} />
            </span>
            <span className="value">{ms(v)}</span>
          </Fragmented>
        ))}
        {t.peak_rss_bytes ? (
          <>
            <span className="label">peak RSS</span>
            <span />
            <span className="value">{bytes(t.peak_rss_bytes)}</span>
          </>
        ) : null}
      </div>
      <p className="faint" style={{ marginBottom: 0 }}>
        queue-inclusive = received → persisted, and includes waiting for a worker.
        service = claimed → persisted, and does not.
      </p>
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
  if (present.length === 0) return <span className="faint">queued</span>;
  return (
    <div className="stack" style={{ gap: 4, minWidth: 130 }}>
      <div className="row" style={{ gap: 6 }}>
        <StatusPill status={batch.status} />
        <span className="faint">
          {present.map((k) => `${batch.counts[k] ?? 0} ${k}`).join(" · ")}
        </span>
      </div>
      <div className="bar">
        {present.map((k) => (
          <span
            key={k}
            style={{
              width: `${((batch.counts[k] ?? 0) / total) * 100}%`,
              background:
                k === "succeeded"
                  ? "var(--ok)"
                  : k === "failed"
                    ? "var(--bad)"
                    : k === "cancelled"
                      ? "var(--warn)"
                      : "var(--line)",
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
  return <div className="error">{text}</div>;
}
