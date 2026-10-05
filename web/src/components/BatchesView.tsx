import type { ReactElement } from "react";
/** Batches view: upload, watch, inspect. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Batch, Candidate, WorkItem } from "../api/client";
import { PageViewer } from "./PageViewer";
import { BatchCounts, ErrorBox, StatusPill, TimingPanel } from "./primitives";

const POLL_MS = 1000;

export function BatchesView({ onChanged }: { onChanged: () => void }): ReactElement {
  const [batches, setBatches] = useState<Batch[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [items, setItems] = useState<WorkItem[]>([]);
  const [activeItem, setActiveItem] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.3);
  const [showActive, setShowActive] = useState(true);
  const [showShadow, setShowShadow] = useState(true);
  const [showDiff, setShowDiff] = useState(true);
  const [shadow, setShadow] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const refreshBatches = useCallback(async () => {
    try {
      const r = await api.listBatches();
      setBatches(r.items);
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, []);

  useEffect(() => {
    void refreshBatches();
    void api.listCandidates().then(setCandidates).catch(setError);
  }, [refreshBatches]);

  // Poll only while a batch is still in flight. Once everything is terminal there
  // is nothing to poll for, and a permanent poll would be a permanent tax.
  const inFlight = useMemo(
    () => batches.some((b) => ["queued", "running"].includes(b.status)),
    [batches],
  );
  useEffect(() => {
    if (!inFlight) return;
    const t = setInterval(() => void refreshBatches(), POLL_MS);
    return () => clearInterval(t);
  }, [inFlight, refreshBatches]);

  useEffect(() => {
    if (!selected) {
      setItems([]);
      return;
    }
    let stop = false;
    const load = async () => {
      try {
        const next = await api.listItems(selected);
        if (!stop) {
          setItems(next);
          setError(null);
        }
      } catch (e) {
        if (!stop) setError(e);
      }
    };
    void load();
    if (!inFlight) return;
    const t = setInterval(() => void load(), POLL_MS);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [selected, inFlight]);

  const current = items.find((i) => i.id === activeItem) ?? items[0] ?? null;
  const hasShadow = items.some((i) => i.shadow_detections.length > 0);

  const upload = async (files: File[]) => {
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      // A key derived from the file names and sizes, so a double-click retries
      // the same batch instead of creating a second one.
      const key = files
        .map((f) => `${f.name}:${f.size}`)
        .sort()
        .join("|")
        .slice(0, 120);
      const b = await api.uploadBatch(files, shadow || null, key);
      setSelected(b.id);
      await refreshBatches();
      onChanged();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (id: string) => {
    setBusy(true);
    try {
      await api.cancelBatch(id);
      await refreshBatches();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid two">
      <div className="stack">
        <div className="panel">
          <h2>Upload pages</h2>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              void upload(Array.from(e.dataTransfer.files));
            }}
            onClick={() => fileInput.current?.click()}
            style={{
              border: `1px dashed ${drag ? "var(--accent)" : "var(--line)"}`,
              borderRadius: 6,
              padding: "22px 14px",
              textAlign: "center",
              cursor: "pointer",
              background: drag ? "var(--panel-2)" : "transparent",
            }}
          >
            <div className="dim">Drop page images or PDFs here</div>
            <div className="faint">PNG, JPEG, WebP, TIFF or unencrypted PDF</div>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept=".png,.jpg,.jpeg,.webp,.tif,.tiff,.pdf"
              style={{ display: "none" }}
              onChange={(e) => {
                void upload(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <label className="faint" style={{ whiteSpace: "nowrap" }}>
              shadow candidate
            </label>
            <select value={shadow} onChange={(e) => setShadow(e.target.value)}>
              <option value="">none</option>
              {candidates.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          {busy ? <p className="faint">uploading…</p> : null}
        </div>

        <div className="panel">
          <h2>Batches</h2>
          {batches.length === 0 ? (
            <p className="faint">no batches yet</p>
          ) : (
            <div className="list">
              {batches.map((b) => (
                <button
                  key={b.id}
                  aria-current={b.id === selected}
                  onClick={() => {
                    setSelected(b.id);
                    setActiveItem(null);
                  }}
                >
                  <div className="row">
                    <StatusPill status={b.status} />
                    <span className="mono faint">{b.id.slice(0, 12)}</span>
                    <span className="spacer" />
                    <span className="faint">{b.candidate_name}</span>
                  </div>
                  <div className="faint">
                    {b.total_items} pages ·{" "}
                    {new Date(b.created_at).toLocaleTimeString()}
                    {b.synthetic ? " · synthetic" : ""}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <ErrorBox error={error} />
      </div>

      <div className="stack">
        {selected ? (
          <>
            <BatchHeader
              batch={batches.find((b) => b.id === selected) ?? null}
              onCancel={() => void cancel(selected)}
              busy={busy}
            />
            <div className="panel">
              <div className="row wrap" style={{ marginBottom: 8 }}>
                <span className="faint">items</span>
                {hasShadow ? (
                  <>
                    <label className="row" style={{ gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={showActive}
                        onChange={(e) => setShowActive(e.target.checked)}
                        style={{ width: "auto" }}
                      />
                      <span className="faint">active</span>
                    </label>
                    <label className="row" style={{ gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={showShadow}
                        onChange={(e) => setShowShadow(e.target.checked)}
                        style={{ width: "auto" }}
                      />
                      <span className="faint">shadow</span>
                    </label>
                    <label className="row" style={{ gap: 4 }}>
                      <input
                        type="checkbox"
                        checked={showDiff}
                        onChange={(e) => setShowDiff(e.target.checked)}
                        style={{ width: "auto" }}
                      />
                      <span className="faint">diff</span>
                    </label>
                  </>
                ) : null}
              </div>

              <div className="grid two" style={{ gridTemplateColumns: "190px 1fr" }}>
                <div className="list">
                  {items.length === 0 ? (
                    <p className="faint" style={{ padding: 8 }}>
                      no items
                    </p>
                  ) : (
                    items.map((it) => (
                      <button
                        key={it.id}
                        aria-current={(current?.id ?? "") === it.id}
                        onClick={() => setActiveItem(it.id)}
                      >
                        <div className="row">
                          <span
                            className="pill"
                            style={{
                              color: stateColor(it.state),
                              borderColor: stateColor(it.state),
                            }}
                          >
                            {it.state}
                          </span>
                          {it.role === "shadow" ? (
                            <span className="pill info">shadow</span>
                          ) : null}
                        </div>
                        <div className="faint">
                          {it.detections.length} boxes
                          {it.timings.service_ms
                            ? ` · ${it.timings.service_ms.toFixed(0)} ms`
                            : ""}
                        </div>
                      </button>
                    ))
                  )}
                </div>

                {current ? (
                  <div className="stack">
                    <PageViewer
                      item={current}
                      imageUrl={current.image_url}
                      showActive={showActive}
                      showShadow={showShadow && hasShadow}
                      showDiff={showDiff && hasShadow}
                      threshold={threshold}
                      maxWidth={760}
                      onThresholdChange={setThreshold}
                    />
                    <TimingPanel t={current.timings} title="Per-item timing" />
                    {current.failure_code ? (
                      <div className="error">
                        {current.failure_code}
                        {current.failure_detail ? `: ${current.failure_detail}` : ""}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <p className="hint">select an item to see its boxes and timings</p>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="panel">
            <p className="hint">
              Upload a batch, or pick one from the list, to see per-page boxes and the
              timing split.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function stateColor(state: string): string {
  if (state === "succeeded") return "var(--ok)";
  if (state === "failed") return "var(--bad)";
  if (state === "cancelled") return "var(--warn)";
  return "var(--accent)";
}

function BatchHeader({
  batch,
  onCancel,
  busy,
}: {
  batch: Batch | null;
  onCancel: () => void;
  busy: boolean;
}) {
  if (!batch) return null;
  return (
    <div className="panel">
      <div className="row wrap">
        <h2 style={{ margin: 0 }}>batch {batch.id.slice(0, 12)}</h2>
        <div className="spacer" />
        <BatchCounts batch={batch} />
        {["queued", "running"].includes(batch.status) ? (
          <button className="danger" onClick={onCancel} disabled={busy}>
            cancel
          </button>
        ) : null}
      </div>
      <div className="faint" style={{ marginTop: 6 }}>
        release <span className="mono">{batch.candidate_name}</span> ·{" "}
        <span className="mono">{batch.release_id.slice(0, 12)}</span>
        {batch.cancel_requested_at ? " · cancel requested" : ""}
      </div>
      {batch.timings ? (
        <div style={{ marginTop: 10 }}>
          <TimingPanel t={batch.timings} title="Batch timing (mean of items)" />
        </div>
      ) : null}
    </div>
  );
}

export { ApiError };
