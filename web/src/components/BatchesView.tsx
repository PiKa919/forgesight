import type { ReactElement } from "react";
/** Batches view: upload, watch, inspect. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  UploadCloud,
  FileText,
  Clock,
  XCircle,
  Loader2,
  Layers,
} from "lucide-react";
import { api, ApiError } from "../api/client";
import type { Batch, Candidate, WorkItem } from "../api/client";
import { PageViewer } from "./PageViewer";
import { BatchCounts, ErrorBox, StatusPill, TimingPanel } from "./primitives";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";

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
    <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-5 items-start">
      <div className="space-y-4">
        {/* Upload Card */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm backdrop-blur-sm">
          <CardHeader className="pb-3 pt-5 px-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <UploadCloud className="size-4 text-sky-400" />
                <CardTitle className="text-sm font-semibold tracking-tight text-zinc-100">
                  Upload Pages
                </CardTitle>
              </div>
              {busy && (
                <span className="flex items-center gap-1.5 text-xs text-sky-400 font-mono animate-pulse">
                  <Loader2 className="size-3 animate-spin" />
                  Uploading…
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="px-5 pb-5 space-y-4">
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
              className={cn(
                "group relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition-all duration-200",
                drag
                  ? "border-sky-500 bg-sky-500/5 shadow-inner"
                  : "border-zinc-700/80 bg-zinc-950/40 hover:border-zinc-500 hover:bg-zinc-900/40"
              )}
            >
              <div
                className={cn(
                  "flex size-11 items-center justify-center rounded-full transition-transform group-hover:scale-105",
                  drag
                    ? "bg-sky-500/20 text-sky-400"
                    : "bg-zinc-800/80 text-zinc-400 group-hover:text-zinc-200"
                )}
              >
                <UploadCloud className="size-5" />
              </div>

              <div className="mt-3 text-xs text-zinc-300 font-medium">
                <span>Drag & drop files or </span>
                <span className="text-sky-400 hover:text-sky-300 underline underline-offset-2">
                  browse
                </span>
              </div>

              <div className="mt-2.5 flex flex-wrap items-center justify-center gap-1.5">
                {["PDF", "PNG", "JPEG", "WEBP"].map((fmt) => (
                  <Badge
                    key={fmt}
                    variant="outline"
                    className="border-zinc-800 bg-zinc-900/80 px-1.5 py-0 text-[10px] font-mono text-zinc-400 tracking-wider"
                  >
                    {fmt}
                  </Badge>
                ))}
              </div>

              <input
                ref={fileInput}
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.webp,.tif,.tiff,.pdf"
                className="hidden"
                onChange={(e) => {
                  void upload(Array.from(e.target.files ?? []));
                  e.target.value = "";
                }}
              />
            </div>

            <div className="flex items-center justify-between gap-3 pt-1">
              <label className="text-xs font-medium text-zinc-400 shrink-0">
                Shadow candidate
              </label>
              <select
                value={shadow}
                onChange={(e) => setShadow(e.target.value)}
                className="bg-zinc-900 border-zinc-700 text-zinc-100 rounded-md px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-sky-500 transition-colors w-full max-w-[200px]"
              >
                <option value="">none</option>
                {candidates.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>
          </CardContent>
        </Card>

        {/* Batches List Card */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm backdrop-blur-sm">
          <CardHeader className="p-4 sm:p-5 pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="size-4 text-sky-400" />
                <CardTitle className="text-sm font-semibold tracking-tight text-zinc-100">
                  Batches
                </CardTitle>
              </div>
              <Badge
                variant="outline"
                className="border-zinc-700/60 bg-zinc-900/80 text-[11px] font-mono text-zinc-400"
              >
                {batches.length}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-4 sm:p-5 pt-0">
            {batches.length === 0 ? (
              <div className="rounded-lg border border-dashed border-zinc-800/80 p-8 text-center text-xs text-zinc-500">
                no batches yet
              </div>
            ) : (
              <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
                {batches.map((b) => {
                  const isSelected = b.id === selected;
                  return (
                    <button
                      key={b.id}
                      type="button"
                      aria-current={isSelected}
                      onClick={() => {
                        setSelected(b.id);
                        setActiveItem(null);
                      }}
                      className={cn(
                        "w-full text-left rounded-lg border p-3 transition-all cursor-pointer flex flex-col gap-2.5",
                        isSelected
                          ? "border-sky-500/70 bg-zinc-900/90 shadow-sm ring-1 ring-sky-500/30 text-zinc-100"
                          : "border-zinc-800/80 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/50 text-zinc-300"
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <StatusPill status={b.status} />
                          <span className="font-mono text-xs text-zinc-200 font-medium truncate">
                            {b.id.slice(0, 12)}
                          </span>
                        </div>
                        <Badge
                          variant="outline"
                          className="border-zinc-700/60 bg-zinc-900/70 text-[11px] font-mono text-zinc-300 shrink-0"
                        >
                          {b.total_items} {b.total_items === 1 ? "page" : "pages"}
                        </Badge>
                      </div>

                      <div className="flex items-center justify-between gap-2 text-xs text-zinc-400">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <Badge
                            variant="secondary"
                            className="text-[10px] font-mono text-zinc-300 bg-zinc-800/90 px-1.5 py-0 truncate"
                          >
                            {b.candidate_name}
                          </Badge>
                          {b.synthetic && (
                            <Badge
                              variant="secondary"
                              className="text-[10px] font-medium text-purple-400 border border-purple-500/20 bg-purple-500/10 px-1.5 py-0"
                            >
                              synthetic
                            </Badge>
                          )}
                        </div>
                        <span className="inline-flex items-center gap-1 text-[11px] font-mono text-zinc-500 shrink-0">
                          <Clock className="size-3 text-zinc-500" />
                          {new Date(b.created_at).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                            second: "2-digit",
                          })}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <ErrorBox error={error} />
      </div>

      <div className="space-y-4">
        {selected ? (
          <>
            <BatchHeader
              batch={batches.find((b) => b.id === selected) ?? null}
              onCancel={() => void cancel(selected)}
              busy={busy}
            />

            <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm backdrop-blur-sm">
              <CardHeader className="p-4 sm:p-5 pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="size-4 text-sky-400" />
                    <CardTitle className="text-sm font-semibold tracking-tight text-zinc-100">
                      Page Items
                    </CardTitle>
                    <Badge
                      variant="outline"
                      className="border-zinc-700/60 bg-zinc-900/80 text-[11px] font-mono text-zinc-400"
                    >
                      {items.length}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-4 sm:p-5 pt-0">
                <div className="flex flex-col md:flex-row gap-4 items-start">
                  <div className="w-full md:w-[210px] shrink-0 space-y-1.5 max-h-[64vh] overflow-y-auto pr-1">
                    {items.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-zinc-800 p-4 text-center text-xs text-zinc-500">
                        no items
                      </div>
                    ) : (
                      items.map((it, idx) => {
                        const isCurrent = (current?.id ?? "") === it.id;
                        return (
                          <button
                            key={it.id}
                            type="button"
                            aria-current={isCurrent}
                            onClick={() => setActiveItem(it.id)}
                            className={cn(
                              "w-full text-left rounded-lg border p-2.5 transition-all cursor-pointer flex flex-col gap-1.5",
                              isCurrent
                                ? "border-sky-500/70 bg-zinc-900/90 shadow-sm ring-1 ring-sky-500/30 text-zinc-100"
                                : "border-zinc-800/80 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/50 text-zinc-300"
                            )}
                          >
                            <div className="flex items-center justify-between gap-1">
                              <span className="text-xs font-semibold tracking-tight">
                                Page {idx + 1}
                              </span>
                              <StatusPill status={it.state} className="scale-90 origin-right" />
                            </div>
                            <div className="flex items-center justify-between gap-1 text-[11px] text-zinc-400">
                              <span className="font-mono text-zinc-400">
                                {it.detections.length}{" "}
                                {it.detections.length === 1 ? "box" : "boxes"}
                              </span>
                              {it.timings.service_ms ? (
                                <span className="font-mono tabular-nums text-zinc-500">
                                  {it.timings.service_ms.toFixed(0)} ms
                                </span>
                              ) : null}
                            </div>
                            {it.role === "shadow" && (
                              <div className="flex items-center gap-1">
                                <Badge
                                  variant="secondary"
                                  className="text-[10px] px-1.5 py-0 text-purple-400 bg-purple-500/10 border-purple-500/20 font-mono"
                                >
                                  shadow
                                </Badge>
                              </div>
                            )}
                          </button>
                        );
                      })
                    )}
                  </div>

                  <div className="flex-1 min-w-0 w-full">
                    {current ? (
                      <div className="space-y-4">
                        <PageViewer
                          item={current}
                          imageUrl={current.image_url}
                          showActive={showActive}
                          showShadow={showShadow && hasShadow}
                          showDiff={showDiff && hasShadow}
                          threshold={threshold}
                          maxWidth={760}
                          onThresholdChange={setThreshold}
                          onToggleActive={setShowActive}
                          onToggleShadow={setShowShadow}
                          onToggleDiff={setShowDiff}
                        />
                        <TimingPanel t={current.timings} title="Per-item timing" />
                        {current.failure_code && (
                          <div className="rounded-md border border-red-500/20 bg-red-500/10 p-3 text-xs font-mono text-red-400">
                            <span className="font-semibold">{current.failure_code}</span>
                            {current.failure_detail ? `: ${current.failure_detail}` : ""}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-800 text-xs text-zinc-500">
                        select an item to see its boxes and timings
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        ) : (
          <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm backdrop-blur-sm">
            <CardContent className="flex flex-col items-center justify-center p-12 text-center space-y-2">
              <FileText className="size-8 text-zinc-600 mb-1" />
              <p className="text-sm font-medium text-zinc-400">No Batch Selected</p>
              <p className="text-xs text-zinc-500 max-w-sm">
                Upload a batch, or pick one from the list, to see per-page boxes and the
                timing split.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
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
  const isCancellable = ["queued", "running"].includes(batch.status);

  return (
    <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm backdrop-blur-sm">
      <CardContent className="p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2.5">
            <h2 className="text-sm font-semibold tracking-tight text-zinc-100 flex items-center gap-1.5">
              <span className="text-zinc-400 font-normal">batch</span>
              <span className="font-mono text-zinc-100">{batch.id.slice(0, 12)}</span>
            </h2>
            <StatusPill status={batch.status} />
            <Badge
              variant="outline"
              className="font-mono text-xs text-zinc-300 border-zinc-700/80 bg-zinc-800/40"
            >
              {batch.candidate_name}
            </Badge>
            <span className="font-mono text-[11px] text-zinc-500">
              {batch.release_id.slice(0, 12)}
            </span>
            {batch.cancel_requested_at && (
              <Badge variant="warning" className="text-[10px] tracking-tight">
                cancel requested
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-3 self-end sm:self-auto">
            <BatchCounts batch={batch} />
            {isCancellable && (
              <Button
                variant="destructive"
                size="sm"
                onClick={onCancel}
                disabled={busy}
                className="gap-1.5 text-xs font-medium"
              >
                <XCircle className="size-3.5" />
                Cancel
              </Button>
            )}
          </div>
        </div>

        {batch.timings && (
          <div className="pt-2 border-t border-zinc-800/60">
            <TimingPanel t={batch.timings} title="Batch timing (mean of items)" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export { ApiError };
