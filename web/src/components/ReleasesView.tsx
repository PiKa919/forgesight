import type { ReactElement } from "react";
/** Releases view: candidate evaluation, gate verdicts matrix, promotion and rollback. */

import { useCallback, useEffect, useState } from "react";
import {
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  RotateCcw,
  Loader2,
  History,
  ArrowRight,
  TrendingUp,
} from "lucide-react";
import { api } from "../api/client";
import type { Candidate, Channel, Evaluation } from "../api/client";
import { ErrorBox, bytes } from "./primitives";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";

export function ReleasesView({ onChanged }: { onChanged: () => void }): ReactElement {
  const [channel, setChannel] = useState<Channel | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [evaluations, setEvaluations] = useState<Record<string, Evaluation>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [promoteReason, setPromoteReason] = useState("");
  const [rollbackReason, setRollbackReason] = useState("");
  const [showRollbackConfirm, setShowRollbackConfirm] = useState(false);
  const [showPromoteConfirm, setShowPromoteConfirm] = useState(false);
  const [evalConfirmCandidate, setEvalConfirmCandidate] = useState<Candidate | null>(null);
  const [evaluatingCandidateId, setEvaluatingCandidateId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [note, setNote] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [ch, cs] = await Promise.all([api.releases(), api.listCandidates()]);
      setChannel(ch);
      setCandidates(cs);
      const firstCandidate = cs[0];
      if (firstCandidate) {
        setSelected((prev) => prev ?? firstCandidate.id);
      }
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const evaluate = async (id: string) => {
    setBusy(true);
    setEvaluatingCandidateId(id);
    setError(null);
    setNote(null);
    try {
      const ev = await api.evaluate(id);
      setEvaluations((prev) => ({ ...prev, [id]: ev }));
      setSelected(id);
      setNote(`Evaluation completed for ${ev.candidate_name}`);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      setEvaluatingCandidateId(null);
    }
  };

  const promote = async () => {
    if (!channel || !selected) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const r = await api.promote(
        selected,
        channel.version,
        promoteReason || "promoted from the UI",
      );
      setChannel(r.channel);
      setPromoteReason("");
      setShowPromoteConfirm(false);
      setNote(`Promoted ${r.release.candidate_name} as version v${r.channel.version}`);
      onChanged();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const rollback = async () => {
    if (!channel) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const r = await api.rollback(
        channel.version,
        rollbackReason || "rolled back from the UI",
      );
      setChannel(r.channel);
      setRollbackReason("");
      setShowRollbackConfirm(false);
      setNote(`Rolled back to ${r.release.candidate_name}`);
      onChanged();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const ev = selected ? evaluations[selected] : undefined;
  const candidate = candidates.find((c) => c.id === selected) ?? null;
  const allGatesPassed = ev?.gates && ev.gates.length > 0 && ev.gates.every((g) => g.passed);

  return (
    <div className="space-y-6">
      {/* Top Notification / Error */}
      <ErrorBox error={error} />
      {evaluatingCandidateId && (
        <div className="flex items-center gap-2 rounded-lg border border-sky-500/20 bg-sky-500/10 px-4 py-2.5 text-xs text-sky-300">
          <Loader2 className="size-4 shrink-0 text-sky-400 animate-spin" />
          <span>Running benchmark evaluation and testing G1–G6 gates...</span>
        </div>
      )}
      {note && !evaluatingCandidateId && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-4 py-2.5 text-xs text-emerald-300">
          <CheckCircle2 className="size-4 shrink-0 text-emerald-400" />
          <span>{note}</span>
        </div>
      )}

      {/* Channel Header Card */}
      <Card className="border-zinc-800 bg-zinc-900/60 shadow-lg backdrop-blur-sm">
        <CardHeader className="p-5 pb-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2.5">
                <CardTitle className="text-base font-semibold text-zinc-100">
                  Release Channel
                </CardTitle>
                {channel && (
                  <Badge variant="info" className="font-mono text-xs">
                    v{channel.version}
                  </Badge>
                )}
              </div>
              <CardDescription className="text-xs text-zinc-400">
                Active serving release and immutable deployment history
              </CardDescription>
            </div>

            <div className="flex items-center gap-2.5">
              {channel?.active_candidate_name ? (
                <div className="flex items-center gap-2 bg-zinc-950/60 border border-zinc-800 rounded-md px-3 py-1.5">
                  <span className="text-[11px] uppercase tracking-wider text-zinc-500 font-semibold">
                    Active
                  </span>
                  <span className="font-mono text-xs font-semibold text-sky-400">
                    {channel.active_candidate_name}
                  </span>
                </div>
              ) : (
                <Badge variant="outline" className="text-xs text-zinc-500">
                  No Active Release
                </Badge>
              )}

              <Button
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5 text-xs font-medium"
                onClick={() => setShowRollbackConfirm(true)}
                disabled={busy || !channel?.active_release_id}
              >
                <RotateCcw className="size-3.5" />
                Roll Back
              </Button>
            </div>
          </div>
        </CardHeader>

        {/* Rollback Confirmation Modal / Inline Drawer */}
        {showRollbackConfirm && (
          <div className="border-t border-red-500/20 bg-red-950/20 p-4 mx-5 mb-4 rounded-lg space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-red-300">
              <AlertCircle className="size-4 text-red-400" />
              <span>Confirm Release Rollback</span>
            </div>
            <p className="text-xs text-zinc-400">
              Rolling back restores the previous stable release configuration for channel v{channel?.version}.
            </p>
            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="text"
                placeholder="Reason for audit log (e.g. latency regression or anomalous outputs)"
                value={rollbackReason}
                onChange={(e) => setRollbackReason(e.target.value)}
                className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-500 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
              />
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs"
                  onClick={() => setShowRollbackConfirm(false)}
                  disabled={busy}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  className="h-8 text-xs font-medium"
                  onClick={() => void rollback()}
                  disabled={busy}
                >
                  {busy ? <Loader2 className="size-3 animate-spin mr-1" /> : null}
                  Confirm Rollback
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Release History */}
        {channel?.history && channel.history.length > 0 && (
          <CardContent className="p-5 pt-0">
            <div className="pt-3 border-t border-zinc-800/80 space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-400">
                <History className="size-3.5 text-zinc-500" />
                <span>Append-Only Ledger</span>
              </div>
              <div className="overflow-x-auto rounded-md border border-zinc-800/80 bg-zinc-950/40">
                <table className="w-full text-xs text-left">
                  <thead className="bg-zinc-900/60 text-zinc-400 border-b border-zinc-800">
                    <tr>
                      <th className="py-2 px-3 font-medium">When</th>
                      <th className="py-2 px-3 font-medium">Action</th>
                      <th className="py-2 px-3 font-medium">Candidate</th>
                      <th className="py-2 px-3 font-medium text-right">Channel v</th>
                      <th className="py-2 px-3 font-medium">Reason</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50">
                    {channel.history.map((h) => (
                      <tr key={h.id} className="hover:bg-zinc-900/30">
                        <td className="py-2 px-3 text-zinc-400 whitespace-nowrap">
                          {new Date(h.created_at).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                            second: "2-digit",
                          })}
                        </td>
                        <td className="py-2 px-3">
                          <Badge
                            variant={
                              h.action === "promote"
                                ? "success"
                                : h.action === "rollback"
                                ? "destructive"
                                : "secondary"
                            }
                            className="text-[10px] uppercase font-mono px-2 py-0"
                          >
                            {h.action}
                          </Badge>
                        </td>
                        <td className="py-2 px-3 font-mono text-zinc-200">
                          {h.candidate_name}
                        </td>
                        <td className="py-2 px-3 font-mono text-right text-zinc-400">
                          {h.channel_version}
                        </td>
                        <td className="py-2 px-3 text-zinc-400 truncate max-w-xs">
                          {h.reason || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Main Grid: Candidates List & Promotion / Gates Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Candidates Grid (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold tracking-tight text-zinc-200">
              Registered Candidates ({candidates.length})
            </h2>
            <span className="text-xs text-zinc-500">Select to inspect or evaluate</span>
          </div>

          <div className="space-y-3">
            {candidates.map((c) => {
              const arch = getModelArchitecture(c);
              const rt = getRuntimeDisplay(c.runtime);
              const isSelected = selected === c.id;
              const isEvaluated = Boolean(evaluations[c.id]);
              const candidateGates = evaluations[c.id]?.gates ?? [];
              const passedGates =
                candidateGates.length > 0 && candidateGates.every((g) => g.passed);

              return (
                <Card
                  key={c.id}
                  onClick={() => {
                    setSelected(c.id);
                    setShowPromoteConfirm(false);
                  }}
                  className={cn(
                    "cursor-pointer transition-all border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-850/60 shadow-sm",
                    isSelected &&
                      "border-sky-500/60 bg-sky-950/20 ring-1 ring-sky-500/40 shadow-md",
                  )}
                >
                  <CardContent className="p-4 space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="space-y-1">
                        <div className="flex items-center flex-wrap gap-1.5">
                          <span className="font-mono text-sm font-semibold text-zinc-100">
                            {c.name}
                          </span>
                          {c.artifact_name.startsWith("negctl") && (
                            <Badge
                              variant="destructive"
                              className="text-[10px] px-1.5 py-0 uppercase"
                            >
                              Negative Control
                            </Badge>
                          )}
                          {c.name.startsWith("ref-") && (
                            <Badge
                              variant="secondary"
                              className="text-[10px] px-1.5 py-0 uppercase"
                            >
                              Reference
                            </Badge>
                          )}
                        </div>
                        <div className="text-xs text-zinc-400">
                          <span className="font-medium text-zinc-300">{arch.name}</span>{" "}
                          <span className="text-sky-400 font-mono">({arch.arch})</span>
                        </div>
                      </div>

                      <Badge variant={rt.variant} className="text-[11px] shrink-0 font-mono">
                        {rt.label}
                      </Badge>
                    </div>

                    {/* Metadata & Provenance */}
                    <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-zinc-800/80">
                      <div>
                        <span className="text-zinc-500 block text-[10px] uppercase font-semibold">
                          Provenance
                        </span>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          {c.valid ? (
                            <span className="inline-flex items-center gap-1 text-emerald-400 font-medium text-xs">
                              <ShieldCheck className="size-3.5" />
                              <span>Verified</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-red-400 font-medium text-xs">
                              <AlertCircle className="size-3.5" />
                              <span>{c.invalid_reason || "Unverified"}</span>
                            </span>
                          )}
                        </div>
                      </div>

                      <div>
                        <span className="text-zinc-500 block text-[10px] uppercase font-semibold">
                          Preprocessing
                        </span>
                        <span className="text-zinc-300 font-mono text-[11px] block mt-0.5">
                          {c.resize} → {c.target}px{c.tile_enabled ? " (tile)" : ""}
                        </span>
                      </div>
                    </div>

                    {/* Footer Actions */}
                    <div className="flex items-center justify-between pt-2 border-t border-zinc-800/80 text-xs">
                      <div className="flex items-center gap-3 text-zinc-400 font-mono text-[11px]">
                        <span>threads: {c.threads}</span>
                        <span>thresh: {c.score_threshold.toFixed(2)}</span>
                      </div>

                      <div className="flex items-center gap-2">
                        {isEvaluated && (
                          <Badge
                            variant={passedGates ? "success" : "destructive"}
                            className="text-[10px] font-mono px-1.5 py-0"
                          >
                            {passedGates ? "GATES PASS" : "GATES BLOCKED"}
                          </Badge>
                        )}

                        <Button
                          size="sm"
                          variant={isEvaluated ? "secondary" : "outline"}
                          className="h-7 text-xs px-2.5"
                          disabled={busy}
                          onClick={(e) => {
                            e.stopPropagation();
                            setEvalConfirmCandidate(c);
                          }}
                        >
                          {evaluatingCandidateId === c.id ? (
                            <>
                              <Loader2 className="size-3 animate-spin mr-1" />
                              Evaluating...
                            </>
                          ) : isEvaluated ? (
                            "Re-evaluate"
                          ) : (
                            "Evaluate"
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Right Column: Promotion Workspace & Gates Matrix (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold tracking-tight text-zinc-200">
              Evaluation & Gate Verdicts
            </h2>
            {candidate && (
              <span className="font-mono text-xs text-sky-400 truncate max-w-xs">
                {candidate.name}
              </span>
            )}
          </div>

          {candidate ? (
            <div className="space-y-4">
              {/* Selected Candidate Technical Profile Card */}
              <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm">
                <CardHeader className="p-4 pb-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-sm font-semibold text-zinc-100">
                        Candidate Technical Profile
                      </CardTitle>
                      <CardDescription className="text-xs text-zinc-400">
                        {candidate.artifact_name} · {candidate.artifact_format} · {candidate.repo_id}@{candidate.revision.slice(0, 8)}
                      </CardDescription>
                    </div>
                    <Badge variant="outline" className="font-mono text-[11px] text-zinc-400">
                      {bytes(0)} artifact
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <div className="rounded-md bg-zinc-950/70 border border-zinc-800/80 p-2.5 font-mono text-[11px] text-zinc-400 break-all">
                    <span className="text-zinc-600 select-none mr-2">SHA256:</span>
                    {candidate.candidate_hash}
                  </div>
                </CardContent>
              </Card>

              {/* Gate Verdicts Matrix */}
              <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm">
                <CardHeader className="p-4 pb-3">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <CardTitle className="text-sm font-semibold text-zinc-100">
                        Gate Verdicts Matrix (G1 – G6)
                      </CardTitle>
                      <CardDescription className="text-xs text-zinc-400">
                        Evaluated relative to pinned reference model to prevent regression creep
                      </CardDescription>
                    </div>
                    {ev && (
                      <Badge
                        variant={allGatesPassed ? "success" : "destructive"}
                        className="text-xs font-mono"
                      >
                        {allGatesPassed ? "ALL GATES PASSED" : "GATE VIOLATIONS"}
                      </Badge>
                    )}
                  </div>
                </CardHeader>

                <CardContent className="p-4 pt-0 space-y-4">
                  {ev?.gates && ev.gates.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                      {ev.gates.map((g) => {
                        const info = parseGateInfo(g.gate);
                        return (
                          <div
                            key={g.gate}
                            className={cn(
                              "rounded-lg border p-3 flex flex-col justify-between transition-colors",
                              g.passed
                                ? "border-zinc-800 bg-zinc-950/50"
                                : "border-red-500/40 bg-red-950/20 ring-1 ring-red-500/30",
                            )}
                          >
                            <div className="space-y-1.5">
                              <div className="flex items-center justify-between gap-1">
                                <Badge
                                  variant={g.passed ? "outline" : "destructive"}
                                  className="text-[10px] font-mono px-1.5 py-0 font-bold"
                                >
                                  {info.code}
                                </Badge>
                                <span
                                  className={cn(
                                    "text-[10px] uppercase font-bold tracking-wider",
                                    g.passed ? "text-emerald-400" : "text-red-400",
                                  )}
                                >
                                  {g.passed ? "PASS" : "BLOCK"}
                                </span>
                              </div>

                              <div className="text-xs font-semibold text-zinc-200">
                                {info.title}
                              </div>

                              {g.detail && (
                                <p className="text-[10px] text-zinc-400 line-clamp-2 leading-relaxed">
                                  {g.detail}
                                </p>
                              )}
                            </div>

                            <div className="grid grid-cols-2 gap-2 pt-2.5 mt-2 border-t border-zinc-800/80 text-[10px] font-mono">
                              <div>
                                <span className="text-zinc-500 block">Observed</span>
                                <span
                                  className={cn(
                                    "font-semibold text-[11px]",
                                    g.passed ? "text-zinc-200" : "text-red-400",
                                  )}
                                >
                                  {fmt(g.observed)}
                                </span>
                              </div>
                              <div>
                                <span className="text-zinc-500 block">Threshold</span>
                                <span className="text-zinc-400 text-[11px]">
                                  {fmt(g.threshold)}
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-dashed border-zinc-800 p-6 text-center space-y-3">
                      <div className="inline-flex size-9 items-center justify-center rounded-full bg-zinc-800/80 text-zinc-400">
                        <TrendingUp className="size-4" />
                      </div>
                      <div className="space-y-1 max-w-sm mx-auto">
                        <p className="text-xs font-medium text-zinc-300">
                          Candidate Unevaluated
                        </p>
                        <p className="text-xs text-zinc-500">
                          Run the evaluation suite to measure latency, quality shift, memory fit, and provenance gates before promotion.
                        </p>
                      </div>
                      <Button
                        variant="default"
                        size="sm"
                        className="text-xs h-8"
                        disabled={busy}
                        onClick={() => setEvalConfirmCandidate(candidate)}
                      >
                        {evaluatingCandidateId === candidate.id ? (
                          <>
                            <Loader2 className="size-3 animate-spin mr-1.5" />
                            Evaluating...
                          </>
                        ) : (
                          "Evaluate Candidate Now"
                        )}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Promotion Controls Card */}
              <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm">
                <CardHeader className="p-4 pb-3">
                  <CardTitle className="text-sm font-semibold text-zinc-100">
                    Channel Promotion Action
                  </CardTitle>
                  <CardDescription className="text-xs text-zinc-400">
                    Promote this validated candidate to active serving status
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-4 pt-0 space-y-3">
                  {!allGatesPassed && ev && (
                    <div className="flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5 text-xs text-amber-300">
                      <AlertCircle className="size-4 shrink-0 text-amber-400" />
                      <span>
                        Promotion is locked because this candidate did not pass all gate criteria.
                      </span>
                    </div>
                  )}

                  {!showPromoteConfirm ? (
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                      <p className="text-xs text-zinc-400">
                        {allGatesPassed
                          ? `All release gates passed. Ready to promote ${candidate.name} to channel v${(channel?.version ?? 0) + 1}.`
                          : "Run and pass all gate criteria to unlock promotion."}
                      </p>
                      <Button
                        variant="default"
                        size="sm"
                        className="h-8 gap-1.5 text-xs font-semibold shrink-0"
                        onClick={() => setShowPromoteConfirm(true)}
                        disabled={busy || !channel || !allGatesPassed}
                      >
                        <ArrowRight className="size-3.5" />
                        Promote to Active
                      </Button>
                    </div>
                  ) : (
                    <div className="border border-sky-500/30 bg-sky-950/20 p-4 rounded-lg space-y-3">
                      <div className="flex items-center justify-between text-xs font-semibold text-sky-300">
                        <div className="flex items-center gap-2">
                          <ShieldCheck className="size-4 text-sky-400" />
                          <span>Confirm Release Promotion</span>
                        </div>
                        <Badge variant="info" className="font-mono text-[11px]">
                          Target Version: v{(channel?.version ?? 0) + 1}
                        </Badge>
                      </div>

                      <div className="text-xs text-zinc-300 space-y-1">
                        <p>
                          Candidate:{" "}
                          <span className="font-mono font-semibold text-zinc-100">
                            {candidate.name}
                          </span>
                        </p>
                        <p className="text-zinc-400">
                          Promoting will set this candidate as the active model and increment the channel to{" "}
                          <span className="font-mono text-sky-300">
                            v{(channel?.version ?? 0) + 1}
                          </span>
                          .
                        </p>
                      </div>

                      <div className="space-y-1.5">
                        <label className="text-[11px] font-medium text-zinc-400">
                          Reason for Audit Log:
                        </label>
                        <div className="flex flex-col sm:flex-row gap-2">
                          <input
                            type="text"
                            placeholder="Reason for audit log (e.g. verified on calibration split)"
                            value={promoteReason}
                            onChange={(e) => setPromoteReason(e.target.value)}
                            disabled={busy}
                            className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-500 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:opacity-50"
                          />
                          <div className="flex items-center gap-2 shrink-0">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-8 text-xs"
                              onClick={() => setShowPromoteConfirm(false)}
                              disabled={busy}
                            >
                              Cancel
                            </Button>
                            <Button
                              variant="default"
                              size="sm"
                              className="h-8 text-xs font-semibold"
                              onClick={() => void promote()}
                              disabled={busy}
                            >
                              {busy ? (
                                <Loader2 className="size-3 animate-spin mr-1" />
                              ) : null}
                              Confirm Promotion
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Per-class AP Comparison Metrics */}
              {ev?.metrics && <MetricsSummary metrics={ev.metrics} />}
            </div>
          ) : (
            <Card className="border-dashed border-zinc-800 bg-zinc-900/30 p-12 text-center">
              <p className="text-xs text-zinc-400">Select a candidate from the left panel to inspect details.</p>
            </Card>
          )}
        </div>
      </div>

      {/* Benchmark Evaluation Confirmation Modal */}
      {evalConfirmCandidate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 p-5 shadow-2xl space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
              <TrendingUp className="size-4 text-sky-400" />
              <span>Confirm Benchmark Evaluation</span>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">
              Run benchmark evaluation suite for candidate{" "}
              <strong className="text-zinc-100 font-mono">
                {evalConfirmCandidate.name}
              </strong>
              ?
            </p>
            <div className="rounded-md border border-zinc-800/80 bg-zinc-950/60 p-3 text-xs space-y-1.5 font-mono">
              <div className="flex justify-between text-zinc-400">
                <span>Runtime:</span>
                <span className="text-zinc-200">{evalConfirmCandidate.runtime}</span>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Artifact:</span>
                <span className="text-zinc-200 truncate max-w-[200px]">
                  {evalConfirmCandidate.artifact_name}
                </span>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Resolution:</span>
                <span className="text-zinc-200">{evalConfirmCandidate.target}px</span>
              </div>
              <div className="flex justify-between text-zinc-400">
                <span>Threads:</span>
                <span className="text-zinc-200">{evalConfirmCandidate.threads}</span>
              </div>
            </div>
            <p className="text-[11px] text-zinc-500">
              This evaluates clean quality, shift robustness, latency parity, and memory limits against the pinned reference baseline.
            </p>
            <div className="flex items-center justify-end gap-2 pt-2 border-t border-zinc-800">
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs"
                onClick={() => setEvalConfirmCandidate(null)}
                disabled={busy}
              >
                Cancel
              </Button>
              <Button
                variant="default"
                size="sm"
                className="h-8 text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white"
                onClick={() => {
                  const cand = evalConfirmCandidate;
                  setEvalConfirmCandidate(null);
                  void evaluate(cand.id);
                }}
                disabled={busy}
              >
                {busy ? <Loader2 className="size-3 animate-spin mr-1" /> : null}
                Start Evaluation
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function parseGateInfo(gateKey: string): { code: string; title: string } {
  const normalized = gateKey.toUpperCase();
  if (normalized.includes("G1") || normalized.includes("QUALITY_CLEAN")) {
    return { code: "G1", title: "Clean Quality (mAP)" };
  }
  if (normalized.includes("G2") || normalized.includes("QUALITY_SHIFT")) {
    return { code: "G2", title: "Scanned Quality Shift" };
  }
  if (normalized.includes("G3") || normalized.includes("CRITICAL_CLASSES")) {
    return { code: "G3", title: "Critical Class Regressions" };
  }
  if (normalized.includes("G4") || normalized.includes("RUNTIME_PARITY")) {
    return { code: "G4", title: "Runtime Parity" };
  }
  if (normalized.includes("G5") || normalized.includes("MEMORY_FIT")) {
    return { code: "G5", title: "Worker Memory Budget" };
  }
  if (normalized.includes("G6") || normalized.includes("PROVENANCE")) {
    return { code: "G6", title: "Artifact Provenance" };
  }
  return { code: gateKey.slice(0, 3), title: gateKey };
}

function getModelArchitecture(candidate: Candidate): { name: string; arch: string } {
  const an = (candidate.artifact_name || "").toLowerCase();
  const repo = (candidate.repo_id || "").toLowerCase();
  const name = (candidate.name || "").toLowerCase();

  if (an.includes("egret") || repo.includes("egret") || name.includes("egret")) {
    return { name: "docling-layout-egret-medium", arch: "D-FINE" };
  }
  if (an.includes("heron") || repo.includes("heron") || name.includes("heron")) {
    return { name: "docling-layout-heron", arch: "RT-DETRv2" };
  }
  return { name: candidate.artifact_name || "Document Layout Model", arch: "Detection" };
}

function getRuntimeDisplay(runtime: string): {
  label: string;
  variant: "default" | "secondary" | "outline";
} {
  const r = runtime.toLowerCase();
  if (r.includes("onnx")) {
    return { label: "ONNX Runtime", variant: "secondary" };
  }
  if (r.includes("torch")) {
    return { label: "PyTorch", variant: "outline" };
  }
  return { label: runtime, variant: "outline" };
}

function MetricsSummary({ metrics }: { metrics: Record<string, unknown> | null }): ReactElement | null {
  if (!metrics) return null;
  const clean = (metrics["clean"] ?? {}) as Record<string, Record<string, unknown>>;
  const ref = clean["reference"] ?? {};
  const cand = clean["candidate"] ?? {};
  const perClass = (cand["per_class_ap"] ?? {}) as Record<string, number>;
  const refClass = (ref["per_class_ap"] ?? {}) as Record<string, number>;
  const rows = Object.keys(perClass).sort();
  if (rows.length === 0) return null;

  return (
    <Card className="border-zinc-800 bg-zinc-900/60 shadow-sm">
      <CardHeader className="p-4 pb-3">
        <CardTitle className="text-sm font-semibold text-zinc-100">
          Per-Class Average Precision (AP@[.5:.95])
        </CardTitle>
        <CardDescription className="text-xs text-zinc-400">
          Comparison between candidate and pinned reference baseline
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="overflow-x-auto rounded-md border border-zinc-800/80 bg-zinc-950/40">
          <table className="w-full text-xs text-left">
            <thead className="bg-zinc-900/60 text-zinc-400 border-b border-zinc-800">
              <tr>
                <th className="py-2 px-3 font-medium">Class</th>
                <th className="py-2 px-3 font-medium text-right font-mono">Candidate</th>
                <th className="py-2 px-3 font-medium text-right font-mono">Reference</th>
                <th className="py-2 px-3 font-medium text-right font-mono">Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {rows.map((k) => {
                const cVal = perClass[k];
                const rVal = refClass[k];
                const delta = rVal !== undefined && cVal !== undefined ? cVal - rVal : null;
                const isPositive = delta !== null && delta >= 0;

                return (
                  <tr key={k} className="hover:bg-zinc-900/30">
                    <td className="py-2 px-3 font-medium text-zinc-300">{k}</td>
                    <td className="py-2 px-3 text-right font-mono text-zinc-200">
                      {cVal?.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-zinc-400">
                      {rVal?.toFixed(4) ?? "—"}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">
                      {delta === null ? (
                        <span className="text-zinc-500">—</span>
                      ) : (
                        <span
                          className={cn(
                            "font-medium",
                            isPositive ? "text-emerald-400" : "text-red-400",
                          )}
                        >
                          {isPositive ? "+" : ""}
                          {delta.toFixed(4)}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
