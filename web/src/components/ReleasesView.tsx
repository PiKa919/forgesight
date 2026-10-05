import type { ReactElement } from "react";
/** Releases view: candidates, gate verdicts, promote and roll back. */

import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Candidate, Channel, Evaluation, GateVerdict } from "../api/client";
import { ErrorBox, StatusPill, bytes } from "./primitives";

export function ReleasesView({ onChanged }: { onChanged: () => void }): ReactElement {
  const [channel, setChannel] = useState<Channel | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [evaluations, setEvaluations] = useState<Record<string, Evaluation>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [note, setNote] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [ch, cs] = await Promise.all([api.releases(), api.listCandidates()]);
      setChannel(ch);
      setCandidates(cs);
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
    setError(null);
    setNote(null);
    try {
      const ev = await api.evaluate(id);
      setEvaluations((prev) => ({ ...prev, [id]: ev }));
      setSelected(id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const promote = async () => {
    if (!channel || !selected) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const r = await api.promote(selected, channel.version, reason || "promoted from the UI");
      setChannel(r.channel);
      setNote(`promoted ${r.release.candidate_name} as version ${r.channel.version}`);
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
      const r = await api.rollback(channel.version, reason || "rolled back from the UI");
      setChannel(r.channel);
      setNote(`rolled back to ${r.release.candidate_name}`);
      onChanged();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const ev = selected ? evaluations[selected] : undefined;
  const candidate = candidates.find((c) => c.id === selected) ?? null;

  return (
    <div className="grid two">
      <div className="stack">
        <div className="panel">
          <h2>Channel</h2>
          {channel ? (
            <>
              <div className="row wrap">
                <span className="pill info">v{channel.version}</span>
                <span className="dim">active</span>
                <span className="mono">{channel.active_candidate_name ?? "none"}</span>
                <div className="spacer" />
                <button
                  className="danger"
                  onClick={() => void rollback()}
                  disabled={busy || !channel.active_release_id}
                >
                  roll back
                </button>
              </div>
              <h3>History (append-only)</h3>
              <table>
                <thead>
                  <tr>
                    <th>when</th>
                    <th>action</th>
                    <th>candidate</th>
                    <th className="num">v</th>
                  </tr>
                </thead>
                <tbody>
                  {channel.history.map((h) => (
                    <tr key={h.id}>
                      <td className="faint">
                        {new Date(h.created_at).toLocaleTimeString()}
                      </td>
                      <td>
                        <StatusPill status={h.action} />
                      </td>
                      <td className="mono">{h.candidate_name}</td>
                      <td className="num">{h.channel_version}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p className="faint">loading…</p>
          )}
        </div>
        <ErrorBox error={error} />
        {note ? <div className="notice">{note}</div> : null}
      </div>

      <div className="stack">
        <div className="panel">
          <h2>Candidates</h2>
          <table>
            <thead>
              <tr>
                <th>name</th>
                <th>runtime</th>
                <th>preprocess</th>
                <th className="num">threads</th>
                <th className="num">threshold</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr
                  key={c.id}
                  onClick={() => setSelected(c.id)}
                  style={{
                    cursor: "pointer",
                    background: c.id === selected ? "var(--panel-2)" : undefined,
                  }}
                >
                  <td className="mono">
                    {c.name}
                    {c.artifact_name.startsWith("negctl") ? (
                      <span className="pill bad" style={{ marginLeft: 6 }}>
                        negative control
                      </span>
                    ) : null}
                  </td>
                  <td>{c.runtime}</td>
                  <td>
                    {c.resize} → {c.target}px
                    {c.tile_enabled ? " · tiled" : ""}
                  </td>
                  <td className="num">{c.threads}</td>
                  <td className="num">{c.score_threshold.toFixed(2)}</td>
                  <td>
                    <button onClick={(e) => { e.stopPropagation(); void evaluate(c.id); }} disabled={busy}>
                      evaluate
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="faint">
            Gates compare against the <span className="mono">ref-heron-torch</span>{" "}
            reference, never against whatever is currently active — otherwise a run of
            individually permitted regressions would accumulate unnoticed.
          </p>
        </div>

        <div className="panel">
          <h2>Promotion</h2>
          {candidate ? (
            <>
              <div className="mono faint">{candidate.candidate_hash.slice(0, 32)}…</div>
              <div className="faint">
                {candidate.artifact_name} · {candidate.artifact_format} ·{" "}
                {candidate.repo_id}@{candidate.revision.slice(0, 12)} ·{" "}
                {bytes(0)} artifact
              </div>
              {ev ? <GateTable gates={ev.gates ?? []} /> : (
                <p className="hint">
                  no evaluation for this candidate yet — a candidate cannot be promoted
                  without one, because an unevaluated candidate is indistinguishable from
                  a passing one.
                </p>
              )}
              <div className="row" style={{ marginTop: 10 }}>
                <input
                  placeholder="reason for the audit log"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
                <button
                  className="primary"
                  onClick={() => void promote()}
                  disabled={busy || !channel}
                >
                  promote
                </button>
              </div>
              {ev ? <Metrics metrics={ev.metrics} /> : null}
            </>
          ) : (
            <p className="hint">select a candidate</p>
          )}
        </div>
      </div>
    </div>
  );
}

function GateTable({ gates }: { gates: GateVerdict[] }): ReactElement | null {
  if (gates.length === 0) return null;
  return (
    <table>
      <thead>
        <tr>
          <th>gate</th>
          <th>result</th>
          <th className="num">observed</th>
          <th className="num">threshold</th>
        </tr>
      </thead>
      <tbody>
        {gates.map((g) => (
          <tr key={g.gate}>
            <td className="mono">
              {g.gate}
              {g.detail ? <div className="faint">{g.detail}</div> : null}
            </td>
            <td>
              <span className={`pill ${g.passed ? "ok" : "bad"}`}>
                {g.passed ? "pass" : "block"}
              </span>
            </td>
            <td className="num">{fmt(g.observed)}</td>
            <td className="num">{fmt(g.threshold)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Metrics({ metrics }: { metrics: Record<string, unknown> | null }): ReactElement | null {
  if (!metrics) return null;
  const clean = (metrics["clean"] ?? {}) as Record<string, Record<string, unknown>>;
  const ref = clean["reference"] ?? {};
  const cand = clean["candidate"] ?? {};
  const perClass = (cand["per_class_ap"] ?? {}) as Record<string, number>;
  const refClass = (ref["per_class_ap"] ?? {}) as Record<string, number>;
  const rows = Object.keys(perClass).sort();
  if (rows.length === 0) return null;
  return (
    <>
      <h3>Per-class AP (candidate vs reference)</h3>
      <table>
        <thead>
          <tr>
            <th>class</th>
            <th className="num">candidate</th>
            <th className="num">reference</th>
            <th className="num">delta</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((k) => (
            <tr key={k}>
              <td>{k}</td>
              <td className="num">{perClass[k]?.toFixed(4)}</td>
              <td className="num">{refClass[k]?.toFixed(4) ?? "—"}</td>
              <td className="num">
                {refClass[k] === undefined
                  ? "—"
                  : (perClass[k]! - refClass[k]).toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="faint">
        operating threshold {(metrics["operating_threshold"] as number)?.toFixed(2)},
        chosen on the calibration split and then frozen. Absolute AP on generated
        pages is not comparable with published numbers.
      </p>
    </>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(5);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
