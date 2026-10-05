import type { ReactElement } from "react";
/** System view: queue depth, budgets, environment, and the recorded reports. */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SystemStatus } from "../api/client";
import { ErrorBox, bytes, pct } from "./primitives";

export function SystemView(): ReactElement {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const s = await api.systemStatus();
        if (!stop) {
          setStatus(s);
          setError(null);
        }
      } catch (e) {
        if (!stop) setError(e);
      }
    };
    void load();
    const t = setInterval(() => void load(), 2000);
    void api.info().then((i) => !stop && setInfo(i));
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  return (
    <div className="grid two">
      <div className="panel">
        <h2>Status</h2>
        {status ? (
          <table>
            <tbody>
              <tr>
                <td>mode</td>
                <td className="mono">{status.mode}</td>
              </tr>
              <tr>
                <td>database</td>
                <td className="mono">{status.dialect}</td>
              </tr>
              <tr>
                <td>queue limit</td>
                <td className="num">{status.queue_limit}</td>
              </tr>
              <tr>
                <td>worker memory budget</td>
                <td className="num">{bytes(status.worker_mem_budget_bytes)}</td>
              </tr>
              <tr>
                <td>safety margin</td>
                <td className="num">{pct(status.memory_safety_margin, 0)}</td>
              </tr>
              <tr>
                <td>host memory free</td>
                <td className="num">{bytes(status.host_free_bytes)}</td>
              </tr>
              <tr>
                <td>admission mispredictions</td>
                <td className="num">{status.admission_mispredictions}</td>
              </tr>
              <tr>
                <td>git sha</td>
                <td className="mono">{(status.git_sha ?? "unknown").slice(0, 12)}</td>
              </tr>
            </tbody>
          </table>
        ) : (
          <p className="faint">loading…</p>
        )}
      </div>

      <div className="stack">
        <div className="panel">
          <h2>Queue depth</h2>
          {status ? (
            <table>
              <thead>
                <tr>
                  <th>pool</th>
                  <th className="num">queued</th>
                  <th className="num">running</th>
                </tr>
              </thead>
              <tbody>
                {status.queue_depth.map((q) => (
                  <tr key={q.pool}>
                    <td className="mono">{q.pool}</td>
                    <td className="num">{q.queued}</td>
                    <td className="num">{q.running}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          <p className="faint">
            One worker process per pool, so each pool's peak RSS is attributable to one
            runtime rather than to two allocators sharing a heap.
          </p>
        </div>

        <div className="panel">
          <h2>What this deployment is</h2>
          {info ? (
            <>
              <p className="faint">{(info["notes"] as string[] | undefined)?.join(" ")}</p>
              {info["synthetic_demo_data"] ? (
                <span className="pill synthetic">synthetic demo data</span>
              ) : null}
            </>
          ) : (
            <p className="faint">loading…</p>
          )}
        </div>

        <ErrorBox error={error} />
      </div>
    </div>
  );
}
