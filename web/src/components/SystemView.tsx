import type { ReactElement } from "react";
/** System view: queue depth, budgets, environment, and host telemetry. */

import { useEffect, useState } from "react";
import {
  Server,
  Database,
  HardDrive,
  Activity,
  Cpu,
  Layers,
  GitCommit,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Info,
} from "lucide-react";
import { api } from "../api/client";
import type { SystemStatus } from "../api/client";
import { ErrorBox, bytes, pct } from "./primitives";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";

export function SystemView(): ReactElement {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const s = await api.systemStatus();
        if (!stop) {
          setStatus(s);
          setLastRefreshed(new Date());
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
    <div className="space-y-6">
      {/* Top Header & Polling Indicator */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-zinc-100">
            System Infrastructure & Telemetry
          </h2>
          <p className="text-xs text-zinc-400">
            Real-time worker telemetry, allocation budgets, and execution host health
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <span className="flex items-center gap-1.5 bg-zinc-900/80 border border-zinc-800 rounded-md px-2.5 py-1">
            <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-[11px] font-mono text-zinc-300">Live 2s polling</span>
          </span>
          <span className="text-zinc-500 text-[11px] font-mono">
            Updated {lastRefreshed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
        </div>
      </div>

      <ErrorBox error={error} />

      {/* 4 Executive Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Card 1: Runtime Mode & Engine */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-md backdrop-blur-sm flex flex-col justify-between">
          <div>
            <CardHeader className="p-5 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="flex size-9 items-center justify-center rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
                    <Server className="size-4.5" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold text-zinc-100">
                      Runtime Mode & Engine
                    </CardTitle>
                    <CardDescription className="text-xs text-zinc-400">
                      Inference orchestration & persistent storage
                    </CardDescription>
                  </div>
                </div>
                <div className="flex size-7 items-center justify-center rounded-md bg-zinc-800/60 text-zinc-400">
                  <Database className="size-3.5" />
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-5 pt-2 space-y-4">
              {status ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Runtime Mode
                      </span>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={status.mode === "live" ? "success" : "info"}
                          className="font-mono text-xs uppercase"
                        >
                          {status.mode}
                        </Badge>
                      </div>
                    </div>

                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Database Engine
                      </span>
                      <span className="font-mono text-xs font-semibold text-zinc-200 uppercase">
                        {status.dialect}
                      </span>
                    </div>
                  </div>

                  <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 flex items-center justify-between">
                    <div>
                      <span className="text-xs text-zinc-400 block font-medium">
                        Global Queue Limit
                      </span>
                      <span className="text-[11px] text-zinc-500">
                        Maximum admitted pending requests
                      </span>
                    </div>
                    <span className="font-mono text-sm font-semibold text-zinc-200">
                      {status.queue_limit} items
                    </span>
                  </div>

                  {Boolean(info?.synthetic_demo_data) && (
                    <div className="flex items-center gap-2 rounded-md border border-purple-500/20 bg-purple-500/10 px-3 py-2 text-xs text-purple-300">
                      <Info className="size-3.5 shrink-0 text-purple-400" />
                      <span>Synthetic demo dataset generator active</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center py-8 text-zinc-500 text-xs">
                  <RefreshCw className="size-4 animate-spin mr-2" /> Loading runtime metrics…
                </div>
              )}
            </CardContent>
          </div>
        </Card>

        {/* Card 2: Memory Budget & Safety Margin */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-md backdrop-blur-sm flex flex-col justify-between">
          <div>
            <CardHeader className="p-5 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="flex size-9 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <HardDrive className="size-4.5" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold text-zinc-100">
                      Memory Budget & Safety Margin
                    </CardTitle>
                    <CardDescription className="text-xs text-zinc-400">
                      Worker RSS headroom and host capacity limits
                    </CardDescription>
                  </div>
                </div>
                <div className="flex size-7 items-center justify-center rounded-md bg-zinc-800/60 text-zinc-400">
                  <Activity className="size-3.5" />
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-5 pt-2 space-y-4">
              {status ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Worker Memory Budget
                      </span>
                      <span className="font-mono text-sm font-semibold text-zinc-200">
                        {bytes(status.worker_mem_budget_bytes)}
                      </span>
                    </div>

                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Safety Margin
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-sm font-semibold text-emerald-400">
                          {pct(status.memory_safety_margin, 0)}
                        </span>
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-zinc-400">
                          Headroom
                        </Badge>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Host Free RAM
                      </span>
                      <span className="font-mono text-sm font-semibold text-zinc-200">
                        {bytes(status.host_free_bytes)}
                      </span>
                    </div>

                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Mispredictions
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-semibold text-zinc-200">
                          {status.admission_mispredictions}
                        </span>
                        {status.admission_mispredictions === 0 ? (
                          <Badge variant="success" className="text-[10px] px-1.5 py-0">
                            Zero
                          </Badge>
                        ) : (
                          <Badge variant="warning" className="text-[10px] px-1.5 py-0">
                            Elevated
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-8 text-zinc-500 text-xs">
                  <RefreshCw className="size-4 animate-spin mr-2" /> Loading memory metrics…
                </div>
              )}
            </CardContent>
          </div>
        </Card>

        {/* Card 3: Worker Pools & Queue Depths */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-md backdrop-blur-sm flex flex-col justify-between">
          <div>
            <CardHeader className="p-5 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="flex size-9 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    <Cpu className="size-4.5" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold text-zinc-100">
                      Worker Pools & Queue Depths
                    </CardTitle>
                    <CardDescription className="text-xs text-zinc-400">
                      Process isolation across execution pools
                    </CardDescription>
                  </div>
                </div>
                <div className="flex size-7 items-center justify-center rounded-md bg-zinc-800/60 text-zinc-400">
                  <Layers className="size-3.5" />
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-5 pt-2 space-y-3">
              {status ? (
                status.queue_depth.length > 0 ? (
                  <div className="space-y-2">
                    <div className="overflow-x-auto rounded-lg border border-zinc-800/80 bg-zinc-950/60">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-zinc-900/70 text-zinc-400 border-b border-zinc-800">
                          <tr>
                            <th className="py-2 px-3 font-medium">Pool Identifier</th>
                            <th className="py-2 px-3 font-medium text-right">Queued</th>
                            <th className="py-2 px-3 font-medium text-right">Running</th>
                            <th className="py-2 px-3 font-medium text-right">State</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/50">
                          {status.queue_depth.map((q) => {
                            const isActive = q.running > 0;
                            const hasQueue = q.queued > 0;

                            return (
                              <tr key={q.pool} className="hover:bg-zinc-900/40">
                                <td className="py-2 px-3 font-mono font-medium text-zinc-200">
                                  {q.pool}
                                </td>
                                <td className="py-2 px-3 font-mono text-right">
                                  <span
                                    className={cn(
                                      hasQueue ? "text-amber-400 font-semibold" : "text-zinc-500",
                                    )}
                                  >
                                    {q.queued}
                                  </span>
                                </td>
                                <td className="py-2 px-3 font-mono text-right">
                                  <span
                                    className={cn(
                                      isActive ? "text-sky-400 font-semibold" : "text-zinc-500",
                                    )}
                                  >
                                    {q.running}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-right">
                                  {isActive ? (
                                    <Badge
                                      variant="info"
                                      className="text-[10px] font-mono px-1.5 py-0"
                                    >
                                      Processing
                                    </Badge>
                                  ) : hasQueue ? (
                                    <Badge
                                      variant="warning"
                                      className="text-[10px] font-mono px-1.5 py-0"
                                    >
                                      Queued
                                    </Badge>
                                  ) : (
                                    <Badge
                                      variant="outline"
                                      className="text-[10px] font-mono px-1.5 py-0 text-zinc-500"
                                    >
                                      Idle
                                    </Badge>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <p className="text-[11px] text-zinc-500 leading-tight">
                      One worker process per pool ensures peak RSS is isolated to a single runtime.
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500 italic py-4 text-center">
                    No active worker pools registered.
                  </p>
                )
              ) : (
                <div className="flex items-center justify-center py-8 text-zinc-500 text-xs">
                  <RefreshCw className="size-4 animate-spin mr-2" /> Loading pool telemetry…
                </div>
              )}
            </CardContent>
          </div>
        </Card>

        {/* Card 4: Host Telemetry & Git SHA */}
        <Card className="border-zinc-800 bg-zinc-900/60 shadow-md backdrop-blur-sm flex flex-col justify-between">
          <div>
            <CardHeader className="p-5 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="flex size-9 items-center justify-center rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <GitCommit className="size-4.5" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold text-zinc-100">
                      Host Telemetry & Git SHA
                    </CardTitle>
                    <CardDescription className="text-xs text-zinc-400">
                      Deployment release fingerprint & host metadata
                    </CardDescription>
                  </div>
                </div>
                <div className="flex size-7 items-center justify-center rounded-md bg-zinc-800/60 text-zinc-400">
                  <ShieldCheck className="size-3.5" />
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-5 pt-2 space-y-4">
              {status ? (
                <div className="space-y-3">
                  <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                      Release Commit SHA
                    </span>
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-semibold text-sky-400 tracking-wider">
                        {(status.git_sha ?? "unknown").slice(0, 12)}
                      </span>
                      <Badge variant="outline" className="font-mono text-[10px] text-zinc-400">
                        {status.git_sha ? status.git_sha.slice(0, 7) : "HEAD"}
                      </Badge>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        API Gateway
                      </span>
                      <div className="flex items-center gap-1.5">
                        <CheckCircle2 className="size-3.5 text-emerald-400" />
                        <span className="text-xs font-medium text-zinc-200">v0.2.0 (FastAPI)</span>
                      </div>
                    </div>

                    <div className="rounded-lg bg-zinc-950/60 border border-zinc-800/80 p-3 space-y-1">
                      <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block">
                        Host Status
                      </span>
                      <div className="flex items-center gap-1.5">
                        <Badge variant="success" className="text-[10px] px-1.5 py-0">
                          HEALTHY
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {Array.isArray(info?.notes) && info.notes.length > 0 && (
                    <div className="rounded-lg bg-zinc-950/40 border border-zinc-800/60 p-3 text-[11px] text-zinc-400 space-y-1">
                      <span className="font-medium text-zinc-300 block">Deployment Notes:</span>
                      <ul className="list-disc list-inside space-y-0.5 text-zinc-400">
                        {info.notes.map((note, idx) => (
                          <li key={idx} className="line-clamp-2">
                            {String(note)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center py-8 text-zinc-500 text-xs">
                  <RefreshCw className="size-4 animate-spin mr-2" /> Loading host telemetry…
                </div>
              )}
            </CardContent>
          </div>
        </Card>
      </div>
    </div>
  );
}
