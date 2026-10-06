/**
 * Typed API client.
 *
 * Hand-written against the OpenAPI schema rather than generated, because the
 * surface is small and the hand-written version can state the one thing a
 * generated client cannot: that a 404 for another workspace's id and a 404 for a
 * missing id are the same answer, and that this client never asks the server
 * which workspace it is in.
 */

export type Role = "viewer" | "operator" | "owner";

export interface Session {
  workspace_id: string;
  token: string;
  expires_at: string;
  mode: string;
}

export interface Detection {
  class_id: number;
  label: string;
  score: number;
  box: [number, number, number, number];
}

export interface TimingSplit {
  queue_inclusive_ms: number | null;
  service_ms: number | null;
  preprocess_ms: number | null;
  infer_ms: number | null;
  postprocess_ms: number | null;
  persist_ms: number | null;
  batch_wait_ms: number | null;
  peak_rss_bytes: number | null;
}

export interface BoxDiff {
  label: string;
  kind: "added" | "missing" | "matched" | "relabelled";
  box: [number, number, number, number];
  score: number | null;
  shadow_score: number | null;
}

export interface ItemDiff {
  agreement: number;
  added: BoxDiff[];
  missing: BoxDiff[];
  relabelled: BoxDiff[];
}

export interface WorkItem {
  id: string;
  page_id: string;
  candidate_id: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  role: "primary" | "shadow";
  attempts: number;
  failure_code: string | null;
  failure_detail: string | null;
  timings: TimingSplit;
  detections: Detection[];
  shadow_detections: Detection[];
  diff: ItemDiff | null;
  image_url: string | null;
  page_size: [number, number] | null;
  synthetic: boolean;
  provenance: string | null;
}

export interface Batch {
  id: string;
  status: string;
  total_items: number;
  counts: Record<string, number>;
  release_id: string;
  candidate_id: string;
  candidate_name: string;
  synthetic: boolean;
  created_at: string;
  finished_at: string | null;
  cancel_requested_at: string | null;
  timings: TimingSplit | null;
}

export interface Candidate {
  id: string;
  name: string;
  candidate_hash: string;
  artifact_name: string;
  artifact_format: string;
  artifact_sha256: string;
  repo_id: string;
  revision: string;
  runtime: string;
  resize: string;
  target: number;
  threads: number;
  score_threshold: number;
  tile_enabled: boolean;
  valid: boolean;
  invalid_reason: string | null;
}

export interface GateVerdict {
  gate: string;
  passed: boolean;
  observed: unknown;
  threshold: unknown;
  detail: string | null;
}

export interface Evaluation {
  id: string;
  candidate_id: string;
  candidate_name: string;
  reference_id: string | null;
  status: string;
  metrics: Record<string, unknown> | null;
  gates: GateVerdict[] | null;
  created_at: string;
  finished_at: string | null;
  failure_detail: string | null;
}

export interface Release {
  id: string;
  action: "promote" | "rollback" | "seed";
  candidate_id: string;
  candidate_name: string;
  previous_release_id: string | null;
  reason: string;
  actor: string;
  channel_version: number;
  created_at: string;
}

export interface Channel {
  name: string;
  version: number;
  active_release_id: string | null;
  active_candidate_name: string | null;
  history: Release[];
}

export interface PoolStatus {
  pool: string;
  queued: number;
  running: number;
}

export interface SystemStatus {
  mode: string;
  dialect: string;
  queue_depth: PoolStatus[];
  queue_limit: number;
  worker_mem_budget_bytes: number;
  memory_safety_margin: number;
  admission_mispredictions: number;
  host_free_bytes: number | null;
  git_sha: string | null;
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    super(describe(status, body));
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function describe(status: number, body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return `${status}: ${detail}`;
    if (detail && typeof detail === "object" && "detail" in detail) {
      const inner = (detail as { detail: unknown }).detail;
      const code = (detail as { code?: unknown }).code;
      const failed = (detail as { failed?: unknown }).failed;
      const parts = [typeof inner === "string" ? inner : JSON.stringify(inner)];
      if (code) parts.push(`code=${String(code)}`);
      if (Array.isArray(failed) && failed.length) parts.push(`failed=${failed.join(",")}`);
      return `${status}: ${parts.join(" ")}`;
    }
    return `${status}: ${JSON.stringify(detail)}`;
  }
  return `${status}`;
}

const TOKEN_KEY = "forgesight.token";
const WORKSPACE_KEY = "forgesight.workspace";

export function storedToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function storedWorkspace(): string | null {
  return localStorage.getItem(WORKSPACE_KEY);
}

export function storeSession(s: Session): void {
  // The token is held in localStorage so a reload keeps the session. That is a
  // deliberate trade for a single-user local tool; a public deployment would put
  // this behind httpOnly cookies and a CSRF token instead.
  localStorage.setItem(TOKEN_KEY, s.token);
  localStorage.setItem(WORKSPACE_KEY, s.workspace_id);
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(WORKSPACE_KEY);
}

export function authHeaders(): Record<string, string> {
  const token = storedToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function getApiBase(): string {
  const custom = localStorage.getItem("forgesight.api_url");
  if (custom) return custom.replace(/\/+$/, "");
  if (import.meta.env.VITE_API_URL) {
    return String(import.meta.env.VITE_API_URL).replace(/\/+$/, "");
  }
  if (typeof window !== "undefined" && window.location.hostname.endsWith("github.io")) {
    return "https://forgesight-production-6891.up.railway.app";
  }
  return "";
}

export function resolveApiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = getApiBase();
  return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

/** Set once per page load so a bad token cannot cause a reload loop. */
const RELOAD_GUARD = "forgesight.reloaded-for-auth";

async function request<T>(
  path: string,
  init: RequestInit & { auth?: boolean } = {},
): Promise<T> {
  const token = storedToken();
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  if (init.auth !== false && token) headers.Authorization = `Bearer ${token}`;

  const url = resolveApiUrl(path);
  const res = await fetch(url, { ...init, headers });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = text;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      /* a non-JSON body is surfaced as-is */
    }
  }

  if (res.status === 401 && token) {
    // The stored token no longer works: it expired, or the database it belonged
    // to was reset. Drop it and reload so the app mints a fresh session instead
    // of showing a permanently broken page. Guarded, because reloading into
    // another 401 would never terminate.
    if (!sessionStorage.getItem(RELOAD_GUARD)) {
      sessionStorage.setItem(RELOAD_GUARD, "1");
      clearSession();
      location.reload();
    } else {
      clearSession();
    }
  }

  if (!res.ok) throw new ApiError(res.status, body);
  return body as T;
}

export const api = {
  info: () => request<Record<string, unknown>>("/v1/system/info", { auth: false }),

  createSession: () =>
    request<Session>("/v1/sessions", { method: "POST", auth: false }),

  listBatches: () => request<{ items: Batch[]; next_cursor: string | null }>("/v1/batches"),

  getBatch: (id: string) => request<Batch>(`/v1/batches/${id}`),

  listItems: (batchId: string) => request<WorkItem[]>(`/v1/batches/${batchId}/items`),

  getItem: (id: string) => request<WorkItem>(`/v1/items/${id}`),

  cancelBatch: (id: string) => request<Batch>(`/v1/batches/${id}/cancel`, { method: "POST" }),

  uploadBatch: (files: File[], shadowCandidateId: string | null, idempotencyKey: string) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    const q = shadowCandidateId ? `?shadow_candidate_id=${encodeURIComponent(shadowCandidateId)}` : "";
    return request<Batch>(`/v1/batches${q}`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: form,
    });
  },

  listCandidates: () => request<Candidate[]>("/v1/candidates"),

  evaluate: (candidateId: string) =>
    request<Evaluation>(`/v1/candidates/${candidateId}/evaluate`, { method: "POST" }),

  getEvaluation: (id: string) => request<Evaluation>(`/v1/evaluations/${id}`),

  releases: () => request<Channel>("/v1/releases"),

  promote: (candidateId: string, expectedChannelVersion: number, reason: string) =>
    request<{ channel: Channel; release: Release }>("/v1/releases/promote", {
      method: "POST",
      body: JSON.stringify({
        candidate_id: candidateId,
        expected_channel_version: expectedChannelVersion,
        reason,
      }),
    }),

  rollback: (expectedChannelVersion: number, reason: string) =>
    request<{ channel: Channel; release: Release }>("/v1/releases/rollback", {
      method: "POST",
      body: JSON.stringify({ expected_channel_version: expectedChannelVersion, reason }),
    }),

  systemStatus: () => request<SystemStatus>("/v1/system/status"),
};
