"""Workspace-scoped repositories.

Every query here filters on `workspace_id`, and no repository method accepts a
workspace id from the caller: the `ws` argument is always the authenticated
principal's workspace. There is deliberately no "get by id without a workspace"
method, because one would be the natural place for a cross-workspace leak to
enter later.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from forgesight.db.pool import Dialect, PoolLike, ph
from forgesight.ledger.claims import new_id
from forgesight.vision.types import Pool, WorkItemState


def _iso(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, datetime):
        return v.astimezone(UTC).isoformat()
    return str(v)


#: SQLite has no timestamp type: CURRENT_TIMESTAMP comes back as the string
#: "YYYY-MM-DD HH:MM:SS" in UTC, while psycopg returns aware datetimes. Every
#: duration therefore goes through this, or the queue-inclusive and service
#: figures silently come back as None on the SQLite backend.
_SQLITE_TS = "%Y-%m-%d %H:%M:%S"


def as_datetime(value):
    """Coerce a database timestamp to an aware datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in (_SQLITE_TS, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def delta_ms(a, b) -> float | None:
    """Milliseconds from `a` to `b`, or None if either is missing."""
    start, end = as_datetime(a), as_datetime(b)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() * 1000.0, 2)


def _ms(a, b) -> float | None:
    return delta_ms(a, b)


class WorkspaceRepo:
    def __init__(self, pool: PoolLike):
        self.pool = pool
        self.d = pool.dialect
        self.ph = ph(self.d)

    def _p(self, n: int) -> str:
        return ", ".join([self.ph] * n)

    # -- workspaces and tokens -------------------------------------------
    def create_workspace(self, name: str, kind: str = "local", ttl_h: int | None = None):
        wid = new_id("ws")
        expires_at = None if ttl_h is None else _shift(datetime.now(UTC), ttl_h)
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO workspace(id, name, kind, expires_at) VALUES ({self._p(4)})",
                (wid, name, kind, expires_at),
            )
        return wid

    def add_token(self, workspace_id: str, token_hash: str, role: str, ttl_h: int):
        tid = new_id("tok")
        expires_at = _shift(datetime.now(UTC), ttl_h)
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO api_token(id, workspace_id, token_hash, role, expires_at) "
                f"VALUES ({self._p(5)})",
                (tid, workspace_id, token_hash, role, expires_at),
            )
        return tid

    def sandbox_count(self) -> int:
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT COUNT(*) AS n FROM workspace WHERE kind = {self.ph}", ("sandbox",)
            )
        return int(row["n"])

    # -- assets and pages -------------------------------------------------
    def find_asset(self, ws: str, sha256: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT * FROM asset WHERE workspace_id = {self.ph} AND sha256 = {self.ph}",
                (ws, sha256),
            )

    def insert_asset(
        self,
        ws: str,
        sha256: str,
        byte_size: int,
        media_type: str,
        object_key: str,
        page_count: int,
        provenance: str,
        synthetic: bool,
        license_: str | None = None,
    ) -> str:
        aid = new_id("as")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO asset(id, workspace_id, sha256, byte_size, media_type, "
                f"object_key, page_count, provenance, synthetic, license) "
                f"VALUES ({self._p(10)})",
                (aid, ws, sha256, byte_size, media_type, object_key, page_count,
                 provenance, synthetic, license_),
            )
        return aid

    def insert_page(
        self,
        ws: str,
        asset_id: str,
        page_index: int,
        width: int,
        height: int,
        dpi: int,
        object_key: str,
        normalized: str | None = None,
    ) -> str:
        """Insert a page row, or return the existing one for the same content.

        Idempotent on purpose. A page is content, not an event: uploading the
        same file into a second batch must reuse the row rather than collide
        with the UNIQUE(workspace_id, asset_id, page_index) constraint. Two
        batches referencing the same page is exactly what that uniqueness is for.
        """
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT id, object_key FROM page WHERE workspace_id = {self.ph} "
                f"AND asset_id = {self.ph} AND page_index = {self.ph}",
                (ws, asset_id, page_index),
            )
        if row is not None:
            # The stored render wins even if this upload would render
            # differently, because it is what predictions may already have been
            # computed from. Replacing it would invalidate them silently.
            return row["id"]
        pid = new_id("pg")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO page(id, workspace_id, asset_id, page_index, width_px, "
                f"height_px, render_dpi, object_key, normalized) VALUES ({self._p(9)})",
                (pid, ws, asset_id, page_index, width, height, dpi, object_key, normalized),
            )
        return pid

    def get_page(self, ws: str, page_id: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT p.*, a.sha256 AS page_sha, a.provenance, a.synthetic "
                f"FROM page p JOIN asset a ON a.id = p.asset_id "
                f"AND a.workspace_id = p.workspace_id "
                f"WHERE p.workspace_id = {self.ph} AND p.id = {self.ph}",
                (ws, page_id),
            )

    # -- catalog ----------------------------------------------------------
    def insert_candidate(self, ws: str, name: str, candidate_hash: str, artifact: Any,
                         preprocess_hash: str, runtime_hash: str, threshold: float,
                         label_map_hash: str, tile_enabled: bool) -> str:
        cid = new_id("cand")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO candidate(id, workspace_id, name, candidate_hash, "
                f"artifact_id, preprocess_id, runtime_id, score_threshold, "
                f"label_map_hash, tile_enabled) VALUES ({self._p(10)})",
                (cid, ws, name, candidate_hash, artifact.id, preprocess_hash,
                 runtime_hash, threshold, label_map_hash, tile_enabled),
            )
        return cid

    def get_candidate(self, ws: str, candidate_id: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT * FROM candidate WHERE workspace_id = {self.ph} AND id = {self.ph}",
                (ws, candidate_id),
            )

    def find_candidate_by_hash(self, ws: str, candidate_hash: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT * FROM candidate WHERE workspace_id = {self.ph} "
                f"AND candidate_hash = {self.ph}",
                (ws, candidate_hash),
            )

    def list_candidates(self, ws: str) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.fetchall(
                f"SELECT * FROM candidate WHERE workspace_id = {self.ph} ORDER BY created_at",
                (ws,),
            )

    def profiles_for(self, ws: str, artifact, candidate) -> tuple[str, str]:
        """Ensure the profile and artifact rows exist for a candidate.

        Profiles live in the owning workspace rather than in a shared sentinel
        workspace. A sentinel would need its own row, and the composite
        foreign keys correctly refuse to invent one -- and per-workspace rows are
        idempotent anyway, because UNIQUE(workspace_id, profile_hash) makes
        re-registering the same configuration a no-op rather than a duplicate.
        """
        pp_id = self._ensure_profile(ws, "preprocess_profile", candidate.preprocess.hash,
                                     json.dumps(candidate.preprocess.as_dict()))
        rt_id = self._ensure_profile(ws, "runtime_profile", candidate.runtime.hash,
                                     json.dumps(candidate.runtime.as_dict()))
        self._ensure_artifact(ws, artifact)
        return pp_id, rt_id

    def _ensure_profile(self, ws: str, table: str, phash: str, body: str) -> str:
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT id FROM {table} WHERE workspace_id = {self.ph} "
                f"AND profile_hash = {self.ph}",
                (ws, phash),
            )
        if row:
            return row["id"]
        pid = new_id(table[:3])
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO {table}(id, workspace_id, profile_hash, body) "
                f"VALUES ({self._p(4)})",
                (pid, ws, phash, body),
            )
        return pid

    def _ensure_artifact(self, ws: str, artifact) -> str:
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT id FROM model_artifact WHERE workspace_id = {self.ph} "
                f"AND sha256 = {self.ph}",
                (ws, artifact.sha256),
            )
        if row:
            return row["id"]
        aid = new_id("art")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO model_artifact(id, workspace_id, name, repo_id, revision, "
                f"format, path, sha256, byte_size, license, parent_sha, exporter, opset) "
                f"VALUES ({self._p(13)})",
                (aid, ws, artifact.name, artifact.repo_id, artifact.revision,
                 artifact.format, artifact.path, artifact.sha256, 0, artifact.license,
                 artifact.parent_sha, artifact.exporter, artifact.opset),
            )
        return aid

    # -- channel and releases ---------------------------------------------
    def ensure_channel(self, ws: str, name: str = "default") -> str:
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT id FROM channel WHERE workspace_id = {self.ph} AND name = {self.ph}",
                (ws, name),
            )
        if row:
            return row["id"]
        cid = new_id("ch")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO channel(id, workspace_id, name, version) VALUES ({self._p(4)})",
                (cid, ws, name, 0),
            )
        return cid

    def channel(self, ws: str, name: str = "default"):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT * FROM channel WHERE workspace_id = {self.ph} AND name = {self.ph}",
                (ws, name),
            )

    def list_releases(self, ws: str, channel_id: str, limit: int = 50) -> list[dict]:
        # rowid is the sqlite tie-break for equal created_at; postgres has no
        # such column, and ties there are broken by the id.
        tail = "r.rowid" if self.d is Dialect.SQLITE else "r.id"
        with self.pool.connection() as conn:
            return conn.fetchall(
                f"SELECT r.*, c.name AS candidate_name FROM release r "
                f"LEFT JOIN candidate c ON c.id = r.candidate_id "
                f"AND c.workspace_id = r.workspace_id "
                f"WHERE r.workspace_id = {self.ph} AND r.channel_id = {self.ph} "
                f"ORDER BY r.created_at DESC, {tail} DESC LIMIT {int(limit)}",
                (ws, channel_id),
            )

    # -- batches ----------------------------------------------------------
    def insert_batch(self, ws: str, release_id: str, shadow_candidate_id: str | None,
                     idempotency_key: str | None, total: int, synthetic: bool) -> str:
        bid = new_id("b")
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO batch(id, workspace_id, release_id, shadow_candidate_id, "
                f"idempotency_key, total_items, synthetic) VALUES ({self._p(7)})",
                (bid, ws, release_id, shadow_candidate_id, idempotency_key, total, synthetic),
            )
        return bid

    def batch_by_key(self, ws: str, idempotency_key: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT * FROM batch WHERE workspace_id = {self.ph} "
                f"AND idempotency_key = {self.ph}",
                (ws, idempotency_key),
            )

    def get_batch(self, ws: str, batch_id: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT b.*, r.candidate_id AS release_candidate_id, "
                f"rc.name AS candidate_name, sc.name AS shadow_candidate_name "
                f"FROM batch b "
                f"JOIN release r ON r.id = b.release_id AND r.workspace_id = b.workspace_id "
                f"LEFT JOIN candidate rc ON rc.id = r.candidate_id "
                f"AND rc.workspace_id = b.workspace_id "
                f"LEFT JOIN candidate sc ON sc.id = b.shadow_candidate_id "
                f"AND sc.workspace_id = b.workspace_id "
                f"WHERE b.workspace_id = {self.ph} AND b.id = {self.ph}",
                (ws, batch_id),
            )

    def enqueue_items(self, ws: str, batch_id: str, rows: list[tuple[str, str, str, str]]):
        """rows: (page_id, candidate_id, pool, role).

        `received_at` is stamped from the application clock rather than by a SQL
        default, for the resolution reason documented on `ledger.claims.now_utc`.
        """
        if not rows:
            return
        now = _now()
        with self.pool.write() as conn:
            for page_id, candidate_id, pool_name, role in rows:
                conn.execute(
                    f"INSERT INTO work_item(id, workspace_id, batch_id, page_id, "
                    f"candidate_id, pool, state, role, enqueued_at, received_at) "
                    f"VALUES ({self._p(10)})",
                    (new_id("wi"), ws, batch_id, page_id, candidate_id, pool_name,
                     WorkItemState.QUEUED.value, role, now, now),
                )

    def refresh_batch(self, batch_id: str) -> None:
        """Re-derive this batch's status from its items.

        The reaper does this on a timer for every batch, but a reader that is
        told "queued" while every item has already failed is being told
        something false. Rolling up on read costs one indexed subquery and makes
        the endpoint honest between reaper ticks.
        """
        from forgesight.ledger.reaper import Reaper
        from forgesight.settings import get_settings

        Reaper(self.pool, get_settings()).refresh_batch_status()

    def batch_counts(self, ws: str, batch_id: str) -> dict[str, int]:
        with self.pool.connection() as conn:
            rows = conn.fetchall(
                f"SELECT state, COUNT(*) AS n FROM work_item "
                f"WHERE workspace_id = {self.ph} AND batch_id = {self.ph} GROUP BY state",
                (ws, batch_id),
            )
        return {r["state"]: int(r["n"]) for r in rows}

    def batch_timings(self, ws: str, batch_id: str) -> dict:
        with self.pool.connection() as conn:
            rows = conn.fetchall(
                f"SELECT received_at, claimed_at, persisted_at, preprocess_ms, "
                f"infer_ms, postprocess_ms, persist_ms, preprocess_start, "
                f"preprocess_end, infer_start, predicted_peak_rss "
                f"FROM work_item WHERE workspace_id = {self.ph} AND batch_id = {self.ph} "
                f"AND state = {self.ph}",
                (ws, batch_id, WorkItemState.SUCCEEDED.value),
            )
        return _aggregate(rows)

    #: A work item's shadow result is the prediction on the SAME PAGE by a
    #: different candidate.
    #:
    #: Joining on `work_item_id` looks right and is always empty. A work item is
    #: created one per (page, candidate) pair, so the shadow candidate's result
    #: lives on a *different* work item that shares `page_id`. The old predicate
    #: -- same work_item_id, different candidate_id -- can therefore never be
    #: true, and `has_shadow` was permanently false, which left the §18 diff view
    #: (added / missing / relabelled boxes) rendering nothing at all. Verified
    #: against a seeded database: the work_item_id join matched 0 rows where the
    #: page_id join matches 8.
    #:
    #: `role = 'shadow'` is the schema's own discriminator and is used as such,
    #: rather than relying on candidate inequality alone. `batch_id` is included
    #: so a diff cannot pair a page against a shadow run from another batch.
    _ITEM_SELECT = """
        SELECT w.*, pr.detections AS primary_detections, pr.timings AS primary_timings,
               pr.id IS NOT NULL AS has_primary,
               sr.detections AS shadow_detections, sr.id IS NOT NULL AS has_shadow
        FROM work_item w
        LEFT JOIN prediction pr
               ON pr.work_item_id = w.id AND pr.workspace_id = w.workspace_id
              AND pr.candidate_id = w.candidate_id
        LEFT JOIN work_item s
               ON s.page_id = w.page_id AND s.workspace_id = w.workspace_id
              AND s.batch_id = w.batch_id AND s.role = 'shadow'
              AND s.candidate_id <> w.candidate_id
        LEFT JOIN prediction sr
               ON sr.work_item_id = s.id
              AND sr.workspace_id = w.workspace_id
              AND sr.candidate_id = s.candidate_id
        WHERE w.workspace_id = {ph} AND w.batch_id = {ph}
    """

    def list_items(self, ws: str, batch_id: str, after: str | None, limit: int = 100):
        # LIMIT is inlined as a clamped integer, not bound, so it must not also
        # appear in the parameter tuple.
        cap = max(1, min(int(limit), 500))
        if after:
            clause = f" AND w.id > {self.ph}"
            params: tuple = (ws, batch_id, after)
        else:
            clause = ""
            params = (ws, batch_id)
        with self.pool.connection() as conn:
            return conn.fetchall(
                self._ITEM_SELECT.format(ph=self.ph) + f"{clause} ORDER BY w.id LIMIT {cap}",
                params,
            )

    def get_item(self, ws: str, item_id: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                "SELECT w.*, pr.detections AS primary_detections, "
                "pr.timings AS primary_timings, "
                "sr.detections AS shadow_detections "
                "FROM work_item w "
                "LEFT JOIN prediction pr ON pr.work_item_id = w.id "
                "AND pr.candidate_id = w.candidate_id AND pr.workspace_id = w.workspace_id "
                "LEFT JOIN work_item s ON s.page_id = w.page_id "
                "AND s.workspace_id = w.workspace_id "
                "AND s.role = 'shadow' "
                "AND s.candidate_id <> w.candidate_id "
                "LEFT JOIN prediction sr ON sr.work_item_id = s.id "
                "AND sr.workspace_id = w.workspace_id "
                "AND sr.candidate_id = s.candidate_id "
                f"WHERE w.workspace_id = {self.ph} AND w.id = {self.ph}",
                (ws, item_id),
            )

    def queue_depth(self) -> dict[str, dict[str, int]]:
        with self.pool.connection() as conn:
            rows = conn.fetchall(
                "SELECT pool, state, COUNT(*) AS n FROM work_item "
                "GROUP BY pool, state"
            )
        out: dict[str, dict[str, int]] = {p.value: {"queued": 0, "running": 0}
                                          for p in Pool}
        for r in rows:
            if r["state"] in out.setdefault(r["pool"], {}):
                out[r["pool"]][r["state"]] = int(r["n"])
        return out

    def mispredictions(self, ws: str) -> int:
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT COUNT(*) AS n FROM work_item WHERE workspace_id = {self.ph} "
                f"AND failure_code = {self.ph}",
                (ws, "admission_misprediction"),
            )
        return int(row["n"])

    def list_datasets(self, ws: str) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.fetchall(
                f"SELECT * FROM dataset_version WHERE workspace_id = {self.ph} "
                f"ORDER BY name, semver",
                (ws,),
            )

    def get_dataset(self, ws: str, name: str, semver: str | None = None):
        with self.pool.connection() as conn:
            if semver:
                return conn.fetchone(
                    f"SELECT * FROM dataset_version WHERE workspace_id = {self.ph} "
                    f"AND name = {self.ph} AND semver = {self.ph}",
                    (ws, name, semver),
                )
            return conn.fetchone(
                f"SELECT * FROM dataset_version WHERE workspace_id = {self.ph} "
                f"AND name = {self.ph} ORDER BY created_at DESC LIMIT 1",
                (ws, name),
            )

    def get_evaluation(self, ws: str, evaluation_id: str):
        with self.pool.connection() as conn:
            return conn.fetchone(
                f"SELECT e.*, c.name AS candidate_name FROM evaluation_run e "
                f"LEFT JOIN candidate c ON c.id = e.candidate_id "
                f"AND c.workspace_id = e.workspace_id "
                f"WHERE e.workspace_id = {self.ph} AND e.id = {self.ph}",
                (ws, evaluation_id),
            )


def _now() -> datetime:
    return datetime.now(UTC)


def _shift(base: datetime, hours: int) -> datetime:
    from datetime import timedelta

    return base + timedelta(hours=hours)


def _aggregate(rows: list[dict]) -> dict:
    qi = [_ms(r.get("received_at"), r.get("persisted_at")) for r in rows]
    svc = [_ms(r.get("claimed_at"), r.get("persisted_at")) for r in rows]
    qi = [v for v in qi if v is not None]
    svc = [v for v in svc if v is not None]

    def pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        k = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
        return round(s[k], 2)

    out: dict[str, Any] = {}
    if qi:
        out["queue_inclusive_p50_ms"] = pct(qi, 0.50)
        out["queue_inclusive_p95_ms"] = pct(qi, 0.95)
    if svc:
        out["service_p50_ms"] = pct(svc, 0.50)
        out["service_p95_ms"] = pct(svc, 0.95)
    for key in ("preprocess_ms", "infer_ms", "postprocess_ms", "persist_ms"):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = round(sum(vals) / len(vals), 2)
    bw = [_ms(r.get("preprocess_end"), r.get("infer_start")) for r in rows]
    bw = [v for v in bw if v is not None]
    if bw:
        out["batch_wait_p50_ms"] = pct(bw, 0.50)
        out["batch_wait_p95_ms"] = pct(bw, 0.95)
    peaks = [int(r["predicted_peak_rss"]) for r in rows if r.get("predicted_peak_rss")]
    if peaks:
        out["peak_rss_bytes_max"] = max(peaks)
    out["n_succeeded"] = len(rows)
    return out
