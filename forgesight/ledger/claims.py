"""The durable job ledger (design §8.2).

Correctness rests on three mechanisms, and it is worth being precise about
which one does what:

1. **A lease.** A claimed item carries `lease_expires_at`. If the worker dies,
   the lease lapses and the reaper returns the item to the queue.

2. **A fencing token.** Every claim allocates a strictly increasing token. A
   worker may only write its result while its token is still the one on the
   row. A worker paused past its lease that lost the item to the reaper, and
   then woke up, is *fenced out*: its write affects zero rows. This is what
   makes a stale worker harmless, and it is acceptance test AT-3.

3. **A unique constraint** on prediction(work_item_id, candidate_id). Even if
   two workers somehow both believed they owned an item, the database refuses
   the second Prediction.

`SELECT ... FOR UPDATE SKIP LOCKED` is deliberately not in that list. It makes
concurrent claims cheap -- disjoint rows, no blocking -- but the at-most-once
guarantee holds without it. See docs/adr/0002-ledger-dialects.md.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from forgesight.db.pool import Dialect, PoolLike, claim_sql, next_fencing_sql, ph
from forgesight.vision.types import Detection


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def now_utc() -> datetime:
    """A microsecond-precision UTC timestamp, supplied by the application.

    Design §8.6 asks for `clock_timestamp()` on cross-process events, which is
    right on PostgreSQL. It is not portable: SQLite's `CURRENT_TIMESTAMP`
    resolves to whole seconds, so a 300 ms inference and a 1.4 s one both report
    the same rounded figure and the queue-inclusive latency this system exists to
    measure comes out as exactly 0 s, 1 s or 2 s. Timestamps are therefore
    passed in from Python, which gives the same wall-clock semantics on both
    backends and keeps sub-second resolution.
    """
    return datetime.now(UTC)


def sql(d: Dialect, postgres: str, sqlite: str) -> str:
    """Pick the statement for a dialect.

    Both variants are written out in full at each call site rather than being
    stitched from fragments, so a reviewer can read the exact statement a given
    backend will run without mentally expanding templates.
    """
    return postgres if d is Dialect.POSTGRES else sqlite


@dataclass(frozen=True, slots=True)
class ClaimedItem:
    id: str
    workspace_id: str
    batch_id: str
    page_id: str
    candidate_id: str
    pool: str
    attempts: int
    fencing_token: int
    lease_owner: str
    claimed_at: Any


@dataclass(frozen=True, slots=True)
class PredictionIn:
    detections: list[Detection]
    raw_digest: str | None = None
    timings: dict | None = None


TIMING_COLUMNS = (
    "enqueue_to_claim_ms",
    "preprocess_ms",
    "infer_ms",
    "postprocess_ms",
    "persist_ms",
)


class Ledger:
    def __init__(self, pool: PoolLike):
        self.pool = pool
        self.d: Dialect = pool.dialect

    # -- claim ------------------------------------------------------------
    def claim(
        self, pool: str, worker_id: str, n: int = 4, lease_s: int = 60
    ) -> list[ClaimedItem]:
        """Take up to `n` queued items for `pool`, oldest first.

        The fencing token is allocated as its own statement inside the claim
        transaction so that it is strictly increasing across claims on every
        backend. One claim may stamp the same token on several rows; what
        matters for fencing is that each claim outranks the last.
        """
        with self.pool.write() as conn:
            token = int(conn.fetchone(next_fencing_sql(self.d))["token"])
            rows = conn.fetchall(
                claim_sql(self.d),
                {
                    "pool": pool,
                    "n": n,
                    "worker": worker_id,
                    "lease_s": lease_s,
                    "token": token,
                    "now": now_utc(),
                },
            )
        return [self._to_claimed(r) for r in rows]

    @staticmethod
    def _to_claimed(r: dict) -> ClaimedItem:
        return ClaimedItem(
            id=r["id"],
            workspace_id=r["workspace_id"],
            batch_id=r["batch_id"],
            page_id=r["page_id"],
            candidate_id=r["candidate_id"],
            pool=r["pool"],
            attempts=int(r["attempts"]),
            fencing_token=int(r["fencing_token"]),
            lease_owner=r["lease_owner"],
            claimed_at=r["claimed_at"],
        )

    # -- lease ------------------------------------------------------------
    def heartbeat(self, items: list[ClaimedItem], lease_s: int = 60) -> int:
        """Extend leases for held items. Returns how many are still ours.

        A fenced-out worker finds nothing to extend, which is its signal to
        abandon the item rather than continue and race the rightful owner.
        """
        if not items:
            return 0
        stmt = sql(
            self.d,
            "UPDATE work_item SET lease_expires_at = "
            "clock_timestamp() + (%s * interval '1 second') "
            "WHERE id = %s AND fencing_token = %s AND state = 'running' RETURNING id",
            "UPDATE work_item SET lease_expires_at = datetime('now', ? || ' seconds') "
            "WHERE id = ? AND fencing_token = ? AND state = 'running' RETURNING id",
        )
        held = 0
        with self.pool.write() as conn:
            for it in items:
                if conn.fetchone(stmt, (lease_s, it.id, it.fencing_token)) is not None:
                    held += 1
        return held

    # -- completion -------------------------------------------------------
    def complete(self, item: ClaimedItem, prediction: PredictionIn) -> bool:
        """Persist a result. False means the caller has been fenced out.

        The state change and the Prediction insert share one transaction, so a
        result can never be recorded while the item stays `running`.
        """
        p = ph(self.d)
        guarded = sql(
            self.d,
            "UPDATE work_item SET state = 'succeeded', "
            "persisted_at = %s, lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = %s AND fencing_token = %s AND state = 'running' RETURNING id",
            "UPDATE work_item SET state = 'succeeded', "
            "persisted_at = ? WHERE id = ? AND fencing_token = ? "
            "AND state = 'running' RETURNING id",
        )
        with self.pool.write() as conn:
            if conn.fetchone(
                guarded, (now_utc(), item.id, item.fencing_token)
            ) is None:
                return False
            conn.execute(
                "INSERT INTO prediction(id, workspace_id, work_item_id, candidate_id, "
                "detections, n_detections, raw_digest, timings) "
                f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})",
                (
                    new_id("pr"),
                    item.workspace_id,
                    item.id,
                    item.candidate_id,
                    json.dumps([d.as_dict() for d in prediction.detections]),
                    len(prediction.detections),
                    prediction.raw_digest,
                    json.dumps(prediction.timings or {}),
                ),
            )
        return True

    def fail(self, item: ClaimedItem, code: str, detail: str = "") -> bool:
        stmt = sql(
            self.d,
            "UPDATE work_item SET state = 'failed', failure_code = %s, failure_detail = %s, "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = %s AND fencing_token = %s AND state = 'running' RETURNING id",
            "UPDATE work_item SET state = 'failed', failure_code = ?, failure_detail = ?, "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = ? AND fencing_token = ? AND state = 'running' RETURNING id",
        )
        with self.pool.write() as conn:
            return conn.fetchone(stmt, (code, detail, item.id, item.fencing_token)) is not None

    def mark_cancelled(self, item: ClaimedItem) -> bool:
        stmt = sql(
            self.d,
            "UPDATE work_item SET state = 'cancelled', failure_code = 'cancelled', "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = %s AND fencing_token = %s AND state = 'running' RETURNING id",
            "UPDATE work_item SET state = 'cancelled', failure_code = 'cancelled', "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = ? AND fencing_token = ? AND state = 'running' RETURNING id",
        )
        with self.pool.write() as conn:
            return conn.fetchone(stmt, (item.id, item.fencing_token)) is not None

    # -- batch cancellation ------------------------------------------------
    def cancel_batch(self, workspace_id: str, batch_id: str) -> int:
        """Cancel a batch and every item that has not started.

        Queued items move straight to `cancelled` rather than waiting to be
        claimed and dropped, so a cancel takes effect immediately for work that
        has not started. AT-4 asserts no `claimed_at` appears afterwards.
        Idempotent: a second call reports zero newly cancelled items.
        """
        touch = sql(
            self.d,
            "UPDATE batch SET cancel_requested_at = "
            "COALESCE(cancel_requested_at, clock_timestamp()) "
            "WHERE workspace_id = %s AND id = %s",
            "UPDATE batch SET cancel_requested_at = "
            "COALESCE(cancel_requested_at, CURRENT_TIMESTAMP) "
            "WHERE workspace_id = ? AND id = ?",
        )
        drop = sql(
            self.d,
            "UPDATE work_item SET state = 'cancelled', failure_code = 'cancelled' "
            "WHERE workspace_id = %s AND batch_id = %s AND state = 'queued' RETURNING id",
            "UPDATE work_item SET state = 'cancelled', failure_code = 'cancelled' "
            "WHERE workspace_id = ? AND batch_id = ? AND state = 'queued' RETURNING id",
        )
        with self.pool.write() as conn:
            conn.execute(touch, (workspace_id, batch_id))
            return len(conn.fetchall(drop, (workspace_id, batch_id)))

    def is_cancel_requested(self, workspace_id: str, batch_id: str) -> bool:
        stmt = sql(
            self.d,
            "SELECT cancel_requested_at FROM batch WHERE id = %s AND workspace_id = %s",
            "SELECT cancel_requested_at FROM batch WHERE id = ? AND workspace_id = ?",
        )
        with self.pool.connection() as conn:
            row = conn.fetchone(stmt, (batch_id, workspace_id))
        return bool(row and row["cancel_requested_at"])


def record_timings(conn, dialect: Dialect, item_id: str, timings: dict) -> None:
    """Store the in-process timing split for one item.

    Queue-inclusive latency is derived from these columns together with
    `received_at`, so it stays available for a batch whose worker died.
    """
    present = [(k, v) for k, v in timings.items() if k in TIMING_COLUMNS and v is not None]
    if not present:
        return
    p = ph(dialect)
    assignments = ", ".join(f"{name} = {p}" for name, _ in present)
    conn.execute(
        f"UPDATE work_item SET {assignments} WHERE id = {p}",
        (*(v for _, v in present), item_id),
    )
