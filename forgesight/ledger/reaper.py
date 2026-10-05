"""The reaper: lease expiry, batch rollup and orphan sweep (design §8.2, §8.1).

Three jobs, each existing because something can die without warning:

* **Requeue expired leases.** A `running` item whose lease has lapsed returns to
  `queued` while it has attempts left, and to `failed(max_attempts)` once it
  does not.

* **Roll item states up into the batch row**, so the API can answer "how is this
  batch doing" without aggregating the item table on every poll.

* **Sweep orphan objects.** The object store is written to *before* the database
  transaction that references it, so an upload that dies in between leaves an
  unreferenced object behind. Anything older than the grace period and not
  referenced by a page row is deleted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from forgesight.db.pool import Dialect, PoolLike
from forgesight.ledger.claims import sql
from forgesight.settings import Settings
from forgesight.vision.types import FailureCode

log = logging.getLogger("forgesight.reaper")


@dataclass(slots=True)
class ReapResult:
    requeued: int = 0
    exhausted: int = 0
    batches_updated: int = 0
    objects_deleted: int = 0

    def __add__(self, other: ReapResult) -> ReapResult:
        return ReapResult(
            self.requeued + other.requeued,
            self.exhausted + other.exhausted,
            self.batches_updated + other.batches_updated,
            self.objects_deleted + other.objects_deleted,
        )


_TERMINAL = "('queued','running')"

# Status ladder, most significant first. The order encodes the rule that a
# failure dominates: a batch with one failed item is "failed" whatever else
# happened, because it did not produce a complete result. "partial" is reserved
# for "some work was cancelled, none of it failed".
#
#   cancelled  cancel was requested and nothing is still running
#   failed     all items terminal, at least one failed
#   partial    all items terminal, none failed, at least one cancelled
#   succeeded  all items terminal, all succeeded
#   running    at least one item running
#   queued     still waiting
_BATCH_ROLLUP_PG = f"""
UPDATE batch b SET
  status = CASE
    WHEN b.cancel_requested_at IS NOT NULL
         AND NOT EXISTS (SELECT 1 FROM work_item w
                         WHERE w.batch_id = b.id AND w.state = 'running')
      THEN 'cancelled'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state IN {_TERMINAL})
         AND EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state = 'failed')
      THEN 'failed'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state IN {_TERMINAL})
         AND EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state = 'cancelled')
      THEN 'partial'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state IN {_TERMINAL})
      THEN 'succeeded'
    WHEN EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                  AND w.state = 'running')
      THEN 'running'
    ELSE b.status
  END,
  finished_at = CASE
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = b.id
                      AND w.state IN {_TERMINAL})
      THEN COALESCE(b.finished_at, clock_timestamp())
    ELSE NULL
  END
WHERE b.finished_at IS NULL OR b.status IN ('queued', 'running', 'succeeded', 'partial')
"""

_BATCH_ROLLUP_SQLITE = f"""
UPDATE batch SET
  status = CASE
    WHEN cancel_requested_at IS NOT NULL
         AND NOT EXISTS (SELECT 1 FROM work_item w
                         WHERE w.batch_id = batch.id AND w.state = 'running')
      THEN 'cancelled'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state IN {_TERMINAL})
         AND EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state = 'failed')
      THEN 'failed'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state IN {_TERMINAL})
         AND EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state = 'cancelled')
      THEN 'partial'
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state IN {_TERMINAL})
      THEN 'succeeded'
    WHEN EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                  AND w.state = 'running')
      THEN 'running'
    ELSE status
  END,
  finished_at = CASE
    WHEN NOT EXISTS (SELECT 1 FROM work_item w WHERE w.batch_id = batch.id
                      AND w.state IN {_TERMINAL})
      THEN COALESCE(finished_at, CURRENT_TIMESTAMP)
    ELSE NULL
  END
WHERE finished_at IS NULL OR status IN ('queued', 'running', 'succeeded', 'partial')
"""


class Reaper:
    def __init__(self, pool: PoolLike, settings: Settings):
        self.pool = pool
        self.s = settings

    def reap_expired_leases(self) -> ReapResult:
        """Return lapsed leases to the queue, or fail them past the attempt cap.

        The requeue clears `fencing_token`. The next claim writes a strictly
        larger one, so the previous owner is fenced out on completion: its
        `complete()` matches no row and returns False.
        """
        d = self.pool.dialect
        requeue = sql(
            d,
            "UPDATE work_item SET state = 'queued', lease_owner = NULL, "
            "lease_expires_at = NULL, claimed_at = NULL, fencing_token = NULL "
            "WHERE state = 'running' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < clock_timestamp() AND attempts < %s RETURNING id",
            "UPDATE work_item SET state = 'queued', lease_owner = NULL, "
            "lease_expires_at = NULL, claimed_at = NULL, fencing_token = NULL "
            "WHERE state = 'running' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < CURRENT_TIMESTAMP AND attempts < ? RETURNING id",
        )
        exhaust = sql(
            d,
            "UPDATE work_item SET state = 'failed', failure_code = %s, "
            "failure_detail = 'lease expired and attempts exhausted', "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE state = 'running' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < clock_timestamp() AND attempts >= %s RETURNING id",
            "UPDATE work_item SET state = 'failed', failure_code = ?, "
            "failure_detail = 'lease expired and attempts exhausted', "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE state = 'running' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < CURRENT_TIMESTAMP AND attempts >= ? RETURNING id",
        )
        with self.pool.write() as conn:
            res = ReapResult(
                requeued=len(conn.fetchall(requeue, (self.s.max_attempts,))),
                exhausted=len(
                    conn.fetchall(
                        exhaust,
                        (FailureCode.MAX_ATTEMPTS.value, self.s.max_attempts),
                    )
                ),
            )
        if res.requeued or res.exhausted:
            log.info("reaped %d requeued, %d exhausted", res.requeued, res.exhausted)
        return res

    def refresh_batch_status(self) -> int:
        """Roll item states up into the batch row."""
        stmt = sql(self.pool.dialect, _BATCH_ROLLUP_PG, _BATCH_ROLLUP_SQLITE)
        with self.pool.write() as conn:
            return conn.execute(stmt).rowcount

    def sweep_orphan_objects(self, store) -> int:
        """Delete store objects that no row references and that are old enough
        to be safely past any in-flight write.

        The store is enumerated rather than the database, because the question
        is "which bytes on disk are unreferenced", and only the store can answer
        that. Two object namespaces exist -- `assets/<sha256>` for the uploaded
        bytes and `pages/<sha256>.png` for renders -- and each is checked
        against the table that owns it.
        """
        hours = self.s.orphan_object_grace_h
        now = (
            "clock_timestamp()"
            if self.pool.dialect is Dialect.POSTGRES
            else "datetime('now')"
        )
        keep_stmt = sql(
            self.pool.dialect,
            f"SELECT object_key FROM asset WHERE created_at > {now} - "
            f"INTERVAL '{hours} hours' "
            "UNION ALL "
            f"SELECT object_key FROM page WHERE created_at > {now} - "
            f"INTERVAL '{hours} hours'",
            "SELECT object_key FROM asset WHERE created_at > datetime('now', ?) "
            "UNION ALL "
            "SELECT object_key FROM page WHERE created_at > datetime('now', ?)",
        )
        keep_params = (
            ()
            if self.pool.dialect is Dialect.POSTGRES
            else (f"-{hours} hours", f"-{hours} hours")
        )
        with self.pool.connection() as conn:
            recent = {r["object_key"] for r in conn.fetchall(keep_stmt, keep_params)}

        deleted = 0
        cutoff = time.time() - hours * 3600
        for key, mtime in store.iter_objects():
            if key in recent or mtime > cutoff:
                continue
            try:
                store.delete(key)
                deleted += 1
            except Exception as exc:
                # A vanished object is the desired end state anyway.
                log.debug("orphan delete %s: %s", key, exc)
        if deleted:
            log.info("swept %d orphan objects", deleted)
        return deleted

    def tick(self, store=None) -> ReapResult:
        res = self.reap_expired_leases()
        res.batches_updated = self.refresh_batch_status()
        if store is not None:
            res.objects_deleted = self.sweep_orphan_objects(store)
        return res
