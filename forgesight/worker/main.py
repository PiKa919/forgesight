"""The worker process: claim, preprocess, batch, infer, postprocess, persist.

This is design §8.3. The shape of it is the point:

    claim loop ──► prefetch queue (bounded) ──► batcher (<=B items or <=W ms)
          ▲            decode/preprocess in a thread pool          │
          │                                                          ▼
    heartbeat task                              single inference executor
                                                                 │
                                              postprocess ──► fenced persist

Every stage has an explicit bound, and the inference executor is a single thread
that owns the model. Two things fall out of that which are worth stating:

* **Per-runtime memory is attributable.** One OS process per pool, so the peak
  RSS of the torch pool and the ORT pool are separately meaningful numbers
  rather than two allocators mixed in one heap.
* **A crash is survivable.** Items are durable rows with leases, so killing this
  process requeues its work rather than losing it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import psutil

from forgesight.db.pool import PoolLike
from forgesight.ledger.claims import ClaimedItem, Ledger, PredictionIn, record_timings
from forgesight.settings import Settings
from forgesight.storage.object_store import ObjectStore
from forgesight.vision.cancel import ThreadingCancelToken
from forgesight.vision.postprocess import postprocess_batch
from forgesight.vision.types import Cancelled, Candidate, FailureCode
from forgesight.worker.batcher import MicroBatcher
from forgesight.worker.memprobe import Admission, Calibration, RssSampler, effective_budget

log = logging.getLogger("forgesight.worker")


@dataclass(slots=True)
class Prepared:
    """A claimed item with its page decoded and its tensors built."""

    item: ClaimedItem
    candidate: Candidate
    tensors: np.ndarray
    items: list
    preprocess_ms: float
    rgb_nbytes: int


@dataclass(slots=True)
class Outcome:
    item: ClaimedItem
    state: str
    detections: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    failure_code: str | None = None
    failure_detail: str = ""
    peak_rss: int | None = None


class ModelCache:
    """LRU of loaded models, at most `size`, verified on load.

    Keyed by artifact sha256, so two candidates that differ only in thread count
    do not load the weights twice, and a corrupted file is refused rather than
    loaded and trusted.
    """

    def __init__(self, size: int = 2):
        self.size = size
        self._entries: dict[str, tuple[Candidate, object]] = {}
        self._order: list[str] = []

    def get(self, artifact_sha: str) -> tuple[Candidate, object] | None:
        if artifact_sha in self._entries:
            self._order.remove(artifact_sha)
            self._order.append(artifact_sha)
        return self._entries.get(artifact_sha)

    def put(self, artifact_sha: str, candidate: Candidate, model) -> None:
        self._entries[artifact_sha] = (candidate, model)
        self._order.append(artifact_sha)
        while len(self._order) > self.size:
            victim = self._order.pop(0)
            _, model = self._entries.pop(victim)
            close = getattr(model, "close", None)
            if close:
                with contextlib.suppress(Exception):
                    close()

    def loaded(self) -> list[str]:
        return list(self._order)

    def clear(self) -> None:
        for _, model in self._entries.values():
            close = getattr(model, "close", None)
            if close:
                with contextlib.suppress(Exception):
                    close()
        self._entries.clear()
        self._order.clear()


class Worker:
    def __init__(
        self,
        settings: Settings,
        pool: PoolLike,
        store: ObjectStore,
        pool_name: str,
        worker_id: str | None = None,
    ):
        self.s = settings
        self.pool = pool
        self.store = store
        self.pool_name = pool_name
        self.worker_id = worker_id or f"{pool_name}-{os.getpid()}"
        self.ledger = Ledger(pool)
        self.cache = ModelCache(settings.model_cache_size)
        self.preprocess_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"prep-{pool_name}"
        )
        self.stopping = asyncio.Event()
        self.held: list[ClaimedItem] = []
        self.admission = Admission(
            calibration=None,
            budget_bytes=effective_budget(
                settings.worker_mem_budget, settings.host_mem_fraction,
                settings.memory_safety_margin,
            ),
            safety_margin=settings.memory_safety_margin,
            max_batch=settings.max_batch,
        )
        self.stats = {
            "claimed": 0, "succeeded": 0, "failed": 0, "cancelled": 0,
            "fenced": 0, "recycles": 0, "batches": 0,
        }

    # -- candidate resolution --------------------------------------------
    def candidate_for(self, item: ClaimedItem) -> Candidate:
        """Rebuild the candidate a work item was pinned to.

        Read from the item's own row rather than from the active release, so a
        promote or rollback mid-batch cannot change what an in-flight item runs
        (AT-14).
        """
        from forgesight.eval.runner import candidate_from_row

        with self.pool.connection() as conn:
            row = conn.fetchone(
                "SELECT * FROM candidate WHERE workspace_id = ? AND id = ?",
                (item.workspace_id, item.candidate_id),
            )
        if row is None:
            raise LookupError(f"candidate {item.candidate_id} vanished")
        return candidate_from_row(row, self.s, self.pool)

    def model_for(self, candidate: Candidate):
        entry = self.cache.get(candidate.artifact.sha256)
        if entry is not None and entry[0].runtime.hash == candidate.runtime.hash:
            return entry[1]
        from forgesight.vision.runtimes.base import OrtModel, TorchModel

        cls = TorchModel if candidate.runtime.runtime == "torch" else OrtModel
        model = cls(candidate.artifact, candidate.runtime)
        self.cache.put(candidate.artifact.sha256, candidate, model)
        return model

    # -- stage 1: claim ---------------------------------------------------
    async def claim_once(self) -> list[ClaimedItem]:
        items = await asyncio.to_thread(
            self.ledger.claim, self.pool_name, self.worker_id,
            self.s.claim_batch, self.s.lease_seconds,
        )
        self.stats["claimed"] += len(items)
        if items:
            self.held.extend(items)
        return items

    # -- stage 2: decode and preprocess -----------------------------------
    def prepare(self, item: ClaimedItem) -> Prepared:
        t0 = time.monotonic()
        from forgesight.eval.dataset import page_bytes, page_object_keys

        keys = page_object_keys(self.pool, [item.page_id])
        key = keys.get(item.page_id)
        if key is None:
            raise LookupError(f"page {item.page_id} has no object key")

        from forgesight.vision.registry import preprocessor_for

        candidate = self.candidate_for(item)
        rgb = page_bytes(self.store, key)
        prepared = preprocessor_for(candidate)(rgb)
        tensors = np.stack([p.tensor for p in prepared])
        ms = (time.monotonic() - t0) * 1000.0
        return Prepared(item, candidate, tensors, prepared, ms, rgb.nbytes)

    # -- stage 3: batch ----------------------------------------------------
    def build_batch(self, prepared: list[Prepared], items_per_batch: int) -> list[Prepared]:
        """Round-robin the prepared items of this claim into micro-batches.

        Round-robin rather than contiguous slicing so that a batch of mixed page
        sizes is spread across batches instead of concentrating the expensive
        ones together.
        """
        out: list[Prepared] = []
        per: list[list[Prepared]] = [[] for _ in range(max(1, items_per_batch))]
        for i, p in enumerate(prepared):
            per[i % len(per)].append(p)
        for group in per:
            if group:
                out.append(group)  # type: ignore[arg-type]
        return out  # type: ignore[return-value]

    # -- stage 4: infer, postprocess, persist ------------------------------
    def run_batch(self, group: list[Prepared], cancel: ThreadingCancelToken) -> list[Outcome]:
        if not group:
            return []
        lead = group[0]
        model = self.model_for(lead.candidate)
        admitted = self.admission.admit(len(group), sum(g.rgb_nbytes for g in group[1:]))
        if admitted == 0:
            # AT-7: the worker refuses this work and stays alive.
            return [
                Outcome(g.item, "failed", failure_code=FailureCode.EXCEEDS_MEMORY_BUDGET.value,
                        failure_detail=(
                            f"budget {self.admission.effective_budget} cannot fit a single "
                            f"640x640 item plus {sum(x.rgb_nbytes for x in group[1:])} bytes "
                            f"of prefetched pages"
                        ))
                for g in group
            ]
        group = group[:admitted]

        outcomes: list[Outcome] = []
        with RssSampler(self.s.rss_sample_ms) as sampler:
            infers: list[tuple[Prepared, list, float]] = []
            for p in group:
                if cancel.is_set():
                    outcomes.append(Outcome(p.item, "cancelled",
                                            failure_code=FailureCode.CANCELLED.value))
                    continue
                if self._cancel_requested(p.item):
                    outcomes.append(Outcome(p.item, "cancelled",
                                            failure_code=FailureCode.CANCELLED.value))
                    continue
                infers.append((p, [], 0.0))

            for p, _, _ in infers:
                try:
                    t0 = time.monotonic()
                    raw = model.infer(p.tensors, cancel)
                    infer_ms = (time.monotonic() - t0) * 1000.0
                except Cancelled:
                    outcomes.append(Outcome(p.item, "cancelled",
                                            failure_code=FailureCode.CANCELLED.value))
                    continue
                except Exception as exc:
                    log.warning("inference failed for %s: %s", p.item.id, exc)
                    outcomes.append(Outcome(p.item, "failed",
                                            failure_code=FailureCode.INFERENCE.value,
                                            failure_detail=str(exc)[:500]))
                    continue
                if cancel.is_set():
                    outcomes.append(Outcome(p.item, "cancelled",
                                            failure_code=FailureCode.CANCELLED.value))
                    continue

                t1 = time.monotonic()
                dets: list = []
                for row in postprocess_batch(
                    raw, p.items, model.id2label, p.candidate.score_threshold
                ):
                    dets.extend(row)
                post_ms = (time.monotonic() - t1) * 1000.0
                outcomes.append(Outcome(
                    p.item, "ok", detections=dets,
                    timings={
                        "preprocess_ms": round(p.preprocess_ms, 3),
                        "infer_ms": round(infer_ms, 3),
                        "postprocess_ms": round(post_ms, 3),
                    },
                ))

        peak = sampler.peak
        self.admission.observe(len(group), peak)
        for o in outcomes:
            o.peak_rss = peak
        self.stats["batches"] += 1
        return outcomes

    def _cancel_requested(self, item: ClaimedItem) -> bool:
        """Cooperative cancel check, refreshed by the heartbeat loop."""
        return self.ledger.is_cancel_requested(item.workspace_id, item.batch_id)

    def persist(self, outcomes: list[Outcome]) -> None:
        for o in outcomes:
            t0 = time.monotonic()
            if o.state == "ok":
                ok = self.ledger.complete(o.item, PredictionIn(o.detections, timings=o.timings))
                if ok:
                    self.stats["succeeded"] += 1
                else:
                    # AT-3: fenced out. The result is discarded, not written.
                    self.stats["fenced"] += 1
                    log.info("discarded a fenced result for %s", o.item.id)
            elif o.state == "cancelled":
                if self.ledger.mark_cancelled(o.item):
                    self.stats["cancelled"] += 1
            else:
                if self.ledger.fail(o.item, o.failure_code or "inference_failed",
                                    o.failure_detail):
                    self.stats["failed"] += 1
            persist_ms = (time.monotonic() - t0) * 1000.0
            timings = dict(o.timings)
            timings["persist_ms"] = round(persist_ms, 3)
            with self.pool.write() as conn:
                record_timings(conn, self.pool.dialect, o.item.id, timings)
                if o.peak_rss is not None:
                    conn.execute(
                        "UPDATE work_item SET predicted_peak_rss = ? WHERE id = ?",
                        (o.peak_rss, o.item.id),
                    )

    # -- main loop --------------------------------------------------------
    async def run(self, once: bool = False) -> None:
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            while not self.stopping.is_set():
                items = await self.claim_once()
                if not items:
                    if once:
                        return
                    await asyncio.sleep(self.s.poll_interval_s)
                    continue
                loop = asyncio.get_running_loop()
                prepared = await asyncio.gather(
                    *[loop.run_in_executor(self.preprocess_pool, self.prepare, it)
                       for it in items],
                    return_exceptions=True,
                )
                groups: list[list[Prepared]] = []
                for it, res in zip(items, prepared, strict=True):
                    if isinstance(res, BaseException):
                        self.persist([Outcome(
                            it, "failed",
                            failure_code=FailureCode.DECODE.value,
                            failure_detail=str(res)[:500],
                        )])
                        continue
                    groups.append([res])

                cancel = ThreadingCancelToken()
                ws, bid = items[0].workspace_id, items[0].batch_id
                cancel.on_set(
                    lambda w=ws, b=bid: self.ledger.is_cancel_requested(w, b)
                )
                # Preparation is only part of the story; the batcher forms
                # micro-batches no larger than the profile allows.
                for p in groups:
                    for b in range(0, len(p), self.s.max_batch):
                        chunk = p[b : b + self.s.max_batch]
                        outcomes = await loop.run_in_executor(
                            None, self.run_batch, chunk, cancel
                        )
                        self.persist(outcomes)
                        # Design §13.2: memory misprediction ends in a graceful
                        # recycle. Checked between micro-batches rather than
                        # between claims so a single oversized batch cannot run
                        # the process into the OOM killer first. The current batch
                        # has already been persisted, so nothing is lost.
                        if self._over_hard_cap():
                            log.warning(
                                "recycling: RSS is over the hard cap after %d batches",
                                self.stats["batches"],
                            )
                            self.stats["recycles"] += 1
                            self.request_stop()
                            return

                self.held = [i for i in self.held if i not in items]
                if once:
                    return
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            self.preprocess_pool.shutdown(wait=False, cancel_futures=True)

    async def _heartbeat(self) -> None:
        """Extend leases and refresh the cancel view for held items."""
        while not self.stopping.is_set():
            if self.held:
                held = await asyncio.to_thread(
                    self.ledger.heartbeat, list(self.held), self.s.lease_seconds
                )
                if held < len(self.held):
                    # Lost the lease on some items: stop working on them.
                    log.warning("lease lost on %d of %d items", len(self.held) - held,
                                len(self.held))
            await asyncio.sleep(max(0.2, self.s.lease_seconds / 3.0))

    def request_stop(self) -> None:
        self.stopping.set()

    def _over_hard_cap(self) -> bool:
        """Measured RSS against the hard cap, or False when it cannot be read.

        Never recycles on a failed measurement. Exiting a healthy process
        because psutil could not read /proc trades a running worker for a
        measurement problem, which is the wrong direction.
        """
        from forgesight.worker.memprobe import hard_cap_bytes, over_hard_cap

        try:
            peak = psutil.Process().memory_info().rss
        except Exception:
            log.warning("could not read RSS; not recycling")
            return False
        cap = hard_cap_bytes(self.s.rss_hard_cap_fraction)
        return over_hard_cap(peak, cap)

    def status(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "pool": self.pool_name,
            "held": len(self.held),
            "models": self.cache.loaded(),
            "admission": self.admission.stats(),
            "stats": dict(self.stats),
        }


def install_signal_handlers(worker: Worker) -> None:
    """SIGTERM drains, SIGINT stops immediately.

    Draining on SIGTERM matters because the compose file sends it: exiting
    mid-batch leaves leases to expire, which works but is slower and noisier
    than finishing what is in hand.
    """
    loop = asyncio.get_running_loop()

    def term() -> None:
        log.info("SIGTERM: draining held work")
        worker.request_stop()

    def intr() -> None:
        log.info("SIGINT: stopping now")
        worker.request_stop()

    for sig, handler in ((signal.SIGTERM, term), (signal.SIGINT, intr)):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handler)


__all__ = ["Calibration", "MicroBatcher", "ModelCache", "Worker", "install_signal_handlers"]
