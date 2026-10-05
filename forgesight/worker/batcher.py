"""Micro-batcher (design §8.3, borrowed from Design B).

Batching trades latency for throughput, and the trade is only worth making if
the caller controls how long a request waits. So a batch closes on whichever
comes first: the size cap `B` or the time window `W` milliseconds.

A batcher that only closes on size is a throughput optimisation that quietly
adds unbounded latency to the last item of a burst. A batcher that only closes
on time never benefits from batching at all. Both bounds are needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class Batch[T]:
    items: list[T]
    waited_ms: float
    closed_by: str  # "size" | "time" | "flush"


class MicroBatcher[T]:
    """Accumulates items until `max_batch` or `window_ms`, whichever comes first."""

    def __init__(self, max_batch: int, window_ms: int):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        if window_ms < 0:
            raise ValueError("window_ms must not be negative")
        self.max_batch = max_batch
        self.window_s = window_ms / 1000.0
        self._items: list[T] = []
        self._first_at: float | None = None
        self.stats = {"batches": 0, "items": 0, "closed_by_size": 0, "closed_by_time": 0,
                      "closed_by_flush": 0, "max_batch_seen": 0}

    def offer(self, item: T) -> Batch[T] | None:
        """Add an item. Returns a batch when this offer closed one."""
        now = time.monotonic()
        if not self._items:
            self._first_at = now
        self._items.append(item)

        if len(self._items) >= self.max_batch:
            return self._close("size")

        if self._first_at is not None and (now - self._first_at) >= self.window_s:
            return self._close("time")
        return None

    def due(self) -> bool:
        """True when the window has elapsed and a batch is waiting."""
        return bool(self._items) and self._first_at is not None and (
            (time.monotonic() - self._first_at) >= self.window_s
        )

    def flush(self) -> Batch[T] | None:
        if not self._items:
            return None
        return self._close("flush")

    def _close(self, why: str) -> Batch[T]:
        items, self._items = self._items, []
        waited = 0.0 if self._first_at is None else (time.monotonic() - self._first_at) * 1000.0
        self._first_at = None
        self.stats["batches"] += 1
        self.stats["items"] += len(items)
        self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(items))
        self.stats[f"closed_by_{why}"] += 1
        return Batch(items=items, waited_ms=round(waited, 3), closed_by=why)

    @property
    def pending(self) -> int:
        return len(self._items)


__all__ = ["Batch", "MicroBatcher"]
