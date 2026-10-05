"""A concrete cancel token shared by both runtimes (design §8.5)."""

from __future__ import annotations

import threading
from collections.abc import Callable


class ThreadingCancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception:
                # A failing cancel hook must not prevent the token from being
                # observed by the worker's own polling checks.
                pass

    def on_set(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._event.is_set():
                fire_now = True
            else:
                self._callbacks.append(cb)
                fire_now = False
        if fire_now:
            cb()

    def reset(self) -> None:
        with self._lock:
            self._event.clear()
            self._callbacks.clear()


class NeverCancel:
    """Token that is never set. Used by benchmarks and evaluation."""

    def is_set(self) -> bool:
        return False

    def on_set(self, cb: Callable[[], None]) -> None:
        return None
