"""Memory probe and the admission controller (design §8.4).

The model is: peak RSS for a batch of b items is roughly `M0 + b * m_item`, with
M0 the steady-state cost of a loaded model and m_item the marginal cost of one
640x640 input. That is fitted per candidate and per host, and admission then
picks the largest batch that fits the budget.

Why a linear fit rather than a measured lookup table: the design's point is that
a configuration can be admitted for work it has never actually run a given batch
size for. A fit predicts the unmeasured point; a table cannot.

The prediction is also checked, not just used. A sampler thread watches real RSS
against the prediction, and a prediction that is exceeded by more than
`rss_misprediction_factor` is recorded. Admission accuracy is itself a reported
result, because a controller that is quietly wrong is worse than none.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass(slots=True)
class Calibration:
    """Fitted memory model for one candidate on one host.

    `valid` is the important field. RSS is a high-water mark that neither Python
    nor either runtime returns to the OS, so a batch smaller than one already
    run adds nothing measurable and the samples are mostly zeros. A least-squares
    line through those has a *negative* intercept, which is not a memory model --
    it is arithmetic on a degenerate sample. When that happens `valid` is False
    and admission falls back to the largest measured peak, which is
    conservative: it assumes more memory than the fit would have.
    """

    m0_bytes: int
    m_item_bytes: float
    max_residual: float
    samples: dict[int, int] = field(default_factory=dict)
    host_fingerprint: str = ""
    measured_at: float = 0.0
    valid: bool = True
    invalid_reason: str = ""

    def peak(self, b: int) -> int:
        if self.valid:
            return int(self.m0_bytes + b * self.m_item_bytes)
        known = [v for v in self.samples.values() if v > 0]
        return int(max(known) if known else 0)

    def as_dict(self) -> dict:
        return {
            "m0_bytes": self.m0_bytes,
            "m_item_bytes": round(self.m_item_bytes, 1),
            "max_residual": round(self.max_residual, 1),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "samples": {str(k): v for k, v in sorted(self.samples.items())},
            "host_fingerprint": self.host_fingerprint,
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(
            m0_bytes=int(d["m0_bytes"]),
            m_item_bytes=float(d["m_item_bytes"]),
            max_residual=float(d.get("max_residual", 0.0)),
            samples={int(k): int(v) for k, v in (d.get("samples") or {}).items()},
            host_fingerprint=d.get("host_fingerprint", ""),
            measured_at=float(d.get("measured_at", 0.0)),
        )


def fit(samples: dict[int, int]) -> Calibration:
    """Least-squares fit of peak(b) = M0 + b*m_item over measured batch sizes.

    With a single sample the slope is unidentifiable, so it falls back to the
    measured marginal cost of that one step, which is the best available
    estimate rather than a fabricated one.
    """
    if not samples:
        raise ValueError("cannot calibrate from zero samples")
    bs = sorted(samples)
    if len(bs) == 1:
        m0 = int(samples[bs[0]] * 0.9)
        slope = float(samples[bs[0]] - m0)
        return Calibration(m0, slope, abs(slope), dict(samples), measured_at=time.time())

    xs = [float(b) for b in bs]
    ys = [float(samples[b]) for b in bs]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / denom
    intercept = mean_y - slope * mean_x
    residuals = [abs(y - (intercept + slope * x)) for x, y in zip(xs, ys, strict=True)]

    # A negative intercept means the fit claims memory is released between
    # batch sizes, which cannot happen. The model is then marked invalid rather
    # than reported, and admission uses the largest measured peak instead.
    invalid = ""
    if intercept < 0:
        invalid = (
            f"the fit gives a negative intercept ({intercept:.0f} B), i.e. it claims "
            f"memory is released as the batch shrinks; RSS is a high-water mark so "
            f"that is impossible. Admission falls back to the largest measured peak."
        )
    return Calibration(
        int(intercept), slope, max(residuals), dict(samples), measured_at=time.time(),
        valid=not invalid, invalid_reason=invalid,
    )


def prefetch_bytes(items: list) -> int:
    """Exact bytes held by the prefetch queue: decoded pages plus tensors."""
    total = 0
    for it in items:
        rgb = getattr(it, "rgb", None)
        if rgb is not None:
            total += int(rgb.nbytes)
        total += it.tensor.nbytes
    return total


def host_total_bytes() -> int:
    """Total host memory, or 0 when it cannot be read.

    Zero on failure rather than an exception: every caller treats an unknown cap
    as "no opinion", so a missing reading must not become a hard failure.
    """
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def hard_cap_bytes(host_fraction: float = 0.85) -> int:
    """The ceiling a worker refuses to keep running past.

    Deliberately a fraction of *total* host memory rather than of
    `worker_mem_budget`. Admission's budget is a soft target and is known to be
    unreliable on this host -- the calibration fit produced a negative intercept
    and was marked invalid (ADR 0004) -- so the backstop is anchored to something
    that cannot drift: the machine.
    """
    return int(host_total_bytes() * host_fraction)


def over_hard_cap(peak_rss: int, cap_bytes: int) -> bool:
    """True when a worker should stop after finishing what it holds.

    Kept separate from `Admission.admit` because the two answer different
    questions. Admission is a prediction about work not yet run and can be wrong
    in either direction; this is a measurement of what the process already did,
    so it belongs as a fallback rather than as the primary control.
    """
    return cap_bytes > 0 and peak_rss > cap_bytes


@dataclass(slots=True)
class Admission:
    """Picks the largest batch that fits, or refuses the work."""

    calibration: Calibration | None
    budget_bytes: int
    safety_margin: float = 0.15
    max_batch: int = 4
    max_residual: float = 0.0
    observations: list[tuple[int, int, int]] = field(default_factory=list)
    mispredictions: int = 0

    @property
    def effective_budget(self) -> int:
        usable = self.budget_bytes * (1.0 - self.safety_margin)
        return int(min(self.budget_bytes, max(0, usable)))

    def admit(self, requested_b: int, pending_bytes: int = 0) -> int:
        """Largest b <= requested that fits. 0 means the item cannot be run."""
        if self.calibration is None:
            return min(requested_b, self.max_batch)
        budget = self.effective_budget - pending_bytes
        if budget <= 0:
            return 0
        cal = self.calibration
        # Widen by the fit's own residual, so a known-noisy fit is not treated as
        # more precise than it is. An invalid fit has no usable residual, so the
        # whole measured peak is treated as the uncertainty.
        slack = cal.max_residual if cal.valid else 0
        for b in range(min(requested_b, self.max_batch), 0, -1):
            if cal.peak(b) + slack <= budget:
                return b
        return 0

    def observe(self, b: int, observed_peak_rss: int) -> None:
        """Record a real peak against the prediction and flag a misprediction."""
        self.observations.append((b, observed_peak_rss, self.predict(b)))
        if self.calibration is None:
            return
        predicted = self.calibration.peak(b)
        if predicted > 0 and observed_peak_rss > predicted * 1.10:
            self.mispredictions += 1

    def predict(self, b: int) -> int:
        return self.calibration.peak(b) if self.calibration else 0

    def stats(self) -> dict:
        return {
            "budget_bytes": self.budget_bytes,
            "effective_budget_bytes": self.effective_budget,
            "calibrated": self.calibration is not None,
            "calibration_valid": bool(self.calibration and self.calibration.valid),
            "calibration_note": (
                self.calibration.invalid_reason if self.calibration else ""
            ),
            "m0_bytes": self.calibration.m0_bytes if self.calibration else None,
            "m_item_bytes": (
                round(self.calibration.m_item_bytes, 1) if self.calibration else None
            ),
            "mispredictions": self.mispredictions,
            "n_observations": len(self.observations),
        }


class RssSampler:
    """Samples this process's RSS on a tight interval for the duration of a block.

    A separate thread is required: the inference call blocks the main thread for
    hundreds of milliseconds, and a peak sampled only between inferences misses
    the peak entirely.
    """

    def __init__(self, interval_ms: int = 5):
        self.interval_s = max(0.001, interval_ms / 1000.0)
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def __enter__(self) -> RssSampler:
        self.baseline = self._proc.memory_info().rss
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except Exception:
                return
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)

    def max_rss_bytes(self) -> int:
        """Peak RSS in bytes.

        `ru_maxrss` is KiB on Linux and bytes on macOS, so it is normalised here
        rather than at each call site, where the difference would be easy to get
        backwards by a factor of 1024.
        """
        import resource
        import sys

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw if sys.platform == "darwin" else raw * 1024


def host_free_bytes() -> int:
    try:
        return psutil.virtual_memory().available
    except Exception:
        return 0


def effective_budget(worker_budget: int, host_fraction: float, safety_margin: float) -> int:
    """min(configured, half of what's actually free) minus a safety margin.

    The host term is what makes this a real constraint rather than a constant: on
    a busy machine a worker that trusts only its configured budget will be the
    thing that tips the machine into swap.
    """
    by_host = int(host_free_bytes() * host_fraction)
    return int(max(0, min(worker_budget, by_host) * (1.0 - safety_margin)))
