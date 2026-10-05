"""Benchmark protocol (design §12).

The protocol is deliberately fussy, because the alternative is a number that
looks like a result and is not one:

* Warmup is discarded. The first inference of a model pays for lazy
  initialisation, allocator growth and first-touch page faults, and including it
  makes a fast configuration look slow.
* Configuration order is randomised per trial, so a slow drift over the run
  cannot systematically favour whichever configuration happened to go first.
* Repetition is reported as a median of medians with a min-max range, never as a
  single best-of number.
* A configuration whose trial throughput varies too much, or drifts from the
  first trial to the last, is marked **unstable** and excluded from comparisons.
  Reporting an unstable measurement as a result is worse than not reporting it.
* Every run stores its env manifest, and two runs are only ever compared when
  their manifests agree on CPU, OS and library versions.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROTOCOL_VERSION = "1"

WARMUP_PAGES = 20
MEASURED_PAGES = 200
TRIALS = 5
COOLDOWN_S = 30
OPEN_LOOP_DURATIONS_S = 300
OPEN_LOOP_RATES = (0.5, 0.8, 0.95)
BATCH_SIZES = (1, 2, 4, 8)
THREAD_COUNTS = (4, 8)
RESIZE_METHODS = ("pil_bilinear", "cv2_linear", "cv2_area")
CV_REL_TOLERANCE = 0.05
DRIFT_TOLERANCE = 0.10


@dataclass(slots=True)
class Config:
    """One measured point: a candidate plus the knobs that vary."""

    name: str
    model: str
    runtime: str
    resize: str
    batch: int
    threads: int
    tile: bool = False
    candidate_hash: str = ""

    def key(self) -> str:
        return (
            f"{self.model}/{self.runtime}/{self.resize}/b{self.batch}"
            f"/t{self.threads}{'/tile' if self.tile else ''}"
        )


@dataclass(slots=True)
class Sample:
    """One page's timing, with the queue wait kept separate from the service."""

    page_id: str
    queue_inclusive_ms: float
    service_ms: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    peak_rss: int


@dataclass(slots=True)
class TrialResult:
    config_key: str
    trial: int
    throughput_pages_per_s: float
    queue_inclusive_p50: float
    queue_inclusive_p95: float
    queue_inclusive_p99: float
    service_p50: float
    service_p95: float
    peak_rss_bytes: int
    steady_rss_bytes: int
    n_pages: int
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "throughput_pages_per_s": round(self.throughput_pages_per_s, 3),
            "queue_inclusive_p50_ms": round(self.queue_inclusive_p50, 2),
            "queue_inclusive_p95_ms": round(self.queue_inclusive_p95, 2),
            "queue_inclusive_p99_ms": round(self.queue_inclusive_p99, 2),
            "service_p50_ms": round(self.service_p50, 2),
            "service_p95_ms": round(self.service_p95, 2),
            "peak_rss_bytes": self.peak_rss_bytes,
            "steady_rss_bytes": self.steady_rss_bytes,
            "n_pages": self.n_pages,
        }


@dataclass(slots=True)
class StabilityVerdict:
    stable: bool
    coefficient_of_variation: float
    first_last_drift: float
    reasons: list[str] = field(default_factory=list)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile on a sorted copy.

    Nearest-rank rather than interpolated so a reported p95 is always a value
    that was actually observed.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, math.ceil(p * len(s)) - 1)
    return s[k]


def judge_stability(trials: list[TrialResult]) -> StabilityVerdict:
    """Apply the design's stability rule to one configuration's trials."""
    reasons: list[str] = []
    if len(trials) < 2:
        return StabilityVerdict(False, float("nan"), float("nan"),
                               ["fewer than two trials"])
    tput = [t.throughput_pages_per_s for t in trials]
    mean = statistics.fmean(tput)
    stdev = statistics.stdev(tput)
    cv = stdev / mean if mean else float("inf")
    drift = (tput[-1] - tput[0]) / tput[0] if tput[0] else float("inf")
    if cv > CV_REL_TOLERANCE:
        reasons.append(
            f"coefficient of variation {cv:.3f} exceeds {CV_REL_TOLERANCE}"
        )
    if abs(drift) > DRIFT_TOLERANCE:
        reasons.append(
            f"first-to-last drift {drift:+.1%} exceeds {DRIFT_TOLERANCE:.0%}"
        )
    return StabilityVerdict(not reasons, cv, drift, reasons)


def bootstrap_ci(
    values: list[float], statistic=percentile, p: float = 0.95,
    resamples: int = 10_000, seed: int = 20261005,
) -> tuple[float, float]:
    """Seeded bootstrap confidence interval.

    The seed is fixed so a reported interval is reproducible; an interval that
    changes when you re-run the analysis is not much of an interval.
    """
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample, p))
    stats.sort()
    lo = stats[int((1 - p) / 2 * resamples)]
    hi = stats[min(resamples - 1, int((1 + p) / 2 * resamples))]
    return (lo, hi)


def aggregate(trials: list[TrialResult]) -> dict:
    """Per-configuration summary: median of medians, plus the min-max range."""
    if not trials:
        return {"n_trials": 0}
    verdict = judge_stability(trials)
    tp = [t.throughput_pages_per_s for t in trials]
    p95 = [t.queue_inclusive_p95 for t in trials]
    lo, hi = bootstrap_ci(p95, lambda v, q: percentile(v, 0.95))
    return {
        "n_trials": len(trials),
        "throughput_pages_per_s": {
            "median_of_medians": round(statistics.median(tp), 3),
            "min": round(min(tp), 3),
            "max": round(max(tp), 3),
        },
        "queue_inclusive_p95_ms": {
            "median_of_medians": round(statistics.median(p95), 2),
            "min": round(min(p95), 2),
            "max": round(max(p95), 2),
            "ci95": [round(lo, 2), round(hi, 2)],
        },
        "service_p50_ms": round(statistics.median(
            [t.service_p50 for t in trials]), 2),
        "peak_rss_bytes": max(t.peak_rss_bytes for t in trials),
        "steady_rss_bytes": max(t.steady_rss_bytes for t in trials),
        "stability": {
            "stable": verdict.stable,
            "coefficient_of_variation": round(verdict.coefficient_of_variation, 4),
            "first_last_drift": round(verdict.first_last_drift, 4),
            "reasons": verdict.reasons,
        },
    }


def trial_order(configs: list[Config], trial: int, seed: int = 20261005) -> list[Config]:
    """Randomised configuration order, seeded so the plan is reproducible."""
    rng = random.Random(seed + trial)
    order = list(configs)
    rng.shuffle(order)
    return order


def poisson_arrivals(rate_per_s: float, duration_s: float, seed: int) -> list[float]:
    """Cumulative arrival offsets from a seeded Poisson process."""
    rng = random.Random(seed)
    t, out = 0.0, []
    while len(out) < 10_000:
        t += rng.expovariate(max(1e-9, rate_per_s))
        if t > duration_s:
            break
        out.append(t)
    return out


def tier1_matrix(models: list[str], runtimes: list[str],
                 batches: tuple[int, ...] = (1, 4), threads: int = 4) -> list[Config]:
    """The 12 required configurations of design §12.4."""
    return [
        Config(name=f"{m}-{r}-b{b}", model=m, runtime=r, resize="pil_bilinear",
               batch=b, threads=threads)
        for m in models for r in runtimes for b in batches
    ]


def tier2_matrix(models: list[str], runtimes: list[str],
                 batches: tuple[int, ...], threads: tuple[int, ...],
                 resizes: tuple[str, ...]) -> list[Config]:
    return [
        Config(name=f"{m}-{r}-{z}-b{b}-t{t}", model=m, runtime=r, resize=z,
               batch=b, threads=t)
        for m in models for r in runtimes
        for z in resizes for b in batches for t in threads
    ]


def is_comparable(a: dict, b: dict) -> tuple[bool, list[str]]:
    """Two aggregates may only be compared within one env manifest."""
    from forgesight.bench.env_capture import comparable

    env_a, env_b = (a or {}).get("env"), (b or {}).get("env")
    if not env_a or not env_b:
        return False, ["one or both runs have no environment manifest"]
    diffs = comparable(env_a, env_b)
    return (not diffs), diffs


def speedup_statement(
    a: dict, b: dict, a_label: str, b_label: str
) -> str:
    """A comparison sentence, or a refusal to make one.

    Design §12.5: "X is N times faster" is allowed only between stable
    configurations measured in the same run with the same env manifest, and must
    carry the interval. Anything else is not stated.
    """
    if not a.get("stability", {}).get("stable") or not b.get("stability", {}).get(
        "stable"
    ):
        return (
            f"No comparison between {a_label} and {b_label}: at least one "
            f"configuration is unstable (see stability reasons). An unstable "
            f"measurement is not quoted as a result."
        )
    ok, diffs = is_comparable(a, b)
    if not ok:
        return (
            f"No comparison between {a_label} and {b_label}: the runs differ in "
            f"{', '.join(diffs)}, so the difference cannot be attributed to the "
            f"configuration."
        )
    ta = a["throughput_pages_per_s"]["median_of_medians"]
    tb = b["throughput_pages_per_s"]["median_of_medians"]
    ci_a = a.get("queue_inclusive_p95_ms", {}).get("ci95")
    ci_b = b.get("queue_inclusive_p95_ms", {}).get("ci95")
    if not tb:
        return f"No comparison: {b_label} measured zero throughput."
    ratio = ta / tb
    return (
        f"{a_label} sustained {ta:.3f} pages/s against {tb:.3f} pages/s for "
        f"{b_label} on the same host in the same run, a factor of {ratio:.2f} "
        f"(queue-inclusive p95 {ci_a} ms vs {ci_b} ms, 95% bootstrap CI over "
        f"{a['n_trials']} trials)."
    )


def summarize_env(env: dict) -> str:
    """The stamp that goes on every chart and table (design §11.3)."""
    return (
        f"{env.get('cpu_brand', 'unknown CPU')} "
        f"{env.get('cpu_perf_levels', {}).get('perflevel0', '?')}P+"
        f"{env.get('cpu_perf_levels', {}).get('perflevel1', '?')}E, "
        f"{env.get('os_version', '?')}, "
        f"torch {env.get('library_versions', {}).get('torch', '?')}, "
        f"ORT {env.get('library_versions', {}).get('onnxruntime', '?')}, "
        f"measured {env.get('captured_at', '?')[:19]}, "
        f"host {env.get('host_fingerprint', '?')}"
    )


def raw_sample_path(trial: TrialResult, out_dir) -> Path:
    """Write one trial's raw samples. Every reported number traces to one of these."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{trial.config_key.replace('/', '_')}_t{trial.trial}.json"
    path.write_text(
        json.dumps(
            {
                "config": trial.config_key,
                "trial": trial.trial,
                "summary": trial.summary(),
                "samples": [asdict(s) for s in trial.samples],
            },
            indent=2,
        )
    )
    return path


__all__ = [
    "BATCH_SIZES",
    "MEASURED_PAGES",
    "PROTOCOL_VERSION",
    "TRIALS",
    "WARMUP_PAGES",
    "Config",
    "Sample",
    "TrialResult",
    "aggregate",
    "bootstrap_ci",
    "is_comparable",
    "judge_stability",
    "percentile",
    "poisson_arrivals",
    "raw_sample_path",
    "speedup_statement",
    "summarize_env",
    "tier1_matrix",
    "tier2_matrix",
    "trial_order",
]
