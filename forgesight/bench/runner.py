"""Benchmark runner (design §12.2).

Runs the protocol in-process for speed, with one caveat that matters: the
throughput it reports is the *pipeline* throughput -- preprocess, inference,
postprocess and persistence all counted -- not bare forward-pass time. That is
the number a deployment cares about, and it is the slower one.

The runner refuses by default when the environment is unfavourable. Passing
`allow_unfavourable` records the refusal reasons in the report rather than
discarding the run, because on a machine that cannot be put on AC power the
useful output is a labelled measurement, not nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forgesight.bench import protocol as proto
from forgesight.bench.env_capture import EnvManifest, Refusal, capture_env, check_runnable
from forgesight.settings import Settings
from forgesight.storage.object_store import ObjectStore
from forgesight.vision.cancel import NeverCancel
from forgesight.vision.postprocess import postprocess_batch
from forgesight.vision.preprocess import Preprocessor, make_profile
from forgesight.vision.registry import artifact_for, load_lock
from forgesight.vision.runtimes.base import OrtModel, TorchModel
from forgesight.vision.types import Candidate, RuntimeProfile
from forgesight.worker.memprobe import RssSampler


class Refused(Exception):
    def __init__(self, refusal: Refusal):
        super().__init__(f"{refusal.reason}: {refusal.detail}")
        self.refusal = refusal


@dataclass(slots=True)
class RunOutcome:
    trials: list[proto.TrialResult]
    env: EnvManifest
    raw_files: list[str]
    refusals: list[str]
    memory: dict


def _build_candidate(cfg: proto.Config, settings: Settings) -> Candidate:
    lock = load_lock(settings)
    fmt = "safetensors" if cfg.runtime == "torch" else "onnx-fp32"
    art = artifact_for(lock, cfg.model, settings, fmt)
    return Candidate(
        name=cfg.key(),
        artifact=art,
        preprocess=make_profile(cfg.resize, target=settings.target_size,
                                render_dpi=settings.render_dpi),
        runtime=RuntimeProfile(
            runtime=cfg.runtime,
            intra_op_threads=cfg.threads,
            inter_op_threads=1,
            graph_opt_level="ORT_ENABLE_ALL",
            max_batch=cfg.batch,
            batch_window_ms=settings.batch_window_ms,
        ),
        score_threshold=settings.score_threshold,
        label_map_hash="",
        tile_enabled=cfg.tile,
    )


def _load(cfg: proto.Config, settings: Settings):
    cand = _build_candidate(cfg, settings)
    cls = TorchModel if cfg.runtime == "torch" else OrtModel
    return cand, cls(cand.artifact, cand.runtime)


def _pages(store: ObjectStore, pool, limit: int) -> list[str]:
    from forgesight.eval.dataset import page_object_keys

    with pool.connection() as conn:
        rows = conn.fetchall(
            "SELECT id FROM page ORDER BY id LIMIT ?", (limit,)
        )
    return page_object_keys(pool, [r["id"] for r in rows])


def run_config(
    cfg: proto.Config,
    settings: Settings,
    store: ObjectStore,
    pool,
    images: list[np.ndarray],
    raw_dir: Path,
    trial: int,
) -> proto.TrialResult:
    """One trial of one configuration over the measured page set."""
    cand, model = _load(cfg, settings)
    proc = Preprocessor(cand.preprocess, tile_enabled=cand.tile_enabled)

    # Warmup, discarded. The first inference pays for lazy init, allocator
    # growth and first-touch faults; including it makes a fast config look slow.
    for img in images[: proto.WARMUP_PAGES]:
        items = proc(img)
        for start in range(0, len(items), cfg.batch):
            chunk = items[start : start + cfg.batch]
            model.infer(np.stack([c.tensor for c in chunk]), NeverCancel())

    samples: list[proto.Sample] = []
    measured = images[: proto.MEASURED_PAGES]
    steady = 0
    with RssSampler(settings.rss_sample_ms) as sampler:
        t_start = time.monotonic()
        for i, img in enumerate(measured):
            t0 = time.monotonic()
            items = proc(img)
            t_pre = time.monotonic()
            dets: list = []
            for start in range(0, len(items), cfg.batch):
                chunk = items[start : start + cfg.batch]
                raw_out = model.infer(np.stack([c.tensor for c in chunk]), NeverCancel())
                for row in postprocess_batch(
                    raw_out, chunk, model.id2label, cand.score_threshold
                ):
                    dets.extend(row)
            t_post = time.monotonic()
            now = time.monotonic()
            samples.append(proto.Sample(
                page_id=f"p{i}",
                queue_inclusive_ms=(now - t0) * 1000.0,
                service_ms=(now - t0) * 1000.0,  # single process: no queue to wait in
                preprocess_ms=(t_pre - t0) * 1000.0,
                infer_ms=(t_post - t_pre) * 1000.0,
                postprocess_ms=(now - t_post) * 1000.0,
                peak_rss=sampler.peak,
            ))
            steady = sampler.peak
        elapsed = time.monotonic() - t_start

    qi = [s.queue_inclusive_ms for s in samples]
    svc = [s.service_ms for s in samples]
    result = proto.TrialResult(
        config_key=cfg.key(),
        trial=trial,
        throughput_pages_per_s=len(samples) / elapsed if elapsed else 0.0,
        queue_inclusive_p50=proto.percentile(qi, 0.50),
        queue_inclusive_p95=proto.percentile(qi, 0.95),
        queue_inclusive_p99=proto.percentile(qi, 0.99),
        service_p50=proto.percentile(svc, 0.50),
        service_p95=proto.percentile(svc, 0.95),
        peak_rss_bytes=sampler.peak,
        steady_rss_bytes=steady,
        n_pages=len(samples),
        samples=samples,
    )
    return result


def run(
    configs: list[proto.Config],
    settings: Settings,
    store: ObjectStore,
    pool,
    trials: int = proto.TRIALS,
    pages: int = proto.MEASURED_PAGES,
    allow_unfavourable: bool = False,
    raw_dir: Path | None = None,
    cooldown_s: float = proto.COOLDOWN_S,
) -> RunOutcome:
    """Run the matrix, honouring the protocol's refusal and stability rules."""
    env = capture_env()
    refusal = check_runnable(
        require_ac=settings.bench_require_ac_power,
        min_free_fraction=settings.bench_min_free_mem_fraction,
    )
    refusals: list[str] = []
    if refusal is not None:
        if not allow_unfavourable:
            raise Refused(refusal)
        refusals.append(f"{refusal.reason}: {refusal.detail}")

    from forgesight.eval.dataset import page_bytes

    keys = _pages(store, pool, pages + proto.WARMUP_PAGES)
    if not keys:
        raise ValueError("no pages to benchmark; run scripts/build_datasets.py first")
    images = [page_bytes(store, k) for k in list(keys.values())]

    raw_dir = raw_dir or (settings.reports_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[proto.TrialResult] = []
    raw_files: list[str] = []
    for trial in range(trials):
        # Randomised per trial so a slow drift cannot systematically favour
        # whichever configuration happened to go first.
        for cfg in proto.trial_order(configs, trial):
            if cooldown_s and results:
                time.sleep(cooldown_s)
            t = run_config(cfg, settings, store, pool, images, raw_dir, trial)
            results.append(t)
            raw_files.append(str(proto.raw_sample_path(t, raw_dir)))

    memory = _calibrate(settings, store, pool, images[:8], raw_dir)
    return RunOutcome(
        trials=results, env=env, raw_files=raw_files, refusals=refusals,
        memory=memory,
    )


def _calibrate(settings, store, pool, images, raw_dir) -> dict:
    """Measure the marginal cost of one 640x640 input and fit the admission model.

    A finding worth stating, because it changed the method: RSS is a high-water
    mark. Python and both runtimes do not return freed memory to the OS, so peak
    RSS is monotonic in the largest batch size seen so far. Measuring
    `peak(b)` for b in 1,2,4,8 in ascending order therefore gives a flat or
    rising staircase that a `M0 + b*m_item` line fits badly -- the first
    measurement of a shape keeps the high-water mark of every earlier one.

    So the baseline is taken *after* a warm pass at the largest batch size, and
    each measurement is the peak above that baseline. What remains is the
    genuinely marginal cost of one more item. Both the raw high-water numbers
    and the baseline-relative ones are reported, so the difference is visible
    rather than hidden.
    """
    from forgesight.worker.memprobe import Admission, fit

    cfg = proto.Config(name="calib", model="egret-medium", runtime="onnxruntime",
                       resize="pil_bilinear", batch=1, threads=4)
    cand, model = _load(cfg, settings)
    proc = Preprocessor(cand.preprocess)
    prepared = [proc(img)[0] for img in images]

    with RssSampler(1) as sampler:
        baseline = sampler.peak
        # Raise the high-water mark first, so later measurements are not simply
        # re-reading it.
        for b in proto.BATCH_SIZES:
            model.infer(np.stack([prepared[0].tensor] * b), NeverCancel())
        baseline = sampler.peak

    samples: dict[int, int] = {}
    high_water: dict[int, int] = {}
    for b in proto.BATCH_SIZES:
        peak_delta = 0
        for item in prepared:
            with RssSampler(1) as s:
                model.infer(np.stack([item.tensor] * b), NeverCancel())
            peak_delta = max(peak_delta, s.peak - baseline)
        samples[b] = peak_delta
        high_water[b] = peak_delta + baseline

    try:
        cal = fit(samples)
    except ValueError:
        return {"error": "calibration produced no samples"}
    admission = Admission(calibration=cal, budget_bytes=settings.worker_mem_budget)
    for b, peak in samples.items():
        admission.observe(b, peak)
    return {
        **cal.as_dict(),
        **admission.stats(),
        "budget_bytes": settings.worker_mem_budget,
        "baseline_after_largest_batch_bytes": baseline,
        "marginal_peak_above_baseline": {str(k): v for k, v in sorted(samples.items())},
        "high_water_including_baseline": {str(k): v for k, v in sorted(high_water.items())},
        "method": (
            "baseline taken after a warm pass at every batch size, so each figure is "
            "the marginal peak above it; measuring ascending batch sizes against a "
            "cold baseline would report a high-water staircase that no linear "
            "model fits"
        ),
    }


__all__ = ["Refused", "RunOutcome", "run", "run_config"]
