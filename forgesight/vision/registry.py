"""Candidate registry: turning pinned artifacts into resolvable configurations.

A `Candidate` (design §6) is the unit a release is made of: one model artifact,
one preprocessing profile, one runtime profile, one score threshold, one label
map. Its hash covers all of them, so two candidates that would produce different
detections cannot share a hash, and a release row is enough to re-derive exactly
what was shipped.

The registry owns the naming convention for built-in candidates. Evaluation and
benchmarking both resolve candidates through it, so a report never has to
hard-code a path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forgesight.settings import Settings
from forgesight.vision.preprocess import Preprocessor, make_profile
from forgesight.vision.runtimes.base import read_id2label
from forgesight.vision.types import (
    Candidate,
    ModelArtifact,
    Pool,
    PreprocessProfile,
    RuntimeProfile,
    label_map_hash,
)

LOCK_PATH = "models.lock.json"
REFERENCE_RESIZE = "pil_bilinear"
GRAPH_OPT = "ORT_ENABLE_ALL"


@dataclass(frozen=True, slots=True)
class BuiltCandidate:
    name: str
    candidate: Candidate
    role: str  # reference | candidate | negative_control
    note: str = ""


def load_lock(settings: Settings) -> dict:
    path = settings.data_dir.parent / LOCK_PATH
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / LOCK_PATH
    return json.loads(path.read_text())


def artifact_for(
    lock: dict, name: str, settings: Settings, fmt: str = "safetensors"
) -> ModelArtifact:
    """Resolve one pinned artifact by name.

    For the ONNX formats the sha comes from the export's own record rather than
    the source weights, because the exported file is what actually gets loaded
    and verified at inference time (AT-22).
    """
    entry = lock["models"][name]
    if fmt == "safetensors":
        path = settings.models_dir / name
        sha = entry["files"]["model.safetensors"]["sha256"]
        parent = None
        exporter = opset = None
    else:
        record_path = settings.artifacts_dir / f"{name}.export.json"
        if not record_path.exists():
            raise FileNotFoundError(
                f"no export record for {name} ({fmt}); run scripts/export_models.py"
            )
        rec = json.loads(record_path.read_text())
        path = settings.artifacts_dir / rec["file"]
        sha = rec["sha256"]
        parent = entry["files"]["model.safetensors"]["sha256"]
        exporter = rec.get("exporter")
        opset = rec.get("opset")

    return ModelArtifact(
        name=name,
        path=str(path),
        sha256=sha,
        format=fmt,  # type: ignore[arg-type]
        repo_id=entry["repo_id"],
        revision=entry["revision"],
        license=entry["license"],
        parent_sha=parent,
        exporter=exporter,
        opset=opset,
    )


def _has_onnx(name: str, settings: Settings) -> bool:
    return (settings.artifacts_dir / f"{name}.export.json").exists()


def registry(settings: Settings) -> list[BuiltCandidate]:
    """Every candidate the system knows how to build right now.

    The reference is pinned as heron on torch with the reference preprocessing,
    because every gate is measured relative to it (design §10). Comparing a
    candidate to whatever happens to be active instead would let a series of
    small permitted regressions accumulate unnoticed.
    """
    lock = load_lock(settings)
    out: list[BuiltCandidate] = []

    heron_dir = settings.models_dir / "heron"
    if heron_dir.exists():
        id2label = read_id2label(heron_dir)
        lmh = label_map_hash(id2label)

        def make(
            model: str,
            runtime: str,
            resize: str = REFERENCE_RESIZE,
            threshold: float | None = None,
            tile: bool = False,
            threads: int | None = None,
        ) -> BuiltCandidate:
            threads = threads or settings.worker_threads
            if runtime == "torch":
                art = artifact_for(lock, model, settings, "safetensors")
                rp = RuntimeProfile(
                    runtime="torch",
                    intra_op_threads=threads,
                    inter_op_threads=1,
                    graph_opt_level=GRAPH_OPT,
                    max_batch=settings.max_batch,
                    batch_window_ms=settings.batch_window_ms,
                )
            else:
                art = artifact_for(lock, model, settings, "onnx-fp32")
                rp = RuntimeProfile(
                    runtime="onnxruntime",
                    intra_op_threads=threads,
                    inter_op_threads=1,
                    graph_opt_level=GRAPH_OPT,
                    max_batch=settings.max_batch,
                    batch_window_ms=settings.batch_window_ms,
                )
            pp = make_profile(
                resize, target=settings.target_size, render_dpi=settings.render_dpi,
                id2label=id2label,
            )
            cand = Candidate(
                name=f"{model}-{runtime}-{resize}",
                artifact=art,
                preprocess=pp,
                runtime=rp,
                score_threshold=(
                    threshold if threshold is not None else settings.score_threshold
                ),
                label_map_hash=lmh,
                tile_enabled=tile,
            )
            return BuiltCandidate(cand.name, cand, "candidate")

        out.append(
            BuiltCandidate(
                "ref-heron-torch",
                make("heron", "torch").candidate,
                "reference",
                "pinned reference; every gate is measured against this",
            )
        )
        out.append(make("egret-medium", "torch"))
        if _has_onnx("heron", settings):
            out.append(make("heron", "onnxruntime"))
        if _has_onnx("egret-medium", settings):
            out.append(make("egret-medium", "onnxruntime"))
        # The negative control must be *worse*, and 320px resizing is the
        # cheapest way to guarantee that while keeping everything else equal.
        out.append(
            BuiltCandidate(
                "negctl-heron-torch-320",
                _resize_candidate(make("heron", "torch").candidate, 320, lmh, settings),
                "negative_control",
                "320px resize; the gates must block this (AT-13)",
            )
        )
    return out


def _resize_candidate(
    base: Candidate, target: int, lmh: str, settings: Settings
) -> Candidate:
    pp = PreprocessProfile(
        resize=base.preprocess.resize,
        target=target,
        render_dpi=base.preprocess.render_dpi,
        tile_long_side_px=base.preprocess.tile_long_side_px,
        tile_aspect=base.preprocess.tile_aspect,
        tile_overlap=base.preprocess.tile_overlap,
        label_map_hash=base.preprocess.label_map_hash,
        processor_class=base.preprocess.processor_class,
    )
    return Candidate(
        name=f"{base.artifact.name}-{base.runtime.runtime}-negctl{target}",
        artifact=base.artifact,
        preprocess=pp,
        runtime=base.runtime,
        score_threshold=base.score_threshold,
        label_map_hash=lmh,
    )


def get(name: str, settings: Settings) -> Candidate:
    for bc in registry(settings):
        if bc.name == name:
            return bc.candidate
    known = [bc.name for bc in registry(settings)]
    raise KeyError(f"unknown candidate {name!r}; known: {known}")


def preprocessor_for(candidate: Candidate) -> Preprocessor:
    return Preprocessor(candidate.preprocess, tile_enabled=candidate.tile_enabled)


def pool_for(candidate: Candidate) -> Pool:
    return candidate.artifact.pool
