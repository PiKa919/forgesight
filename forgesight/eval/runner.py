"""Evaluation runner: the thing that turns a candidate into gate verdicts.

Order matters and is not arbitrary:

1. Resolve the candidate and the **pinned reference** from the registry, not
   from the active release. A gate measured against a moving target is not a
   gate.
2. Require the datasets to exist, with their manifest hashes recorded. If a
   dataset is missing the run fails rather than skipping the gate, because a
   skipped quality gate is indistinguishable from a passed one.
3. Choose the operating threshold on the calibration split, then freeze it.
4. Score clean and shift, compute runtime agreement, then judge G1-G6.
"""

from __future__ import annotations

import time

import numpy as np

from forgesight.eval import dataset as ds
from forgesight.eval.coco_eval import (
    PagePrediction,
    best_f1_threshold,
    detection_agreement,
    evaluate,
    precision_recall,
)
from forgesight.eval.gates import EvaluationResult, judge, negative_control_blocked
from forgesight.settings import Settings
from forgesight.storage.object_store import ObjectStore
from forgesight.vision.cancel import NeverCancel
from forgesight.vision.postprocess import postprocess_batch
from forgesight.vision.preprocess import Preprocessor
from forgesight.vision.registry import preprocessor_for
from forgesight.vision.types import Candidate, Detection

CLEAN = "synth-clean"
SHIFT = "synth-scan"
CALIB = "calib"
TEST = "test"


def _load_model(candidate: Candidate, settings: Settings):
    from forgesight.vision.runtimes.base import OrtModel, TorchModel

    # The candidate already carries the exact artifact identity it was built
    # with, including the export sha, so nothing is re-resolved here: loading a
    # different file than the candidate names is precisely the bug AT-22 guards.
    art = candidate.artifact
    runtime = TorchModel if candidate.runtime.runtime == "torch" else OrtModel
    return runtime(art, candidate.runtime)


def predict_pages(
    model, candidate: Candidate, store: ObjectStore, pages: list, pool
) -> list[PagePrediction]:
    """Run the candidate over a set of dataset pages, one micro-batch at a time."""
    proc: Preprocessor = preprocessor_for(candidate)
    out: list[PagePrediction] = []
    keys = ds.page_object_keys(pool, [p.page_id for p in pages])

    for truth in pages:
        rgb = ds.page_bytes(store, keys[truth.page_id])
        items = proc(rgb)
        dets: list[Detection] = []
        for start in range(0, len(items), 4):
            chunk = items[start : start + 4]
            raw = model.infer(np.stack([c.tensor for c in chunk]), NeverCancel())
            for row in postprocess_batch(
                raw, chunk, model.id2label, candidate.score_threshold
            ):
                dets.extend(row)
        out.append(PagePrediction(truth.page_id, dets))
    return out


def candidate_from_row(row, settings: Settings, pool) -> Candidate:
    """Rebuild the immutable Candidate value from its stored profile rows.

    The pool is a required argument rather than being opened from the ambient
    environment. A caller that already holds a connection pool -- the worker, the
    API -- must not have a second one silently pointed somewhere else, which is
    how code ends up reading and writing two different databases.
    """
    import json as _json

    from forgesight.vision.types import ModelArtifact, PreprocessProfile, RuntimeProfile

    with pool.connection() as conn:
        art = conn.fetchone(
            "SELECT * FROM model_artifact WHERE workspace_id = ? AND id = ?",
            (row["workspace_id"], row["artifact_id"]),
        )
        pre = conn.fetchone(
            "SELECT * FROM preprocess_profile WHERE id = ?", (row["preprocess_id"],)
        )
        rt = conn.fetchone(
            "SELECT * FROM runtime_profile WHERE id = ?", (row["runtime_id"],)
        )

    m = ModelArtifact(
        name=art["name"], path=art["path"], sha256=art["sha256"], format=art["format"],
        repo_id=art["repo_id"], revision=art["revision"], license=art["license"],
        parent_sha=art["parent_sha"], exporter=art["exporter"], opset=art["opset"],
    )
    return Candidate(
        name=row["name"], artifact=m,
        preprocess=PreprocessProfile.from_dict(_json.loads(pre["body"])),
        runtime=RuntimeProfile.from_dict(_json.loads(rt["body"])),
        score_threshold=float(row["score_threshold"]),
        label_map_hash=row["label_map_hash"] or "",
        tile_enabled=bool(row["tile_enabled"]),
    )


def run_evaluation(
    candidate_row, settings: Settings, pool=None, store: ObjectStore | None = None
) -> EvaluationResult:
    ws = candidate_row["workspace_id"]
    from forgesight.db.pool import get_pool
    from forgesight.storage.object_store import build_store

    pool = pool or get_pool()
    store = store or build_store(settings)

    cand = candidate_from_row(candidate_row, settings, pool)
    reference = _reference_candidate(settings, pool, ws)
    if reference is None:
        raise ValueError("no pinned reference candidate is registered")

    clean_test = ds.load_truth(pool, ws, CLEAN, TEST)
    clean_calib = ds.load_truth(pool, ws, CLEAN, CALIB)
    shift_test = ds.load_truth(pool, ws, SHIFT, TEST)
    for label, truth in ((CLEAN, clean_test), (SHIFT, shift_test)):
        if not truth:
            raise ValueError(
                f"dataset {label} is not built; run scripts/build_datasets.py. "
                f"A missing dataset must fail the run, not skip a gate."
            )

    started = time.monotonic()
    cand_model = _load_model(cand, settings)
    ref_model = _load_model(reference, settings)

    # Operating point from the calibration split, then frozen for everything.
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    calib_preds = {t: predict_pages(cand_model, _at(cand, t), store, clean_calib, pool)
                  for t in grid}
    best_t, f1 = best_f1_threshold(clean_calib, calib_preds)
    frozen = _at(cand, best_t)
    ref_frozen = _at(reference, best_t)

    cand_clean = evaluate(clean_test, predict_pages(cand_model, frozen, store, clean_test, pool),
                          ds.category_ids())
    ref_clean = evaluate(clean_test, predict_pages(ref_model, ref_frozen, store, clean_test, pool),
                         ds.category_ids())
    cand_shift = evaluate(shift_test, predict_pages(cand_model, frozen, store, shift_test, pool),
                          ds.category_ids())
    ref_shift = evaluate(shift_test, predict_pages(ref_model, ref_frozen, store, shift_test, pool),
                         ds.category_ids())

    agreement = _agreement(
        cand, reference, cand_model, ref_model, frozen, ref_frozen, store, clean_test, pool
    )

    metrics = {
        "operating_threshold": best_t,
        "calib_f1_at_iou50": round(f1, 5),
        "precision_recall_test": _pr(cand_model, frozen, store, clean_test, pool),
        "clean": {"candidate": cand_clean, "reference": ref_clean},
        "shift": {"candidate": cand_shift, "reference": ref_shift},
        "agreement_with_reference": agreement,
        "elapsed_s": round(time.monotonic() - started, 2),
    }

    manifests = _manifests(pool, ws)
    provenance_ok, provenance_detail = _provenance(cand, pool, ws, manifests)
    verdicts = judge(
        candidate=cand, reference=reference,
        clean={"candidate": cand_clean, "reference": ref_clean},
        shift={"candidate": cand_shift, "reference": ref_shift},
        agreement=agreement, peak_bytes=None, budget_bytes=settings.worker_mem_budget,
        provenance_ok=provenance_ok, provenance_detail=provenance_detail,
    )
    metrics["negative_control_blocked"] = negative_control_blocked(verdicts)
    return EvaluationResult(
        candidate_name=cand.name, metrics=metrics, verdicts=verdicts,
        dataset_versions=manifests, env_manifest=_env_stub(),
    )


def _at(cand: Candidate, threshold: float) -> Candidate:
    return Candidate(
        name=cand.name, artifact=cand.artifact, preprocess=cand.preprocess,
        runtime=cand.runtime, score_threshold=threshold,
        label_map_hash=cand.label_map_hash, tile_enabled=cand.tile_enabled,
    )


def _reference_candidate(settings: Settings, pool, ws: str) -> Candidate | None:
    from forgesight.vision.registry import registry

    for bc in registry(settings):
        if bc.role != "reference":
            continue
        ph = "?" if pool.dialect.value == "sqlite" else "%s"
        with pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT * FROM candidate WHERE workspace_id = {ph} "
                f"AND candidate_hash = {ph}",
                (ws, bc.candidate.hash),
            )
        if row is not None:
            return candidate_from_row(row, settings, pool)
    return None


def _agreement(
    cand, reference, cand_model, ref_model, cand_at, ref_at, store, truths, pool
) -> float | None:
    """Mean detection agreement between the candidate and the reference.

    Returns None when the two differ only in model weights, because a candidate
    that is a different model is supposed to disagree with the reference, and
    gating it on agreeing would be meaningless. G4 is about runtime and
    preprocessing parity, so it applies exactly when those differ and both sides
    run the same weights.
    """
    same_weights = cand.artifact.sha256 == reference.artifact.sha256
    same_path = (
        cand.runtime.runtime == reference.runtime.runtime
        and cand.preprocess.hash == reference.preprocess.hash
    )
    if not same_weights or same_path:
        return None
    a = {p.page_id: p.detections for p in predict_pages(cand_model, cand_at, store, truths, pool)}
    b = {p.page_id: p.detections for p in predict_pages(ref_model, ref_at, store, truths, pool)}
    scores = [
        detection_agreement(a.get(t.page_id, []), b.get(t.page_id, []))
        for t in truths
    ]
    return sum(scores) / len(scores) if scores else None


def _pr(model, cand, store, truths, pool) -> dict:
    preds = predict_pages(model, cand, store, truths, pool)
    p, r = precision_recall(truths, preds, cand.score_threshold)
    return {"precision": round(p, 5), "recall": round(r, 5)}


def _manifests(pool, ws: str) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.fetchall(
            "SELECT name, semver, manifest_hash FROM dataset_version "
            "WHERE workspace_id = ? AND name IN (?, ?)",
            (ws, CLEAN, SHIFT),
        )
    return [{"name": r["name"], "semver": r["semver"],
             "manifest_hash": r["manifest_hash"]} for r in rows]


def _provenance(cand, pool, ws, manifests) -> tuple[bool, str]:
    from pathlib import Path

    from forgesight.vision.runtimes.base import verify_artifact

    if not manifests:
        return False, "no evaluation dataset is registered"
    try:
        verify_artifact(Path(cand.artifact.weights_path), cand.artifact.sha256)
    except Exception as exc:
        return False, f"artifact sha mismatch: {exc}"
    if not cand.artifact.revision or cand.artifact.revision == "main":
        return False, "artifact is not pinned to a revision sha"
    return True, (
        f"artifact {Path(cand.artifact.path).name} sha verified, "
        f"pinned to {cand.artifact.revision[:12]}, "
        f"{len(manifests)} dataset manifest hash(es) recorded"
    )


def _env_stub() -> dict:
    from forgesight.bench.env_capture import capture_env

    return capture_env().to_dict()
