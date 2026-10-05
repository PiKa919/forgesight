"""Candidates, datasets, evaluation and system status (design §11.2)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status

from forgesight.api.deps import Principal, get_pool_singleton, operator
from forgesight.api.repo import WorkspaceRepo, _iso
from forgesight.api.schemas import (
    CandidateCreate,
    CandidateOut,
    DatasetOut,
    EvaluationOut,
    GateVerdict,
    PoolStatus,
    SystemStatus,
)
from forgesight.settings import get_settings
from forgesight.vision.registry import registry

router = APIRouter(prefix="/v1")


def repo() -> WorkspaceRepo:
    return WorkspaceRepo(get_pool_singleton())


@router.get("/candidates", response_model=list[CandidateOut])
def list_candidates(p: Principal = Depends(operator)) -> list[CandidateOut]:
    return [_candidate_out(get_pool_singleton(), r) for r in repo().list_candidates(p.workspace_id)]


@router.post("/candidates", response_model=CandidateOut, status_code=201)
def create_candidate(
    body: CandidateCreate, p: Principal = Depends(operator)
) -> CandidateOut:
    """Create a candidate from existing profiles.

    An unknown artifact or profile is a 422 rather than a fresh row, so a typo
    cannot quietly produce a configuration nobody intended.
    """
    r = repo()
    with get_pool_singleton().connection() as conn:
        art = conn.fetchone(
            "SELECT * FROM model_artifact WHERE workspace_id = ? AND sha256 = ?",
            (p.workspace_id, body.artifact_sha256),
        )
        pre = conn.fetchone(
            "SELECT * FROM preprocess_profile WHERE profile_hash = ?",
            (body.preprocess_hash,),
        )
        rt = conn.fetchone(
            "SELECT * FROM runtime_profile WHERE profile_hash = ?",
            (body.runtime_hash,),
        )
    if art is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown artifact sha256")
    if pre is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown preprocess profile")
    if rt is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown runtime profile")

    from forgesight.vision.types import Candidate

    cand = Candidate(
        name=body.name,
        artifact=_artifact_from_row(art),
        preprocess=pre["profile_hash"],
        runtime=rt["profile_hash"],
        score_threshold=body.score_threshold,
        label_map_hash="",
        tile_enabled=body.tile_enabled,
    )
    del cand  # the hash is assembled from the stored rows below
    import hashlib

    payload = {
        "artifact_sha": body.artifact_sha256,
        "preprocess": pre["body"],
        "runtime": rt["body"],
        "score_threshold": body.score_threshold,
        "tile_enabled": body.tile_enabled,
    }
    chash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    existing = r.find_candidate_by_hash(p.workspace_id, chash)
    if existing:
        return _candidate_out(get_pool_singleton(), existing)

    from forgesight.ledger.claims import new_id

    cid = new_id("cand")
    with get_pool_singleton().write() as conn:
        conn.execute(
            "INSERT INTO candidate(id, workspace_id, name, candidate_hash, artifact_id, "
            "preprocess_id, runtime_id, score_threshold, label_map_hash, tile_enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, p.workspace_id, body.name, chash, art["id"], pre["id"], rt["id"],
             body.score_threshold, "", body.tile_enabled),
        )
    return _candidate_out(get_pool_singleton(), r.get_candidate(p.workspace_id, cid))


def _artifact_from_row(row):
    from forgesight.vision.types import ModelArtifact

    return ModelArtifact(
        name=row["name"], path=row["path"], sha256=row["sha256"], format=row["format"],
        repo_id=row["repo_id"], revision=row["revision"], license=row["license"],
        parent_sha=row["parent_sha"], exporter=row["exporter"], opset=row["opset"],
    )


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(p: Principal = Depends(operator)) -> list[DatasetOut]:
    return [_dataset_out(x) for x in repo().list_datasets(p.workspace_id)]


@router.get("/datasets/{name}", response_model=DatasetOut)
def get_dataset(name: str, p: Principal = Depends(operator)) -> DatasetOut:
    row = repo().get_dataset(p.workspace_id, name)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return _dataset_out(row)


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationOut)
def get_evaluation(evaluation_id: str, p: Principal = Depends(operator)) -> EvaluationOut:
    row = repo().get_evaluation(p.workspace_id, evaluation_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evaluation not found")
    return EvaluationOut(
        id=row["id"],
        candidate_id=row["candidate_id"],
        candidate_name=row["candidate_name"] or "unknown",
        reference_id=row["reference_id"],
        status=row["status"],
        metrics=json.loads(row["metrics"]) if row["metrics"] else None,
        gates=[GateVerdict(**g) for g in json.loads(row["gates"])] if row["gates"] else None,
        created_at=_iso(row["created_at"]) or "",
        finished_at=_iso(row["finished_at"]),
        failure_detail=row["failure_detail"],
    )


@router.post("/candidates/{candidate_id}/evaluate", response_model=EvaluationOut, status_code=202)
def evaluate(candidate_id: str, p: Principal = Depends(operator)) -> EvaluationOut:
    """Queue an evaluation. Disabled in the public demo (design §17)."""
    s = get_settings()
    if s.mode == "public":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "evaluation is disabled in the public demo; recorded reports are shown instead",
        )
    if not s.enable_evaluation:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "evaluation is disabled")
    r = repo()
    cand = r.get_candidate(p.workspace_id, candidate_id)
    if cand is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "candidate not found")

    from forgesight.eval.gates import evaluate_candidate
    from forgesight.ledger.claims import new_id

    eid = new_id("ev")
    with get_pool_singleton().write() as conn:
        conn.execute(
            "INSERT INTO evaluation_run(id, workspace_id, candidate_id, status, "
            "dataset_versions, code_sha, env_manifest) VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (eid, p.workspace_id, candidate_id, "[]", _git_sha(), "{}"),
        )
    result = evaluate_candidate(cand, s, pool=get_pool_singleton())
    with get_pool_singleton().write() as conn:
        conn.execute(
            "UPDATE evaluation_run SET status = 'done', metrics = ?, gates = ?, "
            "dataset_versions = ?, env_manifest = ? WHERE id = ?",
            (json.dumps(result.metrics), json.dumps(result.verdicts),
             json.dumps(result.dataset_versions), json.dumps(result.env_manifest), eid),
        )
    return get_evaluation(eid, p)


@router.get("/system/status", response_model=SystemStatus)
def system_status(p: Principal = Depends(operator)) -> SystemStatus:
    import psutil

    s = get_settings()
    r = repo()
    pool = get_pool_singleton()
    depth = r.queue_depth()

    def free_bytes() -> int | None:
        try:
            return psutil.virtual_memory().available
        except Exception:
            return None

    return SystemStatus(
        mode=s.mode,
        dialect=pool.dialect.value,
        queue_depth=[
            PoolStatus(pool=name, queued=v.get("queued", 0), running=v.get("running", 0))
            for name, v in sorted(depth.items())
        ],
        queue_limit=s.queue_depth_limit,
        worker_mem_budget_bytes=s.worker_mem_budget,
        memory_safety_margin=s.memory_safety_margin,
        admission_mispredictions=r.mispredictions(p.workspace_id),
        host_free_bytes=free_bytes(),
        git_sha=_git_sha(),
    )


def _candidate_out(pool, row) -> CandidateOut:
    with pool.connection() as conn:
        art = conn.fetchone(
            "SELECT * FROM model_artifact WHERE workspace_id = ? AND id = ?",
            (row["workspace_id"], row["artifact_id"]),
        )
        pre = conn.fetchone(
            "SELECT body FROM preprocess_profile WHERE id = ?", (row["preprocess_id"],)
        )
        rt = conn.fetchone(
            "SELECT body FROM runtime_profile WHERE id = ?", (row["runtime_id"],)
        )
    import json as _json

    pp = _json.loads(pre["body"]) if pre else {}
    rp = _json.loads(rt["body"]) if rt else {}
    return CandidateOut(
        id=row["id"],
        name=row["name"],
        candidate_hash=row["candidate_hash"],
        artifact_name=art["name"] if art else "unknown",
        artifact_format=art["format"] if art else "unknown",
        artifact_sha256=art["sha256"] if art else "",
        repo_id=art["repo_id"] if art else "",
        revision=art["revision"] if art else "",
        runtime=rp.get("runtime", "unknown"),
        resize=pp.get("resize", "unknown"),
        target=int(pp.get("target", 640)),
        threads=int(rp.get("intra_op_threads", 0)),
        score_threshold=float(row["score_threshold"]),
        tile_enabled=bool(row["tile_enabled"]),
        valid=bool(row["valid"]),
        invalid_reason=row["invalid_reason"],
    )


def _dataset_out(row) -> DatasetOut:
    return DatasetOut(
        id=row["id"],
        name=row["name"],
        semver=row["semver"],
        manifest_hash=row["manifest_hash"],
        generator=row["generator"],
        generator_version=row["generator_version"],
        license=row["license"],
        synthetic=bool(row["synthetic"]),
        counts=json.loads(row["counts"]),
    )


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_root(),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


__all__ = ["registry", "router"]
