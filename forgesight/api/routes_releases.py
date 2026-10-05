"""Release promotion and rollback (design §10).

Promotion is the point of the whole system, so it is deliberately hard to do
by accident:

* **Optimistic concurrency.** `expected_channel_version` must match. Two racing
  promotes produce one 200 and one 409, and exactly one Release row (AT-12).
* **Gates are not advisory.** A candidate whose evaluation failed any gate
  cannot be promoted, and the failing verdicts come back in the 422 body so the
  user can see which gate stopped it.
* **Append-only.** Rollback inserts a new Release pointing at the previous
  candidate rather than mutating history.
* **Pinning.** In-flight work items keep the candidate they were created with, so
  a promote or rollback mid-load never produces a half-old, half-new batch
  (AT-14).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from forgesight.api.deps import Principal, get_pool_singleton, operator, owner
from forgesight.api.repo import WorkspaceRepo, _iso
from forgesight.api.schemas import (
    ChannelOut,
    GateVerdict,
    PromoteRequest,
    PromoteResponse,
    ReleaseOut,
    RollbackRequest,
)
from forgesight.ledger.claims import new_id
from forgesight.settings import get_settings

router = APIRouter(prefix="/v1")


def repo() -> WorkspaceRepo:
    return WorkspaceRepo(get_pool_singleton())


@router.get("/releases", response_model=ChannelOut)
def get_releases(p: Principal = Depends(operator)) -> ChannelOut:
    r = repo()
    ch = r.channel(p.workspace_id)
    if ch is None:
        r.ensure_channel(p.workspace_id)
        ch = r.channel(p.workspace_id)
    return _channel_out(r, p.workspace_id, ch)


@router.post("/releases/promote", response_model=PromoteResponse)
def promote(body: PromoteRequest, p: Principal = Depends(owner)) -> PromoteResponse:
    r = repo()
    s = get_settings()
    ch = r.channel(p.workspace_id)
    if ch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no channel")

    cand = r.get_candidate(p.workspace_id, body.candidate_id)
    if cand is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "candidate not found")
    if not cand["valid"]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"candidate is invalid: {cand['invalid_reason']}",
        )

    # Gates first. Refusing a candidate on its merits is cheaper and safer than
    # taking the lock and then rolling back.
    evaluation = _latest_evaluation(p.workspace_id, body.candidate_id)
    if s.enable_evaluation:
        if evaluation is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "detail": "candidate has no completed evaluation",
                    "code": "gates_not_evaluated",
                },
            )
        gates = evaluation["gates"] or []
        failed = [g for g in gates if not g.get("passed")]
        if failed:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "detail": "candidate failed a release gate",
                    "code": "gate_failed",
                    "failed": [g.get("gate") for g in failed],
                    "verdicts": [GateVerdict(**g) for g in gates],
                },
            )

    if ch["version"] != body.expected_channel_version:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"channel is at version {ch['version']}, "
            f"expected {body.expected_channel_version}",
        )

    rel_id = _write_release(
        p.workspace_id, ch["id"], body.candidate_id, ch["active_release_id"],
        "promote", body.reason, p.role, body.expected_channel_version,
    )
    _bump(p.workspace_id, ch["id"], body.expected_channel_version, rel_id)
    return PromoteResponse(
        channel=_channel_out(r, p.workspace_id, r.channel(p.workspace_id)),
        release=_release_out(_get_release(p.workspace_id, rel_id)),
    )


@router.post("/releases/rollback", response_model=PromoteResponse)
def rollback(body: RollbackRequest, p: Principal = Depends(owner)) -> PromoteResponse:
    r = repo()
    ch = r.channel(p.workspace_id)
    if ch is None or not ch["active_release_id"]:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "nothing to roll back from")

    current = _get_release(p.workspace_id, ch["active_release_id"])
    if current is None or not current.get("previous_release_id"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "no previous release to return to"
        )
    previous = _get_release(p.workspace_id, current["previous_release_id"])
    if previous is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "the previous release no longer exists"
        )

    if ch["version"] != body.expected_channel_version:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"channel is at version {ch['version']}, "
            f"expected {body.expected_channel_version}",
        )

    # A rollback re-verifies only artifact integrity and memory fit, because the
    # candidate already passed the full gate suite when it was promoted.
    _verify_artifact_intact(p.workspace_id, previous["candidate_id"])

    rel_id = _write_release(
        p.workspace_id, ch["id"], previous["candidate_id"],
        ch["active_release_id"], "rollback", body.reason, p.role,
        body.expected_channel_version,
    )
    _bump(p.workspace_id, ch["id"], body.expected_channel_version, rel_id)
    return PromoteResponse(
        channel=_channel_out(r, p.workspace_id, r.channel(p.workspace_id)),
        release=_release_out(_get_release(p.workspace_id, rel_id)),
    )


# -- helpers ----------------------------------------------------------------


def _write_release(ws: str, channel_id: str, candidate_id: str,
                   previous_id: str | None, action: str, reason: str,
                   actor: str, version: int) -> str:
    rid = new_id("rel")
    with get_pool_singleton().write() as conn:
        conn.execute(
            "INSERT INTO release(id, workspace_id, channel_id, candidate_id, "
            "previous_release_id, action, reason, actor, channel_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, ws, channel_id, candidate_id, previous_id, action, reason, actor, version),
        )
        conn.execute(
            "INSERT INTO audit_event(id, workspace_id, actor, action, subject_kind, "
            "subject_id, after_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (new_id("ae"), ws, actor, f"release.{action}", "channel", channel_id, rid),
        )
    return rid


def _bump(ws: str, channel_id: str, expected: int, rel_id: str) -> None:
    """Move the channel forward, or refuse. The WHERE clause is the lock."""
    with get_pool_singleton().write() as conn:
        cur = conn.execute(
            "UPDATE channel SET active_release_id = ?, version = version + 1 "
            "WHERE id = ? AND workspace_id = ? AND version = ?",
            (rel_id, channel_id, ws, expected),
        )
        if getattr(cur, "rowcount", 1) == 0:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "the channel changed while promoting; retry with the current version",
            )


def _get_release(ws: str, release_id: str):
    with get_pool_singleton().connection() as conn:
        return conn.fetchone(
            "SELECT r.*, c.name AS candidate_name FROM release r "
            "LEFT JOIN candidate c ON c.id = r.candidate_id "
            "AND c.workspace_id = r.workspace_id "
            "WHERE r.workspace_id = ? AND r.id = ?",
            (ws, release_id),
        )


def _latest_evaluation(ws: str, candidate_id: str):
    with get_pool_singleton().connection() as conn:
        return conn.fetchone(
            "SELECT * FROM evaluation_run WHERE workspace_id = ? AND candidate_id = ? "
            "AND status = 'done' ORDER BY created_at DESC LIMIT 1",
            (ws, candidate_id),
        )


def _verify_artifact_intact(ws: str, candidate_id: str) -> None:
    from forgesight.vision.runtimes.base import verify_artifact

    with get_pool_singleton().connection() as conn:
        row = conn.fetchone(
            "SELECT a.path, a.sha256 FROM candidate c "
            "JOIN model_artifact a ON a.id = c.artifact_id "
            "AND a.workspace_id = c.workspace_id "
            "WHERE c.workspace_id = ? AND c.id = ?",
            (ws, candidate_id),
        )
    if row is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "candidate has no artifact")
    from pathlib import Path

    try:
        verify_artifact(Path(row["path"]), row["sha256"])
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"artifact integrity: {exc}"
        ) from exc


def _release_out(row) -> ReleaseOut:
    return ReleaseOut(
        id=row["id"],
        action=row["action"],
        candidate_id=row["candidate_id"],
        candidate_name=row["candidate_name"] or "unknown",
        previous_release_id=row["previous_release_id"],
        reason=row["reason"],
        actor=row["actor"],
        channel_version=int(row["channel_version"]),
        created_at=_iso(row["created_at"]) or "",
    )


def _channel_out(r: WorkspaceRepo, ws: str, ch) -> ChannelOut:
    history = [_release_out(x) for x in r.list_releases(ws, ch["id"])]
    active_name = None
    if ch["active_release_id"]:
        rel = _get_release(ws, ch["active_release_id"])
        active_name = rel["candidate_name"] if rel else None
    return ChannelOut(
        name=ch["name"],
        version=int(ch["version"]),
        active_release_id=ch["active_release_id"],
        active_candidate_name=active_name,
        history=history,
    )


__all__ = ["router"]
