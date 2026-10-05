"""Session creation for the public demo (design §11.2).

`POST /sessions` is the only unauthenticated route. It mints a sandbox workspace
plus an operator token, and in public mode it is metered, because otherwise the
endpoint is an open invitation to create unlimited state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from forgesight.api.deps import Principal, get_pool_singleton, hash_token, new_token
from forgesight.api.repo import WorkspaceRepo
from forgesight.api.schemas import SessionResponse
from forgesight.settings import get_settings
from forgesight.vision.registry import registry

router = APIRouter(prefix="/v1")


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(p: Principal | None = Depends(lambda: None)) -> SessionResponse:
    s = get_settings()
    r = WorkspaceRepo(get_pool_singleton())

    # Metered in public mode only. A local single-user deployment has no reason
    # to ration its own sessions.
    if s.mode == "public" and r.sandbox_count() >= s.public_global_quota:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "the public demo quota is used up; try again later",
            headers={"Retry-After": "60"},
        )

    ws = r.create_workspace(
        "demo", kind="sandbox" if s.mode == "public" else "local",
        ttl_h=s.sandbox_ttl_hours,
    )
    token = new_token()
    r.add_token(ws, hash_token(token), "operator", s.sandbox_ttl_hours)
    _seed(ws, r)

    return SessionResponse(
        workspace_id=ws,
        token=token,
        expires_at=(datetime.now(UTC) + timedelta(hours=s.sandbox_ttl_hours)).isoformat(),
        mode=s.mode,
    )


def _seed(ws: str, r: WorkspaceRepo) -> None:
    """Register the built-in candidates and a seed release.

    Without an active release an upload is refused, so a fresh workspace would
    be unusable. Seeding here means a session is immediately demoable.
    """

    s = get_settings()
    pool = get_pool_singleton()
    ref_id = None
    for bc in registry(s):
        existing = r.find_candidate_by_hash(ws, bc.candidate.hash)
        if existing:
            cid = existing["id"]
        else:
            pp_id, rt_id = r.profiles_for(ws, bc.candidate.artifact, bc.candidate)
            with pool.connection() as conn:
                art = conn.fetchone(
                    "SELECT id FROM model_artifact WHERE workspace_id = ? AND sha256 = ?",
                    (ws, bc.candidate.artifact.sha256),
                )
            # Outside the lookup: insert_candidate opens its own transaction, and
            # the pool is thread-local, so nesting the two would fail.
            cid = r.insert_candidate(
                ws, bc.name, bc.candidate.hash, _ArtifactRow(art["id"]),
                pp_id, rt_id, bc.candidate.score_threshold,
                bc.candidate.label_map_hash, bc.candidate.tile_enabled,
            )
        if bc.role == "reference":
            ref_id = cid

    if ref_id is None:
        return
    channel = r.ensure_channel(ws)
    with pool.write() as conn:
        rel = _seed_release(ws, channel, ref_id, conn)
        conn.execute(
            "UPDATE channel SET active_release_id = ?, version = 1 WHERE id = ?",
            (rel, channel),
        )


class _ArtifactRow:
    """Adapter so the repo can register an artifact it has already inserted."""

    def __init__(self, artifact_id: str):
        self.id = artifact_id


def _seed_release(ws: str, channel: str, candidate_id: str, conn) -> str:
    from forgesight.ledger.claims import new_id

    rid = new_id("rel")
    conn.execute(
        "INSERT INTO release(id, workspace_id, channel_id, candidate_id, action, "
        "reason, actor, channel_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rid, ws, channel, candidate_id, "seed", "initial reference release", "system", 1),
    )
    return rid


__all__ = ["router"]
