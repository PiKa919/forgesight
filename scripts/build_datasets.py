"""Build the evaluation datasets and seed a workspace.

Prints each manifest hash. Two runs on the same generator version and the same
seeds must produce the same hash, and that is asserted by the test suite rather
than assumed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.eval import dataset as ds
from forgesight.settings import get_settings
from forgesight.storage.object_store import build_store
from forgesight.vision.registry import registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", nargs="*", default=["synth-clean", "synth-scan", "synth-bench"])
    ap.add_argument("--limit", type=int, default=None,
                    help="pages per dataset; a truncated dataset gets its own manifest hash")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--workspace", default=None,
                    help="reuse an existing workspace instead of creating one")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args(argv)

    s = get_settings()
    s.ensure_dirs()
    pool = get_pool()
    migrate(pool, verbose=False)
    store = build_store(s)

    from forgesight.api.repo import WorkspaceRepo

    repo = WorkspaceRepo(pool)
    ws = _workspace(repo, args.workspace)
    _seed_candidates(ws, pool, repo)

    out = []
    for name in args.names:
        built = ds.build(name, s, store, limit=args.limit, dpi=args.dpi)
        did = ds.persist(pool, ws, built)
        ds.persist_pages(pool, ws, did, built)
        if not args.json:
            print(
                f"{built.name} v{built.semver}: {built.counts['pages']} pages "
                f"manifest={built.manifest_hash[:16]} id={did}"
            )
        out.append({**built.as_row(), "id": did})

    if args.json:
        print(json.dumps(out, indent=2))
    elif not args.names:
        print("nothing to do")
    return 0


def _workspace(repo, existing: str | None) -> str:
    """Reuse a named workspace, or create one dedicated to benchmarking."""
    if existing:
        return existing
    from forgesight.api.deps import hash_token, new_token

    ws = repo.create_workspace("benchmark")
    # The token exists so the workspace is usable from the API; it is printed by
    # the caller if needed and is not a credential anyone relies on.
    repo.add_token(ws, hash_token(new_token()), "owner", 24 * 365)
    return ws


def _seed_candidates(ws: str, pool, repo) -> None:
    """Register the built-in candidates so evaluations can resolve a reference."""
    s = get_settings()
    for bc in registry(s):
        if repo.find_candidate_by_hash(ws, bc.candidate.hash):
            continue
        pp_id, rt_id = repo.profiles_for(ws, bc.candidate.artifact, bc.candidate)
        with pool.connection() as conn:
            art = conn.fetchone(
                "SELECT id FROM model_artifact WHERE workspace_id = ? AND sha256 = ?",
                (ws, bc.candidate.artifact.sha256),
            )
        repo.insert_candidate(
            ws, bc.name, bc.candidate.hash, _Row(art["id"]), pp_id, rt_id,
            bc.candidate.score_threshold, bc.candidate.label_map_hash,
            bc.candidate.tile_enabled,
        )


class _Row:
    def __init__(self, artifact_id: str):
        self.id = artifact_id


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
