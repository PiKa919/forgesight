"""Seed a demo workspace: candidates, a synthetic batch, and a shadow run.

Produces everything the demo script in the design needs, so the walkthrough can
be run without clicking through the UI first. Everything it creates is generated
by this repository and carries SYNTHETIC provenance.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.settings import get_settings
from forgesight.synth.generator import generate_page
from forgesight.synth.templates import TEMPLATES
from forgesight.vision.registry import registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=12, help="pages in the demo batch")
    ap.add_argument("--shadow", action="store_true",
                    help="also enqueue a shadow candidate on the same pages")
    ap.add_argument("--workspace", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    s.ensure_dirs()
    pool = get_pool()
    migrate(pool, verbose=False)

    from forgesight.api.deps import hash_token, new_token
    from forgesight.api.repo import WorkspaceRepo

    repo = WorkspaceRepo(pool)
    ws = args.workspace or repo.create_workspace("demo")
    if args.workspace is None:
        token = new_token()
        repo.add_token(ws, hash_token(token), "owner", 24 * 365)

    _register_candidates(ws, pool, repo, s)
    channel = repo.ensure_channel(ws)
    ref = _ensure_release(ws, pool, repo, channel)

    from fastapi.testclient import TestClient

    import forgesight.api.deps as deps

    deps.set_pool(pool)
    from forgesight.api.app import create_app

    client = TestClient(create_app())

    files = []
    for i in range(args.pages):
        tpl = sorted(TEMPLATES)[i % len(TEMPLATES)]
        page = generate_page(template=tpl, seed=7_000 + i, dpi=s.render_dpi)
        buf = io.BytesIO()
        Image.fromarray(page["image"]).save(buf, format="PNG")
        files.append(("files", (f"page{i:02d}-{tpl}.png", buf.getvalue(), "image/png")))

    shadow_id = None
    if args.shadow:
        cands = repo.list_candidates(ws)
        # The registry encodes the runtime in the candidate name, and the raw
        # candidate row has no runtime column of its own.
        for c in cands:
            if c["name"].startswith("egret-medium") and "onnxruntime" in c["name"]:
                shadow_id = c["id"]
                break

    q = f"?shadow_candidate_id={shadow_id}" if shadow_id else ""
    # Declared, not inferred. Every page above came from this repository's
    # generator, so saying so is simply true -- and before the API had a way to
    # record it, the demo's own pages were stored with synthetic = FALSE, which
    # made this script's docstring and the README's SYNTHETIC claim untrue.
    r = client.post(f"/v1/batches{q}", files=files,
                    data={"synthetic": "true"},
                    headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 202:
        print(r.text[:800], file=sys.stderr)
        return 1

    out = {
        "workspace_id": ws,
        "channel": ref,
        "batch_id": r.json()["id"],
        "pages": args.pages,
        "shadow_candidate": shadow_id,
        "token": token,
    }
    print(json.dumps(out, indent=2))
    print(
        "\nDrive the workers to finish it:\n"
        "  make worker-torch   # and: make worker-ort\n"
        "  make reaper",
        file=sys.stderr,
    )
    return 0


def _register_candidates(ws: str, pool, repo, s) -> None:
    class Row:
        def __init__(self, i: str):
            self.id = i

    for bc in registry(s):
        if repo.find_candidate_by_hash(ws, bc.candidate.hash):
            continue
        pp, rt = repo.profiles_for(ws, bc.candidate.artifact, bc.candidate)
        with pool.connection() as conn:
            art = conn.fetchone(
                "SELECT id FROM model_artifact WHERE workspace_id = ? AND sha256 = ?",
                (ws, bc.candidate.artifact.sha256),
            )
        repo.insert_candidate(
            ws, bc.name, bc.candidate.hash, Row(art["id"]), pp, rt,
            bc.candidate.score_threshold, bc.candidate.label_map_hash,
            bc.candidate.tile_enabled,
        )


def _ensure_release(ws: str, pool, repo, channel: str) -> str:
    from forgesight.ledger.claims import new_id

    ref = next(c for c in repo.list_candidates(ws) if c["name"].startswith("ref-"))
    with pool.write() as conn:
        row = conn.fetchone(
            "SELECT active_release_id FROM channel WHERE id = ?", (channel,)
        )
        if row and row["active_release_id"]:
            return row["active_release_id"]
        rid = new_id("rel")
        conn.execute(
            "INSERT INTO release(id, workspace_id, channel_id, candidate_id, action, "
            "reason, actor, channel_version) VALUES (?, ?, ?, ?, 'seed', ?, 'seed', 1)",
            (rid, ws, channel, ref["id"], "demo seed release"),
        )
        conn.execute(
            "UPDATE channel SET active_release_id = ?, version = 1 WHERE id = ?",
            (rid, channel),
        )
    return rid


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
