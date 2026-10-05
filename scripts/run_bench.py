"""Run the benchmark protocol and generate the report.

Refuses by default when the environment would make a run incomparable. Pass
`--allow-unfavourable` to take a labelled measurement anyway: the refusal
reasons are recorded in the report, which is more useful than nothing when a
machine cannot be put on AC power.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from forgesight.bench import protocol as proto
from forgesight.bench import report as report_mod
from forgesight.bench.runner import Refused, run
from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="1", choices=["1", "2"])
    ap.add_argument("--models", nargs="*", default=["heron", "egret-medium"])
    ap.add_argument("--runtimes", nargs="*", default=["torch", "onnxruntime"])
    ap.add_argument("--pages", type=int, default=proto.MEASURED_PAGES)
    ap.add_argument("--trials", type=int, default=proto.TRIALS)
    ap.add_argument("--cooldown", type=float, default=proto.COOLDOWN_S)
    ap.add_argument("--allow-unfavourable", action="store_true",
                    help="measure anyway and record why the protocol would refuse")
    ap.add_argument("--quality", action="store_true", help="also run the quality pass")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    s.ensure_dirs()
    pool = get_pool()
    migrate(pool, verbose=False)
    from forgesight.storage.object_store import build_store

    store = build_store(s)

    if args.tier == "1":
        configs = proto.tier1_matrix(args.models, args.runtimes)
    else:
        configs = proto.tier1_matrix(args.models, args.runtimes) + proto.tier2_matrix(
            args.models, args.runtimes, (2, 8), (8,), ("cv2_linear", "cv2_area")
        )
    print(f"matrix: {len(configs)} configurations x {args.trials} trials "
          f"x {args.pages} pages")

    try:
        outcome = run(
            configs, s, store, pool, trials=args.trials, pages=args.pages,
            allow_unfavourable=args.allow_unfavourable, cooldown_s=args.cooldown,
        )
    except Refused as exc:
        print(f"\nrefused to measure: {exc}", file=sys.stderr)
        print("Pass --allow-unfavourable to take a labelled measurement anyway.",
              file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return 1

    data = report_mod.ReportInput(
        env=outcome.env.to_dict(),
        trials=outcome.trials,
        raw_files=outcome.raw_files,
        refusals=outcome.refusals,
        memory_model=outcome.memory,
    )
    if args.quality:
        data.quality = _quality(args.models, s, store, pool)
        data.gating = _gating(args.models, s, store, pool)
    with pool.connection() as conn:
        data.dataset_hashes = [
            {"name": r["name"], "semver": r["semver"],
             "manifest_hash": r["manifest_hash"], "synthetic": bool(r["synthetic"])}
            for r in conn.fetchall(
                "SELECT name, semver, manifest_hash, synthetic FROM dataset_version"
            )
        ]

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    out = Path(args.out) if args.out else s.reports_dir / f"report-{stamp}.md"
    report_mod.write(data, out)
    report_mod.write_json(data, out.with_suffix(".json"))
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    for r in outcome.refusals:
        print(f"NOTE: protocol would have refused -- {r}")
    return 0


def _quality(models: list[str], s, store, pool) -> list[dict]:
    """Score each model on synth-clean/test and synth-scan via the reference."""
    from forgesight.eval import dataset as ds
    from forgesight.eval.coco_eval import evaluate
    from forgesight.eval.runner import predict_pages
    from forgesight.vision.registry import registry

    out: list[dict] = []
    ref = next(bc for bc in registry(s) if bc.role == "reference").candidate
    with pool.connection() as conn:
        ws_row = conn.fetchone("SELECT id FROM workspace ORDER BY created_at LIMIT 1")
    if ws_row is None:
        return out
    ws = ws_row["id"]
    for name, split in (("synth-clean", ds.TEST), ("synth-scan", ds.TEST)):
        truths = ds.load_truth(pool, ws, name, split)
        if not truths:
            continue
        from forgesight.eval.runner import _load_model

        cand = _load_model(ref, s)
        m = evaluate(truths, predict_pages(cand, ref, store, truths, pool),
                     ds.category_ids())
        out.append({"candidate": ref.name, "dataset": name, **m})
    return out


def _gating(models: list[str], s, store, pool) -> list[dict]:
    from forgesight.api.repo import WorkspaceRepo
    from forgesight.eval.runner import run_evaluation

    repo = WorkspaceRepo(pool)
    with pool.connection() as conn:
        row = conn.fetchone("SELECT id FROM workspace ORDER BY created_at LIMIT 1")
    if row is None:
        return []
    out = []
    for c in repo.list_candidates(row["id"]):
        try:
            result = run_evaluation(c, s, pool=pool, store=store)
        except Exception as exc:
            out.append({"candidate": c["name"], "gate": "evaluation", "passed": False,
                        "observed": str(exc)[:200], "threshold": None})
            continue
        for v in result.verdicts:
            out.append({"candidate": c["name"], **v})
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
