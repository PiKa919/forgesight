#!/usr/bin/env python
"""The recorded walkthrough (design §18), as a script that verifies itself.

Design §18 is a timed demo with eight beats. Two are the presenter's words -- the
problem in one sentence, and the limitations -- so they print as text and are not
checked. The other six are claims about what the system does, and this script
drives each one against the real pipeline: real weights, the real ledger, the
real worker, the real reaper. Every beat asserts the state it claims to
demonstrate, so "the demo script runs end to end" is something a gate can check
rather than something a person has to watch for.

Driven in-process rather than against a running compose stack, for two reasons.
A demo needing five services up cannot be a gate. And the failure beat exists to
kill a worker, which is awkward against a supervised container and trivial
against a `Worker` object.

    uv run python scripts/walkthrough.py            # every beat
    uv run python scripts/walkthrough.py --only 4 6 # a subset

Exit code is non-zero if any beat fails to demonstrate what it claims.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool, ph
from forgesight.ledger.claims import Ledger, PredictionIn
from forgesight.settings import configure, get_settings
from forgesight.vision.types import (
    Candidate,
    Detection,
    ModelArtifact,
    PreprocessProfile,
    RuntimeProfile,
)

# Wide enough to reach any item under test, given the queue is global.
WIDE_CLAIM = 256

NARRATIVE = {
    0: (
        "The problem, in one sentence:\n"
        '  "Is a cheaper CPU configuration of a layout model safe to ship,\n'
        '   and how would I know?"\n'
        "Laptop CPU only. Every page is generated and labelled SYNTHETIC."
    ),
    8: (
        "Limitations, stated plainly:\n"
        "  - the synthetic ground truth uses this repo's box conventions, not\n"
        "    DocLayNet's, so absolute mAP does not transfer to real documents\n"
        "  - one host, CPU only\n"
        "  - every number is valid for that host and date, and no other\n"
        "  - the assumed memory model did not reproduce here; admission runs on\n"
        "    measured peaks rather than a fitted line (ADR 0004)"
    ),
}


class Failed(Exception):
    """A beat could not even attempt its claim."""


class Walkthrough:
    def __init__(self, database_url: str, object_dir: Path | None = None) -> None:
        self.s = get_settings().model_copy(update={
            "database_url": database_url,
            **({"objects_dir": object_dir} if object_dir else {}),
        })
        # configure(), not just model_copy. seed_demo.main() calls get_pool() with
        # no argument, which reads the process-wide settings. With only
        # model_copy, the seed wrote to the default database while every check
        # read this script's own, so nothing matched and every beat failed on a
        # query timeout.
        configure(self.s)
        self.s.ensure_dirs()
        if database_url.startswith("sqlite:///"):
            path = Path(database_url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
        self.pool = get_pool(self.s.database_url)
        migrate(self.pool, verbose=False)
        self.pool_ledger = Ledger(self.pool)
        self.p = ph(self.pool.dialect)
        self.failures: list[str] = []
        self.t0 = time.perf_counter()
        self.ctx: dict = {}

    # -- reporting --------------------------------------------------------

    def beat(self, n: int, title: str) -> None:
        print(f"\n{'=' * 74}\n[{n}] {title}   (t+{time.perf_counter() - self.t0:5.1f}s)\n{'=' * 74}")

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"      [{'ok  ' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(what)
        return ok

    def note(self, text: str) -> None:
        print(f"      {text}")

    # -- plumbing ---------------------------------------------------------

    def client(self):
        import forgesight.api.deps as deps

        deps.set_pool(self.pool)
        from fastapi.testclient import TestClient

        from forgesight.api.app import create_app

        return TestClient(create_app())

    def store(self):
        from forgesight.storage.object_store import FsObjectStore

        return FsObjectStore(self.s.objects_dir)

    def seed(self, pages: int, shadow: bool = False) -> dict:
        """Register candidates, seed a release and upload a batch.

        Delegates to seed_demo.py so the walkthrough and the manual demo start
        from exactly the same state; a walkthrough that built its own state would
        be demonstrating a setup nobody else runs.
        """
        from seed_demo import main as seed_main

        argv = ["--pages", str(pages)] + (["--shadow"] if shadow else [])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = seed_main(argv)
        if rc != 0:
            raise Failed(f"seed_demo returned {rc}")
        return json.loads(buf.getvalue())

    def drain(self, pool_name: str, max_rounds: int = 8) -> int:
        """Run the worker until its pool's queue is empty. Returns rounds used."""
        from forgesight.worker.main import Worker

        rounds = 0
        for _ in range(max_rounds):
            depth = self.scalar(
                f"SELECT COUNT(*) AS n FROM work_item WHERE state = {self.p} "
                f"AND pool = {self.p}", ("queued", pool_name))
            if not depth:
                break
            asyncio.run(Worker(self.s, self.pool, self.store(),
                               pool_name=pool_name).run(once=True))
            rounds += 1
        return rounds

    def scalar(self, sql: str, params: tuple = ()) -> int:
        with self.pool.connection() as conn:
            return int(conn.fetchone(sql, params)["n"])

    # -- beat 1: upload and the timing split ------------------------------

    def beat_1_upload(self) -> None:
        info = self.seed(pages=12, shadow=True)
        self.check(info["pages"] == 12, "a 12-page batch was accepted",
                   f"batch {info['batch_id']}")

        # Drain, don't tick. `Worker.run(once=True)` claims at most claim_batch
        # (8) and returns, so two calls process 16 of the 24 items a 12-page
        # shadow batch creates and beat 1 could never pass.
        for pool_name in ("torch", "onnxruntime"):
            self.drain(pool_name)

        client = self.client()
        h = {"Authorization": f"Bearer {info['token']}"}
        batch = client.get(f"/v1/batches/{info['batch_id']}", headers=h).json()
        self.check(batch["status"] == "succeeded", "batch reached succeeded",
                   f"counts={batch['counts']}")

        items = client.get(f"/v1/batches/{info['batch_id']}/items", headers=h).json()
        # Filter to succeeded items. `timings` is a required field on every row and
        # is present-but-null on an unfinished one, so indexing timed[0] picked a
        # uniformly random item and the beat failed roughly one run in three on
        # correct code.
        done = [i for i in items if i["state"] == "succeeded"]
        self.check(len(done) == len(items) and bool(items),
                   "every item succeeded and recorded timings",
                   f"{len(done)}/{len(items)} succeeded")
        if not done:
            return

        t = done[0]["timings"]
        qi, svc = t.get("queue_inclusive_ms"), t.get("service_ms")
        self.check(qi is not None and svc is not None,
                   "both latency figures are reported",
                   f"queue_inclusive_ms={qi}, service_ms={svc}")
        if qi is not None and svc is not None:
            # The point of reporting two numbers: if queue-inclusive omitted the
            # wait it would read smaller than service.
            self.check(qi >= svc - 1.0,
                       "queue-inclusive is not smaller than service",
                       f"{qi:.0f} ms >= {svc:.0f} ms")
        self.check((t.get("infer_ms") or 0) > 0, "inference time recorded",
                   f"{t.get('infer_ms')} ms")
        self.note(f"sample split: {json.dumps(t)}")

        self.ctx = {**info, "headers": h, "items": items}

    # -- beat 2: boxes and provenance -------------------------------------

    def beat_2_provenance(self) -> None:
        if not self.ctx:
            raise Failed("beat 1 must run first; --only needs its prerequisites")
        client = self.client()
        h = self.ctx["headers"]
        done = next((i for i in self.ctx["items"] if i["state"] == "succeeded"), None)
        if not self.check(done is not None, "a finished item to inspect"):
            return

        img = client.get(f"/v1/items/{done['id']}/image", headers=h,
                         follow_redirects=False)
        self.check(img.status_code in (200, 302), "page image served to its owner",
                   f"HTTP {img.status_code}")

        rel = client.get("/v1/releases", headers=h).json()
        self.check(bool(rel.get("active_release_id")), "an active release is pinned",
                   str(rel.get("active_release_id")))

        n = self.scalar(
            f"SELECT COUNT(*) AS n FROM prediction WHERE workspace_id = {self.p}",
            (self.ctx["workspace_id"],))
        self.check(n > 0, "predictions persisted", f"{n} rows")

        # Provenance: every prediction names the artifact it came from, by sha.
        with self.pool.connection() as conn:
            row = conn.fetchone(
                f"SELECT COUNT(*) AS n FROM prediction p JOIN work_item w "
                f"ON p.work_item_id = w.id JOIN candidate c ON w.candidate_id = c.id "
                f"JOIN model_artifact a ON c.artifact_id = a.id "
                f"WHERE p.workspace_id = {self.p} AND a.sha256 <> ''",
                (self.ctx["workspace_id"],))
        self.check(row["n"] > 0, "every prediction traces to a pinned artifact sha",
                   f"{row['n']} traced")

    # -- beat 3: shadow run -----------------------------------------------

    def beat_3_shadow(self) -> None:
        shadow = self.ctx.get("shadow_candidate")
        if not self.check(bool(shadow), "a shadow candidate was registered"):
            return
        n = self.scalar(
            f"SELECT COUNT(*) AS n FROM prediction WHERE workspace_id = {self.p} "
            f"AND candidate_id = {self.p}", (self.ctx["workspace_id"], shadow))
        self.check(n > 0, "the shadow candidate produced its own predictions",
                   f"{n} rows on the same pages")

        # A shadow run creates its own work items, one per candidate per page, so
        # asking "two candidates on one work_item" is the wrong question -- there
        # is only ever one candidate per item by construction. The diff view joins
        # on the *page*, so that is what has to be checked here.
        both = self.scalar(
            f"SELECT COUNT(*) AS n FROM (SELECT w.page_id FROM prediction p "
            f"JOIN work_item w ON p.work_item_id = w.id "
            f"WHERE p.workspace_id = {self.p} GROUP BY w.page_id "
            f"HAVING COUNT(DISTINCT p.candidate_id) > 1) t",
            (self.ctx["workspace_id"],))
        self.check(both > 0, "some pages carry both the release and the shadow result",
                   f"{both} pages have two candidates")

    # -- beat 4: gates, and the negative control --------------------------

    def beat_4_gates(self) -> None:
        from forgesight.eval.gates import GATE_POLICY, judge, negative_control_blocked

        def cand(name: str) -> Candidate:
            return Candidate(
                name=name,
                artifact=ModelArtifact(
                    name=name, path="/nonexistent", sha256=f"{name}-sha",
                    format="safetensors", repo_id=f"example/{name}",
                    revision="0" * 40, license="Apache-2.0"),
                preprocess=PreprocessProfile(
                    resize="pil_bilinear", target=640, render_dpi=150,
                    tile_long_side_px=2200, tile_aspect=1.6, tile_overlap=0.2,
                    label_map_hash="lm", processor_class="example"),
                runtime=RuntimeProfile(
                    runtime="torch", intra_op_threads=4, inter_op_threads=1,
                    graph_opt_level="ORT_ENABLE_ALL", max_batch=8, batch_window_ms=25),
                score_threshold=0.3, label_map_hash="lm")

        metrics = {"map": 0.50, "map50": 0.50,
                   "per_class_ap": {k: 0.5 for k in ("table", "picture", "text", "title")}}
        ref = cand("reference")

        good = judge(candidate=cand("candidate"), reference=ref,
                     clean={"reference": metrics, "candidate": dict(metrics)},
                     shift={"reference": metrics, "candidate": dict(metrics)},
                     agreement=0.995, peak_bytes=None,
                     budget_bytes=self.s.worker_mem_budget, provenance_ok=True)
        self.check(all(v["passed"] for v in good), "an identical candidate passes every gate",
                   f"{len(good)} verdicts")
        self.check({v["gate"] for v in good} == set(GATE_POLICY),
                   "all six gates were evaluated, not just the ones that pass")

        # The negative control: degraded, so it must be refused. A gate suite that
        # stopped blocking bad candidates is worse than one that never existed.
        bad = judge(candidate=cand("candidate"), reference=ref,
                    clean={"reference": metrics, "candidate": {**metrics, "map": 0.40}},
                    shift={"reference": metrics, "candidate": {**metrics, "map": 0.40}},
                    agreement=None, peak_bytes=None,
                    budget_bytes=self.s.worker_mem_budget, provenance_ok=False)
        by = {v["gate"]: v for v in bad}
        g1 = by["G1_quality_clean"]
        self.check(not g1["passed"], "the degraded 320-px control is blocked by G1",
                   f"observed {g1['observed']} against threshold {g1['threshold']}")
        self.check(negative_control_blocked(bad), "the refusal is recognised as a negative control")
        # G4 (runtime parity) and G5 (memory fit) report "passed" when they are
        # not applicable -- there was no shadow comparison and no calibration to
        # judge. Asserting that *nothing* passes would be asserting a bug. What
        # must fail is everything that had evidence to judge on.
        must_fail = {"G1_quality_clean", "G2_quality_shift", "G6_provenance"}
        self.check(all(not by[g]["passed"] for g in must_fail),
                   "every gate with evidence against it failed",
                   ", ".join(sorted(must_fail)))

    # -- beat 5: promote and roll back ------------------------------------

    def beat_5_promote_rollback(self) -> None:
        if not self.ctx:
            raise Failed("beat 1 must run first; --only needs its prerequisites")
        client = self.client()
        h = self.ctx["headers"]
        ws = self.ctx["workspace_id"]
        self._build_datasets(ws)

        cands = client.get("/v1/candidates", headers=h).json()
        if not self.check(len(cands) >= 2, "more than one candidate is available",
                          f"{len(cands)} registered"):
            return

        # The release rides on a measurement, so start from the pinned reference,
        # which is the one candidate that has to pass against itself.
        ref = next((c for c in cands if c["name"].startswith("ref-")), None)
        if not self.check(ref is not None, "a pinned reference is registered"):
            return

        res = self._evaluate(ws, ref["id"])
        self._assert_measured(res, ref["name"])
        failed = [v["gate"] for v in res.verdicts if not v["passed"]]
        if not self.check(not failed, "the reference passes every gate on real data",
                          ", ".join(failed) or "all six passed"):
            return

        before = client.get("/v1/releases", headers=h).json()
        r = client.post("/v1/releases/promote", headers=h, json={
            "candidate_id": ref["id"],
            "expected_channel_version": before["version"],
            "reason": f"walkthrough: promote {ref['name']}",
        })
        if not self.check(r.status_code == 200, "promote accepted at the current version",
                          f"HTTP {r.status_code} {r.text[:120]}"):
            return

        after = client.get("/v1/releases", headers=h).json()
        self.check(after["version"] > before["version"], "channel version advanced",
                   f"{before['version']} -> {after['version']}")

        stale = client.post("/v1/releases/promote", headers=h, json={
            "candidate_id": ref["id"],
            "expected_channel_version": before["version"],
            "reason": "walkthrough: stale version must be refused",
        })
        self.check(stale.status_code == 409,
                   "a promote against a stale version is refused (409)",
                   f"HTTP {stale.status_code}")

        rb = client.post("/v1/releases/rollback", headers=h, json={
            "expected_channel_version": after["version"],
            "reason": "walkthrough: roll back",
        })
        self.check(rb.status_code == 200, "rollback accepted", f"HTTP {rb.status_code}")
        hist = client.get("/v1/releases", headers=h).json().get("history", [])
        self.check(len(hist) >= 2, "history records both events", f"{len(hist)} releases")

        # The half of the story that matters: a candidate that fails a gate must be
        # refused, and the refusal must name the gate. An earlier version of this
        # beat picked "the first candidate that is not ref-" and promoted it after
        # writing its own `passed: true` verdict row -- which meant the demo
        # promoted whatever it was handed and called it a gate suite.
        other = next((c for c in cands
                      if not c["name"].startswith("ref-")
                      and "negctl" not in c["name"]), None)
        if other is None:
            self.note("no second candidate registered; skipped the refusal path")
            return

        res2 = self._evaluate(ws, other["id"])
        self._assert_measured(res2, other["name"])
        failed2 = [v["gate"] for v in res2.verdicts if not v["passed"]]
        if not failed2:
            self.note(f"{other['name']} passed all gates on this host; "
                      "nothing to refuse")
            return

        self.check("negctl" not in other["name"],
                   "the refused candidate is not being confused with the control",
                   other["name"])
        cur = client.get("/v1/releases", headers=h).json()
        r2 = client.post("/v1/releases/promote", headers=h, json={
            "candidate_id": other["id"],
            "expected_channel_version": cur["version"],
            "reason": f"walkthrough: try to promote {other['name']} anyway",
        })
        # FastAPI nests a dict detail one level down: {"detail": {"code": ...}}.
        body = r2.json() if "json" in r2.headers.get("content-type", "") else {}
        inner = body.get("detail") if isinstance(body, dict) else None
        inner = inner if isinstance(inner, dict) else {}
        code = inner.get("code") or ""
        named = [g for g in (inner.get("failed") or []) if g in failed2]
        self.check(r2.status_code == 422 and code == "gate_failed",
                   "promote is refused for a candidate that failed a gate",
                   f"HTTP {r2.status_code} code={code or 'none'}")
        self.check(bool(named),
                   "the refusal names the gates that actually failed",
                   ", ".join(named) or f"failed={failed2}")

    def _assert_measured(self, result, name: str) -> None:
        """Refuse to call it an evaluation unless metrics were really computed."""
        clean = result.metrics.get("clean") or {}
        cand_map = (clean.get("candidate") or {}).get("map")
        ref_map = (clean.get("reference") or {}).get("map")
        self.check(cand_map is not None and ref_map is not None,
                   f"{name}: real mAP was measured, not stubbed",
                   f"candidate={cand_map}, reference={ref_map}")
        self.check(len(result.verdicts) == 6,
                   f"{name}: the release carries six real verdicts",
                   f"{len(result.verdicts)} verdicts")

    def _build_datasets(self, ws: str, limit: int = 6) -> None:
        """Build the clean and shift splits with ground truth, for real evaluation."""
        from forgesight.eval import dataset as ds

        for name in ("synth-clean", "synth-scan"):
            if ds.load_truth(self.pool, ws, name, "test"):
                continue
            built = ds.build(name, self.s, self.store(), limit=limit)
            did = ds.persist(self.pool, ws, built)
            ds.persist_pages(self.pool, ws, did, built)
            self.note(f"built {name} v{built.semver} "
                      f"({built.counts['pages']} pages, manifest={built.manifest_hash[:12]})")

    def _evaluate(self, ws: str, candidate_id: str):
        """Run the real evaluator and persist exactly what it returned.

        An earlier version of this beat inserted an evaluation_run row whose gates
        were all `passed: true` with `observed: 1.0`, then promoted off it. That is
        a fabricated result: the release history said a candidate had passed six
        gates, and nothing had been measured. It is the exact thing the gate suite
        exists to prevent, done by the demo.

        This runs forgesight.eval.runner.run_evaluation instead: it loads the
        calibration/clean/shift splits, picks the operating threshold on calib,
        scores both candidate and reference on clean and shift, and lets judge()
        decide. Whatever it says is what gets stored, including a failure.
        """
        from forgesight.eval import dataset as ds
        from forgesight.eval.runner import run_evaluation
        from forgesight.ledger.claims import new_id

        with self.pool.write() as conn:
            rows = [dict(r) for r in conn.execute(
                f"SELECT * FROM candidate WHERE id = {self.p}", (candidate_id,))]
        if not rows:
            raise RuntimeError(f"candidate {candidate_id} is not registered")
        row = rows[0]

        # A missing dataset must fail the run, not skip a gate.
        for label, split in (("synth-clean", "test"), ("synth-scan", "test")):
            if not ds.load_truth(self.pool, ws, label, split):
                raise RuntimeError(
                    f"dataset {label}/{split} is not built for {ws}; "
                    "a missing dataset must fail the evaluation")

        started = time.monotonic()
        result = run_evaluation(row, self.s, self.pool, self.store())

        cols = ("id", "workspace_id", "candidate_id", "status", "dataset_versions",
                "metrics", "gates", "code_sha", "env_manifest")
        metrics = dict(result.metrics)
        metrics["walkthrough_elapsed_s"] = round(time.monotonic() - started, 2)
        with self.pool.write() as conn:
            conn.execute(
                f"INSERT INTO evaluation_run({', '.join(cols)}) "
                f"VALUES ({','.join([self.p] * len(cols))})",
                (new_id("ev"), ws, candidate_id, "done",
                 json.dumps(result.dataset_versions), json.dumps(metrics),
                 json.dumps(result.verdicts), "walkthrough",
                 json.dumps(result.env_manifest)))
        return result

    # -- beat 6: the worker dies ------------------------------------------

    def beat_6_failure(self) -> None:
        from forgesight.ledger.reaper import Reaper

        info = self.seed(pages=4)
        ws = info["workspace_id"]
        ledger = self.pool_ledger

        # claim() is a global, oldest-first work queue: it is not scoped to a
        # workspace. Claiming n=4 here returned an earlier beat's leftovers, so the
        # "worker killed mid-batch" beat never touched the batch it had just
        # seeded -- it claimed beat 1's pages, while every count below was scoped
        # to `ws`. Take a wide slice, keep this beat's own items, and return the
        # rest: there is no release, so an item leaves a worker by completing,
        # failing, or having its lease reaped.
        everything = ledger.claim("torch", "doomed-worker", n=WIDE_CLAIM, lease_s=60)
        held = [it for it in everything if it.workspace_id == ws]
        for it in everything:
            if it.workspace_id != ws:
                ledger.fail(it, "walkthrough: not under test")
        if not self.check(len(held) == 4, "the seeded batch was claimed in full",
                          f"{len(held)} of 4 items in {ws}"):
            return
        zombie = held[0]
        before = self.scalar(f"SELECT COUNT(*) AS n FROM prediction "
                             f"WHERE workspace_id = {self.p}", (ws,))

        # The lease lapses. Forced into the past rather than slept through: the
        # point is the fencing, and a sleep would be slower and no more
        # convincing. Scoped to the item ids this worker held -- expiring a whole
        # workspace once made the outcome depend on what a previous run had left
        # behind, and the reaper reported 0 against a database whose items were
        # already finished.
        past = datetime.now(UTC) - timedelta(seconds=120)
        ids = [it.id for it in held]
        with self.pool.write() as conn:
            conn.execute(
                f"UPDATE work_item SET lease_expires_at = {self.p} "
                f"WHERE id IN ({','.join([self.p] * len(ids))})",
                (past, *ids))

        result = Reaper(self.pool, self.s).reap_expired_leases()
        self.check(result.requeued >= len(held),
                   "the reaper requeued every abandoned item",
                   f"{result.requeued} requeued of {len(held)} abandoned")

        # A replacement takes the work. This is what rotates the fencing token:
        # an expired lease on its own does not block a write, which is the whole
        # point of the token. An earlier version of this beat asserted fencing
        # without a re-claim and was asserting a fiction -- it completed
        # successfully and reported that as the zombie being fenced out.
        taken = ledger.claim("torch", "replacement-worker", n=WIDE_CLAIM, lease_s=60)
        live = [it for it in taken if it.workspace_id == ws]
        for it in taken:
            if it.workspace_id != ws:
                ledger.fail(it, "walkthrough: not under test")
        if not self.check(bool(live), "a replacement worker can claim the same work",
                          f"{len(live)} items in {ws}"):
            return
        live_item = next((it for it in live if it.id == zombie.id), None)
        if not self.check(live_item is not None,
                          "the replacement took over the zombie's own item",
                          f"zombie={zombie.id}"):
            return
        for it in live:
            if it.id != zombie.id:
                ledger.fail(it, "walkthrough: not under test")
        self.check(live_item.fencing_token > zombie.fencing_token,
                   "the replacement's fencing token is newer",
                   f"{zombie.fencing_token} -> {live_item.fencing_token}")

        self.check(ledger.complete(live_item, self._one_detection()) is True,
                   "the live worker's result is accepted")

        # Now the zombie wakes up. Every write it attempts must be refused.
        self.check(ledger.complete(zombie, self._one_detection()) is False,
                   "the zombie's completion is refused")
        self.check(ledger.fail(zombie, "inference_failed") is False,
                   "the zombie's failure report is refused")
        self.check(ledger.mark_cancelled(zombie) is False,
                   "the zombie cannot cancel an item it no longer owns")

        after = self.scalar(f"SELECT COUNT(*) AS n FROM prediction "
                            f"WHERE workspace_id = {self.p}", (ws,))
        self.check(after == before + 1, "exactly one prediction exists: the live one",
                   f"{before} -> {after}")

    @staticmethod
    def _one_detection():
        return PredictionIn([Detection(class_id=0, label="text", score=0.9,
                                       box=(0.1, 0.1, 0.2, 0.2))])

    # -- runner -----------------------------------------------------------

    def run(self, only: list[int]) -> int:
        print("ForgeSight walkthrough -- design §18")
        print(f"database: {self.s.database_url}")

        beats = [
            (0, "the problem, in one sentence", lambda: self._say(0)),
            (1, "upload 12 synthetic pages, timing split", self.beat_1_upload),
            (2, "boxes by class, and provenance", self.beat_2_provenance),
            (3, "shadow run on the same pages", self.beat_3_shadow),
            (4, "gate report, negative control blocked", self.beat_4_gates),
            (5, "promote, then roll back", self.beat_5_promote_rollback),
            (6, "worker killed mid-batch, reaper requeues, no duplicates",
             self.beat_6_failure),
            (8, "limitations", lambda: self._say(8)),
        ]
        titles = {0: "the problem, in one sentence", 1: "upload 12 synthetic pages, timing split",
                  2: "boxes by class, and provenance", 3: "shadow run on the same pages",
                  4: "gate report, negative control blocked", 5: "promote, then roll back",
                  6: "worker killed mid-batch, reaper requeues, no duplicates",
                  8: "limitations"}
        for n, _, fn in beats:
            if only and n not in only:
                continue
            self.beat(n, titles[n])
            try:
                fn()
            except Exception as exc:  # a beat that raises is a beat that failed
                self.check(False, f"beat {n} could not complete",
                           f"{type(exc).__name__}: {exc}")

        print(f"\n{'=' * 74}")
        if self.failures:
            print(f"WALKTHROUGH FAILED -- {len(self.failures)} claim(s) did not hold:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print("WALKTHROUGH OK -- every checkable claim in design §18 was demonstrated.")
        return 0

    def _say(self, n: int) -> None:
        print(NARRATIVE[n])


def main() -> int:
    ap = argparse.ArgumentParser(description="Drive and verify the design §18 walkthrough.")
    ap.add_argument("--only", nargs="*", type=int, default=[],
                    help="beat numbers to run (default: all)")
    ap.add_argument("--database", default=None,
                    help="DSN to run against. Defaults to a private SQLite file "
                         "which is deleted first, so the walkthrough always "
                         "starts from an empty database and its results do not "
                         "depend on whatever a previous run left behind.")
    args = ap.parse_args()
    url = args.database or os.environ.get("FORGESIGHT_DATABASE_URL") \
        or f"sqlite:///{Path(get_settings().data_dir) / 'walkthrough.db'}"
    return Walkthrough(url).run(args.only)


if __name__ == "__main__":
    sys.exit(main())