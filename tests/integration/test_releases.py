"""Release tests: AT-12, AT-13, AT-14, plus gate arithmetic.

AT-13 is the one that earns the rest. A gate suite that no longer blocks bad
candidates is worse than one that never existed, because it would let them
through silently. So the negative control is asserted to be *blocked*, and the
gate arithmetic is tested against numbers worked out by hand.
"""

from __future__ import annotations

import json

import pytest

from forgesight.api.deps import hash_token
from forgesight.api.repo import WorkspaceRepo
from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.eval.gates import GATE_POLICY, judge, negative_control_blocked
from forgesight.settings import configure, get_settings, reset_settings

pytestmark = pytest.mark.integration


# -- gate arithmetic --------------------------------------------------------


def _cand(name: str = "c"):
    from forgesight.vision.registry import registry

    s = get_settings()
    for bc in registry(s):
        if bc.role == "reference":
            return bc.candidate
    raise AssertionError(f"no reference candidate for {name}")


def _metrics(m: float, per_class: dict | None = None) -> dict:
    return {"map": m, "map50": m, "per_class_ap": per_class or {}}


def _judge(clean_ref, clean_cand, shift_ref, shift_cand,
           agreement=None, provenance=True, per_class=None):
    cand, ref = _cand(), _cand()
    return judge(
        candidate=cand, reference=ref,
        clean={"reference": clean_ref, "candidate": clean_cand},
        shift={"reference": shift_ref, "candidate": shift_cand},
        agreement=agreement, peak_bytes=None,
        budget_bytes=get_settings().worker_mem_budget, provenance_ok=provenance,
    )


def test_identical_candidate_passes_every_gate():
    m = _metrics(0.50, {"table": 0.5, "picture": 0.5, "text": 0.5, "title": 0.5})
    v = _judge(m, dict(m), m, dict(m), agreement=0.995)
    assert all(x["passed"] for x in v), [x for x in v if not x["passed"]]
    assert {x["gate"] for x in v} == set(GATE_POLICY)


def test_a_one_point_quality_drop_fails_g1():
    ref = _metrics(0.50, {"table": 0.5})
    cand = _metrics(0.49, {"table": 0.5})
    v = _judge(ref, cand, ref, dict(cand))
    g1 = {x["gate"]: x for x in v}["G1_quality_clean"]
    assert not g1["passed"]
    assert g1["observed"] == pytest.approx(0.49)
    assert g1["threshold"] == pytest.approx(0.50)


def test_g1_and_g2_use_their_own_budgets():
    ref = _metrics(0.500)
    # 1.5 pt drop: outside G1 (1.0) and outside G2 (2.0)? no, 1.5 < 2.0.
    cand = _metrics(0.485)
    v = {x["gate"]: x for x in _judge(ref, cand, ref, cand)}
    assert not v["G1_quality_clean"]["passed"], "1.5 pt drop exceeds G1's 1.0 pt budget"
    assert v["G2_quality_shift"]["passed"], "1.5 pt drop is inside G2's 2.0 pt budget"

    worse = _metrics(0.470)
    v2 = {x["gate"]: x for x in _judge(ref, worse, ref, worse)}
    assert not v2["G1_quality_clean"]["passed"]
    assert not v2["G2_quality_shift"]["passed"], "3.0 pt drop exceeds G2's 2.0 pt budget"


def test_g3_only_watches_the_critical_classes():
    ref = _metrics(0.50, {"table": 0.50, "picture": 0.50, "text": 0.50, "title": 0.50,
                          "caption": 0.50})
    # A big drop in a non-critical class must not fail G3.
    cand = _metrics(0.50, {"table": 0.50, "picture": 0.50, "text": 0.50, "title": 0.50,
                           "caption": 0.10})
    v = {x["gate"]: x for x in _judge(ref, cand, ref, dict(cand))}
    assert v["G3_critical_classes"]["passed"]

    # A critical class losing more than 3.0 pt must fail it.
    bad = _metrics(0.50, {"table": 0.40, "picture": 0.50, "text": 0.50, "title": 0.50,
                          "caption": 0.50})
    v2 = {x["gate"]: x for x in _judge(ref, bad, ref, dict(bad))}
    assert not v2["G3_critical_classes"]["passed"]
    assert "table" in v2["G3_critical_classes"]["observed"]


def test_g3_treats_a_vanished_critical_class_as_the_worst_case():
    ref = _metrics(0.50, {"table": 0.50, "picture": 0.50, "text": 0.50, "title": 0.50})
    cand = _metrics(0.50, {"picture": 0.50, "text": 0.50, "title": 0.50})
    v = {x["gate"]: x for x in _judge(ref, cand, ref, dict(cand))}
    g3 = v["G3_critical_classes"]
    assert not g3["passed"], "a critical class the candidate never finds is a failure"


def test_g4_threshold_is_exactly_the_policy_value():
    ref = _metrics(0.5)
    assert GATE_POLICY["G4_runtime_parity"]["min"] == 0.98
    assert _judge(ref, dict(ref), ref, dict(ref), agreement=0.98) is not None
    at = {x["gate"]: x for x in _judge(ref, dict(ref), ref, dict(ref), agreement=0.98)}
    assert at["G4_runtime_parity"]["passed"], "0.98 is the inclusive threshold"
    below = {x["gate"]: x for x in _judge(ref, dict(ref), ref, dict(ref), agreement=0.9799)}
    assert not below["G4_runtime_parity"]["passed"]


def test_g4_is_not_applicable_without_a_comparison():
    v = {x["gate"]: x for x in _judge(_metrics(0.5), _metrics(0.5), _metrics(0.5),
                                      _metrics(0.5), agreement=None)}
    g4 = v["G4_runtime_parity"]
    assert g4["passed"]
    assert "not applicable" in g4["detail"]


def test_g6_fails_on_a_broken_provenance_chain():
    ref = _metrics(0.5)
    v = {x["gate"]: x for x in _judge(ref, dict(ref), ref, dict(ref), provenance=False)}
    assert not v["G6_provenance"]["passed"]


def test_verdicts_record_observed_and_threshold():
    ref = _metrics(0.50)
    cand = _metrics(0.40)
    for v in _judge(ref, cand, ref, cand):
        assert set(v) >= {"gate", "passed", "observed", "threshold", "detail"}


# -- negative control (AT-13) -----------------------------------------------


def test_negative_control_detection_helper_needs_both_gates_to_fail():
    both = [
        {"gate": "G1_quality_clean", "passed": False},
        {"gate": "G2_quality_shift", "passed": False},
    ]
    assert negative_control_blocked(both)
    assert not negative_control_blocked([{**both[0]}, {**both[1], "passed": True}])
    assert not negative_control_blocked([both[0]])


@pytest.mark.model
def test_the_320px_candidate_is_actually_blocked_by_the_gates(stack_ctx):
    """AT-13, end to end: the degraded candidate loses quality and is refused."""
    s, pool, _client, ws, _repo = stack_ctx
    from forgesight.api.repo import WorkspaceRepo
    from forgesight.eval import dataset as ds
    from forgesight.eval.runner import run_evaluation
    from forgesight.storage.object_store import FsObjectStore

    store = FsObjectStore(s.objects_dir)
    repo = WorkspaceRepo(pool)
    _candidates(repo, ws)
    for name in ("synth-clean", "synth-scan"):
        built = ds.build(name, s, store, limit=8)
        ds.persist(pool, ws, built)
        ds.persist_pages(pool, ws, None, built)

    neg = [c for c in repo.list_candidates(ws) if "negctl" in c["name"]]
    assert neg, "the negative control candidate is not registered"
    ref = [c for c in repo.list_candidates(ws) if c["name"].startswith("ref-")]
    assert ref, "no reference candidate"

    result = run_evaluation(neg[0], s, pool=pool, store=store)
    by = {v["gate"]: v for v in result.verdicts}
    assert not by["G1_quality_clean"]["passed"], (
        "the 320px negative control passed G1; the gates are not doing their job"
    )
    assert negative_control_blocked(result.verdicts)
    assert not result.passed


# -- promotion (AT-12) and rollback (AT-14) ---------------------------------


@pytest.fixture
def stack_ctx(tmp_path):
    s = get_settings().model_copy(
        update={"data_dir": tmp_path / "data",
                "database_url": f"sqlite:///{tmp_path / 'rel.db'}"}
    )
    s.ensure_dirs()
    configure(s)
    pool = get_pool(s.database_url)
    migrate(pool, verbose=False)
    import forgesight.api.deps as deps

    deps.set_pool(pool)
    repo = WorkspaceRepo(pool)
    ws = repo.create_workspace("releases")
    repo.add_token(ws, hash_token("tok"), "owner", 24)

    from fastapi.testclient import TestClient

    from forgesight.api.app import create_app

    client = TestClient(create_app())
    yield s, pool, client, ws, repo
    pool.close()
    reset_settings()


def _candidates(repo, ws):
    from forgesight.vision.registry import registry

    s = get_settings()
    for bc in registry(s):
        if repo.find_candidate_by_hash(ws, bc.candidate.hash):
            continue
        pp, rt = repo.profiles_for(ws, bc.candidate.artifact, bc.candidate)
        with repo.pool.connection() as conn:
            art = conn.fetchone(
                "SELECT id FROM model_artifact WHERE workspace_id = ? AND sha256 = ?",
                (ws, bc.candidate.artifact.sha256),
            )
        repo.insert_candidate(
            ws, bc.name, bc.candidate.hash, _Row(art["id"]), pp, rt,
            bc.candidate.score_threshold, bc.candidate.label_map_hash,
            bc.candidate.tile_enabled,
        )
    return repo.list_candidates(ws)


class _Row:
    def __init__(self, i):
        self.id = i


def _mark_evaluated(pool, ws, candidate_id, passed=True):
    from forgesight.ledger.claims import new_id

    eid = new_id("ev")
    verdicts = [{"gate": g, "passed": passed, "observed": 1, "threshold": 1, "detail": "t"}
                for g in GATE_POLICY]
    with pool.write() as conn:
        conn.execute(
            "INSERT INTO evaluation_run(id, workspace_id, candidate_id, status, "
            "metrics, gates, dataset_versions, code_sha, env_manifest) "
            "VALUES (?, ?, ?, 'done', '{}', ?, '[]', 'sha', '{}')",
            (eid, ws, candidate_id, json.dumps(verdicts)),
        )
    return eid


def test_promote_with_a_stale_channel_version_conflicts(stack_ctx):
    """AT-12: two promotes against the same version give one 200 and one 409."""
    _s, pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    repo.ensure_channel(ws)
    a, b = cands[0], cands[1]
    _mark_evaluated(pool, ws, a["id"])
    _mark_evaluated(pool, ws, b["id"])
    h = {"Authorization": "Bearer tok"}

    first = client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": a["id"], "expected_channel_version": 0,
        "reason": "first",
    })
    assert first.status_code == 200, first.text

    second = client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": b["id"], "expected_channel_version": 0,  # now stale
        "reason": "second",
    })
    assert second.status_code == 409, second.text

    with pool.connection() as conn:
        rows = conn.fetchall(
            "SELECT id FROM release WHERE workspace_id = ? AND action = 'promote'", (ws,)
        )
    assert len(rows) == 1, "a losing promote must not write a Release row"


def test_promote_is_refused_when_a_gate_failed(stack_ctx):
    _s, pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    target = cands[0]
    _mark_evaluated(pool, ws, target["id"], passed=False)

    r = client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": target["id"], "expected_channel_version": 0, "reason": "no",
    })
    assert r.status_code == 422, r.text
    # FastAPI nests a structured detail under "detail".
    body = r.json()["detail"]
    assert body["code"] == "gate_failed", body
    assert body["failed"], "the 422 must say which gates failed"


def test_promote_requires_a_completed_evaluation(stack_ctx):
    _s, _pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    r = client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": cands[0]["id"], "expected_channel_version": 0, "reason": "x",
    })
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "gates_not_evaluated"


def test_rollback_returns_to_the_previous_candidate(stack_ctx):
    _s, pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    a, b = cands[0], cands[1]
    _mark_evaluated(pool, ws, a["id"])
    _mark_evaluated(pool, ws, b["id"])

    assert client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": a["id"], "expected_channel_version": 0, "reason": "a",
    }).status_code == 200
    assert client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": b["id"], "expected_channel_version": 1, "reason": "b",
    }).status_code == 200

    before = client.get("/v1/releases", headers=h).json()
    assert before["active_candidate_name"] == b["name"]

    rb = client.post("/v1/releases/rollback", headers=h,
                     json={"expected_channel_version": 2, "reason": "revert"})
    assert rb.status_code == 200, rb.text
    after = client.get("/v1/releases", headers=h).json()
    assert after["active_candidate_name"] == a["name"]
    assert after["version"] == 3
    # History is append-only: both the promotes and the rollback are in it.
    assert [x["action"] for x in after["history"]][:1] == ["rollback"]
    assert sum(1 for x in after["history"] if x["action"] == "promote") == 2


def test_rollback_with_no_previous_release_is_422(stack_ctx):
    _s, _pool, client, ws, repo = stack_ctx
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    r = client.post("/v1/releases/rollback", headers=h,
                    json={"expected_channel_version": 0, "reason": "nothing"})
    assert r.status_code == 422, r.text


def test_rollback_is_refused_when_the_artifact_no_longer_matches(stack_ctx):
    """A rolled-back candidate whose bytes changed must not be promoted."""
    _s, pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    a, b = cands[0], cands[1]
    _mark_evaluated(pool, ws, a["id"])
    _mark_evaluated(pool, ws, b["id"])
    client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": a["id"], "expected_channel_version": 0, "reason": "a"})
    client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": b["id"], "expected_channel_version": 1, "reason": "b"})

    # Corrupt exactly one artifact: (workspace_id, sha256) is unique, so a bulk
    # update would fail on the second row rather than test anything.
    with pool.write() as conn:
        row = conn.fetchone(
            "SELECT a.id FROM model_artifact a JOIN candidate c "
            "ON c.artifact_id = a.id AND c.workspace_id = a.workspace_id "
            "WHERE a.workspace_id = ? AND c.id = ?",
            (ws, a["id"]),
        )
        conn.execute("UPDATE model_artifact SET sha256 = ? WHERE id = ?",
                     ("0" * 64, row["id"]))
    r = client.post("/v1/releases/rollback", headers=h,
                    json={"expected_channel_version": 2, "reason": "revert"})
    assert r.status_code == 422, r.text
    assert "integrity" in r.text.lower()


def test_in_flight_items_keep_their_pinned_candidate(stack_ctx):
    """AT-14: a promote mid-load must not change what a running item runs."""
    _s, pool, client, ws, repo = stack_ctx
    cands = _candidates(repo, ws)
    h = {"Authorization": "Bearer tok"}
    repo.ensure_channel(ws)
    a, b = cands[0], cands[1]
    _mark_evaluated(pool, ws, a["id"])
    _mark_evaluated(pool, ws, b["id"])
    client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": a["id"], "expected_channel_version": 0, "reason": "a"})

    import io

    from PIL import Image

    from forgesight.synth.generator import generate_page

    page = generate_page(template="two_column", seed=5, dpi=150)
    r = client.post("/v1/batches", headers=h, files=[
        ("files", ("p.png", (lambda b: (Image.fromarray(page["image"]).save(b, format="PNG"),
                                          b.getvalue())[1])(io.BytesIO()), "image/png")),
    ])
    assert r.status_code == 202, r.text
    batch_id = r.json()["id"]

    items = client.get(f"/v1/batches/{batch_id}/items", headers=h).json()
    assert items, "no work items were created"
    pinned = {i["candidate_id"] for i in items}
    assert pinned == {a["id"]}, f"items should be pinned to the release, got {pinned}"

    # Promote the other candidate, then confirm the pinned rows did not move.
    assert client.post("/v1/releases/promote", headers=h, json={
        "candidate_id": b["id"], "expected_channel_version": 1, "reason": "b",
    }).status_code == 200
    with pool.connection() as conn:
        still = {x["candidate_id"] for x in conn.fetchall(
            "SELECT candidate_id FROM work_item WHERE batch_id = ?", (batch_id,))}
    assert still == {a["id"]}, "in-flight work items changed candidate on promote"

    # A new batch picks up the new release.
    new_batch = client.post("/v1/batches", headers=h, files=[
        ("files", ("q.png", (lambda b: (Image.fromarray(page["image"]).save(b, format="PNG"),
                                          b.getvalue())[1])(io.BytesIO()), "image/png")),
    ])
    new_items = client.get(f"/v1/batches/{new_batch.json()['id']}/items", headers=h).json()
    assert {i["candidate_id"] for i in new_items} == {b["id"]}
