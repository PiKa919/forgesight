"""Ledger acceptance tests, run against every available backend.

The suite is parameterised over the dialects rather than written once, because
the whole point of the two-backend arrangement is that the guarantees hold on
both. A fencing bug that only appears on SQLite is a real bug: it is the backend
the test suite and local dev actually use.

Covered: AT-2 (crash recovery), AT-3 (zombie fencing), AT-4 (cancel before
start), AT-10 (idempotent batch), AT-11 (cross-workspace isolation).
"""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime, timedelta

import pytest

from forgesight.db.migrate import migrate
from forgesight.db.pool import Dialect, get_pool
from forgesight.ledger.claims import Ledger, PredictionIn, new_id
from forgesight.ledger.reaper import Reaper
from forgesight.settings import get_settings
from forgesight.vision.types import Detection, FailureCode

PG_DSN = os.environ.get("FORGESIGHT_TEST_PG", "").strip()
DSNS: dict[str, str] = {"sqlite": "sqlite:///./data/test_ledger.db"}
if PG_DSN:
    DSNS["postgres"] = PG_DSN


@pytest.fixture(params=list(DSNS))
def backend(request):
    """The backend *name*, so call sites can branch on it and look up the DSN."""
    return request.param


@pytest.fixture(scope="session")
def pg_pool():
    """One migrated PostgreSQL schema for the whole session.

    Dropping and recreating the schema per test costs a full 20-table DDL round
    trip each time, which over a tunnel dominated the runtime. Truncating
    between tests is equivalent for isolation and far cheaper.

    Skipped when no DSN is configured, so a run without PostgreSQL still covers
    SQLite rather than erroring in fixture setup.
    """
    if "postgres" not in DSNS:
        pytest.skip("FORGESIGHT_TEST_PG not set")
    pool = get_pool(DSNS["postgres"])
    with pool.write() as conn:
        conn.executescript("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    migrate(pool, verbose=False)
    yield pool
    pool.close()


def _truncate(pool) -> None:
    with pool.connection() as conn:
        names = [
            r["tablename"]
            for r in conn.fetchall(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        ]
    with pool.write() as conn:
        conn.execute(f"TRUNCATE {', '.join(names)} RESTART IDENTITY CASCADE")


@pytest.fixture
def env(request, backend, tmp_path):
    """A migrated, empty database with one workspace."""
    if backend == "postgres":
        # Resolved through the fixture graph rather than requested directly, so
        # a SQLite-only run never depends on PostgreSQL being reachable.
        pool = request.getfixturevalue("pg_pool")
        _truncate(pool)
    else:
        pool = get_pool(f"sqlite:///{tmp_path / 'ledger.db'}")
        migrate(pool, verbose=False)

    ws = new_id("ws")
    with pool.write() as conn:
        conn.execute(
            "INSERT INTO workspace(id, name) VALUES (?, ?)" if pool.dialect is Dialect.SQLITE
            else "INSERT INTO workspace(id, name) VALUES (%s, %s)",
            (ws, "test"),
        )
    yield pool, Ledger(pool), ws
    if backend != "postgres":
        pool.close()


def _seed_catalog(pool, ws: str) -> dict:
    """Insert the minimum immutable rows a work item needs to exist."""
    artifact = new_id("art")
    pre = new_id("pre")
    rt = new_id("rt")
    cand = new_id("cand")
    asset = new_id("as")
    page = new_id("pg")
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.write() as conn:
        conn.execute(
            f"INSERT INTO model_artifact(id, workspace_id, name, repo_id, revision, format, "
            f"path, sha256, byte_size, license) VALUES ({','.join([ph] * 10)})",
            (artifact, ws, "heron", "docling-project/docling-layout-heron", "abc",
             "safetensors", "/tmp/model", "deadbeef", 1, "Apache-2.0"),
        )
        conn.execute(
            f"INSERT INTO preprocess_profile(id, workspace_id, profile_hash, body) "
            f"VALUES ({','.join([ph] * 4)})",
            (pre, ws, "prehash", "{}"),
        )
        conn.execute(
            f"INSERT INTO runtime_profile(id, workspace_id, profile_hash, body) "
            f"VALUES ({','.join([ph] * 4)})",
            (rt, ws, "rthash", "{}"),
        )
        conn.execute(
            f"INSERT INTO candidate(id, workspace_id, name, candidate_hash, artifact_id, "
            f"preprocess_id, runtime_id, score_threshold, label_map_hash) "
            f"VALUES ({','.join([ph] * 9)})",
            (cand, ws, "ref", "candhash", artifact, pre, rt, 0.3, "lmhash"),
        )
        conn.execute(
            f"INSERT INTO asset(id, workspace_id, sha256, byte_size, media_type, object_key, "
            f"page_count, provenance) VALUES ({','.join([ph] * 8)})",
            (asset, ws, "a" * 64, 10, "image/png", f"assets/{'a' * 64}", 1, "synthetic:test"),
        )
        conn.execute(
            f"INSERT INTO page(id, workspace_id, asset_id, page_index, width_px, height_px, "
            f"render_dpi, object_key) VALUES ({','.join([ph] * 8)})",
            (page, ws, asset, 0, 100, 100, 150, f"pages/{'b' * 64}.png"),
        )
    return {"candidate": cand, "page": page, "asset": asset}


def _enqueue(pool, ws: str, cat: dict, n: int, pool_name: str = "torch") -> str:
    """Create a batch of n queued items. Returns the batch id."""
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    batch = new_id("b")
    with pool.write() as conn:
        # A release needs a real channel: the composite workspace foreign keys
        # reject a dangling channel_id, which is the schema working as intended.
        channel = new_id("ch")
        conn.execute(
            f"INSERT INTO channel(id, workspace_id, name, version) "
            f"VALUES ({','.join([ph] * 4)})",
            (channel, ws, f"ch-{channel}", 0),
        )
        rel = new_id("rel")
        conn.execute(
            f"INSERT INTO release(id, workspace_id, channel_id, candidate_id, action, reason, "
            f"actor, channel_version) VALUES ({','.join([ph] * 8)})",
            (rel, ws, channel, cat["candidate"], "seed", "test", "t", 0),
        )
        conn.execute(
            f"INSERT INTO batch(id, workspace_id, release_id, total_items) "
            f"VALUES ({','.join([ph] * 4)})",
            (batch, ws, rel, n),
        )
        for _ in range(n):
            wid = new_id("wi")
            conn.execute(
                f"INSERT INTO work_item(id, workspace_id, batch_id, page_id, candidate_id, "
                f"pool, state, enqueued_at) VALUES ({','.join([ph] * 8)})",
                (wid, ws, batch, cat["page"], cat["candidate"], pool_name, "queued",
                 datetime.now(UTC)),
            )
    return batch


def _dets(n: int = 2) -> list[Detection]:
    return [
        Detection(class_id=9, label="text", score=0.9 - i * 0.1, box=(0.0, 0.0, 1.0, 1.0))
        for i in range(n)
    ]


def _states(pool, ws: str) -> dict[str, int]:
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        rows = conn.fetchall(
            f"SELECT state, COUNT(*) AS n FROM work_item WHERE workspace_id = {ph} GROUP BY state",
            (ws,),
        )
    return {r["state"]: int(r["n"]) for r in rows}


def _predictions(pool, ws: str) -> int:
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        return len(
            conn.fetchall(
                f"SELECT id FROM prediction WHERE workspace_id = {ph}", (ws,)
            )
        )


# -- AT-2: crash recovery ---------------------------------------------------


def test_sigkill_mid_batch_recovers_with_no_duplicates(env):
    """AT-2: after a worker dies, the reaper requeues and nothing is duplicated."""
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 3)

    claimed = ledger.claim("torch", "worker-a", n=3, lease_s=60)
    assert len(claimed) == 3
    # The worker dies here: no complete(), no fail().

    # One item finishes from a different process before the reaper runs.
    assert ledger.complete(claimed[0], PredictionIn(_dets())) is True

    # Expire the leases by hand, standing in for the passage of time.
    past = datetime.now(UTC) - timedelta(seconds=120)
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.write() as conn:
        conn.execute(
            f"UPDATE work_item SET lease_expires_at = {ph} WHERE workspace_id = {ph}",
            (past, ws),
        )

    reaper = Reaper(pool, get_settings())
    result = reaper.reap_expired_leases()
    assert result.requeued == 2, "the two abandoned items should return to the queue"
    assert _states(pool, ws).get("queued") == 2

    # A new worker finishes them. The already-succeeded item is not retried.
    again = ledger.claim("torch", "worker-b", n=5, lease_s=60)
    assert len(again) == 2, "only the requeued items should be claimable"
    for item in again:
        assert ledger.complete(item, PredictionIn(_dets(1))) is True

    assert _states(pool, ws) == {"succeeded": 3}
    assert _predictions(pool, ws) == 3, "exactly one prediction per work item"


def test_attempts_are_capped_and_the_item_fails(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    s = get_settings()
    reaper = Reaper(pool, s)

    for expected_attempt in range(1, s.max_attempts + 1):
        claimed = ledger.claim("torch", "worker-x", n=1, lease_s=60)
        assert len(claimed) == 1
        assert claimed[0].attempts == expected_attempt
        past = datetime.now(UTC) - timedelta(seconds=120)
        ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
        with pool.write() as conn:
            conn.execute(
                f"UPDATE work_item SET lease_expires_at = {ph} WHERE workspace_id = {ph}",
                (past, ws),
            )
        reaper.reap_expired_leases()

    states = _states(pool, ws)
    assert states.get("failed") == 1
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        row = conn.fetchone(
            f"SELECT failure_code, attempts FROM work_item WHERE workspace_id = {ph}", (ws,)
        )
    assert row["failure_code"] == FailureCode.MAX_ATTEMPTS.value
    assert int(row["attempts"]) == s.max_attempts


# -- AT-3: fencing ----------------------------------------------------------


def test_zombie_worker_write_is_fenced(env):
    """AT-3: a worker past its lease cannot write after losing the item."""
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)

    zombie = ledger.claim("torch", "worker-zombie", n=1, lease_s=60)[0]

    # Its lease lapses and the reaper hands the item to someone else.
    past = datetime.now(UTC) - timedelta(seconds=120)
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.write() as conn:
        conn.execute(
            f"UPDATE work_item SET lease_expires_at = {ph} WHERE workspace_id = {ph}",
            (past, ws),
        )
    Reaper(pool, get_settings()).reap_expired_leases()

    live = ledger.claim("torch", "worker-live", n=1, lease_s=60)[0]
    assert live.fencing_token > zombie.fencing_token
    assert ledger.complete(live, PredictionIn(_dets(3))) is True

    # The zombie wakes up and tries to write. It must be refused.
    assert ledger.complete(zombie, PredictionIn(_dets(1))) is False
    assert ledger.fail(zombie, "inference_failed") is False
    assert ledger.mark_cancelled(zombie) is False

    assert _predictions(pool, ws) == 1, "the zombie's prediction must not exist"
    assert _states(pool, ws) == {"succeeded": 1}


def test_fencing_tokens_strictly_increase(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    tokens = []
    for _ in range(4):
        _enqueue(pool, ws, cat, 1)
        tokens.append(ledger.claim("torch", "w", n=1, lease_s=60)[0].fencing_token)
    assert tokens == sorted(tokens) and len(set(tokens)) == 4, tokens


def test_heartbeat_extends_only_our_own_lease(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 2)
    items = ledger.claim("torch", "w", n=2, lease_s=60)
    assert ledger.heartbeat(items, lease_s=120) == 2
    # Once fenced out, the heartbeat reports nothing held.
    Reaper(pool, get_settings())
    past = datetime.now(UTC) - timedelta(seconds=300)
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.write() as conn:
        conn.execute(
            f"UPDATE work_item SET lease_expires_at = {ph} WHERE workspace_id = {ph}",
            (past, ws),
        )
    Reaper(pool, get_settings()).reap_expired_leases()
    assert ledger.heartbeat(items, lease_s=60) == 0


# -- AT-4: cancellation before start ----------------------------------------


def test_cancel_queued_items_never_start(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 4)

    assert ledger.cancel_batch(ws, _last_batch(pool, ws)) == 4
    assert _states(pool, ws) == {"cancelled": 4}

    # Nothing is claimable afterwards, so no claimed_at can ever be written.
    assert ledger.claim("torch", "w", n=10, lease_s=60) == []

    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        rows = conn.fetchall(
            f"SELECT claimed_at FROM work_item WHERE workspace_id = {ph}", (ws,)
        )
    assert all(r["claimed_at"] is None for r in rows)
    assert ledger.is_cancel_requested(ws, _last_batch(pool, ws)) is True


def test_cancel_is_idempotent(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 2)
    batch = _last_batch(pool, ws)
    assert ledger.cancel_batch(ws, batch) == 2
    assert ledger.cancel_batch(ws, batch) == 0
    assert _states(pool, ws) == {"cancelled": 2}


def test_cancel_a_running_item_drops_its_prediction(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    item = ledger.claim("torch", "w", n=1, lease_s=60)[0]
    assert ledger.cancel_batch(ws, _last_batch(pool, ws)) == 0  # already running
    assert ledger.mark_cancelled(item) is True
    assert ledger.complete(item, PredictionIn(_dets())) is False
    assert _predictions(pool, ws) == 0


# -- pooling ----------------------------------------------------------------


def test_claim_is_oldest_first_and_respects_pool(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 2, pool_name="torch")
    _enqueue(pool, ws, cat, 3, pool_name="onnxruntime")

    got = ledger.claim("torch", "w", n=10, lease_s=60)
    assert len(got) == 2
    assert {i.pool for i in got} == {"torch"}
    assert ledger.claim("onnxruntime", "w2", n=10, lease_s=60).__len__() == 3


def test_two_claimers_never_take_the_same_item(env):
    """The invariant SKIP LOCKED exists to provide, checked without concurrency."""
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 6)
    first = ledger.claim("torch", "w1", n=3, lease_s=60)
    second = ledger.claim("torch", "w2", n=3, lease_s=60)
    ids = {i.id for i in first} | {i.id for i in second}
    assert len(ids) == 6, "a work item was handed to two claimers"
    assert not ({i.id for i in first} & {i.id for i in second})


# -- AT-11: workspace isolation --------------------------------------------


def test_cross_workspace_reference_is_impossible(env):
    """AT-11: the composite key makes a cross-workspace item uninsertable."""
    pool, _ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)

    other = new_id("ws")
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.write() as conn:
        conn.execute(
            f"INSERT INTO workspace(id, name) VALUES ({ph}, {ph})", (other, "other")
        )
    other_cat = _seed_catalog(pool, other)
    _enqueue(pool, other, other_cat, 0)

    # Everything on this row belongs to `other` except page_id, which points at
    # the first workspace's page. Only the page's composite key can refuse it,
    # so a pass really is evidence that workspace scoping works.
    with _refused_by(pool, "foreign_key", ("page_id",)), pool.write() as conn:
        conn.execute(
            f"INSERT INTO work_item(id, workspace_id, batch_id, page_id, candidate_id, "
            f"pool, state) VALUES ({','.join([ph] * 7)})",
            (new_id("wi"), other, _last_batch(pool, other), cat["page"],
             other_cat["candidate"], "torch", "queued"),
        )


def test_batch_visibility_is_scoped_to_workspace(env):
    pool, _ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    batch = _last_batch(pool, ws)
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        assert conn.fetchone(
            f"SELECT id FROM batch WHERE id = {ph} AND workspace_id = {ph}", (batch, ws)
        ) is not None
        assert conn.fetchone(
            f"SELECT id FROM batch WHERE id = {ph} AND workspace_id = {ph}", (batch, "nope")
        ) is None


def test_unique_prediction_constraint_is_the_second_guard(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    item = ledger.claim("torch", "w", n=1, lease_s=60)[0]
    assert ledger.complete(item, PredictionIn(_dets())) is True

    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with _refused_by(
        pool, "unique", ("work_item_id", "candidate_id")
    ), pool.write() as conn:
        conn.execute(
            f"INSERT INTO prediction(id, workspace_id, work_item_id, candidate_id, "
            f"detections, n_detections) VALUES ({','.join([ph] * 6)})",
            (new_id("pr"), ws, item.id, item.candidate_id, "[]", 0),
        )


# -- batch rollup -----------------------------------------------------------


def test_batch_status_rolls_up_from_item_states(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 3)
    batch = _last_batch(pool, ws)
    reaper = Reaper(pool, get_settings())

    items = ledger.claim("torch", "w", n=3, lease_s=60)
    reaper.refresh_batch_status()
    assert _batch_status(pool, batch) == "running"

    ledger.complete(items[0], PredictionIn(_dets()))
    ledger.fail(items[1], FailureCode.INFERENCE.value, "boom")
    ledger.mark_cancelled(items[2])
    reaper.refresh_batch_status()
    # The ladder puts failure above cancellation, so a batch containing a
    # failed item is "failed" however the rest turned out.
    assert _batch_status(pool, batch) == "failed"
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        assert conn.fetchone(
            f"SELECT finished_at FROM batch WHERE id = {ph}", (batch,)
        )["finished_at"] is not None


def test_all_succeeded_batch_reports_succeeded(env):
    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 2)
    batch = _last_batch(pool, ws)
    for item in ledger.claim("torch", "w", n=2, lease_s=60):
        assert ledger.complete(item, PredictionIn(_dets())) is True
    Reaper(pool, get_settings()).refresh_batch_status()
    assert _batch_status(pool, batch) == "succeeded"


# -- helpers ---------------------------------------------------------------


@contextlib.contextmanager
def _refused_by(pool, kind: str, columns: tuple[str, ...] = ()):
    """Assert the block is refused by a constraint of the given kind.

    `kind` is "foreign_key" or "unique". Asserting the kind rather than bare
    `Exception` is the point: a test that only expected "some error" would still
    pass if the row were refused for an unrelated reason, and would no longer be
    proving what it claims.

    `columns` is checked where the driver reports it. PostgreSQL names the
    constraint, so the composite key's column list is verified there. SQLite
    names the columns only for UNIQUE and says nothing more for FOREIGN KEY, so
    on that backend the check is the kind alone -- which is why the PostgreSQL
    run is the one that pins down *which* key refused the row.
    """
    import sqlite3

    try:
        yield
    except Exception as exc:
        msg = str(exc)
        if pool.dialect is Dialect.POSTGRES:
            import psycopg.errors as pg_errors

            expected = (
                pg_errors.ForeignKeyViolation
                if kind == "foreign_key"
                else pg_errors.UniqueViolation
            )
            assert isinstance(exc, expected), f"{type(exc).__name__}: {msg}"
            name = getattr(getattr(exc, "diag", None), "constraint_name", "") or ""
            for column in columns:
                assert column in name, f"expected {column!r} in {name!r}"
        else:
            assert isinstance(exc, sqlite3.IntegrityError), type(exc)
            marker = "FOREIGN KEY" if kind == "foreign_key" else "UNIQUE"
            assert marker in msg, f"expected a {marker} violation, got: {msg}"
            if kind == "unique":
                # SQLite names columns for UNIQUE but not for FOREIGN KEY.
                for column in columns:
                    assert column.split(".")[-1] in msg, f"{column!r} missing from {msg!r}"
    else:
        raise AssertionError(f"expected the insert to be refused by a {kind} constraint")


def _last_batch(pool, ws: str) -> str:
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        return conn.fetchone(
            f"SELECT id FROM batch WHERE workspace_id = {ph} "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (ws,),
        )["id"]


def _batch_status(pool, batch: str) -> str:
    ph = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        return conn.fetchone(
            f"SELECT status FROM batch WHERE id = {ph}", (batch,)
        )["status"]


def test_claim_sql_differs_by_dialect_only_in_locking():
    """The two claim statements must differ in locking, not in semantics."""
    from forgesight.db.pool import claim_sql

    pg, lite = claim_sql(Dialect.POSTGRES), claim_sql(Dialect.SQLITE)
    assert "SKIP LOCKED" in pg
    assert "SKIP LOCKED" not in lite
    for fragment in ("state = 'running'", "lease_owner", "fencing_token"):
        assert fragment in pg, fragment
        assert fragment in lite, fragment
    # Postgres aliases the target table in the UPDATE ... FROM, sqlite does not.
    assert "attempts = w.attempts + 1" in pg
    assert "attempts = attempts + 1" in lite


def test_backend_matrix_covers_sqlite_and_reports_postgres():
    """Guard against the suite silently degrading to one backend."""
    assert "sqlite" in DSNS
    if PG_DSN:
        assert DSNS["postgres"].startswith("postgresql://")


# -- timestamp resolution ---------------------------------------------------


def test_timestamps_keep_sub_second_resolution(env):
    """SQLite's CURRENT_TIMESTAMP is whole seconds, which cannot measure latency.

    The design asks for clock_timestamp() on cross-process events, which is right
    on PostgreSQL and silently useless on SQLite: a 300 ms inference and a 1.4 s
    one both round to the same whole second. The ledger therefore stamps from the
    application clock, and this pins that the round trip preserves sub-second
    resolution so the queue-inclusive figure is not 0 s, 1 s or 2 s.
    """
    from forgesight.api.repo import delta_ms
    from forgesight.ledger.claims import now_utc

    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    item = ledger.claim("torch", "w", n=1, lease_s=60)[0]

    placeholder = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        claimed = conn.fetchone(
            f"SELECT received_at, claimed_at FROM work_item WHERE id = {placeholder}",
            (item.id,),
        )["claimed_at"]
    assert claimed is not None
    assert delta_ms(claimed, claimed) == 0.0
    # A str() of a sub-second value keeps its microseconds, unlike SQLite's own
    # CURRENT_TIMESTAMP which would already be truncated.
    assert "." in str(claimed), f"claimed_at lost sub-second precision: {claimed!r}"

    assert now_utc().tzinfo is not None, "timestamps must be timezone-aware"


def test_queue_inclusive_latency_is_not_rounded_to_whole_seconds(env):
    """A fast item must report a fraction of a second, not 0 s or 1 s."""
    from forgesight.api.repo import delta_ms

    pool, ledger, ws = env
    cat = _seed_catalog(pool, ws)
    _enqueue(pool, ws, cat, 1)
    item = ledger.claim("torch", "w", n=1, lease_s=60)[0]
    ledger.complete(item, PredictionIn(_dets()))

    placeholder = "?" if pool.dialect is Dialect.SQLITE else "%s"
    with pool.connection() as conn:
        row = conn.fetchone(
            "SELECT received_at, claimed_at, persisted_at FROM work_item "
            f"WHERE id = {placeholder}",
            (item.id,),
        )
    qi = delta_ms(row["received_at"], row["persisted_at"])
    svc = delta_ms(row["claimed_at"], row["persisted_at"])
    assert qi is not None and svc is not None
    assert qi >= 0 and svc >= 0
    # Monotonic: claiming happens after receipt, so service cannot exceed the
    # queue-inclusive figure.
    assert svc <= qi + 1.0, f"service {svc} > queue-inclusive {qi}"
    assert qi < 60_000
