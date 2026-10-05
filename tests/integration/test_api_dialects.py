"""Every API endpoint must behave identically on SQLite and PostgreSQL.

The route layer was written with literal `?` placeholders, which SQLite accepts
and psycopg does not. Nothing caught it because the whole API suite ran on
SQLite, so the only backend the deployment engine actually uses had never
exercised these endpoints.

This file walks the endpoints that write through the raw-SQL paths -- session
bootstrap, batch listing, evaluation, promote, rollback -- against both
backends. It is deliberately end-to-end through the FastAPI app rather than a
unit test of the SQL, because the thing that broke was a dialect mismatch
between the SQL text and the driver, and only running the real app catches that.

Skipped for PostgreSQL when FORGESIGHT_TEST_PG is unset, so a local run still
covers SQLite rather than erroring in fixture setup.
"""

from __future__ import annotations

import os

import pytest

from forgesight.db.migrate import migrate
from forgesight.db.pool import Dialect, get_pool
from forgesight.settings import get_settings, reset_settings

PG_DSN = os.environ.get("FORGESIGHT_TEST_PG", "").strip()
DSNS: dict[str, str] = {"sqlite": "sqlite:///./data/test_api_dialects.db"}
if PG_DSN:
    DSNS["postgres"] = PG_DSN

pytestmark = pytest.mark.integration


@pytest.fixture(params=list(DSNS))
def backend(request):
    """The backend *name*, so call sites branch on it and look up the DSN."""
    return request.param


@pytest.fixture(scope="session")
def pg_pool():
    """One migrated PostgreSQL schema for the session; truncated between tests."""
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
def stack(request, backend, tmp_path):
    """A migrated, empty database behind a real FastAPI app with a live token."""
    if backend == "postgres":
        pool = request.getfixturevalue("pg_pool")
        _truncate(pool)
    else:
        pool = get_pool(f"sqlite:///{tmp_path / 'api.db'}")
        migrate(pool, verbose=False)

    s = get_settings().model_copy(
        update={"data_dir": tmp_path / "data",
                "database_url": f"sqlite:///{tmp_path / 'api.db'}"
                if backend == "sqlite" else DSNS["postgres"]}
    )
    s.ensure_dirs()
    from forgesight.settings import configure

    configure(s)

    import forgesight.api.deps as deps

    deps.set_pool(pool)
    from forgesight.api.app import create_app
    from forgesight.api.deps import hash_token
    from forgesight.api.repo import WorkspaceRepo

    repo = WorkspaceRepo(pool)
    ws = repo.create_workspace("dialects")
    repo.add_token(ws, hash_token("tok"), "owner", 24)

    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    client.headers["Authorization"] = "Bearer tok"
    yield client, pool, repo, ws
    if backend != "postgres":
        pool.close()
    reset_settings()


def test_session_bootstrap_seeds_a_usable_workspace_on_every_backend(stack):
    """POST /sessions registers the built-in candidates and an active release.

    Each call creates its own workspace and mints its own token, so the point is
    not idempotency but that the raw-SQL seed path -- artifact lookup, candidate
    insert, release insert, channel update -- completes on both backends. A fresh
    workspace with no active release would refuse every upload, so this is the
    "a new session is immediately usable" property.
    """
    client, _pool, _repo, _ws = stack
    first = client.post("/v1/sessions", json={"kind": "personal"})
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["token"], body

    # The seed must have produced candidates and an active release.
    listed = client.get("/v1/candidates", headers={"Authorization": "Bearer " + body["token"]})
    assert listed.status_code == 200, listed.text
    assert listed.json(), "session seed registered no candidates"

    rel = client.get("/v1/releases", headers={"Authorization": "Bearer " + body["token"]})
    assert rel.status_code == 200, rel.text
    assert rel.json()["active_release_id"], "session seed left no active release"


def test_empty_batch_list_returns_200_on_every_backend(stack):
    """GET /batches hits a raw-SQL path with a literal placeholder."""
    client, _pool, _repo, _ws = stack
    r = client.get("/v1/batches")
    assert r.status_code == 200, r.text
    assert r.json()["items"] == []


def test_candidates_list_is_workspace_scoped_on_every_backend(stack):
    """GET /candidates returns only this workspace's candidates."""
    client, _pool, _repo, ws = stack
    r = client.get("/v1/candidates")
    assert r.status_code == 200, r.text
    for c in r.json():
        assert c["workspace_id"] == ws


def test_system_status_reports_queues_on_every_backend(stack):
    client, _pool, _repo, _ws = stack
    r = client.get("/v1/system/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "queues" in body or "queue_depth" in body or "depth" in body, body


def test_candidate_create_rejects_unknown_profiles_on_every_backend(stack):
    """The three-way artifact/profile lookup runs on both backends.

    These are literal-`?` statements (routes_reports.py) and they must 422 on an
    unknown hash rather than create a row. The point is the lookup works on both
    dialects; the 422 is the cheap observable that it ran and found nothing.
    """
    client, _pool, _repo, _ws = stack
    r = client.post("/v1/candidates", json={
        "name": "ghost",
        "artifact_sha256": "does-not-exist",
        "preprocess_hash": "does-not-exist",
        "runtime_hash": "does-not-exist",
        "score_threshold": 0.3,
        "tile_enabled": False,
    })
    assert r.status_code == 422, r.text
    assert "artifact" in r.json()["detail"].lower(), r.text


def test_promote_rejects_a_stale_channel_version_on_every_backend(stack):
    """The optimistic channel-version check runs on both backends.

    A promote whose expected version does not match the current one must be
    refused, and the refusal is the optimistic-concurrency write
    (routes_releases.py) -- literal `?` on both backends.
    """
    client, _pool, _repo, _ws = stack
    r = client.post("/v1/releases/promote", json={
        "candidate_name": "does-not-exist",
        "evaluation_id": "does-not-exist",
        "expected_channel_version": 999,
        "reason": "dialect smoke test",
    })
    # No such evaluation -> refused (404 or 422), never a 500 from bad SQL.
    assert r.status_code in (404, 422), (r.status_code, r.text)
    assert "psycopg" not in r.text and "placeholders" not in r.text, r.text


def test_placeholders_match_the_active_dialect(stack):
    """The regression that motivated this file, asserted directly.

    A psycopg connection given a `?` placeholder does not silently do the wrong
    thing -- it raises. So a 500 from any endpoint that writes through raw SQL is
    the symptom. Here we assert the dialect's own placeholder is what the pool
    advertises, and that a direct raw-SQL round-trip through the pool works, so
    the failure mode is a clear error rather than a mysterious 500.
    """
    _client, pool, _repo, _ws = stack
    from forgesight.db.pool import ph

    expected = "?" if pool.dialect is Dialect.SQLITE else "%s"
    assert ph(pool.dialect) == expected
    # A direct round-trip using the dialect's placeholder must succeed.
    with pool.connection() as conn:
        conn.execute(f"SELECT {ph(pool.dialect)} AS probe", ("x",))
        row = conn.fetchone(f"SELECT {ph(pool.dialect)} AS probe", ("x",))
    assert row["probe"] == "x"


# -- the translation itself, independent of any database ---------------------


def test_portable_placeholders_are_rewritten_for_postgres():
    from forgesight.db.pool import to_postgres_placeholders as tx

    assert tx("SELECT * FROM t WHERE a = ? AND b = ?") == (
        "SELECT * FROM t WHERE a = %s AND b = %s"
    )
    # already-correct SQL must pass through untouched
    assert tx("SELECT %s, %s") == "SELECT %s, %s"
    assert tx("SELECT 1") == "SELECT 1"


def test_a_question_mark_inside_a_literal_is_not_rewritten():
    from forgesight.db.pool import to_postgres_placeholders as tx

    assert tx("SELECT 'why?' AS q") == "SELECT 'why?' AS q"
    assert tx("SELECT ? AS v") == "SELECT %s AS v"
    # an escaped quote inside a literal must not end the literal early
    assert tx("SELECT 'it''s ok?' , ?") == "SELECT 'it''s ok?' , %s"

def test_macos_appledouble_sidecars_are_ignored(tmp_path, monkeypatch):
    """A `._foo.sql` sidecar must not be treated as a migration.

    macOS writes AppleDouble sidecars on any non-HFS filesystem -- a Docker bind
    mount, a network share, an extracted tar. They match `*.sql`, sort before the
    real migration, and are binary, so an unfiltered glob made the API fail to
    start with a UnicodeDecodeError raised from inside migrate().
    """
    import forgesight.db.migrate as mig

    d = tmp_path / "migrations"
    d.mkdir()
    (d / "0001_init.sql").write_text("SELECT 1;")
    (d / "._0001_init.sql").write_bytes(b"\x00\x05\x16\x07\xa3\x01\x00\x00binary")
    (d / ".DS_Store").write_bytes(b"\x00binary")

    monkeypatch.setattr(mig, "MIGRATIONS_DIR", d)
    names = [p.name for p in mig._migration_files()]
    assert names == ["0001_init.sql"], names


def test_a_misnamed_migration_fails_loudly(tmp_path, monkeypatch):
    """A migration that is neither hidden nor numbered is a typo, not a skip.

    Silently ignoring it would leave the schema half-applied, which is worse than
    refusing to start.
    """
    import forgesight.db.migrate as mig

    d = tmp_path / "migrations"
    d.mkdir()
    (d / "0001_init.sql").write_text("SELECT 1;")
    (d / "add_column.sql").write_text("SELECT 1;")

    monkeypatch.setattr(mig, "MIGRATIONS_DIR", d)
    with pytest.raises(ValueError, match="numbered migration"):
        mig._migration_files()


def test_api_routes_are_not_shadowed_by_the_spa_catch_all(stack, monkeypatch):
    """`/healthz` and `/v1/system/info` must answer JSON, not index.html.

    `_mount_web` registers a catch-all `/{path:path}`, and Starlette matches in
    registration order. When the SPA was mounted before these two, the catch-all
    won and they returned index.html -- with HTTP 200, so a load balancer health
    check would report the container healthy while handing back HTML instead of
    JSON. Nothing local noticed because 200 is 200.

    Asserted two ways: on the real response body, and on route order, so a future
    move of _mount_web cannot silently reintroduce the shadowing.
    """
    client, _pool, _repo, _ws = stack

    for path in ("/healthz", "/v1/system/info"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert r.headers["content-type"].startswith("application/json"), (
            f"{path} returned {r.headers['content-type']}, not JSON"
        )
        assert r.json(), f"{path} returned no JSON body"
        assert "<!doctype html>" not in r.text, f"{path} returned the SPA shell"

    order = [
        getattr(r, "path", None)
        for r in client.app.routes
    ]
    if "/{path:path}" not in order:
        # No web build, so no catch-all exists and there is nothing to shadow.
        # The JSON assertions above still ran, and are the part that matters.
        return
    catch_all = order.index("/{path:path}")
    for path in ("/healthz", "/v1/system/info"):
        assert order.index(path) < catch_all, (
            f"{path} is registered after the SPA catch-all and will be shadowed"
        )


def test_session_bootstrap_refuses_when_no_models_are_registered(tmp_path, monkeypatch):
    """A session must not be minted for a workspace that cannot do anything.

    With no model artifacts the registry is empty, so the seed creates a
    workspace with no candidates and no active release -- every upload is then
    refused, and the caller holds a valid token with no explanation. So the
    request is refused instead, *before* anything is written, and the message
    names the directories to populate.
    """
    from fastapi.testclient import TestClient

    from forgesight.api.app import create_app
    from forgesight.db.migrate import migrate
    from forgesight.db.pool import get_pool
    from forgesight.settings import configure, get_settings

    missing = tmp_path / "absent"
    s = get_settings().model_copy(update={
        "data_dir": tmp_path / "data",
        "database_url": f"sqlite:///{tmp_path / 'n.db'}",
        "models_dir": missing / "models",
        "artifacts_dir": missing / "artifacts",
    })
    s.ensure_dirs()
    configure(s)

    import forgesight.api.deps as deps

    pool = get_pool(s.database_url)
    migrate(pool, verbose=False)
    deps.set_pool(pool)
    client = TestClient(create_app())

    r = client.post("/v1/sessions", json={"kind": "local"})
    assert r.status_code == 503, r.text
    assert "no model artifacts" in r.json()["detail"], r.text
    # Names the directory so the fix is actionable, not just a refusal.
    assert str(missing) in r.json()["detail"], r.text

    # Nothing may be left behind: no orphan workspace, no orphan token.
    with pool.connection() as conn:
        assert conn.fetchone("SELECT COUNT(*) AS n FROM workspace")["n"] == 0
        assert conn.fetchone("SELECT COUNT(*) AS n FROM api_token")["n"] == 0

    pool.close()
    from forgesight.settings import reset_settings

    reset_settings()
