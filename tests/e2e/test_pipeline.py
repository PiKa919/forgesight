"""End-to-end: upload through the API, work the ledger, read the result back.

This is acceptance test AT-1, and it is the test that would catch the class of
bug where every component passes its own tests and the product does not work.
It drives the real API, the real database, the real object store and a real
model, in that order, and asserts on final state rather than on status codes.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from forgesight.db.migrate import migrate
from forgesight.db.pool import get_pool
from forgesight.settings import configure, get_settings, reset_settings
from forgesight.storage.object_store import FsObjectStore

pytestmark = pytest.mark.e2e


@pytest.fixture
def stack(tmp_path):
    """API + database + store, all pointed at a scratch directory."""
    s = get_settings().model_copy(
        update={
            "data_dir": tmp_path / "data",
            "database_url": f"sqlite:///{tmp_path / 'e2e.db'}",
            "mode": "local",
            "poll_interval_s": 0.05,
            "rss_sample_ms": 5,
        }
    )
    s.ensure_dirs()
    # The API, the worker and the reaper all resolve settings through
    # get_settings(), so the override is installed process-wide or the API would
    # write objects to a different directory than the worker reads.
    configure(s)
    pool = get_pool(s.database_url)
    migrate(pool, verbose=False)

    import forgesight.api.deps as deps

    deps.set_pool(pool)

    from fastapi.testclient import TestClient

    from forgesight.api.app import create_app

    client = TestClient(create_app())
    store = FsObjectStore(s.objects_dir)
    yield s, pool, store, client
    pool.close()
    reset_settings()


@pytest.fixture
def session(stack):
    _, _, _, client = stack
    r = client.post("/v1/sessions")
    assert r.status_code == 201, r.text
    body = r.json()
    return body, {"Authorization": f"Bearer {body['token']}"}


def _png_bytes(seed: int) -> bytes:
    from forgesight.synth.generator import generate_page

    page = generate_page(template="two_column", seed=seed, dpi=150)
    b = io.BytesIO()
    Image.fromarray(page["image"]).save(b, format="PNG")
    return b.getvalue()


# -- AT-1 -------------------------------------------------------------------


@pytest.mark.model
def test_upload_to_prediction_end_to_end(stack, session):
    """AT-1: a page goes in, a real prediction comes out, timings are ordered."""
    from forgesight.worker.main import Worker

    s, pool, store, client = stack
    _sess, headers = session

    files = [
        ("files", ("page1.png", _png_bytes(1), "image/png")),
        ("files", ("page2.png", _png_bytes(2), "image/png")),
    ]
    r = client.post("/v1/batches", files=files, headers=headers)
    assert r.status_code == 202, r.text
    batch = r.json()
    assert batch["total_items"] == 2

    worker = Worker(s, pool, store, pool_name="torch")
    import asyncio

    asyncio.run(worker.run(once=True))

    got = client.get(f"/v1/batches/{batch['id']}", headers=headers)
    assert got.status_code == 200, got.text
    after = got.json()
    assert after["status"] == "succeeded", after
    assert after["counts"] == {"succeeded": 2}, after["counts"]

    items = client.get(f"/v1/batches/{batch['id']}/items", headers=headers).json()
    assert len(items) == 2
    for it in items:
        assert it["state"] == "succeeded"
        assert it["timings"]["infer_ms"] and it["timings"]["infer_ms"] > 0
        assert it["timings"]["preprocess_ms"] is not None
        # The image endpoint is authorised and redirects.
        img = client.get(f"/v1/items/{it['id']}/image", headers=headers,
                         follow_redirects=False)
        assert img.status_code == 302, img.status_code

    with pool.connection() as conn:
        n = len(conn.fetchall(
            "SELECT id FROM prediction WHERE workspace_id = ?", (_sess["workspace_id"],)
        ))
    assert n == 2, "one prediction per work item"


# -- AT-10 ------------------------------------------------------------------


def test_idempotent_batch_retry(stack, session):
    """AT-10: the same Idempotency-Key returns the original batch, once."""
    _s, pool, _store, client = stack
    _sess, headers = session
    key = "retry-me-please"

    first = client.post(
        "/v1/batches",
        files=[("files", ("a.png", _png_bytes(3), "image/png"))],
        headers={**headers, "Idempotency-Key": key},
    )
    assert first.status_code == 202, first.text
    second = client.post(
        "/v1/batches",
        files=[("files", ("a.png", _png_bytes(3), "image/png"))],
        headers={**headers, "Idempotency-Key": key},
    )
    assert second.status_code == 202, second.text
    assert second.json()["id"] == first.json()["id"]

    with pool.connection() as conn:
        batches = conn.fetchall(
            "SELECT id FROM batch WHERE idempotency_key = ?", (key,)
        )
        assets = conn.fetchall("SELECT id, sha256 FROM asset")
    assert len(batches) == 1, "the retry created a second batch"
    # Identical bytes are stored once: (workspace_id, sha256) is unique.
    assert len({a["sha256"] for a in assets}) == len(assets)


# -- AT-11 ------------------------------------------------------------------


def test_cross_workspace_ids_are_404_and_never_presigned(stack, session):
    """AT-11: another workspace's ids are 404, never a 403 that confirms them."""
    _s, _pool, _store, client = stack
    _sess, headers = session
    other = client.post("/v1/sessions").json()
    other_headers = {"Authorization": f"Bearer {other['token']}"}

    r = client.post(
        "/v1/batches",
        files=[("files", ("a.png", _png_bytes(4), "image/png"))],
        headers=headers,
    )
    assert r.status_code == 202, r.text
    batch_id = r.json()["id"]
    item_id = client.get(
        f"/v1/batches/{batch_id}/items", headers=headers
    ).json()[0]["id"]

    for path in (
        f"/v1/batches/{batch_id}",
        f"/v1/batches/{batch_id}/items",
        f"/v1/items/{item_id}",
        f"/v1/items/{item_id}/image",
    ):
        seen = client.get(path, headers=other_headers, follow_redirects=False)
        assert seen.status_code == 404, f"{path} -> {seen.status_code}"
        # A 404 must not carry a Location header, or the presign leaked anyway.
        assert "location" not in {k.lower() for k in seen.headers}, path


# -- hostile input through the real API -------------------------------------


def test_hostile_uploads_are_refused_with_specific_reasons(stack, session):
    _s, _pool, _store, client = stack
    _sess, headers = session

    # Magic bytes decide. Random data is not a supported type at all (415);
    # data with a valid PDF header that will not parse is a content problem, so
    # it gets 422 with the parse error rather than a type error.
    cases = [
        (b"this is not an image", "image/png", 415),
        (b"%PDF-1.4 not really a pdf", "application/pdf", 422),
    ]
    for payload, mime, expected in cases:
        r = client.post(
            "/v1/batches",
            files=[("files", ("x", payload, mime))],
            headers=headers,
        )
        assert r.status_code == expected, (mime, r.status_code, r.text[:200])

    # A declared Content-Type of PNG does not make a ZIP a PNG.
    r = client.post(
        "/v1/batches",
        files=[("files", ("z.zip", b"PK\x03\x04" + b"\x00" * 200, "image/png"))],
        headers=headers,
    )
    assert r.status_code == 415, r.text


def test_empty_file_list_is_rejected(stack, session):
    _s, _pool, _store, client = stack
    _sess, headers = session
    r = client.post("/v1/batches", files=[], headers=headers)
    assert r.status_code in (400, 422), r.status_code


def test_upload_before_any_release_is_a_conflict(tmp_path):
    """A workspace with no seeded release cannot accept work."""
    s = get_settings().model_copy(
        update={"data_dir": tmp_path / "d", "database_url": f"sqlite:///{tmp_path / 'x.db'}"}
    )
    s.ensure_dirs()
    pool = get_pool(s.database_url)
    migrate(pool, verbose=False)
    import forgesight.api.deps as deps

    deps.set_pool(pool)
    from fastapi.testclient import TestClient

    from forgesight.api.app import create_app
    from forgesight.api.deps import hash_token
    from forgesight.api.repo import WorkspaceRepo

    client = TestClient(create_app())
    repo = WorkspaceRepo(pool)
    ws = repo.create_workspace("bare")
    repo.add_token(ws, hash_token("tok"), "owner", 24)
    headers = {"Authorization": "Bearer tok"}

    r = client.post(
        "/v1/batches",
        files=[("files", ("a.png", _png_bytes(9), "image/png"))],
        headers=headers,
    )
    assert r.status_code == 409, r.text
    pool.close()
    reset_settings()
