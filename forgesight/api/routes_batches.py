"""Batch routes: upload, status, item listing, cancellation (design §11.2).

The limits in §8.1 are enforced here, in the order the design lists them,
because the order is the security property: sniff, then size, then page count,
then decode, then persist. A request that violates any of them is refused
before a row is written, so a rejected upload leaves no state behind.

Objects are written before the database transaction (design §8.1). A crash in
between leaves an unreferenced object, which the reaper sweeps; the reverse
order would leave rows pointing at bytes that do not exist.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from PIL import Image

from forgesight.api.deps import Principal, get_pool_singleton, operator
from forgesight.api.repo import WorkspaceRepo, _iso, delta_ms
from forgesight.api.schemas import (
    BatchListOut,
    BatchOut,
    BoxDiff,
    DetectionOut,
    ItemDiff,
    ItemOut,
    TimingSplit,
)
from forgesight.ingest.sniff import MediaKind, sniff
from forgesight.ingest.validate import (
    ValidationError,
    probe_pdf,
    render_pdf_page,
    validate_image,
)
from forgesight.ledger.claims import Ledger
from forgesight.settings import get_settings
from forgesight.storage.object_store import (
    FsObjectStore,
    SizeExceeded,
    asset_key,
    build_store,
    page_key,
)
from forgesight.vision.postprocess import iou
from forgesight.vision.types import Pool

router = APIRouter(prefix="/v1")


def repo() -> WorkspaceRepo:
    return WorkspaceRepo(get_pool_singleton())


def store():
    return build_store(get_settings())


# -- uploads ----------------------------------------------------------------


@router.post("/batches", response_model=BatchOut, status_code=status.HTTP_202_ACCEPTED)
async def create_batch(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    p: Principal = Depends(operator),
    shadow_candidate_id: str | None = None,
) -> BatchOut:
    """Accept a page batch. The first page's release is pinned for the batch."""
    s = get_settings()
    r = repo()
    st = store()

    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no files supplied")
    if len(files) > s.max_files_per_batch:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(files)} files exceeds the limit of {s.max_files_per_batch}",
        )

    key = request.headers.get("idempotency-key")
    if key:
        existing = r.batch_by_key(p.workspace_id, key)
        if existing:
            # AT-10: a retry with the same key returns the original batch.
            return _batch_out(r, p.workspace_id, existing["id"])

    total_bytes = 0
    for f in files:
        total_bytes += f.size or 0
    if total_bytes > s.max_batch_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"batch is {total_bytes / 1e6:.1f} MB, limit is {s.max_batch_bytes / 1e6:.1f} MB",
        )

    depth = r.queue_depth()
    queued = sum(v.get("queued", 0) for v in depth.values())
    if queued >= s.queue_depth_limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"queue is full ({queued}/{s.queue_depth_limit})",
            headers={"Retry-After": "5"},
        )

    release = _active_release(r, p.workspace_id)
    if release is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "no active release; seed the workspace before uploading",
        )
    candidate_id = release["candidate_id"]

    shadow_id = None
    if shadow_candidate_id:
        shadow = r.get_candidate(p.workspace_id, shadow_candidate_id)
        if shadow is None or not shadow["valid"]:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "unknown or invalid shadow candidate")
        shadow_id = shadow_candidate_id

    # Validate everything before writing anything, so a rejected batch is inert.
    staged: list[dict] = []
    for f in files:
        data = await f.read()
        staged.append(_stage(s, r, st, p.workspace_id, f.filename or "upload", data))

    total_pages = sum(len(x["pages"]) for x in staged)
    if total_pages > s.max_pages_per_batch:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{total_pages} pages exceeds the batch limit of {s.max_pages_per_batch}",
        )

    synthetic = all(x["synthetic"] for x in staged)
    batch_id = r.insert_batch(
        p.workspace_id, release["id"], shadow_id, key, total_pages, synthetic
    )
    rows: list[tuple[str, str, str, str]] = []
    for x in staged:
        for page in x["pages"]:
            rows.append((page["page_id"], candidate_id, Pool.TORCH.value, "primary"))
            if shadow_id:
                rows.append((page["page_id"], shadow_id, Pool.ORT.value, "shadow"))
    r.enqueue_items(p.workspace_id, batch_id, rows)

    return _batch_out(r, p.workspace_id, batch_id)


def _stage(s, r: WorkspaceRepo, st, ws: str, filename: str, data: bytes) -> dict:
    """Validate one upload and persist its asset and page rows.

    Raises ValidationError subclasses as 4xx rather than letting them become a
    500: a rejected file is a client problem with a specific reason.
    """
    res = sniff(data[:16])
    if not res.ok:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, res.reason)
    if len(data) > s.max_file_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{filename}: {len(data)} bytes exceeds {s.max_file_bytes}",
        )

    sha = hashlib.sha256(data).hexdigest()
    pages: list[dict] = []
    synthetic = False
    provenance = f"upload:{filename}"

    if res.kind is MediaKind.PDF:
        try:
            info = probe_pdf(data, s)
        except ValidationError as exc:
            raise HTTPException(_status_for(exc), exc.detail) from exc
        for i in range(info.page_count):
            try:
                rgb = render_pdf_page(data, i, s.render_dpi)
            except ValidationError as exc:
                raise HTTPException(_status_for(exc), exc.detail) from exc
            pages.append({"rgb": rgb, "index": i})
        asset_id = _ensure_asset(r, st, ws, sha, data, "application/pdf",
                                 info.page_count, provenance, False)
    else:
        try:
            v = validate_image(data, s)
        except ValidationError as exc:
            raise HTTPException(_status_for(exc), exc.detail) from exc
        pages.append({"rgb": v.rgb, "index": 0, "normalized": v.normalized})
        asset_id = _ensure_asset(r, st, ws, sha, data, res.kind.value, 1, provenance, False)

    stored: list[dict] = []
    for entry in pages:
        rgb = entry["rgb"]
        ok = encode_and_store(st, rgb, s.max_file_bytes)
        pid = r.insert_page(
            ws, asset_id, entry["index"], rgb.shape[1], rgb.shape[0], s.render_dpi,
            ok["key"], entry.get("normalized"),
        )
        stored.append({
            "page_id": pid, "page_index": entry["index"],
            "width": rgb.shape[1], "height": rgb.shape[0],
        })
    return {"pages": stored, "synthetic": synthetic, "sha256": sha}


def _ensure_asset(r: WorkspaceRepo, st, ws: str, sha: str, data: bytes,
                  media: str, page_count: int, provenance: str, synthetic: bool) -> str:
    existing = r.find_asset(ws, sha)
    if existing:
        return existing["id"]
    key = asset_key(sha)
    st.put_stream(key, io.BytesIO(data), get_settings().max_file_bytes)
    return r.insert_asset(ws, sha, len(data), media, key, page_count, provenance, synthetic)


def encode_and_store(st, rgb: np.ndarray, max_bytes: int) -> dict:
    """Store a rendered page as PNG, content-addressed by the encoded bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    key = page_key(sha)
    if not getattr(st, "exists", lambda _k: True)(key):
        try:
            st.put_stream(key, io.BytesIO(data), max_bytes)
        except SizeExceeded as exc:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"rendered page too large: {exc}",
            ) from exc
    return {"key": key, "sha256": sha, "bytes": len(data)}


def _status_for(exc: ValidationError) -> int:
    from forgesight.vision.types import FailureCode

    if exc.code is FailureCode.PIXEL_LIMIT:
        return status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    if exc.code is FailureCode.UNSUPPORTED_TYPE:
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    return status.HTTP_422_UNPROCESSABLE_ENTITY


def _active_release(r: WorkspaceRepo, ws: str):
    """The channel's active release, not merely the newest release row.

    Resolving "the latest release" by timestamp looks equivalent and is not:
    `created_at` has one-second resolution, so two promotes inside the same
    second tie, and the older release can win. `channel.active_release_id` is
    the field that actually means "what is live", and promote/rollback are the
    only things that change it.
    """
    channel_id = r.ensure_channel(ws)
    with get_pool_singleton().connection() as conn:
        return conn.fetchone(
            "SELECT r.* FROM release r JOIN channel c "
            "ON c.active_release_id = r.id AND c.workspace_id = r.workspace_id "
            "WHERE c.id = ? AND r.workspace_id = ?",
            (channel_id, ws),
        )


# -- reads ------------------------------------------------------------------


@router.get("/batches", response_model=BatchListOut)
def list_batches(p: Principal = Depends(operator), limit: int = 50) -> BatchListOut:
    r = repo()
    with get_pool_singleton().connection() as conn:
        rows = conn.fetchall(
            "SELECT id FROM batch WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
            (p.workspace_id, min(limit, 200)),
        )
    return BatchListOut(items=[_batch_out(r, p.workspace_id, x["id"]) for x in rows])


@router.get("/batches/{batch_id}", response_model=BatchOut)
def get_batch(batch_id: str, p: Principal = Depends(operator)) -> BatchOut:
    r = repo()
    row = r.get_batch(p.workspace_id, batch_id)
    if row is None:
        # Another workspace's id is a 404, never a 403: confirming existence
        # would leak it (AT-11).
        raise HTTPException(status.HTTP_404_NOT_FOUND, "batch not found")
    r.refresh_batch(row["id"])
    return _batch_out(r, p.workspace_id, batch_id)


@router.get("/batches/{batch_id}/items", response_model=list[ItemOut])
def get_items(
    batch_id: str,
    p: Principal = Depends(operator),
    cursor: str | None = None,
    limit: int = 100,
) -> list[ItemOut]:
    r = repo()
    batch = r.get_batch(p.workspace_id, batch_id)
    if batch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "batch not found")
    r.refresh_batch(batch["id"])
    rows = r.list_items(p.workspace_id, batch_id, cursor, min(limit, 500))
    return [_item_out(r, p.workspace_id, row) for row in rows]


@router.post("/batches/{batch_id}/cancel", response_model=BatchOut, status_code=202)
def cancel_batch(batch_id: str, p: Principal = Depends(operator)) -> BatchOut:
    r = repo()
    if r.get_batch(p.workspace_id, batch_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "batch not found")
    Ledger(get_pool_singleton()).cancel_batch(p.workspace_id, batch_id)
    from forgesight.ledger.reaper import Reaper

    Reaper(get_pool_singleton(), get_settings()).refresh_batch_status()
    return _batch_out(r, p.workspace_id, batch_id)


@router.get("/items/{item_id}", response_model=ItemOut)
def get_item(item_id: str, p: Principal = Depends(operator)) -> ItemOut:
    r = repo()
    row = r.get_item(p.workspace_id, item_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "item not found")
    return _item_out(r, p.workspace_id, row)


@router.get("/items/{item_id}/image")
def get_item_image(item_id: str, p: Principal = Depends(operator)):
    """The page image, for a page this caller owns.

    On S3 this redirects to a short-lived presigned URL, which is the shape a
    browser wants: the bytes never pass through the API. The local filesystem
    store has no signatures to give, and a `file://` URL is not something a
    browser will load, so the bytes are streamed instead. Either way the
    workspace check above has already run, so nothing is issued for a page
    belonging to another workspace.
    """
    from fastapi.responses import RedirectResponse, Response

    r = repo()
    row = r.get_item(p.workspace_id, item_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "item not found")
    page = r.get_page(p.workspace_id, row["page_id"])
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "page not found")

    st = store()
    s = get_settings()
    if isinstance(st, FsObjectStore):
        return Response(
            content=st.get_bytes(page["object_key"]),
            media_type="image/png",
            headers={
                # Private and short-lived: an image of someone's document should
                # not sit in a shared cache.
                "Cache-Control": f"private, max-age={min(s.presign_ttl_s, 300)}",
            },
        )
    return RedirectResponse(
        url=st.presign_get(page["object_key"], s.presign_ttl_s),
        status_code=status.HTTP_302_FOUND,
    )


# -- serialisation ----------------------------------------------------------


def _batch_out(r: WorkspaceRepo, ws: str, batch_id: str) -> BatchOut:
    row = r.get_batch(ws, batch_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "batch not found")
    counts = r.batch_counts(ws, batch_id)
    agg = r.batch_timings(ws, batch_id)
    return BatchOut(
        id=row["id"],
        status=row["status"],
        total_items=row["total_items"],
        counts=counts,
        release_id=row["release_id"],
        candidate_id=row["release_candidate_id"],
        candidate_name=row["candidate_name"] or "unknown",
        synthetic=bool(row["synthetic"]),
        created_at=_iso(row["created_at"]) or "",
        finished_at=_iso(row["finished_at"]),
        cancel_requested_at=_iso(row["cancel_requested_at"]),
        timings=_timing_split(agg),
    )


def _timing_split(agg: dict) -> TimingSplit:
    return TimingSplit(
        queue_inclusive_ms=agg.get("queue_inclusive_p50_ms"),
        service_ms=agg.get("service_p50_ms"),
        preprocess_ms=agg.get("preprocess_ms_mean"),
        infer_ms=agg.get("infer_ms_mean"),
        postprocess_ms=agg.get("postprocess_ms_mean"),
        persist_ms=agg.get("persist_ms_mean"),
        batch_wait_ms=agg.get("batch_wait_p50_ms"),
        peak_rss_bytes=agg.get("peak_rss_bytes_max"),
    )


def _item_timings(row: dict) -> TimingSplit:
    received, claimed = row.get("received_at"), row.get("claimed_at")
    persisted, pre_end = row.get("persisted_at"), row.get("preprocess_end")
    infer_start = row.get("infer_start")
    qi = _delta_ms(received, persisted)
    svc = _delta_ms(claimed, persisted)
    bw = _delta_ms(pre_end, infer_start)
    return TimingSplit(
        queue_inclusive_ms=qi,
        service_ms=svc,
        preprocess_ms=row.get("preprocess_ms"),
        infer_ms=row.get("infer_ms"),
        postprocess_ms=row.get("postprocess_ms"),
        persist_ms=row.get("persist_ms"),
        batch_wait_ms=bw,
        peak_rss_bytes=row.get("predicted_peak_rss"),
    )


def _delta_ms(a, b) -> float | None:
    # Routed through the repository's coercion so a SQLite text timestamp and a
    # psycopg datetime are handled identically.
    return delta_ms(a, b)


def _item_out(r: WorkspaceRepo, ws: str, row: dict) -> ItemOut:
    primary = _dets(row.get("primary_detections"))
    shadow = _dets(row.get("shadow_detections"))
    page = r.get_page(ws, row["page_id"])
    diff = _diff(primary, shadow) if shadow else None
    return ItemOut(
        id=row["id"],
        page_id=row["page_id"],
        candidate_id=row["candidate_id"],
        state=row["state"],
        role=row["role"],
        attempts=int(row["attempts"]),
        failure_code=row["failure_code"],
        failure_detail=row["failure_detail"],
        timings=_item_timings(row),
        detections=[DetectionOut(**d) for d in primary],
        shadow_detections=[DetectionOut(**d) for d in shadow],
        diff=ItemDiff(**diff) if diff else None,
        image_url=f"/v1/items/{row['id']}/image",
        page_size=[page["width_px"], page["height_px"]] if page else None,
        synthetic=bool(page["synthetic"]) if page else False,
        provenance=page["provenance"] if page else None,
    )


def _dets(raw) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []


def _diff(primary: list[dict], shadow: list[dict], iou_min: float = 0.9) -> dict:
    """Active vs shadow, matched on class and IoU.

    `relabelled` is a same-region box whose class differs, which is the case a
    naive added/missing pair reports as two unrelated differences and hides.
    """
    added: list[BoxDiff] = []
    missing: list[BoxDiff] = []
    relabelled: list[BoxDiff] = []
    used: set[int] = set()

    for a in primary:
        best, best_iou = None, iou_min
        for j, b in enumerate(shadow):
            if j in used:
                continue
            v = iou(tuple(a["box"]), tuple(b["box"]))
            if v >= best_iou:
                best, best_iou = j, v
        if best is None:
            missing.append(BoxDiff(label=a["label"], kind="missing", box=a["box"],
                                   score=a["score"]))
            continue
        used.add(best)
        b = shadow[best]
        if b["label"] != a["label"]:
            relabelled.append(BoxDiff(
                label=f"{a['label']}->{b['label']}", kind="relabelled", box=a["box"],
                score=a["score"], shadow_score=b["score"],
            ))
    for j, b in enumerate(shadow):
        if j not in used:
            added.append(BoxDiff(label=b["label"], kind="added", box=b["box"],
                                 score=b["score"]))

    denom = len(primary) + len(shadow)
    agreement = (2 * len(used) / denom) if denom else 1.0
    return {
        "agreement": round(agreement, 4),
        "added": added,
        "missing": missing,
        "relabelled": relabelled,
    }


__all__ = ["router"]
