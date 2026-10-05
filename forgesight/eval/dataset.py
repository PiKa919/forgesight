"""Dataset version construction (design §9.1).

A `DatasetVersion` is immutable and content-addressed. Its `manifest_hash` is
the sha256 of a canonical JSON listing every `(page_sha, annotation_sha, split)`,
so two runs of the generator on the same seeds produce the same hash and a
changed label silently changes the version. That is what makes "the quality
gate was measured on synth-clean v1" a checkable statement rather than a claim
about intent.

Version numbers are per name and semver: a new set of pages or any changed
annotation is a new version, never an edit.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from forgesight.db.pool import PoolLike
from forgesight.ledger.claims import new_id
from forgesight.settings import Settings
from forgesight.storage.object_store import ObjectStore
from forgesight.synth.generator import generate_page
from forgesight.synth.templates import CLASS_TO_ID
from forgesight.vision.types import canonical_hash

# Dataset specs from design §9.1. Counts are the design's, and the two_up
# fraction matters because that is the tiling case.
SPECS: dict[str, dict] = {
    "synth-clean": {
        "semver": "1",
        "templates": ["single_column", "two_column", "slide", "two_up"],
        "total": 300, "calib": 60, "test": 240, "degrade": None,
    },
    "synth-scan": {
        "semver": "1",
        "templates": ["single_column", "two_column", "slide", "two_up"],
        "total": 240, "calib": 0, "test": 240, "degrade": "scan",
    },
    "synth-bench": {
        "semver": "1",
        "templates": ["single_column", "two_column", "slide", "two_up"],
        "total": 200, "calib": 0, "test": 200, "degrade": None,
        "two_up_fraction": 0.20,
    },
}


@dataclass(slots=True)
class BuiltPage:
    page_id: str
    page_index: int
    split: str
    width: int
    height: int
    dpi: int
    object_key: str
    boxes: list[list[float]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    provenance: str = ""


@dataclass(slots=True)
class BuiltDataset:
    name: str
    semver: str
    manifest_hash: str
    counts: dict
    pages: list[BuiltPage]
    synthetic: bool = True
    license: str = "Apache-2.0 (generated)"

    def as_row(self) -> dict:
        return {
            "name": self.name,
            "semver": self.semver,
            "manifest_hash": self.manifest_hash,
            "generator": "forgesight-synth",
            "generator_version": "1",
            "license": self.license,
            "synthetic": self.synthetic,
            "counts": self.counts,
        }


def annotation_hash(page: BuiltPage) -> str:
    payload = json.dumps(
        {
            "boxes": [[round(v, 3) for v in b] for b in page.boxes],
            "class_ids": page.class_ids,
        },
        sort_keys=True, separators=(",", ":"),
    )
    return canonical_hash(payload)


def build(
    name: str,
    settings: Settings,
    store: ObjectStore,
    limit: int | None = None,
    dpi: int | None = None,
) -> BuiltDataset:
    """Generate a dataset version and persist its pages to the object store.

    `limit` exists so the fast test suite can build a small dataset; the
    manifest hash still covers exactly the pages it contains, so a truncated
    dataset is a different dataset and says so.
    """
    spec = SPECS[name]
    dpi = dpi or settings.render_dpi
    total = limit or spec["total"]
    templates = spec["templates"]
    calib_n = min(spec["calib"], total // 5) if spec["calib"] else 0

    pages: list[BuiltPage] = []
    per_class: dict[str, int] = {}
    per_split: dict[str, int] = {"calib": 0, "test": 0, "bench": 0}

    for i in range(total):
        tpl = templates[i % len(templates)]
        if spec.get("two_up_fraction") and (i % 5) == 0:
            tpl = "two_up"
        seed = 1_000_000 + i
        page = generate_page(
            template=tpl, seed=seed, dpi=dpi, degrade=spec["degrade"]
        )
        split = "calib" if i < calib_n else ("bench" if name == "synth-bench" else "test")
        per_split[split] = per_split.get(split, 0) + 1

        from forgesight.api.routes_batches import encode_and_store

        stored = encode_and_store(store, page["image"], settings.max_file_bytes)
        bp = BuiltPage(
            page_id=new_id("pg"),
            page_index=i,
            split=split,
            width=page["page_size_px"][0],
            height=page["page_size_px"][1],
            dpi=dpi,
            object_key=stored["key"],
            boxes=page["boxes"],
            class_ids=page["class_ids"],
            class_names=page["labels"],
            provenance=page["provenance"],
        )
        for cn in bp.class_names:
            per_class[cn] = per_class.get(cn, 0) + 1
        pages.append(bp)

    # The manifest covers content only: the rendered page's bytes, the
    # annotations, and the split. Including the freshly generated page id would
    # make the hash differ on every run, which is exactly what a content hash
    # must not do -- the page id is an identity, not a property of the data.
    manifest = [
        {"page_sha": Path(bp.object_key).stem, "annotation": annotation_hash(bp),
         "split": bp.split}
        for bp in pages
    ]
    mhash = canonical_hash(manifest)
    counts = {
        "pages": len(pages),
        "by_split": per_split,
        "by_class": dict(sorted(per_class.items())),
    }
    return BuiltDataset(name=name, semver=spec["semver"], manifest_hash=mhash,
                        counts=counts, pages=pages)


def persist(pool: PoolLike, ws: str, dataset: BuiltDataset) -> str:
    """Write the dataset and its annotations. Immutable, so a rebuild of the
    same manifest hash under a new name is a new row and the same content is
    never silently replaced."""
    did = new_id("ds")
    with pool.write() as conn:
        conn.execute(
            "INSERT INTO dataset_version(id, workspace_id, name, semver, "
            "manifest_hash, generator, generator_version, license, synthetic, counts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (did, ws, dataset.name, dataset.semver, dataset.manifest_hash,
             "forgesight-synth", "1", dataset.license, dataset.synthetic,
             json.dumps(dataset.counts, sort_keys=True)),
        )
    return did


def persist_pages(pool: PoolLike, ws: str, asset_owner: str, dataset: BuiltDataset) -> None:
    """Insert page and annotation rows for a built dataset.

    Pages hang off a single synthetic asset row, because a generated page has no
    uploaded original; its provenance lives in the manifest and on each page row.
    """
    del asset_owner
    ph = "?" if pool.dialect.value == "sqlite" else "%s"
    asset_id = dataset_asset_id(pool, ws, dataset)
    version_id = dataset_version_id(pool, ws, dataset)
    with pool.write() as conn:
        for bp in dataset.pages:
            conn.execute(
                f"INSERT INTO page(id, workspace_id, asset_id, page_index, width_px, "
                f"height_px, render_dpi, object_key) VALUES ({', '.join([ph] * 8)})",
                (bp.page_id, ws, asset_id, bp.page_index, bp.width, bp.height,
                 bp.dpi, bp.object_key),
            )
            for box, cid, cname in zip(bp.boxes, bp.class_ids, bp.class_names, strict=True):
                conn.execute(
                    f"INSERT INTO annotation(id, workspace_id, dataset_version_id, page_id, "
                    f"split, class_name, class_id, x1, y1, x2, y2, source) "
                    f"VALUES ({', '.join([ph] * 12)})",
                    (new_id("an"), ws, version_id, bp.page_id, bp.split, cname, cid,
                     box[0], box[1], box[2], box[3], "generator"),
                )


def dataset_asset_id(pool: PoolLike, ws: str, dataset: BuiltDataset) -> str:
    """The single synthetic asset row that owns a dataset's page renders.

    Looked up by manifest hash rather than cached in a module global, so two
    datasets with the same content share the row and two calls in one process
    cannot disagree about which asset id they are using.
    """
    with pool.connection() as conn:
        row = conn.fetchone(
            "SELECT id FROM asset WHERE workspace_id = ? AND sha256 = ?",
            (ws, dataset.manifest_hash),
        )
    if row is not None:
        return row["id"]
    aid = new_id("as")
    ph = "?" if pool.dialect.value == "sqlite" else "%s"
    with pool.write() as conn:
        conn.execute(
            f"INSERT INTO asset(id, workspace_id, sha256, byte_size, media_type, "
            f"object_key, page_count, provenance, synthetic) "
            f"VALUES ({', '.join([ph] * 9)})",
            (aid, ws, dataset.manifest_hash, 0, "image/png",
             f"datasets/{dataset.name}/{dataset.semver}", len(dataset.pages),
             f"synthetic:forgesight-synth@1#dataset={dataset.name}", True),
        )
    return aid


def dataset_version_id(pool: PoolLike, ws: str, dataset: BuiltDataset) -> str:
    with pool.connection() as conn:
        row = conn.fetchone(
            "SELECT id FROM dataset_version WHERE workspace_id = ? AND manifest_hash = ?",
            (ws, dataset.manifest_hash),
        )
    if row is None:
        raise ValueError(f"persist() must run before persist_pages() for {dataset.name}")
    return row["id"]


def load_truth(pool: PoolLike, ws: str, name: str, split: str) -> list:
    """Load ground truth for a dataset version as PageTruth objects."""
    from forgesight.eval.coco_eval import PageTruth

    with pool.connection() as conn:
        rows = conn.fetchall(
            "SELECT p.id, p.width_px, p.height_px, a.x1, a.y1, a.x2, a.y2, a.class_name "
            "FROM annotation a JOIN page p ON p.id = a.page_id "
            "AND p.workspace_id = a.workspace_id "
            "JOIN dataset_version d ON d.id = a.dataset_version_id "
            "AND d.workspace_id = a.workspace_id "
            "WHERE a.workspace_id = ? AND d.name = ? AND a.split = ? "
            "ORDER BY p.id, a.id",
            (ws, name, split),
        )
    by_page: dict[str, PageTruth] = {}
    for r in rows:
        t = by_page.setdefault(
            r["id"], PageTruth(r["id"], r["width_px"], r["height_px"])
        )
        t.boxes.append([r["x1"], r["y1"], r["x2"], r["y2"]])
        t.class_names.append(r["class_name"])
    return list(by_page.values())


def page_object_keys(pool: PoolLike, page_ids: list[str]) -> dict[str, str]:
    """Map page ids to their object keys.

    Lives here rather than in either caller because the worker and the
    evaluation runner both need it, and a private helper imported across modules
    is just a public helper with a worse name.
    """
    if not page_ids:
        return {}
    marks = ", ".join(["?"] * len(page_ids))
    with pool.connection() as conn:
        rows = conn.fetchall(
            f"SELECT id, object_key FROM page WHERE id IN ({marks})", tuple(page_ids)
        )
    return {r["id"]: r["object_key"] for r in rows}


def page_ids_for(pool: PoolLike, ws: str, name: str, split: str) -> list[str]:
    with pool.connection() as conn:
        rows = conn.fetchall(
            "SELECT DISTINCT a.page_id FROM annotation a "
            "JOIN dataset_version d ON d.id = a.dataset_version_id "
            "AND d.workspace_id = a.workspace_id "
            "WHERE a.workspace_id = ? AND d.name = ? AND a.split = ? ORDER BY a.page_id",
            (ws, name, split),
        )
    return [r["page_id"] for r in rows]


def category_ids() -> list[str]:
    """Class names indexed by Docling class id, which is what the models emit."""
    n = max(CLASS_TO_ID.values()) + 1
    names = [f"class_{i}" for i in range(n)]
    for name, idx in CLASS_TO_ID.items():
        names[idx] = name
    return names


def page_bytes(store: ObjectStore, object_key: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(store.get_bytes(object_key))).convert("RGB"))
