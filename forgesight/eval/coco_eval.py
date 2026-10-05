"""COCO-style detection evaluation (design §12.3).

`pycocotools` does the arithmetic, and a hand-computed fixture pins it
(acceptance test AT-19). A metric implementation nobody has checked against a
known answer is exactly the kind of component that quietly inflates a report.

Two conventions worth stating because they are easy to get wrong:

* Detections are scored from 0.05, not from the operating threshold. The
  operating point is chosen separately on the calibration split and then frozen.
* Greedy matching assigns each ground-truth box to at most one detection, in
  descending score order, at the IoU being evaluated. pycocotools implements
  this; reimplementing it is how you get a number that looks plausible and is
  not comparable to anything.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field

import numpy as np

from forgesight.vision.postprocess import iou
from forgesight.vision.types import Detection

EVAL_SCORE_FLOOR = 0.05


@dataclass(slots=True)
class PageTruth:
    """Ground truth for one page.

    Classes are named, not numbered. The models emit a label string, and storing
    a number as well invites the two from drifting apart -- an index space on one
    side and a name on the other, which is a silent mis-scoring rather than an
    error. `category_ids()` still supplies the full ordered list so that
    per-class AP covers classes a particular page happens not to contain.
    """

    page_id: str
    width: int
    height: int
    boxes: list[list[float]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PagePrediction:
    page_id: str
    detections: list[Detection]


def to_coco(
    truths: list[PageTruth],
    predictions: list[PagePrediction],
    category_ids: list[str],
) -> tuple[dict, list[dict]]:
    """Build a COCO ground-truth dict and a detection list.

    Images are renumbered densely from 0 so that pycocotools does not care what
    the database's page ids are, and so a filtered dataset still has contiguous
    image indices.
    """
    cat_index = {name: i + 1 for i, name in enumerate(category_ids)}
    images, annotations = [], []
    ann_id = 1
    for i, t in enumerate(truths):
        images.append({"id": i, "width": t.width, "height": t.height,
                       "page_id": t.page_id})
        for box, name in zip(t.boxes, t.class_names, strict=True):
            if name not in cat_index:
                raise KeyError(
                    f"ground-truth class {name!r} is not in the category list; "
                    f"refusing to score it under a different class"
                )
            x1, y1, x2, y2 = box
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                # A degenerate box cannot be matched by anyone; dropping it is
                # better than letting it make every candidate's mAP a little
                # lower for a reason unrelated to the model.
                continue
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": cat_index[name],
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_index[name], "name": name} for name in category_ids],
    }

    index = {img["page_id"]: img["id"] for img in images}
    dets = []
    for p in predictions:
        image_id = index.get(p.page_id)
        if image_id is None:
            continue
        for d in p.detections:
            if d.score < EVAL_SCORE_FLOOR:
                continue
            if d.label not in cat_index:
                # A detection of a class outside the scored set is dropped, not
                # folded into some other category.
                continue
            x1, y1, x2, y2 = d.box
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            dets.append({
                "image_id": image_id,
                "category_id": cat_index[d.label],
                "bbox": [x1, y1, w, h],
                "score": d.score,
            })
    return gt, dets


def _coco():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    return COCO, COCOeval


def evaluate(
    truths: list[PageTruth],
    predictions: list[PagePrediction],
    category_ids: list[str],
) -> dict:
    """mAP@[.5:.95], mAP@.5, and per-class AP, at the standard IoU sweep."""
    COCO, COCOeval = _coco()
    gt, dets = to_coco(truths, predictions, category_ids)

    with contextlib.redirect_stdout(io.StringIO()):
        api = COCO()
        api.dataset = gt
        api.createIndex()
        if not dets:
            # COCOeval on an empty detection set returns -1 rather than 0.0,
            # which is indistinguishable from "not implemented". Report an
            # explicit zero so a candidate that predicts nothing cannot pass a
            # quality gate by accident.
            return {
                "map": 0.0, "map50": 0.0, "per_class_ap": {},
                "n_images": len(gt["images"]), "n_annotations": len(gt["annotations"]),
                "n_detections": 0, "note": "no detections above the score floor",
            }
        api_dt = api.loadRes(dets)
        e = COCOeval(api, api_dt, "bbox")
        e.params.catIds = [i + 1 for i in range(len(category_ids))]
        e.evaluate()
        e.accumulate()
        e.summarize()

    per_class: dict[str, float] = {}
    # COCOeval orders precision as [iou, recall, class, area, maxDet], and the
    # class axis runs in the order of params.catIds, which is 1-based while
    # category_ids is a plain 0-based list -- so the position is the index, not
    # the category id.
    precision = e.eval["precision"]
    iou_thr = np.array(e.params.iouThrs)
    for idx in range(len(e.params.catIds)):
        p = precision[:, :, idx, 0, -1]
        p = p[p > -1]
        if p.size:
            per_class[category_ids[idx]] = round(float(p.mean()), 5)

    return {
        "map": round(float(e.stats[0]), 5),
        "map50": round(float(e.stats[1]), 5),
        "map75": round(float(e.stats[2]), 5),
        "per_class_ap": per_class,
        "n_images": len(gt["images"]),
        "n_annotations": len(gt["annotations"]),
        "n_detections": len(dets),
        "iou_thresholds": [round(float(t), 3) for t in iou_thr],
    }


# -- operating point --------------------------------------------------------


def detection_agreement(
    reference: list[Detection], candidate: list[Detection], iou_threshold: float = 0.9
) -> float:
    """2*matched / (|ref| + |cand|) on class and IoU (design §12.3).

    Greedy in descending score order. That is not identical to optimal
    Hungarian assignment, but it is the convention the design specifies, and it
    is stable under a change in the number of detections -- which an optimal
    matching is not.
    """
    if not reference and not candidate:
        return 1.0
    denom = len(reference) + len(candidate)
    if denom == 0:
        return 1.0
    pairs = sorted(
        ((iou(a.box, b.box), i, j) for i, a in enumerate(reference)
         for j, b in enumerate(candidate) if a.class_id == b.class_id),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    for v, i, j in pairs:
        if v < iou_threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
    return 2 * matched / denom


def best_f1_threshold(
    truths: list[PageTruth],
    predictions_by_threshold: dict[float, list[PagePrediction]],
    iou: float = 0.5,
) -> tuple[float, float]:
    """Pick the threshold maximising F1 at IoU 0.5, and return (threshold, f1).

    Computed on the calibration split only, then frozen. Selecting the operating
    point on the same pages the quality gate is measured on would be measuring
    the threshold search as if it were model quality.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in sorted(predictions_by_threshold):
        preds = predictions_by_threshold[t]
        tp = fp = fn = 0
        pred_index = {p.page_id: p for p in preds}
        for truth in truths:
            got = pred_index.get(truth.page_id)
            dets = [d for d in (got.detections if got else []) if d.score >= t]
            used: set[int] = set()
            for box, name in zip(truth.boxes, truth.class_names, strict=True):
                best, best_iou = None, iou
                for j, d in enumerate(dets):
                    if j in used or d.label != name:
                        continue
                    v = iou_overlap(box, d.box)
                    if v >= best_iou:
                        best, best_iou = j, v
                if best is not None:
                    used.add(best)
                    tp += 1
                else:
                    fn += 1
            fp += len(dets) - len(used)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def iou_overlap(box: list[float], other: tuple[float, float, float, float]) -> float:
    return iou((box[0], box[1], box[2], box[3]), other)


def precision_recall(
    truths: list[PageTruth], predictions: list[PagePrediction], threshold: float, iou: float = 0.5
) -> tuple[float, float]:
    tp = fp = fn = 0
    pred_index = {p.page_id: p for p in predictions}
    for truth in truths:
        got = pred_index.get(truth.page_id)
        dets = [d for d in (got.detections if got else []) if d.score >= threshold]
        used: set[int] = set()
        for box, name in zip(truth.boxes, truth.class_names, strict=True):
            best, best_iou = None, iou
            for j, d in enumerate(dets):
                if j in used or d.label != name:
                    continue
                v = iou_overlap(box, d.box)
                if v >= best_iou:
                    best, best_iou = j, v
            if best is not None:
                used.add(best)
                tp += 1
            else:
                fn += 1
        fp += len(dets) - len(used)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def raw_output_parity(a: np.ndarray, b: np.ndarray) -> dict:
    """Isolate the runtime from preprocessing (design §12.3)."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "shape": list(a.shape),
    }


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True)
