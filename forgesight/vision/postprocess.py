"""Shared NumPy postprocessing (design §8.3).

Both runtimes emit the same raw tensors -- `logits [b, Q, C]` and
`pred_boxes [b, Q, 4]` in cxcywh normalized to the model input -- so one
implementation here serves the torch pool and the ORT pool. That is what lets
the ORT worker image ship without torch.

The implementation is validated against
`transformers.RTDetrImageProcessor.post_process_object_detection` in
tests/model/test_postprocess_parity.py (acceptance test AT-15), because the
whole point of the design is that this is a checked property, not an assumption.

Two details of that reference are easy to get wrong and are reproduced exactly:

1. The selection is a **top-k over the flattened (query x class) matrix**, not
   a per-query argmax. HF takes the top `Q` of the `Q*C` scores, so one query
   can contribute several distinct classes. An argmax-per-query implementation
   silently returns fewer detections and would make every runtime comparison
   and every mAP number wrong.
2. Boxes are scaled by the target size and are **not** clipped. Clipping would
   look tidier and would change results, so it is not done here.

No rounding happens in this module. Scores and boxes keep full precision so
that evaluation and detection-agreement metrics are not silently quantized; the
API layer rounds for transport only.
"""

from __future__ import annotations

import numpy as np

from forgesight.vision.types import Detection, PreparedItem, RawOutputs


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def postprocess_batch(
    raw: RawOutputs,
    items: list[PreparedItem],
    id2label: dict[int, str],
    threshold: float,
    top_k: int | None = None,
) -> list[list[Detection]]:
    """Map raw outputs for one micro-batch back to page-pixel detections.

    `items[i]` describes the model input at batch row `i`. The batch dimension
    is asserted rather than assumed, so a runtime that silently ignores the
    batch axis fails loudly instead of returning wrong boxes.
    """
    n_batch, n_queries, n_classes = raw.logits.shape
    if len(items) != n_batch:
        raise ValueError(f"items ({len(items)}) must match batch dim ({n_batch})")
    if raw.boxes.shape[:2] != (n_batch, n_queries):
        raise ValueError(
            f"boxes {raw.boxes.shape} inconsistent with logits {raw.logits.shape}"
        )
    k = n_queries if top_k is None else min(top_k, n_queries * n_classes)

    probs = sigmoid(raw.logits)  # [b, Q, C]
    boxes = cxcywh_to_xyxy(raw.boxes.astype(np.float64))  # [b, Q, 4], normalized

    out: list[list[Detection]] = []
    for b, item in enumerate(items):
        # Top-k over the flattened (query x class) matrix, matching HF.
        flat = probs[b].reshape(-1)
        if k >= flat.size:
            idx = np.argsort(-flat)
        else:
            idx = np.argpartition(-flat, k - 1)[:k]
            idx = idx[np.argsort(-flat[idx])]
        sel_scores = flat[idx]
        sel_labels = idx % n_classes
        sel_queries = idx // n_classes
        sel_boxes = boxes[b][sel_queries]

        keep = sel_scores > threshold  # HF uses a strict comparison
        if not keep.any():
            out.append([])
            continue
        sel_scores, sel_labels, sel_boxes = (
            sel_scores[keep],
            sel_labels[keep],
            sel_boxes[keep],
        )

        # Normalized model coordinates -> page pixels. `scale` is the region
        # extent in page pixels, so a full-extent box lands on the page edge.
        sx, sy = item.scale
        ox, oy = item.tile_origin or (0, 0)
        x1 = sel_boxes[:, 0] * sx + ox
        y1 = sel_boxes[:, 1] * sy + oy
        x2 = sel_boxes[:, 2] * sx + ox
        y2 = sel_boxes[:, 3] * sy + oy

        dets = [
            Detection(
                class_id=int(sel_labels[i]),
                label=id2label.get(int(sel_labels[i]), f"class_{int(sel_labels[i])}"),
                score=float(sel_scores[i]),
                box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            )
            for i in range(len(sel_scores))
        ]
        out.append(dets)
    return out


def merge_tiles(
    per_tile: list[list[Detection]], iou_threshold: float = 0.6
) -> list[Detection]:
    """Merge detections from adjacent tiles with class-wise NMS (design §8.8).

    Boxes cut by a seam appear twice, once per tile, offset by the tile origin.
    """
    flat = [d for tile in per_tile for d in tile]
    if not flat:
        return []
    order = sorted(range(len(flat)), key=lambda i: -flat[i].score)
    keep: list[int] = []
    for i in order:
        di = flat[i]
        if all(
            di.class_id != flat[j].class_id or iou(di.box, flat[j].box) <= iou_threshold
            for j in keep
        ):
            keep.append(i)
    return [flat[i] for i in keep]


def nms(dets: list[Detection], iou_threshold: float = 0.6) -> list[Detection]:
    """Greedy class-wise NMS."""
    return merge_tiles([dets], iou_threshold)


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
