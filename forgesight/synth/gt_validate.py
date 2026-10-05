"""Ground-truth validity checks (design §9.2).

The design's quality claims only mean anything if the ground truth is sound, so
these are hard assertions with thresholds, not warnings. A dataset that fails
validation is not usable for evaluation and `validate_page` says so.

Rules, as written in the design:
  * every box must contain ink (>= 2% non-background pixels),
  * >= 99.5% of ink pixels must fall inside some box, excluding page margins,
  * no two boxes of the same class may overlap with IoU > 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forgesight.vision.postprocess import iou

MIN_INK_FRACTION = 0.02
MIN_INK_COVERAGE = 0.995
MAX_SAME_CLASS_IOU = 0.3
MARGIN_PX = 60  # excluded from the ink-coverage denominator


@dataclass(slots=True)
class GtIssue:
    page_seed: int
    rule: str
    detail: str
    severity: str = "error"  # error | warning


@dataclass(slots=True)
class GtReport:
    pages: int = 0
    boxes: int = 0
    issues: list[GtIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def error_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.issues:
            if i.severity == "error":
                out[i.rule] = out.get(i.rule, 0) + 1
        return out


def _ink_mask(img: np.ndarray) -> np.ndarray:
    """Pixels meaningfully darker than paper."""
    gray = img.astype(np.float32).mean(axis=2)
    return gray < 240.0


def validate_page(page: dict) -> list[GtIssue]:
    issues: list[GtIssue] = []
    seed = int(page.get("seed", -1))
    img = page["image"]
    boxes = page["boxes"]
    labels = page["labels"]
    h, w = img.shape[:2]

    if not boxes:
        issues.append(GtIssue(seed, "empty", "page produced no ground-truth boxes"))
        return issues

    inside = np.zeros((h, w), dtype=bool)
    for i, (box, lbl) in enumerate(zip(boxes, labels, strict=True)):
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            issues.append(
                GtIssue(seed, "out_of_bounds", f"box {i} ({lbl}) {box} vs page {w}x{h}")
            )
            continue
        xi1, yi1 = int(np.floor(x1)), int(np.floor(y1))
        xi2, yi2 = int(np.ceil(x2)), int(np.ceil(y2))
        patch = img[yi1:yi2, xi1:xi2]
        if patch.size == 0:
            issues.append(GtIssue(seed, "empty_box", f"box {i} ({lbl}) has no pixels"))
            continue
        ink = float(np.mean(patch.astype(np.float32).mean(axis=2) < 240.0))
        if ink < MIN_INK_FRACTION:
            issues.append(
                GtIssue(
                    seed, "no_ink", f"box {i} ({lbl}) ink fraction {ink:.4f} < {MIN_INK_FRACTION}"
                )
            )
        inside[yi1:yi2, xi1:xi2] = True

    # Ink coverage, ignoring the page margin band.
    mask = _ink_mask(img)
    core = np.zeros_like(mask)
    m = MARGIN_PX
    core[m : h - m, m : w - m] = mask[m : h - m, m : w - m]
    total = int(core.sum())
    if total:
        covered = int((core & inside).sum())
        frac = covered / total
        if frac < MIN_INK_COVERAGE:
            issues.append(
                GtIssue(
                    seed,
                    "ink_uncovered",
                    f"only {frac:.4f} of core ink is inside a box (< {MIN_INK_COVERAGE})",
                )
            )

    # Same-class overlap.
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if labels[i] != labels[j]:
                continue
            v = iou(tuple(boxes[i]), tuple(boxes[j]))
            if v > MAX_SAME_CLASS_IOU:
                issues.append(
                    GtIssue(
                        seed,
                        "same_class_overlap",
                        f"boxes {i},{j} ({labels[i]}) IoU {v:.3f} > {MAX_SAME_CLASS_IOU}",
                    )
                )
    return issues


def validate_dataset(pages: list[dict]) -> GtReport:
    rep = GtReport()
    for p in pages:
        rep.pages += 1
        rep.boxes += len(p.get("boxes", []))
        rep.issues.extend(validate_page(p))
    return rep
