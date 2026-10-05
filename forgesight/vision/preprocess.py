"""Preprocessing with an explicit, hashable profile (design §8.7).

`pil_bilinear` is the reference: it is bit-exact with
`transformers.RTDetrImageProcessor`, which is the slow PIL-backed class the
model cards name. The other resize methods exist so that preprocessing parity
is a *measured* property rather than an assumption -- the same failure mode as
Anomalib #2944 / #3726, where an exported model silently diverged because
antialiasing was dropped between the framework path and the export path.

cv2.INTER_LINEAR does not antialias when downscaling. That is the whole point
of keeping it as a candidate.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from forgesight.vision.types import (
    PreparedItem,
    PreprocessProfile,
    canonical_hash,
    label_map_hash,
)

REF_PROCESSOR_CLASS = "transformers.RTDetrImageProcessor"
RESCALE = 1.0 / 255.0


def load_id2label(config_path) -> dict[int, str]:
    import json

    d = json.loads(config_path.read_text())
    return {int(k): v for k, v in d["id2label"].items()}


def make_profile(
    resize: str = "pil_bilinear",
    target: int = 640,
    render_dpi: int = 150,
    id2label: dict[int, str] | None = None,
    tile_long_side_px: int = 2200,
    tile_aspect: float = 1.6,
    tile_overlap: float = 0.15,
) -> PreprocessProfile:
    return PreprocessProfile(
        resize=resize,  # type: ignore[arg-type]
        target=target,
        render_dpi=render_dpi,
        tile_long_side_px=tile_long_side_px,
        tile_aspect=tile_aspect,
        tile_overlap=tile_overlap,
        label_map_hash=label_map_hash(id2label or {}),
        processor_class=REF_PROCESSOR_CLASS,
    )


def _resize(page_rgb: np.ndarray, method: str, target: int) -> np.ndarray:
    if method == "pil_bilinear":
        return np.asarray(
            Image.fromarray(page_rgb).resize(
                (target, target), resample=Image.Resampling.BILINEAR
            )
        )
    if method == "pil_reduce":
        # Box-reduce first, then a plain bilinear pass. Antialiased, and a
        # different antialiasing kernel from pil_bilinear.
        return np.asarray(
            Image.fromarray(page_rgb).resize(
                (target, target), resample=Image.Resampling.BILINEAR, reducing_gap=2.0
            )
        )
    if method == "cv2_linear":
        return cv2.resize(
            page_rgb, (target, target), interpolation=cv2.INTER_LINEAR
        )
    if method == "cv2_area":
        return cv2.resize(page_rgb, (target, target), interpolation=cv2.INTER_AREA)
    raise ValueError(f"unknown resize method: {method}")


def to_tensor(resized_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> contiguous CHW float32, rescaled to [0,1]."""
    arr = resized_rgb.astype(np.float32) * RESCALE
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


class Preprocessor:
    """Callable that turns one page image into one or more model inputs."""

    def __init__(self, profile: PreprocessProfile, tile_enabled: bool = False):
        self.profile = profile
        self.tile_enabled = tile_enabled

    # -- tiling ------------------------------------------------------------
    def _tiles(self, page_rgb: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int] | None]]:
        p = self.profile
        h, w = page_rgb.shape[:2]
        long_side = max(h, w)
        aspect = long_side / max(1, min(h, w))
        if long_side <= p.tile_long_side_px and aspect <= p.tile_aspect:
            return [(page_rgb, None)]

        # Split along the long axis into 2 tiles with overlap (design §8.8).
        overlap_px = int(round(long_side * p.tile_overlap))
        if w >= h:
            split = w // 2
            a, b = page_rgb[:, : split + overlap_px], page_rgb[:, split:]
            return [(a, (0, 0)), (b, (split, 0))]
        split = h // 2
        a, b = page_rgb[: split + overlap_px, :], page_rgb[split:, :]
        return [(a, (0, 0)), (b, (0, split))]

    def __call__(self, page_rgb: np.ndarray) -> list[PreparedItem]:
        p = self.profile
        if page_rgb.dtype != np.uint8:
            raise ValueError("Preprocessor expects uint8 RGB")
        h, w = page_rgb.shape[:2]

        regions = self._tiles(page_rgb) if self.tile_enabled else [(page_rgb, None)]
        out: list[PreparedItem] = []
        for region, origin in regions:
            rh, rw = region.shape[:2]
            resized = _resize(region, p.resize, p.target)
            tensor = to_tensor(resized)
            out.append(
                PreparedItem(
                    tensor=tensor,
                    page_size=(w, h),
                    # Normalized [0,1] model coords map to page pixels by the
                    # region's own extent: a model pixel m sits at
                    # m * (region_px / target), and a normalized coord n is
                    # n * target model pixels, so the product is n * region_px.
                    scale=(float(rw), float(rh)),
                    tile_origin=origin,
                    preprocess_hash=p.hash,
                )
            )
        return out


def tensor_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(max_abs, mean_abs) difference between two prepared tensors."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()), float(d.mean())


def build_all_preprocessors(
    id2label: dict[int, str], render_dpi: int = 150, target: int = 640
) -> dict[str, Preprocessor]:
    return {
        m: Preprocessor(make_profile(m, target=target, render_dpi=render_dpi, id2label=id2label))
        for m in ("pil_bilinear", "cv2_linear", "cv2_area", "pil_reduce")
    }
