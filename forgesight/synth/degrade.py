"""Paired synthetic domain shift (design §9.1, `synth-scan`).

The point is *paired* comparison: the same underlying page is degraded with a
fixed seed, so a quality difference between `synth-clean` and `synth-scan`
isolates the degradation rather than the content. A scan of a page looks
different from the clean render in the ways a real scan does: slightly rotated,
slightly blurred, compressed, unevenly lit and noisy.
"""

from __future__ import annotations

import cv2
import numpy as np

DEGRADATIONS = ("rotate", "blur", "jpeg", "illumination", "noise", "scan")


def _rotate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-1.5, 1.5))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = round(max(h, w) * 0.02) + 2
    return cv2.warpAffine(
        img, m, (w + 2 * border, h + 2 * border), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(0.5, 1.2))
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(35, 61))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _illumination(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    strength = float(rng.uniform(0.15, 0.45))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # A smooth linear gradient plus a soft vignette, as a page lit from one side.
    gx, gy = rng.uniform(-1, 1, size=2) * strength
    field = 1.0 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    cy, cx = h / 2, w / 2
    field -= strength * 0.3 * (((yy - cy) ** 2 + (xx - cx) ** 2) / (cx**2 + cy**2))
    out = img.astype(np.float32) * np.clip(field, 0.5, 1.4)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(2.0, 7.0))
    n = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def apply_degradation(img: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "scan":
        # The realistic composite, in the order a scanner would apply it.
        img = _rotate(img, rng)
        img = _illumination(img, rng)
        img = _blur(img, rng)
        img = _noise(img, rng)
        return _jpeg(img, rng)
    if kind == "rotate":
        return _rotate(img, rng)
    if kind == "blur":
        return _blur(img, rng)
    if kind == "jpeg":
        return _jpeg(img, rng)
    if kind == "illumination":
        return _illumination(img, rng)
    if kind == "noise":
        return _noise(img, rng)
    raise ValueError(f"unknown degradation {kind!r}; have {DEGRADATIONS}")
