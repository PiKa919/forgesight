"""SynthLayout: owned synthetic document pages with exact ground truth.

The design's central claim is that quality comparisons on this data are
reproducible, because the ground truth is *generated* rather than labelled.
That only holds if the boxes are tight and the generator is deterministic, so
both properties are tested rather than assumed.

Coordinate frames: ReportLab places `y` at the page bottom and counts up; image
and COCO conventions put the origin at the top-left and count down. The
conversion happens in exactly one place, `Box.to_px`.

Every page carries a visible SYNTHETIC marker and a provenance string of the
form `synthetic:<generator>@<version>#seed=<n>`, so a screenshot of this data
can never be mistaken for a real document.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pypdfium2 as pdfium

from forgesight.synth.templates import (
    CLASS_TO_ID,
    EMITTED_CLASSES,
    TEMPLATES,
)

GENERATOR = "forgesight-synth"
VERSION = "1"
TemplateName = Literal["single_column", "two_column", "slide", "two_up"]


def provenance(template: str, seed: int, degrade: str | None = None) -> str:
    prov = f"synthetic:{GENERATOR}@{VERSION}#seed={seed},template={template}"
    return prov + (f",degrade={degrade}" if degrade else "")


def _render(pdf_bytes: bytes, dpi: int) -> tuple[np.ndarray, float, float]:
    """Rasterize page 1. Page metrics must be read before the doc is closed."""
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        page = pdf[0]
        w_pt, h_pt = float(page.get_width()), float(page.get_height())
        scale = dpi / 72.0
        img = np.array(page.render(scale=scale).to_pil().convert("RGB"))
    finally:
        pdf.close()
    return img, w_pt, h_pt


def generate_page(
    template: TemplateName = "two_column",
    seed: int = 42,
    dpi: int = 150,
    degrade: str | None = None,
) -> dict:
    """Generate one page. Deterministic in (template, seed, dpi, degrade)."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; have {sorted(TEMPLATES)}")
    rng = np.random.default_rng(seed)
    layout = TEMPLATES[template](rng)
    # The layout knows its own page size; re-deriving it from the rendered PDF
    # is what previously inverted every box on the landscape templates.
    page_w_pt, page_h_pt = layout.page_w, layout.page_h
    pdf_bytes = layout.buf.getvalue()

    img, w_pt, h_pt = _render(pdf_bytes, dpi)
    if (round(w_pt, 1), round(h_pt, 1)) != (round(page_w_pt, 1), round(page_h_pt, 1)):
        raise AssertionError(
            f"layout page {page_w_pt}x{page_h_pt} != rendered {w_pt}x{h_pt}"
        )
    pt_to_px = dpi / 72.0

    boxes: list[list[float]] = []
    labels: list[str] = []
    class_ids: list[int] = []
    for b in layout.boxes:
        if b.cls not in EMITTED_CLASSES:
            continue
        boxes.append(b.to_px(page_h_pt, pt_to_px))
        labels.append(b.cls)
        class_ids.append(CLASS_TO_ID[b.cls])

    if degrade:
        from forgesight.synth.degrade import apply_degradation

        img = apply_degradation(img, rng, degrade)

    h, w = img.shape[:2]
    return {
        "image": img,
        "boxes": boxes,
        "labels": labels,
        "class_ids": class_ids,
        "dpi": dpi,
        "template": template,
        "seed": seed,
        "degrade": degrade,
        "provenance": provenance(template, seed, degrade),
        "page_size_px": (w, h),
        "synthetic": True,
    }


def generate_pdf(
    template: TemplateName = "two_column", seed: int = 42, dpi: int = 150
) -> bytes:
    """Return the underlying PDF bytes, for the PDF ingest path."""
    rng = np.random.default_rng(seed)
    layout = TEMPLATES[template](rng)
    return layout.buf.getvalue()


# Backwards-compatible alias for the Phase 0 call site.
generate_synthetic_page = generate_page
