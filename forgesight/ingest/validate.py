"""Upload validation and PDF rendering (design §8.1, acceptance tests AT-8, AT-9).

The ordering here is the security property, not a style choice:

1. **Sniff** the magic bytes. The declared Content-Type is never trusted.
2. **Read the header** to get dimensions, *before* allocating any pixel buffer.
   A 50 000 x 50 000 PNG header is rejected here, so a decompression bomb costs a
   few hundred bytes of parsing rather than 7.5 GB of RAM (AT-8).
3. **Decode**, with `PIL.Image.MAX_IMAGE_PIXELS` as a backstop and
   `ImageFile.LOAD_TRUNCATED_IMAGES` left off so a truncated JPEG fails loudly
   instead of yielding a half-image that looks like a valid result.
4. **Normalize** to 8-bit RGB, applying EXIF orientation. What normalisation was
   applied is recorded on the page row rather than being silently assumed.

PyMuPDF is deliberately absent: it is AGPL. PDF rasterization uses pypdfium2
(BSD-3/Apache-2.0), which the design verified.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile, ImageOps

from forgesight.ingest.dims import probe as probe_dims
from forgesight.ingest.sniff import MediaKind, SniffResult, sniff
from forgesight.settings import Settings
from forgesight.vision.types import FailureCode

# Truncated files must raise rather than decode partially.
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ValidationError(Exception):
    def __init__(self, code: FailureCode, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    rgb: np.ndarray
    width: int
    height: int
    media_type: str
    normalized: str


def probe_image_header(data: bytes, max_pixels: int) -> tuple[int, int, MediaKind]:
    """Read dimensions without decoding. Raises on a pixel-count bomb."""
    res = sniff(data[:16])
    if not res.ok:
        raise ValidationError(FailureCode.UNSUPPORTED_TYPE, res.reason)
    if res.kind is MediaKind.PDF:
        raise ValidationError(
            FailureCode.UNSUPPORTED_TYPE, "use probe_pdf for PDF input"
        )
    # Read the size from the container structure first, so a decompression bomb
    # is refused here rather than by the imaging library's own guard, which
    # reports it as an unidentifiable file instead of a pixel-limit verdict.
    dims = probe_dims(data)
    if dims is None:
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Exception as exc:
            raise ValidationError(
                FailureCode.DECODE, f"unreadable image header: {exc}"
            ) from exc
    else:
        w, h = dims.width, dims.height
    if w <= 0 or h <= 0:
        raise ValidationError(FailureCode.DECODE, f"degenerate size {w}x{h}")
    if w * h > max_pixels:
        raise ValidationError(
            FailureCode.PIXEL_LIMIT,
            f"{w}x{h} = {w * h:,} px exceeds the {max_pixels:,} px limit "
            f"(rejected from the header, before decoding)",
        )
    return w, h, res.kind


def validate_image(data: bytes, s: Settings) -> ValidatedImage:
    """Full validation and normalisation of an image upload."""
    probe_image_header(data, s.max_page_pixels)

    notes: list[str] = []
    fmt = "image"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            fmt = im.format or "image"
            # Alpha must be composited onto white explicitly. `convert("RGB")`
            # would silently discard the alpha channel and leave the underlying
            # colour values, so a fully transparent red region would reach the
            # model as saturated red instead of blank paper.
            has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
                im.info.get("transparency") is not None
            )
            if has_alpha:
                notes.append("alpha composited onto white")
                rgba = im.convert("RGBA")
                flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                flat.alpha_composite(rgba)
                im = flat.convert("RGB")
            elif im.mode != "RGB":
                notes.append(f"mode {im.mode}->RGB")
                im = im.convert("RGB")
            # EXIF orientation is applied, then dropped, so the stored pixels
            # and the reported dimensions agree. exif_transpose returns a copy
            # even when there is nothing to do, so the decision is made on the
            # tag rather than on object identity.
            if im.getexif().get(0x0112):
                notes.append("exif orientation applied")
                im = ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.asarray(im, dtype=np.uint8)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            FailureCode.DECODE, f"{fmt} decode failed: {exc}"
        ) from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValidationError(FailureCode.DECODE, f"unexpected array shape {rgb.shape}")
    if rgb.size == 0:
        raise ValidationError(FailureCode.DECODE, "decoded to zero pixels")

    return ValidatedImage(
        rgb=np.ascontiguousarray(rgb),
        width=rgb.shape[1],
        height=rgb.shape[0],
        media_type=sniff(data[:16]).kind or "image/png",
        normalized=",".join(notes) if notes else "none",
    )


# -- PDF ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PdfProbe:
    page_count: int
    encrypted: bool


def probe_pdf(data: bytes, s: Settings) -> PdfProbe:
    """Page count and encryption from the document structure only.

    Page count is needed at the API boundary to enforce the per-file and
    per-batch page limits, and getting it must not require rendering.
    """
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(io.BytesIO(data))
    except Exception as exc:
        raise ValidationError(FailureCode.DECODE, f"unreadable PDF: {exc}") from exc
    try:
        n = len(doc)
        encrypted = bool(getattr(doc, "is_encrypted", False)) or _is_encrypted(data)
    finally:
        doc.close()
    if encrypted:
        raise ValidationError(
            FailureCode.ENCRYPTED_PDF, "encrypted PDFs are not accepted"
        )
    if n == 0:
        raise ValidationError(FailureCode.ZERO_PAGE_PDF, "PDF has no pages")
    if n > s.max_pages_per_file:
        raise ValidationError(
            FailureCode.RENDER,
            f"{n} pages exceeds the per-file limit of {s.max_pages_per_file}",
        )
    return PdfProbe(page_count=n, encrypted=encrypted)


def _is_encrypted(data: bytes) -> bool:
    """Look for the /Encrypt trailer dictionary."""
    tail = data[-4096:]
    return b"/Encrypt" in tail


def render_pdf_page(data: bytes, index: int, dpi: int) -> np.ndarray:
    """Rasterize one page to RGB at `dpi`."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(data))
    try:
        page = doc[index]
        scale = dpi / 72.0
        img = page.render(scale=scale).to_pil().convert("RGB")
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    except Exception as exc:
        raise ValidationError(
            FailureCode.RENDER, f"page {index} render failed: {exc}"
        ) from exc
    finally:
        doc.close()


def sniff_result_for(data: bytes) -> SniffResult:
    return sniff(data[:16])


def image_limit_exceeded(width: int, height: int, s: Settings) -> bool:
    return width * height > s.max_page_pixels


__all__ = [
    "PdfProbe",
    "ValidatedImage",
    "ValidationError",
    "image_limit_exceeded",
    "probe_image_header",
    "probe_pdf",
    "render_pdf_page",
    "sniff_result_for",
    "validate_image",
]
