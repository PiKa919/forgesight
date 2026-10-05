"""Ingest acceptance tests: AT-8, AT-9, and the sniffing the API depends on.

AT-8 is the security-relevant one. A decompression bomb must be refused from the
header, and the test asserts that by measuring the process's own memory growth
rather than trusting the error code alone.
"""

from __future__ import annotations

import io
import os

import psutil
import pytest
from PIL import Image

from forgesight.ingest.dims import probe
from forgesight.ingest.sniff import MediaKind, sniff
from forgesight.ingest.validate import (
    ValidationError,
    probe_image_header,
    probe_pdf,
    render_pdf_page,
    validate_image,
)
from forgesight.settings import get_settings
from forgesight.synth.generator import generate_page, generate_pdf
from forgesight.vision.types import FailureCode


@pytest.fixture(scope="module")
def s():
    return get_settings()


def _png(img: Image.Image, **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG", **kw)
    return b.getvalue()


# -- sniffing --------------------------------------------------------------


def test_sniff_uses_magic_bytes_not_the_declared_type():
    """A file that claims to be a PNG but is not must not be accepted as one."""
    assert sniff(b"\x89PNG\r\n\x1a\n" + b"rest").kind is MediaKind.PNG
    assert sniff(b"%PDF-1.7\n...").kind is MediaKind.PDF
    assert sniff(b"\xff\xd8\xff\xe0").kind is MediaKind.JPEG
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ").kind is MediaKind.WEBP
    assert sniff(b"II*\x00rest").kind is MediaKind.TIFF
    assert sniff(b"MM\x00*rest").kind is MediaKind.TIFF


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not an image at all",
        b"PK\x03\x04zipfile",  # a zip, not an accepted type
        b"\x89PNG",  # truncated signature
        b"%PDF",  # truncated pdf
    ],
)
def test_sniff_rejects_non_images(payload):
    assert not sniff(payload).ok, f"{payload!r} was accepted"


# -- header probing (AT-8) -------------------------------------------------


def test_header_probe_reads_dimensions_without_decoding(s):
    page = generate_page(template="two_column", seed=5, dpi=150)
    png = _png(Image.fromarray(page["image"]))
    dims = probe(png)
    assert dims is not None
    assert (dims.width, dims.height) == page["page_size_px"]
    assert dims.source == "png-ihdr"


def test_header_probe_handles_jpeg(s):
    page = generate_page(template="two_column", seed=5, dpi=150)
    b = io.BytesIO()
    Image.fromarray(page["image"]).save(b, format="JPEG", quality=85)
    dims = probe(b.getvalue())
    assert dims is not None and dims.source.startswith("jpeg")


def test_decompression_bomb_is_refused_from_the_header(s):
    """AT-8: a 50 000 x 50 000 PNG header must be rejected before decoding."""
    tiny = bytearray(_png(Image.new("RGB", (1, 1))))
    # Overwrite IHDR width/height, keeping the PNG structurally intact.
    tiny[16:24] = (50000).to_bytes(4, "big") + (50000).to_bytes(4, "big")
    data = bytes(tiny)

    # The header alone is enough to refuse it, with no pixel buffer allocated.
    assert probe(data) is not None, "header probe could not read the declared size"

    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    with pytest.raises(ValidationError) as exc:
        validate_image(data, s)
    growth = proc.memory_info().rss - before

    assert exc.value.code is FailureCode.PIXEL_LIMIT
    assert "40000000" in str(exc.value) or "40,000,000" in str(exc.value)
    assert growth < 50 * 1024 * 1024, f"RSS grew {growth / 1e6:.1f} MB while refusing"


def test_pixel_limit_boundary(s):
    page = generate_page(seed=1, dpi=72)
    w, h = page["page_size_px"]
    img = Image.fromarray(page["image"])
    # Just under the limit: accepted.
    probe_image_header(_png(img), max_pixels=w * h)
    # One pixel over: refused.
    with pytest.raises(ValidationError) as exc:
        probe_image_header(_png(img), max_pixels=w * h - 1)
    assert exc.value.code is FailureCode.PIXEL_LIMIT


# -- malformed inputs (AT-9) -----------------------------------------------


def test_truncated_jpeg_fails_loudly(s):
    page = generate_page(seed=2, dpi=150)
    b = io.BytesIO()
    Image.fromarray(page["image"]).save(b, format="JPEG", quality=90)
    full = b.getvalue()
    with pytest.raises(ValidationError) as exc:
        validate_image(full[: len(full) // 2], s)
    assert exc.value.code is FailureCode.DECODE


def test_zero_page_pdf_is_refused(s):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    path = io.BytesIO()
    doc.save(path)
    doc.close()
    with pytest.raises(ValidationError) as exc:
        probe_pdf(path.getvalue(), s)
    assert exc.value.code in {FailureCode.ZERO_PAGE_PDF, FailureCode.DECODE}


def test_encrypted_pdf_is_refused(s):
    data = bytearray(generate_pdf("two_column", 3))
    # Plant an /Encrypt trailer reference; the structural check must catch it.
    data.extend(b"\ntrailer\n<< /Encrypt 9 0 R >>\n")
    with pytest.raises(ValidationError) as exc:
        probe_pdf(bytes(data), s)
    assert exc.value.code is FailureCode.ENCRYPTED_PDF


def test_cmyk_jpeg_is_normalized_to_rgb(s):
    cmyk = Image.new("CMYK", (64, 48), (10, 20, 30, 40))
    b = io.BytesIO()
    cmyk.save(b, format="JPEG")  # PNG cannot carry CMYK
    v = validate_image(b.getvalue(), s)
    assert v.rgb.shape == (48, 64, 3)
    assert "CMYK" in v.normalized


def test_rgba_is_flattened_onto_white(s):
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # fully transparent red
    v = validate_image(_png(rgba), s)
    assert v.rgb.shape == (32, 32, 3)
    assert "alpha" in v.normalized
    # Transparent red over white is white, not black.
    assert v.rgb[0, 0].tolist() == [255, 255, 255]


def test_exif_orientation_is_applied_and_recorded(s):
    page = generate_page(seed=4, dpi=150)
    img = Image.fromarray(page["image"])
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 270 CW
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=95, exif=exif)
    v = validate_image(b.getvalue(), s)
    assert "exif orientation" in v.normalized
    # The stored pixels are rotated, so the reported size is swapped.
    assert (v.width, v.height) == (v.rgb.shape[1], v.rgb.shape[0])
    assert v.width != img.width


def test_image_without_exif_reports_no_normalisation(s):
    """exif_transpose copies unconditionally, so absence must be detected."""
    page = generate_page(seed=4, dpi=150)
    v = validate_image(_png(Image.fromarray(page["image"])), s)
    assert v.normalized == "none"


def test_pdf_page_count_and_render(s):
    pdf = generate_pdf("two_column", 7)
    probe_ = probe_pdf(pdf, s)
    assert probe_.page_count == 1
    img = render_pdf_page(pdf, 0, 150)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > img.shape[1], "letter page should be portrait"


def test_pdf_page_limit(s):
    pdf = generate_pdf("two_column", 8)
    tight = get_settings().model_copy(update={"max_pages_per_file": 0})
    with pytest.raises(ValidationError) as exc:
        probe_pdf(pdf, tight)
    assert exc.value.code is FailureCode.RENDER
