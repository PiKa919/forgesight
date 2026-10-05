"""Header-only dimension probing.

The pixel limit has to be enforced *before* any pixel buffer is allocated, and
it cannot rely on the imaging library to get there first: Pillow's own guard
rejects a malformed header by refusing to identify the file, which surfaces as
an unhelpful `cannot identify image file` rather than a pixel-limit verdict.

So the dimensions are read here, from the container structure alone, with a
minimal parser per format. A 50 000 x 50 000 PNG costs a few dozen bytes of
parsing to refuse.

Parsers return None when they cannot decide, and the caller then falls back to
the imaging library -- this is a fast pre-filter, not a replacement for it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dimensions:
    width: int
    height: int
    source: str  # which parser produced this


def png_dimensions(data: bytes) -> Dimensions | None:
    # 8-byte signature, then an IHDR chunk whose payload starts with w/h.
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return Dimensions(w, h, "png-ihdr")


def jpeg_dimensions(data: bytes) -> Dimensions | None:
    """Walk JPEG markers to the first SOFn frame header."""
    if not data.startswith(b"\xff\xd8"):
        return None
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # Standalone markers carry no length.
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9 or marker == 0xDA:  # EOI or start of scan
            return None
        if i + 4 > n:
            return None
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        # SOF0..SOF15, excluding the non-frame markers DHT (C4), JPG (C8), DAC (CC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return Dimensions(w, h, "jpeg-sofn")
        i += 2 + seg_len
    return None


def gif_dimensions(data: bytes) -> Dimensions | None:
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    w, h = struct.unpack("<HH", data[6:10])
    return Dimensions(w, h, "gif-lsd")


def tiff_dimensions(data: bytes) -> Dimensions | None:
    if data[:4] == b"II*\x00":
        endian = "<"
    elif data[:4] == b"MM\x00*":
        endian = ">"
    else:
        return None
    if len(data) < 8:
        return None
    (offset,) = struct.unpack(endian + "I", data[4:8])
    if offset + 2 > len(data):
        return None
    (count,) = struct.unpack(endian + "H", data[offset : offset + 2])
    # 256 = ImageWidth, 257 = ImageLength
    for tag in (256, 257):
        for i in range(count):
            entry = offset + 2 + i * 12
            if entry + 12 > len(data):
                return None
            (t,) = struct.unpack(endian + "H", data[entry : entry + 2])
            if t == tag:
                kind = data[entry + 2]
                if kind == 3:  # SHORT
                    (v,) = struct.unpack(endian + "H", data[entry + 8 : entry + 10])
                elif kind == 4:  # LONG
                    (v,) = struct.unpack(endian + "I", data[entry + 8 : entry + 12])
                else:
                    continue
                if tag == 256:
                    width = v
                else:
                    height = v
                    return Dimensions(width, height, "tiff-ifd")
    return None


def webp_dimensions(data: bytes) -> Dimensions | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return Dimensions(w, h, "webp-vp8x")
    if chunk == b"VP8 " and len(data) >= 30:
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return Dimensions(w, h, "webp-vp8")
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return Dimensions((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, "webp-vp8l")
    return None


_PARSERS = (
    png_dimensions,
    jpeg_dimensions,
    webp_dimensions,
    tiff_dimensions,
    gif_dimensions,
)


def probe(data: bytes) -> Dimensions | None:
    """Dimensions from container structure, or None if undecidable."""
    for parser in _PARSERS:
        try:
            dims = parser(data)
        except (struct.error, IndexError, ValueError):
            continue
        if dims is not None and dims.width > 0 and dims.height > 0:
            return dims
    return None
