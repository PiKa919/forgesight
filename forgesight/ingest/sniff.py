"""Media-type sniffing by magic bytes (design §8.1).

The client's `Content-Type` header is ignored. A caller that claims
`image/png` while sending a ZIP, or a Python pickle, gets whatever the bytes
actually are -- and bytes that match nothing accepted are rejected.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import StrEnum


class MediaKind(StrEnum):
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    TIFF = "image/tiff"
    PDF = "application/pdf"


@dataclass(frozen=True, slots=True)
class SniffResult:
    kind: MediaKind | None
    reason: str

    @property
    def ok(self) -> bool:
        return self.kind is not None


# Longest-first: several formats share a prefix, and the more specific magic
# must be tested first or a truncated match misclassifies.
_MAGIC: tuple[tuple[bytes, MediaKind], ...] = (
    (b"\x89PNG\r\n\x1a\n", MediaKind.PNG),
    (b"%PDF-", MediaKind.PDF),
    (b"\xff\xd8\xff", MediaKind.JPEG),
    (b"II*\x00", MediaKind.TIFF),  # little-endian TIFF
    (b"MM\x00*", MediaKind.TIFF),  # big-endian TIFF
)


def sniff(head: bytes) -> SniffResult:
    """Classify from the first bytes of a stream."""
    if not head:
        return SniffResult(None, "empty upload")
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return SniffResult(kind, f"matched {kind} magic")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return SniffResult(MediaKind.WEBP, "matched RIFF/WEBP")
    return SniffResult(
        None,
        f"unrecognised magic {head[:8]!r}; accepted: png, jpeg, webp, tiff, pdf",
    )


def sniff_stream(stream, n: int = 16) -> SniffResult:
    head = stream.read(n)
    # A non-seekable stream means the caller already buffered the content.
    with contextlib.suppress(AttributeError, OSError):
        stream.seek(0)
    return sniff(head)
