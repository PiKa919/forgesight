"""Object store (design §11.1, §8.1).

The store is content-addressed: the key is derived from the sha256 of the bytes,
so a retried upload of identical content costs one object, and two workspaces
uploading the same page share the bytes without either being able to see the
other's metadata.

Write order matters. The object is written *first* and the database transaction
second, so a crash can leave an unreferenced object but never a row pointing at
bytes that do not exist. The reaper sweeps the former; the latter cannot happen.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from forgesight.settings import Settings


@dataclass(frozen=True, slots=True)
class PutResult:
    sha256: str
    size: int
    key: str


class ObjectStore(Protocol):
    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> PutResult: ...

    def get_bytes(self, key: str) -> bytes: ...

    def presign_get(self, key: str, ttl_s: int = 300) -> str: ...

    def delete(self, key: str) -> None: ...

    def iter_objects(self) -> list[tuple[str, float]]: ...


class SizeExceeded(Exception):
    pass


def asset_key(sha: str) -> str:
    return f"assets/{sha}"


def page_key(sha: str) -> str:
    return f"pages/{sha}.png"


def _safe_key(key: str) -> str:
    """Reject anything that could escape the store root.

    Keys are derived from hex digests, so a key with a separator or a dot-dot in
    it means a bug or an attack, never a legitimate path.
    """
    if not key or key.startswith("/") or ".." in key or "\\" in key:
        raise ValueError(f"unsafe object key: {key!r}")
    for part in key.split("/"):
        if not part or part in {".", ".."}:
            raise ValueError(f"unsafe object key: {key!r}")
    return key


class FsObjectStore:
    """Local filesystem store. The default for development and tests."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.root / _safe_key(key)
        # Belt and braces: the resolved path must stay under the root.
        if not p.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"unsafe object key: {key!r}")
        return p

    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> PutResult:
        """Write a stream, aborting and cleaning up if it exceeds `max_bytes`.

        The limit is enforced *while* streaming rather than by trusting a
        declared length, because a client can lie about size and a decompressed
        payload is the whole point of the pixel limits in §8.1.
        """
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        tmp = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
        try:
            with open(tmp, "wb") as fh:
                while chunk := stream.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise SizeExceeded(
                            f"object exceeds {max_bytes} bytes (read {size})"
                        )
                    digest.update(chunk)
                    fh.write(chunk)
            if size == 0:
                raise SizeExceeded("refusing to store an empty object")
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return PutResult(sha256=digest.hexdigest(), size=size, key=key)

    def put_bytes(self, key: str, data: bytes, max_bytes: int) -> PutResult:
        return self.put_stream(key, io.BytesIO(data), max_bytes)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_stream(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def presign_get(self, key: str, ttl_s: int = 300) -> str:
        """Local store serves bytes through the API, not a signed URL.

        Returning a path that the auth layer still guards keeps the production
        contract (a short-lived URL the browser can fetch) without pretending a
        local filesystem has signatures.
        """
        return f"file://{self._path(key)}"

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def iter_objects(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for path in self.root.rglob("*"):
            if path.is_file():
                out.append((str(path.relative_to(self.root)), path.stat().st_mtime))
        return out

    def copy_dir(self, src: Path) -> None:
        shutil.copytree(src, self.root, dirs_exist_ok=True)


class S3ObjectStore:
    """S3-compatible store. SeaweedFS gateway in the deployment compose."""

    def __init__(self, settings: Settings):
        import boto3

        self.s = settings
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        self.bucket = settings.s3_bucket

    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> PutResult:
        digest = hashlib.sha256()
        size = 0
        buf = bytearray()
        while chunk := stream.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                raise SizeExceeded(f"object exceeds {max_bytes} bytes (read {size})")
            digest.update(chunk)
            buf.extend(chunk)
        if size == 0:
            raise SizeExceeded("refusing to store an empty object")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=bytes(buf))
        return PutResult(sha256=digest.hexdigest(), size=size, key=key)

    def put_bytes(self, key: str, data: bytes, max_bytes: int) -> PutResult:
        return self.put_stream(key, io.BytesIO(data), max_bytes)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def presign_get(self, key: str, ttl_s: int = 300) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl_s,
        )

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def iter_objects(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        token: str | None = None
        while True:
            kwargs = {"Bucket": self.bucket, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                out.append((obj["Key"], obj["LastModified"].timestamp()))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return out


def build_store(settings: Settings) -> ObjectStore:
    if settings.object_store == "s3":
        return S3ObjectStore(settings)
    return FsObjectStore(settings.objects_dir)


__all__ = [
    "FsObjectStore",
    "ObjectStore",
    "PutResult",
    "S3ObjectStore",
    "SizeExceeded",
    "asset_key",
    "build_store",
    "page_key",
]
