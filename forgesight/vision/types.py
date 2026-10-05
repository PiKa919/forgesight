"""Core vision value types (design §11.1).

These are the types the design fixes as the boundary between the API, the
ledger, the workers and the evaluation code. They are frozen dataclasses or
enums so that an illegal state cannot be constructed.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

LabelName = str
ResizeMethod = Literal["pil_bilinear", "cv2_linear", "cv2_area", "pil_reduce"]
RuntimeName = Literal["torch", "onnxruntime"]
ArtifactFormat = Literal["safetensors", "onnx-fp32", "onnx-int8"]


class Pool(enum.StrEnum):
    TORCH = "torch"
    ORT = "onnxruntime"


class WorkItemState(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureCode(enum.StrEnum):
    """Stable, machine-readable failure reasons. Every failed item carries one."""

    DECODE = "decode_failed"
    PIXEL_LIMIT = "pixel_limit"
    UNSUPPORTED_TYPE = "unsupported_type"
    ENCRYPTED_PDF = "encrypted_pdf"
    ZERO_PAGE_PDF = "zero_page_pdf"
    RENDER = "render_failed"
    EXCEEDS_MEMORY_BUDGET = "exceeds_memory_budget"
    ARTIFACT_CORRUPT = "artifact_corrupt"
    LOAD_FAILED = "load_failed"
    INFERENCE = "inference_failed"
    CANCELLED = "cancelled"
    MAX_ATTEMPTS = "max_attempts"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    LEASE_LOST = "lease_lost"


class Cancelled(Exception):
    """Raised by a runtime when a cancel token fires mid-inference."""


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    label: LabelName
    score: float
    box: tuple[float, float, float, float]  # xyxy in page pixels

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "score": self.score,
            "box": list(self.box),
        }


@dataclass(frozen=True, slots=True)
class PreparedItem:
    """One 640x640 model input plus everything needed to map boxes back."""

    tensor: np.ndarray  # float32 [3, 640, 640]
    page_size: tuple[int, int]  # (w, h) of the *page* in pixels
    # Normalized model coords -> region pixels. A square 640x640 resize of a
    # non-square region gives different x and y factors, so this is a pair.
    scale: tuple[float, float]
    tile_origin: tuple[int, int] | None  # top-left of this tile in page px
    preprocess_hash: str

    @property
    def nbytes(self) -> int:
        return self.tensor.nbytes


@dataclass(frozen=True, slots=True)
class RawOutputs:
    logits: np.ndarray  # [b, Q, C]
    boxes: np.ndarray  # [b, Q, 4] cxcywh normalized to the model input


@dataclass(frozen=True, slots=True)
class PreprocessProfile:
    resize: ResizeMethod
    target: int
    render_dpi: int
    tile_long_side_px: int
    tile_aspect: float
    tile_overlap: float
    label_map_hash: str
    processor_class: str

    @property
    def hash(self) -> str:
        return canonical_hash(self.as_dict())

    def as_dict(self) -> dict:
        return {
            "resize": self.resize,
            "target": self.target,
            "render_dpi": self.render_dpi,
            "tile_long_side_px": self.tile_long_side_px,
            "tile_aspect": self.tile_aspect,
            "tile_overlap": self.tile_overlap,
            "label_map_hash": self.label_map_hash,
            "processor_class": self.processor_class,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessProfile":
        return cls(**d)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    runtime: RuntimeName
    intra_op_threads: int
    inter_op_threads: int
    graph_opt_level: str
    max_batch: int
    batch_window_ms: int

    @property
    def hash(self) -> str:
        return canonical_hash(self.as_dict())

    def as_dict(self) -> dict:
        return {
            "runtime": self.runtime,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "graph_opt_level": self.graph_opt_level,
            "max_batch": self.max_batch,
            "batch_window_ms": self.batch_window_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeProfile":
        return cls(**d)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    path: str
    sha256: str
    format: ArtifactFormat
    repo_id: str
    revision: str
    license: str
    parent_sha: str | None = None
    exporter: str | None = None
    opset: int | None = None
    parent_torch: str | None = None
    parent_ort: str | None = None

    @property
    def pool(self) -> Pool:
        return Pool.ORT if self.format.startswith("onnx") else Pool.TORCH


@dataclass(frozen=True, slots=True)
class Candidate:
    """A fully-resolved, hashable inference configuration (design §6)."""

    name: str
    artifact: ModelArtifact
    preprocess: PreprocessProfile
    runtime: RuntimeProfile
    score_threshold: float
    label_map_hash: str
    tile_enabled: bool = False

    @property
    def hash(self) -> str:
        return canonical_hash(
            {
                "artifact_sha": self.artifact.sha256,
                "preprocess": self.preprocess.as_dict(),
                "runtime": self.runtime.as_dict(),
                "score_threshold": self.score_threshold,
                "label_map_hash": self.label_map_hash,
                "tile_enabled": self.tile_enabled,
            }
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "hash": self.hash,
            "artifact": {
                "name": self.artifact.name,
                "path": self.artifact.path,
                "sha256": self.artifact.sha256,
                "format": self.artifact.format,
                "repo_id": self.artifact.repo_id,
                "revision": self.artifact.revision,
                "license": self.artifact.license,
            },
            "preprocess": self.preprocess.as_dict(),
            "runtime": self.runtime.as_dict(),
            "score_threshold": self.score_threshold,
            "tile_enabled": self.tile_enabled,
        }


def canonical_hash(obj) -> str:
    """sha256 over canonical JSON. Used for every content/provenance hash."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def label_map_hash(id2label: dict[int, str]) -> str:
    return canonical_hash({str(k): v for k, v in sorted(id2label.items())})


# ---- protocols (design §11.1) ------------------------------------------


@runtime_checkable
class CancelToken(Protocol):
    def is_set(self) -> bool: ...

    def on_set(self, cb) -> None: ...


@runtime_checkable
class LoadedModel(Protocol):
    artifact_sha: str

    def infer(self, batch: np.ndarray, cancel: CancelToken) -> RawOutputs: ...

    def close(self) -> None: ...


@runtime_checkable
class InferenceRuntime(Protocol):
    name: RuntimeName

    def load(self, artifact: ModelArtifact, profile: RuntimeProfile) -> LoadedModel: ...


@runtime_checkable
class Preprocessor(Protocol):
    profile: PreprocessProfile

    def __call__(self, page_rgb: np.ndarray) -> list[PreparedItem]: ...
