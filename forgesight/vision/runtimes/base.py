"""Runtime adapters: a single interface over PyTorch eager and ONNX Runtime.

Both expose `infer(batch, cancel) -> RawOutputs` for a whole micro-batch, so
the worker's batching pipeline never knows which runtime is underneath.

A design consequence worth stating: the batch dimension is load-bearing. The
Phase 0 prototype hard-coded batch 1, which would have made the micro-batcher
in Phase 4 impossible to build. `infer` here requires `batch.shape[0] >= 1`
and the exported ONNX graphs carry a dynamic batch axis.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np

from forgesight.vision.types import (
    Cancelled,
    CancelToken,
    ModelArtifact,
    RawOutputs,
    RuntimeProfile,
    label_map_hash,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_id2label(model_dir: Path) -> dict[int, str]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return {int(k): v for k, v in cfg["id2label"].items()}


class ArtifactCorrupt(Exception):
    pass


def verify_artifact(path: Path, expected_sha: str) -> None:
    """Refuse to load a model whose bytes do not match the lockfile.

    This is acceptance test AT-22: a corrupted cache file must fail the load,
    not silently produce wrong detections.
    """
    if not path.exists():
        raise ArtifactCorrupt(f"missing artifact: {path}")
    actual = file_sha256(path)
    if expected_sha and actual != expected_sha:
        raise ArtifactCorrupt(
            f"sha256 mismatch for {path.name}: expected {expected_sha[:12]}, got {actual[:12]}"
        )


class TorchRuntime:
    name = "torch"

    def load(self, artifact: ModelArtifact, profile: RuntimeProfile) -> TorchModel:
        return TorchModel(artifact, profile)


class TorchModel:
    def __init__(self, artifact: ModelArtifact, profile: RuntimeProfile):
        import torch
        from transformers import AutoModelForObjectDetection

        self.artifact_sha = artifact.sha256
        self._torch = torch
        torch.set_num_threads(profile.intra_op_threads)
        # set_num_interop_threads may only be called once, before any parallel
        # region has started. Ignore the RuntimeError if work is already live.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(profile.inter_op_threads)

        model_dir = Path(artifact.path)
        verify_artifact(model_dir / "model.safetensors", artifact.sha256)
        self.id2label = read_id2label(model_dir)
        self.label_map_hash = label_map_hash(self.id2label)
        self.model = AutoModelForObjectDetection.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model.eval()
        self._closed = False

    def infer(self, batch: np.ndarray, cancel: CancelToken) -> RawOutputs:
        torch = self._torch
        if batch.ndim != 4:
            raise ValueError(f"expected [b,3,H,W], got {batch.shape}")
        if cancel.is_set():
            raise Cancelled()
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(batch))
            out = self.model(pixel_values=t)
        if cancel.is_set():
            raise Cancelled()
        return RawOutputs(
            logits=out.logits.detach().cpu().numpy(),
            boxes=out.pred_boxes.detach().cpu().numpy(),
        )

    def close(self) -> None:
        self._closed = True
        del self.model


class OrtruntimeFactory:
    name = "onnxruntime"

    def load(self, artifact: ModelArtifact, profile: RuntimeProfile) -> OrtModel:
        return OrtModel(artifact, profile)


class OrtModel:
    """ONNX Runtime CPU adapter.

    `RunOptions.terminate` gives real in-flight cancellation, which torch eager
    does not have (design §8.5). A forward pass already inside a native kernel
    still has to unwind, but it stops at the next kernel boundary instead of
    running to completion.
    """

    def __init__(self, artifact: ModelArtifact, profile: RuntimeProfile):
        import onnxruntime as ort

        onnx_path = Path(artifact.path)
        verify_artifact(onnx_path, artifact.sha256)

        self.artifact_sha = artifact.sha256
        self.id2label = _id2label_for(artifact)
        self.label_map_hash = label_map_hash(self.id2label)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = profile.intra_op_threads
        opts.inter_op_num_threads = profile.inter_op_threads
        opts.graph_optimization_level = _graph_opt(profile.graph_opt_level)
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input = self.session.get_inputs()[0].name
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._closed = False

    def infer(self, batch: np.ndarray, cancel: CancelToken) -> RawOutputs:
        import onnxruntime as ort

        if batch.ndim != 4:
            raise ValueError(f"expected [b,3,H,W], got {batch.shape}")
        if cancel.is_set():
            raise Cancelled()

        run_opts = ort.RunOptions()
        cancel.on_set(lambda: setattr(run_opts, "terminate", True))
        try:
            outs = self.session.run(
                self._out_names, {self._input: batch}, run_options=run_opts
            )
        except Exception as exc:  # ORT raises on a terminated run
            if cancel.is_set():
                raise Cancelled() from exc
            raise
        if cancel.is_set():
            raise Cancelled()

        by_name = dict(zip(self._out_names, outs, strict=True))
        return RawOutputs(
            logits=np.asarray(by_name["logits"]),
            boxes=np.asarray(by_name["pred_boxes"]),
        )

    def close(self) -> None:
        self._closed = True
        self.session = None  # type: ignore[assignment]


def _id2label_for(artifact: ModelArtifact) -> dict[int, str]:
    """The label map for an exported artifact.

    Read from the export record that sits beside the ONNX file, because the ORT
    worker image intentionally does not carry the source model tree. A
    directory search would work in development and fail in the deployment it
    was written for, which is the worst combination.
    """
    record = Path(artifact.path).with_suffix(".export.json")
    if not record.exists():
        raise ArtifactCorrupt(
            f"no export record beside {Path(artifact.path).name}; the label map "
            f"is required and is not guessed"
        )
    raw = json.loads(record.read_text()).get("id2label")
    if not raw:
        raise ArtifactCorrupt(f"export record for {artifact.name} has no id2label")
    return {int(k): v for k, v in raw.items()}


def _graph_opt(level: str):
    import onnxruntime as ort

    return {
        "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[level]
