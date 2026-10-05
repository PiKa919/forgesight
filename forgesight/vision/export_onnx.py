"""ONNX export with recorded provenance (design §6, §8.9).

The exported file is a *new artifact* with its own sha256, and it records the
sha of the weights it came from, the exporter used, the opset, and the library
versions. Without that chain, "the ONNX model" is not a reproducible thing to
benchmark: a report could name an artifact that no longer exists.

Exporter choice is not a preference. Phase 0 measured that the dynamo exporter
fails on `RTDetrV2ForObjectDetection` (`No ONNX function found for
aten._is_all_true`), so the TorchScript exporter is the only path available for
these architectures, and that is recorded rather than left implicit.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_OPSET = 17
EXPORTER = "torch.onnx.export(dynamo=False)"


@dataclass(frozen=True, slots=True)
class ExportRecord:
    name: str
    file: str
    sha256: str
    byte_size: int
    opset: int
    exporter: str
    parent_sha256: str
    parent_repo: str
    parent_revision: str
    torch_version: str
    transformers_version: str
    onnx_version: str
    dynamic_batch: bool
    input_shape: tuple[int, int, int, int]
    output_names: tuple[str, ...]
    # The ORT worker image ships without the source model tree, so the label map
    # has to travel with the artifact rather than being discovered on disk.
    id2label: dict[int, str]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["output_names"] = list(self.output_names)
        d["id2label"] = {str(k): v for k, v in self.id2label.items()}
        return d


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(
    model_dir: Path,
    out_path: Path,
    name: str,
    parent_sha256: str,
    parent_repo: str,
    parent_revision: str,
    opset: int = DEFAULT_OPSET,
) -> ExportRecord:
    import json as _json

    import onnx
    import torch
    import transformers
    from transformers import AutoModelForObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(str(model_dir)).eval()
    id2label = {
        int(k): v for k, v in _json.loads((model_dir / "config.json").read_text())["id2label"].items()
    }
    dummy = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _run(output_names: list[str]) -> None:
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=output_names,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # RT-DETRv2 and D-FINE return several intermediate tensors alongside the
    # two the worker needs. Passing a two-element output_names list does not
    # truncate the graph: the extra outputs stay, named after internal nodes.
    # So the first pass discovers how many outputs the graph really has, and the
    # second names all of them, keeping the first two as logits and pred_boxes.
    _run(["logits", "pred_boxes"])
    probe = onnx.load(str(out_path)).graph
    n_out = len(probe.output)
    if n_out < 2:
        raise AssertionError(f"{name}: expected at least 2 graph outputs, got {n_out}")
    names = ["logits", "pred_boxes"] + [f"aux_{i}" for i in range(n_out - 2)]
    _run(names)

    # Read the graph back rather than trusting the export call: the dynamic
    # batch axis and the output names are what the worker depends on, so they
    # are asserted on the artifact that will actually be loaded.
    graph = onnx.load(str(out_path)).graph
    dynamic = "batch_size" in {
        d.dim_param for inp in graph.input for d in inp.type.tensor_type.shape.dim
    }
    out_names = tuple(o.name for o in graph.output)
    if not dynamic:
        raise AssertionError(f"{name}: exported graph has no dynamic batch axis")
    if out_names[:2] != ("logits", "pred_boxes"):
        raise AssertionError(f"{name}: unexpected output names {out_names}")

    record = ExportRecord(
        name=name,
        file=out_path.name,
        sha256=file_sha256(out_path),
        byte_size=out_path.stat().st_size,
        opset=opset,
        exporter=EXPORTER,
        parent_sha256=parent_sha256,
        parent_repo=parent_repo,
        parent_revision=parent_revision,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        onnx_version=onnx.__version__,
        dynamic_batch=True,
        input_shape=(1, 3, 640, 640),
        output_names=out_names,
        id2label=id2label,
    )
    record_path = out_path.with_suffix(".export.json")
    record_path.write_text(json.dumps(record.as_dict(), indent=2) + "\n")
    return record


def quantize_int8(src: Path, out_path: Path, name: str) -> dict:
    """Dynamic int8 quantization of MatMul/Gemm weights.

    Reported, never promoted on speed: int8 reaches users only if it passes the
    same quality gates as everything else (design §8.9).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    src_rec = json.loads(src.with_suffix(".export.json").read_text())
    return {
        "name": name,
        "file": out_path.name,
        "sha256": file_sha256(out_path),
        "byte_size": out_path.stat().st_size,
        "format": "onnx-int8",
        "parent_sha256": file_sha256(src),
        "method": "onnxruntime.quantization.quantize_dynamic(QInt8, MatMulConstBOnly)",
        "platform": platform.platform(),
        "id2label": src_rec["id2label"],
    }


def write_int8_record(out_path: Path, payload: dict) -> Path:
    rec = out_path.with_suffix(".export.json")
    rec.write_text(json.dumps(payload, indent=2) + "\n")
    return rec
