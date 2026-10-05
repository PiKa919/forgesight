"""Export both models to ONNX and record each artifact's provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from forgesight.settings import get_settings
from forgesight.vision.export_onnx import export, quantize_int8, write_int8_record

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(with_int8: bool = False) -> int:
    s = get_settings()
    lock = json.loads(
        (Path(__file__).resolve().parents[1] / "models.lock.json").read_text()
    )
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, entry in lock["models"].items():
        model_dir = s.models_dir / name
        if not (model_dir / "model.safetensors").exists():
            print(f"skip {name}: weights absent, run scripts/fetch_models.py")
            continue
        out = s.artifacts_dir / f"{name}.onnx"
        rec = export(
            model_dir,
            out,
            name,
            parent_sha256=entry["files"]["model.safetensors"]["sha256"],
            parent_repo=entry["repo_id"],
            parent_revision=entry["revision"],
        )
        print(
            f"{name}: {rec.file} {rec.byte_size / 1e6:.1f} MB "
            f"sha={rec.sha256[:12]} opset={rec.opset} dynamic_batch={rec.dynamic_batch}"
        )
        if with_int8:
            q_out = s.artifacts_dir / f"{name}.int8.onnx"
            payload = quantize_int8(out, q_out, name)
            write_int8_record(q_out, payload)
            print(
                f"{name}: int8 {payload['byte_size'] / 1e6:.1f} MB "
                f"({payload['byte_size'] / rec.byte_size:.2f}x of fp32)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(with_int8="--int8" in sys.argv))
