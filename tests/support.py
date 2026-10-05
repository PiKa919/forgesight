"""Paths and skip markers shared by the test suite.

Model-dependent tests are opt-in on the presence of real weights, so a fresh
clone runs the full logic suite in seconds and reports honestly which model
tests were skipped and why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_NAMES = ("heron", "egret-medium")


def have_weights(name: str) -> bool:
    return (MODELS_DIR / name / "model.safetensors").exists()


def have_onnx(name: str) -> bool:
    return (ARTIFACTS_DIR / f"{name}.onnx").exists()


requires_weights = pytest.mark.skipif(
    not all(have_weights(n) for n in MODEL_NAMES),
    reason="model weights absent; run `uv run python scripts/fetch_models.py`",
)
requires_onnx = pytest.mark.skipif(
    not all(have_onnx(n) for n in MODEL_NAMES),
    reason="ONNX artifacts absent; run `uv run python scripts/export_models.py`",
)
requires_heron = pytest.mark.skipif(
    not have_weights("heron"),
    reason="heron weights absent; run scripts/fetch_models.py",
)
