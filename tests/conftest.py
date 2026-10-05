"""Shared fixtures.

The model-dependent tests are separated from the pure-logic ones on purpose:
the design's benchmark protocol refuses to mix them, and a fast unit run is
what makes a tight edit/verify loop possible at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_home"))

from forgesight.synth.generator import generate_page  # noqa: E402
from forgesight.synth.templates import TEMPLATES  # noqa: E402
from tests.support import (  # noqa: E402
    MODELS_DIR,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def page():
    """One representative page, generated once for the whole session."""
    return generate_page(template="two_column", seed=42, dpi=150)


@pytest.fixture(scope="session")
def page_variants():
    """One page per template, for tests that must not assume a single layout."""
    return {t: generate_page(template=t, seed=8, dpi=150) for t in sorted(TEMPLATES)}


@pytest.fixture(scope="session")
def id2label():
    from forgesight.vision.runtimes.base import read_id2label

    return read_id2label(MODELS_DIR / "heron")
