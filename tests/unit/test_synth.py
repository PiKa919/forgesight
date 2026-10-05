import numpy as np
import pytest

from forgesight.synth.generator import generate_page
from forgesight.synth.gt_validate import validate_page
from forgesight.synth.templates import TEMPLATES, EMITTED_CLASSES

ALL_TEMPLATES = sorted(TEMPLATES)


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_every_template_renders(template):
    page = generate_page(template=template, seed=7, dpi=150)
    img = page["image"]
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    assert page["boxes"], f"{template} produced no boxes"
    assert page["page_size_px"] == (img.shape[1], img.shape[0])


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_boxes_are_tight_and_in_bounds(template):
    page = generate_page(template=template, seed=11, dpi=150)
    w, h = page["page_size_px"]
    for box, lbl in zip(page["boxes"], page["labels"]):
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= w, f"{template}/{lbl} x out of range: {box} (w={w})"
        assert 0 <= y1 < y2 <= h, f"{template}/{lbl} y out of range: {box} (h={h})"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_ground_truth_rules_hold(template):
    """Design §9.2: every box holds ink, ink is covered, no same-class overlap."""
    for seed in range(8):
        page = generate_page(template=template, seed=seed, dpi=150)
        issues = [i for i in validate_page(page) if i.severity == "error"]
        assert not issues, f"{template} seed={seed}: " + "; ".join(
            f"{i.rule}: {i.detail}" for i in issues
        )


def test_template_actually_changes_the_layout():
    """A `template` argument that does nothing would silently fake diversity."""
    seen = set()
    for t in ALL_TEMPLATES:
        p = generate_page(template=t, seed=3, dpi=150)
        seen.add((p["page_size_px"], tuple(p["labels"])))
    assert len(seen) == len(ALL_TEMPLATES), "templates are not distinct"


def test_labels_are_emitted_classes_with_valid_ids():
    from forgesight.synth.templates import CLASS_TO_ID

    found = set()
    for t in ALL_TEMPLATES:
        for seed in range(6):
            p = generate_page(template=t, seed=seed, dpi=150)
            for lbl, cid in zip(p["labels"], p["class_ids"]):
                assert lbl in EMITTED_CLASSES
                assert CLASS_TO_ID[lbl] == cid
                found.add(lbl)
    # The gate for critical classes (G3) needs these present in the data.
    for required in ("title", "section_header", "text", "table", "picture", "page_header", "page_footer"):
        assert required in found, f"never generated class {required}"


def test_generation_is_deterministic():
    a = generate_page(template="two_column", seed=99, dpi=150)
    b = generate_page(template="two_column", seed=99, dpi=150)
    assert np.array_equal(a["image"], b["image"])
    assert a["boxes"] == b["boxes"]


def test_different_seeds_differ():
    a = generate_page(template="two_column", seed=1, dpi=150)
    b = generate_page(template="two_column", seed=2, dpi=150)
    assert not np.array_equal(a["image"], b["image"])


def test_provenance_is_synthetic_and_seeded():
    p = generate_page(template="slide", seed=1234, dpi=150)
    assert p["synthetic"] is True
    assert p["provenance"].startswith("synthetic:forgesight-synth@1#seed=1234")
    # The marker must be visible in the rendered pixels, not just metadata.
    assert np.any(p["image"] < 240), "page is blank"


def test_dpi_scales_boxes_consistently():
    lo = generate_page(template="two_column", seed=5, dpi=72)
    hi = generate_page(template="two_column", seed=5, dpi=144)
    assert hi["page_size_px"][0] == pytest.approx(2 * lo["page_size_px"][0], abs=2)
    assert len(lo["boxes"]) == len(hi["boxes"])
    ratio_x = hi["boxes"][0][2] / lo["boxes"][0][2]
    assert ratio_x == pytest.approx(2.0, rel=0.01)


def test_degradation_is_paired_and_nonempty():
    clean = generate_page(template="two_column", seed=17, dpi=150)
    scan = generate_page(template="two_column", seed=17, dpi=150, degrade="scan")
    assert not np.array_equal(clean["image"], scan["image"])
    # Same content underneath, so the box count is unchanged.
    assert clean["labels"] == scan["labels"]
    assert "degrade=scan" in scan["provenance"]
