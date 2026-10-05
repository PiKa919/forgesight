"""AT-15: the shared NumPy postprocessor must equal HuggingFace's.

This is the load-bearing test of the whole runtime design. The ORT worker ships
without torch precisely because one NumPy postprocessor serves both pools, so
if that implementation drifts from
`transformers.RTDetrImageProcessor.post_process_object_detection`, every
comparison in the report is comparing two different things.

The check runs on real model outputs, not hand-written fixtures, so it would
catch a change in the models' output convention as well as a bug in our code.
"""

from __future__ import annotations

import numpy as np
import pytest

from forgesight.vision.postprocess import postprocess_batch
from forgesight.vision.preprocess import Preprocessor, make_profile
from forgesight.vision.types import RawOutputs

from tests.support import MODEL_NAMES, MODELS_DIR, requires_weights

TOL = 1e-4


@requires_weights
@pytest.mark.model
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_numpy_postprocess_matches_transformers(model_name, id2label):
    import torch
    from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

    from forgesight.synth.generator import generate_page

    model_dir = MODELS_DIR / model_name
    processor = RTDetrImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir)).eval()

    page = generate_page(template="two_column", seed=42, dpi=150)
    image = page["image"]

    ours = Preprocessor(make_profile("pil_bilinear", id2label=id2label))
    items = ours(image)
    assert len(items) == 1
    item = items[0]

    with torch.no_grad():
        out = model(pixel_values=torch.from_numpy(item.tensor).unsqueeze(0))
    raw = RawOutputs(logits=out.logits.numpy(), boxes=out.pred_boxes.numpy())

    got = postprocess_batch(raw, items, id2label, threshold=0.30, top_k=300)[0]

    h, w = image.shape[:2]
    expected = processor.post_process_object_detection(
        out, threshold=0.30, target_sizes=torch.tensor([[h, w]])
    )[0]
    # `post_process_object_detection` returns a dict of tensors, not a list.
    exp_scores = expected["scores"].tolist()
    exp_labels = expected["labels"].tolist()
    exp_boxes = expected["boxes"].tolist()

    assert len(got) == len(exp_scores), (
        f"detection count differs: ours={len(got)} hf={len(exp_scores)}"
    )

    max_score_diff = 0.0
    max_box_diff = 0.0
    for det, score, label, box in zip(got, exp_scores, exp_labels, exp_boxes):
        assert det.class_id == int(label), f"class {det.class_id} vs {int(label)}"
        max_score_diff = max(max_score_diff, abs(det.score - float(score)))
        for ours_c, theirs_c in zip(det.box, box):
            max_box_diff = max(max_box_diff, abs(ours_c - float(theirs_c)))

    assert max_score_diff < TOL, f"score drift {max_score_diff}"
    assert max_box_diff < TOL, f"box drift {max_box_diff}"


@requires_weights
@pytest.mark.model
def test_postprocess_detects_something_on_a_real_page(id2label):
    """A silent all-empty postprocessor would pass an equality test vacuously."""
    import torch
    from transformers import AutoModelForObjectDetection

    from forgesight.synth.generator import generate_page

    model = AutoModelForObjectDetection.from_pretrained(str(MODELS_DIR / "heron")).eval()
    page = generate_page(template="two_column", seed=42, dpi=150)
    items = Preprocessor(make_profile("pil_bilinear", id2label=id2label))(page["image"])
    with torch.no_grad():
        out = model(pixel_values=torch.from_numpy(items[0].tensor).unsqueeze(0))
    raw = RawOutputs(logits=out.logits.numpy(), boxes=out.pred_boxes.numpy())
    dets = postprocess_batch(raw, items, id2label, threshold=0.30)[0]
    assert len(dets) > 5, f"expected real detections, got {len(dets)}"


def test_postprocess_respects_threshold_and_top_k(id2label):
    """Pure-logic check, so it needs no weights."""
    q, c = 40, len(id2label)
    logits = np.full((1, q, c), -8.0)
    logits[0, :, 9] = 6.0  # every query confidently says "text"
    boxes = np.tile(np.array([[0.5, 0.5, 0.2, 0.2]]), (1, q, 1))
    raw = RawOutputs(logits=logits, boxes=boxes)
    item = _fake_item()

    assert len(postprocess_batch(raw, [item], id2label, threshold=0.30)[0]) == q
    assert len(postprocess_batch(raw, [item], id2label, threshold=0.30, top_k=7)[0]) == 7
    assert postprocess_batch(raw, [item], id2label, threshold=0.999)[0] == []


def test_postprocess_can_return_two_classes_from_one_query(id2label):
    """HF selects over the flattened (query x class) matrix, not a per-query argmax.

    Getting this wrong silently drops detections and invalidates every mAP
    number, so it is pinned here rather than left to the HF parity test.
    """
    q, c = 10, len(id2label)
    logits = np.full((1, q, c), -9.0)
    logits[0, 0, 9] = 5.0  # text
    logits[0, 0, 8] = 4.0  # table, same query
    raw = RawOutputs(logits=logits, boxes=np.tile([[[0.5, 0.5, 0.2, 0.2]]], (1, q, 1)))
    dets = postprocess_batch(raw, [_fake_item()], id2label, threshold=0.5)[0]
    assert {d.class_id for d in dets} == {8, 9}
    assert [d.score for d in dets] == sorted([d.score for d in dets], reverse=True)


def test_postprocess_rejects_batch_mismatch(id2label):
    """A runtime that ignores the batch axis must fail loudly, not return junk."""
    raw = RawOutputs(
        logits=np.zeros((2, 5, len(id2label))),
        boxes=np.zeros((2, 5, 4)),
    )
    with pytest.raises(ValueError, match="batch"):
        postprocess_batch(raw, [_fake_item()], id2label, threshold=0.3)


def test_postprocess_maps_boxes_back_to_page_pixels(id2label):
    """A full-width box at the page centre must land at the page centre."""
    raw = RawOutputs(
        logits=np.full((1, 1, len(id2label)), -8.0),
        boxes=np.array([[[0.5, 0.5, 1.0, 1.0]]]),  # cx=cy=0.5, w=h=1 -> whole page
    )
    raw.logits[0, 0, 10] = 8.0
    item = _fake_item(page=(1000, 500))
    dets = postprocess_batch(raw, [item], id2label, threshold=0.3)[0]
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].box
    assert x1 == pytest.approx(0.0, abs=1e-6)
    assert y1 == pytest.approx(0.0, abs=1e-6)
    assert x2 == pytest.approx(1000.0, abs=1e-6)
    assert y2 == pytest.approx(500.0, abs=1e-6)


def test_postprocess_offsets_tile_origins(id2label):
    """Tiled detections must be shifted by the tile origin, not dropped."""
    raw = RawOutputs(
        logits=np.full((1, 1, len(id2label)), -8.0),
        boxes=np.array([[[0.5, 0.5, 0.0, 0.0]]]),
    )
    raw.logits[0, 0, 9] = 8.0
    item = _fake_item(page=(800, 600), origin=(400, 0))
    dets = postprocess_batch(raw, [item], id2label, threshold=0.3)[0]
    # cx=cy=0.5, zero extent, page 800x600, tile origin (400, 0).
    # x = 0.5 * 800 + 400 = 800, y = 0.5 * 600 + 0 = 300
    x1, y1, x2, y2 = dets[0].box
    assert (x1, x2) == pytest.approx((800.0, 800.0))
    assert (y1, y2) == pytest.approx((300.0, 300.0))


def _fake_item(page=(1000, 500), origin=(0, 0)):
    from forgesight.vision.types import PreparedItem

    w, h = page
    return PreparedItem(
        tensor=np.zeros((3, 640, 640), dtype=np.float32),
        page_size=(w, h),
        scale=(float(w), float(h)),
        tile_origin=origin,
        preprocess_hash="test",
    )
