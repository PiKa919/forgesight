"""Evaluation tests: AT-13, AT-18, AT-19, plus the metric behaviour they guard.

AT-19 is the one that matters most. A metric implementation nobody has checked
against a hand-computed answer is exactly the component that quietly inflates a
report, so the arithmetic is pinned against numbers worked out by hand here
rather than against a reference implementation's own output.
"""

from __future__ import annotations

import pytest

from forgesight.eval.coco_eval import (
    PagePrediction,
    PageTruth,
    best_f1_threshold,
    detection_agreement,
    evaluate,
    precision_recall,
)
from forgesight.vision.postprocess import iou
from forgesight.vision.types import Detection


def _d(cid: int, score: float, box: tuple[float, float, float, float],
       label: str | None = None) -> Detection:
    return Detection(class_id=cid, label=label or f"c{cid}", score=score, box=box)


# -- AT-19: the arithmetic is pinned to hand-computed values ----------------


def test_map_of_a_perfect_single_detection_is_one():
    truth = PageTruth("p1", 100, 100, boxes=[[0, 0, 50, 50]], class_names=["c0"])
    pred = PagePrediction("p1", [_d(0, 0.9, (0, 0, 50, 50))])
    m = evaluate([truth], [pred], ["c0"])
    # One true positive at every IoU threshold, no false positives, no misses.
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)


def test_map_of_a_missed_detection_is_zero():
    truth = PageTruth("p1", 100, 100, boxes=[[0, 0, 50, 50]], class_names=["c0"])
    pred = PagePrediction("p1", [])
    m = evaluate([truth], [pred], ["c0"])
    assert m["map50"] == 0.0
    assert m["map"] == 0.0
    assert m["note"], "an empty prediction set must be labelled, not silently zero"


def test_map_drops_when_the_box_only_overlaps_at_low_iou():
    """A 40% overlap matches at IoU 0.5 but not at 0.75, so mAP@.5 stays 1.0
    while mAP@[.5:.95] falls. Hand-checked: the IoU of a 40x40 box inside a
    100x100 region offset by 30 is 1600/10000 = 0.16, so it matches nothing."""
    truth = PageTruth("p1", 100, 100, boxes=[[0, 0, 100, 100]], class_names=["c0"])
    good = PagePrediction("p1", [_d(0, 0.9, (0, 0, 100, 100))])
    poor = PagePrediction("p1", [_d(0, 0.9, (30, 30, 70, 70))])
    assert iou((0, 0, 100, 100), (30, 30, 70, 70)) == pytest.approx(0.16, abs=1e-6)
    assert evaluate([truth], [good], ["c0"])["map"] == pytest.approx(1.0)
    poor_map = evaluate([truth], [poor], ["c0"])["map"]
    assert poor_map < 0.3, poor_map


def test_perfect_precision_recall_against_hand_counting():
    """Two truths, two correct predictions, one false positive."""
    truths = [
        PageTruth("p1", 100, 100, boxes=[[0, 0, 10, 10], [20, 20, 30, 30]], class_names=["c0", "c1"]),
    ]
    preds = [PagePrediction("p1", [
        _d(0, 0.9, (0, 0, 10, 10)),
        _d(1, 0.8, (20, 20, 30, 30)),
        _d(0, 0.7, (50, 50, 60, 60)),
    ])]
    p, r = precision_recall(truths, preds, threshold=0.5, iou=0.5)
    # tp=2, fp=1, fn=0
    assert (p, r) == (pytest.approx(2 / 3), pytest.approx(1.0))


def test_coco_per_class_ap_keys_match_the_category_names():
    truths = [
        PageTruth("p1", 100, 100, boxes=[[0, 0, 10, 10]], class_names=["alpha"]),
        PageTruth("p2", 100, 100, boxes=[[0, 0, 10, 10]], class_names=["beta"]),
    ]
    preds = [
        PagePrediction("p1", [_d(0, 0.9, (0, 0, 10, 10), label="alpha")]),
        PagePrediction("p2", [_d(1, 0.9, (0, 0, 10, 10), label="beta")]),
    ]
    m = evaluate(truths, preds, ["alpha", "beta", "gamma"])
    assert set(m["per_class_ap"]) == {"alpha", "beta"}
    assert m["per_class_ap"]["alpha"] == pytest.approx(1.0)
    assert m["per_class_ap"]["beta"] == pytest.approx(1.0)
    assert m["n_annotations"] == 2
    assert m["n_images"] == 2


def test_detections_below_the_score_floor_are_excluded():
    truth = PageTruth("p1", 100, 100, boxes=[[0, 0, 50, 50]], class_names=["c0"])
    pred = PagePrediction("p1", [_d(0, 0.01, (0, 0, 50, 50))])
    m = evaluate([truth], [pred], ["c0"])
    assert m["n_detections"] == 0, "a 0.01-score box is below the 0.05 floor"


# -- detection agreement (design §12.3) ------------------------------------


def test_agreement_is_one_for_identical_detection_sets():
    a = [_d(0, 0.9, (0, 0, 10, 10)), _d(1, 0.8, (20, 20, 30, 30))]
    assert detection_agreement(a, list(a)) == pytest.approx(1.0)


def test_agreement_is_one_when_both_sides_are_empty():
    assert detection_agreement([], []) == 1.0


def test_agreement_halves_when_one_side_adds_a_box():
    a = [_d(0, 0.9, (0, 0, 10, 10))]
    b = [*a, _d(1, 0.8, (50, 50, 60, 60))]
    # 2 * 1 / 3
    assert detection_agreement(a, b) == pytest.approx(2 / 3)


def test_agreement_requires_the_same_class():
    a = [_d(0, 0.9, (0, 0, 10, 10))]
    b = [_d(1, 0.9, (0, 0, 10, 10))]
    assert detection_agreement(a, b) == 0.0


def test_agreement_requires_high_iou():
    a = [_d(0, 0.9, (0, 0, 100, 100))]
    b = [_d(0, 0.9, (0, 0, 60, 100))]  # IoU 0.6, below the 0.9 requirement
    assert iou(a[0].box, b[0].box) == pytest.approx(0.6, abs=1e-6)
    assert detection_agreement(a, b) == 0.0


def test_agreement_is_symmetric():
    a = [_d(0, 0.9, (0, 0, 10, 10))]
    b = [_d(0, 0.9, (0, 0, 10, 10)), _d(1, 0.5, (40, 40, 50, 50))]
    assert detection_agreement(a, b) == pytest.approx(detection_agreement(b, a))


def test_agreement_does_not_double_count_one_box():
    a = [_d(0, 0.9, (0, 0, 10, 10))]
    b = [_d(0, 0.9, (0, 0, 10, 10)), _d(0, 0.8, (0, 0, 10, 10))]
    # The duplicate cannot be matched twice, so it counts as a false positive:
    # 2 * 1 / 3, not 4 / 3.
    assert detection_agreement(a, b) == pytest.approx(2 / 3)


# -- operating point (design §12.3) -----------------------------------------


def test_best_f1_threshold_picks_the_operating_point():
    """Two truths. At 0.3 the model finds both; at 0.8 it finds only one."""
    truths = [PageTruth("p1", 100, 100, boxes=[[0, 0, 10, 10], [50, 50, 60, 60]],
                        class_names=["c0", "c0"])]
    preds = {
        0.30: [PagePrediction("p1", [_d(0, 0.95, (0, 0, 10, 10)),
                                     _d(0, 0.35, (50, 50, 60, 60))])],
        0.80: [PagePrediction("p1", [_d(0, 0.95, (0, 0, 10, 10))])],
    }
    t, f1 = best_f1_threshold(truths, preds)
    assert t == pytest.approx(0.30)
    assert f1 == pytest.approx(1.0)


def test_best_f1_threshold_handles_a_model_that_predicts_nothing():
    truths = [PageTruth("p1", 100, 100, boxes=[[0, 0, 10, 10]], class_names=["c0"])]
    t, f1 = best_f1_threshold(truths, {0.5: [PagePrediction("p1", [])]})
    assert f1 == 0.0
    assert t == pytest.approx(0.5)


# -- dataset reproducibility (AT-18) ----------------------------------------


@pytest.mark.slow
def test_manifest_hash_is_reproducible_across_builds(tmp_path):
    """AT-18: the same generator version and seeds must give the same hash."""
    from forgesight.eval import dataset as ds
    from forgesight.settings import get_settings
    from forgesight.storage.object_store import FsObjectStore

    s = get_settings()
    store = FsObjectStore(tmp_path / "objects")
    a = ds.build("synth-clean", s, store, limit=4)
    b = ds.build("synth-clean", s, store, limit=4)
    assert a.manifest_hash == b.manifest_hash
    assert [p.object_key for p in a.pages] == [p.object_key for p in b.pages]
    assert [p.boxes for p in a.pages] == [p.boxes for p in b.pages]

    # A different page count is a different dataset, and says so.
    c = ds.build("synth-clean", s, store, limit=5)
    assert c.manifest_hash != a.manifest_hash


@pytest.mark.slow
def test_dataset_ground_truth_relations_hold(tmp_path):
    """Boxes must be inside the page, well formed, and consistently labelled."""
    from forgesight.eval import dataset as ds
    from forgesight.settings import get_settings
    from forgesight.storage.object_store import FsObjectStore
    from forgesight.synth.templates import CLASS_TO_ID

    s = get_settings()
    store = FsObjectStore(tmp_path / "objects")
    built = ds.build("synth-clean", s, store, limit=8)
    assert built.pages, "no pages built"
    for p in built.pages:
        for box, cid, name in zip(p.boxes, p.class_ids, p.class_names, strict=True):
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= p.width, (box, p.width)
            assert 0 <= y1 < y2 <= p.height, (box, p.height)
            assert CLASS_TO_ID[name] == cid
        assert p.split in {"calib", "test", "bench"}


def test_category_ids_index_by_docling_class_id():
    from forgesight.eval.dataset import category_ids
    from forgesight.synth.templates import CLASS_TO_ID

    names = category_ids()
    assert names[CLASS_TO_ID["title"]] == "title"
    assert names[CLASS_TO_ID["table"]] == "table"
    assert names[CLASS_TO_ID["picture"]] == "picture"
    assert len(names) == max(CLASS_TO_ID.values()) + 1
