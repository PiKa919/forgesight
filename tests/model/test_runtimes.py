"""Runtime acceptance tests: AT-16 (ONNX parity), AT-17 (batch invariance),
AT-22 (artifact integrity), and the preprocessing variants of design 8.7.

These are the tests that decide whether a comparison in the report means
anything. They are separated from the fast unit suite because each one loads
model weights.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from forgesight.settings import get_settings
from forgesight.synth.generator import generate_page
from forgesight.vision.cancel import NeverCancel, ThreadingCancelToken
from forgesight.vision.postprocess import postprocess_batch
from forgesight.vision.preprocess import Preprocessor, make_profile, tensor_diff
from forgesight.vision.registry import artifact_for, load_lock
from forgesight.vision.runtimes.base import (
    ArtifactCorrupt,
    OrtModel,
    TorchModel,
    file_sha256,
    verify_artifact,
)
from forgesight.vision.types import Cancelled, Detection, RuntimeProfile
from tests.support import MODEL_NAMES, requires_onnx, requires_weights


def _profile(runtime: str, **kw) -> RuntimeProfile:
    return RuntimeProfile(
        runtime=runtime,
        intra_op_threads=kw.pop("threads", 4),
        inter_op_threads=1,
        graph_opt_level="ORT_ENABLE_ALL",
        max_batch=kw.pop("max_batch", 8),
        batch_window_ms=25,
    )


def _load(name: str, runtime: str, **kw):
    s = get_settings()
    lock = load_lock(s)
    fmt = "safetensors" if runtime == "torch" else "onnx-fp32"
    art = artifact_for(lock, name, s, fmt)
    profile = _profile(runtime, **kw)
    if runtime == "torch":
        return TorchModel(art, profile)
    return OrtModel(art, profile)


def _pages(n: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        t = ("two_column", "single_column", "slide", "two_up")[i % 4]
        out.append(generate_page(template=t, seed=100 + i, dpi=150)["image"])
    return out


def _agree(a: list[Detection], b: list[Detection], iou_min: float = 0.9) -> float:
    """2*matched / (|a| + |b|), same class, IoU >= threshold (design 12.3)."""
    from forgesight.vision.postprocess import iou

    if not a and not b:
        return 1.0
    used = set()
    matched = 0
    for da in a:
        best, best_iou = None, iou_min
        for j, db in enumerate(b):
            if j in used or db.class_id != da.class_id:
                continue
            v = iou(da.box, db.box)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            matched += 1
    return 2 * matched / (len(a) + len(b))


# -- AT-16: ONNX parity -----------------------------------------------------


@requires_weights
@requires_onnx
@pytest.mark.model
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_raw_outputs_match_torch(name, page):
    """AT-16: the runtimes agree on raw tensors, isolating the runtime from
    preprocessing. Anything above fp32 rounding means the export is wrong."""
    torch_model = _load(name, "torch")
    ort_model = _load(name, "onnxruntime")
    proc = Preprocessor(make_profile("pil_bilinear"))
    item = proc(page["image"])[0]

    t_raw = torch_model.infer(item.tensor[None], NeverCancel())
    o_raw = ort_model.infer(item.tensor[None], NeverCancel())
    t_logits, t_boxes = t_raw.logits, t_raw.boxes
    o_logits, o_boxes = o_raw.logits, o_raw.boxes

    assert t_logits.shape == o_logits.shape
    assert t_boxes.shape == o_boxes.shape
    logit_diff = float(np.max(np.abs(t_logits - o_logits)))
    box_diff = float(np.max(np.abs(t_boxes - o_boxes)))
    print(f"\n[{name}] max logit diff {logit_diff:.2e}, max box diff {box_diff:.2e}")

    # Compared *relatively*, against the magnitude of the logits themselves.
    # An absolute bound was architecture-dependent and failed on x86_64 (1.2e-2
    # against a 1e-2 limit) while passing on arm64 for the same code, because the
    # runtimes pick different SIMD kernels and reduction orders per architecture.
    # The relative form is what "fp32 rounding" actually means, and it separates
    # the two cases by orders of magnitude:
    #
    #   measured, same input   heron 3.6e-4   egret-medium 1.5e-3
    #   mismatched input       heron 7.6e-1   egret-medium 1.0e+0
    #
    # so 5e-3 sits ~3x above the worst real reading and ~150x below the case it
    # exists to catch. The box comparison stays absolute: boxes are normalised
    # coordinates, so their scale does not vary and there is nothing to be
    # relative to.
    logit_scale = max(float(np.max(np.abs(t_logits))), 1e-9)
    logit_rel = logit_diff / logit_scale
    print(f"[{name}] logit scale {logit_scale:.3e}, relative {logit_rel:.2e}")
    assert logit_rel < 5e-3, (
        f"logit divergence {logit_diff:.3e} is {logit_rel:.2e} of the {logit_scale:.3e} "
        f"logit scale, which is not fp32 rounding"
    )
    assert box_diff < 1e-2, f"box divergence {box_diff}"


@requires_weights
@requires_onnx
@pytest.mark.model
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_detections_agree_with_torch(name, page):
    """The gate that actually matters (G4): agreement on detections, not tensors."""
    torch_model = _load(name, "torch")
    ort_model = _load(name, "onnxruntime")
    proc = Preprocessor(make_profile("pil_bilinear"))
    item = proc(page["image"])[0]
    t = torch_model.infer(item.tensor[None], NeverCancel())
    o = ort_model.infer(item.tensor[None], NeverCancel())
    dets_t = postprocess_batch(t, [item], torch_model.id2label, 0.30)[0]
    dets_o = postprocess_batch(o, [item], ort_model.id2label, 0.30)[0]
    agree = _agree(dets_t, dets_o)
    print(f"\n[{name}] detection agreement torch vs ort: {agree:.4f} "
          f"({len(dets_t)} vs {len(dets_o)})")
    assert agree >= 0.98, f"detection agreement {agree:.4f} below the G4 threshold"


@requires_onnx
@pytest.mark.model
def test_onnx_batch_dimension_is_honoured():
    """The exported graph claims a dynamic batch axis; prove it is usable."""
    model = _load("egret-medium", "onnxruntime")
    proc = Preprocessor(make_profile("pil_bilinear"))
    items = proc(_pages(3)[0])[0]
    batch = np.stack([items.tensor] * 3)
    out = model.infer(batch, NeverCancel())
    assert out.logits.shape[0] == 3, out.logits.shape
    assert out.boxes.shape[0] == 3, out.boxes.shape


# -- AT-17: batch invariance ------------------------------------------------


@requires_weights
@requires_onnx
@pytest.mark.model
@pytest.mark.parametrize("runtime", ["torch", "onnxruntime"])
def test_batching_does_not_change_detections(runtime):
    """AT-17: b=1 and b=8 must agree. If batching changed quality, every
    throughput number in the report would be trading accuracy for speed."""
    model = _load("egret-medium", runtime)
    proc = Preprocessor(make_profile("pil_bilinear"))
    images = _pages(8)
    items = [proc(img)[0] for img in images]

    single = []
    for it in items:
        raw = model.infer(it.tensor[None], NeverCancel())
        single.append(postprocess_batch(raw, [it], model.id2label, 0.30)[0])
    batched = postprocess_batch(
        model.infer(np.stack([i.tensor for i in items]), NeverCancel()),
        items,
        model.id2label,
        0.30,
    )

    agreements = [_agree(a, b) for a, b in zip(single, batched, strict=True)]
    worst = min(agreements)
    print(f"\n[{runtime}] b=1 vs b=8 worst-page agreement: {worst:.4f}")
    assert worst >= 0.99, f"batch changed detections (worst agreement {worst:.4f})"


# -- AT-22: artifact integrity ---------------------------------------------


@requires_weights
@pytest.mark.model
def test_artifact_sha_mismatch_refuses_load(tmp_path):
    """AT-22: a corrupted cache file must fail the load, not run anyway."""
    import shutil

    s = get_settings()
    src = s.models_dir / "heron"
    dst = tmp_path / "heron"
    shutil.copytree(src, dst)
    with open(dst / "model.safetensors", "r+b") as fh:
        fh.seek(0)
        fh.write(b"\x00\x00\x00\x00")
    with pytest.raises(ArtifactCorrupt, match="sha256 mismatch"):
        verify_artifact(dst / "model.safetensors", "0" * 64)

    lock = load_lock(s)
    art = artifact_for(lock, "heron", s, "safetensors")
    with pytest.raises(ArtifactCorrupt):
        TorchModel(_with_path(art, dst), _profile("torch"))


def _with_path(art, path):
    from dataclasses import replace

    return replace(art, path=str(path))


@requires_weights
@pytest.mark.model
def test_missing_artifact_refuses_load(tmp_path):
    with pytest.raises(ArtifactCorrupt, match="missing"):
        verify_artifact(tmp_path / "nope.safetensors", "x" * 64)


# -- cancellation -----------------------------------------------------------


@requires_onnx
@pytest.mark.model
def test_ort_respects_a_cancel_token_set_before_inference():
    model = _load("egret-medium", "onnxruntime")
    item = Preprocessor(make_profile("pil_bilinear"))(_pages(1)[0])[0]
    token = ThreadingCancelToken()
    token.set()
    with pytest.raises(Cancelled):
        model.infer(item.tensor[None], token)


def test_cancel_token_fires_hooks_and_handles_late_registration():
    """The token is our code, so its contract is tested directly.

    A previous version of this test raced inference from another thread and
    passed or failed depending on whether the model finished first, which proved
    nothing about the wiring. ONNX Runtime's own terminate behaviour is not ours
    to assert; the pre-set-token test above covers the observable contract.
    """
    token = ThreadingCancelToken()
    assert token.is_set() is False

    seen: list[str] = []
    token.on_set(lambda: seen.append("early"))
    token.set()
    assert token.is_set() and seen == ["early"]

    # Registering after the fact must fire immediately, not silently never run.
    late: list[str] = []
    token.on_set(lambda: late.append("late"))
    assert late == ["late"], "a hook registered after set() never ran"


def test_cancel_token_ignores_a_failing_hook():
    """A broken hook must not stop the token being observed."""
    token = ThreadingCancelToken()
    ran: list[str] = []
    token.on_set(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    token.on_set(lambda: ran.append("ok"))
    token.set()
    assert token.is_set() and ran == ["ok"]


# -- preprocessing variants (design 8.7) ------------------------------------


@requires_weights
@pytest.mark.model
def test_preprocessing_variants_differ_where_it_matters():
    """Downscaling an A4 page to 640 is exactly where antialiasing shows up.

    This is the Anomalib #2944 / #3726 failure mode: an exported model that
    quietly diverged because antialiasing was dropped between two code paths.
    """
    page = generate_page(template="two_column", seed=7, dpi=150)
    ref = Preprocessor(make_profile("pil_bilinear"))(page["image"])[0].tensor
    deltas = {}
    for method in ("cv2_linear", "cv2_area", "pil_reduce"):
        t = Preprocessor(make_profile(method))(page["image"])[0].tensor
        deltas[method] = tensor_diff(ref, t)
    for method, (mx, mean) in deltas.items():
        print(f"\n{method:11s} max {mx:.4f} mean {mean:.5f}")
    # cv2.INTER_LINEAR does not antialias on downscale, so it must differ most.
    assert deltas["cv2_linear"][0] > deltas["cv2_area"][0], deltas
    assert deltas["cv2_linear"][0] > 1e-3, "cv2_linear matched pil_bilinear exactly"


@requires_weights
@pytest.mark.model
def test_preprocess_profiles_have_distinct_hashes():
    base = make_profile("pil_bilinear")
    hashes = {m: make_profile(m).hash for m in
              ("pil_bilinear", "cv2_linear", "cv2_area", "pil_reduce")}
    assert len(set(hashes.values())) == 4, hashes
    assert make_profile("pil_bilinear").hash == base.hash
    assert make_profile("pil_bilinear", target=320).hash != base.hash
    assert make_profile("pil_bilinear", render_dpi=200).hash != base.hash


@requires_weights
@pytest.mark.model
def test_reference_preprocess_is_bit_exact_with_transformers(page):
    """The reference profile must equal RTDetrImageProcessor, not approximate it."""
    from transformers import RTDetrImageProcessor

    from forgesight.settings import get_settings

    s = get_settings()
    hf = RTDetrImageProcessor.from_pretrained(str(s.models_dir / "heron"))
    out = hf(images=page["image"], return_tensors="np")["pixel_values"]
    ours = Preprocessor(make_profile("pil_bilinear"))(page["image"])[0].tensor
    mx, mean = tensor_diff(ours, out[0])
    print(f"\nvs transformers: max {mx:.2e} mean {mean:.2e}")
    assert mx == 0.0, f"reference preprocessing is not bit-exact: max {mx}"


# -- tiling -----------------------------------------------------------------


@requires_onnx
@pytest.mark.model
def test_tiling_splits_an_oversized_page_and_offsets_boxes():
    """Design 8.8: tiling is a hypothesis, so it must at least behave correctly."""
    model = _load("egret-medium", "onnxruntime")
    # 220 DPI puts the two-up spread's long side at 2420 px, over the policy
    # threshold. 200 DPI gives exactly 2200, and the policy is a strict `>`.
    # See the boundary test below for why the default DPI is not enough.
    image = generate_page(template="two_up", seed=3, dpi=220)["image"]

    plain = Preprocessor(make_profile("pil_bilinear"), tile_enabled=False)
    tiled = Preprocessor(make_profile("pil_bilinear"), tile_enabled=True)
    assert len(plain(image)) == 1
    tiles = tiled(image)
    assert len(tiles) == 2
    # The second tile must carry a non-zero origin, or its boxes land on top of
    # the first tile's.
    assert tiles[1].tile_origin is not None
    assert any(t.tile_origin != (0, 0) for t in tiles)

    raw = model.infer(np.stack([t.tensor for t in tiles]), NeverCancel())
    dets = postprocess_batch(raw, tiles, model.id2label, 0.30)
    assert all(len(d) > 0 for d in dets), "tiled inference produced nothing"
    w, h = image.shape[1], image.shape[0]
    for tile_dets in dets:
        for d in tile_dets:
            assert -1 <= d.box[0] <= w + 1 and -1 <= d.box[1] <= h + 1, d


@requires_onnx
@pytest.mark.model
def test_small_pages_are_not_tiled():
    image = generate_page(template="single_column", seed=3, dpi=100)["image"]
    tiled = Preprocessor(make_profile("pil_bilinear"), tile_enabled=True)
    assert len(tiled(image)) == 1, "a small page should not be split"


def test_tiling_threshold_boundary_is_where_the_policy_says():
    """A finding worth pinning.

    Design 8.8 sets the tiling policy at long_side_px > 2200 or aspect > 1.6,
    and design 9.1 names the two-up template as the tiling case. Those two
    statements do not agree: a two-up landscape letter page rendered at the
    default 150 DPI is 1651x1275, so its long side is under the threshold and
    its aspect is 1.29, under the aspect threshold. The two-up page is therefore
    *not* tiled at the default render DPI.

    The threshold is left exactly as the design wrote it, and the tiling
    experiment runs at a render DPI that reaches it. Changing a policy number to
    make a test pass would hide this. See docs/adr/0003-tiling-threshold.md.
    """
    tiled = Preprocessor(make_profile("pil_bilinear"), tile_enabled=True)
    at_150 = generate_page(template="two_up", seed=3, dpi=150)["image"]
    at_200 = generate_page(template="two_up", seed=3, dpi=200)["image"]
    at_220 = generate_page(template="two_up", seed=3, dpi=220)["image"]
    assert len(tiled(at_150)) == 1, "150 DPI two-up is under the policy threshold"
    # 200 DPI lands exactly on 2200 and the policy is a strict greater-than.
    assert len(tiled(at_200)) == 1
    assert len(tiled(at_220)) == 2, "220 DPI two-up crosses it"


def test_extreme_aspect_ratio_tiles_regardless_of_length():
    """The aspect clause is independent of the pixel clause."""
    tall = np.full((4000, 400, 3), 255, dtype=np.uint8)
    tall[100:900, 50:350] = 0
    tall = np.repeat(np.repeat(tall, 1, axis=0), 1, axis=1)
    profile = make_profile("pil_bilinear", tile_long_side_px=10_000, tile_aspect=1.6)
    tiles = Preprocessor(profile, tile_enabled=True)(tall)
    assert len(tiles) == 2
    assert tiles[0].tile_origin == (0, 0)
    assert tiles[1].tile_origin[0] > 0 or tiles[1].tile_origin[1] > 0


# -- AT-5: cancellation during inference -------------------------------------


def _cancel_soon(token: ThreadingCancelToken, delay: float) -> threading.Thread:
    """Fire `token` from another thread after `delay` seconds."""

    def go():
        time.sleep(delay)
        token.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def _time_infer(model, batch, token):
    t0 = time.perf_counter()
    try:
        model.infer(batch, token)
        return time.perf_counter() - t0, None
    except Exception as exc:
        # Recorded and asserted by the caller, which distinguishes Cancelled from
        # any other failure. A bare `except Exception` is right here precisely
        # because the point is to measure and report whatever came out.
        return time.perf_counter() - t0, exc


@requires_weights
@pytest.mark.model
def test_cancel_running_torch_completes_then_drops_the_result():
    """AT-5, torch half: the forward pass runs, and its output is discarded.

    PyTorch eager cannot preempt a kernel mid-op, so cancellation is cooperative
    -- `infer` checks the token before the forward, runs it to completion, then
    checks again and raises `Cancelled`.

    An earlier version of this test asserted that by *timing*: the cancelled run
    should take about as long as an uncancelled one. That was flaky, failing two
    runs in three, and the fault was in the test rather than the code. Both
    timings measure the same completed forward pass, so the only thing separating
    them is measurement noise -- and the baseline was measured cold while the
    cancelled run came second and warm, so a legitimately faster run looked like
    preemption. "No speedup" and "noise" are the same signal at this margin.

    So this asserts the mechanism instead: the model's forward really was called,
    and `Cancelled` still came out. That is deterministic, and it is the thing
    the asymmetry rests on -- ORT raises the same exception without the forward
    ever finishing, which the ORT test below demonstrates by timing, where the
    margin is roughly 25x and the noise cannot reach it.
    """
    model = _load("egret-medium", "torch")
    item = Preprocessor(make_profile("pil_bilinear"))(_pages(1)[0])[0]

    calls = []
    inner = model.model.forward

    def spy(*a, **k):
        calls.append(1)
        return inner(*a, **k)

    model.model.forward = spy

    token = ThreadingCancelToken()
    _cancel_soon(token, 0.05)

    with pytest.raises(Cancelled):
        model.infer(item.tensor[None], token)

    assert calls, (
        "the forward pass never ran, so this was not 'completes then drops' -- "
        "either the token was honoured before the work, or it was never started"
    )
    assert token.is_set()


@requires_weights
@requires_onnx
@pytest.mark.model
def test_cancel_running_ort_terminates_in_flight():
    """AT-5, ORT half: the run is actually stopped, not detected afterwards.

    ONNX Runtime honours `RunOptions.terminate`, so the call returns sooner than
    the inference would have taken. Without real termination the run finishes,
    the token is noticed afterwards, and the operator still pays the full cost --
    which is the entire reason the two runtimes behave differently here.

    Measured rather than asserted structurally, because there is no seam to
    inspect: the measurement is the only evidence that the run stopped. It is
    safe to measure because the margin is about 25x (a terminated run returns in
    roughly 4% of an uncancelled one), nowhere near where timing noise lives.
    """
    model = _load("egret-medium", "onnxruntime")
    item = Preprocessor(make_profile("pil_bilinear"))(_pages(1)[0])[0]
    batch = item.tensor[None]

    # Warm up first. The first inference pays for session setup and first-touch
    # faults; measuring that as the baseline would hand the cancelled run an
    # unfair advantage and make the assertion pass for the wrong reason.
    _time_infer(model, batch, NeverCancel())
    baseline, baseline_exc = _time_infer(model, batch, NeverCancel())
    assert baseline_exc is None, f"baseline inference failed: {baseline_exc!r}"

    token = ThreadingCancelToken()
    _cancel_soon(token, 0.05)
    elapsed, exc = _time_infer(model, batch, token)

    assert isinstance(exc, Cancelled), (
        f"a terminated ORT run must raise Cancelled, got {exc!r} after {elapsed:.2f}s"
    )
    assert elapsed < baseline * 0.75, (
        f"cancelled ORT run took {elapsed:.2f}s against a {baseline:.2f}s baseline; "
        "that looks like the run completed and the token was noticed afterwards, "
        "not an in-flight terminate"
    )


def test_provenance_accepts_a_safetensors_directory(tmp_path):
    """G6 must be able to pass for a torch candidate.

    For safetensors, ModelArtifact.path is the model *directory* and the lock's
    sha256 is of model.safetensors inside it. Callers that handed
    verify_artifact() the directory got IsADirectoryError, so G6_provenance
    returned False for every torch candidate and no torch candidate could ever be
    promoted. Verify the check resolves the same file the hash describes, in both
    directions: a good artifact passes and a corrupted one is still refused.
    """
    from forgesight.eval.runner import _provenance
    from forgesight.vision.types import ModelArtifact

    model_dir = tmp_path / "heron"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"real weights")

    artifact = ModelArtifact(
        name="heron", path=str(model_dir), sha256=file_sha256(weights),
        format="safetensors", repo_id="example/heron", revision="a" * 40,
        license="Apache-2.0")
    candidate = SimpleNamespace(artifact=artifact)

    ok, detail = _provenance(candidate, None, "ws", [{"manifest_hash": "m" * 64}])
    assert ok is True, f"G6 refused a perfectly good artifact: {detail}"
    assert "sha verified" in detail

    # And the check is still doing its job on the same layout.
    weights.write_bytes(b"tampered")
    ok2, detail2 = _provenance(candidate, None, "ws", [{"manifest_hash": "m" * 64}])
    assert ok2 is False
    assert "sha mismatch" in detail2
