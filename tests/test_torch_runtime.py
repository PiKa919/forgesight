from pathlib import Path
import numpy as np
import pytest
from forgesight.synth.generator import generate_synthetic_page
from forgesight.vision.runtimes.torch_rt import TorchLayoutModel

@pytest.mark.parametrize("model_name", ["heron", "egret-medium"])
def test_torch_layout_inference(model_name):
    model_path = Path(f"models/{model_name}")
    if not (model_path / "model.safetensors").exists():
        pytest.skip(f"Model {model_name} not downloaded")
        
    page = generate_synthetic_page(seed=42, dpi=150)
    model = TorchLayoutModel(model_path, num_threads=4)
    
    detections = model.predict(page["image"], score_threshold=0.3)
    assert isinstance(detections, list)
    assert len(detections) > 0, f"Expected detections for {model_name} on synthetic test page"
    
    orig_w, orig_h = page["page_size_px"]
    for det in detections:
        assert "class_id" in det
        assert "class_name" in det
        assert "score" in det
        assert 0.3 <= det["score"] <= 1.0
        assert len(det["bbox"]) == 4
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 <= orig_w + 5
        assert 0 <= y1 <= orig_h + 5
        assert x1 <= x2
        assert y1 <= y2
