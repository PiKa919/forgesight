from pathlib import Path
import numpy as np
import pytest
from forgesight.synth.generator import generate_synthetic_page
from forgesight.vision.preprocess import preprocess_image
from forgesight.vision.runtimes.torch_rt import TorchLayoutModel
from forgesight.vision.runtimes.ort_rt import OrtLayoutModel
from forgesight.vision.export_onnx import export_to_onnx

@pytest.mark.parametrize("model_name", ["heron", "egret-medium"])
def test_torch_onnx_parity(model_name, tmp_path):
    model_dir = Path(f"models/{model_name}")
    if not (model_dir / "model.safetensors").exists():
        pytest.skip(f"Model {model_name} not downloaded")
        
    onnx_file = tmp_path / f"{model_name}.onnx"
    export_to_onnx(model_dir, onnx_file)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 10 * 1024 * 1024  # > 10 MB
    
    torch_model = TorchLayoutModel(model_dir, num_threads=4)
    ort_model = OrtLayoutModel(onnx_file, num_threads=4)
    
    page = generate_synthetic_page(seed=42, dpi=150)
    tensor = preprocess_image(page["image"])
    
    torch_logits, torch_boxes = torch_model.forward_raw(tensor)
    ort_logits, ort_boxes = ort_model.forward_raw(tensor)
    
    # Assert output shapes match
    assert torch_logits.shape == ort_logits.shape
    assert torch_boxes.shape == ort_boxes.shape
    
    # Check max absolute difference between PyTorch CPU and ONNX Runtime CPU
    max_logit_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    max_box_diff = float(np.max(np.abs(torch_boxes - ort_boxes)))
    
    print(f"\n[{model_name}] Max logit diff: {max_logit_diff:.6f}, Max box diff: {max_box_diff:.6f}")
    assert max_logit_diff < 1e-2, f"Logit diff {max_logit_diff} exceeds tolerance"
    assert max_box_diff < 1e-2, f"Box diff {max_box_diff} exceeds tolerance"
