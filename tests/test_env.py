import sys
import os

def test_environment_constraints():
    # Verify Python version >= 3.12
    assert sys.version_info >= (3, 12), f"Expected Python 3.12+, got {sys.version}"
    
    # Verify project-local HF cache variable isolation
    hf_home = os.environ.get("HF_HOME", "")
    assert "forgesight" in hf_home or ".hf_home" in hf_home, f"HF_HOME must be project-local, got: {hf_home}"
    
    # Verify core dependencies import cleanly
    import torch
    import onnxruntime
    import transformers
    import reportlab
    import pypdfium2
    
    # Verify CPU thread limits can be set
    torch.set_num_threads(4)
    assert torch.get_num_threads() == 4
