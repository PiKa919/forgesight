import warnings
from pathlib import Path

import torch
from transformers import AutoModelForObjectDetection


def export_to_onnx(model_dir: Path, output_path: Path, opset: int = 17) -> Path:
    """Export AutoModelForObjectDetection model to standalone ONNX using TorchScript exporter.
    
    Note: We explicitly pass dynamo=False because RT-DETRv2 / D-FINE models contain aten._is_all_true
    control assertions that lack decomposition mappings in Torch Dynamo ONNX translator.
    The TorchScript exporter embeds all parameters and preserves exact dynamic batching.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False
        )
    return output_path
