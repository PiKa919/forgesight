from pathlib import Path
import numpy as np
import onnxruntime as ort
from forgesight.vision.preprocess import preprocess_image
from forgesight.vision.postprocess import postprocess_detections

class OrtLayoutModel:
    """ONNX Runtime CPU layout model wrapper."""
    
    def __init__(self, onnx_path: Path, num_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def forward_raw(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run raw forward pass on preprocessed tensor [3, 640, 640].
        
        Returns:
            (logits, pred_boxes) NumPy arrays for batch index 0.
        """
        batch = np.expand_dims(tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: batch})
        # outputs: [logits, pred_boxes]
        return outputs[0][0], outputs[1][0]

    def predict(self, image_rgb: np.ndarray, score_threshold: float = 0.5) -> list[dict]:
        """Full inference pipeline: preprocess -> ORT -> standalone postprocess."""
        orig_h, orig_w = image_rgb.shape[:2]
        tensor = preprocess_image(image_rgb)
        logits, boxes = self.forward_raw(tensor)
        return postprocess_detections(logits, boxes, (orig_w, orig_h), score_threshold)
