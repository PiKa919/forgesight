from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForObjectDetection
from forgesight.vision.preprocess import preprocess_image
from forgesight.vision.postprocess import postprocess_detections

class TorchLayoutModel:
    """PyTorch eager layout model wrapper."""
    
    def __init__(self, model_dir: Path, num_threads: int = 4):
        torch.set_num_threads(num_threads)
        self.device = torch.device("cpu")
        self.model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def forward_raw(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run raw forward pass on preprocessed tensor [3, 640, 640].
        
        Returns:
            (logits, pred_boxes) NumPy arrays for batch index 0.
        """
        batch = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=batch)
        logits = outputs.logits[0].cpu().numpy()
        boxes = outputs.pred_boxes[0].cpu().numpy()
        return logits, boxes

    def predict(self, image_rgb: np.ndarray, score_threshold: float = 0.5) -> list[dict]:
        """Full inference pipeline: preprocess -> model -> standalone postprocess."""
        orig_h, orig_w = image_rgb.shape[:2]
        tensor = preprocess_image(image_rgb)
        logits, boxes = self.forward_raw(tensor)
        return postprocess_detections(logits, boxes, (orig_w, orig_h), score_threshold)
