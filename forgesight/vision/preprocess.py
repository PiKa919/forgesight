import numpy as np
from PIL import Image

def preprocess_image(image_rgb: np.ndarray, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image matching RTDetrImageProcessor (BILINEAR resize, rescale 1/255.0).
    
    Returns float32 NumPy array of shape [3, target_height, target_width].
    """
    pil_img = Image.fromarray(image_rgb)
    resized = pil_img.resize(target_size, resample=Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    # HWC to CHW
    tensor = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(tensor)
