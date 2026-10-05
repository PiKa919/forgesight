import numpy as np
from forgesight.synth.generator import generate_synthetic_page

def test_generate_synthetic_page_coordinates():
    # Generate deterministic page at seed 42, 150 DPI
    page_data = generate_synthetic_page(template="two_column", seed=42, dpi=150)
    
    # Assert return structure
    assert "image" in page_data
    assert "boxes" in page_data
    assert "labels" in page_data
    assert "dpi" in page_data
    assert page_data["dpi"] == 150
    
    img = page_data["image"]
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    h, w = img.shape[:2]
    
    # Assert coordinates strictly adhere to top-left image origin
    boxes = page_data["boxes"]
    assert len(boxes) > 0
    for box in boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= w, f"Invalid box width bounds: {box} for w={w}"
        assert 0 <= y1 < y2 <= h, f"Invalid box height bounds: {box} for h={h}"
        
        # Verify region has ink (non-white pixels)
        patch = img[int(y1):int(y2), int(x1):int(x2)]
        non_white = np.mean(patch < 250)
        assert non_white > 0.01, f"Bounding box contains no drawn ink: {box}"
