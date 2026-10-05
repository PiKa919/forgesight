import numpy as np

ID2LABEL = {
    0: "caption",
    1: "footnote",
    2: "formula",
    3: "list_item",
    4: "page_footer",
    5: "page_header",
    6: "picture",
    7: "section_header",
    8: "table",
    9: "text",
    10: "title",
    11: "document_index",
    12: "code",
    13: "checkbox_selected",
    14: "checkbox_unselected",
    15: "form",
    16: "key_value_region"
}

def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-based [cx, cy, w, h] to corner-based [x1, y1, x2, y2]."""
    x_c, y_c, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    b = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h)
    ]
    return np.stack(b, axis=-1)

def postprocess_detections(
    logits: np.ndarray,
    boxes: np.ndarray,
    orig_target_size: tuple[int, int],
    score_threshold: float = 0.5
) -> list[dict]:
    """Pure NumPy vectorised postprocessor matching RTDetrImageProcessor.post_process_object_detection.
    
    Args:
        logits: [Q, C] classification logits.
        boxes: [Q, 4] normalized [cx, cy, w, h] bounding boxes in [0, 1].
        orig_target_size: (width, height) in original page image pixels.
        score_threshold: minimum confidence score threshold.
        
    Returns:
        List of detection dictionaries with class_id, class_name, score, and bbox [x1, y1, x2, y2].
    """
    # Sigmoid on classification logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    scores = np.max(probs, axis=-1)
    labels = np.argmax(probs, axis=-1)
    
    keep = scores >= score_threshold
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]
    filtered_boxes = boxes[keep]
    
    if len(filtered_scores) == 0:
        return []
        
    xyxy = box_cxcywh_to_xyxy(filtered_boxes)
    # Scale from normalized [0, 1] to original image [width, height]
    orig_w, orig_h = orig_target_size
    xyxy[..., [0, 2]] = np.clip(xyxy[..., [0, 2]] * orig_w, 0, orig_w)
    xyxy[..., [1, 3]] = np.clip(xyxy[..., [1, 3]] * orig_h, 0, orig_h)
    
    # Sort detections descending by score
    sort_order = np.argsort(-filtered_scores)
    
    results = []
    for idx in sort_order:
        s = float(filtered_scores[idx])
        l = int(filtered_labels[idx])
        b = xyxy[idx]
        results.append({
            "class_id": l,
            "class_name": ID2LABEL.get(l, f"class_{l}"),
            "score": float(round(s, 4)),
            "bbox": [float(round(coord, 2)) for coord in b]
        })
    return results
