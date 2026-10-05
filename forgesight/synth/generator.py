import io
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pypdfium2 as pdfium
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CLASS_MAP = {
    "title": 10,
    "section_header": 7,
    "text": 9,
    "table": 8,
    "picture": 6,
    "page_header": 5,
    "page_footer": 4
}

def generate_synthetic_page(
    template: Literal["single_column", "two_column", "slide"] = "two_column",
    seed: int = 42,
    dpi: int = 150
) -> dict:
    """Generate a synthetic document layout page with mathematically exact ground truth boxes.
    
    Transforms ReportLab's Cartesian coordinate space (origin at bottom-left, y increasing upwards)
    into standard Computer Vision image pixel space (origin at top-left, y increasing downwards).
    """
    pt_to_px = dpi / 72.0
    page_w_pt, page_h_pt = letter
    w_px = int(round(page_w_pt * pt_to_px))
    h_px = int(round(page_h_pt * pt_to_px))
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    boxes_pt = []
    
    # 1. Page Header
    c.setFont("Helvetica-Bold", 10)
    c.drawString(54, page_h_pt - 40, "SYNTHETIC BENCHMARK REPORT - FORGESIGHT")
    boxes_pt.append((54, page_h_pt - 45, 300, 15, "page_header"))
    
    # 2. Document Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(54, page_h_pt - 80, f"Synthetic Document Layout Test Page #{seed}")
    boxes_pt.append((54, page_h_pt - 88, 480, 26, "title"))
    
    # 3. Two columns of layout elements
    col_w = 230
    c.setFont("Helvetica", 9)
    
    # Left column: Section header + text box
    c.drawString(54, page_h_pt - 130, "1.0 Systems Execution & Memory Analysis")
    boxes_pt.append((54, page_h_pt - 135, col_w, 14, "section_header"))
    
    c.rect(54, page_h_pt - 280, col_w, 130, fill=0, stroke=1)
    c.drawString(60, page_h_pt - 160, "Process-isolated workers mitigate heap fragmentation.")
    c.drawString(60, page_h_pt - 180, "Bounded async pipeline guarantees deterministic queue wait.")
    boxes_pt.append((54, page_h_pt - 280, col_w, 130, "text"))
    
    # Right column: Section header + simulated picture/chart
    c.drawString(310, page_h_pt - 130, "2.0 Performance Telemetry")
    boxes_pt.append((310, page_h_pt - 135, col_w, 14, "section_header"))
    
    c.rect(310, page_h_pt - 280, col_w, 130, fill=0, stroke=1)
    c.line(320, page_h_pt - 270, 520, page_h_pt - 170)
    c.drawString(320, page_h_pt - 160, "[Synthetic Benchmark Plot - Latency vs RSS]")
    boxes_pt.append((310, page_h_pt - 280, col_w, 130, "picture"))
    
    # 4. Page Footer
    c.setFont("Helvetica", 9)
    c.drawString(54, 30, f"SYNTHETIC DATA · PROVENANCE: forgesight-synth@v1#seed={seed}")
    boxes_pt.append((54, 25, 450, 15, "page_footer"))
    
    c.showPage()
    c.save()
    
    # Render PDF to RGB image with pypdfium2
    buffer.seek(0)
    pdf = pdfium.PdfDocument(buffer)
    page = pdf[0]
    bitmap = page.render(scale=pt_to_px)
    pil_image = bitmap.to_pil().convert("RGB")
    img_arr = np.array(pil_image)
    
    # Transform boxes: ReportLab (bottom-left) to Image (top-left) in pixels
    # ReportLab rect: (x, y, w, h) where y is bottom coordinate
    boxes_px = []
    box_labels = []
    for x_pt, y_pt, bw_pt, bh_pt, lbl in boxes_pt:
        x1 = x_pt * pt_to_px
        x2 = (x_pt + bw_pt) * pt_to_px
        # y_pt is bottom coordinate; top coordinate in points is y_pt + bh_pt
        y1 = (page_h_pt - (y_pt + bh_pt)) * pt_to_px
        y2 = (page_h_pt - y_pt) * pt_to_px
        boxes_px.append([round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)])
        box_labels.append(lbl)
        
    return {
        "image": img_arr,
        "boxes": boxes_px,
        "labels": box_labels,
        "dpi": dpi,
        "page_size_px": (w_px, h_px)
    }
