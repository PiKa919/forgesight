"""Page templates.

Each template lays a page out and reports the *exact* rectangle it drew, in
PostScript points, in ReportLab's bottom-left coordinate frame. The generator
converts once to top-left pixel space.

The hard requirement is tightness: a ground-truth box that is much larger than
the ink it contains teaches the wrong thing during evaluation, because a loose
box can be "correct" for the wrong reason. So the layout code measures the text
it just drew (`stringWidth`) and the box hugs that, rather than assuming a
character count times a magic number.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from forgesight.synth import textgen

# A stroke=1 line paints half its width outside the path, so every
# rect-derived ground-truth box is grown by this much to contain its stroke.
STROKE_PAD = 0.75

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_MONO = "Courier"

# Docling label map. Kept in one place and asserted against the model config.
CLASS_TO_ID = {
    "caption": 0,
    "footnote": 1,
    "formula": 2,
    "list_item": 3,
    "page_footer": 4,
    "page_header": 5,
    "picture": 6,
    "section_header": 7,
    "table": 8,
    "text": 9,
    "title": 10,
    "document_index": 11,
    "code": 12,
    "checkbox_selected": 13,
    "checkbox_unselected": 14,
    "form": 15,
    "key_value_region": 16,
}

# Classes this generator actually emits. Everything else is out of v1 and is
# excluded from metrics, which every report states explicitly (design §9.1).
EMITTED_CLASSES = (
    "title",
    "section_header",
    "text",
    "list_item",
    "table",
    "picture",
    "caption",
    "page_header",
    "page_footer",
    "footnote",
    "code",
)


@dataclass(slots=True)
class Box:
    x: float
    y: float  # bottom edge, points, ReportLab frame
    w: float
    h: float
    cls: str

    def to_px(self, page_h_pt: float, pt_to_px: float) -> list[float]:
        x1 = self.x * pt_to_px
        x2 = (self.x + self.w) * pt_to_px
        y1 = (page_h_pt - (self.y + self.h)) * pt_to_px
        y2 = (page_h_pt - self.y) * pt_to_px
        return [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]


@dataclass(slots=True)
class Layout:
    c: canvas.Canvas
    buf: io.BytesIO
    page_w: float
    page_h: float
    boxes: list[Box] = field(default_factory=list)
    margin: float = 54.0

    # -- primitives ------------------------------------------------------
    def text_line(
        self, x: float, y_baseline: float, txt: str, size: float, cls: str, bold: bool = False,
        mono: bool = False, pad: float = 1.5,
    ) -> Box:
        font = FONT_MONO if mono else (FONT_BOLD if bold else FONT)
        self.c.setFont(font, size)
        w = self.c.stringWidth(txt, font, size)
        ascent, descent = _metrics(font, size)
        # Box hugs the ink: ascent above the baseline, descent below, plus a pad
        # so antialiased edges are inside. `descent` is already negative, so the
        # bottom edge is baseline + descent.
        box = Box(
            x - pad, y_baseline + descent - pad, w + 2 * pad, (ascent - descent) + 2 * pad, cls
        )
        self.c.drawString(x, y_baseline, txt)
        self.boxes.append(box)
        return box

    def text_block(
        self, x: float, y_top: float, width: float, body: str, size: float, leading: float,
        cls: str = "text",
    ) -> Box:
        """Draw a wrapped block; return one box covering the whole block."""
        font = FONT
        self.c.setFont(font, size)
        lines = self._wrap(body, font, size, width)
        ascent, descent = _metrics(font, size)
        y = y_top
        for ln in lines:
            self.c.drawString(x, y - ascent, ln)
            y -= leading
        height = (y_top - y) + descent
        widest = max(self.c.stringWidth(ln, font, size) for ln in lines)
        # Bound the box to real ink: first line's ink starts at y_top (its
        # baseline is y_top - ascent), and the last line's descenders reach
        # last_baseline + descent. Without this, ascenders and descenders leak
        # a fraction of a point outside the box, which is enough to fail the
        # 99.5% ink-coverage rule once a page has thirty lines of text.
        # The last line was drawn at baseline (y + leading) - ascent, because
        # the loop draws at (y - ascent) and then steps y down by `leading`.
        last_baseline = y + leading - ascent
        pad = 2.0
        box = Box(
            x - pad,
            last_baseline + descent - pad,
            widest + 2 * pad,
            (y_top - last_baseline - descent) + 2 * pad,
            cls,
        )
        self.boxes.append(box)
        return box

    def _wrap(self, body: str, font: str, size: float, width: float) -> list[str]:
        out, cur = [], ""
        for w in body.split():
            trial = f"{cur} {w}".strip()
            if self.c.stringWidth(trial, font, size) <= width or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out

    def rule_box(self, x: float, y: float, w: float, h: float, cls: str) -> Box:
        self.c.rect(x, y, w, h, fill=0, stroke=1)
        box = Box(x - STROKE_PAD, y - STROKE_PAD, w + 2 * STROKE_PAD, h + 2 * STROKE_PAD, cls)
        self.boxes.append(box)
        return box

    def picture(self, x: float, y: float, w: float, h: float, rng: np.random.Generator) -> Box:
        """Procedural chart-like figure. No external image, no photo rights."""
        self.c.rect(x, y, w, h, fill=0, stroke=1)
        n = int(rng.integers(3, 7))
        px0, px1 = x + 8, x + w - 8
        py0, py1 = y + 8, y + h - 8
        prev = None
        for i in range(n):
            t = i / max(1, n - 1)
            gx = px0 + t * (px1 - px0)
            gy = py0 + float(rng.random()) * (py1 - py0)
            if prev is not None:
                self.c.line(prev[0], prev[1], gx, gy)
            prev = (gx, gy)
            self.c.circle(gx, gy, 1.6, fill=1, stroke=0)
        for _ in range(3):
            ly = py0 + float(rng.random()) * (py1 - py0)
            self.c.line(px0, ly, px1, ly)
        box = Box(x - STROKE_PAD, y - STROKE_PAD, w + 2 * STROKE_PAD, h + 2 * STROKE_PAD, "picture")
        self.boxes.append(box)
        return box

    def table(self, x: float, y_top: float, w: float, rows: int, cols: int, rng) -> Box:
        """Grid of cells with text. One box for the whole table (design §9.1)."""
        row_h = 16.0
        h = rows * row_h
        y_bottom = y_top - h
        self.c.rect(x, y_bottom, w, h, fill=0, stroke=1)
        for r in range(1, rows):
            self.c.line(x, y_bottom + r * row_h, x + w, y_bottom + r * row_h)
        col_w = w / cols
        for cidx in range(1, cols):
            self.c.line(x + cidx * col_w, y_bottom, x + cidx * col_w, y_top)
        self.c.setFont(FONT, 7.5)
        for r in range(rows):
            for cidx in range(cols):
                token = f"{rng.choice(('a','b','c','d','e','f'))}{int(rng.integers(1,99))}"
                self.c.drawString(
                    x + cidx * col_w + 3, y_top - r * row_h - 11, token
                )
        box = Box(
            x - STROKE_PAD, y_bottom - STROKE_PAD, w + 2 * STROKE_PAD, h + 2 * STROKE_PAD, "table"
        )
        self.boxes.append(box)
        return box

    def list_block(
        self, x: float, y_top: float, width: float, items: list[str], size: float, leading: float
    ) -> list[Box]:
        out = []
        y = y_top
        for it in items:
            b = self.text_line(x + 10, y - 8, f"• {it}", size, "list_item")
            out.append(b)
            y -= leading
        return out

    def code_block(self, x: float, y_top: float, lines: list[str], size: float, leading: float) -> Box:
        self.c.setFont(FONT_MONO, size)
        ascent, descent = _metrics(FONT_MONO, size)
        y = y_top
        for ln in lines:
            self.c.drawString(x, y - ascent, ln)
            y -= leading
        # The last line was drawn at baseline (y + leading) - ascent, because
        # the loop draws at (y - ascent) and then steps y down by `leading`.
        last_baseline = y + leading - ascent
        widest = max(self.c.stringWidth(ln, FONT_MONO, size) for ln in lines)
        pad = 2.0
        box = Box(
            x - pad,
            last_baseline + descent - pad,
            widest + 2 * pad,
            (y_top - last_baseline - descent) + 2 * pad,
            "code",
        )
        self.boxes.append(box)
        return box


def _metrics(font: str, size: float) -> tuple[float, float]:
    """(ascent, descent) in points for a font at a size.

    ReportLab exposes these on the font face in 1/1000 em units; the canvas has
    no public accessor, so read the face directly.
    """
    face = pdfmetrics.getFont(font).face
    return face.ascent * size / 1000.0, face.descent * size / 1000.0


def _new_canvas(pagesize) -> tuple[canvas.Canvas, io.BytesIO]:
    buf = io.BytesIO()
    return canvas.Canvas(buf, pagesize=pagesize), buf


# ---- templates ------------------------------------------------------------
# Each returns (bytes, Layout). The seed drives every random choice, so a
# (template, seed) pair is fully reproducible.


def template_single_column(rng: np.random.Generator) -> Layout:
    c, buf = _new_canvas(LETTER)
    L = Layout(c, buf, *LETTER)
    m = L.margin
    y = L.page_h - m
    L.text_line(m, y - 9, f"SYNTHETIC · forgesight-synth v1 · seed {int(rng.integers(1000,9999))}", 7.5, "page_header")
    y -= 26
    L.text_line(m, y, f"{textgen.sentence(rng, 4).rstrip('.')}", 17, "title", bold=True)
    y -= 30
    for s in range(3):
        L.text_line(m, y, f"{s + 1}. {textgen.sentence(rng, 3).rstrip('.')}", 11, "section_header", bold=True)
        y -= 16
        L.text_block(m, y, LETTER[0] - 2 * m, textgen.sentence(rng, 55), 9, 11.5)
        y -= 96
    y -= 8
    L.code_block(m, y, textgen.pseudo_code(rng, 6), 7.5, 9.5)
    y -= 64
    L.text_line(m, 34, textgen.sentence(rng, 12), 7, "footnote")
    L.text_line(m, 22, "SYNTHETIC DATA — not a real document", 7.5, "page_footer")
    c.showPage()
    c.save()
    return L


def template_two_column(rng: np.random.Generator) -> Layout:
    c, buf = _new_canvas(LETTER)
    L = Layout(c, buf, *LETTER)
    m = L.margin
    y = L.page_h - m
    L.text_line(m, y - 9, "SYNTHETIC · forgesight-synth v1 · two-column", 7.5, "page_header")
    y -= 26
    L.text_line(m, y, f"{textgen.sentence(rng, 4).rstrip('.')}", 16, "title", bold=True)
    y -= 28
    col_w = (LETTER[0] - 3 * m) / 2
    gutter = m
    top = y

    # Left column
    L.text_line(m, y, "1. " + textgen.sentence(rng, 2).rstrip("."), 10.5, "section_header", bold=True)
    L.text_block(m, y - 16, col_w, textgen.sentence(rng, 40), 8.5, 10.5)
    L.table(m, top - 200, col_w, 5, 3, rng)
    L.text_line(m, top - 206, "Table " + str(int(rng.integers(1, 9))) + ": synthetic rows", 7, "caption")

    # Right column
    rx = m + col_w + gutter
    L.text_line(rx, y, "2. " + textgen.sentence(rng, 2).rstrip("."), 10.5, "section_header", bold=True)
    L.picture(rx, top - 190, col_w, 150, rng)
    L.text_line(rx, top - 196, "Fig. " + str(int(rng.integers(1, 9))) + ": synthetic trace", 7, "caption")
    L.list_block(rx, top - 215, col_w, [textgen.words(rng, 3) for _ in range(5)], 8.5, 11)

    L.text_line(m, 22, "SYNTHETIC DATA — not a real document", 7.5, "page_footer")
    c.showPage()
    c.save()
    return L


def template_slide(rng: np.random.Generator) -> Layout:
    """16:9 slide-like page: wide, few boxes, large picture."""
    pw, ph = landscape(LETTER)
    c, buf = _new_canvas((pw, ph))
    L = Layout(c, buf, pw, ph)
    m = 40.0
    y = ph - m
    L.text_line(m, y - 10, f"SLIDE · synthetic · seed {int(rng.integers(1000,9999))}", 8, "page_header")
    y -= 40
    L.text_line(m, y, textgen.sentence(rng, 4).rstrip("."), 24, "title", bold=True)
    y -= 34
    L.text_block(m, y, pw * 0.42, textgen.sentence(rng, 30), 10, 12.5)
    L.picture(pw * 0.50, ph - m - 240, pw * 0.46, 210, rng)
    y -= 210
    L.list_block(m, y, pw * 0.42, [textgen.words(rng, 3) for _ in range(4)], 10, 13)
    L.text_line(m, 24, "SYNTHETIC DATA — not a real document", 8, "page_footer")
    c.showPage()
    c.save()
    return L


def template_two_up(rng: np.random.Generator) -> Layout:
    """Two A5-ish pages side by side on one landscape sheet.

    This is the tiling case from design §8.8: one long side, aspect 1.41, two
    visually distinct content regions.
    """
    pw, ph = landscape(LETTER)
    c, buf = _new_canvas((pw, ph))
    L = Layout(c, buf, pw, ph)
    m = 30.0
    y = ph - m
    L.text_line(m, y - 9, "SYNTHETIC · two-up spread", 7.5, "page_header")
    y -= 22
    half = (pw - 3 * m) / 2
    for side, x in enumerate((m, m + half + m)):
        L.text_line(x, y, f"{textgen.sentence(rng, 3).rstrip('.')}", 13, "title", bold=True)
        yy = y - 20
        L.text_line(x, yy, "1. " + textgen.sentence(rng, 2).rstrip("."), 9.5, "section_header", bold=True)
        yy -= 14
        L.text_block(x, yy, half, textgen.sentence(rng, 22), 8, 9.5)
        yy -= 80
        if side == 0:
            L.table(x, yy, half, 4, 3, rng)
        else:
            L.picture(x, yy, half, 90, rng)
    L.text_line(m, 18, "SYNTHETIC DATA — not a real document", 7.5, "page_footer")
    c.showPage()
    c.save()
    return L


TEMPLATES = {
    "single_column": template_single_column,
    "two_column": template_two_column,
    "slide": template_slide,
    "two_up": template_two_up,
}
