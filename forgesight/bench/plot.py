"""Admission predicted-vs-observed plot, as a dependency-free SVG.

Design §15 phase 4 asks for "admission predicted-vs-observed plot generated".
This is it, and it is worth having rather than ceremonial: on this host the fit
is invalid, and the plot is what makes that legible instead of a claim.

Why hand-written SVG rather than matplotlib:

  * matplotlib is a ~30 MB dependency for one chart, and this project pins its
    whole dependency set on purpose.
  * SVG is text, so the plot is diffable in git, deterministic byte-for-byte
    across runs, and reviewable in a pull request. A PNG is neither.
  * it renders directly in GitHub markdown, so the figure is visible in the
    README without a build step.

The axes are linear and the figure deliberately keeps the *measured* zero values
rather than clipping them: those zeros are the finding. A plot that starts its
y-axis above zero would hide the exact thing it exists to show.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MB = 1e6


@dataclass(frozen=True, slots=True)
class Series:
    """One candidate configuration's calibration, as a plotter needs it."""

    label: str
    samples: dict[int, float]          # batch size -> measured marginal peak, bytes
    m0: float                          # fitted intercept, bytes
    m_item: float                      # fitted slope, bytes per item
    valid: bool


def _nice_ticks(lo: float, hi: float, want: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, want)
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if raw <= step:
            break
    start = step * int(lo / step)
    if start > lo:
        start -= step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt_bytes(v: float) -> str:
    """Adaptive units, because the interesting values here are sub-megabyte.

    Rounding to whole megabytes made an 82 KB sample label as "0 MB" -- visually
    identical to a genuine zero, which is precisely the distinction the figure
    exists to show. Below 1 MB the axis switches to KB.
    """
    a = abs(v)
    if a >= MB:
        return f"{v / MB:.0f}"
    if a >= 1024:
        return f"{v / 1024:.0f}K"
    return f"{v:.0f}"


def predicted_vs_observed_svg(
    series: list[Series],
    *,
    width: int = 720,
    height: int = 420,
) -> str:
    """Render predicted (fit) against observed (measured) peak RSS per batch size."""
    if not series:
        raise ValueError("no series to plot")

    pad_l, pad_r, pad_t, pad_b = 66, 20, 44, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # Scales, with the intercept included so a negative fit is visible rather
    # than clipped away. It is allowed to reach below the axis origin.
    all_b: list[int] = []
    ys: list[float] = []
    for s in series:
        all_b += list(s.samples)
        ys += list(s.samples.values())
        ys.append(s.m0)
        ys += [s.m0 + b * s.m_item for b in s.samples]
    b_lo, b_hi = 0, max(all_b) * 1.08
    y_lo, y_hi = min([*ys, 0.0]), max(ys)
    if y_hi - y_lo <= 0:
        # A calibration that measured zero everywhere, with a zero-intercept
        # zero-slope fit, collapses the range to a single value and every
        # projection divides by zero. Give the axis a floor so the figure
        # degrades to an empty but well-formed chart instead of raising.
        # Reachable, not hypothetical: this is what an all-zero sample set is.
        pad = max(abs(y_hi), 1.0)
        y_lo, y_hi = y_lo - pad, y_hi + pad

    # Snap the y-axis out to whole tick boundaries.
    #
    # Without this the axis is clipped to the data range, and a tick that falls
    # outside it is simply dropped. With data running to -39 MB and a 50 MB step,
    # the only negative tick is -50 MB -- outside the range, so dropped -- and the
    # axis appears to start at zero even though the plot area extends below it.
    # That is visually indistinguishable from having clipped the negative
    # intercept, which is the one thing this figure must not do.
    y_ticks = _nice_ticks(y_lo, y_hi)
    y_lo = min([y_lo, *y_ticks])
    y_hi = max([y_hi, *y_ticks])

    def sx(b: float) -> float:
        return pad_l + (b - b_lo) / (b_hi - b_lo) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="24" font-size="14" font-weight="600" fill="#111">'
        "Admission: predicted vs observed peak RSS</text>",
        f'<text x="{pad_l}" y="40" font-size="11" fill="#666">'
        "measured marginal peak per batch size; the fit is what admission plans on"
        "</text>",
    ]

    # Axes and grid.
    for v in y_ticks:
        y = sy(v)
        if not (pad_t - 1 <= y <= pad_t + plot_h + 1):
            continue
        out.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="10" fill="#666" '
            f'text-anchor="end">{_fmt_bytes(v)}</text>'
        )
    for b in sorted({*all_b}):
        x = sx(b)
        out.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" font-size="10" fill="#666" '
            f'text-anchor="middle">{b}</text>'
        )
    out.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="#999" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{pad_l}" y1="{sy(0):.1f}" x2="{pad_l + plot_w}" '
        f'y2="{sy(0):.1f}" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    out.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 14}" font-size="11" fill="#333" '
        f'text-anchor="middle">batch size</text>'
    )
    out.append(
        f'<text x="16" y="{pad_t + plot_h / 2}" font-size="11" fill="#333" '
        f'text-anchor="middle" transform="rotate(-90 16 {pad_t + plot_h / 2})">'
        "peak RSS (MB)</text>"
    )

    colours = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    for i, s in enumerate(series):
        c = colours[i % len(colours)]
        # Fitted line, extended across the observed range.
        xs = [b_lo, b_hi]
        ys_line = [s.m0 + b * s.m_item for b in xs]
        dash = "" if s.valid else ' stroke-dasharray="6 4"'
        out.append(
            f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys_line, strict=True))}" '
            f'fill="none" stroke="{c}" stroke-width="1.5" opacity="0.75"{dash} />'
        )
        # Measured points.
        pts = sorted(s.samples.items())
        if len(pts) > 1:
            out.append(
                f'<polyline points="{" ".join(f"{sx(b):.1f},{sy(v):.1f}" for b, v in pts)}" '
                f'fill="none" stroke="{c}" stroke-width="2" />'
            )
        for b, v in pts:
            out.append(
                f'<circle cx="{sx(b):.1f}" cy="{sy(v):.1f}" r="4" fill="{c}" '
                f'stroke="#fff" stroke-width="1.5"/>'
            )
            out.append(
                f'<title>{s.label} b={b}: measured {_fmt_bytes(v)}</title>'
            )
        # Legend entry, drawn only for the observed series.
        ly = pad_t + 14 + i * 15
        out.append(
            f'<line x1="{pad_l + plot_w - 150}" y1="{ly}" x2="{pad_l + plot_w - 130}" '
            f'y2="{ly}" stroke="{c}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{pad_l + plot_w - 124}" y="{ly + 4}" font-size="10" fill="#333">'
            f"{s.label}</text>"
        )

    invalid = [s for s in series if not s.valid]
    if invalid:
        out.append(
            f'<text x="{pad_l}" y="{height - 30}" font-size="11" fill="#b91c1c">'
            "dashed = fit, which is not usable: negative intercept</text>"
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


__all__ = ["Series", "predicted_vs_observed_svg"]