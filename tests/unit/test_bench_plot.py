"""The admission predicted-vs-observed figure.

Design §15 phase 4 lists "admission predicted-vs-observed plot generated" as an
exit criterion, and it did not exist. These tests cover the part that can go
wrong silently: the plot must not hide the finding.

The finding is that the fit has a negative intercept and the measured marginal
peaks are mostly zero. A chart with a truncated y-axis, or one that clips
negative values, would render that as an ordinary-looking graph and defeat the
point of generating it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from forgesight.bench.plot import Series, _nice_ticks, predicted_vs_observed_svg

# The real shape from data/reports/report-2026-10-05.json: the measured marginal
# peaks are ~0 for b=1,2,4 and jump at b=8, and the fitted intercept is negative.
REAL = Series(
    label="egret-medium / onnxruntime",
    samples={1: 81_920, 2: 0, 4: 0, 8: 128_909_312},
    m0=-39_183_405.0,
    m_item=19_050_000.0,
    valid=False,
)


def test_it_is_wellformed_svg():
    svg = predicted_vs_observed_svg([REAL])
    # Parses as XML: a malformed chart is worse than no chart, because a broken
    # image in a tracked report is easy to miss.
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")


def test_it_is_deterministic():
    """The figure is tracked in git, so it must not churn between runs."""
    assert predicted_vs_observed_svg([REAL]) == predicted_vs_observed_svg([REAL])


def test_the_negative_intercept_is_not_clipped_away():
    """A truncated axis would hide the entire finding.

    The fit's intercept is below zero, which is what makes it invalid. Clamping
    the axis at zero would draw a dashed line starting at the floor and make the
    claim look ordinary.

    Asserted on the rendered y-axis tick labels rather than by scraping
    coordinates out of the `points` attribute. An earlier version regexed the
    polyline for `[\d.]+` before a comma, which cannot match a leading minus --
    so the assertion passed for the wrong reason on a chart that did clip. The
    axis labels are what a reader actually sees, and a negative one on the scale
    is direct evidence the range was not clamped.
    """
    svg = predicted_vs_observed_svg([REAL])
    ticks = re.findall(r'text-anchor="end">(-?[\d.]+[KM]?)</text>', svg)
    assert ticks, "no y-axis ticks were rendered"
    assert any(t.startswith("-") for t in ticks), (
        f"no negative tick on the y-axis; the negative intercept was clipped "
        f"(ticks={ticks})"
    )


def test_the_zero_marginal_peaks_are_plotted_not_skipped():
    """Three of the four samples are zero, and those zeros are the result.

    Filtering them would produce a tidy rising line through three points and
    hide the fact that batch size 1 through 4 cost nothing measurable.
    """
    svg = predicted_vs_observed_svg([REAL])
    titles = re.findall(r"<title>([^<]+)</title>", svg)
    assert len(titles) == len(REAL.samples), (
        f"expected one marker per sample, got {len(titles)} for {len(REAL.samples)}"
    )
    # b=2 and b=4 measured exactly zero. b=1 measured 81920 bytes, which is not
    # zero and must not be labelled as such: an earlier version rounded to whole
    # megabytes and rendered all three as "0 MB", collapsing a real 82 KB sample
    # into the very thing the chart is about.
    zeros = [t for t in titles if t.endswith("measured 0")]
    assert len(zeros) == 2, f"expected exactly the two true zeros, got {zeros}"
    tiny = [t for t in titles if "measured 80K" in t]
    assert tiny, f"the 82 KB sample was not labelled in KB; titles={titles}"


def test_an_invalid_fit_is_drawn_dashed_and_labelled():
    svg = predicted_vs_observed_svg([REAL])
    assert "stroke-dasharray" in svg, "an unusable fit is not marked as unusable"
    assert "not usable" in svg, "the figure does not say the fit is unusable"


def test_a_valid_fit_is_not_marked_unusable():
    ok = Series(label="x", samples={1: 10, 2: 20, 4: 40}, m0=5, m_item=10, valid=True)
    svg = predicted_vs_observed_svg([ok])
    assert "not usable" not in svg


def test_a_single_sample_still_renders():
    """One point has no line to join, so the polyline must degrade to markers."""
    svg = predicted_vs_observed_svg([Series(label="one", samples={2: 5},
                                            m0=1, m_item=2, valid=True)])
    assert "<circle" in svg
    ET.fromstring(svg)


def test_multiple_series_are_distinguished():
    a = Series(label="torch", samples={1: 10, 2: 20}, m0=5, m_item=8, valid=True)
    b = Series(label="ort", samples={1: 12, 2: 22}, m0=6, m_item=8, valid=True)
    svg = predicted_vs_observed_svg([a, b])
    assert "torch" in svg and "ort" in svg
    ET.fromstring(svg)


def test_empty_series_is_refused():
    """Better to fail than to emit a chart with no axes labelled."""
    with pytest.raises(ValueError, match="no series"):
        predicted_vs_observed_svg([])


def test_ticks_cover_the_range_and_are_ordered():
    ticks = _nice_ticks(0, 100)
    assert ticks == sorted(ticks)
    assert ticks[0] <= 0 and ticks[-1] >= 100


def test_ticks_survive_a_degenerate_range():
    """All samples identical: the axis must still render rather than divide by zero."""
    flat = Series(label="flat", samples={1: 5, 2: 5}, m0=5, m_item=0, valid=True)
    svg = predicted_vs_observed_svg([flat])
    ET.fromstring(svg)


def test_no_nan_or_inf_reaches_the_output():
    """A NaN coordinate silently drops the whole shape in most renderers."""
    for s in (REAL, Series(label="z", samples={1: 0}, m0=0, m_item=0, valid=True)):
        svg = predicted_vs_observed_svg([s])
        assert "nan" not in svg.lower()
        assert "inf" not in svg.lower()