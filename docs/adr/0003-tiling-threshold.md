# ADR 0003: the tiling threshold does not reach its own test case

- **Status:** accepted, threshold left unchanged
- **Date:** 2026-10-05

## Context

Design §8.8 sets the tiling policy:

> Policy `tile_if(long_side_px > 2200 or aspect > 1.6)`

Design §9.1 then names the tiling experiment in terms of a specific page type:
`synth-bench v1` is specified as "200 pages, **incl. 40 two-up**", and §8.8 says
tiling "may help small classes (footnote, page footer) on **two-up** or large
pages".

Those two statements do not agree.

## The measurement

A two-up spread is a landscape letter page. At the default render DPI of 150:

| Template | DPI | size | long side | aspect | tiled? |
|---|---|---|---|---|---|
| `two_up` | 150 | 1651×1275 | 1651 | 1.29 | no |
| `two_up` | 200 | 2200×1700 | 2200 | 1.29 | no (policy is strict `>`) |
| `two_up` | 220 | 2420×1870 | 2420 | 1.29 | **yes** |

So the page type the design names as the tiling case does **not** tile at the
default render DPI. Its long side is 25% under the pixel threshold and its aspect
is 19% under the aspect threshold.

## Decision

**The threshold is left exactly as the design wrote it.**

The tempting fix is to lower `tile_long_side_px` until the two-up page tiles. That
would be fitting a policy number to make a test pass, and it would be fitting it
to one page size at one DPI — the threshold would then mean nothing as a policy.

Instead:

- The boundary is pinned by
  `tests/model/test_runtimes.py::test_tiling_threshold_boundary_is_where_the_policy_says`,
  which asserts that 150 and 200 DPI do not tile and 220 DPI does. If someone
  later changes the threshold, that test is where the change has to be made
  deliberately.
- The tiling experiment runs at a render DPI that reaches the threshold, and the
  report states the DPI it used.
- The asymmetry clause is tested independently
  (`test_extreme_aspect_ratio_tiles_regardless_of_length`), because a very tall
  page trips the aspect clause with the pixel clause never firing.

## Consequences

- The `synth-bench` tiling experiment is not available at 150 DPI. It is
  available at 220 DPI and above, and the choice is recorded with the result.
- Anyone reading "two-up pages get tiled" from the design text will be wrong at
  the default DPI. This ADR is the correction, and it is short enough to be read.
- A general lesson, which is the reason this is written down rather than just
  fixed: a spec can contain two true statements that do not imply each other, and
  a test written against the prose will pass while the behaviour is wrong. The
  check that found this was a test asserting the *documented* policy, which
  forced the mismatch into the open.
