# ADR 0004: the assumed memory model does not reproduce on this host

- **Status:** accepted; the design's model is marked invalid at runtime
- **Date:** 2026-10-05

## Context

Design §8.4 assumes a linear memory model:

> Calibration: fit `peak(b) = M0 + b·m_item` by least squares. … Admission rule:
> before each micro-batch, pick the largest `b ≤ B` with
> `M0 + b·m_item + prefetch_bytes ≤ budget`.

The reason for the model, and the reason it is a fit rather than a table, is
sound: admission has to size work for a batch size it has never actually run. A
lookup table cannot predict an unmeasured point.

## The measurement

The first calibration run produced this:

| quantity | value |
|---|---|
| steady state M0 | **−39.20 MB** |
| marginal per item | 19.05 MB |
| max fit residual | 37.01 MB |

| batch size | marginal peak above baseline | high-water including baseline |
|---|---|---|
| b=1 | 0.08 MB | 1237.20 MB |
| b=2 | 0.00 MB | 1237.10 MB |
| b=4 | 0.00 MB | 1237.10 MB |
| b=8 | 128.91 MB | 1366.00 MB |

Three of the four marginal figures are zero, and the fit that runs through them
has a negative intercept.

## Why

RSS is a **high-water mark**. Neither CPython nor PyTorch nor ONNX Runtime
returns freed memory to the operating system in any general way. So peak RSS is
monotonic in the largest batch size seen so far: once a batch of 8 has run, the
peak for a batch of 1 is simply the peak that the batch of 8 already set.

Measuring batch sizes in ascending order against a cold baseline therefore does
not measure the cost of a batch. It measures a staircase — the first measurement
of each shape keeps every earlier high-water mark — and no straight line fits a
staircase whose first three rungs are flat.

The design's model implicitly assumes memory is released as the batch shrinks. It
is not, and the negative intercept is the arithmetic saying so out loud.

## Decision

1. **The baseline is taken after a warm pass at every batch size**, so each
   measurement is the genuinely marginal cost of one more input. Both the
   baseline-relative and the high-water numbers are reported, so the difference
   between them is visible rather than hidden by a choice of baseline.
2. **A fit with a negative intercept is marked invalid.** `Calibration.valid` is
   `False`, with the reason recorded.
3. **Admission falls back to the largest measured peak** when the fit is invalid.
   That is conservative: it assumes more memory than the fit would have, so it
   under-admits rather than over-admits.
4. **The report prints the finding**, in a block that says the model did not
   reproduce and that a controller trusting it would be operating on fiction.

## Consequences

- The memory model as specified does not describe this host. Anyone reusing the
  fit needs to re-calibrate and check `valid` first; a `peak()` that silently
  returns a negative number would admit work that then fails.
- Batch sizes 1–4 appear to cost nothing in peak RSS on this host, and only b=8
  grows. That is a real and useful observation about a 16 GiB laptop CPU, and it
  is the opposite of the intuition the design was written from.
- The more interesting result is what the fallback implies. Admission is
  currently doing very little on this host, because 4 GiB of budget is far above
  what a batch of 4 at 640×640 actually needs. The machinery is exercised by
  AT-6 and AT-7 rather than by ordinary operation, which is worth knowing before
  relying on it in production.
