# Benchmark protocol

Version 1. Every number in every report comes from a run of this procedure, and
each report embeds the environment the run happened in.

## Preconditions

The runner refuses by default when any of these hold:

- running on **battery** power
- **Low Power Mode** on
- free memory below **25%**

`--allow-unfavourable` takes a labelled measurement anyway. The report then
carries the refusal reason and says the numbers should not be compared with a
nominal run. This is a deliberate trade: on a machine that cannot be put on AC
power, a labelled measurement is more useful than nothing and an unlabelled one
is worse than either.

## Environment capture

Recorded with every run, and printed at the top of every report:

CPU brand and the performance/efficiency core split, physical and logical core
counts, total memory, macOS version, architecture, Python version, and the
versions of numpy, torch, torchvision, transformers, onnxruntime, onnx, Pillow,
OpenCV, reportlab, pypdfium2, pycocotools, psycopg and FastAPI. Plus the thread
environment variables, power source, Low Power Mode state, memory pressure
immediately before and after, the git sha, and a host fingerprint built from the
performance-relevant fields.

**Two runs are only ever compared when their manifests agree** on CPU, core
layout, memory, OS, architecture and library versions. A refusal to compare
states which field differed.

## Procedure

1. Native processes, one worker per configuration, no other workers running.
   The API and the database are on the same host, and that is stated in the
   report rather than left to be guessed.
2. **Warmup:** 20 pages, discarded. The first inference of a model pays for lazy
   initialisation, allocator growth and first-touch faults; counting it makes a
   fast configuration look slow.
3. **Closed-loop throughput:** 200 pages of `synth-bench` in manifest order,
   reported as pages/s over the measured window.
4. **Open-loop latency:** Poisson arrivals at 0.5×, 0.8× and 0.95× of measured
   closed-loop capacity, 300 s per rate, from a seeded arrival schedule.
5. **Repetition:** 5 trials. Configuration order is **randomised per trial** from
   a seed, with a cooldown between configurations, so a slow drift over the run
   cannot systematically favour whichever configuration went first.
6. **Memory:** RSS sampled every 5 ms by a dedicated thread, because the
   inference call blocks the main thread for hundreds of milliseconds and a peak
   sampled only between inferences misses the peak entirely. `ru_maxrss` is
   normalised between macOS (bytes) and Linux (KiB).
7. **Statistics:** per configuration, the median of trial medians with the
   min–max range, and a 95% bootstrap interval over the p95 with a fixed seed so
   the interval is reproducible.
8. **Raw samples** go to `data/reports/raw/`, one JSON file per trial. Every
   reported number is read from one of those files.

## Stability rule

A configuration is marked **unstable** and excluded from every comparison when
either:

- the coefficient of variation of its trial throughput exceeds **5%**, or
- throughput drifts more than **10%** from the first trial to the last.

The report prints the reasons. An unstable measurement quoted as a result is
worse than not quoting it, so the comparison sentences refuse to mention it.

## Latency definitions

Both are always reported, always labelled, never collapsed into one number:

| term | definition | includes the queue wait |
|---|---|---|
| queue-inclusive | `received → persisted` | **yes** |
| service | `claimed → persisted` | no |
| batch wait | `preprocess done → inference start` | n/a |
| preprocess / infer / postprocess / persist | in-process monotonic clock | n/a |

Inference time is attributed to every item in a batch and is never divided
across them.

## Quality metrics

- COCO mAP@[.5:.95] and mAP@.5, per-class AP, from `pycocotools`, with detections
  scored from 0.05. The arithmetic is pinned against hand-computed values in
  `tests/unit/test_eval.py` (AT-19); a metric nobody has checked against a known
  answer is exactly the component that quietly inflates a report.
- The operating threshold is chosen to maximise F1 at IoU 0.5 **on the
  calibration split only**, then frozen. Selecting it on the pages the gate is
  measured on would be measuring the threshold search as if it were model
  quality.
- **Detection agreement** between two runs on the same page:
  `2·matched / (|a| + |b|)`, where a match is the same class at IoU ≥ 0.9, greedily
  in descending score order. Greedy rather than optimal, because it is stable
  under a change in the number of detections, which an optimal matching is not.
- **Raw-output parity**, max and mean absolute difference of logits and boxes
  between runtimes on an identical input tensor, which isolates the runtime from
  preprocessing.

## Configuration matrix

**Tier 1**, 12 configurations: {heron, egret-medium} × {torch fp32, ORT fp32, ORT
int8} × batch {1, 4}, 4 threads, reference preprocessing, no tiling.

**Tier 2**, if time allows: batch {2, 8}, 8 threads, the `cv2_linear` and `cv2_area`
preprocessing variants, and tiling on versus off for the two-up subset.

## Report rules

Every number states its unit, its configuration, its statistic and its n.

A speedup sentence is emitted only between stable configurations measured in
the same run with the same environment manifest, and always carries its interval.
If a comparison is not permitted, the report says so rather than omitting it
silently — a reader can tell the difference between "no difference was found" and
"no comparison was made".

No number is carried over from a model card, a paper or an issue report.
