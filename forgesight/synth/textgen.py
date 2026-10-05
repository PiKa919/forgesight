"""Seeded pseudo-English word source (design §9.1).

No copyrighted text ships in this repo. Sentences are assembled from a small
in-repo vocabulary with a seeded PRNG, which is enough to give the detector
realistic line lengths, word shapes and ragged right edges without importing
any third-party corpus.
"""

from __future__ import annotations

import numpy as np

NOUNS = ["system", "process", "memory", "model", "runtime", "worker", "queue", "batch", "page", "tensor", "kernel", "latency", "throughput", "cache", "buffer", "tensor", "stream", "schema", "record", "ledger", "release", "candidate", "profile", "dataset", "manifest", "threshold", "calibration", "admission", "pipeline", "segmentation", "detection", "layout", "document", "section", "paragraph", "header", "footer", "caption", "footnote", "formula", "table", "figure", "column", "region", "boundary", "vector", "matrix", "tensor", "signal", "sample", "window", "trace", "counter", "gauge", "sensor", "frame", "packet", "request", "response", "payload", "artifact", "manifest", "revision", "commit", "branch", "merge", "pull", "review", "gate", "budget"]

VERBS = ["measure", "compute", "allocate", "release", "reclaim", "persist", "enqueue", "dequeue", "claim", "renew", "expire", "retry", "recover", "render", "preprocess", "postprocess", "quantize", "export", "import", "verify", "record", "report", "compare", "score", "rank", "sample", "batch", "stream", "decode", "encode", "recover", "reclaim", "drain", "compact", "shard", "replicate", "reconcile", "promote", "rollback", "verify", "observe", "predict", "classify", "detect", "segment", "cluster", "embed"]

ADJECTIVES = ["bounded", "durable", "fenced", "pinned", "immutable", "content-addressed", "idempotent", "isolated", "bounded", "deterministic", "reproducible", "calibrated", "staggered", "throttled", "observable", "replayable", "resumable", "idempotent", "sparse", "dense", "nested", "static", "dynamic", "stable", "unstable", "legacy", "modern", "primary", "secondary", "tertiary"]

ADVERBS = ["safely", "quickly", "slowly", "exactly", "nearly", "always", "never", "often", "rarely", "deterministically", "probabilistically", "measurably", "repeatably", "efficiently"]

MODES = ("assert", "return", "yield", "raise", "await", "break", "continue")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sentence(rng: np.random.Generator, n_words: int | None = None) -> str:
    n = n_words or int(rng.integers(6, 20))
    words: list[str] = []
    for i in range(n):
        r = rng.random()
        if i == 0:
            words.append(str(rng.choice(ADJECTIVES)))
        elif r < 0.22:
            words.append(str(rng.choice(ADVERBS)))
        elif r < 0.45:
            words.append(str(rng.choice(VERBS)))
        else:
            words.append(str(rng.choice(NOUNS)))
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


def words(rng: np.random.Generator, n: int) -> str:
    return " ".join(str(rng.choice(NOUNS)) for _ in range(n))


def pseudo_code(rng: np.random.Generator, n_lines: int) -> list[str]:
    """Short, syntactically plausible code lines for the `code` class."""
    out = []
    for _ in range(n_lines):
        style = rng.random()
        v = str(rng.choice(NOUNS))
        n2 = int(rng.integers(1, 999))
        if style < 0.3:
            out.append(f"def {v}_{n2}() -> int:")
        elif style < 0.55:
            out.append(f"    {v} = {n2}  # {rng.choice(ADJECTIVES)}")
        elif style < 0.75:
            out.append(f"    for i in range({n2}):")
        elif style < 0.9:
            out.append(f"    return {v}_{n2}({n2})")
        else:
            out.append(f"    {rng.choice(MODES)} {v}_{n2}")
    return out
