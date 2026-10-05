"""Release gates (design §10).

Every gate compares against a **pinned reference**, not against whatever is
currently active. That distinction is the whole point: comparing to the active
release would let a series of individually permitted regressions accumulate
without anything ever crossing a line. There is no ratcheting, so the reference
must be fixed.

Each verdict records the observed value, the threshold and whether it passed, so
a report can show *why* a candidate was refused rather than only that it was.
Performance is reported, never gated: a faster configuration that loses quality
does not ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgesight.settings import Settings
from forgesight.vision.types import Candidate

# Policy thresholds. Every one of these is a decision, not a measurement.
GATE_POLICY: dict[str, dict[str, Any]] = {
    "G1_quality_clean": {"metric": "map", "dataset": "synth-clean/test",
                         "max_drop": 0.010, "label": "mAP@[.5:.95] drop vs reference <= 1.0 pt"},
    "G2_quality_shift": {"metric": "map", "dataset": "synth-scan",
                         "max_drop": 0.020, "label": "mAP@[.5:.95] drop on scanned pages <= 2.0 pt"},
    "G3_critical_classes": {"metric": "per_class_ap", "dataset": "synth-clean/test",
                            "classes": ("table", "picture", "text", "title"),
                            "max_drop": 0.030, "label": "critical-class AP drop <= 3.0 pt"},
    "G4_runtime_parity": {"metric": "agreement", "min": 0.98,
                          "label": "detection agreement with the same model on another runtime >= 0.98"},
    "G5_memory_fit": {"metric": "peak_bytes", "max": "worker_budget",
                      "label": "calibrated peak RSS fits the worker memory budget"},
    "G6_provenance": {"metric": "provenance", "label": "artifact sha verified, dataset hashes and code sha recorded"},
}

CRITICAL_CLASSES = ("table", "picture", "text", "title")


@dataclass(slots=True)
class EvaluationResult:
    candidate_name: str
    metrics: dict[str, Any]
    verdicts: list[dict[str, Any]]
    dataset_versions: list[dict[str, str]] = field(default_factory=list)
    env_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v["passed"] for v in self.verdicts)


def _verdict(gate: str, passed: bool, observed: Any, threshold: Any, detail: str = "") -> dict:
    return {
        "gate": gate,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "detail": detail or GATE_POLICY[gate]["label"],
    }


def judge(
    candidate: Candidate,
    reference: Candidate,
    clean: dict,
    shift: dict,
    agreement: float | None,
    peak_bytes: int | None,
    budget_bytes: int,
    provenance_ok: bool,
    provenance_detail: str = "",
) -> list[dict]:
    """Apply G1-G6. `clean`/`shift` are reference and candidate metric dicts."""
    out: list[dict] = []

    ref_map = clean["reference"]["map"]
    cand_map = clean["candidate"]["map"]
    drop = ref_map - cand_map
    out.append(_verdict("G1_quality_clean", drop <= GATE_POLICY["G1_quality_clean"]["max_drop"],
                        round(cand_map, 5), ref_map,
                        f"candidate {cand_map:.5f} vs reference {ref_map:.5f} "
                        f"(drop {drop:+.5f})"))

    ref_s = shift["reference"]["map"]
    cand_s = shift["candidate"]["map"]
    drop_s = ref_s - cand_s
    out.append(_verdict("G2_quality_shift", drop_s <= GATE_POLICY["G2_quality_shift"]["max_drop"],
                        round(cand_s, 5), ref_s,
                        f"candidate {cand_s:.5f} vs reference {ref_s:.5f} "
                        f"(drop {drop_s:+.5f})"))

    ref_pc = clean["reference"].get("per_class_ap", {})
    cand_pc = clean["candidate"].get("per_class_ap", {})
    worst: list[tuple[str, float]] = []
    for cls in CRITICAL_CLASSES:
        if cls not in ref_pc:
            # The reference never found it either, so there is nothing to hold
            # the candidate to.
            continue
        if cls not in cand_pc:
            # The reference found it and the candidate did not: a total loss for
            # that class, which is the largest possible drop. Expressed as a
            # positive drop like every other entry, so the comparison below is
            # the same sign for every case.
            worst.append((cls, ref_pc[cls]))
            continue
        # A drop is reference minus candidate, so a loss is a positive number.
        # Sign matters: computing it the other way round makes every
        # regression look like an improvement and the gate never fires.
        worst.append((cls, ref_pc[cls] - cand_pc[cls]))
    ok_g3 = all(drop <= GATE_POLICY["G3_critical_classes"]["max_drop"] for _, drop in worst)
    out.append(_verdict(
        "G3_critical_classes", ok_g3,
        {c: round(d, 5) for c, d in worst}, GATE_POLICY["G3_critical_classes"]["max_drop"],
        "; ".join(f"{c} -{d:.5f}" for c, d in worst if d > 0) or "no critical class dropped",
    ))

    if agreement is None:
        out.append(_verdict("G4_runtime_parity", True, None, None,
                            "not applicable: the candidate runs the reference's own "
                            "runtime and preprocessing, so there is nothing to compare "
                            "against. Reported, not gated."))
    else:
        out.append(_verdict("G4_runtime_parity",
                            agreement >= GATE_POLICY["G4_runtime_parity"]["min"],
                            round(agreement, 5),
                            GATE_POLICY["G4_runtime_parity"]["min"],
                            f"detection agreement {agreement:.5f}"))

    if peak_bytes is None:
        out.append(_verdict("G5_memory_fit", True, None, None,
                            "not calibrated yet; reported, not gated"))
    else:
        out.append(_verdict("G5_memory_fit", peak_bytes <= budget_bytes, peak_bytes,
                            budget_bytes,
                            f"calibrated peak {peak_bytes / 1e6:.1f} MB of a "
                            f"{budget_bytes / 1e6:.1f} MB budget"))

    out.append(_verdict("G6_provenance", provenance_ok,
                        "ok" if provenance_ok else "failed", None,
                        provenance_detail or "artifact sha, dataset hashes and code sha recorded"))
    return out


def negative_control_blocked(verdicts: list[dict]) -> bool:
    """AT-13: a deliberately degraded candidate must be refused.

    Checked explicitly rather than inferred, because a gate suite that stops
    blocking bad candidates is worse than one that never existed: it would let
    them through silently.
    """
    by_name = {v["gate"]: v for v in verdicts}
    g1 = by_name.get("G1_quality_clean")
    g2 = by_name.get("G2_quality_shift")
    return bool(g1 and g2 and not g1["passed"] and not g2["passed"])


def summarize(verdicts: list[dict]) -> str:
    failed = [v["gate"] for v in verdicts if not v["passed"]]
    if not failed:
        return "all gates passed"
    return "blocked by " + ", ".join(failed)


def evaluate_candidate(candidate_row, settings: Settings, pool=None) -> EvaluationResult:
    """Evaluate a stored candidate row against the pinned reference.

    Raises rather than returning a passing result when inputs are missing, so an
    unevaluated candidate can never look gate-passing. The pool is passed in by
    the caller that owns the connection rather than opened here.
    """
    from forgesight.eval.runner import run_evaluation

    return run_evaluation(candidate_row, settings, pool=pool)
