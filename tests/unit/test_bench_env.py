"""Benchmark environment refusals: AT-21.

`check_runnable` carries the docstring "Design §12.1 and acceptance test AT-21"
and no such test existed. The reason is structural rather than an oversight:
it read the power source and memory pressure through module-level helpers that
nothing could substitute, so there was no way to ask it about a battery without
one.

These tests patch those helpers, which is the seam that was missing.
"""

from __future__ import annotations

import pytest

from forgesight.bench import env_capture as ec
from forgesight.bench.env_capture import Refusal, check_runnable, comparable


@pytest.fixture
def ac(monkeypatch):
    """A healthy machine: plugged in, Low Power Mode off, memory free."""
    monkeypatch.setattr(ec, "_power", lambda: ("ac", False))
    monkeypatch.setattr(ec, "_memory_pressure", lambda: 0.80)
    return True


def test_a_healthy_machine_is_allowed_to_measure(ac):
    assert check_runnable() is None


def test_on_battery_is_refused_with_an_actionable_reason(monkeypatch):
    monkeypatch.setattr(ec, "_power", lambda: ("battery", False))
    monkeypatch.setattr(ec, "_memory_pressure", lambda: 0.90)
    r = check_runnable()
    assert isinstance(r, Refusal)
    assert r.reason == "on_battery"
    # The detail has to say what to do, or the refusal just blocks someone.
    assert "connect AC" in r.detail, r.detail


def test_low_power_mode_is_refused_even_on_ac(monkeypatch):
    monkeypatch.setattr(ec, "_power", lambda: ("ac", True))
    monkeypatch.setattr(ec, "_memory_pressure", lambda: 0.90)
    r = check_runnable()
    assert isinstance(r, Refusal)
    assert r.reason == "low_power_mode"


def test_memory_pressure_is_refused_with_both_numbers_in_the_reason(monkeypatch):
    monkeypatch.setattr(ec, "_power", lambda: ("ac", False))
    monkeypatch.setattr(ec, "_memory_pressure", lambda: 0.10)
    r = check_runnable()
    assert isinstance(r, Refusal)
    assert r.reason == "memory_pressure"
    assert "10.0%" in r.detail and "25%" in r.detail, r.detail


def test_a_refusal_can_be_waived_explicitly(ac, monkeypatch):
    """`require_ac=False` is the deliberate opt-out, not an accident.

    Used by the recorded battery/Low-Power-Mode run in data/reports/, which is
    why it has to keep working: the guard exists to stop an unlabelled
    measurement, and the override produces a labelled one.
    """
    monkeypatch.setattr(ec, "_power", lambda: ("battery", True))
    assert check_runnable() is not None, "precondition: battery should be refused"
    assert check_runnable(require_ac=False) is None


# -- the third clause of AT-21: a mismatched manifest refuses the comparison ---


def _manifest(**over):
    base = {
        "cpu_brand": "Apple M5",
        "physical_cores": 10,
        "total_memory_bytes": 16 << 30,
        "os": "macOS 27.0.1",
        "arch": "arm64",
        "library_versions": {"torch": "2.14.1", "onnxruntime": "1.30.0"},
    }
    base.update(over)
    return base


def test_two_runs_on_the_same_host_are_comparable():
    assert comparable(_manifest(), _manifest()) == []


def test_a_different_cpu_refuses_the_comparison():
    diffs = comparable(_manifest(), _manifest(cpu_brand="Intel Xeon"))
    assert "cpu_brand" in diffs


def test_a_different_library_version_refuses_the_comparison():
    """The case that matters: same machine, upgraded torch.

    Silently comparing across a version bump would produce a speedup number
    that belongs to neither configuration.
    """
    diffs = comparable(_manifest(), _manifest(library_versions={
        "torch": "2.15.0", "onnxruntime": "1.30.0"}))
    assert "library:torch" in diffs, diffs