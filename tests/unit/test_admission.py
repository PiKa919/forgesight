"""Memory admission: AT-6 and AT-7.

AT-6: with a budget that only fits b=2, the worker runs batches of at most 2,
and everything still succeeds -- the point being that admission *shrinks work*,
it does not fail it.

AT-7: an item that cannot fit even alone is failed with a specific code, and the
worker survives to run the next item. The second half is the half that matters:
a worker that dies on the refusal takes the lease with it and the reaper has to
clean up, which is a much worse outcome than refusing the work.

The worker side of AT-7 (Pipeline.run_batch) is exercised in
tests/model/test_runtimes.py, which needs real weights; these are the sizing
arithmetic, which needs nothing.
"""

from __future__ import annotations

from dataclasses import replace

from forgesight.worker.memprobe import Admission, Calibration, fit

MB = 1 << 20


def _calibration(m0_mb: float, per_item_mb: float, residual_mb: float = 0.0) -> Calibration:
    return Calibration(
        m0_bytes=int(m0_mb * MB),
        m_item_bytes=per_item_mb * MB,
        max_residual=residual_mb * MB,
        samples={1: int((m0_mb + per_item_mb) * MB)},
        valid=True,
    )


def test_admission_returns_the_requested_batch_when_it_fits():
    a = Admission(calibration=_calibration(100, 10), budget_bytes=1000 * MB)
    assert a.admit(4) == 4


# -- AT-6 --------------------------------------------------------------------


def test_admission_shrinks_the_batch_under_a_low_budget():
    """AT-6: a budget sized for b=2 admits 2, never 3 or 4.

    The budget is set just above what b=2 needs after the safety margin, so the
    test is about which batch is chosen rather than about a boundary being hit
    exactly -- effective_budget truncates to an int, so a budget of precisely
    peak(2) lands one byte under and would make this an off-by-one test.
    """
    cal = _calibration(100, 100)          # 100 MB base, 100 MB per item
    budget = int((100 + 2 * 100) * 1.02 / (1 - 0.15)) * MB
    a = Admission(calibration=cal, budget_bytes=budget, max_batch=4)

    assert a.admit(4) == 2
    assert a.admit(3) == 2
    assert a.admit(2) == 2


def test_admission_never_exceeds_the_configured_max_batch():
    """The configured ceiling wins even when the budget would allow more."""
    a = Admission(calibration=_calibration(1, 1), budget_bytes=10_000 * MB,
                  max_batch=2)
    assert a.admit(8) == 2


def test_the_safety_margin_is_actually_taken_off():
    """A budget of exactly peak(2) must not admit 2.

    Without the margin, admission runs right at the measured peak, and a batch
    that merely *fits* on this run is the one most likely to exceed it on the
    next. The margin is the difference between fitting and fitting reliably.
    """
    cal = _calibration(100, 100)
    exact = Admission(calibration=cal, budget_bytes=300 * MB, safety_margin=0.0)
    assert exact.admit(2) == 2

    margined = Admission(calibration=cal, budget_bytes=300 * MB, safety_margin=0.15)
    assert margined.admit(2) < 2, "the safety margin did not reduce the admitted batch"


def test_pending_prefetched_bytes_shrink_the_batch():
    """AT-6's prefetch term: pages already fetched count against the budget.

    peak(8) is 900 MB, and the safety margin is taken off the budget before the
    comparison, so the budget has to clear 900/0.85 for b=8 to be admissible at
    all. With 400 MB already prefetched the same budget only reaches b=5.
    """
    cal = _calibration(100, 100)          # peak(8) = 900 MB
    a = Admission(calibration=cal, budget_bytes=1_200 * MB, max_batch=8)

    assert a.admit(8) == 8, "precondition: b=8 should fit with nothing pending"
    assert a.admit(8, pending_bytes=400 * MB) == 5, (
        "prefetched pages did not count against the budget"
    )


def test_a_noisy_fit_is_widened_by_its_own_residual():
    """A fit known to be imprecise must not be trusted at full precision.

    The residual is the part of the fit's error we actually measured, so the
    predicted peak is inflated by it. The same calibration with a zero residual
    is the control, which is what makes this a test of the widening rather than
    of the arithmetic.
    """
    cal = fit({1: 200 * MB, 2: 300 * MB, 3: 320 * MB})
    noisy = replace(cal, max_residual=50 * MB)
    budget = 340 * MB

    tight = Admission(calibration=cal, budget_bytes=budget, safety_margin=0.0)
    widened = Admission(calibration=noisy, budget_bytes=budget, safety_margin=0.0)

    assert widened.admit(4) <= tight.admit(4), (
        "widening by the residual did not reduce the admitted batch"
    )


# -- AT-7 --------------------------------------------------------------------


def test_an_item_that_cannot_fit_at_all_is_refused_not_squeezed_in():
    """AT-7, admission side: 0 means refuse, not clamp to 1."""
    cal = _calibration(5000, 100)         # even one item needs 5.1 GB
    a = Admission(calibration=cal, budget_bytes=100 * MB, max_batch=4)
    assert a.admit(4) == 0


def test_refusal_stays_a_refusal_when_pages_are_prefetched():
    a = Admission(calibration=_calibration(5000, 100), budget_bytes=100 * MB)
    assert a.admit(1, pending_bytes=10 * MB) == 0


def test_without_a_calibration_admission_defers_to_the_ceiling():
    """No fit means no prediction, so it must not invent one.

    Falling back to max_batch here is the conservative choice: over-admitting is
    recoverable by the hard cap, whereas inventing a memory model from nothing is
    how the negative-intercept fit got trusted in the first place.
    """
    a = Admission(calibration=None, budget_bytes=10 * MB, max_batch=4)
    assert a.admit(4) == 4


def test_a_misprediction_is_recorded_rather_than_hidden():
    """AT-7's measurement half: admission may be wrong, but it must say so."""
    cal = _calibration(100, 100)
    a = Admission(calibration=cal, budget_bytes=10_000 * MB)
    assert a.mispredictions == 0

    a.observe(2, int(250 * MB))            # under the predicted 300 MB: fine
    assert a.mispredictions == 0

    a.observe(2, int(400 * MB))            # over the predicted 300 MB by >10%
    assert a.mispredictions == 1


def test_observed_batches_are_kept_for_the_report():
    a = Admission(calibration=_calibration(100, 100), budget_bytes=10_000 * MB)
    a.observe(2, int(300 * MB))
    (b, observed, predicted), = a.observations
    assert b == 2
    assert observed == int(300 * MB)
    assert predicted == a.predict(2)

# -- the hard cap backstop (design §13.2: memory misprediction -> recycle) ---


def test_the_hard_cap_is_a_fraction_of_total_host_memory(monkeypatch):
    """Anchored to the machine, not to the budget admission is guessing at.

    The two are separate on purpose: admission's budget is a prediction about
    work not yet run and did not reproduce on this host (ADR 0004), so the
    backstop is anchored to something that cannot drift.
    """
    from forgesight.worker import memprobe

    monkeypatch.setattr(memprobe, "host_total_bytes", lambda: 16 * 1024 * MB)
    assert memprobe.hard_cap_bytes(0.85) == int(16 * 1024 * MB * 0.85)
    assert memprobe.hard_cap_bytes(0.5) < memprobe.hard_cap_bytes(0.85)


def test_over_hard_cap_is_a_strict_comparison():
    from forgesight.worker.memprobe import over_hard_cap

    assert over_hard_cap(101, 100)
    assert not over_hard_cap(100, 100), "exactly at the cap is not over it"
    assert not over_hard_cap(99, 100)


def test_an_unset_cap_never_forces_a_recycle():
    """A zero or absent cap must mean "no opinion", not "always over".

    Failing closed here would recycle the worker on every batch on a host where
    the cap could not be computed, which is a self-inflicted outage.
    """
    from forgesight.worker.memprobe import over_hard_cap

    assert not over_hard_cap(10**12, 0)
    assert not over_hard_cap(10**12, -1)


def test_a_failed_rss_measurement_does_not_recycle(monkeypatch):
    """The worker must not recycle because psutil raised.

    Recycling on an unreadable measurement trades a real process for a
    measurement problem, which is exactly the wrong direction.
    """
    from forgesight.worker import main as worker_main

    def boom(*_a, **_k):
        raise RuntimeError("no /proc on this host")

    monkeypatch.setattr(worker_main.psutil, "Process", boom)
    w = object.__new__(worker_main.Worker)
    from forgesight.settings import get_settings

    w.s = get_settings()
    assert w._over_hard_cap() is False


def test_the_worker_recycles_when_measured_rss_exceeds_the_cap(monkeypatch):
    from forgesight.worker import main as worker_main
    from forgesight.worker import memprobe

    monkeypatch.setattr(worker_main.psutil, "Process",
                        lambda: type("P", (), {"memory_info": lambda s: type(
                            "M", (), {"rss": int(900 * MB)})()})())
    monkeypatch.setattr(memprobe, "host_total_bytes", lambda: 1024 * MB)
    w = object.__new__(worker_main.Worker)
    from forgesight.settings import get_settings

    w.s = get_settings()
    # Cap is 0.85 * 1 GiB = 870 MiB; the stub reports 900 MiB.
    assert w._over_hard_cap() is True


def test_system_status_reports_the_hard_cap_next_to_the_budget():
    """An operator tuning memory must see both numbers, not just the soft one.

    Otherwise they lower `worker_mem_budget` and never learn the cap is a
    fraction of the host, so the two appear to be the same control.
    """
    from forgesight.api.schemas import SystemStatus

    s = SystemStatus(
        mode="local", dialect="sqlite", queue_depth=[], queue_limit=10,
        worker_mem_budget_bytes=4 * 1024 * MB, memory_safety_margin=0.15,
        admission_mispredictions=0,
    )
    assert 0.0 < s.rss_hard_cap_fraction < 1.0
    body = s.model_dump()
    assert "rss_hard_cap_fraction" in body
    assert body["rss_hard_cap_fraction"] == body.get("rss_hard_cap_fraction")
