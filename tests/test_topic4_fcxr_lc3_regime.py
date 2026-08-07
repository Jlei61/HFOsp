"""Tests for the five-regime classifier, above all for the distinction it exists to make.

A carrier and a burst train both oscillate and both read as a modulated high branch.  What tells
them apart is where the troughs sit, and getting that backwards would label a sequence of separate
re-igniting events as one sustained seizure.
"""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc3_regime import (
    classify_regime,
    interictal_ceiling,
    trough_level,
)

BIN = 1.0
DT = 1.0
RUN_MS = 20000.0
ONSET = 5000.0


def _trace(trough_af, peak_af=0.40, interictal_af=0.02, period_ms=100.0, run_ms=RUN_MS):
    """Interictal until onset, then bursts oscillating between trough_af and peak_af."""
    t = np.arange(0, run_ms, BIN)
    rng = np.random.default_rng(0)
    af = np.abs(rng.normal(0, interictal_af / 3.0, t.size))
    up = t >= ONSET
    phase = (t[up] % period_ms) / period_ms
    af[up] = trough_af + (peak_af - trough_af) * (np.cos(2 * np.pi * phase) * 0.5 + 0.5) ** 6
    return af, t


def _rate(af):
    return af * 32000.0 / (BIN / 1000.0) / 32000.0 * 1000.0    # a monotone stand-in for rate_hz


def _call(af, **kw):
    base = dict(af=af, af_bin_ms=BIN, rate_hz=_rate(af), dt_ms=DT,
                baseline_roll_hi_hz=float(np.percentile(_rate(af)[:int(ONSET)], 99)),
                onset_ms=ONSET, offset_ms=RUN_MS, run_ms=RUN_MS,
                terminated=False, recovered=False)
    base.update(kw)
    return classify_regime(**base)


def test_troughs_that_stay_recruited_read_as_a_carrier():
    af, _ = _trace(trough_af=0.18)
    out = _call(af)
    assert out["regime"] == "R3_carrier"
    assert out["carrier"] is True
    assert out["trough_af"] > out["interictal_ceiling_af"]


def test_troughs_that_fall_back_to_interictal_read_as_a_burst_train_not_a_carrier():
    """The distinction the classifier exists for: same oscillation, different troughs."""
    af, _ = _trace(trough_af=0.0)
    out = _call(af)
    assert out["regime"] == "R4_burst_train"
    assert out["carrier"] is False


def test_a_saturated_tonic_branch_is_runaway_not_a_bounded_state():
    af, _ = _trace(trough_af=0.35)
    out = _call(af, refractory_ceiling_fraction=0.7)
    assert out["regime"] == "R1_runaway"
    assert "refractory ceiling" in out["reason"]


def test_numerical_failure_short_circuits_every_other_label():
    af, _ = _trace(trough_af=0.18)
    assert _call(af, numerical_unsafe=True)["regime"] == "R1_runaway"


def test_no_epoch_is_reported_as_interictal_only():
    af, _ = _trace(trough_af=0.18)
    out = _call(af, onset_ms=None)
    assert out["regime"] == "R0_interictal_only"


def test_terminating_and_recovering_outranks_the_shape_it_had_while_up():
    af, _ = _trace(trough_af=0.18)
    out = _call(af, terminated=True, offset_ms=12000.0, recovered=True)
    assert out["regime"] == "R5_closed_loop"


def test_terminating_without_recovery_is_not_called_a_closed_loop():
    af, _ = _trace(trough_af=0.0)
    out = _call(af, terminated=True, offset_ms=12000.0, recovered=False)
    assert out["regime"] != "R5_closed_loop"
    assert out["regime"] == "R4_burst_train"


def test_the_epoch_for_a_terminated_run_stops_at_the_offset_not_the_record_end():
    """Troughs must be read from the bout, not from the quiet tail that follows it."""
    af, t = _trace(trough_af=0.18, run_ms=20000.0)
    af[t >= 12000.0] = 0.001                      # a silent tail after an early offset
    hot = _call(af, terminated=True, offset_ms=12000.0, recovered=False)
    cold = _call(af, terminated=False, offset_ms=RUN_MS)
    assert hot["trough_af"] > cold["trough_af"], "the tail must not be averaged into the troughs"


def test_trough_and_ceiling_helpers_read_the_windows_they_claim():
    af, _ = _trace(trough_af=0.18)
    assert trough_level(af, BIN, ONSET, RUN_MS) > 0.1
    assert interictal_ceiling(af, BIN, ONSET) < 0.05


def test_helpers_return_nan_rather_than_inventing_a_value_for_an_empty_window():
    af, _ = _trace(trough_af=0.18)
    assert np.isnan(trough_level(af, BIN, RUN_MS + 10, RUN_MS + 20))
    assert np.isnan(interictal_ceiling(af, BIN, 0.0))


def test_a_probe_with_no_pre_onset_stretch_must_be_given_its_reference():
    """The failure this guards: an empty pre-onset window silently sets every trough comparison
    to the same answer, so a whole frozen-state map reads as one regime."""
    import pytest
    af, _ = _trace(trough_af=0.18)
    with pytest.raises(ValueError, match="no interictal reference"):
        _call(af, onset_ms=0.0)


def test_an_externally_supplied_reference_is_used_and_decides_the_label():
    af, _ = _trace(trough_af=0.18)
    hot = _call(af, onset_ms=0.0, interictal_ceiling_af=0.05)   # troughs clear it
    cold = _call(af, onset_ms=0.0, interictal_ceiling_af=0.30)  # troughs do not
    assert hot["regime"] == "R3_carrier" and hot["carrier"] is True
    assert cold["regime"] == "R4_burst_train" and cold["carrier"] is False
