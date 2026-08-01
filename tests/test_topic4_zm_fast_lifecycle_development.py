"""Focused contracts for the development-first dynamic-threshold screen."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_zm_fast_lifecycle_development.py"
SPEC = importlib.util.spec_from_file_location("zm_fast_lifecycle_development", SCRIPT)
DEV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEV)


def test_delta_phi_uses_seconds_and_locked_phase_c_rate():
    expected = 0.30 * 6.5 / (0.100 * DEV.REFERENCE_RATE_HZ)
    assert DEV.delta_phi_mV(100.0, 0.30) == pytest.approx(expected)


def test_initial_panel_is_exactly_six_unique_dynamic_threshold_points():
    values = {
        (tau, fraction): DEV.delta_phi_mV(tau, fraction)
        for tau in DEV.TAUS_MS
        for fraction in DEV.FRACTIONS
    }
    assert len(values) == 6
    assert len(set(values.values())) == 6
    for tau in DEV.TAUS_MS:
        assert values[tau, 0.30] == pytest.approx(2.0 * values[tau, 0.15])


@pytest.mark.parametrize(
    "tau,fraction,gap,rate",
    [
        (0.0, 0.30, 6.5, DEV.REFERENCE_RATE_HZ),
        (100.0, 0.0, 6.5, DEV.REFERENCE_RATE_HZ),
        (100.0, 1.0, 6.5, DEV.REFERENCE_RATE_HZ),
        (100.0, 0.30, 0.0, DEV.REFERENCE_RATE_HZ),
        (100.0, 0.30, 6.5, 0.0),
    ],
)
def test_delta_phi_rejects_nonphysical_inputs(tau, fraction, gap, rate):
    with pytest.raises(ValueError):
        DEV.delta_phi_mV(tau, fraction, gap_mV=gap, rate_hz=rate)


def test_state_tags_are_the_locked_four_phase_c_checkpoints():
    assert DEV.STATES == (
        "bounded_mid__rising",
        "bounded_mid__peak",
        "bounded_late__rising",
        "bounded_late__peak",
    )
    assert DEV._state_parts("bounded_late__peak") == ("bounded_late", "peak")
    with pytest.raises(ValueError):
        DEV._state_parts("unregistered__peak")
