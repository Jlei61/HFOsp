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

ANALYSIS_SCRIPT = ROOT / "scripts/analyze_topic4_zm_fast_lifecycle_development.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_zm_fast_lifecycle_development", ANALYSIS_SCRIPT
)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


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


def test_i2e_target_mapping_caps_only_the_nonphysical_corner():
    capped = DEV.i2e_use_from_target(100.0, 0.35)
    assert capped["U_nominal"] > 1.0
    assert capped["U_applied"] == pytest.approx(0.95)
    assert capped["use_was_capped"] is True
    assert capped["d_star_attainable_at_reference_rate"] > 0.35

    physical = DEV.i2e_use_from_target(600.0, 0.75)
    assert 0.0 < physical["U_applied"] < 0.95
    assert physical["use_was_capped"] is False
    assert physical["d_star_attainable_at_reference_rate"] == pytest.approx(0.75)


def test_i_adaptation_calibration_uses_inhibitory_threshold_gap():
    expected = 0.25 * 7.0 / (0.300 * DEV.RACE_I_REFERENCE_RATE_HZ)
    assert DEV.delta_i_adaptation_mV(300.0, 0.25) == pytest.approx(expected)


def test_control_window_is_shifted_from_branch_relative_to_engine_absolute_time():
    t0, t1 = DEV.control_window_in_engine_time(
        source_t_ms=7350.0,
        relative_t0_ms=2520.0,
        duration_ms=50.0,
    )
    assert t0 == pytest.approx(9870.0)
    assert t1 == pytest.approx(9920.0)
    assert t0 > 7350.0


def test_control_artifact_stem_versions_the_clock_fix_away_from_invalid_runs():
    args = type("Args", (), {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "control_uplift_mV": 1.0,
        "control_target": "all_E", "control_t0_ms": 2520.0,
        "control_duration_ms": 50.0,
    })()
    assert DEV._mechanism_stem(args).endswith("__clkrel2")


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
    assert DEV._state_parts("pre_entry__natural") == ("pre_entry", "natural")
    assert set(DEV.STATES).issubset(DEV.FROZEN_MODE_STATES)
    with pytest.raises(ValueError):
        DEV._state_parts("unregistered__peak")


def test_frozen_mode_stem_carries_the_slow_state_identity():
    args = type("Args", (), {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "control_uplift_mV": 0.0,
        "use_mode_H": True, "rho_mode_H": 0.5,
        "tau_mode_H_ms": 250.0, "m_mode_half": 30.0,
        "freeze_zm": True, "state": "pre_entry__natural",
    })()
    assert DEV._mechanism_stem(args).endswith(
        "__modeH0.5t250__mc30__freeze_pre_entry__natural"
    )


def _subtractive_args(beta):
    return type("Args", (), {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "control_uplift_mV": 0.0,
        "use_mode_H": False, "freeze_zm": True,
        "state": "bounded_late__peak", "beta_SG": beta,
    })()


def test_subtractive_pool_strength_is_part_of_the_artifact_identity():
    """Two subtractive strengths are two arms, not one overwritten artifact."""
    assert "bSG" not in DEV._mechanism_stem(_subtractive_args(0.0))
    weak = DEV._mechanism_stem(_subtractive_args(0.25))
    strong = DEV._mechanism_stem(_subtractive_args(1.0))
    assert "__bSG0.25" in weak and "__bSG1" in strong
    assert weak != strong


def test_subtractive_pool_off_leaves_the_stem_byte_identical_to_prior_runs():
    """beta=0 is the literal pre-change path, so frozen artifacts must not move."""
    args = type("Args", (), {
        "arm": "i2e", "tau_D_ms": 300.7, "d_star": 0.7281,
        "strength_scale": 1.0, "control_uplift_mV": 0.0,
        "use_mode_H": True, "rho_mode_H": 0.5,
        "tau_mode_H_ms": 250.0, "m_mode_half": 30.0,
        "freeze_zm": True, "state": "pre_entry__natural", "beta_SG": 0.0,
    })()
    assert DEV._mechanism_stem(args).endswith(
        "__modeH0.5t250__mc30__freeze_pre_entry__natural"
    )


def test_vseeg_energy_floor_separates_continuous_carrier_from_pulse_train():
    import numpy as np

    fs = 1000.0
    t = np.arange(5000) / fs
    continuous = np.sin(2 * np.pi * 40 * t)[:, None]
    pulsed = continuous.copy()
    gate = np.zeros_like(t, dtype=bool)
    for start in range(0, len(t), 500):
        gate[start:start + 50] = True
    pulsed *= gate[:, None]

    _, continuous_metrics = ANALYSIS._vseeg_energy(continuous, fs)
    _, pulsed_metrics = ANALYSIS._vseeg_energy(pulsed, fs)
    assert continuous_metrics["energy_floor_fraction"] > 0.90
    assert continuous_metrics["energy_gap_fraction"] == 0.0
    assert pulsed_metrics["energy_floor_fraction"] < 1e-12
    assert pulsed_metrics["energy_gap_fraction"] > 0.70


def test_low_variance_surround_makes_correlation_undefined():
    import numpy as np

    out = ANALYSIS._conditional_corr(
        np.linspace(0, 10, 100), np.full(100, 3.0), sigma_min=1.0
    )
    assert out["status"] == "low_variance_undefined"
    assert out["value"] is None


def test_post_entry_spatial_metrics_ignore_entry_flash():
    import numpy as np

    kymo = np.zeros((8, 80), dtype=float)
    kymo[:, :40] = np.arange(8)[:, None] + 1.0
    kymo[3, 40:] = 1.0
    got = ANALYSIS._post_entry_spatial_metrics(
        kymo, bin_ms=25.0, skip_ms=1000.0
    )
    assert got["centroid_excursion_bins"] == 0.0
    assert got["status"] == "spatial_variance_too_low"


def test_pareto_front_keeps_distinct_non_dominated_phenotypes():
    rows = [
        {"energy": 10.0, "motion": 0.0},
        {"energy": 3.0, "motion": 5.0},
        {"energy": 2.0, "motion": 1.0},
    ]
    assert ANALYSIS._pareto_indices(rows, ("energy", "motion")) == [0, 1]
