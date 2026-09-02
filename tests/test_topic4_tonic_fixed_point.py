import numpy as np
import pytest

from src.topic4_tonic_fixed_point import (
    classify_tonic_fixed_point,
    population_rate_modulation,
)


def _rate(mean_hz, depth, frequency_hz, dt_ms=0.1, duration_ms=1000.0, seed=3):
    time = np.arange(0.0, duration_ms, dt_ms)
    rng = np.random.default_rng(seed)
    clean = mean_hz * (1.0 + 0.5 * depth * np.sin(
        2.0 * np.pi * frequency_hz * time / 1000.0))
    return clean + 0.01 * mean_hz * rng.standard_normal(len(time)), time


def test_modulation_depth_recovers_a_known_sinusoidal_swing():
    rate, _ = _rate(400.0, 0.60, 45.0)
    got = population_rate_modulation(rate, dt_ms=0.1)
    assert got["dominant_hz"] == pytest.approx(45.0, abs=2.0)
    assert got["modulation_depth"] == pytest.approx(0.60, rel=0.15)


def test_a_two_percent_ripple_on_a_plateau_is_called_a_tonic_fixed_point():
    """The exact failure mode the LFP spectral clauses cannot see."""
    rate, _ = _rate(400.0, 0.02, 45.0)
    got = population_rate_modulation(rate, dt_ms=0.1)
    assert got["modulation_depth"] < 0.05
    verdict = classify_tonic_fixed_point(
        np.concatenate([np.full(20000, 30.0), rate]), dt_ms=0.1,
        onset_ms=2000.0, post_ms=900.0)
    assert verdict["status"] == "TONIC_HIGH_RATE_FIXED_POINT_WITH_RIPPLE"
    assert verdict["all_checks_pass"] is False


def test_a_deeply_modulated_high_state_passes_criterion_ten():
    rate, _ = _rate(400.0, 0.70, 45.0, duration_ms=1400.0)
    verdict = classify_tonic_fixed_point(
        np.concatenate([np.full(20000, 30.0), rate]), dt_ms=0.1,
        onset_ms=2000.0, post_ms=1000.0)
    assert verdict["status"] == "OSCILLATORY_HIGH_STATE"
    assert verdict["detail"]["high_state"]["modulation_depth"] > 0.2


def test_saturated_activity_is_reported_alongside_the_depth():
    rate, _ = _rate(400.0, 0.02, 45.0, duration_ms=1400.0)
    verdict = classify_tonic_fixed_point(
        np.concatenate([np.full(20000, 30.0), rate]), dt_ms=0.1,
        onset_ms=2000.0, post_ms=1000.0,
        active_fraction_20ms=np.ones(60))
    detail = verdict["detail"]
    assert detail["median_active_E_fraction_20ms_post"] == 1.0
    assert detail["fraction_of_20ms_windows_with_every_E_active"] == 1.0
