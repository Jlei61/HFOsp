import numpy as np

from src.topic4_global_recruited_oscillation import (
    classify_global_recruited_oscillation,
    contact_rhythm_metrics,
    detect_sustained_high_state_onset,
    fixed_state_contact_rhythm_metrics,
)


def test_scientific_onset_accepts_oscillatory_troughs_but_rejects_short_burst():
    dt = 1.0
    time = np.arange(0.0, 2000.0, dt)
    oscillatory = np.full(len(time), 20.0)
    post = time >= 800.0
    oscillatory[post] = 140.0 + 80.0 * np.sin(
        2.0 * np.pi * 40.0 * time[post] / 1000.0)
    onset = detect_sustained_high_state_onset(oscillatory, dt_ms=dt)
    assert onset == 800.0

    short_burst = np.full(len(time), 20.0)
    short_burst[(time >= 800.0) & (time < 920.0)] = 300.0
    assert detect_sustained_high_state_onset(short_burst, dt_ms=dt) is None


def test_recurring_global_50hz_rhythm_passes_contact_gate():
    dt = 0.5
    time = np.arange(0.0, 2500.0, dt)
    rng = np.random.default_rng(7)
    trace = 0.05 * rng.standard_normal((len(time), 10))
    post = time >= 1000.0
    for contact in range(10):
        phase = 0.08 * contact
        trace[post, contact] += 3.0 * np.sin(
            2.0 * np.pi * 50.0 * time[post] / 1000.0 + phase)
    got = contact_rhythm_metrics(
        trace, dt_ms=dt, onset_ms=1000.0,
        settle_ms=300.0, post_ms=1000.0)
    assert got["contact_fraction_consistently_rhythmic"] == 1.0
    assert np.isclose(got["median_contact_peak_hz"], 50.0, atol=2.1)
    assert got["contact_peak_mad_hz"] <= 2.1


def test_flat_tonic_step_does_not_masquerade_as_rhythm():
    dt = 0.5
    time = np.arange(0.0, 2500.0, dt)
    trace = np.zeros((len(time), 8))
    trace[time >= 1000.0] = 4.0
    got = contact_rhythm_metrics(
        trace, dt_ms=dt, onset_ms=1000.0,
        settle_ms=300.0, post_ms=1000.0)
    assert got["contact_fraction_consistently_rhythmic"] == 0.0


def test_fixed_state_rhythm_finds_stationary_40hz_mode_over_q1_reference():
    dt_ms = 1.0
    time = np.arange(2500, dtype=float) * dt_ms / 1000.0
    reference = np.zeros((len(time), 5), float)
    candidate = np.column_stack([
        3.0 * np.sin(2.0 * np.pi * 40.0 * time + phase)
        for phase in np.linspace(0.0, 0.4, 5)
    ])
    metrics = fixed_state_contact_rhythm_metrics(
        candidate, reference, dt_ms=dt_ms, start_ms=1500.0)
    assert metrics["contact_fraction_consistently_rhythmic"] == 1.0
    assert metrics["median_contact_peak_hz"] == 40.0
    assert metrics["contact_peak_mad_hz"] == 0.0
    assert metrics["median_peak_power_fraction"] > 0.6
    assert metrics["median_band_power_ratio_over_q1_reference"] > 2.0


def test_classifier_accepts_troughs_when_recruitment_duty_and_rhythm_pass():
    rates = {
        "median_pre_hz": 20.0,
        "q95_pre_hz": 70.0,
        "median_post_hz": 180.0,
        "q05_post_hz": 55.0,
        "median_ratio_post_over_pre": 9.0,
    }
    recruitment = {
        "joint_global_recruitment_duty": 0.80,
    }
    rhythm = {
        "contact_fraction_consistently_rhythmic": 0.90,
        "median_contact_peak_hz": 52.0,
        "contact_peak_mad_hz": 4.0,
        "median_peak_power_fraction": 0.45,
        "median_band_power_ratio_post_over_pre": 8.0,
    }
    got = classify_global_recruited_oscillation(
        onset_ms=2400.0, rates=rates, recruitment=recruitment, rhythm=rhythm)
    assert got["all_checks_pass"]
