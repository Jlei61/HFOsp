import numpy as np

from src.topic4_runaway_morphology import (
    classify_sustained_runaway,
    contact_oscillation_metrics,
    population_rate_frequency_metrics,
    rolling_full_field_recruitment,
    summarize_runaway_morphology,
)


def test_full_field_recruitment_distinguishes_global_from_local_activity():
    positions = np.asarray([(x + 0.5, y + 0.5) for x in range(10) for y in range(10)])
    spikes = np.zeros((1000, 100), bool)
    spikes[500::10, :] = True
    out = rolling_full_field_recruitment(
        spikes, positions, dt_ms=1.0, sheet_l_mm=10.0,
        window_ms=20.0, stride_ms=5.0)
    post = out["time_ms"] >= 520.0
    assert np.allclose(out["active_neuron_fraction"][post], 1.0)
    assert np.allclose(out["recruited_spatial_fraction"][post], 1.0)

    local = spikes.copy()
    local[:, 10:] = False
    local_out = rolling_full_field_recruitment(
        local, positions, dt_ms=1.0, sheet_l_mm=10.0,
        window_ms=20.0, stride_ms=5.0)
    assert np.max(local_out["active_neuron_fraction"][post]) == 0.1
    assert np.max(local_out["recruited_spatial_fraction"][post]) == 0.1


def test_persistent_high_amplitude_oscillation_beats_pre_state():
    dt_ms = 1.0
    time = np.arange(1000) * dt_ms / 1000.0
    carrier = np.sin(2.0 * np.pi * 40.0 * time)
    trace = np.column_stack([carrier, 0.8 * carrier])
    fast = np.sin(2.0 * np.pi * 90.0 * time[500:])
    trace[500:, 0] = 5.0 * fast
    trace[500:, 1] = 4.0 * fast
    out = contact_oscillation_metrics(
        trace, dt_ms=dt_ms, onset_ms=500.0, pre_ms=400.0, post_ms=400.0,
        band_hz=(30.0, 120.0))
    assert out["median_band_rms_ratio_post_over_pre"] > 3.5
    assert out["contact_fraction_high_for_half_post_window"] == 1.0
    assert out["median_post_high_envelope_duty"] > 0.95
    assert out["median_peak_frequency_pre_hz"] == 40.0
    assert out["median_peak_frequency_post_hz"] == 90.0
    assert out["median_peak_frequency_shift_hz"] == 50.0

    population_rate = np.where(
        time < 0.5,
        40.0 + 10.0 * np.sin(2.0 * np.pi * 40.0 * time),
        130.0 + 30.0 * np.sin(2.0 * np.pi * 90.0 * time),
    )
    population = population_rate_frequency_metrics(
        population_rate, dt_ms=dt_ms, onset_ms=500.0,
        pre_ms=400.0, post_ms=400.0)
    assert population["peak_frequency_pre_hz"] == 40.0
    assert population["peak_frequency_post_hz"] == 90.0
    assert population["spectral_centroid_shift_hz"] > 0


def test_classification_requires_broad_sustained_and_faster_state():
    summary = {
        "full_field_recruitment": {
            "q05_active_neuron_fraction_20ms": 0.7,
            "q05_recruited_spatial_fraction_1mm": 0.8,
        },
        "contact_oscillation": {
            "median_post_high_envelope_duty": 0.9,
            "contact_fraction_high_for_half_post_window": 0.9,
            "median_band_rms_ratio_post_over_pre": 3.0,
        },
        "population_rate_frequency": {
            "frequency_resolution_hz": 2.0,
            "spectral_centroid_shift_hz": 20.0,
            "median_rate_ratio_post_over_pre": 3.0,
        },
    }
    accepted = classify_sustained_runaway(summary)
    assert accepted["all_checks_pass"]
    summary["population_rate_frequency"]["spectral_centroid_shift_hz"] = -1.0
    rejected = classify_sustained_runaway(summary)
    assert not rejected["all_checks_pass"]
    assert not rejected["checks"]["population_frequency_increased"]


def test_summary_requires_post_onset_recruitment_samples():
    recruitment = {
        "time_ms": np.asarray([10.0, 20.0]),
        "active_neuron_fraction": np.asarray([0.1, 0.2]),
        "recruited_spatial_fraction": np.asarray([0.1, 0.2]),
    }
    with np.testing.assert_raises_regex(ValueError, "no post-onset"):
        summarize_runaway_morphology(recruitment, {}, onset_ms=30.0)


def test_contact_frequency_rejects_truncated_post_window():
    trace = np.zeros((750, 2), float)
    with np.testing.assert_raises_regex(ValueError, "windows are incomplete"):
        contact_oscillation_metrics(
            trace, dt_ms=1.0, onset_ms=500.0, pre_ms=400.0, post_ms=400.0,
            band_hz=(30.0, 120.0))
