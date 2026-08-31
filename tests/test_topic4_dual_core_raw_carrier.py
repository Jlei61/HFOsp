import numpy as np
import pytest

from scripts.aggregate_topic4_dual_core_carrier_kinetics import _finite_summary

from src.topic4_dual_core_carrier import (
    arrays_equal_with_nan,
    baseline_mask_from_events,
    bin_continuous_trace,
    binned_group_rates_hz,
    dual_core_region_masks,
    event_window_indices,
    raw_population_burst_summary,
)


def test_finite_summary_ignores_missing_values_without_emitting_nan():
    assert _finite_summary([None, np.nan]) is None
    assert _finite_summary([None, 0.2, np.nan, 0.4]) == pytest.approx(0.3)
    assert _finite_summary([None, 20.0, 10.0], reducer=np.median) == 15.0


def test_exact_parity_accepts_aligned_nan_but_not_finite_drift():
    left = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    right = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    assert arrays_equal_with_nan(left, right)
    right[-1] = np.nextafter(right[-1], np.float32(np.inf))
    assert not arrays_equal_with_nan(left, right)


def test_dual_core_region_masks_form_disjoint_partition():
    positions = np.array([[0, 0], [1.5, 0], [3, 0], [10, 0], [8.5, 0], [7, 0]])
    masks, names = dual_core_region_masks(
        positions, np.array([[0, 0], [10, 0]]),
        core_radius_mm=1.0, annulus_outer_radius_mm=2.0,
    )
    assert names == ["core_1", "core_2", "annulus_1", "annulus_2", "background"]
    assert np.all(masks.sum(axis=1) == 1)
    assert np.array_equal(masks.sum(axis=0), [1, 1, 1, 1, 2])


def test_binned_group_rates_are_unsmoothed_and_per_neuron():
    spikes = np.zeros((20, 4), bool)
    spikes[[0, 10], 0] = True
    spikes[[1, 11], 1] = True
    masks = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    rates, time_ms, sizes = binned_group_rates_hz(
        spikes, masks, dt_ms=0.1, bin_ms=1.0,
    )
    assert np.array_equal(sizes, [2, 2])
    assert np.array_equal(time_ms, [0.5, 1.5])
    assert np.array_equal(rates[:, 0], [1000.0, 1000.0])
    assert np.array_equal(rates[:, 1], [0.0, 0.0])


def test_continuous_trace_binning_only_averages_within_bin():
    trace = np.arange(20, dtype=float).reshape(10, 2)
    binned, time_ms = bin_continuous_trace(trace, dt_ms=0.1, bin_ms=0.5)
    assert np.array_equal(time_ms, [0.25, 0.75])
    assert np.array_equal(binned, [[4, 5], [14, 15]])


def test_event_windows_do_not_pad_incomplete_events():
    indices, valid = event_window_indices(
        [10.0, 100.0], trace_length=200, bin_ms=1.0,
        before_ms=20.0, after_ms=40.0,
    )
    assert not valid[0]
    assert valid[1]
    assert indices.shape == (2, 60)


def test_raw_three_cycle_detector_rejects_one_pulse_and_accepts_native_cycles():
    baseline = np.zeros(1000)
    one_pulse = np.zeros(256)
    one_pulse[100] = 50.0
    rejected = raw_population_burst_summary(
        one_pulse, bin_ms=1.0, baseline_values=baseline,
    )
    assert rejected["raw_peak_count"] == 1
    assert not rejected["regular_three_cycle_burst"]
    assert not rejected["population_three_cycle_burst"]

    cycles = np.zeros(256)
    cycles[[80, 96, 112, 128]] = 50.0
    accepted = raw_population_burst_summary(
        cycles, bin_ms=1.0, baseline_values=baseline,
    )
    assert accepted["raw_peak_count"] == 4
    assert accepted["regular_three_cycle_burst"]
    assert accepted["population_three_cycle_burst"]
    assert accepted["raw_peak_interval_frequency_hz"] == 62.5


def test_timing_only_cycles_do_not_count_as_population_carrier():
    signal = np.zeros(256)
    signal[[80, 96, 112, 128]] = 4.0
    result = raw_population_burst_summary(
        signal, bin_ms=1.0, baseline_values=np.zeros(512),
    )
    assert result["regular_three_cycle_burst"]
    assert not result["population_three_cycle_burst"]


def test_baseline_mask_excludes_event_and_guard_intervals():
    times = np.arange(1000, dtype=float)
    keep = baseline_mask_from_events(
        times, [{"t_on_ms": 400.0, "t_off_ms": 450.0}],
        guard_before_ms=100.0, guard_after_ms=200.0,
    )
    assert keep[299]
    assert not keep[300]
    assert not keep[650]
    assert keep[651]
