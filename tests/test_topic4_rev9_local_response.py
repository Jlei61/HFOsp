import numpy as np

from src.topic4_rev9_local_response import (fit_response_slope,
                                            paired_spike_response)


def test_fit_response_slope_recovers_linear_response():
    result = fit_response_slope([0.25, 0.5, 1.0], [1.5, 2.0, 3.0])
    assert np.isclose(result["slope"], 2.0)
    assert np.isclose(result["intercept"], 1.0)
    assert np.isclose(result["r2"], 1.0)


def test_paired_response_separates_source_and_downstream_and_builds_maps():
    positions = np.asarray([[0.5, 0.5], [0.8, 0.5], [2.0, 0.5], [3.0, 0.5]])
    sham = np.zeros((20, 4), bool)
    kick = sham.copy()
    kick[10:12, 0] = True
    kick[12:14, 2] = True
    result = paired_spike_response(
        kick, sham, positions, [0.5, 0.5], [1.0, 0.0],
        dt=1.0, pulse_end_ms=10.0, windows_after_ms=[[0.0, 2.0], [2.0, 4.0]],
        source_radius_mm=0.5, L=4.0, spatial_bins_per_axis=4)
    assert result["n_source"] == 2
    assert result["windows"][0]["source_signed_per_cell"] == 1.0
    assert result["windows"][0]["downstream_signed_per_cell"] == 0.0
    assert result["windows"][1]["source_signed_per_cell"] == 0.0
    assert result["windows"][1]["downstream_signed_per_cell"] == 1.0
    assert result["windows"][0]["positive_map_per_cell"].shape == (4, 4)


def test_paired_response_marks_truncated_window():
    positions = np.asarray([[0.5, 0.5], [2.0, 0.5]])
    spikes = np.zeros((10, 2), bool)
    result = paired_spike_response(
        spikes, spikes, positions, [0.5, 0.5], [1.0, 0.0],
        dt=1.0, pulse_end_ms=8.0, windows_after_ms=[[0.0, 4.0]],
        source_radius_mm=0.2, L=3.0, spatial_bins_per_axis=3)
    assert result["status"] == "truncated"
    assert result["windows"][0]["status"] == "truncated"
