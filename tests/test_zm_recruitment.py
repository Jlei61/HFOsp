"""Local recruitment must separate sequential spread from simultaneous ignition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_recruitment import (  # noqa: E402
    axial_lag, bin_baseline, bin_rate_traces, local_recruitment, spatial_bins)


def _traveling(n_bins=20, n_steps=4000, dt=0.1, speed_bins_per_ms=0.1, base=10.0, hi=200.0):
    traces = np.full((n_bins, n_steps), base, np.float32)
    for b in range(n_bins):
        onset = int(round((b / speed_bins_per_ms) / dt))
        if onset < n_steps:
            traces[b, onset:] = hi
    return traces


def _simultaneous(n_bins=20, n_steps=4000, base=10.0, hi=200.0, onset_step=1000):
    traces = np.full((n_bins, n_steps), base, np.float32)
    traces[:, onset_step:] = hi
    return traces


def test_bins_partition_every_neuron_exactly_once():
    rng = np.random.default_rng(0)
    positions = rng.random((5000, 2)) * 20.0
    out = spatial_bins(positions, bin_mm=1.0, sheet_l_mm=20.0)
    assert out["bin_index"].shape == (5000,)
    assert out["bin_index"].min() >= 0
    assert out["bin_counts"].sum() == 5000
    assert out["bin_xy_mm"].shape == (out["bin_counts"].size, 2)


def test_traveling_wave_gives_a_long_spread_duration():
    traces = _traveling()
    thresholds = np.full(traces.shape[0], 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 1.0
    assert 120.0 <= out["spread_10_90_ms"] <= 200.0
    assert out["first_recruited_bin"] == 0


def test_simultaneous_ignition_gives_a_near_zero_spread_duration():
    traces = _simultaneous()
    thresholds = np.full(traces.shape[0], 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 1.0
    assert out["spread_10_90_ms"] <= 1.0


def test_a_brief_blip_shorter_than_the_persistence_floor_is_not_recruitment():
    traces = np.full((5, 4000), 10.0, np.float32)
    traces[:, 1000:1050] = 200.0          # 5 ms, below the 15 ms floor
    thresholds = np.full(5, 100.0)
    out = local_recruitment(traces, thresholds, dt_ms=0.1,
                            search_window_steps=4000, minimum_persistence_ms=15.0)
    assert out["recruited_fraction"] == 0.0
    assert np.all(np.isnan(out["recruitment_step"]))


def test_threshold_is_each_bin_s_own_baseline_not_a_global_one():
    traces = np.zeros((2, 2000), np.float32)
    traces[0] = 5.0
    traces[1] = 500.0                     # a permanently busy bin
    thresholds = bin_baseline(traces, dt_ms=0.1, window_ms=(0.0, 200.0), quantile=0.99)
    assert thresholds[1] > thresholds[0] * 10.0


def test_axial_lag_recovers_a_planted_axial_gradient():
    xy = np.stack([np.linspace(0, 19, 20), np.zeros(20)], axis=-1)
    recruitment_step = np.arange(20, dtype=float) * 100.0     # 10 ms per mm at dt=0.1
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["axial_slope_ms_per_mm"], 10.0, atol=0.5)
    assert abs(out["axial_r"]) > 0.99


def test_symmetric_offaxis_spread_is_not_cancelled_by_a_signed_coordinate():
    """The decisive off-axis regression: bins spread symmetrically on BOTH sides
    of the axis. A signed perpendicular coordinate averages the two sides to a
    slope near zero; the absolute distance recovers the real 8 ms/mm."""
    offsets = np.concatenate([np.arange(1.0, 11.0), -np.arange(1.0, 11.0)])
    xy = np.stack([np.zeros(20), offsets], axis=-1)
    recruitment_step = np.abs(offsets) * 80.0                 # 8 ms per mm at dt=0.1
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["offaxial_slope_ms_per_mm"], 8.0, atol=0.5)
    assert abs(out["offaxial_r"]) > 0.99
    assert abs(out["axial_slope_ms_per_mm"]) < 1e-6


def test_offaxis_slope_is_not_confounded_by_the_axial_gradient():
    rng = np.random.default_rng(9)
    along = rng.uniform(-8.0, 8.0, 60)
    perp = rng.uniform(-6.0, 6.0, 60)
    xy = np.stack([along, perp], axis=-1)
    recruitment_step = along * 50.0 + np.abs(perp) * 20.0     # 5 and 2 ms/mm
    out = axial_lag(recruitment_step, xy, dt_ms=0.1,
                    axis_unit=np.array([1.0, 0.0]), origin_xy=np.zeros(2))
    assert np.isclose(out["axial_slope_ms_per_mm"], 5.0, atol=0.3)
    assert np.isclose(out["offaxial_slope_ms_per_mm"], 2.0, atol=0.3)


def test_bin_rate_traces_are_per_neuron_hz():
    spikes = np.zeros((1000, 4), bool)
    spikes[::10, :] = True                # one spike per ms = 1000 Hz per neuron
    bin_index = np.zeros(4, int)
    traces = bin_rate_traces(spikes, bin_index, 1, dt_ms=0.1, kernel_ms=5.0)
    assert traces.shape == (1, 1000)
    assert np.isclose(traces[0, 200:800].mean(), 1000.0, rtol=0.05)
