"""Contract tests for the per-SNN-run event-diagnostic figure helpers.

The figure (src/sef_hfo_event_figure.py) shows one representative (kick, seed):
raster (cells sorted by distance from the kick) + early per-bin response heatmap +
return-to-quiet trace. Plotting needs in-run spikes; the pure data helpers are pinned here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sef_hfo_event_figure as ef  # noqa: E402


def test_distance_sort_index_orders_by_distance_from_kick():
    posE = np.array([[10.0, 10.0], [0.0, 0.0], [11.0, 10.0], [5.0, 5.0]])
    order = ef.distance_sort_index(posE, kick_center=(10.0, 10.0))
    assert list(order) == [0, 2, 3, 1]   # nearest the kick first


def test_reshape_bins_to_grid_is_row_major_iy_ix():
    # spatial_bins lays bins out as iy*nb+ix (ix fast). grid[iy, ix] must match.
    nb = 3
    vals = np.arange(9)                  # bin index == value
    g = ef.reshape_bins_to_grid(vals, nb)
    assert g.shape == (3, 3)
    assert g[0, 0] == 0 and g[0, 2] == 2 and g[2, 0] == 6 and g[1, 1] == 4


def test_active_fraction_trace_counts_fraction_active_per_bin():
    # 4 time steps, 2 cells; dt=1ms, bin_ms=2 -> 2 bins. cell0 spikes step0, cell1 step3.
    spk = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=bool)
    tr = ef.active_fraction_trace(spk, dt=1.0, bin_ms=2.0)
    assert tr.shape == (2,)
    assert np.isclose(tr[0], 0.5)        # bin0 (steps 0,1): cell0 active -> 1/2
    assert np.isclose(tr[1], 0.5)        # bin1 (steps 2,3): cell1 active -> 1/2


def test_active_fraction_trace_all_silent_is_zero():
    spk = np.zeros((4, 3), dtype=bool)
    assert np.allclose(ef.active_fraction_trace(spk, dt=1.0, bin_ms=2.0), 0.0)


def test_median_representative_picks_seed_closest_to_median():
    # values [4,1,3,9] -> median 3.5; closest is 3 (seed 'c') or 4 (seed 'a'), tie 0.5/0.5
    # use an odd set to make it unambiguous: [1,3,9] -> median 3 -> seed 'b'
    seeds = ["a", "b", "c"]
    vals = [1.0, 3.0, 9.0]
    assert ef.median_representative(seeds, vals) == "b"


def test_median_representative_empty_returns_none():
    assert ef.median_representative([], []) is None
