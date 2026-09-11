from __future__ import annotations

import numpy as np

from scripts.paper_figures.plot_fig5_dual_core_transition_story import (
    _fit_plane_gradient,
    _select_interictal_event,
    _timescale_latency_matrix,
    _spatial_rate_maps,
    _candidate_eta_m,
)


def test_fit_plane_gradient_recovers_direction() -> None:
    x, y = np.meshgrid(np.arange(4.0), np.arange(5.0), indexing="ij")
    values = 2.0 + 3.0 * x - 4.0 * y
    actual = _fit_plane_gradient(values, x, y)
    expected = np.asarray([3.0, -4.0]) / 5.0
    assert np.allclose(actual, expected)


def test_adaptation_current_uses_candidate_gain_not_round_reference():
    manifest={'candidates':[
        {'candidate_id':'other','slow_variables':{'eta_m':.0074}},
        {'candidate_id':'selected','slow_variables':{'eta_m':.0037}}]}
    assert _candidate_eta_m({'candidate_id':'selected'},manifest)==.0037


def test_spatial_rate_uses_producer_yx_order_and_correct_occupancy():
    # Unequal occupancies ensure a transpose cannot cancel in normalization.
    arrays = {
        'positions_E':np.array([[1.,1.],[11.,1.],[12.,1.],[1.,11.]]),
        'transition_spatial_frame_time_ms':np.array([10.,30.,50.]),
        'transition_spatial_spike_count_20ms':np.tile(
            np.array([[1.,4.],[3.,0.]])[None,:,:],(3,1,1)),
        'transition_spatial_bin_mm':10.,
    }
    stages={'interictal_ms':[0.,20.],'pre_onset_ms':[20.,40.],'early_ictal_ms':[40.,60.]}
    maps,centers,occupancy=_spatial_rate_maps(arrays,stages)
    assert np.array_equal(occupancy,np.array([[1.,1.],[2.,0.]]))
    assert np.array_equal(maps[0],np.array([[50.,150.],[100.,0.]]))


def test_select_interictal_event_is_latest_returned_complete_event() -> None:
    meta = {
        "events": [
            {"event_index": 0, "returned": 1, "t_on_ms": 100.0},
            {"event_index": 1, "returned": 1, "t_on_ms": 300.0},
            {"event_index": 2, "returned": 1, "t_on_ms": 500.0},
            {"event_index": 3, "returned": 0, "t_on_ms": 700.0},
        ]
    }
    arrays = {
        "onsets": np.asarray([
            [1.0, 2.0],
            [2.0, 3.0],
            [np.nan, 4.0],
            [5.0, 6.0],
        ])
    }
    selected = _select_interictal_event(meta, arrays)
    assert selected["event_index"] == 1


def test_timescale_latency_matrix_preserves_parameter_axes() -> None:
    summaries = []
    for tau_z in (3000, 5000):
        for tau_m in (250, 1000):
            summaries.append({
                "candidate_id": f"rev21_ts_tz{tau_z}_ta{tau_m}",
                "operational_onset_ms": {
                    "median": float(tau_z + tau_m),
                    "n": 4,
                },
            })
    z, m, latency, count = _timescale_latency_matrix(
        {"candidate_summaries": summaries})
    assert z.tolist() == [3.0, 5.0]
    assert m.tolist() == [0.25, 1.0]
    assert np.allclose(latency, [[3.25, 4.0], [5.25, 6.0]])
    assert np.array_equal(count, np.full((2, 2), 4))
