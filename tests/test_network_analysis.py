from __future__ import annotations

from pathlib import Path

import numpy as np

from src.network_analysis import (
    _apply_physics_constraint,
    _nearest_lags_within_window,
    build_broad_graph,
    build_hfo_network,
    compute_pairwise_association,
    inject_direction,
    load_network_result,
    prune_network,
    save_network_result,
)


def test_build_broad_graph_simpson_dice() -> None:
    coact = np.array(
        [
            [10, 5],
            [5, 20],
        ],
        dtype=np.int64,
    )
    w_simpson = build_broad_graph(coact, method="simpson")
    w_dice = build_broad_graph(coact, method="dice")
    assert np.isclose(w_simpson[0, 1], 0.5)
    assert np.isclose(w_dice[0, 1], 5.0 / 15.0)
    assert np.allclose(np.diag(w_simpson), 0.0)
    assert np.allclose(np.diag(w_dice), 0.0)


def test_nearest_lags_within_window() -> None:
    assert _nearest_lags_within_window(np.array([]), np.array([0.1]), 0.05).size == 0
    src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dst = np.array([1.02, 2.02, 3.02], dtype=np.float64)
    lags = _nearest_lags_within_window(src, dst, 0.021)
    assert lags.shape[0] == 3
    assert np.allclose(lags, np.array([-0.02, -0.02, -0.02]), atol=1e-8)
    assert _nearest_lags_within_window(src, dst, 0.005).size == 0


def test_compute_pairwise_association() -> None:
    starts = {
        "A1": np.array([0.10, 0.20, 0.30], dtype=np.float64),
        "B1": np.array([0.11, 0.21, 0.31], dtype=np.float64),
    }
    out = compute_pairwise_association(starts, ["A1", "B1"], assoc_window_ms=20.0)
    assert out["assoc_count"][0, 1] == 3
    assert out["lag_median_ms"][0, 1] < 0
    assert out["lag_median_ms"][1, 0] > 0


def test_apply_physics_constraint() -> None:
    edge_mask = np.array([[False, True], [False, False]], dtype=bool)
    lag_abs_ms = np.array([[np.inf, 1.0], [np.inf, np.inf]], dtype=np.float64)
    assert np.array_equal(
        _apply_physics_constraint(edge_mask, lag_abs_ms, None),
        edge_mask,
    )
    dist = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
    filtered = _apply_physics_constraint(
        edge_mask,
        lag_abs_ms,
        dist,
        min_dist_mm=10.0,
        lag_vc_ms=3.0,
    )
    assert not filtered[0, 1]


def test_inject_direction() -> None:
    w_pruned = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    lag_raw = np.array(
        [
            np.full(12, 0.010, dtype=np.float64),
            np.full(12, 0.025, dtype=np.float64),
        ]
    )
    events_bool = np.ones((2, 12), dtype=bool)
    adj, direction_mask, edge_stats = inject_direction(
        w_pruned,
        lag_raw,
        events_bool,
        min_events=5,
        lag_thresh_ms=3.0,
        consistency_thresh=0.6,
        p_value_thresh=0.05,
    )
    assert direction_mask[0, 1]
    assert adj[0, 1] > 0
    assert any(es.get("reason") == "directed" for es in edge_stats)


def test_prune_network() -> None:
    w = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.8, 0.0, 0.1],
            [0.2, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    x = np.array([1.0, 0.05, 1.2], dtype=np.float64)  # node 1 fails min_rate
    y = np.array([0.2, 0.3, 0.95], dtype=np.float64)  # node 2 fails max_entropy
    z = np.zeros(3, dtype=np.float64)
    selected_idx, w_pruned, labels = prune_network(
        w,
        x,
        y,
        z,
        min_rate=0.5,
        max_entropy=0.85,
        use_spectral=False,
    )
    assert np.array_equal(selected_idx, np.array([0], dtype=np.int64))
    assert w_pruned.shape == (1, 1)
    assert labels[0] >= 0
    assert labels[1] < 0 and labels[2] < 0


def test_build_hfo_network_integration(
    synthetic_group_analysis_npz: str,
    synthetic_gpu_npz: str,
) -> None:
    dist = np.array(
        [
            [0.0, 20.0, 30.0],
            [20.0, 0.0, 20.0],
            [30.0, 20.0, 0.0],
        ],
        dtype=np.float64,
    )
    result = build_hfo_network(
        synthetic_group_analysis_npz,
        dist,
        detections_npz_path=synthetic_gpu_npz,
        run_surrogate=False,
        min_events=3,
        min_pair_events=3,
        use_spectral=False,
    )
    assert result.n_pool_channels == 3
    assert result.n_selected >= 2
    assert result.adj.shape[0] == result.n_selected
    assert "causal_diagnostics" in result.metrics
    assert "velocity_diagnostics" in result.metrics


def test_save_load_roundtrip(
    synthetic_group_analysis_npz: str,
    synthetic_gpu_npz: str,
    tmp_path: Path,
) -> None:
    result = build_hfo_network(
        synthetic_group_analysis_npz,
        detections_npz_path=synthetic_gpu_npz,
        run_surrogate=False,
        min_events=3,
        min_pair_events=3,
        use_spectral=False,
    )
    out_npz = tmp_path / "network_result.npz"
    save_network_result(result, str(out_npz))
    loaded = load_network_result(str(out_npz))
    assert loaded.node_names == result.node_names
    assert np.allclose(loaded.adj, result.adj)
    assert loaded.metrics["n_edges"] == result.metrics["n_edges"]
