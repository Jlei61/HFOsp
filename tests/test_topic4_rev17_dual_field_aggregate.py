from __future__ import annotations

import numpy as np

from scripts.aggregate_topic4_rev17_dual_field_residual_atlas import (
    _configured_network_seeds,
    finite_differences,
    summarize_differences,
)


def _row(candidate, seed, channel=None, mode=None, orientation=None,
         j14=2.0, a=1.0, b=0.5, sa=8.0, sb=9.0):
    return {
        "candidate_id": candidate, "seed": seed,
        "channel": channel, "mode_index": mode, "orientation": orientation,
        "j14": j14, "mode_0_mean": a, "mode_1_mean": b,
        "mode_0_effective_events": sa, "mode_1_effective_events": sb,
    }


def test_central_difference_is_paired_within_network_and_channel():
    rows = []
    for seed in (1, 2, 3):
        rows.extend([
            _row("exact_dual_anchor", seed),
            _row("mean_f00_m_a15", seed, "mean", 0, -1,
                 j14=2.0 + 0.3, a=1.0 + 0.15),
            _row("mean_f00_p_a15", seed, "mean", 0, 1,
                 j14=2.0 - 0.3, a=1.0 - 0.15),
        ])
    differences = finite_differences(rows, 0.15)
    assert len(differences) == 3
    assert all(np.isclose(row["j14_derivative"], -2.0) for row in differences)
    assert all(np.isclose(row["mode_0_mean_derivative"], -1.0) for row in differences)
    assert all(np.isclose(row["j14_curvature"], 0.0) for row in differences)


def test_response_summary_counts_network_signs_without_pooling_events():
    rows = [
        {"channel": "dispersion", "mode_index": 2, "seed": seed,
         "j14_derivative": value, "j14_curvature": 0.0,
         "mode_0_mean_derivative": value,
         "mode_0_mean_curvature": 0.0,
         "mode_1_mean_derivative": -value,
         "mode_1_mean_curvature": 0.0,
         "mode_0_effective_events_derivative": 1.0,
         "mode_0_effective_events_curvature": 0.0,
         "mode_1_effective_events_derivative": 1.0,
         "mode_1_effective_events_curvature": 0.0}
        for seed, value in zip((1, 2, 3), (-1.0, -2.0, 0.5))
    ]
    summary = summarize_differences(rows, 0.15)
    assert len(summary) == 1
    assert summary[0]["negative_networks_mode_0_mean_derivative"] == 2
    assert summary[0]["positive_networks_mode_0_mean_derivative"] == 1


def test_reference_geometry_uses_the_active_stage_seed_pool():
    assert _configured_network_seeds({"search": {
        "fit_network_seeds": [],
        "selection_network_seeds": [2373, 2371, 2372],
        "confirmation_network_seeds": [],
    }}) == [2371, 2372, 2373]
    assert _configured_network_seeds({"search": {
        "fit_network_seeds": [],
        "selection_network_seeds": [],
        "confirmation_network_seeds": [2381, 2382, 2383],
    }}) == [2381, 2382, 2383]
