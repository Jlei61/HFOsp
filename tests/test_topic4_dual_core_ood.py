from unittest.mock import patch

import numpy as np
import pytest

from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds
from scripts.run_topic4_dual_core_ood_controller import (
    launch_capacity,
    _unit_token,
)
from src.topic4_dual_core_ood import (
    candidate_field_sha256,
    candidate_sort_key,
    generate_sobol_candidates,
    score_returned_event_support,
    spatial_event_activity_grid,
)


FAMILY = {
    "sheet_bounds_mm": [0.75, 19.25],
    "minimum_center_separation_mm": 4.0,
    "maximum_center_separation_mm": 18.0,
    "target_count_bounds": [700, 1600],
    "minimum_reference_budget_fraction_per_core": 0.25,
    "sobol_candidates": 8,
    "sobol_seed": 20260830,
    "historical_anchor": {
        "candidate_id": "anchor",
        "centers_mm": [[4.2, 9.1], [16.5, 4.0]],
        "target_count": 1129,
    },
}


def test_sobol_dual_core_library_is_deterministic_and_strictly_two_core():
    left = generate_sobol_candidates(FAMILY)
    right = generate_sobol_candidates(FAMILY)
    assert left == right
    assert len(left) == 9
    for row in left:
        centers = np.asarray(row["centers_mm"], float)
        assert centers.shape == (2, 2)
        assert 4.0 <= np.linalg.norm(centers[0] - centers[1]) <= 18.0
        assert min(row["reference_budget_fraction_per_core"]) >= 0.25


def test_field_hash_ignores_component_label_exchange():
    centers = np.asarray([[3.0, 4.0], [14.0, 7.0]])
    assert candidate_field_sha256(centers, 1000) == candidate_field_sha256(
        centers[::-1], 1000,
    )


def test_unreadable_returned_event_counts_as_primary_ood():
    onsets = np.asarray([
        [0.0, 1.0, 2.0, np.nan],
        [0.0, np.nan, np.nan, np.nan],
        [0.0, 1.0, 2.0, 3.0],
    ])
    assigned = {
        "labels": np.asarray([0, 1, 1]),
        "ood": np.asarray([False, False, True]),
        "ood_distance": np.asarray([0.2, 0.2, 2.0]),
    }
    classifier = {"ood_distance_thresholds": [1.0, 1.0]}
    with patch(
        "src.topic4_dual_core_ood.assign_direction_modes",
        return_value=assigned,
    ):
        result = score_returned_event_support(
            onsets, np.ones(3, bool), contract={"contacts": []},
            embedding={}, classifier=classifier,
        )
    assert result["n_in_support"] == 1
    assert result["ood_all_returned"] == pytest.approx(2 / 3)
    assert result["ood_returned_readable"] == 0.5
    assert result["unreadable_returned_fraction"] == pytest.approx(1 / 3)


def test_ranking_protects_both_modes_before_lower_ood():
    both = {
        "candidate_id": "both", "networks_with_both_modes": 2,
        "equal_network_ood_all_returned": 0.4,
        "weakest_mode_normalized_support_distance": 0.5,
        "equal_network_returned_events": 20,
    }
    collapsed = {
        "candidate_id": "collapsed", "networks_with_both_modes": 0,
        "equal_network_ood_all_returned": 0.1,
        "weakest_mode_normalized_support_distance": None,
        "equal_network_returned_events": 30,
    }
    assert sorted([collapsed, both], key=candidate_sort_key)[0][
        "candidate_id"
    ] == "both"


def test_activity_grid_preserves_event_and_spatial_identity():
    spikes = np.zeros((10, 2), bool)
    spikes[2, 0] = True
    spikes[7, 1] = True
    positions = np.asarray([[1.0, 1.0], [19.0, 19.0]])
    events = [{"t_on_ms": 2.0, "t_off_ms": 7.0}]
    result = spatial_event_activity_grid(
        spikes, positions, events, dt_ms=1.0, bin_ms=5.0,
        spatial_bins=2, sheet_l_mm=20.0, pad_before_ms=0.0,
        pad_after_ms=1.0,
    )
    assert result["activity_grid"].shape == (2, 2, 2)
    assert result["activity_grid"][0, 0, 0] == 1
    assert result["activity_grid"][1, 1, 1] == 1
    assert result["activity_grid_event_count"].tolist() == [2]


def test_new_config_exposes_all_phase_seeds_without_changing_old_behavior():
    new = {
        "schema_version": "topic4_dual_core_ood_node_pathways_v1",
        "search": {
            "fit_network_seeds": [1], "selection_network_seeds": [2],
            "confirmation_network_seeds": [3], "pathway_network_seeds": [4],
        },
    }
    old = {
        "search": {
            "phase": "selection", "fit_network_seeds": [1],
            "selection_network_seeds": [2], "confirmation_network_seeds": [3],
        },
    }
    assert active_network_seeds(new) == [1, 2, 3, 4]
    assert active_network_seeds(old) == [2]


def test_controller_capacity_reserves_memory_and_counts_active_workers():
    assert launch_capacity(
        100.0, reserve_gib=32.0, peak_gib=16.0,
        worker_cap=8, active_workers=2,
    ) == 4
    assert launch_capacity(
        45.0, reserve_gib=32.0, peak_gib=16.0,
        worker_cap=8, active_workers=0,
    ) == 0


def test_controller_unit_name_is_candidate_safe_and_distinct():
    left = _unit_token("fit", "candidate/with unsafe name", 2401, "abcdef1234")
    right = _unit_token("fit", "another candidate", 2401, "abcdef1234")
    assert "/" not in left and " " not in left
    assert left != right
