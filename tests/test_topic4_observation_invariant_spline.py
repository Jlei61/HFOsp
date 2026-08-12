import inspect

import numpy as np

from scripts.launch_topic4_rev10_sa_spectral_field_search import NUMERIC_ENV

from src.topic4_continuous_field import continuous_surface, tensor_basis
from src.topic4_observation_invariant_spline import (
    allocation_direction,
    fit_uniform_surface,
    sample_smooth_residual_pairs,
    uniform_allocation_centers,
)
from src.topic4_spectral_field import uniform_sheet_grid


def test_spline_search_limits_each_numeric_runtime_to_one_thread():
    assert NUMERIC_ENV
    assert set(NUMERIC_ENV.values()) == {"1"}
    assert {
        "BLIS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    } <= set(NUMERIC_ENV)


def test_uniform_spline_coordinates_are_stable_and_observation_free():
    from scripts.freeze_topic4_rev10_sa_spline_field_v4_candidates import (
        build_candidates,
    )
    from scripts.freeze_topic4_rev10_sa_spline_bridge_v41_candidates import (
        build_candidates as build_bridge_candidates,
    )
    from scripts.freeze_topic4_rev10_sa_spline_interpolation_v5_candidates import (
        build_candidates as build_interpolation_candidates,
    )

    grid = uniform_sheet_grid(32, L=20.0)
    basis = tensor_basis(grid, 10, degree=3, L=20.0)
    condition = np.linalg.cond(basis)
    forbidden = {"contacts", "contact_xy", "shaft_ids", "onsets", "labels"}

    assert condition < 100.0
    for function in (
        uniform_allocation_centers, fit_uniform_surface,
        allocation_direction, sample_smooth_residual_pairs, build_candidates,
        build_bridge_candidates, build_interpolation_candidates,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_smooth_random_pairs_are_antithetic_at_frozen_rms():
    grid = uniform_sheet_grid(32, L=20.0)
    pairs = sample_smooth_residual_pairs(
        n_pairs=2, n_basis=10, seed=3, rms_amplitudes=[2.0, 3.0],
        positions=grid, smoothing_controls=1.0, degree=3, L=20.0,
    )
    for pair, expected in zip(pairs, (2.0, 3.0)):
        positive = continuous_surface(
            pair["positive"], grid, n_basis=10, degree=3, L=20.0,
        )
        negative = continuous_surface(
            pair["negative"], grid, n_basis=10, degree=3, L=20.0,
        )
        assert np.allclose(positive, -negative)
        assert np.isclose(
            np.sqrt(np.mean((positive - positive.mean()) ** 2)), expected,
        )


def test_joint_shaft_selection_is_fail_closed_but_keeps_diagnostic():
    from scripts.aggregate_topic4_rev10_sa_spline_field_search import (
        _selection_verdict,
    )

    config = {"search": {"objective": {
        "minimum_joint_events_for_selection": 1,
    }}}
    rows = [
        {
            "candidate_id": "route_only", "selection_score": 1.0,
            "n_runaway_networks": 0, "n_joint": 0,
        },
        {
            "candidate_id": "other", "selection_score": 2.0,
            "n_runaway_networks": 0, "n_joint": 0,
        },
    ]
    verdict = _selection_verdict(rows, config)
    assert verdict["selected"] is None
    assert verdict["diagnostic"]["candidate_id"] == "route_only"
    assert verdict["status"] == "REV10SA_V4_NO_JOINT_SHAFT_CANDIDATE"

    rows[1]["n_joint"] = 1
    verdict = _selection_verdict(rows, config)
    assert verdict["selected"]["candidate_id"] == "other"


def test_v5_anchor_selection_uses_scores_not_spatial_metadata():
    from scripts.freeze_topic4_rev10_sa_spline_interpolation_v5_candidates import (
        select_anchor_ids,
    )

    rows = [
        {"candidate_id": "reference", "n_joint": 0, "joint_fraction": 0.0,
         "selection_score": 8.0, "route_score": 8.0, "n_runaway_networks": 0},
        {"candidate_id": "joint_high", "n_joint": 2, "joint_fraction": 0.4,
         "selection_score": 5.0, "route_score": 5.0, "n_runaway_networks": 0},
        {"candidate_id": "joint_low", "n_joint": 1, "joint_fraction": 0.2,
         "selection_score": 4.0, "route_score": 4.0, "n_runaway_networks": 0},
        {"candidate_id": "route", "n_joint": 0, "joint_fraction": 0.0,
         "selection_score": 3.0, "route_score": 2.0, "n_runaway_networks": 0},
    ]
    selected = select_anchor_ids(
        {"candidate_rows": rows}, reference_id="reference",
        joint_count=2, route_count=1,
    )
    assert selected == ["reference", "joint_high", "joint_low", "route"]


def test_selection_confirmation_requires_joint_events_in_both_networks():
    from scripts.aggregate_topic4_rev10_sa_spline_field_search import (
        _selection_verdict,
    )

    config = {"search": {"objective": {
        "minimum_joint_events_for_selection": 2,
        "minimum_seeds_with_joint_for_selection": 2,
    }}}
    rows = [{
        "candidate_id": "one_seed", "selection_score": 1.0,
        "n_runaway_networks": 0, "n_joint": 3,
        "n_seeds_with_joint": 1,
    }]
    assert _selection_verdict(rows, config)["selected"] is None
    rows[0]["n_seeds_with_joint"] = 2
    assert _selection_verdict(rows, config)["selected"] is rows[0]


def test_selection_can_require_patient_supported_joint_events_in_both_modes():
    from scripts.aggregate_topic4_rev10_sa_spline_field_search import (
        _selection_verdict,
    )

    config = {"search": {"objective": {
        "minimum_joint_events_for_selection": 2,
        "minimum_seeds_with_joint_for_selection": 1,
        "minimum_joint_in_distribution_events_per_mode_for_selection": 1,
        "minimum_seeds_with_joint_in_distribution_per_mode_for_selection": 1,
    }}}
    row = {
        "candidate_id": "shaft_partition", "selection_score": 1.0,
        "n_runaway_networks": 0, "n_joint": 20, "n_seeds_with_joint": 3,
        "weak_mode_joint_in_distribution_count": 0,
        "weak_mode_joint_in_distribution_seed_count": 0,
    }
    assert _selection_verdict([row], config)["selected"] is None
    row["weak_mode_joint_in_distribution_count"] = 1
    row["weak_mode_joint_in_distribution_seed_count"] = 1
    assert _selection_verdict([row], config)["selected"] is row


def test_selection_can_require_both_modes_in_the_same_network():
    from scripts.aggregate_topic4_rev10_sa_spline_field_search import _selection_verdict

    config = {"search": {"objective": {
        "minimum_joint_events_for_selection": 2,
        "minimum_seeds_with_joint_for_selection": 1,
        "minimum_joint_in_distribution_events_per_mode_for_selection": 1,
        "minimum_seeds_with_joint_in_distribution_per_mode_for_selection": 1,
        "minimum_same_networks_with_both_joint_in_distribution_modes_for_selection": 1,
    }}}
    row = {
        "candidate_id": "different_networks", "selection_score": 1.0,
        "n_runaway_networks": 0, "n_joint": 4, "n_seeds_with_joint": 2,
        "weak_mode_joint_in_distribution_count": 1,
        "weak_mode_joint_in_distribution_seed_count": 1,
        "same_network_both_modes_joint_in_distribution_count": 0,
    }
    assert _selection_verdict([row], config)["selected"] is None
    row["same_network_both_modes_joint_in_distribution_count"] = 1
    assert _selection_verdict([row], config)["selected"] is row


def test_v51_diversity_rule_keeps_pareto_anchors_and_winner_path():
    from scripts.freeze_topic4_rev10_sa_v5_selection_candidates import (
        select_confirmation_ids,
    )

    candidates = [
        {"candidate_id": "winner", "role": "adaptive_density_mixture_interpolation",
         "anchor_pair_index": 3},
        {"candidate_id": "neighbor_a", "role": "adaptive_density_mixture_interpolation",
         "anchor_pair_index": 3},
        {"candidate_id": "neighbor_b", "role": "adaptive_density_mixture_interpolation",
         "anchor_pair_index": 3},
        {"candidate_id": "anchor_joint", "role": "adaptive_training_anchor"},
        {"candidate_id": "pareto", "role": "adaptive_training_anchor"},
    ]
    rows = [
        {"candidate_id": "winner", "role": candidates[0]["role"],
         "joint_fraction": 0.5, "selection_score": 1.0, "n_joint": 2,
         "pareto_nondominated": True},
        {"candidate_id": "neighbor_a", "role": candidates[1]["role"],
         "joint_fraction": 0.2, "selection_score": 3.0, "n_joint": 1,
         "pareto_nondominated": False},
        {"candidate_id": "neighbor_b", "role": candidates[2]["role"],
         "joint_fraction": 0.0, "selection_score": 4.0, "n_joint": 0,
         "pareto_nondominated": False},
        {"candidate_id": "anchor_joint", "role": "adaptive_training_anchor",
         "joint_fraction": 0.3, "selection_score": 2.0, "n_joint": 1,
         "pareto_nondominated": False},
        {"candidate_id": "pareto", "role": "adaptive_training_anchor",
         "joint_fraction": 0.0, "selection_score": 2.5, "n_joint": 0,
         "pareto_nondominated": True},
    ]
    selected = select_confirmation_ids(
        {"candidate_set": {"candidates": candidates}},
        {"selected_candidate_id": "winner", "candidate_rows": rows},
    )
    assert selected == [
        "winner", "anchor_joint", "pareto", "neighbor_a", "neighbor_b",
    ]


def test_v52_uses_distinct_score_support_and_stage3_fields():
    from scripts.freeze_topic4_rev10_sa_v52_final_candidates import (
        select_final_sources,
    )

    summary = {
        "selected_candidate_id": "score_winner",
        "minimum_seeds_with_joint_for_selection": 2,
        "candidate_rows": [
            {"candidate_id": "score_winner", "n_runaway_networks": 0,
             "n_joint": 4, "joint_fraction": 0.5, "n_seeds_with_joint": 2,
             "selection_score": 1.0},
            {"candidate_id": "support", "n_runaway_networks": 0,
             "n_joint": 12, "joint_fraction": 0.4, "n_seeds_with_joint": 2,
             "selection_score": 3.0},
            {"candidate_id": "one_network", "n_runaway_networks": 0,
             "n_joint": 20, "joint_fraction": 0.8, "n_seeds_with_joint": 1,
             "selection_score": 0.5},
        ],
    }
    assert select_final_sources(summary, "stage3") == (
        "score_winner", "support", "stage3",
    )


def test_v6_boundary_reproduces_known_v5_endpoint():
    import json
    from pathlib import Path

    from scripts.freeze_topic4_rev10_sa_v6_mode_boundary_candidates import (
        build_candidates,
    )

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/topic4_rev10_sa_observation_invariant_field_v6.json").read_text())
    manifest = json.loads((root / config["inputs"]["v5_candidate_manifest"]["path"]).read_text())
    audit = json.loads((root / config["inputs"]["v5_mode_conditioned_audit"]["path"]).read_text())
    candidates, _ = build_candidates(config, manifest, audit)

    assert len(candidates) == 11
    known = next(
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == "v5_density_p05_t025"
    )
    assert candidates[-1]["field_sha256"] == known["field_sha256"]


def test_v62_keeps_all_adjacent_selection_positive_fields():
    from scripts.freeze_topic4_rev10_sa_v62_final_candidates import select_sources

    summary = {"candidate_rows": [
        {"candidate_id": "v61_t075",
         "same_network_both_modes_joint_in_distribution_count": 1},
        {"candidate_id": "v61_t000",
         "same_network_both_modes_joint_in_distribution_count": 0},
        {"candidate_id": "v61_t025",
         "same_network_both_modes_joint_in_distribution_count": 1},
        {"candidate_id": "v61_t050",
         "same_network_both_modes_joint_in_distribution_count": 1},
    ]}
    assert select_sources(summary) == ["v61_t025", "v61_t050", "v61_t075"]
