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
