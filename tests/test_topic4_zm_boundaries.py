import numpy as np

from src.topic4_zm_boundaries import (
    boundary_reachability,
    bootstrap_half_boundary,
    half_boundary,
    hysteresis_summary,
    jeffreys_probability_curve,
    interpolate_slow_state,
    slow_state_coordinate_values,
    trajectory_crossing,
)


def _rows(levels, successes, n=20):
    out = []
    for x, k in zip(levels, successes):
        out.extend({"q": float(x), "outcome": i < k} for i in range(n))
    return out


def test_monotonic_probability_curve_has_bracketed_half_boundary():
    curve = jeffreys_probability_curve(
        _rows([0, 1, 2, 3], [1, 5, 15, 19]), "q", "outcome"
    )
    out = half_boundary(curve, expected_direction="increasing")
    assert out["status"] == "bracketed"
    assert 1.0 < out["q_half"] < 2.0
    assert out["direction"] == "increasing"


def test_unbracketed_and_nonmonotonic_curves_fail_closed():
    low = jeffreys_probability_curve(
        _rows([0, 1, 2], [0, 1, 2]), "q", "outcome"
    )
    assert half_boundary(low, expected_direction="increasing")["status"] == "unbracketed"

    nonmono = jeffreys_probability_curve(
        _rows([0, 1, 2, 3], [1, 17, 3, 19]), "q", "outcome"
    )
    assert half_boundary(nonmono, expected_direction="increasing")["status"] == "nonmonotonic"


def test_bootstrap_boundary_reports_uncertainty_only_when_replicates_bracket():
    rows = _rows([0, 1, 2, 3], [1, 4, 16, 19], n=20)
    out = bootstrap_half_boundary(
        rows, "q", "outcome", expected_direction="increasing",
        n_boot=300, seed=5,
    )
    assert out["status"] == "bracketed"
    assert out["n_valid_bootstrap"] > 250
    assert out["q_half_ci"][0] < out["q_half"] < out["q_half_ci"][1]


def test_clustered_boundary_bootstrap_resamples_seeds_before_replicates():
    rows = []
    for seed in (1, 3, 4):
        for q, k in zip([0, 1, 2, 3], [0, 1, 4, 5]):
            rows.extend(
                {
                    "seed": seed,
                    "q": float(q),
                    "outcome": i < k,
                }
                for i in range(5)
            )
    out = bootstrap_half_boundary(
        rows,
        "q",
        "outcome",
        expected_direction="increasing",
        n_boot=300,
        seed=5,
        cluster_key="seed",
    )
    assert out["status"] == "bracketed"
    assert out["bootstrap_structure"] == "hierarchical_seed_then_replicate"
    assert out["n_clusters"] == 3
    assert out["q_half_ci"] is not None


def test_clustered_boundary_bootstrap_fails_closed_with_one_seed():
    rows = [
        {"seed": 1, "q": q, "outcome": outcome}
        for q, outcome in [(0.0, False), (1.0, True)]
    ]
    out = bootstrap_half_boundary(
        rows,
        "q",
        "outcome",
        expected_direction="increasing",
        n_boot=100,
        seed=5,
        cluster_key="seed",
    )
    assert out["status"] == "bootstrap_indeterminate"
    assert out["q_half"] is None
    assert out["n_clusters"] == 1


def test_clustered_boundary_requires_seed_even_when_point_is_unbracketed():
    rows = [
        {"q": 0.0, "outcome": False},
        {"q": 1.0, "outcome": False},
    ]
    with np.testing.assert_raises(ValueError):
        bootstrap_half_boundary(
            rows,
            "q",
            "outcome",
            expected_direction="increasing",
            n_boot=100,
            seed=5,
            cluster_key="seed",
        )


def test_actual_trajectory_crossing_direction_is_explicit():
    inc = trajectory_crossing([0.0, 0.8, 1.2, 1.8], 1.0, expected_direction="increasing")
    wrong = trajectory_crossing([1.8, 1.2, 0.8, 0.0], 1.0, expected_direction="increasing")
    assert inc["crossed"] and inc["direction_ok"]
    assert wrong["crossed"] and not wrong["direction_ok"]


def test_boundary_reachability_requires_ci_range_and_actual_direction():
    boundary = {
        "status": "bracketed",
        "q_half": 0.6,
        "q_half_ci": [0.5, 0.7],
    }
    reached = boundary_reachability(
        boundary,
        [0.0, 1.0],
        expected_direction="increasing",
        reachable_range=(0.0, 1.0),
    )
    wrong_direction = boundary_reachability(
        boundary,
        [1.0, 0.0],
        expected_direction="increasing",
        reachable_range=(0.0, 1.0),
    )
    missing_ci = boundary_reachability(
        {**boundary, "q_half_ci": None},
        [0.0, 1.0],
        expected_direction="increasing",
        reachable_range=(0.0, 1.0),
    )
    outside_ci = boundary_reachability(
        {**boundary, "q_half_ci": [0.5, 1.1]},
        [0.0, 1.0],
        expected_direction="increasing",
        reachable_range=(0.0, 1.0),
    )
    assert reached["reached"]
    assert not wrong_direction["reached"]
    assert not missing_ci["reached"]
    assert not outside_ci["reached"]


def test_onset_and_offset_surfaces_report_hysteresis_not_one_threshold():
    h = hysteresis_summary(
        0.7, 1.4, scale=1.0, onset_ci=[0.6, 0.8], offset_ci=[1.2, 1.6]
    )
    same = hysteresis_summary(
        1.0, 1.01, scale=1.0, onset_ci=[0.8, 1.2], offset_ci=[0.9, 1.2]
    )
    missing = hysteresis_summary(0.7, 1.4, scale=1.0)
    assert h["distinct_surfaces"]
    assert np.isclose(h["signed_separation"], 0.7)
    assert not same["distinct_surfaces"]
    assert not missing["distinct_surfaces"]
    assert missing["status"] == "uncertainty_missing"


def _state(z, m, sg):
    return {
        "slow.z": np.asarray(z, float),
        "slow.m": np.asarray(m, float),
        "slow.S_G": np.asarray(float(sg)),
        "V": np.arange(len(z), dtype=float),
    }


def test_actual_slow_fields_are_interpolated_jointly_without_scalar_clipping():
    early = _state([0.9, 0.8, 1.0], [1.0, 2.0, 0.0], 0.2)
    late = _state([0.5, 0.4, 1.0], [5.0, 6.0, 0.0], 0.8)
    mid = interpolate_slow_state(
        early, late, 0.5, coordinates=("z", "m", "sg"), nE=2
    )
    assert np.allclose(mid["slow.z"], [0.7, 0.6, 1.0])
    assert np.allclose(mid["slow.m"], [3.0, 4.0, 0.0])
    assert np.isclose(mid["slow.S_G"], 0.5)
    assert np.array_equal(mid["V"], early["V"])

    q = slow_state_coordinate_values(
        [early, mid, late], early, late, coordinates=("z", "m", "sg"), nE=2
    )
    assert np.allclose(q["joint_lambda"], [0.0, 0.5, 1.0])
    assert np.allclose(q["per_coordinate_lambda"]["z"], [0.0, 0.5, 1.0])


def test_invalid_extrapolated_slow_field_is_rejected_not_clipped():
    early = _state([0.9, 0.8], [1.0, 2.0], 0.2)
    late = _state([0.5, 0.4], [5.0, 6.0], 0.8)
    with np.testing.assert_raises(ValueError):
        interpolate_slow_state(
            early, late, 3.0, coordinates=("z",), nE=2,
            allow_extrapolation=True,
        )
