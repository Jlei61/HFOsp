import numpy as np

from src.topic4_zm_boundaries import (
    bootstrap_half_boundary,
    half_boundary,
    hysteresis_summary,
    jeffreys_probability_curve,
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


def test_actual_trajectory_crossing_direction_is_explicit():
    inc = trajectory_crossing([0.0, 0.8, 1.2, 1.8], 1.0, expected_direction="increasing")
    wrong = trajectory_crossing([1.8, 1.2, 0.8, 0.0], 1.0, expected_direction="increasing")
    assert inc["crossed"] and inc["direction_ok"]
    assert wrong["crossed"] and not wrong["direction_ok"]


def test_onset_and_offset_surfaces_report_hysteresis_not_one_threshold():
    h = hysteresis_summary(0.7, 1.4, scale=1.0)
    same = hysteresis_summary(1.0, 1.01, scale=1.0)
    assert h["distinct_surfaces"]
    assert np.isclose(h["signed_separation"], 0.7)
    assert not same["distinct_surfaces"]
