"""Pre-registered statistics: censoring, pairing, and an enumerated spatial null."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_statistics import (  # noqa: E402
    covariate_collinearity, exact_toroidal_shifts, paired_bootstrap,
    paired_onset_difference, phase2_decision, restricted_ictal_free_time,
    spatial_correlation_exact_shift)


def test_paired_bootstrap_is_paired_and_deterministic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a + 0.5
    one = paired_bootstrap(a, b, draws=1000, seed=5)
    two = paired_bootstrap(a, b, draws=1000, seed=5)
    assert one == two
    assert np.isclose(one["mean_difference"], -0.5)
    assert one["q95"] < 0.0
    assert one["n"] == 4


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap(np.zeros(3), np.zeros(4), draws=10, seed=1)


def test_restricted_ictal_free_time_treats_none_as_censored_at_the_cap():
    out = restricted_ictal_free_time([5000.0, None, 15000.0], cap_ms=20000.0)
    assert np.isclose(out["restricted_mean_ms"], (5000.0 + 20000.0 + 15000.0) / 3.0)
    assert out["n_censored"] == 1
    assert out["n"] == 3
    assert np.isclose(out["entered_fraction"], 2.0 / 3.0)


def test_paired_onset_difference_uses_only_networks_where_both_entered():
    out = paired_onset_difference([1000.0, None, 3000.0], [2000.0, 4000.0, None])
    assert out["n"] == 1
    assert np.isclose(out["mean_difference_ms"], -1000.0)
    assert out["n_dropped"] == 2


def test_the_shift_group_is_enumerated_in_full():
    shifts = exact_toroidal_shifts(7)
    assert shifts.shape == (49, 2)
    assert len({tuple(s) for s in shifts}) == 49
    assert (0, 0) in {tuple(s) for s in shifts}


def test_exact_shift_null_reports_49_shifts_and_a_1_over_49_floor():
    rng = np.random.default_rng(0)
    values = rng.random(49)
    covariate = rng.random(49)
    out = spatial_correlation_exact_shift(values, covariate, grid_n=7)
    assert out["n_distinct_shifts"] == 49
    assert np.isclose(out["p_floor"], 1.0 / 49.0)
    assert out["p_value"] >= out["p_floor"] - 1e-12
    assert len(out["null_r"]) == 49


def test_exact_shift_null_hits_its_floor_on_a_perfect_match():
    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7), indexing="ij"),
                    axis=-1).reshape(-1, 2)
    field = np.sin(grid[:, 0] * 0.9) + np.cos(grid[:, 1] * 0.7)
    out = spatial_correlation_exact_shift(field, field, grid_n=7)
    assert np.isclose(out["spearman_r"], 1.0)
    assert np.isclose(out["p_value"], 1.0 / 49.0)


def test_exact_shift_null_is_deterministic_and_takes_no_draws():
    rng = np.random.default_rng(3)
    values, covariate = rng.random(49), rng.random(49)
    assert (spatial_correlation_exact_shift(values, covariate, grid_n=7)
            == spatial_correlation_exact_shift(values, covariate, grid_n=7))


def test_collinearity_reports_every_pair_and_decides_nothing():
    rng = np.random.default_rng(4)
    h = rng.random(49)
    covariates = {"h": h, "ee_gain": h * 2.0 + 0.01 * rng.random(49),
                  "etoi_gain": h * -1.5 + 0.01 * rng.random(49)}
    out = covariate_collinearity(covariates)
    assert out["max_abs_r"] > 0.9
    assert set(out["pairwise_spearman"]) == {("ee_gain", "etoi_gain"),
                                             ("ee_gain", "h"), ("etoi_gain", "h")}
    assert "report_as_single_family" not in out    # h is primary by design, not by data


def test_phase2_decision_is_three_way_and_directional():
    """A significantly NEGATIVE interval also excludes zero; continuing on it
    would spend five more hours on a result pointing the other way."""
    assert phase2_decision({"q05": 0.4, "q95": 1.2})["action"] == "continue"
    stop_neg = phase2_decision({"q05": -1.2, "q95": -0.4})
    assert stop_neg["action"] == "stop"
    assert stop_neg["reason"] == "opposite_direction"
    straddle = phase2_decision({"q05": -0.3, "q95": 0.9})
    assert straddle["action"] == "stop"
    assert straddle["reason"] == "unresolved"
    assert "no effect" not in straddle["permitted_wording"].lower()
