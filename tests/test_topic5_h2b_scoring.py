"""B1 contract tests: discrete-time survival scoring and the nested increment.

The censoring arithmetic is the part that silently corrupts a survival result,
so it is pinned first: a row that merely ran out of monitoring must contribute
only the bins it actually survived, and must never be scored as "no seizure".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topic5_h2b_transfer.scoring import (
    brier_by_bin,
    discrete_time_log_score,
    nested_increment,
    field_score,
)


# --- discrete-time log score ---------------------------------------------------


def test_event_row_scores_the_hazard_of_its_bin_and_survival_before_it():
    h = np.array([[0.1, 0.2, 0.5]])
    ll = discrete_time_log_score(h, outcome_bin=[1], last_observed_bin=[2], censored=[False])
    assert np.isclose(ll[0], np.log(1 - 0.1) + np.log(0.2))


def test_censored_row_scores_only_the_bins_it_actually_survived():
    """Coverage ended after bin 0: bins 1-2 carry no information about it."""
    h = np.array([[0.1, 0.2, 0.5]])
    ll = discrete_time_log_score(h, outcome_bin=[None], last_observed_bin=[0], censored=[True])
    assert np.isclose(ll[0], np.log(1 - 0.1))


def test_a_censored_row_is_not_scored_as_a_survivor_of_the_whole_horizon():
    h = np.array([[0.1, 0.2, 0.5]])
    censored = discrete_time_log_score(h, [None], [0], [True])[0]
    survived_all = discrete_time_log_score(h, [None], [2], [False])[0]
    assert censored > survived_all  # fewer terms, so strictly less negative


def test_beyond_horizon_row_scores_survival_through_every_bin():
    h = np.array([[0.1, 0.2, 0.5]])
    ll = discrete_time_log_score(h, [None], [2], [False])
    expected = np.log(0.9) + np.log(0.8) + np.log(0.5)
    assert np.isclose(ll[0], expected)


def test_a_row_that_observed_nothing_contributes_nothing():
    h = np.array([[0.1, 0.2, 0.5]])
    ll = discrete_time_log_score(h, [None], [-1], [True])
    assert ll[0] == 0.0


def test_hazards_are_clipped_so_a_confident_wrong_call_is_finite():
    h = np.array([[0.0, 0.0, 0.0]])
    ll = discrete_time_log_score(h, outcome_bin=[0], last_observed_bin=[2], censored=[False])
    assert np.isfinite(ll[0])


def test_event_after_the_last_observed_bin_is_rejected_as_inconsistent():
    h = np.array([[0.1, 0.2, 0.5]])
    with pytest.raises(ValueError, match="beyond"):
        discrete_time_log_score(h, outcome_bin=[2], last_observed_bin=[0], censored=[False])


# --- Brier ---------------------------------------------------------------------


def test_brier_only_counts_bins_the_row_was_actually_at_risk_in():
    h = np.array([[0.5, 0.5, 0.5]])
    b = brier_by_bin(h, outcome_bin=[None], last_observed_bin=[0], censored=[True])
    assert np.isfinite(b[0]) and np.isnan(b[1]) and np.isnan(b[2])


def test_brier_is_zero_for_a_perfect_call():
    h = np.array([[0.0, 1.0]])
    b = brier_by_bin(h, outcome_bin=[1], last_observed_bin=[1], censored=[False])
    assert np.allclose(b[np.isfinite(b)], 0.0)


# --- nested increment ----------------------------------------------------------


def test_increment_requires_the_two_arms_to_cover_identical_rows():
    with pytest.raises(ValueError, match="same rows"):
        nested_increment(np.array([1.0, 2.0]), np.array([1.0]))


def test_increment_is_the_paired_mean_gain_of_the_richer_arm():
    base = np.array([-2.0, -3.0])
    full = np.array([-1.0, -1.0])
    out = nested_increment(base, full)
    assert np.isclose(out["mean_gain"], 1.5)
    assert out["n"] == 2


# --- early field score ----------------------------------------------------------


def test_field_score_is_rank_based_and_ignores_missing_contacts():
    pred = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    obs = np.array([10.0, 20.0, 30.0, 40.0, 999.0])
    assert np.isclose(field_score(pred, obs), 1.0)


def test_field_score_of_a_reversed_field_is_minus_one():
    assert np.isclose(
        field_score(np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])), -1.0
    )


def test_field_score_below_the_minimum_contact_guard_is_nan():
    """Three surviving contacts cannot support a spatial claim."""
    assert np.isnan(field_score(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])))


def test_field_score_with_too_few_overlapping_contacts_is_nan():
    assert np.isnan(field_score(np.array([1.0, np.nan]), np.array([np.nan, 2.0])))
