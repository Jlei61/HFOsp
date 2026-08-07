"""Tests for reading a frozen-state map.

The error these guard is the one that would make the map say more than it can: calling a point
bistable because two probes came back with different regime names, when with one probe per start
a name can flip for reasons that are not the state.
"""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc3_bistability import (
    bistable_points,
    boundary_along,
    evidence_summary,
    is_high,
    jump_size,
)

CEIL = 0.04


def _row(ad, ax, ic, mean_af, regime):
    return dict(alpha_d=ad, alpha_x=ax, ic=ic, mean_af=mean_af, regime=regime,
                interictal_ceiling_af=CEIL)


def _map(high_rule):
    """A synthetic map; ``high_rule(ad, ax, ic)`` says whether that probe ends up high."""
    rows = []
    for ad in (0.6, 1.0, 1.5):
        for ax in (0.0, 1.0, 2.0):
            for ic in ("interictal", "ictal"):
                hi = high_rule(ad, ax, ic)
                rows.append(_row(ad, ax, ic,
                                 0.30 if hi else 0.01,
                                 "R4_burst_train" if hi else "R0_interictal_only"))
    return rows


def test_high_is_decided_by_activity_not_by_the_regime_name():
    """A label alone must not promote a probe sitting at interictal activity to a high branch."""
    quiet_but_labelled = _row(1.0, 0.0, "ictal", mean_af=0.02, regime="R4_burst_train")
    assert is_high(quiet_but_labelled) is False
    loud = _row(1.0, 0.0, "ictal", mean_af=0.30, regime="R4_burst_train")
    assert is_high(loud) is True


def test_a_point_is_bistable_only_when_the_two_starts_disagree():
    rows = _map(lambda ad, ax, ic: ic == "ictal" and ad >= 1.0)
    pts = {(p["alpha_d"], p["alpha_x"]): p for p in bistable_points(rows)}
    assert pts[(1.0, 0.0)]["bistable"] is True
    assert pts[(0.6, 0.0)]["bistable"] is False


def test_a_point_high_from_both_starts_is_not_bistable():
    rows = _map(lambda ad, ax, ic: True)
    assert not any(p["bistable"] for p in bistable_points(rows))


def test_a_point_low_from_both_starts_is_not_bistable():
    rows = _map(lambda ad, ax, ic: False)
    assert not any(p["bistable"] for p in bistable_points(rows))


def test_hysteresis_is_reported_when_the_two_boundaries_sit_apart():
    rows = _map(lambda ad, ax, ic: (ad >= 1.5) if ic == "interictal" else (ad >= 1.0))
    pts = bistable_points(rows)
    up = boundary_along(pts, "alpha_d", "interictal")
    down = boundary_along(pts, "alpha_d", "ictal")
    assert up and down and any(up[k] != down[k] for k in set(up) & set(down))
    assert evidence_summary(rows)["hysteresis"] == "supported"


def test_no_hysteresis_is_reported_when_the_boundaries_coincide():
    rows = _map(lambda ad, ax, ic: ad >= 1.0)
    assert evidence_summary(rows)["hysteresis"] == "not seen"


def test_the_jump_is_measured_not_asserted():
    rows = _map(lambda ad, ax, ic: ic == "ictal" and ad >= 1.0)
    j = jump_size(rows)
    assert j["separated"] is True and j["gap_ratio"] > 5.0


def test_a_map_with_one_branch_only_reports_that_it_cannot_measure_a_jump():
    rows = _map(lambda ad, ax, ic: False)
    assert jump_size(rows)["separated"] is False


def test_the_allowed_claim_needs_both_bistability_and_hysteresis():
    both = _map(lambda ad, ax, ic: (ad >= 1.5) if ic == "interictal" else (ad >= 1.0))
    assert "saddle-node-like" in evidence_summary(both)["claim_allowed"]
    neither = _map(lambda ad, ax, ic: ad >= 1.0)
    assert "not enough" in evidence_summary(neither)["claim_allowed"]


def test_the_forbidden_claim_is_stated_whatever_the_map_shows():
    for rows in (_map(lambda a, x, i: True), _map(lambda a, x, i: False)):
        assert "proven saddle-node" in evidence_summary(rows)["claim_forbidden"]


def test_an_unusable_ceiling_falls_back_to_the_label_rather_than_dividing_by_it():
    r = dict(regime="R4_burst_train", mean_af=0.3, interictal_ceiling_af=float("nan"))
    assert is_high(r) is True
    r2 = dict(regime="R0_interictal_only", mean_af=0.3, interictal_ceiling_af=float("nan"))
    assert is_high(r2) is False


def test_points_missing_one_of_their_two_starts_are_skipped_not_guessed():
    rows = [_row(1.0, 0.0, "ictal", 0.30, "R4_burst_train")]
    assert bistable_points(rows) == []


def test_the_summary_counts_what_it_says_it_counts():
    rows = _map(lambda ad, ax, ic: ic == "ictal" and ad >= 1.0)
    s = evidence_summary(rows)
    assert s["n_points"] == 9
    assert s["n_bistable"] == sum(1 for p in bistable_points(rows) if p["bistable"])
    assert np.isfinite(s["jump"]["gap_ratio"])


def test_the_boundary_is_where_the_branch_begins_not_where_the_grid_ends():
    """Taking the far edge would report the grid's own limit for both starts and hide any
    hysteresis, however large."""
    rows = _map(lambda ad, ax, ic: (ad >= 1.5) if ic == "interictal" else (ad >= 1.0))
    up = boundary_along(rows_points(rows), "alpha_d", "interictal")
    down = boundary_along(rows_points(rows), "alpha_d", "ictal")
    assert set(up.values()) == {1.5} and set(down.values()) == {1.0}


def test_the_relay_axis_uses_the_other_end_because_it_opposes_the_high_branch():
    rows = _map(lambda ad, ax, ic: ax <= 1.0)
    b = boundary_along(rows_points(rows), "alpha_x", "ictal")
    assert set(b.values()) == {1.0}, "the boundary on X is the highest load still high"


def rows_points(rows):
    return bistable_points(rows)
