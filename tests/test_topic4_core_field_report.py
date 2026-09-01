import numpy as np
import pytest
from src.topic4_core_field_report import (
    COMPARISONS, SCORE_KEYS, SEEDS, SIM_ARMS, concordance, stage1_report,
    tiered_paired_stats,
)

CFG = {"checksum": "abc123", "delta_eq": 0.05}


def _runs(separable=True, n_dir=2, cov=0.9):
    out = {}
    for arm in SIM_ARMS:
        for seed in SEEDS:
            base = 0.80 if arm.startswith("manual") else 0.60
            if not separable:
                base = 0.70
            jitter = 0.001 * ((seed * 7 + SIM_ARMS.index(arm)) % 3)
            for key in SCORE_KEYS:
                out[(arm, seed) + key] = dict(
                    n_dir=n_dir, S_rank=base + jitter,
                    coverage_forward=cov, coverage_reverse=cov)
    return out


def test_there_is_no_event_gate_dimension():
    """gate=4 admits zero signed events, so it is not an axis."""
    assert all(len(k) == 3 for k in SCORE_KEYS)
    assert {k[2] for k in SCORE_KEYS} == {"spearman", "pair"}


def test_comparisons_use_manual_smooth_as_the_shape_baseline():
    shape = {c["name"]: c for c in COMPARISONS if c["group"] == "shape"}
    assert set(shape) == {"B1", "B2", "B3", "B4"}
    assert all(c["a"] == "manual_smooth" for c in shape.values())
    assert {c["name"] for c in COMPARISONS if c["group"] == "equivalence"} == {"A", "A2"}


def test_s_rank_is_never_differenced_across_direction_tiers():
    """spec 5.3: only same-tier seeds contribute a numeric difference; the rest
    are counted as direction-tier wins and losses."""
    st = tiered_paired_stats([(2, 0.8, 2, 0.5), (2, 0.9, 1, float("nan")),
                              (0, float("nan"), 2, 0.4)])
    assert st["n_same_tier"] == 1
    assert st["tier_wins"] == 1
    assert st["tier_losses"] == 1
    assert st["mean"] == pytest.approx(0.3)


def test_tiered_stats_report_a_confidence_interval():
    st = tiered_paired_stats([(2, 0.6 + 0.01 * i, 2, 0.5) for i in range(12)])
    assert st["ci_low"] < st["mean"] < st["ci_high"]


def test_report_is_a_pure_function():
    runs = _runs()
    snapshot = {k: dict(v) for k, v in runs.items()}
    stage1_report(runs, CFG)
    assert runs == snapshot


def test_integrity_fails_closed_on_a_missing_cell():
    runs = _runs(); del runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]
    assert stage1_report(runs, CFG)["integrity"]["status"] == "FAIL_CLOSED"


def test_integrity_fails_closed_on_a_nan_with_two_directions():
    runs = _runs()
    runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]["S_rank"] = float("nan")
    assert stage1_report(runs, CFG)["integrity"]["status"] == "FAIL_CLOSED"


def test_a_nan_with_no_directions_is_legitimate():
    runs = _runs()
    cell = runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]
    cell["n_dir"], cell["S_rank"] = 0, float("nan")
    assert stage1_report(runs, CFG)["integrity"]["status"] == "ok"


def test_separable_and_flat_arms_give_different_recommendations():
    sep = stage1_report(_runs(separable=True), CFG)
    flat = stage1_report(_runs(separable=False), CFG)
    assert sep["recommendation"]["shape_separates"] is True
    assert flat["recommendation"]["shape_separates"] is False
    for r in (sep, flat):
        assert r["integrity"]["status"] == "ok"
        assert "verdict" not in r          # exploratory: no automatic gate


def test_low_coverage_is_flagged_but_does_not_stop_anything():
    runs = _runs()
    for seed in SEEDS:
        for key in SCORE_KEYS:
            runs[("uniform_axial", seed) + key]["coverage_forward"] = 0.2
    rep = stage1_report(runs, CFG)
    assert "uniform_axial" in rep["coverage"]["low_coverage_arms"]
    assert rep["integrity"]["status"] == "ok"


def test_transverse_sign_flip_does_not_change_the_report():
    runs = _runs(separable=True)
    swap = {"transverse_plus": "transverse_minus", "transverse_minus": "transverse_plus"}
    flipped = {(swap.get(k[0], k[0]),) + k[1:]: dict(v) for k, v in runs.items()}
    assert stage1_report(flipped, CFG)["recommendation"] == \
           stage1_report(runs, CFG)["recommendation"]
