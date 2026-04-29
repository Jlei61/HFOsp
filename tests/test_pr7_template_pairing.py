"""TDD tests for PR-7 template antagonistic temporal pairing.

Contract: docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md §8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.template_temporal_pairing import (  # noqa: E402
    compute_pairing_lift,
    compute_pairing_with_nulls,
    compute_transition_odds,
    evaluate_pass_criteria,
    shuffle_labels_block_aware,
    shuffle_labels_circular,
    shuffle_labels_global,
    shuffle_labels_local_window,
)


# ---------------------------------------------------------------------------
# T1: alternating schedule -> p_opposite >> p_same
# ---------------------------------------------------------------------------
def test_compute_pairing_lift_perfectly_alternating():
    times = np.arange(0.0, 100.0, 1.0)
    labels = np.array([i % 2 for i in range(times.size)], dtype=int)
    block = [(times[0] - 0.1, times[-1] + 0.1)]

    out = compute_pairing_lift(times, labels, delta_t_seconds=1.5, block_time_ranges=block)

    # In a 1.5s window after each event, exactly the next 1 event lies inside.
    # Alternating labels => that next event is always opposite.
    assert out["p_opposite"] == pytest.approx(1.0, abs=0.05)
    assert out["p_same"] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# T2: independent (i.i.d. labels) -> excess approx 0
# ---------------------------------------------------------------------------
def test_compute_pairing_lift_independent():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 1000, size=2000))
    labels = rng.integers(0, 2, size=times.size)
    block = [(times[0] - 0.1, times[-1] + 0.1)]

    out = compute_pairing_lift(times, labels, delta_t_seconds=1.0, block_time_ranges=block)

    # 50/50 labels with rate ~2/s and Δt=1 => expected ~1 event in window,
    # ~50% opposite, ~50% same. Excess ≈ 0.
    excess = out["p_opposite"] - out["p_same"]
    assert abs(excess) < 0.1


# ---------------------------------------------------------------------------
# T3: block boundary must isolate counting -> events in block B not counted from block A
# ---------------------------------------------------------------------------
def test_compute_pairing_lift_block_boundary_isolation():
    # Block A ends at t=10 with label=0; Block B starts at t=11 with label=1
    times = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    block_time_ranges = [(0.0, 10.5), (10.6, 14.0)]

    # Δt=5s spans the block gap from t=10 (label=0) into [11..15]
    # If block-aware: t=10 (label 0) sees nothing in (10, 15] within block A => 0 opposite, 0 same
    # If naive (cross-block): t=10 would count t=11,12,13 as 3 opposite => bug
    out = compute_pairing_lift(
        times, labels, delta_t_seconds=5.0, block_time_ranges=block_time_ranges
    )

    # Per anchor counts:
    #   block A anchors (3 events at 8, 9, 10):
    #     t=8: window (8,13] truncated to block A end 10.5 -> sees t=9, t=10 (both label 0) -> 0 opp, 2 same
    #     t=9: window (9,14] truncated to 10.5 -> sees t=10 (label 0) -> 0 opp, 1 same
    #     t=10: window (10,15] truncated to 10.5 -> empty -> 0 opp, 0 same
    #   block B anchors (3 events at 11, 12, 13):
    #     t=11: window (11,16] truncated to 14.0 -> sees t=12,13 (label 1) -> 0 opp, 2 same
    #     t=12: sees t=13 (label 1) -> 0 opp, 1 same
    #     t=13: empty
    # Total: 0 opposite / 6 events = 0.0; 6 same / 6 events = 1.0
    assert out["p_opposite"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_same"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T4: global shuffle preserves overall label count
# ---------------------------------------------------------------------------
def test_shuffle_labels_global_preserves_count():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 30 + [1] * 70, dtype=int)
    shuffled = shuffle_labels_global(labels, rng)

    assert shuffled.shape == labels.shape
    assert int(np.sum(shuffled == 0)) == 30
    assert int(np.sum(shuffled == 1)) == 70
    # Should not be identical (with high probability)
    assert not np.array_equal(shuffled, labels)


# ---------------------------------------------------------------------------
# T5: block-aware shuffle never crosses block boundary
# ---------------------------------------------------------------------------
def test_shuffle_labels_block_aware_within_block_only():
    rng = np.random.default_rng(2)
    times = np.array([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0])
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=int)
    block_time_ranges = [(0.0, 5.0), (10.0, 15.0)]

    for _ in range(50):
        shuffled = shuffle_labels_block_aware(labels, times, block_time_ranges, rng)
        # Block A (idx 0-3): always 2x0 + 2x1
        block_a = shuffled[:4]
        assert int(np.sum(block_a == 0)) == 2
        assert int(np.sum(block_a == 1)) == 2
        # Block B (idx 4-7): always 2x1 + 2x0
        block_b = shuffled[4:]
        assert int(np.sum(block_b == 1)) == 2
        assert int(np.sum(block_b == 0)) == 2


# ---------------------------------------------------------------------------
# T6: local-window shuffle preserves per-window proportions (modulo boundaries)
# ---------------------------------------------------------------------------
def test_shuffle_labels_local_window_proportions():
    rng = np.random.default_rng(3)
    # 60 minutes of events, one per second
    times = np.arange(0.0, 3600.0, 1.0)
    # First half label=0, second half label=1
    labels = np.array([0] * 1800 + [1] * 1800, dtype=int)
    window_seconds = 1800.0  # 30 min

    shuffled = shuffle_labels_local_window(labels, times, window_seconds, rng)

    assert shuffled.shape == labels.shape
    # Within each non-overlapping 30-min half, counts must be preserved
    assert int(np.sum(shuffled[:1800] == 0)) == 1800
    assert int(np.sum(shuffled[1800:] == 1)) == 1800


# ---------------------------------------------------------------------------
# T7: circular shift preserves lag-1 transition pattern (shift invariance)
# ---------------------------------------------------------------------------
def test_shuffle_labels_circular_preserves_autocorrelation():
    rng = np.random.default_rng(4)
    labels = np.array([0] * 100 + [1] * 100, dtype=int)
    n = labels.size

    def lag1_same(arr):
        return float(np.mean(arr[:-1] == arr[1:]))

    base = lag1_same(labels)

    for _ in range(20):
        shifted = shuffle_labels_circular(labels, rng)
        assert shifted.shape == labels.shape
        assert int(np.sum(shifted == 0)) == 100
        assert int(np.sum(shifted == 1)) == 100
        # Circular shift preserves lag-1 same fraction up to wrap point (1/n flip)
        s = lag1_same(shifted)
        assert abs(s - base) <= 2.0 / n


# ---------------------------------------------------------------------------
# T8: transition odds discriminate alternating vs independent
# ---------------------------------------------------------------------------
def test_compute_transition_odds_alternating_vs_independent():
    # Alternating schedule
    times_alt = np.arange(0.0, 100.0, 1.0)
    labels_alt = np.array([i % 2 for i in range(100)], dtype=int)
    out_alt = compute_transition_odds(times_alt, labels_alt)

    assert out_alt["p_next_opposite"] == pytest.approx(1.0, abs=0.02)
    assert out_alt["transition_odds"] > 50.0  # huge

    # Independent schedule (i.i.d. 50/50)
    rng = np.random.default_rng(5)
    times_ind = np.sort(rng.uniform(0, 1000, size=2000))
    labels_ind = rng.integers(0, 2, size=2000)
    out_ind = compute_transition_odds(times_ind, labels_ind)

    assert abs(out_ind["p_next_opposite"] - 0.5) < 0.05
    assert abs(out_ind["transition_odds"] - out_ind["baseline_odds"]) < 0.2


# ---------------------------------------------------------------------------
# T9: full pipeline with nulls discriminates antagonistic vs independent
# ---------------------------------------------------------------------------
def test_compute_pairing_with_nulls_full_pipeline():
    # Antagonistic schedule (alternating)
    times = np.arange(0.0, 200.0, 1.0)
    labels = np.array([i % 2 for i in range(200)], dtype=int)
    block = [(times[0] - 0.1, times[-1] + 0.1)]

    out = compute_pairing_with_nulls(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block,
        delta_t_grid=(1.5,),
        n_perm=200,
        nulls=("N0", "N1", "N2", "N3"),
        seed=0,
    )

    # Empirical: p_opposite ≈ 1, p_same ≈ 0
    emp = out["empirical"][1.5]
    assert emp["p_opposite"] > 0.9
    assert emp["p_same"] < 0.1

    # Each null should produce excess ≈ 0 on average
    for null_name in ("N0", "N1", "N2", "N3"):
        excess = out["lift"][null_name][1.5]["excess"]
        # Empirical excess (opposite_lift − same_lift) should be strongly positive
        assert excess > 1.0

    # Now an independent schedule
    rng = np.random.default_rng(6)
    times2 = np.sort(rng.uniform(0, 1000, size=1000))
    labels2 = rng.integers(0, 2, size=1000)
    block2 = [(times2[0] - 0.1, times2[-1] + 0.1)]

    out2 = compute_pairing_with_nulls(
        event_abs_times=times2,
        cluster_labels=labels2,
        block_time_ranges=block2,
        delta_t_grid=(1.0,),
        n_perm=200,
        nulls=("N2",),
        seed=0,
    )
    excess2 = out2["lift"]["N2"][1.0]["excess"]
    assert abs(excess2) < 0.3


# ---------------------------------------------------------------------------
# T10: evaluate_pass_criteria triple gate
# ---------------------------------------------------------------------------
def test_evaluate_pass_criteria_triple_gate():
    # case 1: all three gates pass
    res = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.3, "s2": 0.2, "s3": 0.5, "s4": 0.4, "s5": 0.6},
        cohort_excess_30s={"s1": 0.1, "s2": 0.05, "s3": 0.2, "s4": 0.15, "s5": 0.1},
    )
    assert res["pass"] is True
    assert res["wilcoxon_10s"] < 0.05
    assert res["sign_10s"] < 0.05
    assert res["median_30s_positive"] is True

    # case 2: 10s strong but 30s reverses (negative median)
    res2 = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.3, "s2": 0.2, "s3": 0.5, "s4": 0.4, "s5": 0.6},
        cohort_excess_30s={"s1": -0.05, "s2": -0.1, "s3": -0.02, "s4": -0.08, "s5": -0.03},
    )
    assert res2["pass"] is False
    assert res2["median_30s_positive"] is False

    # case 3: 10s wilcoxon fails (mostly small / mixed)
    res3 = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.05, "s2": -0.02, "s3": 0.01, "s4": 0.03, "s5": -0.01},
        cohort_excess_30s={"s1": 0.05, "s2": 0.02, "s3": 0.03, "s4": 0.04, "s5": 0.01},
    )
    assert res3["pass"] is False
    # 10s either wilcoxon or sign should fail
    assert (res3["wilcoxon_10s"] > 0.05) or (res3["sign_10s"] > 0.05)
