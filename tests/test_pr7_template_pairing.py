"""TDD tests for PR-7 template antagonistic temporal pairing.

Contract: docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md §8.
Includes T11–T15 added 2026-04-28 in response to scientific-contract review:
- T11: N2 50% overlap + first-covering-window partition behavior
- T12: N2 must not cross block boundaries
- T13: evaluate_pass_criteria enforces subject-key match
- T14: compute_transition_odds is block-aware (no cross-block transitions)
- T15: H1b direction asymmetry visible via p_a_to_b vs p_b_to_a fields
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
    resample_isi_per_cluster,
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

    excess = out["p_opposite"] - out["p_same"]
    assert abs(excess) < 0.1


# ---------------------------------------------------------------------------
# T3: block boundary must isolate counting
# ---------------------------------------------------------------------------
def test_compute_pairing_lift_block_boundary_isolation():
    times = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    block_time_ranges = [(0.0, 10.5), (10.6, 14.0)]

    out = compute_pairing_lift(
        times, labels, delta_t_seconds=5.0, block_time_ranges=block_time_ranges
    )

    # Cross-block events filtered: see plan analysis in test docstring.
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
        block_a = shuffled[:4]
        assert int(np.sum(block_a == 0)) == 2
        assert int(np.sum(block_a == 1)) == 2
        block_b = shuffled[4:]
        assert int(np.sum(block_b == 1)) == 2
        assert int(np.sum(block_b == 0)) == 2


# ---------------------------------------------------------------------------
# T6: local-window shuffle preserves per-window counts (single block, simple case)
# ---------------------------------------------------------------------------
def test_shuffle_labels_local_window_proportions():
    rng = np.random.default_rng(3)
    times = np.arange(0.0, 3600.0, 1.0)
    # First half label=0, second half label=1
    labels = np.array([0] * 1800 + [1] * 1800, dtype=int)
    window_seconds = 1800.0  # 30 min
    block_time_ranges = [(0.0, 3600.0)]

    shuffled = shuffle_labels_local_window(
        labels, times, window_seconds, block_time_ranges, rng
    )

    assert shuffled.shape == labels.shape
    # First-covering rule with 50% overlap puts:
    #   pool 0 = events at t in [0, 1800)  -> 1800 events all label 0
    #   pool 1 = events at t in [1800, 2700) -> 900 events all label 1
    #   pool 2 = events at t in [2700, 3600) -> 900 events all label 1
    # All pools homogeneous => permutation is identity in label space.
    assert int(np.sum(shuffled[:1800] == 0)) == 1800
    assert int(np.sum(shuffled[1800:] == 1)) == 1800


# ---------------------------------------------------------------------------
# T7: circular shift preserves lag-1 same fraction (shift invariance)
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
        s = lag1_same(shifted)
        assert abs(s - base) <= 2.0 / n


# ---------------------------------------------------------------------------
# T8: transition odds discriminate alternating vs independent (block-aware)
# ---------------------------------------------------------------------------
def test_compute_transition_odds_alternating_vs_independent():
    times_alt = np.arange(0.0, 100.0, 1.0)
    labels_alt = np.array([i % 2 for i in range(100)], dtype=int)
    block_alt = [(times_alt[0] - 0.1, times_alt[-1] + 0.1)]
    out_alt = compute_transition_odds(times_alt, labels_alt, block_alt)

    assert out_alt["p_next_opposite"] == pytest.approx(1.0, abs=0.02)
    assert out_alt["transition_odds"] > 50.0

    rng = np.random.default_rng(5)
    times_ind = np.sort(rng.uniform(0, 1000, size=2000))
    labels_ind = rng.integers(0, 2, size=2000)
    block_ind = [(times_ind[0] - 0.1, times_ind[-1] + 0.1)]
    out_ind = compute_transition_odds(times_ind, labels_ind, block_ind)

    assert abs(out_ind["p_next_opposite"] - 0.5) < 0.05
    assert abs(out_ind["transition_odds"] - out_ind["baseline_odds"]) < 0.2


# ---------------------------------------------------------------------------
# T9: full pipeline with nulls discriminates antagonistic vs independent
# ---------------------------------------------------------------------------
def test_compute_pairing_with_nulls_full_pipeline():
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
        n2_window_seconds=50.0,  # short window for short test span
        seed=0,
    )

    emp = out["empirical"][1.5]
    assert emp["p_opposite"] > 0.9
    assert emp["p_same"] < 0.1

    for null_name in ("N0", "N1", "N2", "N3"):
        excess = out["lift"][null_name][1.5]["excess"]
        assert excess > 1.0

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
        n2_window_seconds=300.0,
        seed=0,
    )
    excess2 = out2["lift"]["N2"][1.0]["excess"]
    assert abs(excess2) < 0.3


# ---------------------------------------------------------------------------
# T10: evaluate_pass_criteria triple gate
# ---------------------------------------------------------------------------
def test_evaluate_pass_criteria_triple_gate():
    res = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.3, "s2": 0.2, "s3": 0.5, "s4": 0.4, "s5": 0.6},
        cohort_excess_30s={"s1": 0.1, "s2": 0.05, "s3": 0.2, "s4": 0.15, "s5": 0.1},
    )
    assert res["pass"] is True
    assert res["wilcoxon_10s"] < 0.05
    assert res["sign_10s"] < 0.05
    assert res["median_30s_positive"] is True

    res2 = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.3, "s2": 0.2, "s3": 0.5, "s4": 0.4, "s5": 0.6},
        cohort_excess_30s={"s1": -0.05, "s2": -0.1, "s3": -0.02, "s4": -0.08, "s5": -0.03},
    )
    assert res2["pass"] is False
    assert res2["median_30s_positive"] is False

    res3 = evaluate_pass_criteria(
        cohort_excess_10s={"s1": 0.05, "s2": -0.02, "s3": 0.01, "s4": 0.03, "s5": -0.01},
        cohort_excess_30s={"s1": 0.05, "s2": 0.02, "s3": 0.03, "s4": 0.04, "s5": 0.01},
    )
    assert res3["pass"] is False
    assert (res3["wilcoxon_10s"] > 0.05) or (res3["sign_10s"] > 0.05)


# ---------------------------------------------------------------------------
# T11: N2 50% overlap + first-covering rule (precise partition test)
# ---------------------------------------------------------------------------
def test_n2_first_covering_partition_with_50pct_overlap():
    """Verify N2 partitions events into first-covering windows with 50% step.

    Construct labels that diagnose the partition:
      - t in [0, 900):   label 0  -> first-covering window 0 (covers [0, 1800))
      - t in [900, 1800): label 1 -> first-covering window 0 (still covers it)
      - t in [1800, 2700): label 0 -> first-covering window 1 (covers [900, 2700))
      - t in [2700, 3600): label 1 -> first-covering window 2 (covers [1800, 3600))

    After shuffle:
      - pool 0 (events at t in [0, 1800)) has 900 zeros + 900 ones.
        Permutation produces some arrangement but total counts preserved.
      - pool 1 (events at t in [1800, 2700)) is homogeneous: 900 zeros.
      - pool 2 (events at t in [2700, 3600)) is homogeneous: 900 ones.

    The diagnostic: pool 0 mixes 0s and 1s; pool 1 stays all 0; pool 2 stays all 1.
    """
    rng = np.random.default_rng(11)
    times = np.arange(0.0, 3600.0, 1.0)
    labels = np.concatenate(
        [
            np.zeros(900, dtype=int),  # [0, 900)   label 0
            np.ones(900, dtype=int),   # [900, 1800) label 1
            np.zeros(900, dtype=int),  # [1800, 2700) label 0
            np.ones(900, dtype=int),   # [2700, 3600) label 1
        ]
    )
    block_time_ranges = [(0.0, 3600.0)]

    shuffled = shuffle_labels_local_window(
        labels, times, window_seconds=1800.0, block_time_ranges=block_time_ranges, rng=rng
    )

    # Pool 0 (idx 0..1799): contains 900x label0 + 900x label1; counts preserved
    pool0 = shuffled[:1800]
    assert int(np.sum(pool0 == 0)) == 900
    assert int(np.sum(pool0 == 1)) == 900
    # Some 0s should appear in the second-half of the pool indices (mixing)
    assert int(np.sum(pool0[900:] == 0)) > 0  # actual mixing happened

    # Pool 1 (idx 1800..2699): homogeneous label 0 -> stays all 0
    assert int(np.sum(shuffled[1800:2700] == 0)) == 900

    # Pool 2 (idx 2700..3599): homogeneous label 1 -> stays all 1
    assert int(np.sum(shuffled[2700:3600] == 1)) == 900


# ---------------------------------------------------------------------------
# T12: N2 must not cross block boundaries
# ---------------------------------------------------------------------------
def test_n2_does_not_cross_block_boundaries():
    """Two blocks separated by a gap. Block A is all label 0, block B is all
    label 1. N2 with a window large enough to span both blocks must NOT
    shuffle labels across the block gap.
    """
    rng = np.random.default_rng(12)
    # Block A at [0, 1000), 100 events all label 0
    # Gap [1000, 5000)
    # Block B at [5000, 6000), 100 events all label 1
    times_a = np.linspace(0.0, 999.0, 100)
    times_b = np.linspace(5000.0, 5999.0, 100)
    times = np.concatenate([times_a, times_b])
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    block_time_ranges = [(0.0, 1000.0), (5000.0, 6000.0)]

    # Use a HUGE window (10000s) that would, if not block-aware, lump
    # everything into one pool and mix labels across blocks.
    for _ in range(20):
        shuffled = shuffle_labels_local_window(
            labels, times, window_seconds=10000.0,
            block_time_ranges=block_time_ranges, rng=rng,
        )
        # Block A indices stay all 0
        assert int(np.sum(shuffled[:100] == 0)) == 100
        # Block B indices stay all 1
        assert int(np.sum(shuffled[100:] == 1)) == 100


# ---------------------------------------------------------------------------
# T13: evaluate_pass_criteria enforces subject-key match
# ---------------------------------------------------------------------------
def test_evaluate_pass_criteria_rejects_key_mismatch():
    """Subject-level paired design REQUIRES identical subject keys for 10s
    and 30s cohort dicts. Mismatch must raise ValueError."""
    excess_10s = {"s1": 0.3, "s2": 0.2, "s3": 0.5}
    excess_30s_extra = {"s1": 0.1, "s2": 0.05, "s4": 0.2}  # s3 -> s4 mismatch

    with pytest.raises(ValueError, match="key mismatch"):
        evaluate_pass_criteria(excess_10s, excess_30s_extra)

    excess_30s_missing = {"s1": 0.1, "s2": 0.05}  # s3 missing
    with pytest.raises(ValueError, match="key mismatch"):
        evaluate_pass_criteria(excess_10s, excess_30s_missing)


# ---------------------------------------------------------------------------
# T14: transition odds is block-aware (cross-block pairs must not count)
# ---------------------------------------------------------------------------
def test_transition_odds_does_not_cross_blocks():
    """Block A all label 0, big gap, Block B all label 1.

    Naively (cross-block): the transition from last event of A (label 0) to
    first event of B (label 1) would be an OPPOSITE transition.

    Block-aware: that pair is dropped. Transition odds within block A are 0
    same-different (all same), within block B 0 same-different, total
    next-opposite count is 0.
    """
    times_a = np.linspace(0.0, 100.0, 50)
    times_b = np.linspace(10000.0, 10100.0, 50)
    times = np.concatenate([times_a, times_b])
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    block_time_ranges = [(0.0, 200.0), (9990.0, 10200.0)]

    out = compute_transition_odds(times, labels, block_time_ranges)

    # All within-block transitions are SAME (label 0->0 or 1->1).
    assert out["p_next_opposite"] == pytest.approx(0.0, abs=1e-9)
    # n_pairs counts only same-block consecutive pairs: 49 + 49 = 98
    assert out["n_pairs"] == 98

    # Sanity: if we DROP the block constraint by making everything one block,
    # the cross-block pair WOULD be counted as opposite (1 transition out of 99).
    one_block = [(0.0, 11000.0)]
    out_naive = compute_transition_odds(times, labels, one_block)
    assert out_naive["p_next_opposite"] == pytest.approx(1.0 / 99.0, abs=1e-6)


# ---------------------------------------------------------------------------
# T15: H1b direction asymmetry visible in p_a_to_b vs p_b_to_a
# ---------------------------------------------------------------------------
def test_pairing_lift_direction_asymmetry():
    """Construct a schedule where T_a (label 0) is reliably followed by T_b
    (label 1) but T_b is rarely followed by T_a — i.e. an asymmetric pairing.
    The directional lift fields must reflect the asymmetry.
    """
    # Schedule: 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, ...
    # Each label-0 event is followed by exactly one nearby label-1 event.
    # Each label-1 event is mostly followed by another label-1, then occasionally label-0.
    times = np.arange(0.0, 100.0, 1.0)
    labels = []
    for i in range(100):
        if i % 5 == 0:
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels, dtype=int)
    block = [(times[0] - 0.1, times[-1] + 0.1)]

    out = compute_pairing_lift(times, labels, delta_t_seconds=1.5, block_time_ranges=block)

    # Per anchor of class a (label 0), almost always sees one label-1 next.
    # p_a_to_b should be near 1.0
    # Per anchor of class b (label 1), sees label-1 (same) more often than label-0.
    # p_b_to_a should be much smaller than p_a_to_b
    assert out["p_a_to_b"] > 0.9
    assert out["p_b_to_a"] < 0.3
    assert out["p_a_to_b"] - out["p_b_to_a"] > 0.5  # clear asymmetry

    # Anchor counts are accurate
    assert out["n_a_anchors"] == 20  # every 5th event is label 0
    assert out["n_b_anchors"] == 80


# ---------------------------------------------------------------------------
# T16 (extra): N4 stub raises NotImplementedError
# ---------------------------------------------------------------------------
def test_resample_isi_per_cluster_raises():
    """N4 is a conditional follow-up surrogate; the stub MUST raise
    NotImplementedError so it cannot silently leak into primary results."""
    rng = np.random.default_rng(99)
    with pytest.raises(NotImplementedError, match="follow-up"):
        resample_isi_per_cluster(
            event_abs_times=np.array([1.0, 2.0]),
            cluster_labels=np.array([0, 1]),
            local_window_seconds=300.0,
            rng=rng,
        )
