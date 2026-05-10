"""TDD tests for PR-6 Step 6 — held-out time template stability.

Plan-of-record:
    docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_plan_2026-05-10.md

This test module covers the asymmetric train/test pipeline:
1. ``_split_events_by_time``           — first/second half time-split helper
2. ``compute_held_out_endpoint_validation`` — train on first half, test on second
3. Endpoint geometric stability edge cases (drift to baseline)
4. Stub / contract guards (block_time_ranges required, n_valid >= 2 * top_n)
5. swap_class concordance via §8 variable-k classifier integration
"""

from __future__ import annotations

import numpy as np
import pytest

from src.interictal_propagation import (
    _split_events_by_time,
    compute_held_out_endpoint_validation,
)


# =====================================================================
# Task 1 — _split_events_by_time
# =====================================================================


def test_split_events_by_time_simple_median():
    """Six events evenly spaced inside one block split at median into 3+3."""
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    block_time_ranges = [(0.0, 60.0)]

    idx_first, idx_second, info = _split_events_by_time(
        times, block_time_ranges
    )

    assert sorted(idx_first.tolist()) == [0, 1, 2]
    assert sorted(idx_second.tolist()) == [3, 4, 5]
    assert info["n_first"] == 3
    assert info["n_second"] == 3
    assert np.isfinite(info["median_t"])


def test_split_events_by_time_unsorted_input():
    """Helper must sort by abs_time internally; input order should not matter."""
    times = np.array([30.0, 0.0, 50.0, 20.0, 10.0, 40.0])
    block_time_ranges = [(0.0, 60.0)]

    idx_first, idx_second, info = _split_events_by_time(
        times, block_time_ranges
    )

    # First half = three smallest times (0, 10, 20) at indices 1, 4, 3
    assert sorted(idx_first.tolist()) == [1, 3, 4]
    assert sorted(idx_second.tolist()) == [0, 2, 5]


def test_split_events_by_time_two_blocks_with_gap():
    """Events span two recording blocks separated by a gap; median split respects time, not block_id."""
    # Block A: t in [0, 100], 4 events
    # gap from 100 to 1000
    # Block B: t in [1000, 1100], 4 events
    times = np.array([10.0, 30.0, 60.0, 90.0, 1010.0, 1030.0, 1060.0, 1090.0])
    block_time_ranges = [(0.0, 100.0), (1000.0, 1100.0)]

    idx_first, idx_second, info = _split_events_by_time(
        times, block_time_ranges
    )

    # Median of times = (90 + 1010) / 2 = 550 → first half = block A, second half = block B
    assert sorted(idx_first.tolist()) == [0, 1, 2, 3]
    assert sorted(idx_second.tolist()) == [4, 5, 6, 7]


def test_split_events_by_time_raises_on_event_outside_block():
    """Event with time outside any block_time_range should raise ValueError (boundary contract)."""
    times = np.array([10.0, 30.0, 200.0])  # t=200 outside [(0,100)]
    block_time_ranges = [(0.0, 100.0)]

    with pytest.raises(ValueError, match=r"outside.*block_time_range"):
        _split_events_by_time(times, block_time_ranges)


def test_split_events_by_time_with_day_night_labels():
    """When day_night_labels passed, info dict carries per-half day/night counts."""
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    labels = np.array(["day", "day", "night", "day", "night", "night"])
    block_time_ranges = [(0.0, 60.0)]

    idx_first, idx_second, info = _split_events_by_time(
        times, block_time_ranges, day_night_labels=labels
    )

    assert info["day_night_ratio_first"]["day"] == 2
    assert info["day_night_ratio_first"]["night"] == 1
    assert info["day_night_ratio_second"]["day"] == 1
    assert info["day_night_ratio_second"]["night"] == 2


def test_split_events_by_time_balanced_mode_requires_labels():
    """balance_day_night=True without day_night_labels must raise ValueError (stub contract)."""
    times = np.array([0.0, 10.0, 20.0, 30.0])
    block_time_ranges = [(0.0, 40.0)]

    with pytest.raises(ValueError, match=r"day_night_labels.*required"):
        _split_events_by_time(
            times, block_time_ranges, balance_day_night=True
        )


def test_split_events_by_time_balanced_mode_distributes_day_night_evenly():
    """balance_day_night=True splits day and night events independently at their respective medians."""
    times = np.arange(8, dtype=float) * 10.0  # 0,10,...,70
    # 4 day events at t=0,20,40,60; 4 night events at t=10,30,50,70
    labels = np.array(["day", "night", "day", "night", "day", "night", "day", "night"])
    block_time_ranges = [(0.0, 80.0)]

    idx_first, idx_second, info = _split_events_by_time(
        times, block_time_ranges, day_night_labels=labels, balance_day_night=True
    )

    # First half should contain 2 day + 2 night; second half similarly
    first_labels = labels[idx_first]
    second_labels = labels[idx_second]
    assert int(np.sum(first_labels == "day")) == 2
    assert int(np.sum(first_labels == "night")) == 2
    assert int(np.sum(second_labels == "day")) == 2
    assert int(np.sum(second_labels == "night")) == 2


# =====================================================================
# Task 2 / 3 — compute_held_out_endpoint_validation (placeholders below;
# implemented after Task 1 passes)
# =====================================================================


def _make_synthetic_subject(
    n_events_per_cluster: int = 100,
    n_valid: int = 10,
    cluster_b_reverse: bool = True,
    second_half_drift: float = 0.0,
    rng_seed: int = 0,
):
    """Build a synthetic subject with stable_k=2 stereotyped templates.

    Cluster A template: rank ascending [1, 2, ..., n_valid] (channel 0 fastest).
    Cluster B template: descending if cluster_b_reverse else ascending again.

    second_half_drift in [0, 1]: 0 = test half identical to train half,
    1 = test half completely random ranks.

    Returns dict keyed like load_subject_propagation_events output, plus
    pre-computed adaptive cluster labels for k=2.
    """
    rng = np.random.default_rng(rng_seed)
    template_a = np.arange(1, n_valid + 1, dtype=float)
    template_b = template_a[::-1].copy() if cluster_b_reverse else template_a.copy()

    n_total = 2 * n_events_per_cluster

    # Time order: alternate clusters across full duration so first/second halves
    # each contain ~equal mix of cluster A and B.
    abs_times = np.linspace(0.0, 1000.0, n_total)
    block_time_ranges = [(0.0, 1000.0)]
    block_ids = np.zeros(n_total, dtype=int)
    labels = np.zeros(n_total, dtype=int)
    labels[1::2] = 1  # cluster B at odd indices

    ranks = np.zeros((n_valid, n_total), dtype=float)
    bools = np.ones((n_valid, n_total), dtype=bool)

    for ev in range(n_total):
        in_first_half = abs_times[ev] <= np.median(abs_times)
        cluster = int(labels[ev])
        base_template = template_a if cluster == 0 else template_b

        if in_first_half:
            # Train half: tight stereotypy with mild noise
            noisy = base_template + rng.normal(0.0, 0.3, n_valid)
        else:
            # Test half: blend base_template with random based on drift
            random_template = rng.permutation(np.arange(1, n_valid + 1)).astype(float)
            noisy = (1 - second_half_drift) * base_template + second_half_drift * random_template
            noisy = noisy + rng.normal(0.0, 0.3, n_valid)

        # Convert to integer rank: argsort-of-argsort then +1
        rk = np.argsort(np.argsort(noisy)) + 1
        ranks[:, ev] = rk

    return {
        "ranks": ranks,
        "bools": bools,
        "event_abs_times": abs_times,
        "block_ids": block_ids,
        "block_time_ranges": block_time_ranges,
        "channel_names": [f"ch{i:02d}" for i in range(n_valid)],
        "valid_event_indices": np.arange(n_total, dtype=int),
        "adaptive_labels": labels,
        "valid_mask": np.ones(n_valid, dtype=bool),
    }


def test_held_out_validation_recovers_train_template_on_zero_drift():
    """Synthetic stereotyped subject with no test-half drift → 4 metrics all strong."""
    subj = _make_synthetic_subject(
        n_events_per_cluster=200, n_valid=10, second_half_drift=0.0, rng_seed=0
    )

    result = compute_held_out_endpoint_validation(
        ranks=subj["ranks"],
        bools=subj["bools"],
        event_abs_times=subj["event_abs_times"],
        block_ids=subj["block_ids"],
        block_time_ranges=subj["block_time_ranges"],
        chosen_k=2,
        valid_event_indices=subj["valid_event_indices"],
        channel_names=subj["channel_names"],
        soz_channels=set(),
    )

    val = result["validation"]
    assert val["template_spearman"] > 0.7, val
    assert val["endpoint_position_recall"] > 0.6, val
    assert val["swap_class_concordant"] is True, val
    assert val["tier"] == "strong", val


def test_held_out_validation_endpoint_drifts_to_middle_under_random_test_half():
    """When test half is fully randomized rank, endpoint_position_recall ≈ baseline."""
    subj = _make_synthetic_subject(
        n_events_per_cluster=200, n_valid=10, second_half_drift=1.0, rng_seed=1
    )

    result = compute_held_out_endpoint_validation(
        ranks=subj["ranks"],
        bools=subj["bools"],
        event_abs_times=subj["event_abs_times"],
        block_ids=subj["block_ids"],
        block_time_ranges=subj["block_time_ranges"],
        chosen_k=2,
        valid_event_indices=subj["valid_event_indices"],
        channel_names=subj["channel_names"],
        soz_channels=set(),
    )

    # n_valid=10, direction-preserving baseline ~0.3 under random ranks.
    # template_spearman has a selection-bias floor (assign_events_to_templates
    # picks nearest template even on random data), so the strict-tier
    # threshold 0.7 is the real diagnostic — expect it NOT to clear.
    val = result["validation"]
    assert val["template_spearman"] < 0.75, val
    assert val["endpoint_position_recall"] < 0.65, val
    assert val["swap_class_concordant"] is False, val
    assert val["tier"] in {"weak", "fail"}, val


# =====================================================================
# Task 4 — stub / contract guards
# =====================================================================


def test_held_out_validation_raises_on_missing_block_time_ranges():
    """block_time_ranges is required (no default); passing None must raise."""
    subj = _make_synthetic_subject(rng_seed=0)
    with pytest.raises((ValueError, TypeError)):
        compute_held_out_endpoint_validation(
            ranks=subj["ranks"],
            bools=subj["bools"],
            event_abs_times=subj["event_abs_times"],
            block_ids=subj["block_ids"],
            block_time_ranges=None,
            chosen_k=2,
            valid_event_indices=subj["valid_event_indices"],
            channel_names=subj["channel_names"],
            soz_channels=set(),
        )


def test_held_out_validation_raises_on_n_valid_below_endpoint_threshold():
    """n_valid < 2 * endpoint_top_n must raise (PR-6 endpoint_defined contract)."""
    # endpoint_top_n=3 default → need n_valid >= 6
    subj = _make_synthetic_subject(n_valid=5, rng_seed=0)
    with pytest.raises(ValueError, match=r"n_valid.*endpoint"):
        compute_held_out_endpoint_validation(
            ranks=subj["ranks"],
            bools=subj["bools"],
            event_abs_times=subj["event_abs_times"],
            block_ids=subj["block_ids"],
            block_time_ranges=subj["block_time_ranges"],
            chosen_k=2,
            valid_event_indices=subj["valid_event_indices"],
            channel_names=subj["channel_names"],
            soz_channels=set(),
        )


# =====================================================================
# Task 5 — swap_class concordance integration
# =====================================================================


def test_held_out_validation_swap_class_concordance_when_geometry_preserved():
    """Geometry-preserved test half should yield the same swap_class label."""
    subj = _make_synthetic_subject(
        n_events_per_cluster=200, n_valid=10, second_half_drift=0.0, rng_seed=0
    )

    result = compute_held_out_endpoint_validation(
        ranks=subj["ranks"],
        bools=subj["bools"],
        event_abs_times=subj["event_abs_times"],
        block_ids=subj["block_ids"],
        block_time_ranges=subj["block_time_ranges"],
        chosen_k=2,
        valid_event_indices=subj["valid_event_indices"],
        channel_names=subj["channel_names"],
        soz_channels=set(),
    )

    fh = result["first_half"]
    sh = result["second_half"]
    assert fh["swap_class"] == sh["swap_class_projected"]
    assert result["validation"]["swap_class_concordant"] is True
