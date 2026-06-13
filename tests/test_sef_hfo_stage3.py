"""TDD for src/sef_hfo_stage3 — pure helpers for the Stage 3 stochastic-template-train pilot.

Covers the core-level onset + collision/source-labeling contract from
docs/superpowers/specs/2026-06-13-sef-hfo-snn-stage3-stochastic-template-train-design.md §1-§2.
Each test pins ONE contract clause (CLAUDE.md §6: surrogate/labeling construction details ARE the
contract — every non-trivial clause needs a test that would fail if violated).
"""
import numpy as np

from src.sef_hfo_stage3 import (
    core_participation_threshold,
    first_crossing_time,
    label_event,
    collision_free_blocks,
    synthetic_label_sequence,
    entry_jitter_stats,
)


# --- core_participation_threshold = max(0.01, n_min / n_core_cells) (spec §1) ---

def test_threshold_one_percent_floor_dominates_for_large_core():
    # n_min/n_core = 5/2000 = 0.0025 < 0.01 -> the 1% floor wins.
    assert core_participation_threshold(n_core_cells=2000, n_min=5) == 0.01


def test_threshold_count_dominates_for_small_core():
    # n_min/n_core = 5/100 = 0.05 > 0.01 -> the >=N_min count wins.
    assert core_participation_threshold(n_core_cells=100, n_min=5) == 0.05


# --- first_crossing_time: first bin >= threshold, in absolute ms (spec §1) ---

def test_first_crossing_returns_absolute_time_of_first_bin_at_or_above_threshold():
    series = np.array([0.0, 0.005, 0.02, 0.5])   # first >= 0.01 is index 2
    # bin_w=1.0 ms, window starts at t_offset=10.0 ms -> crossing at 10 + 2*1 = 12.0
    assert first_crossing_time(series, bin_w=1.0, threshold=0.01, t_offset=10.0) == 12.0


def test_first_crossing_returns_none_when_never_crosses():
    series = np.array([0.0, 0.001, 0.005])
    assert first_crossing_time(series, bin_w=1.0, threshold=0.01, t_offset=0.0) is None


# --- label_event: neg/pos/collision/ambiguous (spec §1, §2) ---

def test_label_neg_when_neg_core_ignites_clearly_first():
    # neg_onset + delta < pos_onset -> 'neg'
    assert label_event(onset_neg=10.0, onset_pos=14.0, delta_onset=1.0, readable=True) == "neg"


def test_label_pos_when_pos_core_ignites_clearly_first():
    assert label_event(onset_neg=14.0, onset_pos=10.0, delta_onset=1.0, readable=True) == "pos"


def test_label_collision_when_onsets_within_delta():
    # |neg - pos| = 0.4 <= delta 1.0 -> both ends ignite near-simultaneously
    assert label_event(onset_neg=10.0, onset_pos=10.4, delta_onset=1.0, readable=True) == "collision"


def test_label_ambiguous_when_axis_unreadable():
    # readable=False (n_part < PART_MIN or axis_err is None) overrides any onset values.
    assert label_event(onset_neg=10.0, onset_pos=14.0, delta_onset=1.0, readable=False) == "ambiguous"


def test_label_ambiguous_when_neither_core_crosses():
    # both onsets None (no_core_crossing) -> ambiguous, even if 'readable'.
    assert label_event(onset_neg=None, onset_pos=None, delta_onset=1.0, readable=True) == "ambiguous"


def test_label_single_core_crossing_is_that_core_not_collision():
    # Only neg crossed (pos None). None is treated as +inf, so neg is the unambiguous source.
    # This must NOT be 'collision' (|neg - inf| is not <= delta) nor 'ambiguous' (one end DID fire).
    assert label_event(onset_neg=10.0, onset_pos=None, delta_onset=1.0, readable=True) == "neg"
    assert label_event(onset_neg=None, onset_pos=10.0, delta_onset=1.0, readable=True) == "pos"


# --- collision_free_blocks: collisions as sequence boundaries (spec §2, P1-1) ---

def test_collision_free_blocks_segments_at_censored_events():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    clean = np.array([True, True, False, True, True])
    blocks, block_id = collision_free_blocks(times, clean)
    assert blocks == [(0.0, 1.0), (3.0, 4.0)]      # two segments; censored event between them
    assert list(block_id) == [0, 0, -1, 1, 1]      # censored -> -1


def test_collision_free_blocks_isolated_clean_event_is_its_own_block():
    times = np.array([0.0, 1.0, 2.0])
    clean = np.array([False, True, False])
    blocks, block_id = collision_free_blocks(times, clean)
    assert blocks == [(1.0, 1.0)]
    assert list(block_id) == [-1, 0, -1]


def test_collision_free_blocks_unsorted_input_is_time_ordered():
    times = np.array([4.0, 0.0, 2.0, 1.0, 3.0])    # same as test 1 but scrambled
    clean = np.array([True, True, False, True, True])  # aligned to the scrambled order
    blocks, block_id = collision_free_blocks(times, clean)
    # time order is 0(clean),1(clean),2(censored),3(clean),4(clean) -> same blocks
    assert blocks == [(0.0, 1.0), (3.0, 4.0)]


# --- synthetic_label_sequence: marginal-preserving controls (spec §4, P1-5) ---

def test_synthetic_alternating_preserves_marginal_and_alternates():
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="alternating", rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())   # marginal preserved
    assert int(np.sum(out[:-1] == out[1:])) == 0             # equal counts -> 0 same-adjacent


def test_synthetic_sticky_makes_two_runs():
    labels = np.array([0, 1, 0, 1, 0, 1])
    out = synthetic_label_sequence(labels, mode="sticky", rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())
    n_runs = 1 + int(np.sum(out[1:] != out[:-1]))
    assert n_runs == 2                                       # all-A then all-B


def test_synthetic_shuffle_preserves_marginal():
    labels = np.array([0, 0, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="shuffle", rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())


# --- entry_jitter_stats: first-active contact dispersion (spec §3.3) ---

def test_entry_jitter_stats_single_fixed_contact_has_zero_dispersion():
    s = entry_jitter_stats(["A0", "A0", "A0"])
    assert s["n_unique"] == 1
    assert s["top1_fraction"] == 1.0


def test_entry_jitter_stats_wandering_group():
    s = entry_jitter_stats(["A0", "A1", "A0", "A2", "A0", "A1"])
    assert s["n_unique"] == 3
    assert 0.0 < s["top1_fraction"] < 1.0
    assert s["top3_fraction"] == 1.0
