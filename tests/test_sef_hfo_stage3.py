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
    core_active_fraction,
    build_sidecar,
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


# --- synthetic controls must be BLOCK-AWARE (P1-b: collision splits the sequence; the control
#     has to be maximally dependent AS MEASURED by the block-aware timing test, which resets runs
#     at block boundaries — so generate it independently within each collision-free block) ---

def test_synthetic_alternating_is_block_local_when_block_id_given():
    # block 0 = events 0-3 (counts 2,2); censored event 4 (block -1); block 1 = events 5-8 (2,2).
    labels   = np.array([0, 0, 1, 1,   1,   0, 1, 0, 1])
    block_id = np.array([0, 0, 0, 0,  -1,   1, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="alternating",
                                   rng=np.random.default_rng(0), block_id=block_id)
    for b in (0, 1):
        seg = out[block_id == b]
        assert int(np.sum(seg[:-1] == seg[1:])) == 0          # alternating WITHIN each block
        assert sorted(seg.tolist()) == [0, 0, 1, 1]            # per-block marginal preserved
    assert out[4] == labels[4]                                 # censored event untouched


def test_synthetic_sticky_is_block_local_when_block_id_given():
    # each block must collapse to TWO runs internally (AABB), not one global run spanning blocks.
    labels   = np.array([0, 1, 0, 1,   0, 1, 0, 1])
    block_id = np.array([0, 0, 0, 0,   1, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="sticky",
                                   rng=np.random.default_rng(0), block_id=block_id)
    for b in (0, 1):
        seg = out[block_id == b]
        n_runs = 1 + int(np.sum(seg[1:] != seg[:-1]))
        assert n_runs == 2
        assert sorted(seg.tolist()) == [0, 0, 1, 1]


def test_synthetic_block_local_tolerates_single_class_block():
    # a block where only one end fired (all 0) cannot alternate -> stays all-0, no raise.
    labels   = np.array([0, 0,   0, 1])
    block_id = np.array([0, 0,   1, 1])
    out = synthetic_label_sequence(labels, mode="alternating",
                                   rng=np.random.default_rng(0), block_id=block_id)
    assert out[block_id == 0].tolist() == [0, 0]               # single-class block unchanged


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


# --- Task 6 writer contract: build_sidecar aligns to returned events + core-level label (P1-3/P1-4) ---
# Unit-tested WITHOUT a sim: a tiny cm-SNN can't reach the discrete-event regime (it saturates),
# so the writer contract is pinned on synthetic spikes here; the real sim path is the pilot (Task 7).

def _spk_two_end(n_steps=300, NE=10):
    """Synthetic E-spike matrix: neg cells 0-4 ignite at bin0 of windows [0,10] and [20,30] ms;
    pos cells 5-9 ignite only at bin5 of window [0,10] ms."""
    spk = np.zeros((n_steps, NE), bool)
    spk[0:10, 0:5] = True       # event0 [0,10]: neg @ bin0 (steps 0-9, dt=0.1 -> 0-1ms)
    spk[50:60, 5:10] = True     # event0 [0,10]: pos @ bin5 (steps 50-59 -> 5-6ms)
    spk[200:210, 0:5] = True    # event2 [20,30]: neg @ bin0-of-window
    return spk


def _ev(t_on, t_off, returned, axis_err=5.0):
    return dict(t_on=t_on, t_off=t_off, returned=returned, event_peak_t=t_on + 2.0,
                n_part=8, axis_err=axis_err, sign=1.0, readability=0.9)


def test_build_sidecar_aligns_to_returned_events_only():
    spk = _spk_two_end()
    core_masks = [np.array([True] * 5 + [False] * 5), np.array([False] * 5 + [True] * 5)]
    ev_recs = [_ev(0.0, 10.0, True), _ev(10.0, 20.0, False), _ev(20.0, 30.0, True)]
    payload = build_sidecar(ev_recs, spk, core_masks, NE=10, dt=0.1, bin_ms=1.0,
                            part_min=7, delta_onset=1.0, n_min=1)
    ev = payload["events"]
    # only the 2 RETURNED events; event_id contiguous; raw_event_index keeps original positions
    assert payload["n_record_events"] == 2
    assert [e["event_id"] for e in ev] == [0, 1]
    assert [e["raw_event_index"] for e in ev] == [0, 2]
    # t_on preserved -> packedTimes (t_on/1000) align to the record by construction (P1-3)
    assert [e["t_on"] for e in ev] == [0.0, 20.0]


def test_build_sidecar_labels_earlier_core_as_source():
    spk = _spk_two_end()
    core_masks = [np.array([True] * 5 + [False] * 5), np.array([False] * 5 + [True] * 5)]
    payload = build_sidecar([_ev(0.0, 10.0, True)], spk, core_masks, NE=10, dt=0.1, bin_ms=1.0,
                            part_min=7, delta_onset=1.0, n_min=1)
    # neg ignites at ~0ms, pos at ~5ms, |Δ|=5 > delta 1 -> 'neg', clean for timing
    assert payload["events"][0]["hidden_source_label"] == "neg"
    assert payload["events"][0]["clean_for_timing"] is True


def test_build_sidecar_unreadable_axis_is_ambiguous_and_not_clean():
    spk = _spk_two_end()
    core_masks = [np.array([True] * 5 + [False] * 5), np.array([False] * 5 + [True] * 5)]
    # axis_err None (unreadable) -> ambiguous regardless of onsets, never clean_for_timing
    payload = build_sidecar([_ev(0.0, 10.0, True, axis_err=None)], spk, core_masks, NE=10,
                            dt=0.1, bin_ms=1.0, part_min=7, delta_onset=1.0, n_min=1)
    e0 = payload["events"][0]
    assert e0["hidden_source_label"] == "ambiguous"
    assert e0["collision_reason"] == "unreadable_axis"
    assert e0["clean_for_timing"] is False


def test_core_active_fraction_window_binning():
    spk = _spk_two_end()
    # neg cells 0-4 all fire in bin0 of [0,10] -> af[0] == 1.0
    af = core_active_fraction(spk, np.arange(5), dt=0.1, bin_ms=1.0, t_on=0.0, t_off=10.0)
    assert af.shape[0] == 10 and af[0] == 1.0 and af[5] == 0.0


# --- sidecar diagnostic fields (user 2026-06-13: distinguish REAL co-ignition from a too-sensitive
#     1% onset threshold, and tag block membership) — needs per-core first-bin curves + block_id ---

def test_build_sidecar_saves_per_core_first_bins():
    # [0,10]: neg core fully active in bin0, pos core silent until bin5. The first-bin curves make
    # 'is this a real co-ignition or did one core just tickle the 1% threshold' auditable offline.
    spk = _spk_two_end()
    core_masks = [np.array([True] * 5 + [False] * 5), np.array([False] * 5 + [True] * 5)]
    payload = build_sidecar([_ev(0.0, 10.0, True)], spk, core_masks, NE=10, dt=0.1, bin_ms=1.0,
                            part_min=7, delta_onset=1.0, n_min=1)
    e0 = payload["events"][0]
    assert e0["core_frac_neg_first_bins"][0] == 1.0     # neg fully active in bin0
    assert e0["core_frac_pos_first_bins"][0] == 0.0     # pos silent in bin0
    assert e0["core_frac_pos_first_bins"][5] == 1.0     # pos ramps at bin5 (captured, not a stray spike)
    # n_min=1 over a 5-cell core -> max(0.01, 1/5) = 0.2 (the N_min count dominates on a tiny core;
    # the 1% floor only wins on the real ~587-cell pilot core).
    assert e0["core_threshold_neg"] == 0.2 and e0["core_threshold_pos"] == 0.2


def test_build_sidecar_assigns_block_id_censoring_breaks_run():
    # clean, censored(unreadable), clean -> block_id 0, -1, 1 (reuses collision_free_blocks).
    spk = _spk_two_end()
    core_masks = [np.array([True] * 5 + [False] * 5), np.array([False] * 5 + [True] * 5)]
    ev_recs = [_ev(0.0, 10.0, True), _ev(10.0, 20.0, True, axis_err=None), _ev(20.0, 30.0, True)]
    payload = build_sidecar(ev_recs, spk, core_masks, NE=10, dt=0.1, bin_ms=1.0,
                            part_min=7, delta_onset=1.0, n_min=1)
    assert [e["block_id_after_collision_censoring"] for e in payload["events"]] == [0, -1, 1]
