import numpy as np
import pytest

from src.sef_hfo_axial_intervention import (
    band_mask, split_near_target_far, core_source_raw,
    participation_ratio, exclude_target_contacts,
    CLAMP_LEVEL, intervention_vth_at_time, make_on_axis_target,
    make_off_axis_target, make_static_deadzone_schedule,
    baseline_eligibility, select_first_eligible_event,
    build_replay_schedule, build_late_schedule,
)


# ---------------- Task 1: geometry + source helpers ----------------

def test_band_mask_perpendicular_to_axis():
    axis = np.array([1.0, 0.0]); center = np.array([5.0, 5.0])
    coords = np.array([[5.0, 5.0], [5.4, 9.0], [5.6, 0.0], [8.0, 5.0]])
    m = band_mask(coords, axis, center, thickness=1.0)   # |x-5| <= 0.5
    assert m.tolist() == [True, True, False, False]


def test_split_near_target_far_orients_by_source():
    axis = np.array([1.0, 0.0]); center = np.array([5.0, 5.0])
    coords = np.array([[9.0, 5.0], [1.0, 5.0], [5.0, 5.0]])
    s = split_near_target_far(coords, axis, center, np.array([9.0, 5.0]), target_thickness=1.0)
    assert s["near"].tolist() == [True, False, False]
    assert s["far"].tolist() == [False, True, False]
    assert s["target"].tolist() == [False, False, True]
    # source on the -axis side flips near/far
    s2 = split_near_target_far(coords, axis, center, np.array([1.0, 5.0]), target_thickness=1.0)
    assert s2["near"].tolist() == [False, True, False]
    assert s2["far"].tolist() == [True, False, False]


def test_core_source_raw_independent_of_readability():
    assert core_source_raw(10.0, 60.0, 30.0) == "neg"          # |10-60|=50 > 30
    assert core_source_raw(60.0, 10.0, 30.0) == "pos"
    assert core_source_raw(10.0, 25.0, 30.0) == "collision"    # |10-25|=15 <= 30
    assert core_source_raw(10.0, None, 30.0) == "neg"          # single core fires
    assert core_source_raw(None, 15.0, 30.0) == "pos"
    assert core_source_raw(None, None, 30.0) == "none"         # neither -> none (not ambiguous)


def test_participation_ratio_excludes_clamped_denominator():
    fired = np.array([True, True, False, False])
    region = np.array([True, True, True, True])
    assert participation_ratio(fired, region) == 0.5            # 2/4
    free = np.array([True, True, False, False])                 # idx 2,3 clamped -> out of denom
    assert participation_ratio(fired, region, valid=free) == 1.0   # 2/2 (fair)
    r = participation_ratio(fired, np.zeros(4, bool))
    assert r != r                                              # NaN on empty denom


def test_exclude_target_contacts():
    valid = np.array([True, True, True, False])
    target = np.array([False, True, False, True])
    out = exclude_target_contacts(valid, target)
    assert out.tolist() == [True, False, True, False]
    assert valid.tolist() == [True, True, True, False]         # input not mutated


# ---------------- Task 2: target masks + dynamic clamp schedule ----------------

def _toy_sheet():
    xs, ys = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    posE = np.column_stack([xs.ravel(), ys.ravel()])
    posI = np.array([[5.0, 5.0], [2.0, 2.0]])
    pos = np.vstack([posE, posI]); NE = posE.shape[0]; N = pos.shape[0]
    is_E = np.zeros(N, bool); is_E[:NE] = True
    return pos, is_E, NE, N


def test_dynamic_vth_E_only_only_inside_window():
    base = np.full(6, 18.0)
    is_E = np.array([True, True, True, False, False, False])
    target = np.array([True, False, True, True, False, True])
    # inside window [100,140): E cells in target clamped (idx 0,2); I in target (3,5) untouched
    inside = intervention_vth_at_time(base, target, is_E, 120.0, 100.0, 140.0)
    assert inside[0] == CLAMP_LEVEL and inside[2] == CLAMP_LEVEL
    assert inside[3] == 18.0 and inside[5] == 18.0     # I cells never clamped
    assert inside[1] == 18.0                           # E not in target
    assert base[0] == 18.0                             # input not mutated


def test_dynamic_vth_off_window_identity():
    base = np.full(6, 18.0); base[1] = 16.5
    is_E = np.array([True] * 6)
    target = np.array([True, True, False, False, False, False])
    before = intervention_vth_at_time(base, target, is_E, 50.0, 100.0, 140.0)
    after = intervention_vth_at_time(base, target, is_E, 140.0, 100.0, 140.0)  # off is exclusive
    assert np.array_equal(before, base)
    assert np.array_equal(after, base)
    # no schedule (on_ms=None) -> always identity
    assert np.array_equal(intervention_vth_at_time(base, target, is_E, 120.0, None, None), base)


def test_count_matched_off_axis_target_avoids_cores():
    pos, is_E, NE, N = _toy_sheet()
    axis = np.array([1.0, 0.0]); center = np.array([5.0, 5.0])
    on = make_on_axis_target(pos, is_E, axis, center, thickness=1.0)   # |x-5|<=0.5 -> x=5 column
    n_on = int((on & is_E).sum())
    no_cores = [np.zeros(N, bool), np.zeros(N, bool)]
    off = make_off_axis_target(pos, is_E, axis, center, thickness=1.0, n_match=n_on,
                               core_masks=no_cores, rng=np.random.default_rng(1), L=10.0, mode="lateral")
    assert int((off & is_E).sum()) == n_on
    # off-axis cells are laterally displaced from the propagation axis (the y=5 line here),
    # i.e. off the corridor (a small crossing overlap with the perpendicular on-axis band is
    # geometrically unavoidable and not a violation).
    assert np.all(np.abs(pos[off][:, 1] - 5.0) >= 2.0)
    # if the off-axis lateral region overlaps a core, raise
    bad_core = np.zeros(N, bool)
    bad_core[(np.abs(pos[:, 1] - 8.5) <= 1.0) & is_E] = True   # covers lateral band y~8.5
    with pytest.raises(ValueError):
        make_off_axis_target(pos, is_E, axis, center, thickness=1.0, n_match=n_on,
                             core_masks=[bad_core], rng=np.random.default_rng(1), L=10.0, mode="lateral")


def test_static_target_is_upper_bound():
    base = np.full(6, 18.0)
    is_E = np.array([True, True, True, True, False, False])
    target = np.array([True, False, True, False, True, False])
    sched = make_static_deadzone_schedule()
    assert sched["on_ms"] == 0.0 and sched["off_ms"] == float("inf")
    vth_t = intervention_vth_at_time(base, target, is_E, 12345.0, sched["on_ms"], sched["off_ms"])
    expect = base.copy(); expect[target & is_E] = CLAMP_LEVEL    # idx 0,2 (E + target)
    assert np.array_equal(vth_t, expect)


# ---------------- Task 3: baseline eligibility + replay schedule ----------------

def _ev(src, far_ratio, so, fo, eid=0):
    return dict(event_id=eid, core_source_raw=src, oracle_far_ratio=far_ratio,
                source_onset=so, far_onset_time=fo)


def test_baseline_eligibility_requires_cross_midline_opportunities():
    cross = [_ev("neg", 0.2, 100.0, 150.0) for _ in range(6)]   # crossing + far after source
    ok, reason, flags = baseline_eligibility(dict(n_returned=25, n_neg=10, n_pos=8, events=cross))
    assert ok and reason == "eligible"
    assert flags["n_cross_midline"] >= 5 and flags["n_trigger_opportunity"] >= 5
    assert baseline_eligibility(dict(n_returned=10, n_neg=10, n_pos=8, events=cross))[1] == "too_few_events"
    assert baseline_eligibility(dict(n_returned=25, n_neg=1, n_pos=8, events=cross))[1] == "one_end_silent"
    # crossing but far_onset == source_onset -> no temporal window to intervene
    nowindow = [_ev("neg", 0.2, 100.0, 100.0) for _ in range(6)]
    assert baseline_eligibility(dict(n_returned=25, n_neg=10, n_pos=8, events=nowindow))[1] == "no_trigger_opportunity"
    flat = [_ev("neg", 0.0, 100.0, None) for _ in range(6)]
    assert baseline_eligibility(dict(n_returned=25, n_neg=10, n_pos=8, events=flat))[1] == "no_cross_midline"


def test_select_first_eligible_event_prefers_single_source_cross_midline():
    events = [
        _ev("collision", 0.9, 100.0, 150.0, eid=0),
        _ev("none", 0.0, None, None, eid=1),
        _ev("neg", 0.02, 100.0, 150.0, eid=2),     # below cross-midline frac
        _ev("pos", 0.30, 100.0, 150.0, eid=3),     # first eligible
        _ev("neg", 0.50, 100.0, 150.0, eid=4),
    ]
    assert select_first_eligible_event(events, 0.05)["event_id"] == 3
    assert select_first_eligible_event([], 0.05) is None


def test_build_replay_schedule_starts_after_source_onset():
    s = build_replay_schedule(_ev("neg", 0.2, 100.0, 160.0), trigger_delay_ms=8.0, duration_ms=40.0)
    assert s["on_ms"] == 108.0 and s["off_ms"] == 148.0 and s["trigger_status"] == "fired"
    with pytest.raises(ValueError):     # trigger would fire after far onset
        build_replay_schedule(_ev("neg", 0.2, 100.0, 105.0), trigger_delay_ms=8.0)
    s2 = build_replay_schedule(_ev("neg", 0.2, 100.0, 105.0), trigger_delay_ms=8.0, allow_late=True)
    assert s2["on_ms"] == 108.0


def test_late_schedule_marks_late_control():
    s = build_late_schedule(_ev("neg", 0.2, 100.0, 160.0), late_delay_ms=8.0, duration_ms=40.0)
    assert s["on_ms"] == 168.0 and s["off_ms"] == 208.0 and s["trigger_status"] == "late"
