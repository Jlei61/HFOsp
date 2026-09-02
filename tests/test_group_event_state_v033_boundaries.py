"""Frozen data-boundary contract (v0.3.3 plan Task 3, clauses B1-B5).

Mainline: a target window never crosses a recording gap, a partition boundary
or a seizure; seizure and immediate-postictal events never update the state;
the state keeps its autonomous flow across the excluded interval; only a real
gap / session edge is a hard reset; a hard reset at seizure onset is a *named
sensitivity variant*, never the default.
"""
from __future__ import annotations

import numpy as np
import inspect
import pytest
import torch

from src.topic5_group_event_state.v02.timeline import RecordedSession, build_carry_segments
from src.topic5_group_event_state.v032_eval.partition import eval_partition
from src.topic5_group_event_state.v032_model.state import anchor_states, leaky_bank_trajectory
from src.topic5_group_event_state.v033_evaluator import boundaries as B


def test_real_boundary_audit_checks_the_kept_stream_not_a_tautology():
    import scripts.audit_group_event_state_v033_boundaries as audit

    source = inspect.getsource(audit.audit_subject)
    assert "update[kept_positions].all()" in source
    assert "update & ~update" not in source

SESSIONS = [RecordedSession(0, 0.0, 10_000.0), RecordedSession(1, 20_000.0, 30_000.0)]
SEIZURES = [{"onset_epoch": 5_000.0, "offset_epoch": 5_100.0}]
POSTICTAL = 600.0


def _segments():
    return build_carry_segments(SESSIONS, SEIZURES, postictal_exclusion_seconds=POSTICTAL,
                                min_segment_seconds=300.0)


def test_target_window_never_crosses_gap_split_or_seizure():
    segments = _segments()                      # [0,5000) [5700,10000) [20000,30000)
    partition = eval_partition(segments)        # 60/70/80 % of 19 300 recorded seconds
    # recorded seconds 5000 + 4300 + 10000 = 19300; 60 % -> 11580 s recorded -> epoch 22280 in segment 2
    t = np.array([100.0, 4_000.0, 9_000.0, 5_800.0, 20_100.0])
    ok = B.target_window_valid(t, 1800.0, segments, partition)
    assert ok.tolist() == [True, False, False, True, True]
    # a window that straddles the 60 % boundary is invalid even inside one segment
    b0 = float(partition.boundary_epochs[0])
    inside_seg = any(s.start_epoch <= b0 - 100.0 and b0 + 100.0 <= s.stop_epoch for s in segments)
    assert inside_seg
    assert B.target_window_valid(np.array([b0 - 100.0]), 200.0, segments, partition).tolist() == [False]
    # start outside every segment is invalid
    assert B.target_window_valid(np.array([5_300.0, 15_000.0]), 60.0, segments, partition).tolist() == [False, False]


def test_seizure_and_immediate_postictal_events_never_update_state():
    events = np.array([4_990.0, 5_000.0, 5_050.0, 5_400.0, 5_699.9, 5_700.0, 9_000.0])
    mask = B.event_update_mask(events, SEIZURES, postictal_seconds=POSTICTAL)
    assert mask.tolist() == [True, False, False, False, False, True, True]


def test_state_carry_units_are_recorded_sessions_and_reject_overlap():
    units = B.state_carry_units(SESSIONS)
    assert [(u.session_id, u.start_epoch, u.stop_epoch) for u in units] == [(0, 0.0, 10_000.0), (1, 20_000.0, 30_000.0)]
    with pytest.raises(ValueError):
        B.state_carry_units([RecordedSession(0, 0.0, 100.0), RecordedSession(1, 50.0, 200.0)])


def test_autonomous_flow_continues_across_a_seizure_inside_one_session():
    events = np.array([1_000.0, 4_990.0, 5_050.0, 5_400.0, 8_000.0])
    units = B.state_carry_units(SESSIONS)
    ev_unit = B.anchor_carry_index(events, units)
    update = B.event_update_mask(events, SEIZURES, postictal_seconds=POSTICTAL)
    t_anchor = np.array([6_000.0])
    an_unit = B.anchor_carry_index(t_anchor, units)
    last = B.carry_last_event(events, ev_unit, update, t_anchor, an_unit)
    assert last.tolist() == [1]                      # 4990 s event, not the ictal/postictal ones, not -1
    u = torch.ones((events.size, 1))
    _pre, post = leaky_bank_trajectory(u, torch.from_numpy(events), torch.from_numpy(ev_unit),
                                       torch.tensor([1800.0]), chunk_seconds=3600.0)
    s = anchor_states(post, torch.from_numpy(events), torch.from_numpy(t_anchor), torch.from_numpy(last),
                      torch.tensor([1800.0]))
    assert float(s[0, 0]) > 0.0                      # decayed, not reset
    # sensitivity variant: segments cut at the seizure -> the anchor has no in-segment past -> reset to 0
    segments = _segments()
    seg_of_event = B.anchor_carry_index(events, segments)
    seg_of_anchor = B.anchor_carry_index(t_anchor, segments)
    last_hard = B.carry_last_event(events, seg_of_event, update, t_anchor, seg_of_anchor)
    assert last_hard.tolist() == [-1]


def test_gap_or_session_edge_is_a_hard_reset():
    events = np.array([1_000.0, 9_000.0])
    units = B.state_carry_units(SESSIONS)
    ev_unit = B.anchor_carry_index(events, units)
    update = np.ones(events.size, bool)
    t_anchor = np.array([20_500.0])
    last = B.carry_last_event(events, ev_unit, update, t_anchor, B.anchor_carry_index(t_anchor, units))
    assert last.tolist() == [-1]
    assert B.anchor_carry_index(np.array([15_000.0]), units).tolist() == [-1]


def test_hard_seizure_reset_is_a_named_sensitivity_variant_not_the_default():
    assert B.DEFAULT_VARIANT == "mainline"
    main = B.boundary_variant("mainline")
    assert main["state_reset_at"] == "recorded_gap_or_session_edge_only"
    assert main["seizure_and_immediate_postictal_events"] == "excluded_from_state_update_autonomous_flow_continues"
    sens = B.boundary_variant("sensitivity_hard_seizure_reset")
    assert sens["state_reset_at"] == "recorded_gap_or_session_edge_or_seizure_onset"
    assert sens["role"] == "sensitivity_only"
    with pytest.raises(KeyError):
        B.boundary_variant("something_else")


def test_events_before_an_anchor_but_in_another_unit_are_never_its_history():
    events = np.array([9_500.0, 9_900.0])
    units = B.state_carry_units(SESSIONS)
    ev_unit = B.anchor_carry_index(events, units)
    update = np.ones(2, bool)
    t_anchor = np.array([9_950.0, 20_300.0])
    last = B.carry_last_event(events, ev_unit, update, t_anchor, B.anchor_carry_index(t_anchor, units))
    assert last.tolist() == [1, -1]
