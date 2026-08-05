import numpy as np
import pytest

from src.topic5_slow_state_sessions import (
    assign_events,
    build_blocks,
    build_sessions,
    dropped_remainders,
    session_gaps,
)


def _seg(sid, start, stop, group="g", montage="m"):
    return {
        "source_id": sid, "start_time": start, "stop_time": stop,
        "continuity_group": group, "montage_hash": montage,
    }


def test_sessions_use_metadata_bounds_not_event_bounds():
    # the recording runs 0-1000 s but the only events are at 400-500 s;
    # the session must still be 1000 s long
    sessions = build_sessions([_seg("a", 0.0, 1000.0)], join_seconds=300.0)
    assert sessions[0]["t_start"] == 0.0 and sessions[0]["t_end"] == 1000.0
    with_events = assign_events(sessions, np.array([400.0, 500.0]), np.array(["a", "a"]))
    assert with_events[0]["t_start"] == 0.0 and with_events[0]["t_end"] == 1000.0
    assert with_events[0]["first_event_time"] == 400.0


def test_a_quiet_but_recorded_stretch_is_not_a_gap():
    # two sessions far apart in metadata time, but first session has early events
    # and second session has late events — gap must reflect metadata bounds, not event span
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 5000.0, 6000.0)]
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0),
        np.array([100.0, 150.0, 5900.0, 5950.0]),
        np.array(["a", "a", "b", "b"]),
    )
    gaps = session_gaps(sessions)
    assert len(gaps) == 1
    # metadata gap is 5000 - 1000 = 4000 seconds, not the event spread 5900 - 150
    assert gaps[0]["metadata_gap_seconds"] == pytest.approx(4000.0)


def test_metadata_gap_and_event_silence_are_reported_separately():
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 2000.0, 3000.0)]
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0),
        np.array([100.0, 900.0, 2100.0, 2900.0]),
        np.array(["a", "a", "b", "b"]),
    )
    gap = session_gaps(sessions)[0]
    assert gap["metadata_gap_seconds"] == pytest.approx(1000.0)
    assert gap["event_silence_seconds"] == pytest.approx(1200.0)
    assert gap["event_silence_seconds"] > gap["metadata_gap_seconds"]
    assert gap["observed_events_during_gap"] is False


def test_segments_within_the_join_threshold_merge():
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 1200.0, 2000.0)]
    assert len(build_sessions(segments, join_seconds=300.0)) == 1


def test_segments_with_a_different_montage_never_merge():
    segments = [_seg("a", 0.0, 1000.0, montage="m1"), _seg("b", 1100.0, 2000.0, montage="m2")]
    assert len(build_sessions(segments, join_seconds=300.0)) == 2


def test_segments_in_a_different_continuity_group_never_merge():
    segments = [_seg("a", 0.0, 1000.0, group="g1"), _seg("b", 1100.0, 2000.0, group="g2")]
    assert len(build_sessions(segments, join_seconds=300.0)) == 2


def test_blocks_never_span_a_session_boundary():
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(6.0), 10000.0 + np.arange(6.0)])
    names = np.array(["a"] * 6 + ["b"] * 6)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert len(blocks) == 2
    assert {b["session_index"] for b in blocks} == {0, 1}


def test_the_first_block_after_a_gap_is_labelled_cross_gap():
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(8.0), 10000.0 + np.arange(8.0)])
    names = np.array(["a"] * 8 + ["b"] * 8)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert [b["transition_stratum"] for b in blocks] == [
        None, "within_session", "cross_gap", "within_session",
    ]


def test_delta_t_bridges_the_gap_rather_than_resetting_to_zero():
    # rev3 R3-D: delta_t_from_previous is replaced by inter_block_gap (same
    # first-event-minus-previous-last-event formula) plus transition_delta_t and
    # metadata_gap_seconds.
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(4.0), 10000.0 + np.arange(4.0)])
    names = np.array(["a"] * 4 + ["b"] * 4)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert blocks[0]["inter_block_gap"] is None
    assert blocks[1]["inter_block_gap"] == pytest.approx(10000.0 - 3.0)
    # beyond the literal brief: the only fixture in this file with a genuine cross-
    # session transition, so it is also the only place that can exercise
    # metadata_gap_seconds' non-None numeric branch (session "a" metadata ends at
    # 100.0, session "b" metadata starts at 10000.0 -- neither equals the event-time
    # bound 3.0/10000.0 used by inter_block_gap, so a mutant that aliased
    # metadata_gap_seconds to inter_block_gap would still be caught here).
    assert blocks[0]["metadata_gap_seconds"] is None
    assert blocks[1]["metadata_gap_seconds"] == pytest.approx(10000.0 - 100.0)


def test_transition_delta_t_is_centre_to_centre_not_edge_to_edge():
    # rev3 fix-round-4 ITEM 2: the original fixture (0,1,2,3,10,11,12,13) is symmetric
    # within each block, so the mean, the midpoint of first-and-last, and the median all
    # give the same 10.0 -- nothing in that fixture actually pinned "centre" to MEAN
    # specifically. Re-fixtured with skewed blocks: block 0 = 0,1,2,30 (mean 8.25;
    # first/last-midpoint (0+30)/2=15.0), block 1 = 100,101,102,103 (mean 101.5;
    # first/last-midpoint (100+103)/2=101.5). transition_delta_t (mean-to-mean) ==
    # 101.5-8.25 == 93.25; an implementation using first/last-midpoint instead would
    # give 101.5-15.0 == 86.5. inter_block_gap (edge-to-edge, unaffected by this
    # distinction) == 100.0-30.0 == 70.0, distinct from both of the above.
    segments = [_seg("a", 0.0, 120.0)]
    times = np.array([0.0, 1.0, 2.0, 30.0, 100.0, 101.0, 102.0, 103.0])
    names = np.array(["a"] * 8)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert blocks[0]["transition_delta_t"] is None
    assert blocks[0]["inter_block_gap"] is None
    assert blocks[1]["transition_delta_t"] == pytest.approx(93.25)
    assert blocks[1]["inter_block_gap"] == pytest.approx(70.0)


def test_metadata_gap_is_none_within_one_session():
    # rev3 R3-D: metadata_gap_seconds is None whenever the previous block is in the
    # same session (ambiguity-resolution note) -- exercised here on a single-session,
    # two-block fixture where transition_stratum is "within_session".
    segments = [_seg("a", 0.0, 20.0)]
    times = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    names = np.array(["a"] * 8)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert blocks[1]["transition_stratum"] == "within_session"
    assert blocks[1]["metadata_gap_seconds"] is None


def test_session_remainders_are_dropped_and_counted_not_padded():
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(10.0)
    names = np.array(["a"] * 10)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    assert len(build_blocks(sessions, block_events=4, event_times=times)) == 2
    assert dropped_remainders(sessions, block_events=4) == [
        {"session_index": 0, "n_dropped": 2}
    ]


def test_block_size_below_two_is_rejected():
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(10.0)
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0), times, np.array(["a"] * 10)
    )
    with pytest.raises(ValueError):
        build_blocks(sessions, block_events=1, event_times=times)


def test_event_silence_is_none_when_a_session_has_no_events():
    # one session with events, one with none; event_silence_seconds must be None
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 2000.0, 3000.0)]
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0),
        np.array([100.0, 900.0]),  # only events in session a, none in b
        np.array(["a", "a"]),
    )
    gaps = session_gaps(sessions)
    assert len(gaps) == 1
    gap = gaps[0]
    # metadata gap is still a float
    assert isinstance(gap["metadata_gap_seconds"], float)
    assert gap["metadata_gap_seconds"] == pytest.approx(1000.0)
    # event_silence_seconds must be None because session b has no events
    assert gap["event_silence_seconds"] is None


def test_a_session_dividing_evenly_is_omitted_from_dropped_remainders():
    # a session with event count equal to a multiple of block_events has no remainder
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(8.0)  # 8 events = 2 blocks of size 4 exactly
    names = np.array(["a"] * 8)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    # should produce 2 blocks with 0 remainder
    assert len(build_blocks(sessions, block_events=4, event_times=times)) == 2
    # session should not appear in dropped_remainders at all
    assert dropped_remainders(sessions, block_events=4) == []


def test_a_session_shorter_than_one_block_reports_all_its_events_dropped():
    # a session with fewer events than block_size has all events dropped
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(3.0)  # 3 events < block_size of 4
    names = np.array(["a"] * 3)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    # should produce 0 blocks
    assert len(build_blocks(sessions, block_events=4, event_times=times)) == 0
    # all 3 events should be reported as dropped
    assert dropped_remainders(sessions, block_events=4) == [
        {"session_index": 0, "n_dropped": 3}
    ]
