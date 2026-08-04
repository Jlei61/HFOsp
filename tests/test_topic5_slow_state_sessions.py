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
    sessions = build_sessions([_seg("a", 0.0, 5000.0)], join_seconds=300.0)
    assert session_gaps(sessions) == []


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
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(4.0), 10000.0 + np.arange(4.0)])
    names = np.array(["a"] * 4 + ["b"] * 4)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert blocks[0]["delta_t_from_previous"] is None
    assert blocks[1]["delta_t_from_previous"] == pytest.approx(10000.0 - 3.0)


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
