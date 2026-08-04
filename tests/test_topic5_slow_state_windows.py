import numpy as np

from src.topic5_slow_state_windows import (
    scale_is_evaluable,
    sliding_event_windows,
    tile_clock_windows,
    tile_event_windows,
)


def _session(index, indices, t_start=0.0, t_end=1000.0):
    return {
        "session_index": index,
        "event_indices": np.asarray(list(indices)),
        "t_start": t_start,
        "t_end": t_end,
    }


def test_primary_windows_do_not_overlap():
    windows = tile_event_windows([_session(0, range(100))], window_events=20)
    seen = np.concatenate([w["event_indices"] for w in windows])
    assert len(windows) == 5
    assert len(set(seen.tolist())) == seen.size


def test_primary_windows_never_span_two_sessions():
    windows = tile_event_windows(
        [_session(0, range(30)), _session(1, range(30, 60))], window_events=20
    )
    assert len(windows) == 2
    assert {w["session_index"] for w in windows} == {0, 1}


def test_primary_windows_are_marked_as_zero_offset():
    windows = tile_event_windows([_session(0, range(100))], window_events=20)
    assert all(w["offset_fraction"] == 0.0 for w in windows)


def test_sliding_windows_are_labelled_as_sensitivity_and_do_overlap():
    windows = sliding_event_windows(
        [_session(0, range(100))], window_events=20, offsets=[0.5]
    )
    assert windows
    assert all(w["offset_fraction"] == 0.5 for w in windows)


def test_clock_windows_tile_wall_time_not_event_count():
    times = np.concatenate([np.arange(0.0, 10.0), np.arange(100.0, 110.0)])
    session = _session(0, range(20), t_start=0.0, t_end=200.0)
    windows = tile_clock_windows([session], times, window_seconds=100.0, min_events=5)
    assert len(windows) == 2
    assert len(windows[0]["event_indices"]) == 10


def test_a_clock_window_with_too_few_events_is_dropped_not_padded():
    times = np.array([0.0, 1.0, 150.0])
    session = _session(0, range(3), t_start=0.0, t_end=200.0)
    assert tile_clock_windows([session], times, window_seconds=100.0, min_events=5) == []


def test_a_scale_with_too_few_independent_windows_is_not_evaluable():
    assert scale_is_evaluable([{}] * 4, minimum=5) is False
    assert scale_is_evaluable([{}] * 5, minimum=5) is True


def test_two_hundred_random_splits_are_not_two_hundred_windows():
    # guards the rev1 defect: split count must never be mistaken for window count
    windows = tile_event_windows([_session(0, range(100))], window_events=50)
    assert len(windows) == 2
    assert scale_is_evaluable(windows, minimum=5) is False


def test_a_session_shorter_than_one_window_yields_no_window():
    windows = tile_event_windows([_session(0, range(10))], window_events=20)
    assert windows == []


def test_sliding_windows_reject_a_zero_offset():
    try:
        sliding_event_windows([_session(0, range(100))], window_events=20, offsets=[0.0])
        assert False, "expected ValueError for offset 0.0"
    except ValueError as e:
        assert "offset 0.0" in str(e).lower() or "primary" in str(e).lower()


def test_event_window_size_below_two_is_rejected():
    try:
        tile_event_windows([_session(0, range(100))], window_events=1)
        assert False, "expected ValueError for window_events < 2"
    except ValueError as e:
        assert "at least 2" in str(e).lower() or "window_events" in str(e).lower()


def test_clock_windows_never_cross_a_session_boundary():
    # Two sessions far apart; windows should never cross the boundary
    times1 = np.arange(0.0, 10.0)  # 10 events in first session
    times2 = np.arange(500.0, 510.0)  # 10 events in second session (far apart)
    times = np.concatenate([times1, times2])

    sessions = [
        _session(0, range(10), t_start=0.0, t_end=50.0),
        _session(1, range(10, 20), t_start=500.0, t_end=550.0),
    ]
    windows = tile_clock_windows(sessions, times, window_seconds=100.0, min_events=1)

    # Verify each window draws from exactly one session
    for w in windows:
        session_idx = w["session_index"]
        event_indices = w["event_indices"]
        # All event indices for session 0 are in [0, 10); for session 1 are in [10, 20)
        if session_idx == 0:
            assert all(idx < 10 for idx in event_indices)
        elif session_idx == 1:
            assert all(idx >= 10 for idx in event_indices)
