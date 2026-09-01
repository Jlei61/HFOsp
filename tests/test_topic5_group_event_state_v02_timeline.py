"""A0 regression tests: coverage segments, physical-time split, fixed anchor grid.

Every test here encodes one clause of
``docs/archive/topic5/group_event_state_v0_2_agent_a_contract_clauses_2026-09-01.md``.
A test name that does not name a clause is a test that cannot fail for a
scientific reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topic5_group_event_state.v02 import timeline as T


def _sessions(*pairs: tuple[float, float]) -> list[T.RecordedSession]:
    return [
        T.RecordedSession(session_id=i, start_epoch=float(a), stop_epoch=float(b))
        for i, (a, b) in enumerate(pairs)
    ]


# --------------------------------------------------------------------- C2 / C3


def test_segments_break_at_recording_gaps_and_never_bridge_them() -> None:
    """C2: two recorded sessions are two carry segments, never one."""

    segments = T.build_carry_segments(
        _sessions((0.0, 3600.0), (10_000.0, 13_600.0)), seizures=[]
    )
    assert len(segments) == 2
    assert (segments[0].start_epoch, segments[0].stop_epoch) == (0.0, 3600.0)
    assert (segments[1].start_epoch, segments[1].stop_epoch) == (10_000.0, 13_600.0)
    assert segments[0].session_id != segments[1].session_id


def test_segments_break_at_seizure_and_skip_the_postictal_window() -> None:
    """C3: a seizure cuts the session; the new segment starts offset+postictal."""

    seizures = [{"onset_epoch": 5_000.0, "offset_epoch": 5_100.0}]
    segments = T.build_carry_segments(
        _sessions((0.0, 20_000.0)),
        seizures=seizures,
        postictal_exclusion_seconds=3600.0,
        min_segment_seconds=0.0,
    )
    assert len(segments) == 2
    assert segments[0].stop_epoch == pytest.approx(5_000.0)          # stops at onset
    assert segments[1].start_epoch == pytest.approx(5_100.0 + 3600.0)  # offset + 60 min
    # no segment covers any instant of [onset, offset + postictal)
    for probe in (5_000.0, 5_050.0, 5_100.0, 8_000.0, 8_699.0):
        assert not any(s.start_epoch <= probe < s.stop_epoch for s in segments)


def test_postictal_length_is_a_parameter_not_a_hard_coded_constant() -> None:
    """C3: the 60 min primary must be swappable for the sensitivity analysis."""

    seizures = [{"onset_epoch": 5_000.0, "offset_epoch": 5_100.0}]
    short = T.build_carry_segments(
        _sessions((0.0, 20_000.0)), seizures=seizures,
        postictal_exclusion_seconds=0.0, min_segment_seconds=0.0,
    )
    assert short[1].start_epoch == pytest.approx(5_100.0)


def test_a_seizure_that_swallows_a_session_leaves_no_segment() -> None:
    """C3: no silent 'zero-length' segment survives a fully excluded session."""

    seizures = [{"onset_epoch": -10.0, "offset_epoch": 100.0}]
    segments = T.build_carry_segments(
        _sessions((0.0, 3000.0)), seizures=seizures,
        postictal_exclusion_seconds=3600.0, min_segment_seconds=0.0,
    )
    assert segments == []


# --------------------------------------------------------------------- C1


def test_split_is_by_recorded_time_not_event_count() -> None:
    """C1: an event-dense early session must not swallow the whole TRAIN split.

    Two sessions of equal recorded length; the first holds 100x more events.
    An event-count split would put the boundary inside session 0.  The contract
    requires the boundary to sit at 70% of *recorded time*.
    """

    segments = T.build_carry_segments(_sessions((0.0, 10_000.0), (50_000.0, 60_000.0)))
    split = T.physical_time_split(segments, fractions=(0.7, 0.1, 0.2))
    # 20 000 s recorded in total -> train ends after 14 000 recorded seconds,
    # i.e. 10 000 s in segment 0 plus 4 000 s into segment 1.
    assert split.boundary_epochs[0] == pytest.approx(54_000.0)
    assert split.boundary_epochs[1] == pytest.approx(56_000.0)
    assert split.recorded_seconds["train"] == pytest.approx(14_000.0)
    assert split.recorded_seconds["val"] == pytest.approx(2_000.0)
    assert split.recorded_seconds["test"] == pytest.approx(4_000.0)


def test_split_label_of_an_instant_is_a_pure_function_of_recorded_time() -> None:
    """C1: the split of a timestamp never depends on how many events are near it."""

    segments = T.build_carry_segments(_sessions((0.0, 10_000.0), (50_000.0, 60_000.0)))
    split = T.physical_time_split(segments, fractions=(0.7, 0.1, 0.2))
    assert split.label_of(5_000.0) == "train"
    assert split.label_of(53_999.0) == "train"
    assert split.label_of(54_001.0) == "val"
    assert split.label_of(56_001.0) == "test"


# --------------------------------------------------------------------- C4 / C11


def test_anchor_grid_is_uniform_in_time_not_in_events() -> None:
    """C4: doubling the event density must not add a single anchor."""

    segments = T.build_carry_segments(_sessions((0.0, 7200.0)))
    split = T.physical_time_split(segments, fractions=(1.0, 0.0, 0.0))
    sparse = T.build_anchor_grid(
        segments, split, event_times=np.arange(0.0, 7200.0, 60.0),
        horizons_seconds=(300.0,),
    )
    dense = T.build_anchor_grid(
        segments, split, event_times=np.arange(0.0, 7200.0, 1.0),
        horizons_seconds=(300.0,),
    )
    assert sparse.n_anchors == dense.n_anchors
    assert np.array_equal(sparse.t_anchor, dense.t_anchor)
    steps = np.diff(sparse.t_anchor)
    assert np.allclose(steps, T.ANCHOR_GRID_SECONDS)


def test_anchor_windows_never_leave_their_segment_or_their_split() -> None:
    """C1 + C3: eligibility is 'window fully inside one segment and one split'."""

    segments = T.build_carry_segments(_sessions((0.0, 10_000.0), (50_000.0, 60_000.0)))
    split = T.physical_time_split(segments, fractions=(0.7, 0.1, 0.2))
    grid = T.build_anchor_grid(
        segments, split, event_times=np.arange(0.0, 60_000.0, 30.0),
        horizons_seconds=(300.0, 1800.0),
    )
    for h_i, horizon in enumerate((300.0, 1800.0)):
        ok = grid.eligible[:, h_i]
        starts = grid.t_anchor[ok]
        stops = starts + horizon
        seg_lo = np.array([segments[s].start_epoch for s in grid.segment_index[ok]])
        seg_hi = np.array([segments[s].stop_epoch for s in grid.segment_index[ok]])
        assert np.all(starts >= seg_lo) and np.all(stops <= seg_hi)
        assert all(
            split.label_of(a) == split.label_of(b - 1e-6) for a, b in zip(starts, stops)
        )


def test_independent_windows_is_time_over_horizon_not_anchor_count() -> None:
    """C11: 5-min-spaced anchors on a 2 h horizon are not 24 independent windows."""

    segments = T.build_carry_segments(_sessions((0.0, 36_000.0)))
    split = T.physical_time_split(segments, fractions=(1.0, 0.0, 0.0))
    grid = T.build_anchor_grid(
        segments, split, event_times=np.arange(0.0, 36_000.0, 30.0),
        horizons_seconds=(7200.0,),
    )
    n_anchor = int(grid.eligible[:, 0].sum())
    n_indep = T.effective_independent_windows(segments, split, "train", 7200.0)
    assert n_anchor > 20
    assert n_indep == 5  # 36 000 s / 7 200 s
    assert n_indep < n_anchor


# --------------------------------------------------------------------- C8 / C12


def test_anchor_reads_only_the_last_event_at_or_before_itself() -> None:
    """C8: the pointer used to propagate state is strictly causal."""

    segments = T.build_carry_segments(_sessions((0.0, 3600.0)))
    split = T.physical_time_split(segments, fractions=(1.0, 0.0, 0.0))
    events = np.array([10.0, 20.0, 301.0, 900.0], dtype=np.float64)
    grid = T.build_anchor_grid(
        segments, split, event_times=events, horizons_seconds=(300.0,),
        min_warmup_seconds=0.0,
    )
    for t, pos in zip(grid.t_anchor, grid.last_event_pos):
        if pos < 0:
            assert not np.any(events < t)
        else:
            assert events[pos] < t
            assert not np.any((events > events[pos]) & (events < t))


def test_absolute_times_stay_float64() -> None:
    """C12: far-epoch timestamps in float32 lose ~64 s of resolution."""

    base = 1_240_466_978.0
    segments = T.build_carry_segments(_sessions((base, base + 7200.0)))
    split = T.physical_time_split(segments, fractions=(1.0, 0.0, 0.0))
    grid = T.build_anchor_grid(
        segments, split,
        event_times=np.arange(base, base + 7200.0, 30.0),
        horizons_seconds=(300.0,),
    )
    assert grid.t_anchor.dtype == np.float64
    assert split.boundary_epochs.dtype == np.float64
    assert np.unique(np.diff(grid.t_anchor)).size == 1  # float32 would jitter


# --------------------------------------------------------------------- C2 end-to-end


def test_events_outside_every_segment_are_dropped_not_reassigned() -> None:
    """C3: a postictal event belongs to no segment; it must not join a neighbour."""

    seizures = [{"onset_epoch": 5_000.0, "offset_epoch": 5_100.0}]
    segments = T.build_carry_segments(
        _sessions((0.0, 20_000.0)), seizures=seizures,
        postictal_exclusion_seconds=3600.0, min_segment_seconds=0.0,
    )
    events = np.array([4_000.0, 5_050.0, 6_000.0, 9_000.0], dtype=np.float64)
    seg_of = T.assign_events_to_segments(events, segments)
    assert seg_of.tolist() == [0, -1, -1, 1]
