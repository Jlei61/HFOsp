"""M3A-A2 dynamic recorder: per-event q-landmark sampler.

The dynamic RegionalResource run produces per-step q traces (trace_core,
trace_global, trace_gk). This sampler turns those traces + detected event times
into per-event landmark rows (pre/onset/peak/end/post_50ms/post_200ms/post_1s).
Pure function -- tested on synthetic traces, no SNN run. The downstream
conversion of these rows into the canonical phase_trajectory schema is covered by
the runner-layer export test (test_m3a_export.py) against src/sef_hfo_m3_interface.

Spec: docs/superpowers/plans/2026-06-24-sef-hfo-m3a-dynamic-slowvars-plan.md
      Task 1 (dynamic recorder) + §3 required state samples.

Contract clauses locked here:
  - sampled values align with simulation time (value at a stage == trace[step])
  - pre is before onset; post is after end
  - every event emits all seven required stages (canonical enum members)
  - multiple events keep their own event_id
  - a post landmark beyond the trace clamps to the last step (no IndexError)
  - ragged traces (mismatched lengths) are rejected
"""
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_a2 import sample_event_landmarks  # noqa: E402


def _index_traces(T):
    """q_core[t] == t so a sampled value reveals exactly which step was read."""
    return {
        "q_core": list(range(T)),
        "q_global": [0.5 * t for t in range(T)],
        "g_K": [0.01 * t for t in range(T)],
    }


def _event(eid=1, onset=100, peak=150, end=200):
    return {"event_id": eid, "onset_ms": onset, "peak_ms": peak, "end_ms": end}


def test_landmark_sampler_aligns_values_with_event_times():
    rows = sample_event_landmarks(_index_traces(300), dt_ms=1.0, events=[_event()])
    by = {r["event_stage"]: r for r in rows}
    assert by["onset"]["q_core"] == 100
    assert by["peak"]["q_core"] == 150
    assert by["end"]["q_core"] == 200


def test_landmark_sampler_orders_pre_before_onset_post_after_end():
    rows = sample_event_landmarks(_index_traces(1500), 1.0, [_event(onset=300, peak=350, end=400)])
    by = {r["event_stage"]: r for r in rows}
    assert by["pre"]["time_ms"] < by["onset"]["time_ms"]
    assert by["post_1s"]["time_ms"] > by["end"]["time_ms"]


def test_landmark_sampler_emits_all_required_stages():
    rows = sample_event_landmarks(_index_traces(1500), 1.0, [_event(onset=300, peak=350, end=400)])
    assert {r["event_stage"] for r in rows} == {
        "pre", "onset", "peak", "end", "post_50ms", "post_200ms", "post_1s"}


def test_landmark_sampler_keeps_event_ids_for_multiple_events():
    events = [_event(eid="a", onset=100, peak=120, end=150),
              _event(eid="b", onset=600, peak=650, end=700)]
    rows = sample_event_landmarks(_index_traces(2000), 1.0, events)
    assert {r["event_id"] for r in rows} == {"a", "b"}
    assert sum(1 for r in rows if r["event_id"] == "a") == 7


def test_landmark_sampler_clamps_post_beyond_trace_end():
    # post_1s (end+1000=1200) is past the 210-step trace -> clamp to last step, no error.
    rows = sample_event_landmarks(_index_traces(210), 1.0, [_event()])
    by = {r["event_stage"]: r for r in rows}
    assert by["post_1s"]["q_core"] == 209


def test_landmark_sampler_rejects_ragged_traces():
    bad = {"q_core": [1.0, 0.9, 0.8], "q_global": [1.0, 0.9]}  # mismatched lengths
    with pytest.raises(ValueError):
        sample_event_landmarks(bad, 1.0, [_event()])
