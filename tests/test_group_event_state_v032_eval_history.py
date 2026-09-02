"""Causality and standardisation contract of the explicit history H and controls."""
from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v032_eval.history import (
    HistoryFeatureBuilder,
    HistoryInputs,
    Standardiser,
)
from src.topic5_group_event_state.v032_eval.controls import (
    linear_marked_ema,
    random_reservoir_state,
    times_only_state,
)


def _inputs(seed: int = 0) -> HistoryInputs:
    rng = np.random.default_rng(seed)
    # two segments in one session, then a second session
    t0 = np.sort(rng.uniform(0.0, 7200.0, 300))
    t1 = np.sort(rng.uniform(9000.0, 12000.0, 120))
    t2 = np.sort(rng.uniform(50000.0, 60000.0, 200))
    times = np.concatenate([t0, t1, t2])
    seg = np.concatenate([np.zeros(300, int), np.ones(120, int), np.full(200, 2)])
    sess = np.concatenate([np.zeros(420, int), np.ones(200, int)])
    n, c = times.size, 6
    part = rng.random((n, c)) < 0.5
    part[np.arange(n), rng.integers(0, c, n)] = True
    vocab = np.array([True, True, True, True, True, False])
    delay = np.where(part, rng.uniform(0, 0.08, (n, c)), np.nan).astype(np.float32)
    marks = rng.normal(size=(n, 3))
    group_count = rng.integers(1, 4, n)
    return HistoryInputs(
        event_times=times, event_segment=seg, event_session=sess,
        segment_start={0: 0.0, 1: 8800.0, 2: 49000.0}, session_start={0: 0.0, 1: 49000.0},
        recording_start=0.0, participation=part, vocab_mask=vocab,
        shaft_of_contact=("A", "A", "B", "B", "C", "C"), group_count=group_count,
        relative_delay=delay, mark_continuous=marks, mark_names=("m0", "m1", "m2"),
    )


def _builder(inputs: HistoryInputs) -> HistoryFeatureBuilder:
    return HistoryFeatureBuilder(inputs, lookback_seconds=(300.0, 1800.0),
                                 ewma_tau_seconds=(300.0, 1800.0), field_tau_seconds=(300.0,))


def test_history_depends_only_on_events_strictly_before_the_query_time():
    inputs = _inputs()
    builder = _builder(inputs)
    q = np.array([3600.0, 10000.0, 55000.0])
    seg = np.array([0, 1, 2])
    x_full, names = builder.features(q, seg, variant="H_strong")
    # Delete every event at or after each query time (per segment) -> identical features.
    keep = np.ones(inputs.event_times.size, bool)
    for t, s in zip(q, seg):
        keep &= ~((inputs.event_segment == s) & (inputs.event_times >= t))
    trimmed = HistoryInputs(
        event_times=inputs.event_times[keep], event_segment=inputs.event_segment[keep],
        event_session=inputs.event_session[keep], segment_start=inputs.segment_start,
        session_start=inputs.session_start, recording_start=inputs.recording_start,
        participation=inputs.participation[keep], vocab_mask=inputs.vocab_mask,
        shaft_of_contact=inputs.shaft_of_contact, group_count=inputs.group_count[keep],
        relative_delay=inputs.relative_delay[keep], mark_continuous=inputs.mark_continuous[keep],
        mark_names=inputs.mark_names,
    )
    x_trim, _ = _builder(trimmed).features(q, seg, variant="H_strong")
    assert np.allclose(x_full, x_trim)
    # an event exactly at the query time must be excluded (pre-event query)
    exact = inputs.event_times[10]
    x_a, _ = builder.features(np.array([exact]), np.array([0]), variant="H_rate")
    x_c, _ = builder.features(np.array([np.nextafter(exact, np.inf)]), np.array([0]), variant="H_rate")
    # moving the query a hair after the event changes the "since last event" column
    assert x_a[0, names.index("log_seconds_since_last_event")] != x_c[0, names.index("log_seconds_since_last_event")]


def test_history_resets_at_segment_start_and_never_uses_segment_end():
    inputs = _inputs()
    builder = _builder(inputs)
    x, names = builder.features(np.array([8801.0]), np.array([1]), variant="H_strong")
    # one second into segment 1: no event before it in that segment
    assert x[0, names.index("has_previous_event")] == 0.0
    assert x[0, names.index("log_count_300s")] == 0.0
    assert not any("left" in n or "duration" in n or "fraction_through" in n for n in names)


def test_h_rate_is_a_prefix_of_h_strong_and_names_align():
    inputs = _inputs()
    builder = _builder(inputs)
    q = np.array([3600.0, 5000.0])
    seg = np.array([0, 0])
    x_rate, names_rate = builder.features(q, seg, variant="H_rate")
    x_strong, names_strong = builder.features(q, seg, variant="H_strong")
    assert names_strong[: len(names_rate)] == names_rate
    assert np.allclose(x_strong[:, : len(names_rate)], x_rate)
    assert x_rate.shape[1] == len(names_rate)
    # vocabulary-excluded contact never appears in the participation field
    assert not any(n.startswith("participation[5]") for n in names_strong)


def test_standardiser_is_frozen_from_fit_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(100, 4))
    std = Standardiser.fit(x[:60], phase="base_fit")
    z = std.apply(x[60:])
    assert std.n_rows == 60 and std.phase == "base_fit"
    assert np.allclose(std.mean, x[:60].mean(axis=0))
    assert z.shape == (40, 4)


def test_controls_are_causal_deterministic_and_reset_per_segment():
    inputs = _inputs()
    q = np.array([3600.0, 8801.0, 55000.0])
    seg = np.array([0, 1, 2])
    t1, n1 = times_only_state(inputs, q, seg)
    t2, _ = times_only_state(inputs, q, seg)
    assert np.array_equal(t1, t2) and t1.shape == (3, 12) and len(n1) == 12
    assert np.all(t1[1] == 0.0)  # fresh segment -> nothing accumulated
    r1, rn = random_reservoir_state(inputs, q, seg, seed=7)
    r2, _ = random_reservoir_state(inputs, q, seg, seed=7)
    r3, _ = random_reservoir_state(inputs, q, seg, seed=8)
    assert np.array_equal(r1, r2) and not np.allclose(r1, r3)
    assert np.all(r1[1] == 0.0)
    e1, en = linear_marked_ema(inputs, q, seg, taus=(300.0, 1800.0))
    assert e1.shape == (3, 6) and len(en) == 6
    assert np.all(e1[1] == 0.0)
    # deleting future events leaves every control unchanged
    keep = ~((inputs.event_segment == 0) & (inputs.event_times >= 3600.0))
    trimmed = HistoryInputs(**{**inputs.__dict__,
                               "event_times": inputs.event_times[keep],
                               "event_segment": inputs.event_segment[keep],
                               "event_session": inputs.event_session[keep],
                               "participation": inputs.participation[keep],
                               "group_count": inputs.group_count[keep],
                               "relative_delay": inputs.relative_delay[keep],
                               "mark_continuous": inputs.mark_continuous[keep]})
    r_trim, _ = random_reservoir_state(trimmed, q[:1], seg[:1], seed=7)
    assert np.allclose(r_trim[0], r1[0])
