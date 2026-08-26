from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state.bridge import (
    BridgeArrays,
    _explicit_history,
    _raw_features,
    _spectral_features,
    fit_bridge_arm,
)
from src.topic5_continuous_marked_state.exposure import (
    exposure_pair,
    pre_event_innovation_predictors,
)
from src.topic5_continuous_marked_state.event_anchor import fit_event_anchor


def test_explicit_history_resets_at_session_boundary() -> None:
    times = np.asarray([0.0, 10.0, 1000.0, 1010.0])
    session = np.asarray([0, 0, 1, 1])
    part = np.asarray([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
    groups = np.asarray([1, 2, 1, 2])
    load = part.mean(axis=1)
    rank = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    out = _explicit_history(times, session, part, groups, load, rank, "yuquan")
    assert out.shape == (4, 16)
    assert out[2, 0] == 0.0
    assert out[2, 1] == 1.0


def test_masked_raw_and_spectral_features_are_finite() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30 * 256, 5)).astype(np.float32)
    x[100:700] = np.nan
    x[:, 4] = np.nan
    raw = _raw_features(x)
    spectral = _spectral_features(x, 256)
    assert raw.shape == (5 * 7,)
    assert spectral.shape == (5 * 8,)
    assert np.isfinite(raw).all()
    assert np.isfinite(spectral).all()


def test_convex_bridge_fit_is_seed_invariant() -> None:
    rng = np.random.default_rng(4)
    n, n_contacts = 48, 3
    split = np.r_[np.zeros(36, dtype=np.int8), np.ones(12, dtype=np.int8)]
    participation = (rng.random((n, n_contacts)) > 0.45).astype(np.float32)
    arrays = BridgeArrays(
        subject="synthetic",
        history=rng.normal(size=(n, 7)).astype(np.float32),
        spectral=rng.normal(size=(n, 8)).astype(np.float32),
        raw=rng.normal(size=(n, 9)).astype(np.float32),
        log_next_iei=rng.normal(4.0, 0.4, size=n).astype(np.float32),
        participation=participation,
        rank=(rng.random((n, n_contacts)) * participation).astype(np.float32),
        stop_fraction=rng.random(n).astype(np.float32),
        split=split,
        current_time=np.arange(n, dtype=np.float64) * 100.0,
        next_time=np.arange(n, dtype=np.float64) * 100.0 + 30.0,
        current_event_index=np.arange(n, dtype=np.int64),
        observation_valid_fraction=np.ones(n, dtype=np.float32),
    )
    first = fit_bridge_arm(arrays, "b0_history", seed=0, epochs=25)
    second = fit_bridge_arm(arrays, "b0_history", seed=999, epochs=25)
    for key in ("joint_nll", "timing_nll", "mark_nll", "participation_nll",
                "rank_nll", "stop_nll"):
        assert first["validation"][key] == second["validation"][key]
    observation = fit_bridge_arm(arrays, "b1_spectral", seed=4, epochs=25)
    for split_name in ("train", "validation"):
        for key in ("joint_nll", "timing_nll", "mark_nll"):
            assert (
                observation["shared_frozen_history_baseline"][split_name][key]
                == first[split_name][key]
            )
    assert observation["train"]["joint_nll"] <= first["train"]["joint_nll"] + 1e-6
    assert observation["n_parameters"] == first["n_parameters"]


def test_innovation_predictors_exclude_current_event_mark() -> None:
    history = np.zeros((3, 16), dtype=np.float32)
    participation = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=bool)
    history[:, 1:3] = 2.0
    history[:, -2:] = participation + 0.5
    first = pre_event_innovation_predictors(history, participation)
    changed = history.copy()
    changed[:, 3:7] += 1000.0
    changed[:, 10:14] += 1000.0
    second = pre_event_innovation_predictors(changed, participation)
    assert np.array_equal(first, second)


def test_placebo_uses_only_older_innovations_without_wrap() -> None:
    times = np.arange(9, dtype=np.float64) * 60.0
    innovation = np.arange(1, 10, dtype=np.float32)
    session = np.zeros(9, dtype=np.int64)
    split = np.zeros(9, dtype=np.int8)
    _, placebo, shifts = exposure_pair(times, innovation, session, split, 1.0)
    effective = shifts[0]["effective_delay_minutes"]
    assert effective > 0.0
    first_nonzero = int(np.flatnonzero(placebo)[0])
    assert first_nonzero > 0
    assert placebo[first_nonzero] == innovation[0]


def test_near_zero_tau_is_current_event_innovation_limit() -> None:
    times = np.asarray([0.0, 1.0, 3.0, 10.0], dtype=np.float64)
    innovation = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    session = np.zeros(len(times), dtype=np.int64)
    split = np.zeros(len(times), dtype=np.int8)
    real, _, _ = exposure_pair(
        times, innovation, session, split, tau_minutes=1e-6
    )
    assert np.array_equal(real, innovation)


def test_event_count_clock_ignores_irregular_physical_interval() -> None:
    times = np.asarray([0.0, 60.0, 660.0], dtype=np.float64)
    innovation = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    session = np.zeros(len(times), dtype=np.int64)
    split = np.zeros(len(times), dtype=np.int8)
    physical, _, _ = exposure_pair(
        times, innovation, session, split, tau_minutes=1.0
    )
    count, _, metadata = exposure_pair(
        times, innovation, session, split, tau_minutes=1.0,
        decay_clock="event_count", event_count_step_minutes=1.0,
    )
    assert np.isclose(physical[1], count[1])
    assert count[2] > physical[2] * 1000.0
    assert metadata[0]["decay_clock"] == "event_count"


def test_fixed_event_count_memory_is_rate_invariant() -> None:
    times = np.asarray([0.0, 1.0, 1000.0, 1001.0], dtype=np.float64)
    innovation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    session = np.zeros(len(times), dtype=np.int64)
    split = np.zeros(len(times), dtype=np.int8)
    memory_events = 50.0
    fast_step = 0.01
    slow_step = 0.2
    fast, _, _ = exposure_pair(
        times, innovation, session, split,
        tau_minutes=memory_events * fast_step,
        decay_clock="event_count", event_count_step_minutes=fast_step,
    )
    slow, _, _ = exposure_pair(
        times, innovation, session, split,
        tau_minutes=memory_events * slow_step,
        decay_clock="event_count", event_count_step_minutes=slow_step,
    )
    assert np.allclose(fast, slow)


def test_event_anchor_rejects_sampled_event_timeline() -> None:
    rng = np.random.default_rng(7)
    n, n_contacts = 24, 2
    arrays = BridgeArrays(
        subject="synthetic",
        history=rng.normal(size=(n, 16)).astype(np.float32),
        spectral=rng.normal(size=(n, 8)).astype(np.float32),
        raw=rng.normal(size=(n, 9)).astype(np.float32),
        log_next_iei=np.ones(n, dtype=np.float32),
        participation=np.ones((n, n_contacts), dtype=np.float32),
        rank=np.zeros((n, n_contacts), dtype=np.float32),
        stop_fraction=np.ones(n, dtype=np.float32),
        split=np.r_[np.zeros(16, dtype=np.int8), np.ones(8, dtype=np.int8)],
        current_time=np.arange(n, dtype=np.float64) * 60.0,
        next_time=np.arange(n, dtype=np.float64) * 60.0 + 30.0,
        current_event_index=np.arange(n, dtype=np.int64) * 2,
        observation_valid_fraction=np.ones(n, dtype=np.float32),
    )
    import pytest
    with pytest.raises(ValueError, match="requires every event"):
        fit_event_anchor(arrays, "t1", epochs=1)
