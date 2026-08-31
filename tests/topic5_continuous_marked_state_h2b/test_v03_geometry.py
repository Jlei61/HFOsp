from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_geometry import (
    assign_basins,
    evaluate_oos_geometry_fold,
    evaluate_oos_geometry_fold_full_grid,
    fit_decoder_projection,
    fit_two_basins,
)


def test_decoder_projection_and_basins_are_training_defined() -> None:
    rng = np.random.default_rng(2)
    left = rng.normal(loc=-2.0, scale=0.2, size=(100, 3))
    right = rng.normal(loc=2.0, scale=0.2, size=(100, 3))
    training = np.vstack([left, right])
    projection = fit_decoder_projection(training)
    centres = fit_two_basins(projection.transform(training))
    labels = assign_basins(projection.transform(training), centres)
    assert min(np.sum(labels == 0), np.sum(labels == 1)) >= 90


def _synthetic_geometry() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    times, sessions, decoder = [], [], []
    for session in range(3):
        local = np.arange(400, dtype=np.float64) * 30.0 + session * 20_000.0
        state = np.column_stack([
            np.where(np.arange(400) % 120 < 60, -2.0, 2.0),
            np.sin(np.arange(400) / 18.0),
        ]) + rng.normal(scale=0.15, size=(400, 2))
        times.append(local)
        sessions.append(np.full(400, session, dtype=np.int64))
        decoder.append(state)
    onsets = np.asarray([12_000.0, 32_000.0, 52_000.0])
    risk_time, risk_group, risk_decoder = [], [], []
    for segment, onset in enumerate(onsets):
        local = onset - np.arange(60, 0, -1, dtype=np.float64) * 30.0
        progress = np.linspace(-2.0, 2.2, len(local))
        value = np.column_stack([progress, np.zeros(len(local))])
        value += rng.normal(scale=0.04, size=value.shape)
        risk_time.append(local)
        risk_group.append(np.full(len(local), segment, dtype=np.int64))
        risk_decoder.append(value)
    return {
        "interictal_time": np.concatenate(times),
        "interictal_session": np.concatenate(sessions),
        "interictal_decoder": np.vstack(decoder),
        "risk_time": np.concatenate(risk_time),
        "risk_segment": np.concatenate(risk_group),
        "risk_decoder": np.vstack(risk_decoder),
        "onset_time": onsets,
        "onset_segment": np.arange(3, dtype=np.int64),
    }


def test_oos_geometry_recovers_directed_approach_without_heldout_fit() -> None:
    observed = evaluate_oos_geometry_fold(
        **_synthetic_geometry(), heldout_position=2, lookback_minutes=30.0,
    )
    assert observed["status"] == "COMPLETE_EXPLORATORY"
    assert observed["fit_read_heldout_seizure"] is False
    assert observed["n_prior_entry_trajectories"] == 2
    assert observed["n_controls"] >= 5
    assert observed["family_scores"]["directed_approach"] > 0


def test_oos_geometry_requires_two_prior_seizures() -> None:
    observed = evaluate_oos_geometry_fold(
        **_synthetic_geometry(), heldout_position=1, lookback_minutes=30.0,
    )
    assert observed == {
        "status": "NOT_ESTIMABLE", "reason": "insufficient_prior_seizures",
    }


def test_full_grid_geometry_uses_one_extraction_domain() -> None:
    rng = np.random.default_rng(13)
    onsets = np.asarray([12_000.0, 32_000.0, 52_000.0])
    times, groups, decoder = [], [], []
    for group, onset in enumerate(onsets):
        local = onset - np.arange(400, 0, -1, dtype=np.float64) * 30.0
        value = rng.normal(scale=0.25, size=(400, 2))
        value[:, 0] += np.where(np.arange(400) % 100 < 50, -1.5, 1.5)
        value[-60:, 0] = np.linspace(-1.5, 2.0, 60)
        value[-60:, 1] = rng.normal(scale=0.03, size=60)
        times.append(local)
        groups.append(np.full(400, group, dtype=np.int64))
        decoder.append(value)
    observed = evaluate_oos_geometry_fold_full_grid(
        grid_time=np.concatenate(times), grid_segment=np.concatenate(groups),
        grid_decoder=np.vstack(decoder), onset_time=onsets,
        onset_segment=np.arange(3, dtype=np.int64), heldout_position=2,
        lookback_minutes=30.0, grid_spacing_seconds=30.0,
    )
    assert observed["status"] == "COMPLETE_EXPLORATORY"
    assert observed["fit_and_case_extraction_domain_identical"] is True
    assert observed["fit_read_heldout_seizure"] is False
    assert abs(observed["family_scores"]["abrupt_transition"]) < 100.0
