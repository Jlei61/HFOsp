from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_hazard import (
    HazardDesign,
    downsample_recorded_grid,
    horizon_outcome,
    lagged_persistent_state,
    prequential_nested_hazard,
)


def _design() -> HazardDesign:
    rng = np.random.default_rng(5)
    n = 720
    time = np.arange(n, dtype=np.float64) * 300.0
    onsets = np.arange(90, 690, 75, dtype=np.int64)
    persistent = rng.normal(scale=0.2, size=(n, 3))
    for onset in onsets:
        rows = np.arange(max(0, onset - 6), onset)
        persistent[rows, 0] += np.linspace(0.5, 2.0, len(rows))
    value = HazardDesign(
        source_index=np.arange(n, dtype=np.int64),
        time_epoch=time,
        segment=np.zeros(n, dtype=np.int64),
        history=rng.normal(size=(n, 11)),
        current_observation=rng.normal(size=(n, 4)),
        persistent_state=persistent,
        memoryless_state=rng.normal(scale=0.2, size=(n, 3)),
        onset_time=time[onsets],
        onset_segment=np.zeros(len(onsets), dtype=np.int64),
    )
    value.validate()
    return value


def test_horizon_outcome_requires_complete_negative_window() -> None:
    design = _design()
    outcome, eligible = horizon_outcome(design, 30.0)
    assert outcome[89] == 1
    assert eligible[89]
    assert not eligible[-1]


def test_prequential_hazard_uses_strictly_later_seizures() -> None:
    observed = prequential_nested_hazard(_design(), initial_k=2)
    assert observed["status"] == "COMPLETE_EXPLORATORY"
    assert observed["n_oof_seizures"] >= 5
    assert all(
        row["train_cutoff_epoch"] < row["heldout_onset_epoch"]
        for row in observed["folds"]
    )
    assert all(row["training_labels_known_by_cutoff"] for row in observed["folds"])
    assert all(row["M4_residual_fit_outer_training_only"] for row in observed["folds"])
    assert observed["M4_residual_fit_outer_training_only"] is True
    assert observed["model_definition"]["M1"] != observed["model_definition"]["M3"]


def test_lagged_state_never_uses_future_or_crosses_segment() -> None:
    design = _design()
    state, valid = lagged_persistent_state(design, 10.0)
    assert not valid[:2].any()
    assert np.array_equal(state[2], design.persistent_state[0])


def test_downsampling_restarts_at_each_segment() -> None:
    time = np.asarray([0.0, 30.0, 300.0, 10_000.0, 10_030.0, 10_300.0])
    segment = np.asarray([0, 0, 0, 1, 1, 1])
    observed = downsample_recorded_grid(time, segment, spacing_seconds=300.0)
    assert observed.tolist() == [0, 2, 3, 5]
