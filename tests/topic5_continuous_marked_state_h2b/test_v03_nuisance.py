from __future__ import annotations

import numpy as np
from dataclasses import replace

from src.topic5_continuous_marked_state_h2b.v03_nuisance import (
    build_nonoverlap_future_targets,
    nested_prequential_increment,
    prior_seizure_features,
)
from src.topic5_continuous_marked_state_r1.r1_2 import FullAnchorDesign


def _design() -> FullAnchorDesign:
    anchor_time = np.arange(21, dtype=np.float64) * 30.0
    event_time = np.asarray([100.0, 400.0], dtype=np.float64)
    return FullAnchorDesign(
        subject="synthetic",
        anchor_time=anchor_time,
        anchor_split=np.ones(21, dtype=np.int8),
        anchor_session=np.zeros(21, dtype=np.int64),
        anchor_history=np.zeros((21, 2), dtype=np.float32),
        event_time=event_time,
        event_split=np.ones(2, dtype=np.int8),
        event_session=np.zeros(2, dtype=np.int64),
        event_source_anchor=np.asarray([3, 13], dtype=np.int64),
        event_history=np.zeros((2, 2), dtype=np.float32),
        event_group_ids=np.asarray([[0, -1], [0, 1]], dtype=np.int64),
        event_group_count=np.asarray([1, 2], dtype=np.int64),
        quadrature_time=np.empty(0, dtype=np.float64),
        quadrature_split=np.empty(0, dtype=np.int8),
        quadrature_session=np.empty(0, dtype=np.int64),
        quadrature_source_anchor=np.empty(0, dtype=np.int64),
        quadrature_history=np.empty((0, 2), dtype=np.float32),
        quadrature_weight_seconds=np.empty(0, dtype=np.float64),
        session_label=np.asarray([0], dtype=np.int64),
        session_start=np.asarray([0.0], dtype=np.float64),
    )


def test_future_targets_are_nonoverlapping() -> None:
    observed = build_nonoverlap_future_targets(
        _design(), np.ones(21, dtype=bool), horizon_seconds=300.0,
    )
    assert observed["anchor"].tolist() == [0, 10]
    assert observed["event_count"].tolist() == [1.0, 1.0]
    assert observed["first_event_delay_seconds"].tolist() == [100.0, 100.0]
    assert np.diff(observed["time"])[0] == 300.0


def test_future_targets_keep_no_event_window_with_censored_delay() -> None:
    design = _design()
    design = replace(
        design,
        event_time=design.event_time[:1],
        event_split=design.event_split[:1],
        event_session=design.event_session[:1],
        event_source_anchor=design.event_source_anchor[:1],
        event_history=design.event_history[:1],
        event_group_ids=design.event_group_ids[:1],
        event_group_count=design.event_group_count[:1],
    )
    observed = build_nonoverlap_future_targets(
        design, np.ones(21, dtype=bool), horizon_seconds=300.0,
    )
    assert observed["event_count"].tolist() == [1.0, 0.0]
    assert observed["first_event_delay_seconds"].tolist() == [100.0, 300.0]
    assert observed["has_event"].tolist() == [True, False]


def test_future_target_rejects_window_that_crosses_anchor_gap() -> None:
    design = _design()
    keep = np.ones(21, dtype=bool)
    keep[5:9] = False
    observed = build_nonoverlap_future_targets(
        design, keep, horizon_seconds=300.0,
    )
    assert observed["anchor"].tolist() == [10]


def test_prior_seizure_feature_is_strictly_past_only() -> None:
    observed = prior_seizure_features(
        np.asarray([100.0, 200.0, 201.0]), [200.0],
    )
    assert observed[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert observed[2, 1] > 0.0


def test_prequential_increment_recovers_added_signal() -> None:
    rng = np.random.default_rng(12)
    n = 400
    base = rng.normal(size=(n, 3))
    increment = rng.normal(size=(n, 2))
    target = np.column_stack([
        0.2 * base[:, 0] + 1.5 * increment[:, 0],
        -0.3 * base[:, 1] + 1.2 * increment[:, 1],
    ]) + rng.normal(scale=0.05, size=(n, 2))
    observed = nested_prequential_increment(
        base, increment, target, np.arange(n, dtype=np.float64),
    )
    assert observed["status"] == "COMPLETE"
    assert observed["pass"] is True
    assert observed["median_relative_improvement"] > 0.8


def test_prequential_first_fold_has_more_rows_than_full_design() -> None:
    rng = np.random.default_rng(21)
    base = rng.normal(size=(240, 50))
    increment = rng.normal(size=(240, 20))
    target = increment[:, :2] + rng.normal(scale=0.1, size=(240, 2))
    observed = nested_prequential_increment(
        base, increment, target, np.arange(240, dtype=np.float64),
    )
    assert observed["minimum_train_rows"] >= 90
    assert observed["folds"][0]["train_rows"] >= 90
