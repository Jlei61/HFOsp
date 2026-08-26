from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.baseline import (
    ExactHistoryMarkDecoder, HistoryIntensity,
)
from src.topic5_continuous_marked_state_r1.r1_2 import (
    FullAnchorDesign,
    FrozenEmbeddingStateModel,
    _latest_anchor_source,
    _subtract_intervals,
    evaluate_full_t1,
    load_full_design,
    save_full_design,
)


def test_subtract_intervals_preserves_labels_and_splits_at_exclusions() -> None:
    start, stop, label = _subtract_intervals(
        np.asarray([0.0, 20.0]), np.asarray([10.0, 30.0]),
        np.asarray([4, 5]), np.asarray([3.0, 22.0]),
        np.asarray([7.0, 40.0]),
    )
    assert np.array_equal(start, [0.0, 7.0, 20.0])
    assert np.array_equal(stop, [3.0, 10.0, 22.0])
    assert np.array_equal(label, [4, 4, 5])


def _design() -> FullAnchorDesign:
    value = FullAnchorDesign(
        subject="synthetic",
        anchor_time=np.asarray([10.0, 20.0, 30.0, 40.0]),
        anchor_split=np.asarray([0, 0, 1, 1], dtype=np.int8),
        anchor_session=np.zeros(4, dtype=np.int64),
        anchor_history=np.zeros((4, 3), dtype=np.float32),
        event_time=np.asarray([5.0, 15.0, 25.0, 35.0]),
        event_split=np.asarray([0, 0, 1, 1], dtype=np.int8),
        event_session=np.zeros(4, dtype=np.int64),
        event_source_anchor=np.asarray([-1, 0, 1, 2], dtype=np.int64),
        event_history=np.zeros((4, 3), dtype=np.float32),
        event_group_ids=np.asarray([
            [0, -1], [0, 0], [-1, 0], [0, -1],
        ], dtype=np.int64),
        event_group_count=np.ones(4, dtype=np.int64),
        quadrature_time=np.asarray([2.5, 7.5, 12.5, 22.5, 32.5, 42.5]),
        quadrature_split=np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int8),
        quadrature_session=np.zeros(6, dtype=np.int64),
        quadrature_source_anchor=np.asarray([-1, -1, 0, 1, 2, 3], dtype=np.int64),
        quadrature_history=np.zeros((6, 3), dtype=np.float32),
        quadrature_weight_seconds=np.full(6, 5.0),
        session_label=np.asarray([0], dtype=np.int64),
        session_start=np.asarray([0.0]),
    )
    value.validate()
    return value


def _checkpoint() -> dict:
    timing = HistoryIntensity(3, history_visible=True)
    mark = ExactHistoryMarkDecoder(3, 2, np.zeros((1, 2, 2), dtype=np.float32))
    return {
        "timing": {"history": timing.state_dict()},
        "mark": {"history": mark.state_dict()},
    }


def test_latest_anchor_source_preserves_pre_anchor_recorded_support() -> None:
    source = _latest_anchor_source(
        np.asarray([5.0, 10.0, 19.0, 25.0]), np.zeros(4, dtype=np.int64),
        np.asarray([10.0, 20.0]), np.zeros(2, dtype=np.int64),
    )
    assert np.array_equal(source, [-1, -1, 0, 1])


def test_full_design_round_trip_is_pickle_free(tmp_path: Path) -> None:
    path = tmp_path / "design.npz"
    save_full_design(path, _design())
    got = load_full_design(path)
    assert got.subject == "synthetic"
    assert np.array_equal(got.event_source_anchor, [-1, 0, 1, 2])
    assert got.quadrature_weight_seconds.sum() == 30.0


def test_zero_effect_full_support_state_has_exact_correction_off_parity() -> None:
    design = _design()
    model = FrozenEmbeddingStateModel(
        _checkpoint(), 3, 2, np.zeros((1, 2, 2), dtype=np.float32),
        observation_dim=4, state_dim=2,
    )
    embedding = np.random.default_rng(2).normal(size=(4, 4)).astype(np.float32)
    filtered = evaluate_full_t1(
        model, design, embedding, "validation", device="cpu"
    )
    off = evaluate_full_t1(
        model, design, embedding, "validation", device="cpu",
        validation_correction_off=True,
    )
    assert filtered.n_events == 2
    assert filtered.recorded_seconds == 10.0
    assert abs(filtered.joint_nll_per_event - off.joint_nll_per_event) < 1e-7
