from __future__ import annotations

import numpy as np
import torch

from src.topic5_group_event_state.v037.h2a import _same_prefix_shift_context
from src.topic5_group_event_state.v037.h2a_marks import (
    fit_rich_mark_readout,
    leaked_event_context,
    mean_squared_mark,
)


def test_leaked_context_contains_participation_and_relative_group_order() -> None:
    ranks = np.asarray([[0, 1, -1], [1, 0, 2]], dtype=np.int16)
    value = leaked_event_context(ranks)
    assert value.shape == (2, 8)
    assert np.array_equal(value[0, :3], [1.0, 1.0, 0.0])
    assert value[1, 3 + 1] < value[1, 3 + 0] < value[1, 3 + 2]


def test_rich_mark_ridge_uses_inner_only_for_regularisation() -> None:
    rng = np.random.default_rng(7)
    x_fit = rng.normal(size=(100, 5))
    beta = rng.normal(size=(5, 3))
    y_fit = x_fit @ beta + rng.normal(scale=0.02, size=(100, 3))
    x_inner = rng.normal(size=(40, 5)); y_inner = x_inner @ beta
    model, audit = fit_rich_mark_readout(x_fit, y_fit, x_inner, y_inner)
    assert audit["selected_ridge"] in {1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}
    assert mean_squared_mark(model, x_inner, y_inner) < np.mean(y_inner * y_inner)


def test_same_prefix_wrong_time_requires_same_session_and_physical_distance() -> None:
    context = torch.arange(8, dtype=torch.float32)[:, None]
    ranks = np.asarray([
        [0, 1, -1], [0, 1, -1], [0, 1, -1], [0, 1, -1],
        [0, 1, -1], [0, 1, -1], [0, 1, -1], [0, 1, -1],
    ], dtype=np.int16)
    time = np.asarray([0, 10, 1000, 1010, 0, 10, 1000, 1010], dtype=np.float64)
    segment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    shifted, valid = _same_prefix_shift_context(
        context, ranks, time, segment, np.arange(8),
        minimum_events=4, minimum_seconds=500.0,
    )
    assert np.all(valid)
    donor = shifted[:, 0].numpy().astype(int)
    assert np.all(segment[donor] == segment)
    assert np.all(np.abs(time[donor] - time) >= 500.0)
