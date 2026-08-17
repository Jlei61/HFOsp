from __future__ import annotations

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_cumulative_response import (
    CumulativeRows,
    matched_cumulative_donor,
    matched_cumulative_donor_pool,
    project_exposure_innovations,
)
from src.topic5_event_innovation_v3_0 import RankStateBasis


def _basis():
    return RankStateBasis(
        backbone=np.zeros(3),
        loadings=np.array([[1.0], [0.5], [-1.0]]),
        singular_values=np.ones(1),
    )


def test_exposure_projection_recovers_sum_dose_and_alignment():
    values = {
        index: (
            np.array([0.2, 0.1, -0.2]),
            np.ones(3, dtype=bool),
        )
        for index in range(3)
    }
    result = project_exposure_innovations(np.arange(3), values, _basis())
    assert result is not None
    cumulative, dose, alignment = result
    np.testing.assert_allclose(cumulative, [0.6], atol=1e-4)
    assert dose > 0
    assert alignment == 1.0


def test_iei_decay_makes_within_exposure_order_identifiable():
    first = np.array([0.2, 0.1, -0.2])
    second = np.array([-0.1, -0.05, 0.1])
    valid = np.ones(3, dtype=bool)
    times = np.array([0.0, 10.0])
    forward = project_exposure_innovations(
        np.array([0, 1]),
        {0: (first, valid), 1: (second, valid)},
        _basis(),
        event_times=times,
        tau_seconds=5.0,
    )
    reverse = project_exposure_innovations(
        np.array([0, 1]),
        {0: (second, valid), 1: (first, valid)},
        _basis(),
        event_times=times,
        tau_seconds=5.0,
    )
    assert forward is not None and reverse is not None
    assert not np.allclose(forward[0], reverse[0])


def test_cumulative_donor_reassigns_complete_vectors():
    n = 20
    rows = CumulativeRows(
        anchor_event=np.arange(n),
        group=np.zeros(n, dtype=int),
        pre_state=np.arange(n, dtype=float)[:, None],
        future_state=np.zeros((n, 1)),
        cumulative_innovation=np.column_stack([np.arange(n), -np.arange(n)]),
        dose=np.arange(n, dtype=float) + 1,
        alignment=np.linspace(0, 1, n),
        nuisance=np.column_stack([np.linspace(0, 1, n), np.zeros((n, 2))]),
        observed_future_field=np.zeros((n, 2)),
        future_support=np.ones((n, 2)),
        future_windows=[np.array([index]) for index in range(n)],
    )
    donor, audit = matched_cumulative_donor(rows, seed=3, top_k=2)
    assert not np.any(np.all(donor == rows.cumulative_innovation, axis=1))
    assert audit["eligible_anchor_fraction"] == 1.0
    pools, _ = matched_cumulative_donor_pool(rows, top_k=2)
    assert all(index not in pool for index, pool in enumerate(pools))
