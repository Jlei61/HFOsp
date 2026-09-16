from __future__ import annotations

import numpy as np

from src.topic5_event_innovation_response_v3_0 import (
    fit_weighted_local_projection,
    future_precedence_brier,
    masked_innovation_projection,
    masked_state_projection,
)
from src.topic5_event_innovation_v3_0 import RankStateBasis


def _basis():
    return RankStateBasis(
        backbone=np.array([0.1, 0.4, 0.7]),
        loadings=np.array([[1.0], [0.5], [-1.0]]),
        singular_values=np.ones(1),
    )


def test_masked_projection_recovers_state_without_treating_absence_as_zero():
    basis = _basis()
    state = np.array([[0.2], [-0.3]])
    fields = basis.inverse(state)
    valid = np.ones_like(fields, dtype=bool)
    valid[0, 1] = False
    fields[0, 1] = 999.0
    recovered, estimable = masked_state_projection(fields, valid, basis, alpha=0)
    np.testing.assert_allclose(recovered, state)
    assert estimable.all()


def test_masked_innovation_projection_does_not_subtract_rank_backbone():
    basis = _basis()
    residual = np.array([[0.2, 0.1, -0.2]])
    recovered, estimable = masked_innovation_projection(
        residual, np.ones_like(residual, dtype=bool), basis, alpha=1e-8
    )
    np.testing.assert_allclose(recovered, [[0.2]], atol=1e-7)
    assert estimable[0]


def test_innovation_projection_allows_regularized_underdetermined_event():
    basis = RankStateBasis(
        backbone=np.zeros(4),
        loadings=np.eye(4),
        singular_values=np.ones(4),
    )
    residual = np.array([[0.2, -0.1, 0.0, 0.0]])
    valid = np.array([[1, 1, 0, 0]], dtype=bool)
    recovered, estimable = masked_innovation_projection(
        residual, valid, basis, alpha=1e-4
    )
    assert estimable[0]
    np.testing.assert_allclose(recovered[0, :2], [0.2, -0.1], atol=1e-3)


def test_precedence_brier_uses_only_co_participating_non_tied_pairs():
    predicted = np.array([[0.0, 1.0, 2.0]])
    ranks = np.array([[0.0, 0.0, 1.0]])
    participation = np.ones_like(ranks, dtype=bool)
    ties = np.array([[0, 0, 1]])
    value = future_precedence_brier(
        predicted, [np.array([0])], ranks, participation, ties
    )
    assert np.isfinite(value)
    assert 0 <= value <= 1

    probability = 1.0 / (1.0 + np.exp(-(predicted[0, 2] - predicted[0, 0])))
    expected = 0.5 * ((1.0 - probability) ** 2 + (1.0 - (1.0 / (1.0 + np.exp(-(predicted[0, 2] - predicted[0, 1]))))) ** 2)
    np.testing.assert_allclose(value, expected)


def test_weighted_local_projection_recovers_known_event_update():
    rng = np.random.default_rng(3)
    pre = rng.normal(size=(500, 2))
    innovation = rng.normal(size=(500, 2))
    future = 0.8 * pre + 0.3 * innovation
    fit = fit_weighted_local_projection(
        pre,
        future,
        innovation,
        alpha=1e-8,
        sample_weight=np.ones(500),
    )
    np.testing.assert_allclose(fit.impulse, np.eye(2) * 0.3, atol=1e-6)
