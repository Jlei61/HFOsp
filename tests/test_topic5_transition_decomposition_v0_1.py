from __future__ import annotations

import numpy as np

from src.topic5_transition_decomposition_v0_1 import (
    StopParameters,
    contact_shaft,
    cross_shaft_conditional_nll,
    estimate_pair_residual,
    event_nll,
    fibonacci_axes,
    history_contacts,
    positive_contact_nll_by_shaft,
    symmetric_skew,
    weighted_ridge_residual,
)


def _events() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 2, -1],
            [0, 1, -1, 2],
            [2, 1, 0, -1],
            [1, 0, 2, -1],
        ],
        dtype=np.int64,
    )


def test_contact_shaft_keeps_internal_digits_and_removes_contact_number() -> None:
    assert contact_shaft("FLA12") == "FLA"
    assert contact_shaft("A1") == "A"


def test_symmetric_skew_reconstructs_residual() -> None:
    matrix = np.asarray([[0.0, 2.0], [-1.0, 0.0]])
    symmetric, skew = symmetric_skew(matrix)
    assert np.allclose(symmetric, symmetric.T)
    assert np.allclose(skew, -skew.T)
    assert np.allclose(symmetric + skew, matrix)


def test_pair_residual_uses_only_requested_training_events() -> None:
    groups = _events()
    first = estimate_pair_residual(groups, np.asarray([0, 1]))
    second = estimate_pair_residual(groups, np.asarray([2, 3]))
    assert not np.allclose(first.residual, second.residual)


def test_node_model_has_finite_exact_set_likelihood() -> None:
    groups = _events()
    pair = estimate_pair_residual(groups, np.asarray([0, 1, 2]))
    nll = event_nll(
        groups[3],
        node_logit=pair.node_logit,
        residual=np.zeros_like(pair.residual),
        stop=StopParameters(c0=-1.0, c_n=1.0),
    )
    assert np.isfinite(nll)
    assert nll > 0


def test_source_only_history_keeps_the_observed_source() -> None:
    event = np.asarray([0, 1, 2, 3, -1], dtype=np.int64)
    contacts, weights = history_contacts(event, 2, "source_only")
    assert contacts.tolist() == [0]
    assert weights.tolist() == [1.0]


def test_weighted_ridge_and_axis_inventory_are_deterministic() -> None:
    target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    feature = target.copy()
    fitted, coefficient, mse = weighted_ridge_residual(
        target, [feature], np.ones_like(target), ridge=1.0e-6
    )
    assert coefficient[0] > 0.99
    assert mse < 1.0e-10
    assert np.array_equal(fibonacci_axes(8), fibonacci_axes(8))


def test_positive_contact_readout_separates_same_and_cross_shaft() -> None:
    groups = np.asarray([[0, 1, 2, -1]], dtype=np.int64)
    result = positive_contact_nll_by_shaft(
        groups,
        np.asarray([0]),
        names=["A1", "A2", "B1", "B2"],
        node_logit=np.zeros(4),
        residual=np.zeros((4, 4)),
    )
    assert result["n_same_shaft_positive_contacts"] == 1
    assert result["n_cross_shaft_positive_contacts"] == 1


def test_cross_shaft_endpoint_includes_negative_contact_competition() -> None:
    groups = np.asarray([[0, 1, 2, -1]], dtype=np.int64)
    node_logit = np.zeros(4)
    baseline, n_prefixes = cross_shaft_conditional_nll(
        groups,
        np.asarray([0]),
        names=["A1", "A2", "B1", "C1"],
        node_logit=node_logit,
        residual=np.zeros((4, 4)),
    )
    misleading = np.zeros((4, 4))
    misleading[1, 3] = 5.0
    with_false_positive, _ = cross_shaft_conditional_nll(
        groups,
        np.asarray([0]),
        names=["A1", "A2", "B1", "C1"],
        node_logit=node_logit,
        residual=misleading,
    )
    assert n_prefixes == 1
    assert len(baseline) == 1
    assert with_false_positive[0] > baseline[0]
