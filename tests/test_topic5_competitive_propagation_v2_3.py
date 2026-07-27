from __future__ import annotations

import numpy as np
import torch

from src.topic5_competitive_propagation_v2_3 import (
    CompetitivePropagationRNN,
    has_non_source_tie,
)


def _model(axis: np.ndarray, **kwargs: bool) -> CompetitivePropagationRNN:
    return CompetitivePropagationRNN(
        coords=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.3, 0.0],
                [3.0, 0.4, 0.2],
            ]
        ),
        axis=axis,
        node_logit=np.zeros(4),
        rho_propagation=0.5,
        rho_competition=0.75,
        **kwargs,
    )


def test_symmetric_and_directional_operators_have_required_structure() -> None:
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    symmetric, directed = model.operators()
    assert torch.allclose(symmetric, symmetric.T)
    assert torch.all(symmetric >= 0)
    assert torch.allclose(directed, -directed.T)


def test_axis_sign_does_not_change_source_conditioned_logits() -> None:
    positive = _model(np.asarray([1.0, 0.0, 0.0]))
    negative = _model(np.asarray([-1.0, 0.0, 0.0]))
    with torch.no_grad():
        positive.raw_source_beta.fill_(0.4)
        negative.raw_source_beta.copy_(positive.raw_source_beta)
    event = np.asarray([0, 1, 2, 3])
    first = positive.forward_event(event).probabilities
    second = negative.forward_event(event).probabilities
    assert len(first) == len(second)
    assert all(torch.allclose(a, b, atol=1.0e-10) for a, b in zip(first, second))


def test_categorical_probabilities_sum_to_one_and_mask_seen_contacts() -> None:
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    result = model.forward_event(np.asarray([0, 1, 2, 3]))
    for step, (probability, eligible) in enumerate(
        zip(result.probabilities, result.eligible_indices)
    ):
        assert torch.allclose(probability.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert not np.any(np.isin(np.arange(step + 1), eligible.cpu().numpy()))


def test_event_state_is_reset_between_calls() -> None:
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    event = np.asarray([0, 1, 2, 3])
    first = model.forward_event(event)
    second = model.forward_event(event)
    assert torch.allclose(first.losses, second.losses)
    assert torch.allclose(
        first.propagation_states[-1], second.propagation_states[-1]
    )


def test_tied_non_source_rank_is_rejected() -> None:
    assert has_non_source_tie(np.asarray([0, 1, 1, 2]))
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    try:
        model.forward_event(np.asarray([0, 1, 1, 2]))
    except ValueError as error:
        assert "tied ranks" in str(error)
    else:
        raise AssertionError("tied event should be rejected")


def test_ablations_remove_the_intended_terms() -> None:
    local = _model(np.asarray([1.0, 0.0, 0.0]), local_only=True)
    one_state = _model(
        np.asarray([1.0, 0.0, 0.0]), no_competition=True
    )
    no_source = _model(np.asarray([1.0, 0.0, 0.0]), no_source=True)
    no_history = _model(np.asarray([1.0, 0.0, 0.0]), no_history=True)
    assert float(local.gamma) == 0.0
    assert float(one_state.gain_competition) == 0.0
    assert float(no_source.source_beta) == 0.0
    assert float(no_history.rho_propagation) == 0.0
    assert float(no_history.rho_competition) == 0.0


def test_gradients_only_reach_four_scalar_parameters() -> None:
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    loss = model.mean_event_nll(np.asarray([0, 1, 2, 3]))
    loss.backward()
    trainable = {name for name, parameter in model.named_parameters()}
    assert trainable == {
        "raw_gamma",
        "raw_gain_propagation",
        "raw_gain_competition",
        "raw_source_beta",
    }
    assert all(
        parameter.grad is not None for parameter in model.parameters()
    )


def test_vectorized_batch_matches_eventwise_nll() -> None:
    model = _model(np.asarray([1.0, 0.0, 0.0]))
    groups = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 2, 1, -1],
        ]
    )
    counts = np.asarray([4, 3])
    batch = model.forward_batch(groups, counts)
    eventwise = torch.stack(
        [model.mean_event_nll(event) for event in groups]
    )
    assert torch.allclose(batch.event_losses, eventwise, atol=1.0e-10)
    assert torch.equal(
        batch.decision_count,
        torch.tensor([3.0, 2.0], dtype=torch.float64),
    )
