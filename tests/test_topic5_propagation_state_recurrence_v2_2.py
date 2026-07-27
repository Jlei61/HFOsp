import numpy as np
import torch

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    SymmetricAxisPropagationStateRNN,
    estimate_node_hazard_bias,
    exact_one_step_log_probability,
)


def test_node_bias_counts_terminal_decision_and_is_not_event_participation():
    events = [
        np.array([0, 1, -1]),
        np.array([0, -1, 1]),
    ]
    result = estimate_node_hazard_bias(events)
    np.testing.assert_array_equal(result["n_next"], [0, 1, 1])
    # A contact remains eligible at the terminal decision even when it never
    # participates in that event.
    np.testing.assert_array_equal(result["n_eligible"], [0, 3, 3])
    assert not np.allclose(
        result["hazard_probability"],
        result["event_participation_probability"],
    )


def test_exact_likelihood_separates_nonempty_set_from_stop():
    logits = torch.tensor([0.2, -0.3, -torch.inf], dtype=torch.float64)
    eligible = torch.tensor([True, True, False])
    target = torch.tensor([True, False, False])
    stop_logit = torch.tensor(-0.4, dtype=torch.float64)
    nonterminal = exact_one_step_log_probability(
        node_logits=logits,
        eligible=eligible,
        next_set=target,
        stop_logit=stop_logit,
        terminal=False,
    )
    terminal = exact_one_step_log_probability(
        node_logits=logits,
        eligible=eligible,
        next_set=torch.zeros(3, dtype=torch.bool),
        stop_logit=stop_logit,
        terminal=True,
    )
    assert torch.isfinite(nonterminal)
    assert torch.allclose(terminal, torch.nn.functional.logsigmoid(stop_logit))


def test_event_state_resets_and_does_not_carry_across_events():
    coords = np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)])
    model = SymmetricAxisPropagationStateRNN(coords=coords, node_bias=np.zeros(4))
    first = torch.tensor([True, False, False, False])
    state_a = model.observe(model.reset_state(), first)
    state_b = model.observe(model.reset_state(), first)
    assert torch.allclose(state_a, state_b)
    assert torch.all(model.reset_state() == 0)


def test_seen_contact_is_masked_to_zero_hazard():
    coords = np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)])
    model = SymmetricAxisPropagationStateRNN(coords=coords, node_bias=np.zeros(4))
    state = model.observe(
        model.reset_state(), torch.tensor([True, False, False, False])
    )
    decision = model.decision(
        state, torch.tensor([True, False, False, False])
    )
    assert torch.isneginf(decision["node_logits"][0])
    assert torch.sigmoid(decision["node_logits"][0]) == 0


def test_parameter_sign_constraints_are_structural():
    coords = np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)])
    model = SymmetricAxisPropagationStateRNN(coords=coords, node_bias=np.zeros(4))
    assert 1.0 <= float(model.anisotropy_ratio) <= 4.0
    assert 0.0 <= float(model.rho_p) < 1.0
    assert float(model.c_p) <= 0.0
    assert float(model.c_n) >= 0.0
