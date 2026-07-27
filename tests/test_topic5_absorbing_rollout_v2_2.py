import torch

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    absorbing_mean_field_rollout,
)


def test_absorbing_rollout_conserves_event_and_per_node_mass():
    dtype = torch.float64
    operator = torch.tensor(
        [[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]],
        dtype=dtype,
    )
    result = absorbing_mean_field_rollout(
        initial_state=torch.tensor([0.2, 0.1, 0.0], dtype=dtype),
        operator=operator,
        node_bias=torch.tensor([-0.5, -0.2, -0.4], dtype=dtype),
        eligible=torch.tensor([False, True, True]),
        rho_p=0.6,
        c0=-1.0,
        c_p=-0.5,
        c_n=1.0,
        seen_count=1,
        horizon=3,
    )
    event_total = result.stop_mass.sum() + result.event_survival[-1]
    assert torch.allclose(event_total, torch.tensor(1.0, dtype=dtype), atol=1e-10)
    per_node = (
        result.first_arrival_mass.sum(dim=0)
        + result.stop_before_arrival_mass.sum(dim=0)
        + result.event_survival[-1] * result.not_arrived_survival[-1]
    )
    assert torch.allclose(
        per_node[1:], torch.ones(2, dtype=dtype), atol=1e-10
    )
    assert torch.all(result.first_arrival_mass[:, 0] == 0)


def test_empty_eligible_set_forces_absorbing_stop():
    result = absorbing_mean_field_rollout(
        initial_state=torch.zeros(2),
        operator=torch.zeros(2, 2),
        node_bias=torch.zeros(2),
        eligible=torch.tensor([False, False]),
        rho_p=0.5,
        c0=-10.0,
        c_p=-1.0,
        c_n=1.0,
        seen_count=2,
        horizon=2,
    )
    assert result.stop_probability[0] == 1
    assert result.event_survival[1] == 0
    assert torch.all(result.first_arrival_mass == 0)


def test_zero_horizon_returns_residual_without_fabricating_arrivals():
    result = absorbing_mean_field_rollout(
        initial_state=torch.zeros(2),
        operator=torch.zeros(2, 2),
        node_bias=torch.zeros(2),
        eligible=torch.tensor([False, True]),
        rho_p=0.5,
        c0=0.0,
        c_p=-1.0,
        c_n=1.0,
        seen_count=1,
        horizon=0,
    )
    assert result.first_arrival_mass.shape == (0, 2)
    assert result.event_survival.tolist() == [1.0]
    assert result.not_arrived_survival[-1].tolist() == [0.0, 1.0]

