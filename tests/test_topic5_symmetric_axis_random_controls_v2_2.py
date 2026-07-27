from __future__ import annotations

import numpy as np
import torch

from scripts.train_topic5_symmetric_axis_propagation_state_v2_2 import (
    batch_event_losses,
)
from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    SymmetricAxisPropagationStateRNN,
)
from src.topic5_symmetric_axis_random_controls_v2_2 import (
    fixed_axis_event_losses_batch,
    fixed_axis_operator_batch,
)


def _events() -> tuple[torch.Tensor, torch.Tensor]:
    groups = torch.tensor(
        [
            [0, 1, 2, -1, -1],
            [1, 0, 1, 2, -1],
            [0, 2, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    counts = torch.tensor([3, 3, 4], dtype=torch.long)
    return groups, counts


def _scalar_models(
    coords: np.ndarray,
    bias: np.ndarray,
    axes: torch.Tensor,
    gamma_raw: torch.Tensor,
    gain_raw: torch.Tensor,
) -> list[SymmetricAxisPropagationStateRNN]:
    models = []
    for index in range(len(axes)):
        model = SymmetricAxisPropagationStateRNN(
            coords=coords,
            node_bias=bias,
        )
        with torch.no_grad():
            model.axis_raw.copy_(axes[index])
            model.gamma_raw.copy_(gamma_raw[index])
            model.gain_raw.copy_(gain_raw[index])
            model.raw_anisotropy.copy_(torch.tensor(0.37))
            model.raw_rho.copy_(torch.tensor(-0.21))
            model.c0.copy_(torch.tensor(-0.83))
            model.raw_c_p.copy_(torch.tensor(0.19))
            model.raw_c_n.copy_(torch.tensor(-0.14))
        models.append(model)
    return models


def test_batched_operator_matches_scalar_model() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.1, -0.2, 0.2],
            [0.2, 1.4, -0.1],
            [1.2, 1.1, 0.3],
        ]
    )
    bias = np.linspace(-1.2, -0.4, len(coords))
    axes = torch.tensor(
        [[1.0, 0.3, -0.2], [-0.4, 0.8, 0.5], [0.2, -0.3, 0.9]]
    )
    gamma_raw = torch.tensor([-0.7, 0.1, 0.9])
    gain_raw = torch.tensor([0.2, -0.4, 0.6])
    models = _scalar_models(coords, bias, axes, gamma_raw, gain_raw)
    batched = fixed_axis_operator_batch(
        coords=models[0].coords,
        axes=axes,
        anisotropy_ratio=models[0].anisotropy_ratio,
        gamma_raw=gamma_raw,
        gain_raw=gain_raw,
        local_scale=models[0].local_scale,
    )
    scalar = torch.stack(
        [model.operator_components()["W"] for model in models]
    )
    torch.testing.assert_close(batched, scalar, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        batched, batched.transpose(1, 2), rtol=1.0e-6, atol=1.0e-7
    )


def test_batched_h3_losses_match_scalar_model() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.1, -0.2, 0.2],
            [0.2, 1.4, -0.1],
            [1.2, 1.1, 0.3],
        ]
    )
    bias = np.linspace(-1.2, -0.4, len(coords))
    axes = torch.tensor(
        [[1.0, 0.3, -0.2], [-0.4, 0.8, 0.5], [0.2, -0.3, 0.9]]
    )
    gamma_raw = torch.tensor([-0.7, 0.1, 0.9])
    gain_raw = torch.tensor([0.2, -0.4, 0.6])
    models = _scalar_models(coords, bias, axes, gamma_raw, gain_raw)
    groups, counts = _events()
    operator = fixed_axis_operator_batch(
        coords=models[0].coords,
        axes=axes,
        anisotropy_ratio=models[0].anisotropy_ratio,
        gamma_raw=gamma_raw,
        gain_raw=gain_raw,
        local_scale=models[0].local_scale,
    )
    batched = fixed_axis_event_losses_batch(
        operator=operator,
        groups=groups,
        counts=counts,
        node_bias=models[0].node_bias,
        rho_p=models[0].rho_p,
        c0=models[0].c0,
        c_p=models[0].c_p,
        c_n=models[0].c_n,
        training_horizon=3,
    )
    for direction, model in enumerate(models):
        scalar = batch_event_losses(
            model=model,
            groups=groups,
            counts=counts,
            training_horizon=3,
            evaluate_full_future=False,
        )
        for key in (
            "event_next_nll",
            "event_future_nll",
            "event_objective",
        ):
            torch.testing.assert_close(
                batched[key][direction],
                scalar[key],
                rtol=2.0e-6,
                atol=2.0e-6,
            )


def test_batched_next_only_gradient_is_finite() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.1, -0.2, 0.2],
            [0.2, 1.4, -0.1],
            [1.2, 1.1, 0.3],
        ]
    )
    axes = torch.tensor([[1.0, 0.2, 0.1], [0.1, 1.0, -0.2]])
    gamma_raw = torch.zeros(2, requires_grad=True)
    gain_raw = torch.zeros(2, requires_grad=True)
    groups, counts = _events()
    operator = fixed_axis_operator_batch(
        coords=coords,
        axes=axes,
        anisotropy_ratio=2.1,
        gamma_raw=gamma_raw,
        gain_raw=gain_raw,
        local_scale=1.0,
    )
    losses = fixed_axis_event_losses_batch(
        operator=operator,
        groups=groups,
        counts=counts,
        node_bias=torch.linspace(-1.2, -0.4, 5),
        rho_p=torch.tensor(0.45),
        c0=torch.tensor(-0.8),
        c_p=torch.tensor(-0.7),
        c_n=torch.tensor(0.6),
        training_horizon=0,
    )
    losses["event_objective"].sum().backward()
    assert gamma_raw.grad is not None
    assert gain_raw.grad is not None
    assert torch.isfinite(gamma_raw.grad).all()
    assert torch.isfinite(gain_raw.grad).all()
