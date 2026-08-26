from __future__ import annotations

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.baseline import (
    ExactHistoryMarkDecoder,
    HistoryIntensity,
    intensity_loss,
)
from src.topic5_continuous_marked_state_r1.design import SplitDesign


def test_intensity_loss_has_correct_constant_rate_optimum_direction() -> None:
    design = SplitDesign(
        split="train", event_index=np.arange(2),
        event_history=np.zeros((2, 1), dtype=np.float32),
        quadrature_history=np.zeros((4, 1), dtype=np.float32),
        quadrature_weight_seconds=np.full(4, 2.5),
        recorded_seconds=10.0,
    )
    model = HistoryIntensity(1, history_visible=False)
    with torch.no_grad():
        model.intercept.fill_(np.log(0.2))
    loss = intensity_loss(model, design)
    expected = -(2 * np.log(0.2) - 2.0) / 2
    np.testing.assert_allclose(float(loss), expected, rtol=1e-6)


def test_exact_mark_decoder_is_finite_and_differentiable() -> None:
    history = torch.randn(3, 5)
    group_ids = torch.tensor([[0, 1, 0, -1], [0, -1, -1, -1], [1, 0, 2, -1]])
    group_count = torch.tensor([2, 1, 3])
    adjacency = np.stack([np.eye(4), np.eye(4), np.eye(4)]).astype(np.float32)
    model = ExactHistoryMarkDecoder(5, 4, adjacency)
    terms = model(history, group_ids, group_count)
    loss = -terms.event_log_prob.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
