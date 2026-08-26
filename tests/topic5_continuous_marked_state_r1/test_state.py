from __future__ import annotations

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.state import (
    ControlledPersistentState, StableGenerator,
)


def test_stable_generator_has_strictly_negative_symmetric_part() -> None:
    torch.manual_seed(4)
    generator = StableGenerator(8)
    with torch.no_grad():
        generator.omega_raw.normal_()
        generator.q_raw.uniform_(-8.0, 2.0)
    matrix = generator.matrix()
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalue = torch.linalg.eigvalsh(symmetric)
    assert float(eigenvalue.max()) < 0.0


def test_batched_anchor_flow_matches_sequential_semigroup() -> None:
    torch.manual_seed(2)
    generator = StableGenerator(4)
    state = torch.randn(4)
    direct = generator.from_anchor(state, torch.tensor([3.0]))[0]
    split = generator.propagate(generator.propagate(state, 1.25), 1.75)
    torch.testing.assert_close(direct, split, rtol=1e-5, atol=1e-6)


def test_observation_correction_is_bounded_and_can_be_disabled() -> None:
    torch.manual_seed(8)
    core = ControlledPersistentState(6, 4)
    state = torch.zeros(4)
    observation = torch.randn(6)
    disabled = core.assimilate(state, 0.0, observation, enabled=False)
    enabled = core.assimilate(state, 0.0, observation, enabled=True)
    torch.testing.assert_close(disabled, state)
    assert np.isfinite(enabled.detach().numpy()).all()
    assert float(enabled.abs().max()) <= 1.0
