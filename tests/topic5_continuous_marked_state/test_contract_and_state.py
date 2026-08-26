from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import BridgeHead
from src.topic5_continuous_marked_state.regular_t1 import (
    FrozenBaselineStateHead,
    PreparedRegularT1,
    RegularT1Model,
    _future_transition_rows_from_anchor,
)
from src.topic5_continuous_marked_state.state import (
    ExposureState,
    T1T2Core,
    correction_off_rollout,
)


def test_split_contract_rejects_sealed_time() -> None:
    split = contract.load_split("epilepsiae_620")
    contract.assert_development_times(
        "epilepsiae_620", np.asarray([split.train_end_epoch - 1.0]), "train"
    )
    with pytest.raises(ValueError, match="SEALED/SPLIT VIOLATION"):
        contract.assert_development_times(
            "epilepsiae_620", np.asarray([split.dev_end_epoch]), "validation"
        )


def test_stable_generator_has_negative_real_eigenvalues() -> None:
    core = T1T2Core(observation_dim=3, state_dim=4, t2=False)
    eig = torch.linalg.eigvals(core.generator.matrix()).real
    assert bool((eig < 0).all())


def test_batched_generator_matches_scalar_exact_propagation() -> None:
    torch.manual_seed(17)
    core = T1T2Core(observation_dim=3, state_dim=8, t2=False)
    with torch.no_grad():
        core.generator.omega_raw.normal_(0.0, 0.03)
        core.generator.q_raw.normal_(-4.0, 0.2)
        core.generator.mu.normal_(0.0, 0.1)
    z = torch.randn(8)
    times = torch.tensor([0.0, 0.1, 1.0, 7.5, 120.0])
    batched = core.generator.propagate_many_from_same_state(z, times)
    scalar = torch.stack([core.generator.propagate(z, value) for value in times])
    assert torch.allclose(batched, scalar, atol=2e-6, rtol=2e-6)


def test_correction_off_ignores_all_future_observations() -> None:
    torch.manual_seed(2)
    core = T1T2Core(observation_dim=3, state_dim=4, t2=True)
    obs = [torch.randn(3) for _ in range(10)]
    altered = [x.clone() for x in obs]
    for i in range(4, 10):
        altered[i] += 1000.0
    kwargs = dict(
        core=core, z0=torch.zeros(4), delta_minutes=[1.0] * 10,
        innovations=[0.1] * 10, tau_minutes=60.0, anchor_index=3,
    )
    a = correction_off_rollout(observations=obs, **kwargs)
    b = correction_off_rollout(observations=altered, **kwargs)
    assert torch.equal(a[4:], b[4:])


def test_t1_has_no_exposure_forcing_even_if_parameter_changes() -> None:
    core = T1T2Core(observation_dim=2, state_dim=3, t2=False)
    with torch.no_grad():
        core.exposure_to_state.fill_(100.0)
    obs = [torch.zeros(2) for _ in range(4)]
    a = correction_off_rollout(
        core, torch.zeros(3), obs, [1.0] * 4, [0.0] * 4, 60.0, 0
    )
    b = correction_off_rollout(
        core, torch.zeros(3), obs, [1.0] * 4, [10.0] * 4, 60.0, 0
    )
    assert torch.equal(a, b)


def test_repeated_observation_corrections_do_not_become_event_counter() -> None:
    torch.manual_seed(5)
    core = T1T2Core(observation_dim=3, state_dim=4, t2=False)
    z = torch.zeros(4)
    exposure = ExposureState(torch.zeros(()), 60.0)
    for _ in range(500):
        z, exposure = core.step(
            z, 0.0, torch.full((3,), 100.0), exposure,
            correction_enabled=True,
        )
    assert float(z.abs().max()) <= 1.0


def test_frozen_history_plus_zero_initial_state_exactly_equals_baseline() -> None:
    torch.manual_seed(8)
    baseline = BridgeHead(
        input_dim=5, n_contacts=3,
        time_sigma=0.7, rank_sigma=0.4, stop_sigma=0.2,
    )
    head = FrozenBaselineStateHead(baseline, state_dim=4)
    history = torch.randn(7, 5)
    state = torch.zeros(7, 4)
    log_iei = torch.randn(7)
    participation = (torch.rand(7, 3) > 0.3).float()
    rank = torch.rand(7, 3)
    stop = torch.rand(7)
    expected = baseline.losses(
        history, log_iei, participation, rank, stop
    )
    got = head.losses(
        history, state, log_iei, participation, rank, stop
    )
    for key in (
        "joint_nll", "timing_nll", "mark_nll",
        "participation_nll", "rank_nll", "stop_nll",
    ):
        assert torch.equal(expected[key], got[key])
    assert not any(parameter.requires_grad for parameter in head.baseline.parameters())
    assert all(torch.count_nonzero(module.weight) > 0 for module in (
        head.state_time, head.state_participation, head.state_rank, head.state_stop,
    ))


def test_identity_observer_has_zero_effect_but_nonzero_first_gradient() -> None:
    torch.manual_seed(12)
    baseline = BridgeHead(
        input_dim=5, n_contacts=3,
        time_sigma=0.7, rank_sigma=0.4, stop_sigma=0.2,
    )
    model = RegularT1Model(
        history_dim=5, n_contacts=3,
        scales={"time_sigma": 0.7, "rank_sigma": 0.4, "stop_sigma": 0.2},
        baseline=baseline, state_dim=4,
    )
    z, _ = model.core.step(
        torch.zeros(4), 0.0,
        model.observation_project(torch.randn(contract.STATE_OBSERVATION_DIM)),
        ExposureState(torch.zeros(()), 60.0), correction_enabled=True,
    )
    assert torch.equal(z, torch.zeros_like(z))
    loss = model.event_losses(
        torch.randn(5), z, torch.tensor(0.3),
        torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.2, 0.0, 0.8]),
        torch.tensor(0.5),
    )["joint_nll"].mean()
    loss.backward()
    gradient = model.core.observer.correction.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_post_anchor_transition_rows_target_strictly_future_events() -> None:
    n_events = 9
    sequence = PreparedRegularT1(
        subject="synthetic",
        history=torch.zeros(n_events, 1),
        observation=torch.zeros(0, contract.STATE_OBSERVATION_DIM),
        observation_time=np.empty(0),
        observation_split=np.empty(0, dtype=np.int8),
        event_time=np.arange(n_events, dtype=float),
        next_time=np.arange(1, n_events + 1, dtype=float),
        session=np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1]),
        split=np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8),
        log_iei=torch.zeros(n_events),
        participation=torch.ones(n_events, 1),
        rank=torch.zeros(n_events, 1),
        stop=torch.zeros(n_events),
    )
    rows = _future_transition_rows_from_anchor(sequence, anchor=1, horizon=3)
    assert rows is not None
    assert rows.tolist() == [1, 2, 3]
    # Each row i predicts event i+1, so the actual targets are the next three
    # events and never the anchor itself.
    assert (rows + 1).tolist() == [2, 3, 4]
    assert _future_transition_rows_from_anchor(
        sequence, anchor=4, horizon=3
    ) is None
