import numpy as np
import torch
from types import SimpleNamespace

from src.topic5_continuous_marked_state_r1.t2_s1 import (
    OneStepDesign, SignedExposureEdge, build_one_step_design, evaluate_edge,
    fit_edge, fit_load_innovation, rolling_event_exposure,
    state_matched_placebo,
)


def test_rolling_exposure_resets_at_recorded_gap() -> None:
    innovation = np.arange(1, 9, dtype=np.float32)
    segment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    exposure, eligible = rolling_event_exposure(innovation, segment, 3)
    assert eligible.tolist() == [False, False, True, True, False, False, True, True]
    assert np.isclose(exposure[2], (1 + 2 + 3) / np.sqrt(3))
    assert np.isclose(exposure[6], (5 + 6 + 7) / np.sqrt(3))


def test_load_innovation_uses_only_train_outcomes() -> None:
    rng = np.random.default_rng(3)
    state = rng.normal(size=(40, 2))
    history = rng.normal(size=(40, 11))
    load = 2 * state[:, 0] - history[:, 2] + rng.normal(scale=0.1, size=40)
    train = np.arange(40) < 30
    first, audit = fit_load_innovation(state, history, load, train)
    changed = load.copy(); changed[~train] += 1000
    second, _ = fit_load_innovation(state, history, changed, train)
    assert np.allclose(first[train], second[train])
    assert audit["uses_validation_outcome"] is False


def test_one_step_builder_never_crosses_recorded_segment() -> None:
    event_time = np.asarray([0.0, 10.0, 20.0, 30.0])
    q_time = np.concatenate([
        np.linspace(1, 9, 4), np.linspace(11, 19, 4), np.linspace(21, 29, 4)
    ])
    design = SimpleNamespace(
        event_time=event_time,
        event_split=np.zeros(4, dtype=np.int8),
        event_session=np.zeros(4, dtype=np.int64),
        event_history=np.zeros((4, 3), dtype=np.float32),
        event_group_ids=np.zeros((4, 1), dtype=np.int64),
        event_group_count=np.ones(4, dtype=np.int64),
        quadrature_time=q_time,
        quadrature_split=np.zeros(12, dtype=np.int8),
        quadrature_session=np.zeros(12, dtype=np.int64),
        quadrature_history=np.zeros((12, 3), dtype=np.float32),
        quadrature_weight_seconds=np.full(12, 2.5),
    )
    result = build_one_step_design(
        design, np.zeros((4, 2), dtype=np.float32),
        np.asarray([0, 0, 1, 1]), np.arange(4, dtype=np.float32),
        np.ones(4, dtype=bool),
    )
    assert result.current_index.tolist() == [0, 2]
    assert result.quadrature_delta_minutes.shape == (2, 4)


def test_state_matched_placebo_uses_train_and_excludes_local_window() -> None:
    n = 300
    exposure = np.arange(n, dtype=np.float32)
    state = np.arange(n, dtype=np.float32)[:, None]
    history = np.zeros((n, 11), dtype=np.float32)
    train = np.arange(n) < 240
    eligible = np.ones(n, dtype=bool)
    placebo, matched, audit = state_matched_placebo(
        exposure, state, history, train, eligible,
        exclusion_events=20, neighbours=8,
    )
    assert matched.all()
    # Exposure equals the donor row index in this construction.
    donor = placebo.astype(int)
    # Every target, not only the TRAIN ones.  A validation row just past the
    # split boundary would otherwise draw a donor whose rolling window overlaps
    # its own by up to N-1 events, i.e. a placebo that is mostly the real
    # exposure, which biases the real-minus-placebo contrast toward zero.
    assert np.all(np.abs(donor - np.arange(n)) >= 20)
    assert np.all(donor[~train] < 240)
    assert audit["validation_donors_from_train_only"] is True
    assert audit["exclusion_events_all_targets"] == 20


class _DummyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(2))

    def matrix(self) -> torch.Tensor:
        return self.mu.new_zeros((2, 2))


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = torch.nn.Module()
        self.state.generator = _DummyGenerator()

    def timing_log_rate(self, history, state):
        return state[:, 0]

    def mark_terms(self, history, state, group_ids, group_count):
        batch = len(state)
        zero = state[:, 0] * 0.0
        step = zero[:, None].expand(batch, 2)
        return SimpleNamespace(
            event_log_prob=zero,
            group_size_log_prob=zero,
            subset_log_prob=zero,
            group_size_step_log_prob=step,
            subset_step_log_prob=step,
            active_step=torch.ones((batch, 2), dtype=torch.bool, device=state.device),
            select_step=torch.tensor([True, False], device=state.device).expand(batch, 2),
        )


def test_synthetic_signed_edge_is_recovered() -> None:
    rng = np.random.default_rng(11)
    n = 1200
    exposure = rng.normal(size=n).astype(np.float32)
    truth = 0.7
    interval_seconds = rng.exponential(1.0 / np.exp(truth * exposure)).astype(np.float32)
    design = OneStepDesign(
        current_state=np.zeros((n, 2), dtype=np.float32),
        current_index=np.arange(n),
        next_history=np.zeros((n, 3), dtype=np.float32),
        next_group_ids=np.zeros((n, 1), dtype=np.int64),
        next_group_count=np.ones(n, dtype=np.int64),
        delta_minutes=interval_seconds / 60.0,
        quadrature_delta_minutes=np.tile(interval_seconds[:, None] / 120.0, (1, 4)),
        quadrature_history=np.zeros((n, 4, 3), dtype=np.float32),
        quadrature_weight_seconds=np.tile(interval_seconds[:, None] / 4.0, (1, 4)),
        exposure=exposure,
        split=np.r_[np.zeros(800, dtype=np.int8), np.ones(400, dtype=np.int8)],
    )
    design.validate()
    model = _DummyModel()
    null = SignedExposureEdge(2)
    null_metrics = evaluate_edge(model, null, design, split="validation", device="cpu")
    edge, audit = fit_edge(
        model, design, device="cpu", seed=3, epochs=30,
        learning_rate=0.03, batch_size=256,
    )
    fitted = evaluate_edge(model, edge, design, split="validation", device="cpu")
    assert edge.vector[0].item() > 0.3
    assert fitted.joint_nll_per_event < null_metrics.joint_nll_per_event
    assert audit["selected_epoch"] > 0
