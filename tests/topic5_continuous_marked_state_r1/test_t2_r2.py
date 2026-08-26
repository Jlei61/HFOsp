from types import SimpleNamespace
import json
import sys

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.t2_r2 import (
    ExposureEdge,
    crossfit_expected_mark,
    edge_estimability_audit,
    exponential_event_exposure,
    build_horizon_mark_design,
    evaluate_horizon_mark,
    fit_load_innovation_crossfit,
    fit_r2_edge,
    state_matched_nonoverlap_placebo,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import OneStepDesign
from scripts.topic5_continuous_marked_state_r1 import (
    aggregate_t2_r2,
    run_t2_r2_human,
)


def test_crossfit_innovation_never_uses_validation_outcome() -> None:
    rng = np.random.default_rng(7)
    n = 120
    state = rng.normal(size=(n, 3))
    history = rng.normal(size=(n, 11))
    observation = rng.normal(size=(n, 4))
    load = 2 * state[:, 0] - observation[:, 1] + rng.normal(scale=.1, size=n)
    train = np.arange(n) < 90
    first, audit = fit_load_innovation_crossfit(
        state, history, observation, load, train, folds=5
    )
    changed = load.copy()
    changed[~train] += 10_000
    second, _ = fit_load_innovation_crossfit(
        state, history, observation, changed, train, folds=5
    )
    assert np.allclose(first[train], second[train])
    assert audit["train_predictions_are_out_of_fold"] is True
    assert audit["uses_validation_outcome"] is False


def test_crossfit_train_predictions_are_not_in_sample_predictions() -> None:
    x = np.eye(30, dtype=np.float64)
    y = np.arange(30, dtype=np.float64)
    train = np.arange(30) < 25
    prediction, audit = crossfit_expected_mark(x, y, train, folds=5, ridge=1e-6)
    # Each held-out identity column is absent from its fitting rows, so a
    # true out-of-fold prediction cannot interpolate its own outcome.
    assert not np.allclose(prediction[train], y[train])
    assert audit["folds"] == 5


def test_exponential_exposure_resets_and_uses_frozen_n() -> None:
    innovation = np.ones(8, dtype=np.float32)
    segment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    exposure, eligible, audit = exponential_event_exposure(
        innovation, segment, scale_events=3, burn_in_events=3
    )
    alpha = np.exp(-1 / 3)
    assert np.isclose(exposure[2], 1 + alpha + alpha**2)
    assert np.isclose(exposure[4], 1.0)
    assert eligible.tolist() == [False, False, True, True, False, False, True, True]
    assert audit["resets_at_recorded_segment"] is True


def test_placebo_donor_is_train_only_and_effective_histories_do_not_overlap() -> None:
    n = 1200
    exposure = np.arange(n, dtype=np.float32)
    rng = np.random.default_rng(4)
    state = rng.normal(size=(n, 2))
    history = rng.normal(size=(n, 11))
    observation = rng.normal(size=(n, 3))
    train = np.arange(n) < 900
    eligible = np.ones(n, dtype=bool)
    segment = np.repeat(np.arange(6), 200)
    placebo, matched, audit = state_matched_nonoverlap_placebo(
        exposure, state, history, observation, train, eligible, segment,
        scale_events=20, history_multiples=5, neighbours=32,
    )
    assert matched.all()
    donor = placebo.astype(int)
    assert np.all(donor < 900)
    same = segment[donor] == segment[np.arange(n)]
    assert np.all((~same) | (np.abs(donor - np.arange(n)) >= 100))
    assert audit["all_matched"] is True


class _Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(2))

    def matrix(self) -> torch.Tensor:
        return self.mu.new_zeros((2, 2))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = torch.nn.Module()
        self.state.generator = _Generator()

    def timing_log_rate(self, history, state):
        return state[:, 0]

    def mark_terms(self, history, state, group_ids, group_count):
        n = len(state)
        zero = state[:, 0] * 0
        step = zero[:, None].expand(n, 2)
        return SimpleNamespace(
            event_log_prob=zero,
            group_size_log_prob=zero,
            subset_log_prob=zero,
            group_size_step_log_prob=step,
            subset_step_log_prob=step,
            active_step=torch.ones((n, 2), dtype=torch.bool, device=state.device),
            select_step=torch.tensor([True, False], device=state.device).expand(n, 2),
        )


def _synthetic_design(sign: float) -> OneStepDesign:
    rng = np.random.default_rng(11)
    n = 1200
    exposure = rng.normal(size=n).astype(np.float32)
    interval = rng.exponential(1 / np.exp(sign * .7 * exposure)).astype(np.float32)
    return OneStepDesign(
        current_state=np.zeros((n, 2), dtype=np.float32),
        current_index=np.arange(n),
        next_history=np.zeros((n, 3), dtype=np.float32),
        next_group_ids=np.zeros((n, 1), dtype=np.int64),
        next_group_count=np.ones(n, dtype=np.int64),
        delta_minutes=interval / 60,
        quadrature_delta_minutes=np.tile(interval[:, None] / 120, (1, 4)),
        quadrature_history=np.zeros((n, 4, 3), dtype=np.float32),
        quadrature_weight_seconds=np.tile(interval[:, None] / 4, (1, 4)),
        exposure=exposure,
        split=np.r_[np.zeros(800, dtype=np.int8), np.ones(400, dtype=np.int8)],
    )


def test_r2_edge_has_nonzero_gradient_and_recovers_both_signs() -> None:
    model = _Model()
    for sign in (1.0, -1.0):
        design = _synthetic_design(sign)
        audit = edge_estimability_audit(model, design, device="cpu", batch_size=256)
        assert audit["gradient_at_zero_norm"] > 0
        assert audit["exposure_rank"] == 1
        edge, fit = fit_r2_edge(
            model, design, device="cpu", seed=3, epochs=30,
            learning_rate=.03, batch_size=256,
        )
        assert np.sign(edge.matrix[0, 0].item()) == np.sign(sign)
        assert fit["edge_left_zero_initialisation"] is True


def test_zero_edge_is_exact_no_edge_state() -> None:
    edge = ExposureEdge(3, 1)
    state = torch.randn(5, 3)
    exposure = torch.randn(5)
    assert torch.equal(edge(state, exposure), state)


def test_horizon_scores_future_state_accuracy_not_only_nonzero_displacement() -> None:
    n = 20
    full = SimpleNamespace(
        event_time=np.arange(n, dtype=np.float64) * 10,
        event_split=np.zeros(n, dtype=np.int8),
        event_history=np.zeros((n, 3), dtype=np.float32),
        event_group_ids=np.zeros((n, 1), dtype=np.int64),
        event_group_count=np.ones(n, dtype=np.int64),
    )
    state = np.zeros((n, 2), dtype=np.float32)
    state[5:, 0] = 1.0
    horizon = build_horizon_mark_design(
        full, state, np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.float32), np.arange(n) < 5, 5,
    )
    model = _Model()
    zero = ExposureEdge(2, 1)
    shifted = ExposureEdge(2, 1)
    with torch.no_grad():
        shifted.matrix[0, 0] = 1.0
    base = evaluate_horizon_mark(model, zero, horizon, split="train", device="cpu")
    fitted = evaluate_horizon_mark(
        model, shifted, horizon, split="train", device="cpu"
    )
    assert fitted.mean_state_displacement_from_no_edge > 0
    assert fitted.state_mse_to_filtered_target < base.state_mse_to_filtered_target


def test_support_limited_seed_is_persisted_and_aggregated_without_blocking(
    tmp_path, monkeypatch,
) -> None:
    r1_root = tmp_path / "r1_4"
    t2_root = tmp_path / "t2_r2"
    (r1_root / "reports").mkdir(parents=True)
    (r1_root / "reports/r1_4_summary.json").write_text(json.dumps({
        "revision": aggregate_t2_r2.R1_4_REVISION,
        "sealed_opened": False,
        "by_subject": {
            "epilepsiae_620": {"stable_explicit_t1_for_t2": True},
        },
    }))
    context = SimpleNamespace(audit={
        "r1_4_experiment_label": aggregate_t2_r2.R1_4_REVISION,
    })
    for source in aggregate_t2_r2.SOURCES:
        for seed in aggregate_t2_r2.SEEDS:
            args = SimpleNamespace(
                subject="epilepsiae_620", source=source, seed=seed,
            )
            output = (
                t2_root / "human/epilepsiae_620"
                / f"{source}_seed_{seed}_n_100"
            )
            run_t2_r2_human.persist_not_estimable(
                args, context, output,
                "insufficient N=100 support (500 TRAIN, 42 validation)",
                n_train=500, n_validation=42,
            )
    monkeypatch.setattr(sys, "argv", [
        "aggregate_t2_r2.py", "--r1-4-root", str(r1_root),
        "--root", str(t2_root),
    ])
    aggregate_t2_r2.main()
    summary = json.loads((t2_root / "reports/t2_r2_summary.json").read_text())
    assert len(summary["patient_source"]) == 2
    assert all(row["support_ineligible_seeds"] == 3
               for row in summary["patient_source"])
    assert all(row["estimable_seeds"] == 0 for row in summary["patient_source"])
    assert summary["scale_expansion_candidates"] == []


def test_only_known_support_failures_are_downgraded() -> None:
    assert run_t2_r2_human.support_limited(
        ValueError("state-matched placebo has too few TRAIN donors")
    )
    assert not run_t2_r2_human.support_limited(
        ValueError("unexpected tensor shape")
    )
