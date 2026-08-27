from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.h3_long import (
    affine_estimability_audit,
    classify_affine_estimability,
    exact_boxcar_event_exposure,
    exact_previous_block_placebo,
    fit_affine_edge,
    independent_endpoint_rows,
    standardise_exposure_on_train,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import OneStepDesign
from scripts.topic5_continuous_marked_state_r1.audit_r1_5_h3_long_support import (
    greedy_disjoint_count,
)
from scripts.topic5_continuous_marked_state_r1 import run_h3_long_queue


def test_exact_boxcar_resets_and_contains_exactly_n_events() -> None:
    innovation = np.arange(1, 9, dtype=np.float32)
    segment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    exposure, eligible, audit = exact_boxcar_event_exposure(
        innovation, segment, scale_events=3
    )
    assert np.allclose(exposure, [0, 0, 6, 9, 0, 0, 18, 21])
    assert eligible.tolist() == [False, False, True, True, False, False, True, True]
    assert audit["exponential_tail"] is False


def test_previous_block_placebo_is_causal_and_exactly_disjoint() -> None:
    innovation = np.ones(12, dtype=np.float32)
    segment = np.zeros(12, dtype=np.int64)
    exposure, _, _ = exact_boxcar_event_exposure(
        innovation, segment, scale_events=3
    )
    placebo, eligible, audit = exact_previous_block_placebo(
        exposure, segment, scale_events=3
    )
    assert eligible.tolist() == [False] * 5 + [True] * 7
    assert np.allclose(placebo[eligible], 3.0)
    assert audit["real_and_placebo_windows_exactly_disjoint"] is True
    assert audit["strictly_past"] is True


def test_standardisation_uses_train_only() -> None:
    exposure = np.asarray([1, 2, 3, 100, 200], dtype=np.float32)
    train = np.asarray([True, True, True, False, False])
    eligible = np.ones(5, dtype=bool)
    first, audit = standardise_exposure_on_train(exposure, train, eligible)
    changed = exposure.copy(); changed[~train] += 10000
    second, _ = standardise_exposure_on_train(changed, train, eligible)
    assert np.allclose(first[train], second[train])
    assert audit["validation_statistics_used"] is False


def test_greedy_independent_blocks_do_not_count_sliding_rows() -> None:
    assert greedy_disjoint_count(np.arange(1000, 3000), 1000) == 2


def test_full_control_independent_units_use_two_n_width() -> None:
    current = np.arange(1999, 8000, dtype=np.int64)
    segment = np.zeros(8000, dtype=np.int64)
    rows = independent_endpoint_rows(current, segment, width_events=2000)
    assert current[rows].tolist() == [1999, 3999, 5999, 7999]


def test_raw_operator_scaling_is_removed_by_train_standardisation() -> None:
    exposure = np.linspace(-3, 7, 100, dtype=np.float32)
    train = np.arange(100) < 70
    eligible = np.ones(100, dtype=bool)
    first, _ = standardise_exposure_on_train(exposure, train, eligible)
    second, _ = standardise_exposure_on_train(1000 * exposure, train, eligible)
    assert np.allclose(first, second, atol=1e-6)


def test_estimability_classes_keep_zero_selection_and_rank_failure_separate() -> None:
    base = {
        "gradient_finite": True, "affine_design_rank": 2,
        "expected_affine_rank": 2, "exposure_sd": [1.0],
        "matrix_gradient_at_zero_norm": 1.0,
    }
    assert classify_affine_estimability(
        base, {"edge_left_zero_initialisation": False}
    ) == "ZERO_SELECTED"
    rank = {**base, "affine_design_rank": 1}
    assert classify_affine_estimability(
        rank, {"edge_left_zero_initialisation": True}
    ) == "RANK_DEGENERATE"
    zero_gradient = {**base, "matrix_gradient_at_zero_norm": 0.0}
    assert classify_affine_estimability(
        zero_gradient, {"edge_left_zero_initialisation": False}
    ) == "ZERO_GRADIENT"


def test_resume_rejects_a_cell_from_the_wrong_package(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_h3_long_queue, "cell_package_fingerprint",
        lambda *args, **kwargs: ("expected", {"bundle": "current"}),
    )
    path = tmp_path / "result.json"
    path.write_text(__import__("json").dumps({
        "status": "COMPLETE",
        "revision": run_h3_long_queue.H3_LONG_REVISION,
        "sealed_opened": False,
        "formal_test_partition_opened": False,
        "subject": "epilepsiae_384", "seed": 0, "source": "load",
        "scale_events": 1000, "support_role": "full_control",
        "package_fingerprint": "stale",
        "package_components": {"bundle": "old"},
    }))
    cell = {"scale_events": 1000, "role": "full_control"}
    assert not run_h3_long_queue.result_complete(
        path, subject="epilepsiae_384", seed=0, source="load",
        cell=cell, root=tmp_path,
    )


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
            select_step=torch.tensor(
                [True, False], device=state.device
            ).expand(n, 2),
        )


def _design(exposure: np.ndarray, log_rate: np.ndarray) -> OneStepDesign:
    n = len(exposure)
    interval = np.exp(-np.asarray(log_rate, dtype=np.float64)).astype(np.float32)
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
        exposure=np.asarray(exposure, dtype=np.float32),
        split=np.r_[np.zeros(800, dtype=np.int8), np.ones(n - 800, dtype=np.int8)],
    )


def test_affine_edge_recovers_both_signs_with_full_rank_design() -> None:
    model = _Model()
    exposure = np.tile(np.asarray([-1.0, 1.0]), 600)
    for sign in (1.0, -1.0):
        design = _design(exposure, .6 * sign * exposure + .25)
        audit = affine_estimability_audit(
            model, design, device="cpu", batch_size=2048
        )
        assert audit["affine_design_rank"] == audit["expected_affine_rank"]
        edge, fit = fit_affine_edge(
            model, design, device="cpu", seed=4, epochs=20,
            learning_rate=.03, batch_size=2048,
        )
        assert np.sign(edge.matrix[0, 0].item()) == np.sign(sign)
        assert fit["edge_left_zero_initialisation"] is True


def test_constant_offset_is_absorbed_without_exposure_edge() -> None:
    model = _Model()
    exposure = np.tile(np.asarray([-1.0, 1.0]), 600)
    design = _design(exposure, np.full(len(exposure), .7))
    edge, fit = fit_affine_edge(
        model, design, device="cpu", seed=2, epochs=20,
        learning_rate=.03, batch_size=2048,
    )
    assert fit["intercept_left_zero_initialisation"] is True
    assert abs(edge.intercept[0].item()) > .1
    assert abs(edge.matrix[0, 0].item()) < 1e-3
