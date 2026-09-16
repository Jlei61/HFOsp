from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
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
    classify_one_shot_persistence,
    standardise_exposure,
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
    assert audit["unique_donors"] > 1
    assert audit["effective_donors"] > 1
    assert 0 < audit["maximum_donor_reuse_fraction"] < 1
    assert audit["match_distance_q95"] >= audit["median_match_distance"]


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


def test_persistence_label_requires_estimable_nonzero_real_edge() -> None:
    favourable = {
        "mark_nll_per_event": -0.2,
        "state_mse_to_filtered_target": -0.1,
    }
    moved = {"mean_state_displacement_from_no_edge": 0.5}
    zero = {"mean_state_displacement_from_no_edge": 0.0}
    assert classify_one_shot_persistence(
        favourable, moved, real_edge_estimable=True
    )["state_and_mark_persist"] is True
    assert classify_one_shot_persistence(
        favourable, moved, real_edge_estimable=False
    )["state_and_mark_persist"] is False
    assert classify_one_shot_persistence(
        favourable, zero, real_edge_estimable=True
    )["state_and_mark_persist"] is False


def test_aggregator_recomputes_and_rejects_stale_zero_edge_flag() -> None:
    payload = {
        "real_edge_estimable": False,
        "comparisons": {
            "H5": {"real_minus_state_matched_placebo": {
                "mark_nll_per_event": -0.2,
                "state_mse_to_filtered_target": -0.1,
            }},
        },
        "validation": {"horizons": {"H5": {"real_cumulative": {
            "mean_state_displacement_from_no_edge": 0.0,
        }}}},
    }
    assert aggregate_t2_r2.corrected_persistence(
        payload, "H5"
    )["state_and_mark_persist"] is False


def test_aggregator_excludes_structural_zero_from_effect_summary() -> None:
    fitted = {"analysis_status": "ESTIMATED", "real_edge_estimable": True}
    structural_zero = {
        "analysis_status": "ESTIMATED", "real_edge_estimable": False,
    }
    support_limited = {
        "analysis_status": "NOT_ESTIMABLE", "real_edge_estimable": False,
    }
    assert aggregate_t2_r2.edge_estimable_payloads([
        fitted, structural_zero, support_limited,
    ]) == [fitted]


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


def _scaled_design(design: OneStepDesign, factor: float) -> OneStepDesign:
    return OneStepDesign(**{
        **{k: getattr(design, k) for k in design.__dataclass_fields__},
        "exposure": (design.exposure * factor).astype(np.float32),
    })


def test_standardise_exposure_puts_every_arm_on_one_train_scale() -> None:
    rng = np.random.default_rng(5)
    n = 400
    train = np.r_[np.ones(300, dtype=bool), np.zeros(100, dtype=bool)]
    eligible = np.ones(n, dtype=bool)
    cumulative = rng.normal(scale=18.0, size=n)
    current = rng.normal(scale=1.0, size=n)
    scaled_cumulative, audit_c = standardise_exposure(
        cumulative, train, eligible, label="real_cumulative"
    )
    scaled_current, audit_e = standardise_exposure(
        current, train, eligible, label="current_event_only"
    )
    assert audit_c["train_sd_after"][0] == pytest.approx(1.0, abs=1e-6)
    assert audit_e["train_sd_after"][0] == pytest.approx(1.0, abs=1e-6)
    assert audit_c["train_sd_before"][0] / audit_e["train_sd_before"][0] > 10
    # A constant column is left alone rather than divided by zero.
    constant, audit_k = standardise_exposure(
        np.ones((n, 1)), train, eligible, label="fitted_intercept_diagnostic"
    )
    assert audit_k["constant_columns_left_unscaled"] == [0]
    assert np.allclose(constant, 1.0)
    # B @ x is invariant, so this is a reparameterisation of the same model.
    edge = ExposureEdge(2, 1)
    with torch.no_grad():
        edge.matrix.fill_(0.3)
    state = torch.zeros(n, 2)
    raw = edge(state, torch.as_tensor(cumulative, dtype=torch.float32))
    rescaled = ExposureEdge(2, 1)
    with torch.no_grad():
        rescaled.matrix.fill_(0.3 * float(audit_c["train_sd_before"][0]))
    assert torch.allclose(
        raw, rescaled(state, torch.as_tensor(scaled_cumulative)), atol=1e-3
    )


def test_unstandardised_exposure_scale_changes_the_early_stopped_fit() -> None:
    # The regression that motivates standardisation.  AdamW is largely
    # scale-free, so the *achieved* fit barely moves: the scale-adjusted edge
    # norms agree to about one percent here.  What does move is the epoch the
    # chronological inner-TRAIN search stops at, and that is the quantity the
    # arms are supposed to share.  Two arms whose raw exposures differ 12-21x
    # are therefore stopped at different points of their own trajectories.
    model = _Model()
    base = _synthetic_design(1.0)
    _, small = fit_r2_edge(
        model, _scaled_design(base, 1.0), device="cpu", seed=3, epochs=30,
        learning_rate=.03, batch_size=256,
    )
    _, large = fit_r2_edge(
        model, _scaled_design(base, 18.0), device="cpu", seed=3, epochs=30,
        learning_rate=.03, batch_size=256,
    )
    assert large["selected_epoch"] != small["selected_epoch"]
    assert large["edge_norm"] * 18.0 == pytest.approx(
        small["edge_norm"], rel=.05
    )
    # ...and standardising both back onto one scale removes the difference.
    train = base.split == 0
    eligible = np.ones(len(base.split), dtype=bool)
    fits = []
    for factor in (1.0, 18.0):
        exposure, _ = standardise_exposure(
            base.exposure * factor, train, eligible, label=f"x{factor}"
        )
        _, fit = fit_r2_edge(
            model, _scaled_design(base, 1.0).__class__(**{
                **{k: getattr(base, k) for k in base.__dataclass_fields__},
                "exposure": exposure,
            }),
            device="cpu", seed=3, epochs=30, learning_rate=.03, batch_size=256,
        )
        fits.append(fit)
    assert fits[0]["selected_epoch"] == fits[1]["selected_epoch"]
    assert fits[0]["edge_norm"] == pytest.approx(fits[1]["edge_norm"], rel=1e-6)


def _t2_payload(*, status="ESTIMATED", left_zero=True, current_beats=False,
                real=0.0):
    return {
        "analysis_status": status,
        "real_edge_estimable": bool(left_zero),
        "real_edge_status": "FITTED" if left_zero else "ZERO_EDGE_SELECTED",
        "primary_next_event_increment": False,
        "fits": {"real_cumulative": {
            "edge_left_zero_initialisation": bool(left_zero)
        }},
        "validation": {
            "next_event": {
                "no_edge": {"joint_nll_per_event": 10.0},
                "current_event_only": {
                    "joint_nll_per_event": 9.9 if current_beats else 10.1
                },
            },
            "horizons": {},
        },
        "comparisons": {},
    }


def test_zero_edge_on_an_identifiable_design_is_a_negative_not_a_blank(
    tmp_path, monkeypatch,
) -> None:
    # The strongest-supported patient in the shipped run selected the zero edge
    # in 3/3 seeds while the current-event arm on the same rows beat no-edge.
    # That is an answer to H3a, so the row must carry it rather than print n/a
    # across the board.
    payloads = [
        _t2_payload(left_zero=False, current_beats=True) for _ in range(3)
    ]
    zero_edge = [
        value for value in payloads
        if value["real_edge_status"] == "ZERO_EDGE_SELECTED"
    ]
    assert len(zero_edge) == 3
    assert aggregate_t2_r2.edge_estimable_payloads(payloads) == []
    sibling = sum(
        value["validation"]["next_event"]["current_event_only"][
            "joint_nll_per_event"
        ] < value["validation"]["next_event"]["no_edge"]["joint_nll_per_event"]
        for value in payloads
    )
    assert sibling == 3


def test_support_ineligible_is_the_only_missing_measurement() -> None:
    # A seed with no usable one-step pairs cannot answer anything; a seed whose
    # search returned the zero edge answered "no".  They must not share a label.
    no_support = _t2_payload(status="NOT_ESTIMABLE")
    zero_edge = _t2_payload(left_zero=False)
    assert no_support["analysis_status"] == "NOT_ESTIMABLE"
    assert zero_edge["analysis_status"] == "ESTIMATED"
    assert zero_edge["real_edge_status"] == "ZERO_EDGE_SELECTED"
    assert aggregate_t2_r2.edge_estimable_payloads([no_support, zero_edge]) == []
