"""Task 7: training card and adequacy rule (design §8, clauses C1-C5)."""

from __future__ import annotations

import pytest

from src.topic5_group_event_state.v033_training_lab.card import (
    ADEQUACY_RULE,
    CARD_FIELDS,
    adequacy,
    assert_card_has_no_dev_test,
    build_training_card,
    seed_dispersion,
)


def _recipe_result(**over):
    base = {
        "status": "complete", "subject": "toy", "seed": 1, "arm": "learned", "config": {"arch": {"write_width": 4}},
        "config_hash": "cfg", "split_hash": "split", "input_hash": "input", "h_source": "provisional_local",
        "selected_step": 40, "selected_in_warmup": False, "selected_at_budget_edge": False, "n_steps_run": 60,
        "stopped_reason": "patience", "first_active_step": {"encoder_weights": 1, "adapter_w": 1},
        "all_groups_active_before_selection": True, "clipping_fraction": 0.1,
        "plateau": {"reached": True, "since_step": 40, "stale_validations": 3, "lr_reductions": 0},
        "best_validation": {"step": 40, "inner_val_nll": 9.0, "inner_val_nll_h": 9.3, "gain_h_minus_model": 0.3},
        "history": [
            {"step": 10, "train_nll": 9.4, "inner_val_nll": 9.2, "grad_norm_by_group": {"adapter_w": 1.0},
             "update_norm_by_group": {"adapter_w": 0.01}, "lr_by_group": {"adapter_w": 1e-3}},
            {"step": 40, "train_nll": 8.8, "inner_val_nll": 9.0, "grad_norm_by_group": {"adapter_w": 0.5},
             "update_norm_by_group": {"adapter_w": 0.005}, "lr_by_group": {"adapter_w": 1e-3}},
        ],
        "curves_path": "/tmp/curves.parquet", "checkpoint": "/tmp/checkpoint.pt", "checkpoint_sha256": "abc",
        "parameter_sha256": "def", "elapsed_seconds": 3.0, "source_commit": "233f3ad1", "optimizer": "adamw",
        "schedule": "constant", "warmup_steps": 0, "lr_by_group": {"adapter_w": 1e-3},
    }
    base.update(over)
    return base


def _t0(**over):
    base = {"tiny_slice_overfit": {"pass": True, "gap_closed": 0.8, "threshold": 0.5},
            "state_write_jacobian": {"pass": True}, "optimizer_membership": {"pass": True},
            "amp_small_gradient": {"status": "skipped", "pass": None}, "state_output_modulation": {"pass": True},
            "oracle_head_fit": {"status": "skipped"}, "gradient_path_ok": True}
    base.update(over)
    return base


def _diag(**over):
    base = {"blocked_inner_val_gain": {"mean": 0.3, "ci_low": 0.1, "ci_high": 0.5, "n_anchors": 26, "n_blocks": 5},
            "shift_null": {"delta_shifted_minus_correct": {"mean": 0.2, "ci_low": 0.05}, "n_valid_donors": 20},
            "random_reservoir_delta": {"learned_minus_random": {"mean": -0.1, "ci_low": -0.2, "ci_high": 0.0}},
            "synthetic_recovery": {"pass": True, "source": "v032_residual_positive_proxy", "beta": 0.7},
            "state_variance_rank": {"participation_ratio": 4.2, "readout_rank": 3},
            "state_output_modulation": {"pass": True, "inner_val": {"modulation_rms": 0.1}}}
    base.update(over)
    return base


REQUEST = {"request_id": "req1", "scientific_target": {"family": "S_N", "objective": "count_profile"},
           "input_view": {"kind": "toy"}, "state_family": "fixed_leaky", "split_hash": "split", "input_hash": "input",
           "code_commit": "233f3ad1", "requested_by": "agent_c"}


def test_c1_all_six_conditions_give_training_adequate_and_any_failure_gives_diagnostic_with_reason():
    seeds = [_recipe_result(seed=s, best_validation={"step": 40, "inner_val_nll": 9.0 + 0.01 * s,
                                                     "inner_val_nll_h": 9.3, "gain_h_minus_model": 0.3 - 0.01 * s})
             for s in range(3)]
    card = build_training_card(request=REQUEST, recipe_result=seeds[0], seed_results=seeds, t0=_t0(), diagnostics=_diag())
    assert card["evidence_label"] == "TRAINING-ADEQUATE" and card["adequacy_rule"] == ADEQUACY_RULE
    assert card["adequacy_reasons"] == []
    failures = {
        "tiny_overfit": dict(t0=_t0(tiny_slice_overfit={"pass": False, "gap_closed": 0.1, "threshold": 0.5})),
        "synthetic_recovery": dict(diagnostics=_diag(synthetic_recovery={"pass": False, "source": "x", "beta": 0.7})),
        "blocked_inner_val_gain": dict(diagnostics=_diag(blocked_inner_val_gain={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1})),
        "selected_in_warmup": dict(recipe_result=_recipe_result(selected_in_warmup=True)),
        "selected_at_budget_edge": dict(recipe_result=_recipe_result(selected_at_budget_edge=True)),
        "all_groups_active_before_selection": dict(recipe_result=_recipe_result(all_groups_active_before_selection=False)),
    }
    for name, over in failures.items():
        kwargs = dict(request=REQUEST, recipe_result=seeds[0], seed_results=seeds, t0=_t0(), diagnostics=_diag())
        kwargs.update(over)
        bad = build_training_card(**kwargs)
        assert bad["evidence_label"] == "DIAGNOSTIC", name
        assert any(name in reason for reason in bad["adequacy_reasons"]), (name, bad["adequacy_reasons"])
    label, detail = adequacy(card)
    assert label == "TRAINING-ADEQUATE" and set(detail["conditions"]) == {
        "tiny_overfit", "synthetic_recovery", "blocked_inner_val_gain", "selected_in_warmup",
        "selected_at_budget_edge", "all_groups_active_before_selection"}


def test_c2_seed_dispersion_needs_at_least_two_seeds():
    seeds = [_recipe_result(seed=s, selected_step=40 + 10 * s,
                            best_validation={"step": 40, "inner_val_nll": 9.0 + 0.1 * s, "inner_val_nll_h": 9.3,
                                             "gain_h_minus_model": 0.3 - 0.1 * s}) for s in range(3)]
    d = seed_dispersion(seeds)
    assert d["n_seeds"] == 3 and d["insufficient_seeds"] is False
    assert d["inner_val_nll"]["std"] == pytest.approx(0.1, abs=1e-6)          # sample std (ddof=1)
    assert d["selected_step"]["min"] == 40 and d["selected_step"]["max"] == 60
    single = seed_dispersion(seeds[:1])
    assert single["n_seeds"] == 1 and single["insufficient_seeds"] is True


def test_c3_card_carries_every_required_field_and_is_not_canonical_by_default():
    card = build_training_card(request=REQUEST, recipe_result=_recipe_result(), seed_results=[_recipe_result()],
                               t0=_t0(), diagnostics=_diag())
    for name in CARD_FIELDS:
        assert name in card, name
    assert card["selection_metric_is_canonical"] is False and card["evaluator_hash"] is None
    assert card["sealed_partition_opened"] is False and card["development_evaluation_read"] is False
    assert card["training_adequacy_is_not_a_scientific_result"] is True


def test_c4_card_with_a_development_evaluation_key_is_refused():
    card = build_training_card(request=REQUEST, recipe_result=_recipe_result(), seed_results=[_recipe_result()],
                               t0=_t0(), diagnostics=_diag())
    assert_card_has_no_dev_test(card)
    card["diagnostics"]["dev_test_gain"] = 0.1
    with pytest.raises(ValueError):
        assert_card_has_no_dev_test(card)
    with pytest.raises(ValueError):
        build_training_card(request=REQUEST, recipe_result=_recipe_result(), seed_results=[_recipe_result()],
                            t0=_t0(), diagnostics=_diag(dev_test_gain=0.1))
