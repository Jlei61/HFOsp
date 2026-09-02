from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from src.topic5_group_event_state.v033_evaluator import dgp as D
from src.topic5_group_event_state.v033_evaluator import grammar_recovery as G
from tests.test_group_event_state_v033_toyutil import toy_scaffold


def test_recipe_grid_is_small_predeclared_and_marks_only_not_selection_eligible():
    assert tuple(G.RECIPES) == (
        "t0_lr3e3_constant", "t0_lr1e2_cosine", "t1_nested_h16_w2_full",
        "t1_nested_h64_w4_full", "t1_nested_h16_w2_marks_only",
    )
    assert sum(r.batch == "T0" for r in G.RECIPES.values()) == 2
    assert sum(r.batch == "T1" for r in G.RECIPES.values()) == 3
    assert G.recipe("t1_nested_h16_w2_marks_only").eligible_for_selection is False


def test_nested_model_freezes_calibrated_h_intercept():
    cfg = G.recipe("t1_nested_h16_w2_full")
    h = np.linspace(-1, 1, 6)
    model = G.GrammarLevel2Model(8, 6, cfg, h)
    assert model.a.requires_grad is False
    assert np.allclose(model.a.detach().numpy(), h)
    old = G.GrammarLevel2Model(8, 6, G.recipe("t0_lr3e3_constant"), h)
    assert old.a.requires_grad is True


def test_h_only_calibration_selects_inner_checkpoint_instead_of_last_iterate():
    rng = np.random.default_rng(8)
    participation = torch.from_numpy(rng.uniform(size=(80, 6)) < 0.4)
    owner = torch.arange(80, dtype=torch.long)
    event = torch.arange(80, dtype=torch.long)
    weight = torch.full((80,), 1 / 80, dtype=torch.float64)
    train = (owner[:60], event[:60], torch.full((60,), 1 / 60, dtype=torch.float64))
    select = (owner[:20], event[60:], torch.full((20,), 1 / 20, dtype=torch.float64))
    got = G._fit_h_only_selected(participation, train, select, np.zeros(6),
                                  max_steps=40, validate_every=10, patience_checks=4)
    assert got["selected_step"] in (10, 20, 30, 40)
    assert got["selected_inner_nll"] == min(x["inner_nll"] for x in got["curve"])


def test_toy_recovery_card_never_claims_human_targets_and_uses_inner_selection():
    sc = toy_scaffold(seed=4, rate_per_second=0.03, n_contacts=6)
    data = D.generate(sc, "D3", beta_count=0.7, beta_grammar=2.5,
                      generator_seed=21, noise_seed=22)
    cfg = replace(G.recipe("t1_nested_h16_w2_marks_only"), max_steps=20, min_steps=10,
                  validate_every=5, patience_checks=4)
    card = G.run_recovery(sc, data, cfg=cfg, horizon=1800.0, seed=0, device=torch.device("cpu"))
    assert card["human_targets_used"] is False
    assert card["development_human_targets_read"] is False
    assert card["seizure_outcomes_used"] is False and card["sealed_partition_opened"] is False
    assert card["selection_phase"] == "inner_val" and card["score_target"] == "synthetic_contact_subsets_only"
    assert card["head_contract"] == "frozen_H_intercept_plus_state_residual"
    assert card["first_encoder_gradient_step"] is not None


def test_recipe_selection_excludes_marks_only_and_prefers_simple_within_tolerance():
    cards = [
        {"kind": "D3", "recipe_selection_eligible": True,
         "recipe": {"name": "t1_nested_h16_w2_full"}, "selected_inner_nll": 1.001},
        {"kind": "D3", "recipe_selection_eligible": True,
         "recipe": {"name": "t1_nested_h64_w4_full"}, "selected_inner_nll": 1.000},
        {"kind": "D3", "recipe_selection_eligible": False,
         "recipe": {"name": "t1_nested_h16_w2_marks_only"}, "selected_inner_nll": 0.1},
    ]
    got = G.select_full_input_recipe(cards, tolerance=0.002)
    assert got["selected_recipe"] == "t1_nested_h16_w2_full"
    assert all(x["recipe"] != "t1_nested_h16_w2_marks_only" for x in got["eligible_candidates"])
