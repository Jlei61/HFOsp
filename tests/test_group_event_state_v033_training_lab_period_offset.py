"""Period-offset control for the multi-seed card diagnostics (2026-09-03 audit).

A same-segment block-circular shift cannot reject the explanation "the state
only carries a constant period-level offset of the baseline": the shifted
state has the same mean.  The card therefore reports, next to the shift null,
the increment that remains after the state is replaced by its inner-val mean.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v033_training_lab.card import CARD_FIELDS, build_training_card
from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.diagnostics import (
    merge_seed_anchor_diagnostics,
    multi_seed_card_diagnostics,
)
from src.topic5_group_event_state.v033_training_lab.objective import ResidualCountTrainable
from src.topic5_group_event_state.v033_training_lab.trainer import RecipeConfig, train_recipe
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")
FAST = dict(max_steps=40, min_steps=20, validate_every=10, patience=3)


def test_merge_reports_period_offset_control_and_its_identity():
    rng = np.random.default_rng(0)
    n, k = 40, 3
    h = rng.normal(5.0, 0.5, n)
    learned = np.stack([h - 0.3 + rng.normal(0, 0.05, n) for _ in range(k)])
    period = np.stack([h - 0.25 + rng.normal(0, 0.05, n) for _ in range(k)])
    shifted = learned + 0.02
    random = learned + 0.1
    seg = np.repeat([0, 1], n // 2)
    ok = np.ones(n, dtype=bool)
    out = merge_seed_anchor_diagnostics(h_nll=h, learned_nll=learned, shifted_nll=shifted, random_nll=random,
                                        shift_valid=ok, segments=seg, period_mean_nll=period)
    control = out["period_offset_control"]
    assert control is not None
    total = out["blocked_inner_val_gain"]["mean"]
    offset = control["gain_period_offset_h_minus_period_mean"]["mean"]
    beyond = control["beyond_period_offset_period_mean_minus_learned"]["mean"]
    assert total == pytest.approx(offset + beyond, abs=1e-9)          # decomposition identity
    assert "period_mean_nll_seed_median" in out["arrays"]
    without = merge_seed_anchor_diagnostics(h_nll=h, learned_nll=learned, shifted_nll=shifted, random_nll=random,
                                            shift_valid=ok, segments=seg)
    assert without["period_offset_control"] is None
    with pytest.raises(ValueError):
        merge_seed_anchor_diagnostics(h_nll=h, learned_nll=learned, shifted_nll=shifted, random_nll=random,
                                      shift_valid=ok, segments=seg, period_mean_nll=period[:, :10])


def test_multi_seed_card_diagnostics_carries_the_control_into_the_card(tmp_path):
    bundle, _ = make_toy_bundle(seed=3, planted_beta=0.0)
    view = build_view(bundle)
    trainable = ResidualCountTrainable()
    cfg = RecipeConfig(**FAST)
    dirs = []
    for seed in (11, 12):
        out = tmp_path / f"seed_{seed}"
        result = train_recipe(trainable, view, cfg, seed=seed, device=CPU, out_dir=out)
        assert result["status"] == "complete"
        dirs.append(out)
    merged = multi_seed_card_diagnostics(trainable, view, cfg, dirs, device=CPU, out_dir=tmp_path / "card")
    control = merged["period_offset_control"]
    assert control is not None and np.isfinite(control["beyond_period_offset_period_mean_minus_learned"]["mean"])
    for row in merged["per_seed"]:
        assert np.isfinite(row["beyond_period_offset_period_mean_minus_learned"]["mean"])
    total = merged["blocked_inner_val_gain"]["mean"]
    assert total == pytest.approx(control["gain_period_offset_h_minus_period_mean"]["mean"]
                                  + control["beyond_period_offset_period_mean_minus_learned"]["mean"], abs=1e-6)
    seeds = [__import__("json").loads((d / "result.json").read_text()) for d in dirs]
    t0 = {"tiny_slice_overfit": {"pass": True}, "config_hash": "x"}
    diagnostics = {
        "blocked_inner_val_gain": merged["blocked_inner_val_gain"], "shift_null": merged["shift_null"],
        "random_reservoir_delta": merged["random_reservoir_delta"], "period_offset_control": control,
        "synthetic_recovery": {"pass": False}, "state_variance_rank": {}, "state_output_modulation": {},
        "multi_seed_diagnostics": {k: v for k, v in merged.items()
                                   if k not in ("blocked_inner_val_gain", "shift_null", "random_reservoir_delta",
                                                "period_offset_control")},
    }
    card = build_training_card(request=None, recipe_result=seeds[0], seed_results=seeds, t0=t0, diagnostics=diagnostics)
    assert "period_offset_control" in CARD_FIELDS
    assert card["period_offset_control"] == control
