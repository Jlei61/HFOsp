"""Task 7: evaluation arms on one anchor set (E1-E5) + block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v032_model.config import ModelConfig
from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci, evaluate_arms
from src.topic5_group_event_state.v032_model.model import build_model
from src.topic5_group_event_state.v032_model.trainer import (
    bundle_tensors,
    refresh_statistics,
    resolve_log_r_h,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")


def _ready_model(bundle, cfg, seed):
    model = build_model(cfg, in_dim=bundle.x_std.shape[1], log_r_init=0.0, seed=seed)
    refresh_statistics(model, bundle_tensors(bundle, CPU, horizon=1800.0))
    return model.eval()


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, size=200)
    groups = np.repeat(np.arange(4), 50)
    a = block_bootstrap_mean_ci(values, groups, block_len=6, n_boot=500, seed=1)
    b = block_bootstrap_mean_ci(values, groups, block_len=6, n_boot=500, seed=1)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"] and a["se"] > 0
    assert a["n"] == 200 and a["n_blocks"] > 0
    assert abs(a["mean"] - values.mean()) < 1e-12


def test_e1_e5_five_arms_share_anchors_mean_arm_is_constant_and_h_uses_given_dispersion():
    bundle, _ = make_toy_bundle(seed=11)
    cfg = ModelConfig()
    model = _ready_model(bundle, cfg, seed=1)
    random_model = _ready_model(bundle, cfg, seed=99)
    log_r_h, _src = resolve_log_r_h(bundle, 1800.0)
    out = evaluate_arms(model, bundle, cfg, device=CPU, phase="dev_test", horizon=1800.0,
                        log_r_h=log_r_h, random_model=random_model)
    n = int(bundle.anchor_mask("dev_test", 1800.0).sum())
    per = out["per_anchor"]
    for key in ("nll_h", "nll_correct", "nll_shifted", "nll_mean", "nll_random"):
        assert len(per[key]) == n, key                                              # E1
    assert out["arms"]["h_plus_mean_s"]["modulation_std"] < 1e-6                      # E2
    donor = np.asarray(per["donor"])
    shifted = np.asarray(per["nll_shifted"], dtype=float)
    assert np.all(np.isnan(shifted[donor < 0])) and np.all(np.isfinite(shifted[donor >= 0]))
    assert out["contrasts"]["shifted_minus_correct"]["n"] == int((donor >= 0).sum())  # E3
    assert out["effective_independent_windows"] == bundle.effective_independent_windows("dev_test", 1800.0)  # E4
    assert out["arms"]["h"]["log_r"] == pytest.approx(log_r_h)                         # E5
    diff = np.asarray(per["nll_h"]) - np.asarray(per["nll_correct"])
    assert out["contrasts"]["h_minus_correct"]["mean"] == pytest.approx(float(diff.mean()))
    assert out["contrasts"]["random_minus_correct"]["n"] == n
    assert set(out["shift_alternatives"]) == {"0.25", "0.75"}
    # the shifted arm never uses a donor outside the anchor's own segment
    seg = bundle.anchor_segment[np.asarray(per["idx"])]
    ok = donor >= 0
    assert np.all(seg[ok] == seg[donor[ok]])


def test_e6_intercept_recalibrated_h_arm_is_fit_on_train_anchors_only():
    from src.topic5_group_event_state.v032_model.evaluate import fit_train_intercept

    bundle, _ = make_toy_bundle(seed=12)
    cfg = ModelConfig()
    model = _ready_model(bundle, cfg, seed=1)
    log_r_h, _src = resolve_log_r_h(bundle, 1800.0)
    c = fit_train_intercept(bundle, horizon=1800.0, log_r_h=log_r_h)
    assert np.isfinite(c) and abs(c) < 3.0
    # perturbing dev_test counts leaves the intercept unchanged
    h_i = bundle.horizon_index(1800.0)
    test = bundle.anchor_mask("dev_test", 1800.0)
    bundle.counts[test, h_i] = bundle.counts[test, h_i] * 3 + 7
    assert fit_train_intercept(bundle, horizon=1800.0, log_r_h=log_r_h) == pytest.approx(c)
    out = evaluate_arms(model, bundle, cfg, device=CPU, phase="dev_test", horizon=1800.0, log_r_h=log_r_h)
    arm = out["arms"]["h_plus_intercept"]
    assert arm["intercept"] == pytest.approx(c) and arm["n"] == out["n_anchors"]
    assert out["contrasts"]["intercept_minus_correct"]["n"] == out["n_anchors"]
    assert out["contrasts"]["h_minus_intercept"]["n"] == out["n_anchors"]
    assert len(out["per_anchor"]["nll_intercept"]) == out["n_anchors"]
