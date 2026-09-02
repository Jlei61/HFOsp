"""Task 5: T0 diagnostics and training-card inputs (design §6, clauses G1-G8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.diagnostics import (
    amp_small_gradient_check,
    blocked_inner_val_gain,
    optimizer_membership,
    oracle_head_fit,
    random_reservoir_delta,
    run_t0,
    shift_null,
    state_output_modulation,
    state_variance_rank,
    state_write_jacobian,
    synthetic_recovery,
    tiny_slice_overfit,
)
from src.topic5_group_event_state.v033_training_lab.models import ArchConfig
from src.topic5_group_event_state.v033_training_lab.objective import ResidualCountTrainable
from src.topic5_group_event_state.v033_training_lab.synthetic import plant_residual_signal
from src.topic5_group_event_state.v033_training_lab.trainer import RecipeConfig, train_recipe
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")
FAST = dict(max_steps=60, min_steps=20, validate_every=10, patience=3)


def _view(seed=0):
    bundle, _ = make_toy_bundle(seed=seed, planted_beta=0.0)
    return build_view(bundle)


def _keys(obj, found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.append(str(k))
            _keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, found)
    return found


def test_g1_tiny_slice_overfit_passes_with_a_live_path_and_fails_with_a_dead_one(tmp_path):
    view = _view(0)
    trainable = ResidualCountTrainable()
    live = tiny_slice_overfit(trainable, view, RecipeConfig(), seed=1, device=CPU, n_slice=12, steps=300)
    assert live["pass"] is True and live["gap_closed"] >= live["threshold"] == 0.5
    assert live["n_slice"] == 12 and len(live["slice_anchor_idx"]) == 12
    assert live["nll_saturated"] < live["nll_end"] < live["nll_h"]
    dead_lr = {g: 0.0 for g in RecipeConfig().lr}
    dead = tiny_slice_overfit(trainable, view, RecipeConfig(lr=dead_lr), seed=1, device=CPU, n_slice=12, steps=50)
    assert dead["pass"] is False
    assert abs(dead["nll_end"] - dead["nll_start"]) < 1e-6          # nothing moved: dead gradient path


def test_g2_state_and_output_jacobians_match_closed_forms_for_the_bank():
    view = _view(1)
    trainable = ResidualCountTrainable()
    model = trainable.build(ArchConfig(), view, seed=2)
    result = state_write_jacobian(trainable, view, model, device=CPU)
    assert result["pass"] is True
    assert result["bank"]["max_relative_error"] < 1e-4
    assert result["output"]["max_abs_error"] < 1e-5 and result["output"]["jacobian_norm"] > 0
    gated = trainable.build(ArchConfig(state_family="gated_exploratory"), view, seed=2)
    g = state_write_jacobian(trainable, view, gated, device=CPU)
    assert g["bank"] is None and g["state"]["finite"] is True and g["state"]["nonzero"] is True


def test_g3_optimizer_membership_fails_on_an_unassigned_parameter():
    view = _view(2)
    trainable = ResidualCountTrainable()
    model = trainable.build(ArchConfig(), view, seed=3)
    lrs = RecipeConfig().effective_lr()
    ok = optimizer_membership(model, model.param_groups(lrs, 1e-4))
    assert ok["pass"] is True and ok["unassigned"] == [] and ok["duplicates"] == []
    model.extra = nn.Parameter(torch.zeros(3))
    bad = optimizer_membership(model, model.param_groups(lrs, 1e-4))
    assert bad["pass"] is False and bad["unassigned"] == ["extra"]


def test_g4_amp_check_is_skipped_on_cpu_and_reports_ratios_on_cuda():
    view = _view(3)
    trainable = ResidualCountTrainable()
    cpu = amp_small_gradient_check(trainable, view, RecipeConfig(), seed=4, device=CPU)
    assert cpu["status"] == "skipped" and cpu["pass"] is None
    if torch.cuda.is_available():
        cuda = amp_small_gradient_check(trainable, view, RecipeConfig(), seed=4, device=torch.device("cuda:0"))
        assert cuda["status"] == "complete" and isinstance(cuda["pass"], bool)
        for name, ratio in cuda["ratio_amp_over_fp32"].items():
            if cuda["fp32_grad_norm"][name] > 0:
                assert np.isfinite(ratio), name


def test_g5_shift_null_uses_the_v032_block_circular_donor_rule(tmp_path):
    view = _view(4)
    trainable = ResidualCountTrainable()
    result = train_recipe(trainable, view, RecipeConfig(**FAST), seed=5, device=CPU, out_dir=tmp_path)
    assert result["status"] == "complete"
    from src.topic5_group_event_state.v033_training_lab.trainer import load_trained
    model = load_trained(tmp_path, trainable, view, CPU)
    shift = shift_null(trainable, view, model, device=CPU, fraction=0.5)
    idx = view.phase_index["inner_val"]
    donor = block_circular_donor(view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=0.5)
    assert shift["n_valid_donors"] == int((donor >= 0).sum()) > 0
    assert shift["fraction"] == 0.5 and np.isfinite(shift["delta_shifted_minus_correct"]["mean"])
    assert "ci_low" in shift["delta_shifted_minus_correct"]


def test_g6_synthetic_recovery_recovers_a_strong_planted_signal_on_toy(tmp_path):
    view = _view(5)
    trainable = ResidualCountTrainable()
    cfg = RecipeConfig(max_steps=200, min_steps=40, validate_every=10, patience=6)
    rec = synthetic_recovery(trainable, view, cfg, seed=6, device=CPU, out_dir=tmp_path, beta=1.0, dispersion_r=8.0,
                             generator_seed=1, noise_seed=2)
    assert rec["source"] == "v032_residual_positive_proxy" and rec["beta"] == 1.0
    assert rec["blocked_inner_val_gain"]["ci_low"] > 0
    assert rec["shift_null"]["delta_shifted_minus_correct"]["mean"] > 0
    assert rec["pass"] is True


def test_g7_card_inputs_report_state_rank_modulation_random_reservoir_and_blocked_gain(tmp_path):
    view, _ = plant_residual_signal(_view(6), beta=1.0, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    trainable = ResidualCountTrainable()
    cfg = RecipeConfig(max_steps=120, min_steps=40, validate_every=10, patience=6)
    learned = train_recipe(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path / "learned")
    from src.topic5_group_event_state.v033_training_lab.trainer import load_trained
    model = load_trained(tmp_path / "learned", trainable, view, CPU)
    rank = state_variance_rank(trainable, view, model, device=CPU)
    assert rank["state_dim"] == 12 and 1.0 <= rank["participation_ratio"] <= 12.0
    assert rank["readout_rank"] == 3 and len(rank["covariance_eigenvalues"]) == 12
    mod = state_output_modulation(trainable, view, model, device=CPU)
    assert mod["pass"] is True and mod["inner_val"]["modulation_rms"] > 0
    assert "nll_mean_state_minus_dynamic" in mod["inner_val"]
    gain = blocked_inner_val_gain(trainable, view, model, device=CPU)
    assert gain["n_anchors"] == view.n("inner_val") and gain["n_blocks"] >= 2
    assert gain["mean"] == pytest.approx(learned["best_validation"]["gain_h_minus_model"], abs=1e-4)
    delta = random_reservoir_delta(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path,
                                   learned_dir=tmp_path / "learned")
    assert delta["random_result"]["arm"] == "random_reservoir"
    assert np.isfinite(delta["learned_minus_random"]["mean"])


def test_g8_run_t0_bundles_every_check_echoes_thresholds_and_never_mentions_dev_test(tmp_path):
    view, info = plant_residual_signal(_view(7), beta=1.0, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    trainable = ResidualCountTrainable()
    t0 = run_t0(trainable, view, RecipeConfig(**FAST), seed=8, device=CPU, out_dir=tmp_path,
                tiny_steps=200, probe_steps=40, true_state=info["z"][:, None])
    for key in ("optimizer_membership", "tiny_slice_overfit", "state_write_jacobian", "amp_small_gradient",
                "state_output_modulation", "state_variance_rank", "oracle_head_fit", "first_active_step",
                "clipping_fraction", "gradient_update_norms"):
        assert key in t0, key
    assert t0["tiny_slice_overfit"]["threshold"] == 0.5
    assert t0["oracle_head_fit"]["gain_h_minus_oracle"]["ci_low"] > 0
    assert isinstance(t0["gradient_path_ok"], bool)
    assert "dev_test" not in " ".join(_keys(t0)).lower()
    assert (tmp_path / "t0.json").exists()
