"""Task 6: full-batch NB residual trainer (T1-T8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v032_model.config import ModelConfig
from src.topic5_group_event_state.v032_model.trainer import (
    anchor_terms,
    load_checkpoint_model,
    train_residual_model,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")
FAST = dict(max_steps=60, min_steps=20, validate_every=10, patience=3, alpha_freeze_steps=15)


def test_t1_t3_alpha_frozen_early_and_selection_metadata_recorded(tmp_path):
    bundle, _ = make_toy_bundle(seed=0)
    cfg = ModelConfig(**FAST)
    result = train_residual_model(bundle, cfg, seed=1, device=CPU, out_dir=tmp_path)
    assert result["status"] == "complete"
    history = result["history"]
    alphas = [row["alpha"] for row in history]
    steps = [row["step"] for row in history]
    early = [a for s, a in zip(steps, alphas) if s <= cfg.alpha_freeze_steps]
    late = [a for s, a in zip(steps, alphas) if s > cfg.alpha_freeze_steps + 5]
    assert all(abs(a - cfg.alpha_init) < 1e-7 for a in early) and early
    assert late and any(abs(a - cfg.alpha_init) > 1e-6 for a in late)
    for key in ("selected_step", "selected_first_validation", "selected_at_budget_edge",
                "selection_metric", "n_train_anchors", "n_val_anchors", "h_source", "config_hash"):
        assert key in result
    assert result["selection_metric"] == "dev_val_nb_nll_h_plus_s_correct_1800s"
    assert isinstance(result["selected_first_validation"], bool)
    assert result["alpha_frozen_until_step"] == cfg.alpha_freeze_steps
    assert (tmp_path / "checkpoint.pt").exists() and (tmp_path / "result.json").exists()
    assert json.loads((tmp_path / "result.json").read_text())["config_hash"] == cfg.config_hash()


def test_learns_planted_residual_beyond_h_on_dev_val(tmp_path):
    bundle, _ = make_toy_bundle(seed=3, planted_beta=0.8)
    cfg = ModelConfig(max_steps=300, min_steps=50, validate_every=10, patience=8, alpha_freeze_steps=10)
    result = train_residual_model(bundle, cfg, seed=2, device=CPU, out_dir=tmp_path)
    best = result["best_validation"]
    assert best["nll_h_plus_s_correct"] < best["nll_h"] - 0.05
    assert result["selected_step"] > 0


def test_t4_t8_checkpoint_contents_replay_and_dtypes(tmp_path):
    bundle, _ = make_toy_bundle(seed=4)
    cfg = ModelConfig(**FAST)
    result = train_residual_model(bundle, cfg, seed=5, device=CPU, out_dir=tmp_path)
    payload = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "config", "config_hash", "standardizer", "feature_names",
                "fingerprint", "h_source", "log_r_h", "selected_step", "parameter_sha256"):
        assert key in payload, key
    assert "phi_mean" in payload["model_state"] and "train_mean_state" in payload["model_state"]
    model = load_checkpoint_model(tmp_path / "checkpoint.pt", in_dim=bundle.x_std.shape[1], device=CPU)
    terms = anchor_terms(model, bundle, phase="dev_val", horizon=1800.0, device=CPU)
    assert terms["nll"].dtype == torch.float32 and terms["state"].dtype == torch.float32
    assert terms["nll"].shape[0] == int(bundle.anchor_mask("dev_val", 1800.0).sum())
    assert float(terms["nll"].mean()) == pytest.approx(result["best_validation"]["nll_h_plus_s_correct"], abs=1e-4)


def test_t5_interrupted_run_resumes_to_identical_parameters(tmp_path):
    bundle, _ = make_toy_bundle(seed=6)
    cfg = ModelConfig(**FAST)
    full = train_residual_model(bundle, cfg, seed=7, device=CPU, out_dir=tmp_path / "full")
    partial = train_residual_model(
        bundle, cfg, seed=7, device=CPU, out_dir=tmp_path / "resume", interrupt_after_step=25
    )
    assert partial["status"] == "interrupted" and (tmp_path / "resume" / "last.pt").exists()
    resumed = train_residual_model(bundle, cfg, seed=7, device=CPU, out_dir=tmp_path / "resume")
    assert resumed["status"] == "complete" and resumed["resumed_from_step"] == 25
    a = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "resume" / "checkpoint.pt", weights_only=False)["model_state"]
    for key in a:
        assert torch.allclose(a[key].float(), b[key].float(), atol=1e-6), key
    assert resumed["selected_step"] == full["selected_step"]
    again = train_residual_model(bundle, cfg, seed=7, device=CPU, out_dir=tmp_path / "resume")
    assert again["status"] == "complete" and again.get("skipped_existing") is True


def test_t6_t7_gradient_records_and_random_reservoir_freezes_encoder(tmp_path):
    bundle, _ = make_toy_bundle(seed=8)
    cfg = ModelConfig(**FAST)
    result = train_residual_model(bundle, cfg, seed=9, device=CPU, out_dir=tmp_path / "learned")
    first = result["history"][0]
    assert first["grad_norm_pre_clip"] > 0 and first["grad_norm_post_clip"] <= cfg.grad_clip + 1e-6
    assert set(first["grad_norm_by_group"]) >= {"encoder_weights", "adapter_w", "adapter_gate_alpha"}
    random_result = train_residual_model(
        bundle, cfg, seed=9, device=CPU, out_dir=tmp_path / "random", arm="random_reservoir"
    )
    assert random_result["arm"] == "random_reservoir"
    assert random_result["history"][0]["grad_norm_by_group"]["encoder_weights"] == 0.0
    init = torch.load(tmp_path / "random" / "checkpoint.pt", weights_only=False)
    assert init["encoder_frozen"] is True
