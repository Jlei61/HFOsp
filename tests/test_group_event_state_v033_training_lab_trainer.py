"""Task 4: recipe trainer (design §5, clauses T1-T10)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.models import ArchConfig
from src.topic5_group_event_state.v033_training_lab.objective import ResidualCountTrainable
from src.topic5_group_event_state.v033_training_lab.synthetic import plant_residual_signal
from src.topic5_group_event_state.v033_training_lab.trainer import (
    DEFAULT_LR,
    PlateauController,
    RecipeConfig,
    load_trained,
    lr_multiplier,
    train_recipe,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")
FAST = dict(max_steps=60, min_steps=20, validate_every=10, patience=3)


def _view(seed=0, planted_beta=0.0):
    bundle, _ = make_toy_bundle(seed=seed, planted_beta=planted_beta)
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


def test_t1_lr_controller_definitions_and_every_optimizer_and_schedule_runs(tmp_path):
    cfg = RecipeConfig(schedule="cosine", warmup_fraction=0.1, max_steps=100)
    assert cfg.warmup_steps() == 10
    assert lr_multiplier(5, cfg) == pytest.approx(0.5)
    assert lr_multiplier(10, cfg) == pytest.approx(1.0)
    assert lr_multiplier(100, cfg) == pytest.approx(0.0, abs=1e-12)
    assert lr_multiplier(55, cfg) == pytest.approx(0.5)
    const = RecipeConfig(schedule="constant", warmup_fraction=0.0, max_steps=100)
    assert lr_multiplier(1, const) == 1.0 and lr_multiplier(100, const) == 1.0
    plateau = PlateauController(factor=0.5, patience=2, tol=1e-6)
    for value in (1.0, 0.9, 0.95, 0.95):
        plateau.observe(value)
    assert plateau.factor == pytest.approx(0.5) and plateau.n_reductions == 1
    view = _view(0)
    trainable = ResidualCountTrainable()
    for i, (opt, sched) in enumerate((("adamw", "constant"), ("adam", "cosine"), ("rmsprop", "plateau"))):
        cfg = RecipeConfig(optimizer=opt, schedule=sched, plateau_patience=1, **FAST)
        result = train_recipe(trainable, view, cfg, seed=1, device=CPU, out_dir=tmp_path / f"run{i}")
        assert result["status"] == "complete", result.get("reason")
        assert result["optimizer"] == opt and result["schedule"] == sched
        assert all("lr_by_group" in row for row in result["history"])
    cos = train_recipe(trainable, view, RecipeConfig(schedule="cosine", max_steps=40, min_steps=40, patience=100,
                                                     validate_every=10), seed=1, device=CPU, out_dir=tmp_path / "cos")
    lrs = [row["lr_by_group"]["adapter_w"] for row in cos["history"]]
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12) and lrs[0] > 0.5 * DEFAULT_LR["adapter_w"]


def test_t2_warmup_scales_lr_linearly_and_flags_selection_inside_warmup(tmp_path):
    view = _view(1)
    cfg = RecipeConfig(warmup_fraction=0.5, max_steps=40, min_steps=40, validate_every=10, patience=100)
    assert cfg.warmup_steps() == 20
    result = train_recipe(ResidualCountTrainable(), view, cfg, seed=2, device=CPU, out_dir=tmp_path)
    by_step = {row["step"]: row for row in result["history"]}
    assert by_step[10]["lr_by_group"]["encoder_weights"] == pytest.approx(0.5 * DEFAULT_LR["encoder_weights"])
    assert by_step[20]["lr_by_group"]["encoder_weights"] == pytest.approx(DEFAULT_LR["encoder_weights"])
    assert result["warmup_steps"] == 20
    assert result["selected_in_warmup"] == (result["selected_step"] <= 20)


def test_t3_gate_alpha_moves_from_the_first_step(tmp_path):
    view, _info = plant_residual_signal(_view(2), beta=0.8, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    cfg = RecipeConfig(**FAST)
    result = train_recipe(ResidualCountTrainable(), view, cfg, seed=3, device=CPU, out_dir=tmp_path)
    assert abs(result["history"][0]["alpha"] - cfg.arch.alpha_init) > 1e-6
    assert result["first_active_step"]["adapter_gate_alpha"] == 1


def test_t4_first_active_step_per_group_and_random_reservoir_encoder_never_activates(tmp_path):
    view = _view(3)
    trainable = ResidualCountTrainable()
    learned = train_recipe(trainable, view, RecipeConfig(dispersion="low_lr", **FAST), seed=4, device=CPU,
                           out_dir=tmp_path / "learned")
    active = learned["first_active_step"]
    for name in ("encoder_weights", "encoder_bias", "adapter_w", "adapter_gate_alpha", "adapter_dispersion"):
        assert active[name] == 1, (name, active)
    assert active["state_weights"] is None and active["state_bias"] is None      # bank has no state params
    assert learned["all_groups_active_before_selection"] is True
    random = train_recipe(trainable, view, RecipeConfig(dispersion="low_lr", **FAST), seed=4, device=CPU,
                          out_dir=tmp_path / "random", arm="random_reservoir")
    assert random["arm"] == "random_reservoir"
    assert random["first_active_step"]["encoder_weights"] is None
    assert random["first_active_step"]["adapter_w"] == 1
    assert random["history"][0]["grad_norm_by_group"]["encoder_weights"] == 0.0


def test_t5_clipping_fraction_counts_clipped_steps(tmp_path):
    view = _view(4)
    always = train_recipe(ResidualCountTrainable(), view, RecipeConfig(grad_clip=1e-8, **FAST), seed=5,
                          device=CPU, out_dir=tmp_path / "a")
    never = train_recipe(ResidualCountTrainable(), view, RecipeConfig(grad_clip=1e9, **FAST), seed=5,
                         device=CPU, out_dir=tmp_path / "b")
    assert always["clipping_fraction"] == pytest.approx(1.0)
    assert never["clipping_fraction"] == pytest.approx(0.0)


def test_t6_dispersion_frozen_versus_low_lr(tmp_path):
    view = _view(5)
    trainable = ResidualCountTrainable()
    frozen = train_recipe(trainable, view, RecipeConfig(dispersion="frozen", **FAST), seed=6, device=CPU,
                          out_dir=tmp_path / "frozen")
    assert np.allclose(frozen["final_log_r"], view.log_r_h)
    assert frozen["first_active_step"]["adapter_dispersion"] is None
    assert "adapter_dispersion" not in frozen["optimizer_groups"]
    low = train_recipe(trainable, view, RecipeConfig(dispersion="low_lr", **FAST), seed=6, device=CPU,
                       out_dir=tmp_path / "low")
    assert not np.allclose(low["final_log_r"], view.log_r_h)
    assert low["lr_by_group"]["adapter_dispersion"] == pytest.approx(0.1 * DEFAULT_LR["adapter_w"])


def test_t7_rung_budget_resume_matches_single_run_and_repeat_is_skipped(tmp_path):
    view = _view(6)
    trainable = ResidualCountTrainable()
    cfg = RecipeConfig(max_steps=60, min_steps=60, validate_every=10, patience=100)
    partial = train_recipe(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path / "rung", steps_budget=30)
    assert partial["status"] == "complete" and partial["n_steps_run"] == 30 and partial["resumable"] is True
    assert partial["steps_budget"] == 30 and partial["budget_is_full"] is False
    assert (tmp_path / "rung" / "last.pt").exists()
    resumed = train_recipe(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path / "rung", steps_budget=60)
    assert resumed["resumed_from_step"] == 30 and resumed["n_steps_run"] == 60 and resumed["resumable"] is False
    single = train_recipe(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path / "single", steps_budget=60)
    a = torch.load(tmp_path / "rung" / "checkpoint.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "single" / "checkpoint.pt", weights_only=False)["model_state"]
    for key in a:
        assert torch.allclose(a[key].float(), b[key].float(), atol=1e-6), key
    assert [r["step"] for r in resumed["history"]] == [r["step"] for r in single["history"]]
    assert resumed["selected_step"] == single["selected_step"]
    again = train_recipe(trainable, view, cfg, seed=7, device=CPU, out_dir=tmp_path / "rung", steps_budget=60)
    assert again.get("skipped_existing") is True
    assert not (tmp_path / "rung" / "last.pt").exists()


class _NaNAfter(ResidualCountTrainable):
    def __init__(self, after: int):
        super().__init__()
        self.after = after
        self.calls = 0

    def loss_terms(self, *args, **kwargs):
        terms = super().loss_terms(*args, **kwargs)
        if kwargs.get("differentiable_statistics"):
            self.calls += 1
            if self.calls > self.after:
                terms.nll = terms.nll * float("nan")
        return terms


def test_t8_non_finite_loss_is_dumped_not_raised(tmp_path):
    view = _view(7)
    result = train_recipe(_NaNAfter(after=2), view, RecipeConfig(**FAST), seed=8, device=CPU, out_dir=tmp_path)
    assert result["status"] == "nan"
    dump = json.loads((tmp_path / "nan_dump.json").read_text())
    assert dump["step"] == 3 and dump["first_non_finite"] == "loss"
    assert "grad_norm_by_group" in dump and "lr_by_group" in dump


def test_t9_learning_curves_parquet_has_one_row_per_validation(tmp_path):
    view = _view(8)
    result = train_recipe(ResidualCountTrainable(), view, RecipeConfig(**FAST), seed=9, device=CPU, out_dir=tmp_path)
    frame = pd.read_parquet(result["curves_path"])
    assert len(frame) == len(result["history"])
    for column in ("step", "inner_val_nll", "train_nll", "grad_norm.adapter_w", "update_norm.adapter_w",
                   "lr.adapter_w", "alpha", "clipped"):
        assert column in frame.columns, column


def test_t10_selection_is_inner_val_only_and_result_never_mentions_dev_test(tmp_path):
    view, _info = plant_residual_signal(_view(9), beta=0.8, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    trainable = ResidualCountTrainable()
    cfg = RecipeConfig(max_steps=200, min_steps=40, validate_every=10, patience=6)
    result = train_recipe(trainable, view, cfg, seed=10, device=CPU, out_dir=tmp_path)
    keys = " ".join(_keys(result)).lower()
    assert "dev_test" not in keys and "sealed" not in keys or result["sealed_partition_opened"] is False
    assert result["selection_phase"] == "inner_val"
    assert result["best_validation"]["inner_val_nll"] < result["best_validation"]["inner_val_nll_h"] - 0.05
    model = load_trained(tmp_path, trainable, view, CPU)
    terms = trainable.loss_terms(model, view, "inner_val", device=CPU, differentiable_statistics=False,
                                 sampling="anchor_balanced", lookback_seconds=7200.0)
    assert float(terms.nll.mean()) == pytest.approx(result["best_validation"]["inner_val_nll"], abs=1e-4)
    assert math.isfinite(result["best_validation"]["gain_h_minus_model"])
