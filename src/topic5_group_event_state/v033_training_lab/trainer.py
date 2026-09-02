"""Recipe trainer: T1 knobs with per-parameter-group bookkeeping (design §5).

One optimizer step is one full pass over every TRAIN anchor (complete
chronological scan for the fixed leaky bank).  Nothing here reads a
development-evaluation number: selection is on the inner-validation anchors
exposed by the data view and nothing else.

Contract clauses (plan Task 4):
  [T1] optimizer adamw/adam/rmsprop, schedule constant/cosine/plateau with the LR trajectories of
       ``lr_multiplier`` / ``PlateauController``;  [T2] global linear warm-up, ``selected_in_warmup``;
  [T3] the gate alpha is trainable from step 1;  [T4] ``first_active_step`` per group = first step with a
       non-zero gradient AND a non-zero parameter update;  [T5] ``clipping_fraction``;
  [T6] dispersion frozen (not in the optimizer) vs low_lr (0.1 x adapter LR);  [T7] rung budgets resume
       from ``last.pt`` and are bit-for-bit the single run;  [T8] non-finite loss / gradient -> ``nan_dump``;
  [T9] ``learning_curves.parquet`` one row per validation;  [T10] no dev_test anywhere.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data import DataView, SAMPLINGS, SCALINGS
from .models import ArchConfig, GROUP_NAMES
from .objective import Trainable
from .paths import atomic_write_json, atomic_write_torch, current_commit, file_hash, payload_hash

OPTIMIZERS = ("adamw", "adam", "rmsprop")
SCHEDULES = ("constant", "cosine", "plateau")
DISPERSION_MODES = ("frozen", "low_lr")
ARMS = ("learned", "random_reservoir")
DEFAULT_LR: dict[str, float] = {
    "encoder_weights": 1e-3, "encoder_bias": 1e-3, "state_weights": 1e-3, "state_bias": 1e-3,
    "adapter_w": 3e-3, "adapter_gate_alpha": 3e-3, "adapter_dispersion": 3e-4,
}
CHECKPOINT_FORMAT = "group_event_state_v0_3_3_training_lab_checkpoint"
RESULT_FORMAT = "group_event_state_v0_3_3_training_lab_train_result"


@dataclass(frozen=True)
class RecipeConfig:
    arch: ArchConfig = field(default_factory=ArchConfig)
    optimizer: str = "adamw"
    schedule: str = "constant"
    warmup_fraction: float = 0.0
    lr: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LR))
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dispersion: str = "frozen"
    sampling: str = "anchor_balanced"
    scaling: str = "zscore"
    max_steps: int = 600
    min_steps: int = 50
    validate_every: int = 10
    patience: int = 10
    amp_encoder: bool = False
    checkpointing: bool = False
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    improvement_tol: float = 1e-6

    def validate(self) -> "RecipeConfig":
        self.arch.validate()
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer {self.optimizer!r} not in {OPTIMIZERS}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule {self.schedule!r} not in {SCHEDULES}")
        if self.dispersion not in DISPERSION_MODES:
            raise ValueError(f"dispersion {self.dispersion!r} not in {DISPERSION_MODES}")
        if self.sampling not in SAMPLINGS or self.scaling not in SCALINGS:
            raise ValueError("unknown sampling / scaling")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        missing = [g for g in GROUP_NAMES if g not in self.lr]
        if missing:
            raise ValueError(f"lr missing for groups {missing}")
        if self.min_steps > self.max_steps or self.validate_every < 1 or self.max_steps < 1:
            raise ValueError("invalid step schedule")
        return self

    @property
    def lookback_seconds(self) -> float:
        return float(max(self.arch.taus_seconds))

    def warmup_steps(self) -> int:
        return int(round(self.warmup_fraction * self.max_steps))

    def effective_lr(self) -> dict[str, float]:
        lrs = {g: float(self.lr[g]) for g in GROUP_NAMES}
        if self.dispersion == "low_lr":
            lrs["adapter_dispersion"] = 0.1 * float(self.lr["adapter_w"])
        return lrs

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arch"] = self.arch.as_dict()
        payload["lr"] = {k: float(v) for k, v in sorted(self.lr.items())}
        return payload

    def config_hash(self) -> str:
        return payload_hash(self.as_dict())

    def with_overrides(self, **over: Any) -> "RecipeConfig":
        return replace(self, **over)


def lr_multiplier(step: int, cfg: RecipeConfig) -> float:
    """Warm-up x schedule factor at optimizer step ``step`` (1-based) -- [T1][T2]."""

    warm = cfg.warmup_steps()
    factor = 1.0
    if warm > 0 and step <= warm:
        factor = float(step) / float(warm)
    if cfg.schedule == "cosine":
        span = max(cfg.max_steps - warm, 1)
        progress = min(max(step - warm, 0), span) / span
        factor *= 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


class PlateauController:
    """ReduceLROnPlateau on the inner-validation metric (definition in one place)."""

    def __init__(self, factor: float, patience: int, tol: float, min_factor: float = 1e-3) -> None:
        self.reduction = float(factor)
        self.patience = int(patience)
        self.tol = float(tol)
        self.min_factor = float(min_factor)
        self.factor = 1.0
        self.best = math.inf
        self.stale = 0
        self.n_reductions = 0

    def observe(self, value: float) -> float:
        if math.isfinite(value) and value < self.best - self.tol:
            self.best = value
            self.stale = 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.factor = max(self.factor * self.reduction, self.min_factor)
                self.n_reductions += 1
                self.stale = 0
        return self.factor

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "best": self.best, "stale": self.stale, "n_reductions": self.n_reductions}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.factor = float(payload["factor"])
        self.best = float(payload["best"])
        self.stale = int(payload["stale"])
        self.n_reductions = int(payload["n_reductions"])


def build_optimizer(name: str, groups: list[dict[str, Any]]) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(groups)
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups)
    raise ValueError(f"unknown optimizer {name!r}")


# ------------------------------------------------------------------ helpers
def _norm_of(tensors: list[Tensor]) -> float:
    squares = [t.detach().float().pow(2).sum() for t in tensors]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0


def _grad_norms(groups: list[dict[str, Any]]) -> dict[str, float]:
    return {g["name"]: _norm_of([p.grad for p in g["params"] if p.grad is not None]) for g in groups}


def _first_non_finite_grad(model: torch.nn.Module) -> str | None:
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return f"grad:{name}"
    return None


def _flatten(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _curves_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in history:
        flat = _flatten({k: v for k, v in row.items() if k not in ("grad_norm_by_group", "update_norm_by_group",
                                                                     "lr_by_group")})
        flat.update({f"grad_norm.{k}": v for k, v in row["grad_norm_by_group"].items()})
        flat.update({f"update_norm.{k}": v for k, v in row["update_norm_by_group"].items()})
        flat.update({f"lr.{k}": v for k, v in row["lr_by_group"].items()})
        rows.append(flat)
    return pd.DataFrame(rows)


def load_trained(out_dir: Path, trainable: Trainable, view: DataView, device: torch.device) -> torch.nn.Module:
    payload = torch.load(Path(out_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    arch = ArchConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in payload["config"]["arch"].items()})
    model = trainable.build(arch, view, seed=int(payload["seed"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if payload.get("encoder_frozen"):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    model.amp_encoder = bool(payload["config"].get("amp_encoder", False))
    return model.to(device).eval()


# ------------------------------------------------------------------- training
def train_recipe(
    trainable: Trainable,
    view: DataView,
    cfg: RecipeConfig,
    seed: int,
    *,
    device: torch.device,
    out_dir: Path,
    arm: str = "learned",
    steps_budget: int | None = None,
    overwrite: bool = False,
    interrupt_after_step: int | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    cfg.validate()
    if cfg.scaling != view.scaling:
        raise ValueError(f"recipe scaling {cfg.scaling!r} but the view was built with {view.scaling!r}")
    budget = int(cfg.max_steps if steps_budget is None else min(int(steps_budget), cfg.max_steps))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, result_path, last_path = out_dir / "checkpoint.pt", out_dir / "result.json", out_dir / "last.pt"
    config_hash = cfg.config_hash()
    if result_path.exists() and checkpoint_path.exists() and not overwrite:
        previous = json.loads(result_path.read_text())
        same = (previous.get("config_hash") == config_hash and previous.get("arm") == arm
                and int(previous.get("seed", -1)) == int(seed) and previous.get("status") == "complete")
        if same and (int(previous.get("n_steps_run", 0)) >= budget or previous.get("stopped_reason") == "patience"):
            previous["skipped_existing"] = True
            return previous

    started = time.time()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    tensors = trainable.tensors(view, device)
    model = trainable.build(cfg.arch, view, seed).to(device)
    model.amp_encoder = bool(cfg.amp_encoder)
    model.checkpointing = bool(cfg.checkpointing)
    encoder_frozen = arm == "random_reservoir"
    if encoder_frozen:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    if cfg.dispersion == "frozen":                                                       # [T6]
        model.adapter.log_r.requires_grad_(False)
    lrs = cfg.effective_lr()
    groups = trainable.param_groups(model, lrs, cfg.weight_decay)
    optimizer_groups = [
        {k: v for k, v in g.items() if k != "params"} | {"params": [p for p in g["params"] if p.requires_grad]}
        for g in groups
    ]
    optimizer_groups = [g for g in optimizer_groups if g["params"]]
    base_lr = {g["name"]: float(g["lr"]) for g in optimizer_groups}
    optimizer = build_optimizer(cfg.optimizer, optimizer_groups)
    plateau = PlateauController(cfg.plateau_factor, cfg.plateau_patience, cfg.improvement_tol) \
        if cfg.schedule == "plateau" else None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    init_state = copy.deepcopy(model.state_dict())
    h_train = float(trainable.h_only_nll(view, "train").mean())
    h_val = float(trainable.h_only_nll(view, "inner_val").mean())

    base_report: dict[str, Any] = {
        "format": RESULT_FORMAT, "subject": view.subject, "seed": int(seed), "arm": arm,
        "config": cfg.as_dict(), "config_hash": config_hash, "split_hash": view.split_hash,
        "input_hash": view.input_hash, "h_source": view.h_source, "bins_seconds": [list(b) for b in view.bins],
        "optimizer": cfg.optimizer, "schedule": cfg.schedule, "warmup_steps": cfg.warmup_steps(),
        "lr_by_group": base_lr, "optimizer_groups": {g["name"]: len(g["params"]) for g in optimizer_groups},
        "steps_budget": budget, "budget_is_full": budget == cfg.max_steps,
        "n_train_anchors": view.n("train"), "n_inner_val_anchors": view.n("inner_val"),
        "n_train_events": int(view.train_event_mask.sum()),
        "effective_independent_windows": {"train": view.effective_independent_windows("train"),
                                          "inner_val": view.effective_independent_windows("inner_val")},
        "selection_phase": "inner_val", "selection_metric": "inner_val_nb_nll_mean_per_anchor",
        "selection_metric_is_canonical": False, "development_evaluation_read": False,
        "sealed_partition_opened": False, "source_commit": current_commit(), "device": str(device),
    }

    history: list[dict[str, Any]] = []
    best_value, best_step, stale = math.inf, -1, 0
    best_state: dict[str, Tensor] | None = None
    first_active: dict[str, int | None] = {g["name"]: None for g in groups}
    n_clipped, start_step, resumed_from = 0, 0, None
    if last_path.exists() and not overwrite:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("config_hash") == config_hash and saved.get("arm") == arm and saved.get("seed") == int(seed):
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            history = list(saved["history"])
            best_value, best_step, stale = saved["best_value"], saved["best_step"], saved["stale"]
            best_state = saved.get("best_state")
            first_active = dict(saved["first_active"])
            n_clipped = int(saved["n_clipped"])
            start_step = int(saved["step"])
            resumed_from = start_step
            init_state = saved["init_state"]
            torch.set_rng_state(saved["rng_state"])
            if plateau is not None and saved.get("plateau") is not None:
                plateau.load_state_dict(saved["plateau"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _save_last(step: int) -> None:
        atomic_write_torch(last_path, {
            "format": CHECKPOINT_FORMAT + "_last", "config_hash": config_hash, "arm": arm, "seed": int(seed),
            "step": int(step), "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "history": history, "best_value": best_value,
            "best_step": best_step, "stale": stale, "best_state": best_state, "first_active": first_active,
            "n_clipped": n_clipped, "init_state": init_state, "rng_state": torch.get_rng_state(),
            "plateau": None if plateau is None else plateau.state_dict(),
        })

    def _nan_dump(step: int, what: str, loss_value: float, grad_by_group: dict[str, float], lr_now: dict[str, float]) -> dict[str, Any]:
        dump = {"step": int(step), "first_non_finite": what, "loss": loss_value, "grad_norm_by_group": grad_by_group,
                "lr_by_group": lr_now, "alpha": float(model.adapter.alpha.detach()),
                "log_r": model.adapter.log_r.detach().cpu().tolist(), "config_hash": config_hash, "seed": int(seed)}
        atomic_write_json(out_dir / "nan_dump.json", dump)
        report = {**base_report, "status": "nan", "reason": f"non-finite {what} at step {step}", "nan": dump,
                  "history": history, "n_steps_run": int(step), "elapsed_seconds": time.time() - started}
        atomic_write_json(result_path, report)
        return report

    stopped_reason = "budget"
    step = start_step
    for step in range(start_step + 1, budget + 1):
        lr_now = {name: base_lr[name] * lr_multiplier(step, cfg) * (plateau.factor if plateau else 1.0)
                  for name in base_lr}
        for g in optimizer.param_groups:
            g["lr"] = lr_now[g["name"]]
        model.train()
        terms = trainable.loss_terms(model, view, "train", device=device, differentiable_statistics=True,
                                     sampling=cfg.sampling, lookback_seconds=cfg.lookback_seconds, tensors=tensors)
        loss = (terms.nll * terms.weights).sum() / terms.weights.sum()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):                                                   # [T8]
            return _nan_dump(step, "loss", loss_value, {}, lr_now)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bad = _first_non_finite_grad(model)
        grad_by_group = _grad_norms(groups)
        if bad is not None:
            return _nan_dump(step, bad, loss_value, grad_by_group, lr_now)
        pre_clip = _norm_of([p.grad for p in trainable_params if p.grad is not None])
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        post_clip = _norm_of([p.grad for p in trainable_params if p.grad is not None])
        clipped = bool(pre_clip > cfg.grad_clip)
        n_clipped += int(clipped)                                                            # [T5]
        before = {g["name"]: [p.detach().clone() for p in g["params"]] for g in groups}
        optimizer.step()
        update_by_group = {g["name"]: _norm_of([p.detach() - b for p, b in zip(g["params"], before[g["name"]])])
                           for g in groups}
        for name in first_active:                                                            # [T4]
            if first_active[name] is None and grad_by_group[name] > 0.0 and update_by_group[name] > 0.0:
                first_active[name] = int(step)
        validate = step % cfg.validate_every == 0 or step == budget
        if validate:
            model.eval()
            trainable.refresh_statistics(model, view, device, tensors)
            with torch.no_grad():
                val = trainable.loss_terms(model, view, "inner_val", device=device, differentiable_statistics=False,
                                           sampling="anchor_balanced", lookback_seconds=cfg.lookback_seconds,
                                           tensors=tensors)
                tr = trainable.loss_terms(model, view, "train", device=device, differentiable_statistics=False,
                                          sampling="anchor_balanced", lookback_seconds=cfg.lookback_seconds,
                                          tensors=tensors)
            val_nll = float(val.nll.mean())
            row = {
                "step": int(step), "train_loss_weighted": loss_value, "train_nll": float(tr.nll.mean()),
                "train_nll_h": h_train, "inner_val_nll": val_nll, "inner_val_nll_h": h_val,
                "inner_val_per_bin_nll": val.per_bin_nll.mean(dim=0).cpu().tolist(),
                "inner_val_modulation_rms": float(val.modulation.pow(2).mean().sqrt()),
                "train_modulation_rms": float(tr.modulation.pow(2).mean().sqrt()),
                "alpha": float(model.adapter.alpha.detach()), "log_r": model.adapter.log_r.detach().cpu().tolist(),
                "grad_norm_pre_clip": pre_clip, "grad_norm_post_clip": post_clip, "clipped": clipped,
                "grad_norm_by_group": grad_by_group, "update_norm_by_group": update_by_group, "lr_by_group": lr_now,
                "plateau_factor": plateau.factor if plateau else 1.0, "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            if plateau is not None:
                plateau.observe(val_nll)
            if math.isfinite(val_nll) and val_nll < best_value - cfg.improvement_tol:
                best_value, best_step, stale = val_nll, int(step), 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            _save_last(step)
            atomic_write_json(out_dir / "progress.json", {**base_report, "status": "running", "step": step,
                                                           "history": history, "best_step": best_step})
            if step >= cfg.min_steps and stale >= cfg.patience:
                stopped_reason = "patience"
                break
        if interrupt_after_step is not None and step >= int(interrupt_after_step):
            _save_last(step)
            report = {**base_report, "status": "interrupted", "step": int(step), "history": history}
            atomic_write_json(out_dir / "progress.json", report)
            return report

    if best_state is None:
        raise RuntimeError("training never produced a finite inner-validation value")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    validation_steps = [row["step"] for row in history]
    at_edge = bool(stopped_reason == "budget" and best_step == validation_steps[-1])
    warm = cfg.warmup_steps()
    active_groups = {n: s for n, s in first_active.items()
                     if any(p.requires_grad for p in next(g for g in groups if g["name"] == n)["params"])}
    all_active = all(s is not None and s <= best_step for s in active_groups.values())
    resumable = bool(stopped_reason == "budget" and budget < cfg.max_steps)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    curves_path = out_dir / "learning_curves.parquet"
    _curves_frame(history).to_parquet(curves_path, index=False)                            # [T9]
    payload = {
        "format": CHECKPOINT_FORMAT, "subject": view.subject, "seed": int(seed), "arm": arm,
        "config": cfg.as_dict(), "config_hash": config_hash, "split_hash": view.split_hash,
        "input_hash": view.input_hash, "bins_seconds": [list(b) for b in view.bins],
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "init_state": init_state,
        "scaler_stats": view.scaler_stats, "feature_names": list(view.feature_names), "fingerprint": view.fingerprint,
        "h_source": view.h_source, "log_r_h": [float(v) for v in view.log_r_h], "selected_step": int(best_step),
        "encoder_frozen": encoder_frozen, "parameter_sha256": payload_hash(
            {k: v.detach().cpu().numpy().tobytes().hex() for k, v in sorted(model.state_dict().items())}),
        "source_commit": base_report["source_commit"],
    }
    atomic_write_torch(checkpoint_path, payload)
    report = {
        **base_report, "status": "complete", "selected_step": int(best_step),
        "selected_in_warmup": bool(best_step <= warm),                                       # [T2]
        "selected_at_budget_edge": at_edge, "selected_first_validation": bool(best_step == validation_steps[0]),
        "n_steps_run": int(step), "stopped_reason": stopped_reason, "resumable": resumable,
        "resumed_from_step": resumed_from, "first_active_step": first_active,
        "all_groups_active_before_selection": bool(all_active),
        "clipping_fraction": float(n_clipped / max(step, 1)),
        "plateau": {"reached": bool(stale >= cfg.patience), "since_step": int(best_step), "stale_validations": int(stale),
                    "lr_reductions": plateau.n_reductions if plateau else 0},
        "best_validation": {"step": int(best_step), "inner_val_nll": float(best_value), "inner_val_nll_h": h_val,
                            "gain_h_minus_model": float(h_val - best_value)},
        "final_alpha": float(model.adapter.alpha.detach()), "final_log_r": model.adapter.log_r.detach().cpu().tolist(),
        "history": history, "curves_path": str(curves_path), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path), "parameter_sha256": payload["parameter_sha256"],
        "elapsed_seconds": time.time() - started, "peak_gpu_memory_bytes": peak_memory,
    }
    atomic_write_json(result_path, report)
    atomic_write_json(out_dir / "progress.json", report)
    if not resumable:
        last_path.unlink(missing_ok=True)
    return report
