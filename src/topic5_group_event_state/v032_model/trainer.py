"""Full-batch negative-binomial residual trainer (design §4, plan Task 6).

One optimizer step = one pass over every state-train anchor of the primary
horizon.  The state trajectory is recomputed from the segment start at every
step and at every validation, so no stale trajectory can ever be scored
(``detach_replay_audit`` re-checks this from the saved checkpoint).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ModelConfig, load_config
from .data import SubjectBundle
from .model import ResidualStateModel, build_model
from .paths import atomic_write_json, atomic_write_torch, file_hash, source_commit, repo_root
from .readout import fit_nb_log_dispersion, nb_log_prob

CHECKPOINT_FORMAT = "group_event_state_v0_3_2_model_checkpoint"
RESULT_FORMAT = "group_event_state_v0_3_2_model_train_result"
ARMS = ("learned", "random_reservoir")


# --------------------------------------------------------------------------- tensors
def bundle_tensors(bundle: SubjectBundle, device: torch.device) -> dict[str, Tensor]:
    return {
        "x_std": torch.from_numpy(np.ascontiguousarray(bundle.x_std)).to(device, torch.float32),
        "times": torch.from_numpy(bundle.event_times).to(device, torch.float64),
        "segment": torch.from_numpy(bundle.event_segment).to(device, torch.long),
        "train_event_mask": torch.from_numpy(bundle.train_event_mask()).to(device),
        "t_anchor": torch.from_numpy(bundle.t_anchor).to(device, torch.float64),
        "last_event_pos": torch.from_numpy(bundle.last_event_pos).to(device, torch.long),
    }


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def anchor_terms(
    model: ResidualStateModel,
    bundle: SubjectBundle,
    *,
    phase: str,
    horizon: float,
    device: torch.device,
    tensors: dict[str, Tensor] | None = None,
    state_override: Tensor | None = None,
    return_trajectory: bool = False,
) -> dict[str, Any]:
    """Per-anchor NB terms of ``H+S`` for one nested phase (FP32 outputs)."""

    t = tensors or bundle_tensors(bundle, device)
    idx = np.flatnonzero(bundle.anchor_mask(phase, horizon))
    h_i = bundle.horizon_index(horizon)
    pre, post = model.trajectory(t["x_std"], t["times"], t["segment"])
    idx_t = torch.from_numpy(idx).to(device)
    state = model.anchor_states(post, t["times"], t["t_anchor"][idx_t], t["last_event_pos"][idx_t])
    if state_override is not None:
        state = state_override.to(state.dtype)
    log_mu_h = torch.from_numpy(bundle.log_mu_h(horizon)[idx]).to(device, torch.float32)
    log_mu = model.log_mu(log_mu_h, state)
    y = torch.from_numpy(bundle.counts[idx, h_i]).to(device, torch.float32)
    nll = -nb_log_prob(y, torch.exp(log_mu), model.adapter.log_r)
    out = {"idx": idx, "state": state.to(torch.float32), "log_mu": log_mu, "nll": nll,
           "log_mu_h": log_mu_h, "y": y, "modulation": (log_mu - log_mu_h)}
    if return_trajectory:
        out["state_pre"] = pre
        out["state_post"] = post
    return out


def h_only_nll(bundle: SubjectBundle, *, phase: str, horizon: float, log_r_h: float) -> np.ndarray:
    idx = np.flatnonzero(bundle.anchor_mask(phase, horizon))
    h_i = bundle.horizon_index(horizon)
    y = torch.from_numpy(bundle.counts[idx, h_i]).float()
    mu = torch.exp(torch.from_numpy(bundle.log_mu_h(horizon)[idx]).float())
    return (-nb_log_prob(y, mu, torch.tensor(float(log_r_h)))).numpy()


def resolve_log_r_h(bundle: SubjectBundle, horizon: float) -> tuple[float, str]:
    """H-only NB dispersion: registry value if present, else TRAIN-anchor MLE."""

    given = bundle.history.nb_log_dispersion.get(int(horizon))
    if given is not None and math.isfinite(float(given)):
        return float(given), "history_registry"
    idx = np.flatnonzero(bundle.anchor_mask("state_train", horizon))
    h_i = bundle.horizon_index(horizon)
    value = fit_nb_log_dispersion(bundle.counts[idx, h_i], np.exp(bundle.log_mu_h(horizon)[idx]))
    return float(value), "state_train_mle"


# --------------------------------------------------------------------------- gradients
def _group_grad_norms(groups: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in groups:
        squares = [p.grad.detach().float().pow(2).sum() for p in group["params"] if p.grad is not None]
        out[group["name"]] = float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0
    return out


def _global_grad_norm(params: list[torch.nn.Parameter]) -> float:
    squares = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0


# --------------------------------------------------------------------------- checkpoint
def load_checkpoint_model(path: Path, *, in_dim: int, device: torch.device) -> ResidualStateModel:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = load_config(None, **payload["config"])
    model = ResidualStateModel(cfg, in_dim, log_r_init=0.0)
    model.load_state_dict(payload["model_state"], strict=True)
    if payload.get("encoder_frozen"):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return model.to(device).eval()


@torch.no_grad()
def train_mean_anchor_state(
    model: ResidualStateModel, bundle: SubjectBundle, *, horizon: float, device: torch.device,
    tensors: dict[str, Tensor] | None = None,
) -> Tensor:
    terms = anchor_terms(model, bundle, phase="state_train", horizon=horizon, device=device, tensors=tensors)
    return terms["state"].mean(dim=0)


# --------------------------------------------------------------------------- training
def train_residual_model(
    bundle: SubjectBundle,
    cfg: ModelConfig,
    seed: int,
    *,
    device: torch.device,
    out_dir: Path,
    arm: str = "learned",
    overwrite: bool = False,
    interrupt_after_step: int | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    result_path = out_dir / "result.json"
    last_path = out_dir / "last.pt"
    config_hash = cfg.config_hash()
    if result_path.exists() and checkpoint_path.exists() and not overwrite:
        previous = json.loads(result_path.read_text())
        if previous.get("config_hash") == config_hash and previous.get("status") == "complete":
            previous["skipped_existing"] = True
            return previous

    started = time.time()
    horizon = float(cfg.horizon_seconds)
    h_i = bundle.horizon_index(horizon)
    train_idx = np.flatnonzero(bundle.anchor_mask("state_train", horizon))
    val_idx = np.flatnonzero(bundle.anchor_mask("dev_val", horizon))
    base_report = {
        "format": RESULT_FORMAT,
        "subject": bundle.subject,
        "seed": int(seed),
        "architecture": cfg.architecture,
        "arm": arm,
        "config": cfg.as_dict(),
        "config_hash": config_hash,
        "h_source": bundle.history.source,
        "fingerprint": bundle.fingerprint,
        "n_train_anchors": int(train_idx.size),
        "n_val_anchors": int(val_idx.size),
        "n_train_events": int(bundle.train_event_mask().sum()),
        "effective_independent_windows": {
            "state_train": bundle.effective_independent_windows("state_train", horizon),
            "dev_val": bundle.effective_independent_windows("dev_val", horizon),
        },
        "source_commit": source_commit(repo_root()),
        "development_test_used_for_selection": False,
        "sealed_partition_opened": False,
    }
    if train_idx.size == 0 or val_idx.size == 0:
        report = {**base_report, "status": "insufficient_anchors",
                  "reason": "no eligible state_train or dev_val anchor at the primary horizon; "
                            "this is a coverage limit, not a scientific result"}
        atomic_write_json(result_path, report)
        return report

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    tensors = bundle_tensors(bundle, device)
    log_r_h, log_r_h_source = resolve_log_r_h(bundle, horizon)
    model = build_model(cfg, in_dim=bundle.x_std.shape[1], log_r_init=log_r_h, seed=seed).to(device)
    encoder_frozen = arm == "random_reservoir"
    if encoder_frozen:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    groups = model.param_groups(cfg)
    optimizer_groups = [
        {k: v for k, v in g.items() if k != "params"} | {"params": [p for p in g["params"] if p.requires_grad]}
        for g in groups
    ]
    optimizer_groups = [g for g in optimizer_groups if g["params"]]
    optimizer = torch.optim.AdamW(optimizer_groups)
    trainable = [p for p in model.parameters() if p.requires_grad]
    init_state = copy.deepcopy(model.state_dict())

    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    best_row: dict[str, Any] | None = None
    stale = 0
    start_step = 0
    resumed_from: int | None = None
    if last_path.exists() and not overwrite:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("config_hash") == config_hash and saved.get("arm") == arm and saved.get("seed") == int(seed):
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            history = list(saved["history"])
            best_value, best_step, stale = saved["best_value"], saved["best_step"], saved["stale"]
            best_state = saved.get("best_state")
            best_row = saved.get("best_row")
            start_step = int(saved["step"])
            resumed_from = start_step
            init_state = saved["init_state"]
    h_val_nll = float(h_only_nll(bundle, phase="dev_val", horizon=horizon, log_r_h=log_r_h).mean())
    h_train_nll = float(h_only_nll(bundle, phase="state_train", horizon=horizon, log_r_h=log_r_h).mean())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _save_last(step: int) -> None:
        atomic_write_torch(last_path, {
            "format": CHECKPOINT_FORMAT + "_last",
            "config_hash": config_hash, "arm": arm, "seed": int(seed), "step": int(step),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "history": history, "best_value": best_value, "best_step": best_step, "stale": stale,
            "best_state": best_state, "best_row": best_row, "init_state": init_state,
        })

    stopped_reason = "max_steps"
    step = start_step
    for step in range(start_step + 1, cfg.max_steps + 1):
        model.train()
        model.adapter.set_alpha_trainable(step > cfg.alpha_freeze_steps)
        model.refresh_train_mean(tensors["x_std"], tensors["train_event_mask"])
        terms = anchor_terms(model, bundle, phase="state_train", horizon=horizon, device=device, tensors=tensors)
        loss = terms["nll"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_by_group = _group_grad_norms(groups)
        pre_clip = _global_grad_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        post_clip = _global_grad_norm(trainable)
        if not math.isfinite(pre_clip):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        validate = step % cfg.validate_every == 0 or step == cfg.max_steps
        if validate:
            model.eval()
            model.refresh_train_mean(tensors["x_std"], tensors["train_event_mask"])
            with torch.no_grad():
                val = anchor_terms(model, bundle, phase="dev_val", horizon=horizon, device=device, tensors=tensors)
                val_nll = float(val["nll"].mean())
                train_eval = anchor_terms(model, bundle, phase="state_train", horizon=horizon,
                                          device=device, tensors=tensors)
            row = {
                "step": int(step),
                "train_nll": float(loss.detach()),
                "train_nll_post_step": float(train_eval["nll"].mean()),
                "train_nll_h": h_train_nll,
                "val_nll": val_nll,
                "val_nll_h": h_val_nll,
                "val_modulation_rms": float(val["modulation"].pow(2).mean().sqrt()),
                "alpha": float(model.adapter.alpha.detach()),
                "log_r": float(model.adapter.log_r.detach()),
                "grad_norm_pre_clip": pre_clip,
                "grad_norm_post_clip": post_clip,
                "grad_norm_by_group": grad_by_group,
                "alpha_trainable": bool(step > cfg.alpha_freeze_steps),
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            if math.isfinite(val_nll) and val_nll < best_value - 1e-6:
                best_value, best_step, stale = val_nll, int(step), 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_row = dict(row)
            else:
                stale += 1
            _save_last(step)
            atomic_write_json(out_dir / "progress.json", {**base_report, "status": "running",
                                                           "step": step, "history": history,
                                                           "best_step": best_step})
            if step >= cfg.min_steps and stale >= cfg.patience:
                stopped_reason = "patience"
                break
        if interrupt_after_step is not None and step >= int(interrupt_after_step):
            _save_last(step)
            report = {**base_report, "status": "interrupted", "step": int(step), "history": history}
            atomic_write_json(out_dir / "progress.json", report)
            return report

    if best_state is None:
        raise RuntimeError("training never produced a finite validation value")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        model.train_mean_state.copy_(
            train_mean_anchor_state(model, bundle, horizon=horizon, device=device, tensors=tensors)
        )
    validation_steps = [row["step"] for row in history]
    selected_first = bool(validation_steps and best_step == validation_steps[0])
    at_edge = bool(step >= cfg.max_steps and best_step == validation_steps[-1])
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    payload = {
        "format": CHECKPOINT_FORMAT,
        "subject": bundle.subject,
        "seed": int(seed),
        "arm": arm,
        "architecture": cfg.architecture,
        "config": cfg.as_dict(),
        "config_hash": config_hash,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "init_state": init_state,
        "standardizer": bundle.standardizer.to_dict(),
        "feature_names": list(bundle.feature_names),
        "fingerprint": bundle.fingerprint,
        "h_source": bundle.history.source,
        "h_meta": bundle.history.meta,
        "log_r_h": log_r_h,
        "log_r_h_source": log_r_h_source,
        "selected_step": int(best_step),
        "encoder_frozen": encoder_frozen,
        "horizon_seconds": horizon,
        "parameter_sha256": parameter_sha256(model),
        "source_commit": base_report["source_commit"],
    }
    atomic_write_torch(checkpoint_path, payload)
    report = {
        **base_report,
        "status": "complete",
        "selected_step": int(best_step),
        "selected_first_validation": selected_first,
        "selected_at_budget_edge": at_edge,
        "selection_metric": f"dev_val_nb_nll_h_plus_s_correct_{int(horizon)}s",
        "n_steps_run": int(step),
        "stopped_reason": stopped_reason,
        "resumed_from_step": resumed_from,
        "alpha_frozen_until_step": int(cfg.alpha_freeze_steps),
        "best_validation": {
            "step": int(best_step),
            "nll_h_plus_s_correct": float(best_value),
            "nll_h": h_val_nll,
            "gain_h_minus_h_plus_s": float(h_val_nll - best_value),
        },
        "final_alpha": float(model.adapter.alpha.detach()),
        "final_log_r": float(model.adapter.log_r.detach()),
        "log_r_h": log_r_h,
        "log_r_h_source": log_r_h_source,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "parameter_sha256": payload["parameter_sha256"],
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "device": str(device),
    }
    atomic_write_json(result_path, report)
    atomic_write_json(out_dir / "progress.json", {**report, "history": history})
    last_path.unlink(missing_ok=True)
    return report
