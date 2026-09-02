"""T0 numerical / gradient-path diagnostics and training-card inputs (design §6, §8).

Every function returns a JSON-serialisable dict that echoes the constants it
judged against.  Nothing here reads a development-evaluation anchor: all
increments are inner-validation, all fits are TRAIN.

Contract clauses (plan Task 5):
  [G1] tiny-slice overfit judged by the closed fraction of the H -> saturated NB gap;
  [G2] bank state Jacobian equals exp(-dt/tau) and the output Jacobian equals alpha*W/scale;
  [G3] optimizer membership fails on any unassigned / duplicated parameter or a buffer in a group;
  [G4] the AMP audit is ``skipped`` without CUDA, never faked;
  [G5] the shift null re-uses v0.3.2 ``block_circular_donor`` (same segment, >= one horizon away);
  [G6] synthetic recovery = blocked inner-val gain CI_low > 0 AND shifted worse than correct;
  [G7] thresholds are constants echoed into every report;  [G8] no dev_test anywhere.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci  # re-use
from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v032_model.state import anchor_states
from src.topic5_group_event_state.v033_evaluator import canonical as C

from .data import DataView
from .models import FlexibleResidualStateModel
from .objective import Trainable
from .paths import atomic_write_json
from .synthetic import SOURCE as SYNTHETIC_SOURCE, plant_residual_signal
from .trainer import RecipeConfig, load_trained, train_recipe

TINY_OVERFIT_THRESHOLD = 0.5
AMP_RATIO_RANGE = (1e-2, 1e2)
JACOBIAN_REL_TOL = 1e-4
OUTPUT_JACOBIAN_ABS_TOL = 1e-5
MODULATION_NONZERO_ABS = 1e-3
BOOT = dict(block_len=6, n_boot=1000, seed=0)
T0_FORMAT = "group_event_state_v0_3_3_training_lab_t0"


def _paired_ci(values: np.ndarray, segments: np.ndarray) -> dict[str, Any]:
    out = block_bootstrap_mean_ci(np.asarray(values, dtype=np.float64), np.asarray(segments), **BOOT)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def _terms(trainable: Trainable, view: DataView, model: nn.Module, phase: str, device: torch.device, *,
           state_override: Tensor | None = None):
    model.eval()
    with torch.no_grad():
        return trainable.loss_terms(model, view, phase, device=device, differentiable_statistics=False,
                                    sampling="anchor_balanced", lookback_seconds=float(max(model.arch.taus_seconds)),
                                    state_override=state_override)


def _saturated_nll(y: np.ndarray, log_r: np.ndarray) -> float:
    total = np.zeros(y.shape[0])
    for b in range(y.shape[1]):
        yb = torch.from_numpy(y[:, b].astype(np.float32))
        mu = torch.clamp(yb, min=1e-8)
        total += C.nb_nll_torch(yb, torch.log(mu), torch.tensor(float(log_r[b]))).numpy()
    return float(total.mean())


# ------------------------------------------------------------ tiny overfit
def tiny_slice_overfit(trainable: Trainable, view: DataView, cfg: RecipeConfig, seed: int, *, device: torch.device,
                       n_slice: int = 12, steps: int = 300, threshold: float = TINY_OVERFIT_THRESHOLD,
                       out_dir: Path | None = None) -> dict[str, Any]:
    """[G1] Fit ``n_slice`` contiguous TRAIN anchors of one segment with no regularisation."""

    train = view.phase_index["train"]
    seg = view.anchor_segment[train]
    seg_ids, seg_counts = np.unique(seg, return_counts=True)
    members = train[seg == seg_ids[int(np.argmax(seg_counts))]]
    members = members[np.argsort(view.t_anchor[members], kind="stable")][: int(n_slice)]
    if members.size < 2:
        raise ValueError("tiny-slice overfit needs at least two TRAIN anchors in one segment")
    slice_view = replace(view, phase_index={"train": members, "inner_val": view.phase_index["inner_val"]})
    over = cfg.with_overrides(
        weight_decay=0.0, arch=replace(cfg.arch, dropout=0.0), max_steps=int(steps), min_steps=int(steps),
        patience=10 ** 9, validate_every=max(int(steps) // 10, 1),
        schedule="constant" if cfg.schedule == "plateau" else cfg.schedule,
    )
    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="v033_tiny_overfit_")
        out_dir = Path(tmp.name)
    try:
        result = train_recipe(trainable, slice_view, over, seed, device=device, out_dir=Path(out_dir), overwrite=True)
    finally:
        if tmp is not None:
            tmp.cleanup()
    if result["status"] != "complete":
        return {"pass": False, "status": result["status"], "reason": result.get("reason"), "threshold": threshold,
                "n_slice": int(members.size), "steps": int(steps), "slice_anchor_idx": members.tolist()}
    history = result["history"]
    nll_h = float(history[-1]["train_nll_h"])
    nll_end = float(history[-1]["train_nll"])
    nll_start = float(history[0]["train_nll"])
    saturated = _saturated_nll(slice_view.counts[members], np.asarray(result["final_log_r"], dtype=np.float64))
    gap_closed = (nll_h - nll_end) / max(nll_h - saturated, 1e-9)
    return {
        "pass": bool(gap_closed >= threshold), "gap_closed": float(gap_closed), "threshold": float(threshold),
        "n_slice": int(members.size), "steps": int(steps), "nll_h": nll_h, "nll_start": nll_start,
        "nll_end": nll_end, "nll_saturated": saturated, "slice_anchor_idx": members.tolist(),
        "slice_segment": int(seg_ids[int(np.argmax(seg_counts))]), "config_hash": over.config_hash(),
        "first_active_step": result["first_active_step"],
        "definition": "gap_closed = (NLL_H - NLL_end) / (NLL_H - NLL_saturated) on the slice; "
                      "no weight decay, no dropout, constant/cosine LR, TRAIN slice only",
    }


# ---------------------------------------------------------- oracle head
def oracle_head_fit(trainable: Trainable, view: DataView, cfg: RecipeConfig, true_state: np.ndarray, seed: int, *,
                    device: torch.device, steps: int = 300, lr: float = 1e-2) -> dict[str, Any]:
    """Level 0: the true state is given, only the output head is trained (TRAIN), scored on inner-val."""

    z = np.asarray(true_state, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    train, val = view.phase_index["train"], view.phase_index["inner_val"]
    mean, std = z[train].mean(axis=0), z[train].std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    zs = torch.from_numpy(((z - mean) / std).astype(np.float32)).to(device)
    torch.manual_seed(int(seed))
    head = nn.Linear(z.shape[1], view.n_bins, bias=False).to(device)
    alpha = nn.Parameter(torch.tensor(float(cfg.arch.alpha_init), device=device))
    log_r = torch.tensor(view.log_r_h, dtype=torch.float32, device=device)
    log_r = nn.Parameter(log_r, requires_grad=cfg.dispersion == "low_lr")
    params = [p for p in (*head.parameters(), alpha, log_r) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(lr))
    log_mu_h = torch.from_numpy(np.nan_to_num(view.log_mu_h, nan=0.0).astype(np.float32)).to(device)
    y = torch.from_numpy(view.counts.astype(np.float32)).to(device)
    idx_train = torch.from_numpy(train).to(device)
    idx_val = torch.from_numpy(val).to(device)

    def nll(idx: Tensor) -> Tensor:
        log_mu = log_mu_h[idx] + alpha * head(zs[idx])
        per_bin = torch.stack([
            C.nb_nll_torch(y[idx, b], log_mu[:, b], log_r[b]) for b in range(view.n_bins)
        ], dim=1)
        return per_bin.sum(dim=1)

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = nll(idx_train).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_nll = nll(idx_val).cpu().numpy().astype(np.float64)
        train_nll = float(nll(idx_train).mean())
    h = trainable.h_only_nll(view, "inner_val")
    gain = _paired_ci(h - val_nll, view.bootstrap_segment(val))
    return {
        "pass": bool(gain["ci_low"] > 0), "gain_h_minus_oracle": gain, "inner_val_nll_oracle": float(val_nll.mean()),
        "inner_val_nll_h": float(h.mean()), "train_nll_oracle": train_nll, "steps": int(steps), "lr": float(lr),
        "true_state_dim": int(z.shape[1]), "final_alpha": float(alpha.detach()),
        "definition": "Level 0 oracle: TRAIN-standardised true state, trained W/alpha head, inner-val paired gain vs H",
    }


# --------------------------------------------------------------- jacobians
def state_write_jacobian(trainable: Trainable, view: DataView, model: FlexibleResidualStateModel, *,
                         device: torch.device) -> dict[str, Any]:
    """[G2] d(anchor state)/d(event write) and d(log mu)/d(raw state) against their closed forms."""

    model = model.to(device).eval()
    t = trainable.tensors(view, device)
    train = view.phase_index["train"]
    candidates = [int(a) for a in train if view.last_event_pos[a] >= 0]
    if not candidates:
        raise ValueError("no TRAIN anchor with a preceding event")
    a = candidates[len(candidates) // 2]
    j_last = int(view.last_event_pos[a])
    j = j_last - 3 if j_last - 3 >= 0 and view.event_segment[j_last - 3] == view.event_segment[j_last] else j_last
    with torch.enable_grad():
        u = model.writes(t["x"]).detach().requires_grad_(True)
        _pre, post = model.state(u, t["times"], t["segment"])
        s_a = anchor_states(post, t["times"], t["t_anchor"][[a]], t["last_event_pos"][[a]], model.state.taus_full)[0]
        d = int(s_a.numel())
        jac = torch.zeros(d, u.shape[1], dtype=torch.float64)
        for k in range(d):
            g = torch.autograd.grad(s_a[k], u, retain_graph=k < d - 1)[0]
            jac[k] = g[j].detach().cpu().double()
    dt = float(view.t_anchor[a] - view.event_times[j])
    bank: dict[str, Any] | None = None
    if model.arch.state_family == "fixed_leaky":
        width = int(model.arch.write_width)
        taus_full = model.state.taus_full.detach().cpu().double()
        expected = torch.zeros_like(jac)
        for k in range(d):
            expected[k, k % width] = math.exp(-dt / float(taus_full[k]))
        rel = float((jac - expected).abs().max() / expected.abs().max())
        bank = {"max_relative_error": rel, "tolerance": JACOBIAN_REL_TOL, "pass": bool(rel < JACOBIAN_REL_TOL),
                "dt_seconds": dt, "anchor_idx": a, "event_idx": j}
    finite = bool(torch.isfinite(jac).all())
    nonzero = bool(jac.abs().max() > 0)
    with torch.enable_grad():
        state = torch.randn(1, d, device=device, requires_grad=True)
        log_mu = model.log_mu(torch.zeros(1, model.n_bins, device=device), state)
        rows = [torch.autograd.grad(log_mu[0, b], state, retain_graph=b < model.n_bins - 1)[0][0] for b in range(model.n_bins)]
        auto = torch.stack(rows).detach().cpu()
    closed = model.modulation_jacobian().cpu()
    out_err = float((auto - closed).abs().max())
    output = {"max_abs_error": out_err, "tolerance": OUTPUT_JACOBIAN_ABS_TOL, "pass": bool(out_err < OUTPUT_JACOBIAN_ABS_TOL),
              "jacobian_norm": float(closed.norm())}
    passed = finite and nonzero and output["pass"] and (bank is None or bank["pass"])
    return {"pass": bool(passed), "bank": bank, "state": {"finite": finite, "nonzero": nonzero, "max_abs": float(jac.abs().max()),
                                                          "state_family": model.arch.state_family},
            "output": output, "definition": "autograd d S_a / d u_j vs exp(-(t_a - t_j)/tau); d log mu / d S vs alpha W / scale"}


# ------------------------------------------------------------- membership
def optimizer_membership(model: nn.Module, groups: list[dict[str, Any]]) -> dict[str, Any]:
    """[G3] Every requires_grad parameter in exactly one group; buffers in none."""

    names = {id(p): n for n, p in model.named_parameters()}
    assigned: dict[int, list[str]] = {}
    for g in groups:
        for p in g["params"]:
            assigned.setdefault(id(p), []).append(g["name"])
    unassigned = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in assigned]
    duplicates = [names.get(i, "?") for i, gs in assigned.items() if len(gs) > 1]
    buffers = [n for n, b in model.named_buffers() if id(b) in assigned]
    frozen = [names.get(i, "?") for i in assigned if not next((p for p in model.parameters() if id(p) == i)).requires_grad]
    return {"pass": bool(not unassigned and not duplicates and not buffers), "unassigned": unassigned,
            "duplicates": duplicates, "buffers_in_groups": buffers, "frozen_listed": frozen,
            "groups": {g["name"]: int(sum(p.numel() for p in g["params"])) for g in groups},
            "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


# ------------------------------------------------------------------- AMP
def amp_small_gradient_check(trainable: Trainable, view: DataView, cfg: RecipeConfig, seed: int, *,
                             device: torch.device) -> dict[str, Any]:
    """[G4] bf16 autocast on the encoder only: per-group gradient norm ratio against FP32."""

    if device.type != "cuda":
        return {"status": "skipped", "pass": None, "reason": "AMP audit needs a CUDA device", "ratio_range": AMP_RATIO_RANGE}

    def norms(amp: bool) -> dict[str, float]:
        model = trainable.build(cfg.arch, view, seed).to(device)
        model.amp_encoder = amp
        if cfg.dispersion == "frozen":
            model.adapter.log_r.requires_grad_(False)
        groups = trainable.param_groups(model, cfg.effective_lr(), cfg.weight_decay)
        model.train()
        terms = trainable.loss_terms(model, view, "train", device=device, differentiable_statistics=True,
                                     sampling=cfg.sampling, lookback_seconds=cfg.lookback_seconds)
        loss = (terms.nll * terms.weights).sum() / terms.weights.sum()
        loss.backward()
        return {g["name"]: float(torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in g["params"] if p.grad is not None)))
                if any(p.grad is not None for p in g["params"]) else 0.0 for g in groups}

    fp32, amp = norms(False), norms(True)
    ratio = {n: (amp[n] / fp32[n] if fp32[n] > 0 else None) for n in fp32}
    zero_under_amp = [n for n in fp32 if fp32[n] > 0 and amp[n] == 0.0]
    out_of_range = [n for n, r in ratio.items() if r is not None and not (AMP_RATIO_RANGE[0] <= r <= AMP_RATIO_RANGE[1])]
    finite = all(math.isfinite(v) for v in amp.values())
    return {"status": "complete", "pass": bool(finite and not zero_under_amp and not out_of_range),
            "fp32_grad_norm": fp32, "amp_grad_norm": amp, "ratio_amp_over_fp32": ratio,
            "zero_under_amp": zero_under_amp, "out_of_range": out_of_range, "ratio_range": AMP_RATIO_RANGE,
            "definition": "one full-batch TRAIN forward/backward, encoder-only bf16 autocast vs FP32"}


# -------------------------------------------------------------- modulation
def state_output_modulation(trainable: Trainable, view: DataView, model: FlexibleResidualStateModel, *,
                            device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for phase in ("train", "inner_val"):
        terms = _terms(trainable, view, model, phase, device)
        mod = terms.modulation.detach().cpu().numpy().astype(np.float64)
        mean_state = model.train_mean_state.to(terms.state_raw.dtype).unsqueeze(0).expand_as(terms.state_raw)
        mean_terms = _terms(trainable, view, model, phase, device, state_override=mean_state)
        out[phase] = {
            "modulation_rms": float(np.sqrt(np.mean(mod ** 2))),
            "modulation_std_per_bin": mod.std(axis=0).tolist(),
            "fraction_abs_modulation_gt_threshold": float(np.mean(np.abs(mod) > MODULATION_NONZERO_ABS)),
            "threshold": MODULATION_NONZERO_ABS,
            "nll_dynamic": float(terms.nll.mean()), "nll_mean_state": float(mean_terms.nll.mean()),
            "nll_mean_state_minus_dynamic": float(mean_terms.nll.mean() - terms.nll.mean()),
        }
    out["pass"] = bool(out["train"]["modulation_rms"] > 0 and out["inner_val"]["modulation_rms"] > 0)
    out["definition"] = "modulation = log mu(H+S) - log mu(H); mean-state arm replaces S by the TRAIN mean state"
    return out


def state_variance_rank(trainable: Trainable, view: DataView, model: FlexibleResidualStateModel, *,
                        device: torch.device) -> dict[str, Any]:
    terms = _terms(trainable, view, model, "train", device)
    s_std = terms.state_std.detach().cpu().numpy().astype(np.float64)
    s_raw = terms.state_raw.detach().cpu().numpy().astype(np.float64)
    cov = np.atleast_2d(np.cov(s_std, rowvar=False))
    eig = np.clip(np.sort(np.linalg.eigvalsh(cov))[::-1], 0.0, None)
    total = float(eig.sum())
    pr = float(total ** 2 / max(float((eig ** 2).sum()), 1e-12)) if total > 0 else 0.0
    w = model.adapter.W.weight.detach().cpu().numpy()
    return {"state_dim": int(s_std.shape[1]), "n_anchors": int(s_std.shape[0]),
            "covariance_eigenvalues": eig.tolist(), "participation_ratio": pr,
            "fraction_variance_top1": float(eig[0] / total) if total > 0 else 0.0,
            "readout_rank": int(np.linalg.matrix_rank(w)), "readout_dim": int(w.shape[0]),
            "temporal_variance_raw": s_raw.var(axis=0).tolist(), "temporal_variance_std": s_std.var(axis=0).tolist(),
            "definition": "TRAIN anchor states standardised by TRAIN stats; participation ratio (sum l)^2 / sum l^2"}


# --------------------------------------------------------------- shift null
def shift_null(trainable: Trainable, view: DataView, model: FlexibleResidualStateModel, *, device: torch.device,
               fraction: float = 0.5) -> dict[str, Any]:
    """[G5] Same-segment block-circular donor at least one horizon away (v0.3.2 rule)."""

    idx = view.phase_index["inner_val"]
    correct = _terms(trainable, view, model, "inner_val", device)
    donor = block_circular_donor(view.t_anchor, view.anchor_segment, idx, horizon=view.horizon, fraction=float(fraction))
    ok = donor >= 0
    state = correct.state_raw
    shifted_state = state.clone()
    if ok.any():
        src = torch.from_numpy(donor[ok]).to(device)
        dst = torch.from_numpy(np.flatnonzero(ok)).to(device)
        shifted_state[dst] = state[src]
    shifted = _terms(trainable, view, model, "inner_val", device, state_override=shifted_state)
    nll_c = correct.nll.detach().cpu().numpy().astype(np.float64)
    nll_s = shifted.nll.detach().cpu().numpy().astype(np.float64)
    delta = nll_s - nll_c
    delta[~ok] = np.nan
    return {"fraction": float(fraction), "n_anchors": int(idx.size), "n_valid_donors": int(ok.sum()),
            "delta_shifted_minus_correct": _paired_ci(delta, view.bootstrap_segment(idx)),
            "nll_correct_on_valid": float(nll_c[ok].mean()) if ok.any() else float("nan"),
            "nll_shifted_on_valid": float(nll_s[ok].mean()) if ok.any() else float("nan"),
            "definition": "inner-val anchor state replaced by a same-segment donor >= one horizon away; positive favours correct time"}


def blocked_inner_val_gain(trainable: Trainable, view: DataView, model: FlexibleResidualStateModel, *,
                           device: torch.device) -> dict[str, Any]:
    idx = view.phase_index["inner_val"]
    terms = _terms(trainable, view, model, "inner_val", device)
    nll_model = terms.nll.detach().cpu().numpy().astype(np.float64)
    h = trainable.h_only_nll(view, "inner_val")
    ci = _paired_ci(h - nll_model, view.bootstrap_segment(idx))
    return {**ci, "n_anchors": int(idx.size), "n_blocks": int(np.unique(view.blocks("inner_val")).size),
            "effective_independent_windows": view.effective_independent_windows("inner_val"),
            "nll_h_mean": float(h.mean()), "nll_model_mean": float(nll_model.mean()),
            "definition": "per-anchor H-only NLL minus model NLL on inner-val, within-segment moving-block bootstrap"}


def random_reservoir_delta(trainable: Trainable, view: DataView, cfg: RecipeConfig, seed: int, *, device: torch.device,
                           out_dir: Path, learned_dir: Path) -> dict[str, Any]:
    random_dir = Path(out_dir) / "random_reservoir"
    random = train_recipe(trainable, view, cfg, seed, device=device, out_dir=random_dir, arm="random_reservoir")
    keep = ("arm", "status", "selected_step", "best_validation", "n_steps_run", "stopped_reason", "config_hash")
    summary = {k: random.get(k) for k in keep}
    if random["status"] != "complete":
        return {"status": random["status"], "random_result": summary, "learned_minus_random": None}
    learned_model = load_trained(Path(learned_dir), trainable, view, device)
    random_model = load_trained(random_dir, trainable, view, device)
    idx = view.phase_index["inner_val"]
    nll_l = _terms(trainable, view, learned_model, "inner_val", device).nll.cpu().numpy().astype(np.float64)
    nll_r = _terms(trainable, view, random_model, "inner_val", device).nll.cpu().numpy().astype(np.float64)
    return {"status": "complete", "random_result": summary, "learned_inner_val_nll": float(nll_l.mean()),
            "random_inner_val_nll": float(nll_r.mean()),
            "learned_minus_random": _paired_ci(nll_l - nll_r, view.bootstrap_segment(idx)),
            "definition": "same recipe with a frozen random encoder; negative delta = trained encoder helps"}


def synthetic_recovery(trainable: Trainable, view: DataView, cfg: RecipeConfig, seed: int, *, device: torch.device,
                       out_dir: Path, beta: float = 0.7, dispersion_r: float = 5.0, generator_seed: int | None = None,
                       noise_seed: int | None = None) -> dict[str, Any]:
    """[G6] Recipe recovers a planted residual signal (inner-val CI_low > 0) with correct-time specificity."""

    gs = 1000 + int(seed) if generator_seed is None else int(generator_seed)
    ns = 2000 + int(seed) if noise_seed is None else int(noise_seed)
    planted, info = plant_residual_signal(view, beta=beta, dispersion_r=dispersion_r, generator_seed=gs, noise_seed=ns)
    run_dir = Path(out_dir) / "synthetic_recovery"
    result = train_recipe(trainable, planted, cfg, seed, device=device, out_dir=run_dir)
    header = {"source": SYNTHETIC_SOURCE, "beta": float(beta), "dispersion_r": float(dispersion_r), "generator_seed": gs,
              "noise_seed": ns, "r2_hidden_vs_baseline_train": float(info["r2_hidden_vs_baseline_train"]),
              "rule": "blocked inner-val gain CI_low > 0 AND mean(shifted - correct) > 0"}
    if result["status"] != "complete":
        return {**header, "pass": False, "status": result["status"], "reason": result.get("reason")}
    model = load_trained(run_dir, trainable, planted, device)
    gain = blocked_inner_val_gain(trainable, planted, model, device=device)
    shift = shift_null(trainable, planted, model, device=device)
    passed = bool(gain["ci_low"] > 0 and shift["delta_shifted_minus_correct"]["mean"] > 0)
    keep = ("selected_step", "selected_in_warmup", "selected_at_budget_edge", "n_steps_run", "stopped_reason",
            "best_validation", "all_groups_active_before_selection")
    return {**header, "pass": passed, "status": "complete", "blocked_inner_val_gain": gain, "shift_null": shift,
            "train": {k: result.get(k) for k in keep}}


# --------------------------------------------------------------------- T0
def run_t0(trainable: Trainable, view: DataView, cfg: RecipeConfig, seed: int, *, device: torch.device, out_dir: Path,
           tiny_steps: int = 300, probe_steps: int = 50, n_slice: int = 12, true_state: np.ndarray | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model0 = trainable.build(cfg.arch, view, seed).to(device)
    if cfg.dispersion == "frozen":
        model0.adapter.log_r.requires_grad_(False)
    membership = optimizer_membership(model0, trainable.param_groups(model0, cfg.effective_lr(), cfg.weight_decay))
    jacobian = state_write_jacobian(trainable, view, model0, device=device)
    tiny = tiny_slice_overfit(trainable, view, cfg, seed, device=device, n_slice=n_slice, steps=tiny_steps,
                              out_dir=out_dir / "tiny_overfit")
    amp = amp_small_gradient_check(trainable, view, cfg, seed, device=device)
    probe_cfg = cfg.with_overrides(max_steps=int(probe_steps), min_steps=int(probe_steps), patience=10 ** 9,
                                   validate_every=max(int(probe_steps) // 5, 1))
    probe = train_recipe(trainable, view, probe_cfg, seed, device=device, out_dir=out_dir / "probe", overwrite=True)
    if probe["status"] != "complete":
        modulation: dict[str, Any] = {"pass": False, "status": probe["status"]}
        rank: dict[str, Any] = {"status": probe["status"]}
        grad_update: dict[str, Any] = {}
    else:
        model = load_trained(out_dir / "probe", trainable, view, device)
        modulation = state_output_modulation(trainable, view, model, device=device)
        rank = state_variance_rank(trainable, view, model, device=device)
        first, last = probe["history"][0], probe["history"][-1]
        grad_update = {"first_validation": {"step": first["step"], "grad_norm_by_group": first["grad_norm_by_group"],
                                            "update_norm_by_group": first["update_norm_by_group"]},
                       "last_validation": {"step": last["step"], "grad_norm_by_group": last["grad_norm_by_group"],
                                           "update_norm_by_group": last["update_norm_by_group"]}}
    oracle = (oracle_head_fit(trainable, view, cfg, true_state, seed, device=device) if true_state is not None
              else {"status": "skipped", "pass": None, "reason": "no true state available (human view)"})
    report = {
        "format": T0_FORMAT, "subject": view.subject, "seed": int(seed), "config_hash": cfg.config_hash(),
        "split_hash": view.split_hash, "input_hash": view.input_hash, "h_source": view.h_source,
        "optimizer_membership": membership, "tiny_slice_overfit": tiny, "state_write_jacobian": jacobian,
        "amp_small_gradient": amp, "state_output_modulation": modulation, "state_variance_rank": rank,
        "oracle_head_fit": oracle, "first_active_step": probe.get("first_active_step"),
        "clipping_fraction": probe.get("clipping_fraction"), "gradient_update_norms": grad_update,
        "probe": {k: probe.get(k) for k in ("status", "n_steps_run", "selected_step", "best_validation", "config_hash")},
        "gradient_path_ok": bool(membership["pass"] and tiny["pass"] and jacobian["pass"] and amp["pass"] in (True, None)
                                 and modulation.get("pass", False)),
        "evidence_label": "DIAGNOSTIC", "development_evaluation_read": False, "sealed_partition_opened": False,
    }
    atomic_write_json(out_dir / "t0.json", report)
    return report
