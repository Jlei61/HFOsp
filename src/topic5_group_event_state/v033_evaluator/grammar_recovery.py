"""Bounded synthetic-only recovery laboratory for the S_G Level-2 assay.

This module deliberately does not expose a human-data entry point.  It uses a
frozen :class:`Scaffold`, replaces every target with D0/D3 synthetic truth, and
asks where the grammar cascade loses the planted state.  The small recipe set
is fixed in source so an assay result cannot silently grow into a hyperparameter
sweep.

Two diagnostic batches are kept separate:

* T0 changes optimiser budget/schedule while retaining the historical,
  non-nested Level-2 head.
* T1 first restores the scientific nesting contract: the calibrated H-only
  contact grammar is copied into the state arm and frozen, so the trainable
  model can only add a state-dependent residual.  Capacity and nuisance-input
  isolation are then each changed once.

The ``marks_only`` recipe is diagnostic and is never eligible to become the
selected full-input recipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import resource
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from src.topic5_group_event_state.v032_model.state import anchor_states, leaky_bank_trajectory

from . import canonical as C
from . import oracle as O
from .dgp import SyntheticData
from .scaffold import Scaffold


@dataclass(frozen=True)
class GrammarRecoveryRecipe:
    name: str
    batch: str
    lr: float
    schedule: str
    max_steps: int
    hidden_dim: int = 16
    write_dim: int = 2
    feature_mode: str = "marks_plus_scaffold"
    nested_h_baseline: bool = False
    validate_every: int = 10
    min_steps: int = 50
    patience_checks: int = 10
    grad_clip: float = 1.0
    improvement_tol: float = 1e-7

    def validate(self) -> "GrammarRecoveryRecipe":
        if self.batch not in ("T0", "T1"):
            raise ValueError("batch must be T0 or T1")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("schedule must be constant or cosine")
        if self.feature_mode not in ("marks_plus_scaffold", "marks_only"):
            raise ValueError("unknown feature_mode")
        if self.lr <= 0 or self.max_steps < 1 or self.hidden_dim < 1 or self.write_dim < 1:
            raise ValueError("invalid positive recipe parameter")
        if self.validate_every < 1 or self.min_steps < 1 or self.patience_checks < 1:
            raise ValueError("invalid validation/early-stop parameter")
        return self

    @property
    def eligible_for_selection(self) -> bool:
        return self.feature_mode == "marks_plus_scaffold"


# Predeclared before any new recovery result was inspected.  The historical
# 120-step recipe remains in oracle.py and is the audit reference, not a grid
# cell.  T0 tests optimisation.  T1 tests correct residual nesting, then one
# capacity increase and one synthetic mark-isolation diagnostic.
RECIPES: dict[str, GrammarRecoveryRecipe] = {
    "t0_lr3e3_constant": GrammarRecoveryRecipe(
        name="t0_lr3e3_constant", batch="T0", lr=3e-3, schedule="constant", max_steps=600,
    ),
    "t0_lr1e2_cosine": GrammarRecoveryRecipe(
        name="t0_lr1e2_cosine", batch="T0", lr=1e-2, schedule="cosine", max_steps=600,
    ),
    "t1_nested_h16_w2_full": GrammarRecoveryRecipe(
        name="t1_nested_h16_w2_full", batch="T1", lr=3e-3, schedule="constant", max_steps=600,
        nested_h_baseline=True,
    ),
    "t1_nested_h64_w4_full": GrammarRecoveryRecipe(
        name="t1_nested_h64_w4_full", batch="T1", lr=3e-3, schedule="constant", max_steps=600,
        hidden_dim=64, write_dim=4, nested_h_baseline=True,
    ),
    "t1_nested_h16_w2_marks_only": GrammarRecoveryRecipe(
        name="t1_nested_h16_w2_marks_only", batch="T1", lr=3e-3, schedule="constant", max_steps=600,
        feature_mode="marks_only", nested_h_baseline=True,
    ),
}


def recipe(name: str) -> GrammarRecoveryRecipe:
    try:
        return RECIPES[name].validate()
    except KeyError as exc:
        raise ValueError(f"unknown recipe {name!r}; allowed={tuple(RECIPES)}") from exc


class GrammarLevel2Model(nn.Module):
    def __init__(self, in_dim: int, n_contacts: int, cfg: GrammarRecoveryRecipe,
                 h_only_logits: np.ndarray) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(int(in_dim), int(cfg.hidden_dim)), nn.GELU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.write_dim)),
        )
        self.register_buffer("taus", torch.tensor(list(O.BANK_TAUS), dtype=torch.float32))
        self.register_buffer(
            "taus_full",
            torch.tensor(list(O.BANK_TAUS), dtype=torch.float32).repeat_interleave(int(cfg.write_dim)),
        )
        self.a = nn.Parameter(torch.as_tensor(np.asarray(h_only_logits), dtype=torch.float32).clone())
        self.w = nn.Parameter(torch.zeros(int(n_contacts), len(O.BANK_TAUS) * int(cfg.write_dim)))
        if cfg.nested_h_baseline:
            self.a.requires_grad_(False)

    def anchor_state(self, x: torch.Tensor, times: torch.Tensor, codes: torch.Tensor,
                     t_anchor: torch.Tensor, last: torch.Tensor, train_rows: torch.Tensor) -> torch.Tensor:
        writes = torch.tanh(self.encoder(x))
        _pre, post = leaky_bank_trajectory(
            writes, times, codes, self.taus, chunk_seconds=3600.0,
        )
        state = anchor_states(post, times, t_anchor, last, self.taus_full).to(torch.float64)
        ref = state[train_rows]
        scale = ref.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        return (state - ref.mean(dim=0)) / scale


def _features(scaffold: Scaffold, data: SyntheticData, cfg: GrammarRecoveryRecipe) -> tuple[np.ndarray, str]:
    if data.marks is None:
        raise ValueError("S_G recovery requires a visible synthetic mark channel")
    marks = np.asarray(data.marks, dtype=np.float64)
    if cfg.feature_mode == "marks_only":
        return marks, "visible_synthetic_marks_only"
    return np.column_stack([marks, O._real_tokens(scaffold)]), "visible_synthetic_marks_plus_frozen_scaffold_tokens"


def _lr_factor(cfg: GrammarRecoveryRecipe, step: int) -> float:
    if cfg.schedule == "constant":
        return 1.0
    progress = min(max(float(step) / float(cfg.max_steps), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _weighted_grammar_loss(model: GrammarLevel2Model, states: torch.Tensor,
                           participation: torch.Tensor,
                           pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    anchor_rows, event_rows, weight = pairs
    logits = model.a.to(torch.float64)[None, :] + states[anchor_rows] @ model.w.to(torch.float64).T
    return -(weight * O.conditional_bernoulli_logpmf_torch(logits, participation[event_rows])).sum()


def _fit_h_only_selected(participation: torch.Tensor,
                         train_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         select_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         initial_logits: np.ndarray, *, max_steps: int = 600,
                         validate_every: int = 10, patience_checks: int = 10,
                         lr: float = 0.05) -> dict[str, Any]:
    """Fit the H-only intercept on TRAIN and select its step on inner-validation.

    The historical helper returns its last iterate, so simply increasing the
    Level-2 budget from 120 to 600 silently makes H itself less calibrated.
    This helper freezes that second optimisation axis before a state residual
    is considered.
    """

    a = nn.Parameter(torch.as_tensor(np.asarray(initial_logits), dtype=torch.float64,
                                     device=participation.device).clone())
    opt = torch.optim.Adam([a], lr=float(lr))

    def loss(pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _owner, event, weight = pairs
        logits = a[None, :].expand(event.numel(), -1)
        return -(weight * O.conditional_bernoulli_logpmf_torch(logits, participation[event])).sum()

    best, best_step, best_a, stale = math.inf, -1, None, 0
    curve = []
    for step in range(1, int(max_steps) + 1):
        opt.zero_grad()
        train = loss(train_pairs)
        train.backward()
        opt.step()
        if step % int(validate_every) == 0 or step == int(max_steps):
            with torch.no_grad():
                inner = float(loss(select_pairs))
            curve.append({"step": int(step), "train_nll": float(train.detach()), "inner_nll": inner})
            if inner < best - 1e-7:
                best, best_step, best_a, stale = inner, int(step), a.detach().cpu().numpy().copy(), 0
            else:
                stale += 1
            if step >= 50 and stale >= int(patience_checks):
                break
    if best_a is None:
        raise RuntimeError("H-only grammar calibration produced no checkpoint")
    return {"a": best_a, "selected_step": best_step, "selected_inner_nll": float(best),
            "steps_run": int(curve[-1]["step"]), "curve": curve}


def _pairs(scaffold: Scaffold, rows: np.ndarray, horizon: float,
           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    owner, event, weight = O._window_pairs(scaffold, rows, horizon)
    norm = max(float(weight.sum()), 1e-12)
    return (
        torch.from_numpy(owner.astype(np.int64)).to(device),
        torch.from_numpy(event.astype(np.int64)).to(device),
        torch.from_numpy(weight / norm).to(device, torch.float64),
    )


def _truth_alignment(states: np.ndarray, truth: np.ndarray, rows: np.ndarray) -> dict[str, float | None]:
    if rows.size < 3 or states.shape[1] == 0:
        return {"max_abs_correlation": None, "median_abs_correlation": None}
    z = np.asarray(truth, dtype=np.float64)[rows]
    vals = []
    for j in range(states.shape[1]):
        s = states[rows, j]
        if np.std(s) > 1e-9 and np.std(z) > 1e-9:
            vals.append(abs(float(np.corrcoef(s, z)[0, 1])))
    return {
        "max_abs_correlation": max(vals) if vals else None,
        "median_abs_correlation": float(np.median(vals)) if vals else None,
    }


def run_recovery(
    scaffold: Scaffold,
    data: SyntheticData,
    *,
    cfg: GrammarRecoveryRecipe,
    horizon: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train one predeclared recipe and score only synthetic D0/D3 targets."""

    cfg.validate()
    if data.kind not in ("D0", "D3"):
        raise ValueError("bounded S_G recovery accepts D0 or D3 only")
    if data.marks is None:
        raise ValueError("visible synthetic mark channel is required")
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    train_rows_np = scaffold.anchor_rows(O.TRAIN_PHASE, horizon)
    select_rows_np = scaffold.anchor_rows(O.SELECT_PHASE, horizon)
    score_rows_np = scaffold.anchor_rows(O.EVALUATION_PHASE, horizon)
    train_events = scaffold.event_rows(O.TRAIN_PHASE)
    x_np, input_contract = _features(scaffold, data, cfg)
    x = torch.from_numpy(O._standardise_columns(x_np, train_events)).to(device, torch.float32)
    times = torch.from_numpy(scaffold.event_times).to(device, torch.float64)
    codes = torch.from_numpy(O._monotone_codes(scaffold.event_carry)).to(device, torch.long)
    t_anchor = torch.from_numpy(scaffold.t_anchor).to(device, torch.float64)
    last = torch.from_numpy(scaffold.last_event_pos).to(device, torch.long)
    train_rows = torch.from_numpy(train_rows_np).to(device, torch.long)
    participation = torch.from_numpy(data.participation).to(device)
    train_pairs = _pairs(scaffold, train_rows_np, horizon, device)
    select_pairs = _pairs(scaffold, select_rows_np, horizon, device)

    # H-only calibration is fitted on exactly the same TRAIN anchor/event
    # weighting and selected on the same inner-validation phase.  It is both
    # the scoring control and, for T1, the immutable intercept of the nested
    # residual state arm.
    owner_np, event_np, weight_np = O._window_pairs(scaffold, train_rows_np, horizon)
    h_fit = _fit_h_only_selected(
        participation, train_pairs, select_pairs,
        O._base_logits(data.participation[train_events]),
        max_steps=cfg.max_steps, validate_every=cfg.validate_every,
        patience_checks=cfg.patience_checks,
    )
    h_head = {"a": h_fit["a"], "W": np.zeros((scaffold.n_contacts, 0), dtype=np.float64)}
    model = GrammarLevel2Model(x.shape[1], scaffold.n_contacts, cfg, h_head["a"]).to(device)
    if not cfg.nested_h_baseline:
        # Historical Level-2 did not start from the calibrated H head.
        with torch.no_grad():
            model.a.copy_(torch.from_numpy(O._base_logits(data.participation[train_events])).to(device, torch.float32))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(cfg.lr))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best = math.inf
    best_step = -1
    best_state = None
    stale = 0
    curve: list[dict[str, float | int]] = []
    first_encoder_grad = None
    for step in range(1, cfg.max_steps + 1):
        factor = _lr_factor(cfg, step)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * factor
        model.train()
        opt.zero_grad()
        states = model.anchor_state(x, times, codes, t_anchor, last, train_rows)
        loss = _weighted_grammar_loss(model, states, participation, train_pairs)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite TRAIN loss at step {step}")
        loss.backward()
        enc_grad = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum())
                                 for p in model.encoder.parameters() if p.grad is not None))
        if first_encoder_grad is None and enc_grad > 0:
            first_encoder_grad = int(step)
        total_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        opt.step()
        if step % cfg.validate_every == 0 or step == cfg.max_steps:
            model.eval()
            with torch.no_grad():
                states = model.anchor_state(x, times, codes, t_anchor, last, train_rows)
                select = float(_weighted_grammar_loss(model, states, participation, select_pairs))
            row = {"step": int(step), "train_nll": float(loss.detach()), "inner_nll": select,
                   "lr": float(cfg.lr * factor), "grad_norm_before_clip": total_norm,
                   "encoder_grad_norm": enc_grad}
            curve.append(row)
            if select < best - cfg.improvement_tol:
                best, best_step, stale = select, int(step), 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if step >= cfg.min_steps and stale >= cfg.patience_checks:
                break
    if best_state is None:
        raise RuntimeError("no finite checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        states = model.anchor_state(x, times, codes, t_anchor, last, train_rows).cpu().numpy()

    logits_h = O.grammar_logits(h_head, np.zeros((score_rows_np.size, 0)))
    head = {"a": model.a.detach().cpu().numpy().astype(np.float64),
            "W": model.w.detach().cpu().numpy().astype(np.float64)}
    logits_hs = O.grammar_logits(head, states[score_rows_np])
    scores = O._grammar_block_scores(
        scaffold, data, horizon=horizon, rows=score_rows_np, logits_h=logits_h, logits_hs=logits_hs,
    )
    table = C.build_per_anchor_table_from_scores(
        subject=f"synthetic_{data.kind}_{scaffold.subject}", seed=seed,
        checkpoint_hash=f"sg-recovery:{cfg.name}", split=O.EVALUATION_PHASE,
        anchor_time=scaffold.t_anchor[score_rows_np], target=scores["n_future"],
        per_anchor_nll={"H": scores["block_H"], "H_plus_state": scores["block_H_plus_state"]},
        score_family="conditional_subset_nll", mask=scores["n_future"] > 0, weight=None,
        eligibility="synthetic_n_future_positive", evidence_label="DIAGNOSTIC_SYNTHETIC_ASSAY_ONLY",
        extra_nll={"first_H": scores["first_H"], "first_H_plus_state": scores["first_H_plus_state"]},
    )
    gain = C.paired_gain(table)
    boot = O._bootstrap(table, O._blocks(scaffold, score_rows_np, horizon), seed)
    first = C.paired_gain(table, control="first_H", treated="first_H_plus_state")
    peak_cuda = 0.0
    if device.type == "cuda":
        peak_cuda = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    return {
        "format": "group_event_state_v0_3_3_sg_level2_recovery_card",
        "recipe": asdict(cfg), "recipe_selection_eligible": cfg.eligible_for_selection,
        "kind": data.kind, "subject_role": "frozen_scaffold_only", "scaffold_subject": scaffold.subject,
        "generator_seed": int(data.generator_seed), "noise_seed": int(data.noise_seed),
        "estimator_seed": int(seed), "inputs": input_contract,
        "human_targets_used": False, "development_human_targets_read": False,
        "seizure_outcomes_used": False, "sealed_partition_opened": False,
        "score_target": "synthetic_contact_subsets_only", "score_phase": O.EVALUATION_PHASE,
        "selection_phase": O.SELECT_PHASE, "horizon_seconds": float(horizon),
        "n_train_anchors": int(train_rows_np.size), "n_inner_anchors": int(select_rows_np.size),
        "n_score_anchors": int(score_rows_np.size), "n_train_pairs": int(event_np.size),
        "h_only_inner_nll": float(h_fit["selected_inner_nll"]),
        "h_calibration": h_fit,
        "selected_step": best_step, "selected_inner_nll": float(best), "steps_run": int(curve[-1]["step"]),
        "stopped_by_patience": bool(curve[-1]["step"] < cfg.max_steps),
        "first_encoder_gradient_step": first_encoder_grad, "training_curve": curve,
        "gain_level2": float(gain["gain"]), "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
        "detected": bool(boot["ci_lower"] is not None and boot["ci_lower"] > O.DETECTION_FLOOR_NATS),
        "gain_first_future_event": float(first["gain"]), "n_blocks": int(boot["n_blocks"]),
        "n_rows_used": int(boot["n_rows_used"]),
        "truth_alignment": {
            "train": _truth_alignment(states, data.z_grammar_anchor, train_rows_np),
            "inner": _truth_alignment(states, data.z_grammar_anchor, select_rows_np),
            "score": _truth_alignment(states, data.z_grammar_anchor, score_rows_np),
        },
        "head_contract": "frozen_H_intercept_plus_state_residual" if cfg.nested_h_baseline
                         else "historical_joint_recalibration_plus_state",
        "resources": {"wall_seconds": float(time.perf_counter() - started),
                      "peak_rss_mib": peak_rss, "peak_cuda_allocated_mib": peak_cuda,
                      "device": str(device)},
    }


def select_full_input_recipe(cards: list[dict[str, Any]], *, tolerance: float = 2e-3) -> dict[str, Any]:
    """Select on D3 replicate-0 inner NLL only; prefer the simpler nested model within tolerance."""

    eligible = [c for c in cards if c.get("kind") == "D3" and c.get("recipe_selection_eligible")]
    if not eligible:
        raise ValueError("no full-input D3 tuning cards")
    best = min(float(c["selected_inner_nll"]) for c in eligible)
    near = [c for c in eligible if float(c["selected_inner_nll"]) <= best + float(tolerance)]
    order = {"t1_nested_h16_w2_full": 0, "t0_lr3e3_constant": 1,
             "t0_lr1e2_cosine": 2, "t1_nested_h64_w4_full": 3}
    chosen = min(near, key=lambda c: order.get(c["recipe"]["name"], 99))
    return {
        "selection_metric": "synthetic_D3_replicate0_inner_nll",
        "selection_does_not_read_score_phase": True,
        "tolerance": float(tolerance), "best_inner_nll": best,
        "selected_recipe": chosen["recipe"]["name"],
        "eligible_candidates": [{"recipe": c["recipe"]["name"],
                                 "selected_inner_nll": c["selected_inner_nll"]} for c in eligible],
    }
