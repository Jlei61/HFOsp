"""T1-T5: search space, seeded sampler, synchronous ASHA, seed policy, stop rules,
failure classification (design §7).

Contract clauses (plan Task 6):
  [S1] group LRs log-uniform in [1e-5, 3e-3]; every categorical value reachable; seeded;
  [S2] each rung keeps ceil(n / eta); a config whose parameter groups activated too late for the rung
       (``all_groups_active_step + validate_every > rung``) is never pruned there (``grace_deferred``);
  [S3] seeds: rung 0 -> seeds_low, rung >= 1 -> seeds_mid, final rung -> top n_final x seeds_final; scores
       compare the seed-median inner-validation NLL;
  [S4] two consecutive batches without improvement -> stop; stable plateau -> stop;
  [S5] failure classification is a fixed-priority table echoed into the JSON;
  [S6] search_trace.json records every unit; [S7] no dev_test anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .data import DataView
from .models import ArchConfig, GROUP_NAMES, TIME_BANKS
from .objective import Trainable
from .paths import atomic_write_json, payload_hash
from .trainer import RecipeConfig, train_recipe

LR_LOG_RANGE = (1e-5, 3e-3)
CATEGORICAL: dict[str, tuple[Any, ...]] = {
    "optimizer": ("adamw", "adam", "rmsprop"),
    "schedule": ("constant", "cosine", "plateau"),
    "warmup_fraction": (0.0, 0.05, 0.10),
    "init": ("xavier", "orthogonal"),
    "write_scale": (0.01, 0.1, 1.0),
    "alpha_init": (0.01, 0.03, 0.1),
    "scaling": ("zscore", "robust"),
    "hidden_norm": ("none", "layernorm"),
    "depth": (1, 2, 3),
    "width": (32, 64, 128),
    "activation": ("relu", "gelu", "silu"),
    "dropout": (0.0, 0.1),
    "write_width": (2, 4, 8),
    "time_bank": tuple(TIME_BANKS.values()),
    "dispersion": ("frozen", "low_lr"),
    "sampling": ("anchor_balanced", "event_balanced"),
}
GATED_EXTRA: dict[str, tuple[Any, ...]] = {
    "gate_bias_init": (-1.0, 0.0, 1.0),
    "tbptt_seconds": (1800.0, 3600.0, 7200.0),
}
ARCH_KEYS = ("init", "write_scale", "alpha_init", "hidden_norm", "depth", "width", "activation", "dropout",
             "write_width", "gate_bias_init", "tbptt_seconds")
FAILURE_PRIORITY = ("support_insufficient", "gradient_path", "budget_edge", "underfit", "objective_or_support",
                    "overfit_or_objective", "search_exhausted")
NEXT_ACTION = {
    "support_insufficient": "return to Agent A/C: not enough independent windows for this endpoint/horizon",
    "gradient_path": "inspect path / capacity / LR / normalisation before any further search",
    "budget_edge": "extend the step budget, then judge again",
    "underfit": "increase capacity / LR (TRAIN does not beat H)",
    "objective_or_support": "return to Agent C: synthetic recovers but human inner-val does not and the "
                            "random reservoir is equivalent -> objective / support, not optimisation",
    "overfit_or_objective": "regularise / reduce capacity; if persistent, return to Agent C for objective/support",
    "search_exhausted": "stop blind search; classify and report",
    "none": "no failure observed",
}


# ---------------------------------------------------------------- space
@dataclass
class SearchSpace:
    state_family: str
    categorical: dict[str, tuple[Any, ...]]
    lr_range: tuple[float, float] = LR_LOG_RANGE

    @classmethod
    def for_family(cls, state_family: str, *, gated_approved: bool = False,
                   restrict: Mapping[str, Sequence[Any]] | None = None) -> "SearchSpace":
        if state_family == "gated_exploratory" and not gated_approved:
            raise ValueError("gated_exploratory space needs an explicitly approved request")
        cats = dict(CATEGORICAL)
        if state_family == "gated_exploratory":
            cats.update(GATED_EXTRA)
        for key, values in (restrict or {}).items():
            if key not in cats:
                raise ValueError(f"unknown search dimension {key!r}")
            allowed = set(cats[key]) if key != "time_bank" else {tuple(v) for v in cats[key]}
            picked = tuple(tuple(v) if key == "time_bank" else v for v in values)
            bad = [v for v in picked if v not in allowed]
            if bad:
                raise ValueError(f"restriction {key}={bad} outside the pre-registered values")
            cats[key] = picked
        return cls(state_family=state_family, categorical=cats)

    def sample(self, rng: np.random.Generator, *, budget: "SearchBudget | None" = None) -> RecipeConfig:
        pick = {name: values[int(rng.integers(len(values)))] for name, values in self.categorical.items()}
        lo, hi = math.log(self.lr_range[0]), math.log(self.lr_range[1])
        lr = {g: float(math.exp(rng.uniform(lo, hi))) for g in GROUP_NAMES}
        arch = ArchConfig(
            state_family=self.state_family, taus_seconds=tuple(pick["time_bank"]),
            **{k: pick[k] for k in ARCH_KEYS if k in pick},
        )
        cfg = RecipeConfig(arch=arch, optimizer=pick["optimizer"], schedule=pick["schedule"],
                           warmup_fraction=float(pick["warmup_fraction"]), lr=lr, dispersion=pick["dispersion"],
                           sampling=pick["sampling"], scaling=pick["scaling"])
        if budget is not None:
            cfg = budget.apply(cfg)
        return cfg.validate()

    def describe(self) -> dict[str, Any]:
        return {"state_family": self.state_family, "lr_log_uniform_range": list(self.lr_range),
                "lr_groups": list(GROUP_NAMES),
                "categorical": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in self.categorical.items()}}


# --------------------------------------------------------------- budget
@dataclass(frozen=True)
class SearchBudget:
    n_configs: int
    max_steps: int
    rung_steps: tuple[int, ...]
    eta: int = 2
    seeds_low: int = 1
    seeds_mid: int = 3
    seeds_final: int = 5
    n_final: int = 2
    validate_every: int = 10
    min_steps: int = 0
    patience: int = 10 ** 9

    def __post_init__(self) -> None:
        rungs = tuple(int(r) for r in self.rung_steps)
        if not rungs or any(b <= a for a, b in zip(rungs, rungs[1:])):
            raise ValueError("rung_steps must be strictly increasing")
        if rungs[-1] != int(self.max_steps):
            raise ValueError("last rung must equal max_steps")
        if any(r < self.validate_every for r in rungs) or self.eta < 2 or self.n_configs < 1:
            raise ValueError("every rung needs at least one validation; eta >= 2; n_configs >= 1")
        if not 1 <= self.n_final:
            raise ValueError("n_final >= 1")
        object.__setattr__(self, "rung_steps", rungs)

    @classmethod
    def from_request(cls, payload: Mapping[str, Any]) -> "SearchBudget":
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        known["rung_steps"] = tuple(int(v) for v in known.get("rung_steps", (int(payload["max_steps"]),)))
        return cls(**known)

    def apply(self, cfg: RecipeConfig) -> RecipeConfig:
        return cfg.with_overrides(max_steps=int(self.max_steps), min_steps=int(self.min_steps),
                                  validate_every=int(self.validate_every), patience=int(self.patience))

    def seeds_for(self, rung_index: int) -> int:
        if rung_index == 0:
            return int(self.seeds_low)
        if rung_index == len(self.rung_steps) - 1 and len(self.rung_steps) > 1:
            return int(self.seeds_final)
        return int(self.seeds_mid)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------- ASHA
def asha_promote(rows: Sequence[Mapping[str, Any]], *, eta: int) -> dict[str, Any]:
    """[S2] Keep the best ceil(n/eta) by score (lower = better); never prune a grace-deferred config."""

    ordered = sorted(rows, key=lambda r: (not math.isfinite(float(r["score"])), float(r["score"])))
    n_keep = int(math.ceil(len(ordered) / float(eta))) if ordered else 0
    keep = [r["config_id"] for r in ordered[:n_keep]]
    deferred = [r["config_id"] for r in ordered[n_keep:] if not bool(r.get("grace_ok", True))]
    pruned = [r["config_id"] for r in ordered[n_keep:] if bool(r.get("grace_ok", True))]
    return {"n_keep": n_keep, "promoted": keep + deferred, "grace_deferred": deferred, "pruned": pruned}


def _all_groups_active_step(result: Mapping[str, Any]) -> int | None:
    active = result.get("first_active_step") or {}
    groups = result.get("optimizer_groups") or {}
    steps = [active.get(name) for name in groups]
    if any(s is None for s in steps):
        return None
    return int(max(steps)) if steps else 1


def _unit_row(config_id: str, cfg: RecipeConfig, seed: int, seed_index: int, rung_index: int, rung: int,
              result: Mapping[str, Any]) -> dict[str, Any]:
    active = _all_groups_active_step(result)
    grace_ok = active is not None and active + int(cfg.validate_every) <= int(rung)
    best = result.get("best_validation") or {}
    return {
        "config_id": config_id, "config_hash": cfg.config_hash(), "seed": int(seed), "seed_index": int(seed_index),
        "rung_index": int(rung_index), "steps_budget": int(rung), "n_steps_run": result.get("n_steps_run"),
        "status": result.get("status"), "inner_val_nll": best.get("inner_val_nll"),
        "inner_val_nll_h": best.get("inner_val_nll_h"), "gain_h_minus_model": best.get("gain_h_minus_model"),
        "selected_step": result.get("selected_step"), "selected_in_warmup": result.get("selected_in_warmup"),
        "selected_at_budget_edge": result.get("selected_at_budget_edge"), "resumed_from_step": result.get("resumed_from_step"),
        "all_groups_active_step": active, "grace_ok": bool(grace_ok), "plateau_reached": (result.get("plateau") or {}).get("reached"),
        "stopped_reason": result.get("stopped_reason"), "elapsed_seconds": result.get("elapsed_seconds"),
    }


def run_search_batch(
    trainable: Trainable,
    views: Mapping[str, DataView],
    space: SearchSpace,
    budget: SearchBudget,
    *,
    base_seed: int,
    device: torch.device | str,
    out_dir: Path,
    batch_index: int,
    incumbent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One synchronous successive-halving bracket over ``budget.n_configs`` fresh samples."""

    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(base_seed) + 1000 * int(batch_index))
    configs: dict[str, RecipeConfig] = {}
    for i in range(int(budget.n_configs)):
        cfg = space.sample(rng, budget=budget)
        configs[f"b{batch_index}_c{i:03d}"] = cfg
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    survivors = list(configs)
    started = time.time()
    per_rung_scores: dict[int, dict[str, dict[str, Any]]] = {}
    for rung_index, rung in enumerate(budget.rung_steps):
        is_final = rung_index == len(budget.rung_steps) - 1
        if is_final and len(budget.rung_steps) > 1:
            previous = per_rung_scores[rung_index - 1]
            ranked = sorted(survivors, key=lambda c: previous[c]["score"])
            survivors = ranked[: int(budget.n_final)]
        n_seeds = budget.seeds_for(rung_index)
        scores: dict[str, dict[str, Any]] = {}
        for cid in survivors:
            cfg = configs[cid]
            view = views[cfg.scaling]
            seed_rows = []
            for seed_index in range(n_seeds):
                seed = int(base_seed) + 100 * seed_index + 7 * int(batch_index)
                result = train_recipe(trainable, view, cfg, seed, device=device, out_dir=out_dir / cid / f"seed_{seed_index}",
                                      steps_budget=int(rung))
                row = _unit_row(cid, cfg, seed, seed_index, rung_index, int(rung), result)
                rows.append(row)
                seed_rows.append(row)
            finite = [r["inner_val_nll"] for r in seed_rows if r["status"] == "complete" and r["inner_val_nll"] is not None]
            score = float(np.median(finite)) if finite else float("inf")
            gains = [r["gain_h_minus_model"] for r in seed_rows if r["status"] == "complete"]
            scores[cid] = {"config_id": cid, "config_hash": cfg.config_hash(), "score": score,
                           "gain_median": float(np.median(gains)) if gains else None, "n_seeds": len(seed_rows),
                           "grace_ok": all(bool(r["grace_ok"]) for r in seed_rows),
                           "plateau_reached": any(bool(r["plateau_reached"]) for r in seed_rows),
                           "budget_edge": any(bool(r["selected_at_budget_edge"]) for r in seed_rows),
                           "rung_index": rung_index, "rung": int(rung)}
        per_rung_scores[rung_index] = scores
        if not is_final:
            decision = asha_promote(list(scores.values()), eta=budget.eta)
            decisions.append({"rung_index": rung_index, "rung": int(rung), **decision})
            survivors = decision["promoted"]
    final_scores = per_rung_scores[len(budget.rung_steps) - 1]
    best_id = min(final_scores, key=lambda c: final_scores[c]["score"])
    inc = dict(final_scores[best_id])
    inc["recipe"] = configs[best_id].as_dict()
    inc["run_dir"] = str(out_dir / best_id)
    summary = {
        "format": "group_event_state_v0_3_3_training_lab_search_batch", "batch_index": int(batch_index),
        "base_seed": int(base_seed), "budget": budget.as_dict(), "space": space.describe(),
        "configs": {cid: {"config_hash": cfg.config_hash(), "recipe": cfg.as_dict()} for cid, cfg in configs.items()},
        "rows": rows, "decisions": decisions, "scores_by_rung": {str(k): v for k, v in per_rung_scores.items()},
        "incumbent": inc, "previous_incumbent": None if incumbent is None else dict(incumbent),
        "elapsed_seconds": time.time() - started, "selection_phase": "inner_val", "development_evaluation_read": False,
    }
    atomic_write_json(out_dir / "search_trace.json", {"batch_index": int(batch_index), "rows": rows, "decisions": decisions})
    atomic_write_json(out_dir / "batch_summary.json", summary)
    return summary


def run_search(
    trainable: Trainable,
    views: Mapping[str, DataView],
    space: SearchSpace | None,
    budget: SearchBudget | None,
    *,
    base_seed: int,
    device: torch.device | str,
    out_dir: Path,
    max_batches: int = 4,
    tol: float = 1e-3,
    batch_runner: Callable[..., dict[str, Any]] = run_search_batch,
) -> dict[str, Any]:
    """[S4] Batches until two without improvement, a stable plateau, or ``max_batches``."""

    out_dir = Path(out_dir)
    incumbent: dict[str, Any] | None = None
    no_improve = 0
    batches: list[dict[str, Any]] = []
    stop_reason = "max_batches"
    for b in range(int(max_batches)):
        batch = batch_runner(trainable, views, space, budget, base_seed=base_seed, device=device,
                             out_dir=out_dir / f"batch_{b:02d}", batch_index=b, incumbent=incumbent)
        cand = dict(batch["incumbent"])
        improved = incumbent is None or float(cand["score"]) < float(incumbent["score"]) - float(tol)
        batches.append({"batch_index": b, "incumbent": cand, "improved": bool(improved),
                        "n_rows": len(batch.get("rows", []))})
        if improved:
            incumbent, no_improve = cand, 0
        else:
            no_improve += 1
        if no_improve >= 2:
            stop_reason = "no_improvement_two_batches"
            break
        if no_improve >= 1 and bool(incumbent.get("plateau_reached")):
            stop_reason = "stable_plateau"
            break
    report = {"format": "group_event_state_v0_3_3_training_lab_search", "incumbent": incumbent, "batches": batches,
              "stop_reason": stop_reason, "n_batches": len(batches), "tol": float(tol), "max_batches": int(max_batches)}
    if space is not None and budget is not None:
        atomic_write_json(out_dir / "search_summary.json", report)
    return report


# ----------------------------------------------------- failure classification
def classify_failure(obs: Mapping[str, Any]) -> dict[str, Any]:
    """[S5] Fixed-priority classification of a training outcome (design §7 table)."""

    matches: list[str] = []
    if int(obs.get("effective_windows", 10 ** 9)) < int(obs.get("min_effective_windows", 0)):
        matches.append("support_insufficient")
    if obs.get("tiny_overfit_pass") is False or obs.get("all_groups_active") is False:
        matches.append("gradient_path")
    if bool(obs.get("selected_at_budget_edge")):
        matches.append("budget_edge")
    if obs.get("train_learned") is False:
        matches.append("underfit")
    gain_low = obs.get("inner_val_gain_ci_low")
    no_gain = gain_low is not None and float(gain_low) <= 0.0
    if obs.get("train_learned") and no_gain:
        if bool(obs.get("synthetic_recovery_pass")) and bool(obs.get("random_reservoir_equivalent")):
            matches.append("objective_or_support")
        else:
            matches.append("overfit_or_objective")
    if int(obs.get("search_no_improvement_batches", 0)) >= 2:
        matches.append("search_exhausted")
    category = next((c for c in FAILURE_PRIORITY if c in matches), "none")
    return {"category": category, "all_matches": matches, "priority": list(FAILURE_PRIORITY),
            "next_action": NEXT_ACTION[category],
            "rule": "first match in priority order; objective_or_support requires synthetic recovery pass AND "
                    "random-reservoir equivalence, otherwise overfit_or_objective", "observations": dict(obs)}
