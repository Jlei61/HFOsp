"""T6 training card and the TRAINING-ADEQUATE rule (design §8).

A card is the *only* evidence a model recipe can carry into the science
workstream.  It states whether the network provably learned the assigned
task; it never states anything about H1 / H2.

Contract clauses (plan Task 7):
  [C1] six conditions -> TRAINING-ADEQUATE, any failure -> DIAGNOSTIC with the failing condition named;
  [C2] seed dispersion needs >= 2 seeds, otherwise flagged; [C3] every §8 field present;
  [C4] a card mentioning a development-evaluation key is refused; [C5] non-canonical by default.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .paths import current_commit

ADEQUACY_RULE = ("tiny_overfit.pass and synthetic_recovery.pass and blocked_inner_val_gain.ci_low > 0 "
                 "and not selected_in_warmup and not selected_at_budget_edge and all_groups_active_before_selection")
FORBIDDEN_KEY_FRAGMENTS = ("dev_test", "development_test", "sealed_partition_data")
CARD_FIELDS = (
    "format", "created_epoch", "request", "recipe", "config_hash", "split_hash", "input_hash", "code_commit",
    "curves", "best_step", "plateau", "seed_dispersion", "gradient_update", "clipping_fraction",
    "first_active_step", "state_variance_rank", "random_reservoir_delta", "shift_null", "output_modulation",
    "tiny_overfit", "synthetic_recovery", "blocked_inner_val_gain", "selected_in_warmup",
    "selected_at_budget_edge", "all_groups_active_before_selection", "t0", "diagnostics", "search",
    "evidence_label", "adequacy_rule", "adequacy_reasons", "adequacy_conditions",
    "selection_metric_is_canonical", "evaluator_hash", "sealed_partition_opened",
    "development_evaluation_read", "training_adequacy_is_not_a_scientific_result", "source_commit",
)
CARD_FORMAT = "group_event_state_v0_3_3_training_card"


def _walk_keys(obj: Any, found: list[str]) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            found.append(str(k))
            _walk_keys(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_keys(v, found)


def assert_card_has_no_dev_test(card: Mapping[str, Any]) -> None:
    """[C4] Refuse any card that carries a development-evaluation number."""

    keys: list[str] = []
    _walk_keys(card, keys)
    bad = sorted({k for k in keys if any(f in k.lower() for f in FORBIDDEN_KEY_FRAGMENTS)})
    if bad:
        raise ValueError(f"training card must not carry development-evaluation keys: {bad}")


def _stat(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "range": None}
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "min": float(arr.min()), "max": float(arr.max()), "range": float(arr.max() - arr.min())}


def seed_dispersion(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """[C2] Spread of best inner-val NLL / gain / selected step across seeds of one recipe."""

    complete = [r for r in seed_results if r.get("status") == "complete"]
    best = [r.get("best_validation") or {} for r in complete]
    return {
        "n_seeds": len(complete), "insufficient_seeds": len(complete) < 2,
        "seeds": [int(r.get("seed")) for r in complete],
        "inner_val_nll": _stat([b.get("inner_val_nll") for b in best]),
        "gain_h_minus_model": _stat([b.get("gain_h_minus_model") for b in best]),
        "selected_step": _stat([r.get("selected_step") for r in complete]),
        "selected_at_budget_edge_any": any(bool(r.get("selected_at_budget_edge")) for r in complete),
        "selected_in_warmup_any": any(bool(r.get("selected_in_warmup")) for r in complete),
    }


def adequacy(card: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """[C1] Evaluate the six conditions; every failed one is named."""

    conditions = {
        "tiny_overfit": bool((card.get("tiny_overfit") or {}).get("pass")),
        "synthetic_recovery": bool((card.get("synthetic_recovery") or {}).get("pass")),
        "blocked_inner_val_gain": (card.get("blocked_inner_val_gain") or {}).get("ci_low") is not None
                                  and float(card["blocked_inner_val_gain"]["ci_low"]) > 0.0,
        "selected_in_warmup": not bool(card.get("selected_in_warmup")),
        "selected_at_budget_edge": not bool(card.get("selected_at_budget_edge")),
        "all_groups_active_before_selection": bool(card.get("all_groups_active_before_selection")),
    }
    reasons = [f"{name} condition failed" for name, ok in conditions.items() if not ok]
    label = "TRAINING-ADEQUATE" if not reasons else "DIAGNOSTIC"
    return label, {"conditions": conditions, "reasons": reasons, "rule": ADEQUACY_RULE}


def build_training_card(
    *,
    request: Mapping[str, Any] | None,
    recipe_result: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    t0: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    search_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    history = list(recipe_result.get("history") or [])
    first, last = (history[0] if history else {}), (history[-1] if history else {})
    curves = {
        "n_validations": len(history), "curves_path": recipe_result.get("curves_path"),
        "train_nll_first": first.get("train_nll"), "train_nll_last": last.get("train_nll"),
        "inner_val_nll_first": first.get("inner_val_nll"), "inner_val_nll_last": last.get("inner_val_nll"),
        "inner_val_nll_best": (recipe_result.get("best_validation") or {}).get("inner_val_nll"),
        "inner_val_nll_h": (recipe_result.get("best_validation") or {}).get("inner_val_nll_h"),
        "n_steps_run": recipe_result.get("n_steps_run"), "stopped_reason": recipe_result.get("stopped_reason"),
    }
    gradient_update = {
        "first_validation": {"step": first.get("step"), "grad_norm_by_group": first.get("grad_norm_by_group"),
                             "update_norm_by_group": first.get("update_norm_by_group"), "lr_by_group": first.get("lr_by_group")},
        "last_validation": {"step": last.get("step"), "grad_norm_by_group": last.get("grad_norm_by_group"),
                            "update_norm_by_group": last.get("update_norm_by_group"), "lr_by_group": last.get("lr_by_group")},
    }
    request_ref = None
    if request is not None:
        request_ref = {k: request.get(k) for k in ("request_id", "scientific_target", "input_view", "state_family",
                                                    "split_hash", "input_hash", "code_commit", "requested_by")}
    card: dict[str, Any] = {
        "format": CARD_FORMAT, "created_epoch": time.time(), "request": request_ref,
        "recipe": recipe_result.get("config"), "config_hash": recipe_result.get("config_hash"),
        "split_hash": recipe_result.get("split_hash"), "input_hash": recipe_result.get("input_hash"),
        "code_commit": recipe_result.get("source_commit"), "subject": recipe_result.get("subject"),
        "h_source": recipe_result.get("h_source"), "arm": recipe_result.get("arm"),
        "curves": curves, "best_step": recipe_result.get("selected_step"), "plateau": recipe_result.get("plateau"),
        "seed_dispersion": seed_dispersion(seed_results), "gradient_update": gradient_update,
        "clipping_fraction": recipe_result.get("clipping_fraction"),
        "first_active_step": recipe_result.get("first_active_step"),
        "state_variance_rank": diagnostics.get("state_variance_rank"),
        "random_reservoir_delta": diagnostics.get("random_reservoir_delta"),
        "shift_null": diagnostics.get("shift_null"), "output_modulation": diagnostics.get("state_output_modulation"),
        "tiny_overfit": t0.get("tiny_slice_overfit"), "synthetic_recovery": diagnostics.get("synthetic_recovery"),
        "blocked_inner_val_gain": diagnostics.get("blocked_inner_val_gain"),
        "selected_in_warmup": bool(recipe_result.get("selected_in_warmup")),
        "selected_at_budget_edge": bool(recipe_result.get("selected_at_budget_edge")),
        "all_groups_active_before_selection": bool(recipe_result.get("all_groups_active_before_selection")),
        "t0": {k: v for k, v in t0.items() if k != "tiny_slice_overfit"},
        "diagnostics": {k: v for k, v in diagnostics.items()
                        if k not in ("state_variance_rank", "random_reservoir_delta", "shift_null",
                                     "state_output_modulation", "synthetic_recovery", "blocked_inner_val_gain")},
        "search": None if search_summary is None else {k: search_summary.get(k) for k in ("incumbent", "stop_reason", "n_batches")},
        "selection_metric_is_canonical": False, "evaluator_hash": None,                     # [C5]
        "sealed_partition_opened": False, "development_evaluation_read": False,
        "training_adequacy_is_not_a_scientific_result": True, "source_commit": current_commit(),
    }
    label, detail = adequacy(card)
    card["evidence_label"] = label
    card["adequacy_rule"] = ADEQUACY_RULE
    card["adequacy_reasons"] = detail["reasons"]
    card["adequacy_conditions"] = detail["conditions"]
    assert_card_has_no_dev_test(card)
    return card
