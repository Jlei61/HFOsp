#!/usr/bin/env python3
"""One FIT/INNER-only optimizer/capacity-search unit for v0.3.7 H1."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_dual_train import H1DualTrainConfig, train_h1_dual_subject
from src.topic5_group_event_state.v037.h1_train import H1TrainConfig, train_h1_subject
from src.topic5_group_event_state.v037.contracts import atomic_json


RECIPES = {
    "base_small": dict(lr_state=3e-4, lr_state_head=1e-3, weight_decay=1e-4,
                       warmup_steps_state=0, state_readout_init_std=1e-2,
                       burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "low_lr_small": dict(lr_state=1e-4, lr_state_head=5e-4, weight_decay=1e-4,
                         warmup_steps_state=0, state_readout_init_std=1e-2,
                         burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "high_lr_small": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                          warmup_steps_state=0, state_readout_init_std=1e-2,
                          burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "high_lr_warm_small": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                               warmup_steps_state=100, state_readout_init_std=1e-2,
                               burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "no_decay_warm_small": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=0.0,
                                warmup_steps_state=100, state_readout_init_std=1e-2,
                                burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "strong_decay_warm_small": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-3,
                                    warmup_steps_state=100, state_readout_init_std=1e-2,
                                    burden_channels_per_tau=2, grammar_channels_per_tau=3),
    "warm_medium": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                        warmup_steps_state=100, state_readout_init_std=1e-2,
                        burden_channels_per_tau=4, grammar_channels_per_tau=6),
    "warm_medium_low_init": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                                 warmup_steps_state=100, state_readout_init_std=1e-3,
                                 burden_channels_per_tau=4, grammar_channels_per_tau=6),
    "warm_medium_high_init": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                                  warmup_steps_state=100, state_readout_init_std=5e-2,
                                  burden_channels_per_tau=4, grammar_channels_per_tau=6),
    "responsive_medium": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                               warmup_steps_state=100, state_readout_init_std=1e-2,
                               burden_channels_per_tau=4, grammar_channels_per_tau=6,
                               adam_beta1=0.8, adam_beta2=0.98, adam_epsilon=1e-8),
    "long_budget_medium": dict(lr_state=1e-3, lr_state_head=3e-3, weight_decay=1e-4,
                                warmup_steps_state=100, state_readout_init_std=1e-2,
                                burden_channels_per_tau=4, grammar_channels_per_tau=6,
                                max_steps_state=3600),
}


# The transparent multi-scale baseline is the arm every learned state must beat.
# In the first round it was left at one untested setting and stopped at step 0 in
# 102 of 126 searched cells, which made "the state beats the baseline"
# uninterpretable.  It now gets its own recipe grid and its own gate.  The
# baseline stage is fitted after the rate stage and updates only its own
# parameters, so it is swept without paying for a state fit.
# Both control stages are swept.  The rate stage is the innermost control: when
# it stops at its 900-step budget edge (all six E253 pilot runs did) every arm
# nested above it is fitted on top of an unconverged foundation, and a gain
# credited to the mark bank may be nothing but rate information the rate stage
# failed to reach.
BASELINE_RECIPES = {
    "baseline_legacy": dict(lr_q=3e-3, max_steps_q=900,
                            lr_bmark=2e-3, warmup_steps_bmark=0,
                            bmark_readout_init_std=0.0, max_steps_bmark=1200,
                            weight_decay_bmark=1e-4),
    "baseline_warm_init": dict(lr_q=3e-3, max_steps_q=3600,
                               lr_bmark=2e-3, warmup_steps_bmark=100,
                               bmark_readout_init_std=1e-2, max_steps_bmark=1800,
                               weight_decay_bmark=1e-4),
    "baseline_low_lr": dict(lr_q=1e-3, max_steps_q=3600,
                            lr_bmark=5e-4, warmup_steps_bmark=100,
                            bmark_readout_init_std=1e-2, max_steps_bmark=1800,
                            weight_decay_bmark=1e-4),
    "baseline_high_lr": dict(lr_q=8e-3, max_steps_q=3600,
                             lr_bmark=8e-3, warmup_steps_bmark=100,
                             bmark_readout_init_std=1e-2, max_steps_bmark=1800,
                             weight_decay_bmark=1e-4),
    "baseline_strong_decay": dict(lr_q=3e-3, max_steps_q=3600,
                                  lr_bmark=2e-3, warmup_steps_bmark=100,
                                  bmark_readout_init_std=1e-2, max_steps_bmark=1800,
                                  weight_decay_bmark=1e-2),
    "baseline_no_decay_long": dict(lr_q=3e-3, max_steps_q=7200,
                                   lr_bmark=2e-3, warmup_steps_bmark=100,
                                   bmark_readout_init_std=1e-2, max_steps_bmark=3600,
                                   weight_decay_bmark=0.0),
}

ALL_RECIPES = {**RECIPES, **BASELINE_RECIPES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("event", "grid", "dual", "bmark"), required=True)
    parser.add_argument("--recipe", choices=tuple(ALL_RECIPES), required=True)
    parser.add_argument("--baseline-recipe", choices=tuple(BASELINE_RECIPES), default=None)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/optimizer_search"))
    args = parser.parse_args()
    out = args.out_root / args.model / args.recipe / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        print(f"already complete: {out}")
        return
    if (args.model == "bmark") != (args.recipe in BASELINE_RECIPES):
        raise SystemExit("model 'bmark' requires a baseline recipe and vice versa")
    if args.model != "bmark" and args.baseline_recipe is None:
        raise SystemExit(
            "learned-model search requires the already frozen --baseline-recipe; "
            "state hyperparameters must be selected on top of the fitted control"
        )
    data = build_h1_subject_data(args.subject, seed=int(args.seed))
    if args.model == "bmark":
        config = replace(H1TrainConfig(seed=int(args.seed)), **BASELINE_RECIPES[args.recipe])
        card = train_h1_subject(
            data, config, device=torch.device(args.device), out_dir=out,
            state_mode="event", grid_seconds=300.0, stages_only=("q", "bmark"),
        )
    elif args.model == "dual":
        values = {**BASELINE_RECIPES[args.baseline_recipe], **RECIPES[args.recipe]}
        medium = int(values["burden_channels_per_tau"]) > 2
        config = replace(
            H1DualTrainConfig(seed=int(args.seed)), **values,
            lr_background_state=float(values["lr_state"]),
            lr_background_head=float(values["lr_state_head"]),
            warmup_steps_background=int(values["warmup_steps_state"]),
            background_channels_per_tau=4 if medium else 2,
        )
        card = train_h1_dual_subject(
            data, config, device=torch.device(args.device), out_dir=out,
        )
    else:
        values = {**BASELINE_RECIPES[args.baseline_recipe], **RECIPES[args.recipe]}
        config = replace(H1TrainConfig(seed=int(args.seed)), **values)
        card = train_h1_subject(
            data, config, device=torch.device(args.device), out_dir=out,
            state_mode=args.model, grid_seconds=300.0,
        )
    card["optimizer_search"] = {
        "model_recipe": args.recipe if args.model != "bmark" else None,
        "frozen_baseline_recipe": args.recipe if args.model == "bmark" else args.baseline_recipe,
        "selection_partition": "INNER", "selection_scores_used": False,
    }
    atomic_json(out / "card.json", card)
    # This file makes explicit that recipe selection may inspect INNER only.
    (out / "search_contract.json").write_text(json.dumps({
        "model": args.model,
        "recipe": args.recipe,
        "frozen_baseline_recipe": args.recipe if args.model == "bmark" else args.baseline_recipe,
        "recipe_parameters": ALL_RECIPES[args.recipe],
        "selection_fields_allowed": [
            "stages.state.* for event/grid",
            "stages.background_state.* and stages.event.* for dual",
            "stages.bmark.* for the transparent-baseline sweep",
        ],
        "selection_scores_forbidden_for_recipe_choice": True,
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reported = {"event": "state", "grid": "state", "dual": "event", "bmark": "bmark"}[args.model]
    print({"subject": card["subject"], "model": args.model, "recipe": args.recipe,
           "best_inner": card["stages"][reported]["best_inner_loss"]})


if __name__ == "__main__":
    main()
