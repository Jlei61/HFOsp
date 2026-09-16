#!/usr/bin/env python3
"""Run one formal H1 unit using the FIT/INNER-selected recipe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_group_event_state_v037_optimizer_search import BASELINE_RECIPES, RECIPES
from src.topic5_group_event_state.v037 import CheckpointEntry, update_checkpoint_registry
from src.topic5_group_event_state.v037.contracts import atomic_json, sha256_file
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_dual_train import H1DualTrainConfig, train_h1_dual_subject
from src.topic5_group_event_state.v037.h1_train import H1TrainConfig, train_h1_subject


FAMILIES = {
    "event": "S_event_v037_inner_selected",
    "grid": "S_grid_v037_inner_selected",
    "dual": "S_dual_v037_inner_selected",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--horizons-hours", nargs="+", type=float, default=(0.5, 2.0, 6.0, 8.0),
        help="Shared producer horizons; every head uses the same frozen state trajectory.",
    )
    parser.add_argument("--search-summary", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/optimizer_search/summary.json"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_optimized"))
    args = parser.parse_args()
    summary = json.loads(args.search_summary.read_text(encoding="utf-8"))
    if summary["models"][args.model].get("selection_status") != "TRAINABLE_RECIPE_SELECTED":
        raise RuntimeError(
            f"{args.model}: optimizer search did not identify a trainable recipe; "
            "formal selection scoring is not released"
        )
    # The control arm is released under the same gate as the model arm.  A
    # formal selection score computed against a baseline that never left its
    # initialisation cannot separate "no extra information" from "not fitted".
    baseline = summary.get("baseline", {})
    if baseline.get("baseline_selection_status") != "TRAINABLE_BASELINE_RECIPE_SELECTED":
        raise RuntimeError(
            "transparent-baseline sweep did not identify a trainable recipe "
            f"(status={baseline.get('baseline_selection_status')}); formal selection "
            "scoring is not released because the control arm would be unfitted"
        )
    baseline_recipe = baseline["selected_recipe"]
    recipe = summary["models"][args.model]["selected_recipe"]
    values = dict(RECIPES[recipe])
    values.update(BASELINE_RECIPES[baseline_recipe])
    out = args.out_root / args.model / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        card = json.loads((out / "card.json").read_text(encoding="utf-8"))
    else:
        horizons_seconds = tuple(float(value) * 3600.0 for value in args.horizons_hours)
        data = build_h1_subject_data(
            args.subject, seed=int(args.seed), horizons_seconds=horizons_seconds,
        )
        if args.model == "dual":
            medium = int(values["burden_channels_per_tau"]) > 2
            config = replace(
                H1DualTrainConfig(seed=int(args.seed)), **values,
                lr_background_state=float(values["lr_state"]),
                lr_background_head=float(values["lr_state_head"]),
                warmup_steps_background=int(values["warmup_steps_state"]),
                background_channels_per_tau=4 if medium else 2,
            )
            card = train_h1_dual_subject(data, config, device=torch.device(args.device), out_dir=out)
        else:
            config = replace(H1TrainConfig(seed=int(args.seed)), **values)
            card = train_h1_subject(
                data, config, device=torch.device(args.device), out_dir=out,
                state_mode=args.model, grid_seconds=300.0,
            )
        card["optimizer_search"] = {
            "recipe": recipe, "baseline_recipe": baseline_recipe, "parameters": values,
            "baseline_selection_status": baseline["baseline_selection_status"],
            "summary": str(args.search_summary),
            "selection_partition": "INNER", "selection_scores_used": False,
        }
    card["dependency_provenance"] = {
        "h1_train.py": sha256_file(
            ROOT / "src/topic5_group_event_state/v037/h1_train.py"
        ),
        "reason": (
            "records shared fixed-history and event-state query alignment code; "
            "required to distinguish the event-free-segment zero-history fix"
        ),
    }
    atomic_json(out / "card.json", card)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hash = sha256_file(ROOT / (
        "src/topic5_group_event_state/v037/h1_dual_train.py" if args.model == "dual" else
        "src/topic5_group_event_state/v037/h1_train.py"
    ))
    streams = (
        ("fixed_clock_non_event_background", "group_event_burden", "group_event_conditional_grammar")
        if args.model == "dual" else
        (("five_minute_group_event_burden", "five_minute_conditional_grammar") if args.model == "grid" else
         ("group_event_burden", "group_event_conditional_grammar"))
    )
    update_checkpoint_registry(
        args.out_root / "registry/checkpoint_registry.json",
        CheckpointEntry(
            key=f"{FAMILIES[args.model]}::{args.subject}::seed{args.seed}",
            model_family=FAMILIES[args.model], state_semantics="observer",
            subject=args.subject, seed=int(args.seed), checkpoint_path=card["checkpoint_path"],
            maximum_training_time=float(card["maximum_training_time"]),
            input_streams=streams,
            objectives=("future_burden", "conditional_grammar_multi_horizon"),
            code_commit=f"{commit}+source_sha256:{source_hash}",
            normalization_provenance=(
                f"recipe {recipe} selected on FIT/INNER only; representation and scales fit before INNER; "
                f"shared_horizons_hours={tuple(float(v) for v in args.horizons_hours)}"
            ),
        ),
    )
    print({"model": args.model, "recipe": recipe, "baseline_recipe": baseline_recipe,
           "subject": args.subject, "seed": args.seed,
           "horizons_hours": tuple(float(v) for v in args.horizons_hours)})


if __name__ == "__main__":
    main()
