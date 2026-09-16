#!/usr/bin/env python3
"""Select model and transparent-baseline recipes using FIT/INNER diagnostics only.

The control arm is gated exactly like the model arms.  A learned state that
beats a baseline which never left its initialisation is not evidence, so a
formal H1 is released only when both families have a trainable recipe.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import statistics

from run_group_event_state_v037_optimizer_search import BASELINE_RECIPES, RECIPES


DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/optimizer_search")
SUBJECTS = (
    "epilepsiae_253", "epilepsiae_958",
    "epilepsiae_1077", "epilepsiae_1125",
)
SEEDS = (20260903, 20260904)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    frozen_baseline_path = root / "baseline_summary.json"
    if not frozen_baseline_path.exists():
        raise FileNotFoundError("baseline must be frozen before model recipe finalization")
    frozen_baseline = json.loads(frozen_baseline_path.read_text(encoding="utf-8"))
    frozen_baseline_recipe = frozen_baseline.get("selected_recipe")
    if frozen_baseline.get("status") != "TRAINABLE_BASELINE_RECIPE_SELECTED":
        raise RuntimeError("model recipe finalization is forbidden without a trainable frozen baseline")
    report = {"format": "group_event_state_v0_3_7_optimizer_search_summary_v1", "models": {}}
    for model in ("event", "grid", "dual"):
        cells = {}
        for recipe in RECIPES:
            cells[recipe] = []
            for subject in SUBJECTS:
                for seed in SEEDS:
                    card = json.loads((root / model / recipe / subject / f"seed{seed}" / "card.json").read_text())
                    stage = card["stages"]["state" if model != "dual" else "event"]
                    if card.get("optimizer_search", {}).get("frozen_baseline_recipe") != frozen_baseline_recipe:
                        raise RuntimeError(
                            f"{model}/{recipe}/{subject}/seed{seed} was not trained on the frozen "
                            f"baseline recipe {frozen_baseline_recipe}"
                        )
                    row = {
                        "subject": subject, "seed": seed,
                        "best_inner_loss": stage["best_inner_loss"],
                        "initial_inner_loss": stage["initial_inner_loss"],
                        "gain": stage["initial_inner_loss"] - stage["best_inner_loss"],
                        "selected_step": stage["selected_step"], "steps_run": stage["steps_run"],
                        "selected_at_init": stage["selected_at_init"],
                        "selected_at_budget_edge": stage["selected_at_budget_edge"],
                        "training_budget_exhausted": stage["training_budget_exhausted"],
                        "first_step_gradient_norm": stage["first_step_gradient_norm"],
                        "peak_parameter_delta": stage["peak_parameter_delta_from_stage_start"],
                    }
                    if model == "dual":
                        background = card["stages"]["background_state"]
                        row.update({
                            "background_best_inner_loss": background["best_inner_loss"],
                            "background_initial_inner_loss": background["initial_inner_loss"],
                            "background_gain": background["initial_inner_loss"] - background["best_inner_loss"],
                            "background_selected_at_init": background["selected_at_init"],
                            "background_selected_at_budget_edge": background["selected_at_budget_edge"],
                            "background_training_budget_exhausted": background["training_budget_exhausted"],
                            "background_first_step_gradient_norm": background["first_step_gradient_norm"],
                            "background_peak_parameter_delta": background["peak_parameter_delta_from_stage_start"],
                        })
                    cells[recipe].append(row)
        ranks = {recipe: [] for recipe in RECIPES}
        for subject in SUBJECTS:
            for seed in SEEDS:
                ordered = sorted(RECIPES, key=lambda r: next(
                    row["best_inner_loss"] for row in cells[r]
                    if row["subject"] == subject and row["seed"] == seed
                ))
                for rank, recipe in enumerate(ordered, 1):
                    ranks[recipe].append(rank)
        summary = {}
        for recipe, rows in cells.items():
            non_init_fraction = statistics.fmean(float(not row["selected_at_init"]) for row in rows)
            budget_exhausted_fraction = statistics.fmean(
                float(row["training_budget_exhausted"])
                for row in rows
            )
            selected_at_edge_fraction = statistics.fmean(
                float(row["selected_at_budget_edge"]) for row in rows
            )
            median_gain = statistics.median(row["gain"] for row in rows)
            pathway_explored_fraction = statistics.fmean(
                row["first_step_gradient_norm"] > 0.0
                and row["peak_parameter_delta"] > 1e-8
                for row in rows
            )
            background_non_init_fraction = (
                statistics.fmean(float(not row["background_selected_at_init"]) for row in rows)
                if model == "dual" else None
            )
            background_budget_exhausted_fraction = (
                statistics.fmean(float(row["background_training_budget_exhausted"]) for row in rows)
                if model == "dual" else None
            )
            background_median_gain = (
                statistics.median(row["background_gain"] for row in rows)
                if model == "dual" else None
            )
            background_pathway_explored_fraction = (
                statistics.fmean(
                    row["background_first_step_gradient_norm"] > 0.0
                    and row["background_peak_parameter_delta"] > 1e-8
                    for row in rows
                ) if model == "dual" else None
            )
            dual_background_ok = (
                model != "dual" or (
                    background_pathway_explored_fraction >= 0.75
                    and background_budget_exhausted_fraction <= 0.25
                )
            )
            summary[recipe] = {
                "parameters": RECIPES[recipe],
                "mean_within_cell_rank": statistics.fmean(ranks[recipe]),
                "median_best_inner_loss": statistics.median(row["best_inner_loss"] for row in rows),
                "median_inner_gain": median_gain,
                "selected_at_init_fraction": statistics.fmean(float(row["selected_at_init"]) for row in rows),
                "selected_after_training_fraction": non_init_fraction,
                "selected_at_budget_edge_fraction": selected_at_edge_fraction,
                "training_budget_exhausted_fraction": budget_exhausted_fraction,
                "pathway_explored_fraction": pathway_explored_fraction,
                "background_selected_after_training_fraction": background_non_init_fraction,
                "background_training_budget_exhausted_fraction": background_budget_exhausted_fraction,
                "background_median_inner_gain": background_median_gain,
                "background_pathway_explored_fraction": background_pathway_explored_fraction,
                "inner_increment_supported": (
                    non_init_fraction >= 0.75 and median_gain > 1e-4
                ),
                "background_inner_increment_supported": (
                    model == "dual"
                    and background_non_init_fraction >= 0.75
                    and background_median_gain > 1e-4
                ) if model == "dual" else None,
                "trainability_eligible": (
                    pathway_explored_fraction >= 0.75
                    and budget_exhausted_fraction <= 0.25
                    and dual_background_ok
                ),
                "cells": rows,
            }
        eligible = [recipe for recipe in summary if summary[recipe]["trainability_eligible"]]
        pool = eligible if eligible else list(summary)
        selected = min(pool, key=lambda r: (
            summary[r]["mean_within_cell_rank"],
            -summary[r]["median_inner_gain"],
            summary[r]["median_best_inner_loss"], r,
        ))
        report["models"][model] = {
            "selected_recipe": selected,
            "selection_status": "TRAINABLE_RECIPE_SELECTED" if eligible else "NO_RECIPE_PASSED_TRAINABILITY",
            "trainability_eligible_recipes": eligible,
            "inner_increment_supported_recipes": [
                recipe for recipe in summary if summary[recipe]["inner_increment_supported"]
            ],
            "recipes": summary,
        }
    # ---- the control arm, with nested parity and an optimisation audit -------
    baseline_summary = {}
    baseline_cells = {}
    for recipe in BASELINE_RECIPES:
        rows = []
        for subject in SUBJECTS:
            for seed in SEEDS:
                path = root / "bmark" / recipe / subject / f"seed{seed}" / "card.json"
                if not path.exists():
                    continue
                stages = json.loads(path.read_text())["stages"]
                stage = stages["bmark"]
                rate = stages["q"]
                rows.append({
                    "subject": subject, "seed": seed,
                    "rate_selected_at_init": rate["selected_at_init"],
                    "rate_selected_at_budget_edge": rate["selected_at_budget_edge"],
                    "rate_training_budget_exhausted": rate["training_budget_exhausted"],
                    "rate_best_inner_loss": rate["best_inner_loss"],
                    "rate_initial_inner_loss": rate["initial_inner_loss"],
                    "rate_gain": rate["initial_inner_loss"] - rate["best_inner_loss"],
                    "rate_first_step_gradient_norm": rate["first_step_gradient_norm"],
                    "rate_peak_parameter_delta": rate["peak_parameter_delta_from_stage_start"],
                    "best_inner_loss": stage["best_inner_loss"],
                    "initial_inner_loss": stage["initial_inner_loss"],
                    "gain": stage["gain_over_parent"],
                    "selected_step": stage["selected_step"], "steps_run": stage["steps_run"],
                    "selected_at_init": stage["selected_at_init"],
                    "selected_at_budget_edge": stage["selected_at_budget_edge"],
                    "training_budget_exhausted": stage["training_budget_exhausted"],
                    "first_step_gradient_norm": stage["first_step_gradient_norm"],
                    "peak_parameter_delta": stage["peak_parameter_delta_from_stage_start"],
                })
        baseline_cells[recipe] = rows
    complete = {r: rows for r, rows in baseline_cells.items() if len(rows) == len(SUBJECTS) * len(SEEDS)}
    for recipe, rows in complete.items():
        non_init = statistics.fmean(float(not row["selected_at_init"]) for row in rows)
        edge = statistics.fmean(float(row["training_budget_exhausted"]) for row in rows)
        gain = statistics.median(row["gain"] for row in rows)
        pathway_explored = statistics.fmean(
            row["first_step_gradient_norm"] > 0.0
            and row["peak_parameter_delta"] > 1e-8
            for row in rows
        )
        # The mark bank is fitted on top of the rate stage.  A rate stage still
        # at its budget edge means the nested gain above it cannot be
        # attributed, so such a recipe is not eligible however good it looks.
        rate_edge = statistics.fmean(float(row["rate_training_budget_exhausted"]) for row in rows)
        rate_non_init = statistics.fmean(float(not row["rate_selected_at_init"]) for row in rows)
        rate_gain = statistics.median(row["rate_gain"] for row in rows)
        rate_pathway_explored = statistics.fmean(
            row["rate_first_step_gradient_norm"] > 0.0
            and row["rate_peak_parameter_delta"] > 1e-8
            for row in rows
        )
        rate_ok = rate_edge <= 0.25 and rate_pathway_explored >= 0.75
        frozen_recipe = frozen_baseline.get("recipes", {}).get(recipe, {})
        baseline_summary[recipe] = {
            "parameters": BASELINE_RECIPES[recipe],
            "median_best_inner_loss": statistics.median(row["best_inner_loss"] for row in rows),
            "median_inner_gain": gain,
            "selected_after_training_fraction": non_init,
            "pathway_explored_fraction": pathway_explored,
            "training_budget_exhausted_fraction": edge,
            "rate_stage_converged_fraction": 1.0 - rate_edge,
            "rate_stage_selected_after_training_fraction": rate_non_init,
            "rate_stage_median_inner_gain": rate_gain,
            "rate_stage_pathway_explored_fraction": rate_pathway_explored,
            "rate_stage_eligible": rate_ok,
            # Keep the control finalizer's stricter patient-first decision.
            # Recomputing only a cohort-average here can hide a completely
            # exhausted parent stage in one subject (the legacy recipe did so
            # for both E253 seeds) and contradict baseline_summary.json.
            "baseline_trainability_eligible": bool(
                frozen_recipe.get(
                    "eligible",
                    pathway_explored >= 0.75 and edge <= 0.25 and rate_ok,
                )
            ),
            "per_subject_stage_eligible": frozen_recipe.get(
                "per_subject_stage_eligible", {}
            ),
            "cells": rows,
        }
    baseline_eligible = list(frozen_baseline.get("eligible_recipes", ()))
    baseline_ranks = {recipe: [] for recipe in complete}
    for subject in SUBJECTS:
        for seed in SEEDS:
            ordered = sorted(complete, key=lambda recipe: next(
                row["best_inner_loss"] for row in complete[recipe]
                if row["subject"] == subject and row["seed"] == seed
            ))
            for rank, recipe in enumerate(ordered, 1):
                baseline_ranks[recipe].append(rank)
    for recipe in baseline_summary:
        baseline_summary[recipe]["mean_within_cell_rank"] = statistics.fmean(
            baseline_ranks[recipe]
        )
    baseline_selected = frozen_baseline_recipe
    if baseline_selected not in baseline_summary:
        raise RuntimeError("frozen baseline recipe is missing from the complete control sweep")
    report["baseline"] = {
        "selected_recipe": baseline_selected,
        "baseline_selection_status": (
            "TRAINABLE_BASELINE_RECIPE_SELECTED" if baseline_eligible else
            "NO_BASELINE_RECIPE_PASSED_TRAINABILITY" if baseline_summary else
            "BASELINE_SWEEP_NOT_RUN"
        ),
        "baseline_trainability_eligible_recipes": baseline_eligible,
        "incomplete_recipes": sorted(set(baseline_cells) - set(complete)),
        "recipes": baseline_summary,
        "why": (
            "the transparent multi-scale baseline is the arm the learned state must beat; "
            "in the first round it had one untested setting and stopped at step 0 in most "
            "cells, so a state gain over it could not be separated from an unfitted control"
        ),
    }

    report.update({
        "baseline_frozen_before_model_search": True,
        "frozen_baseline_recipe": frozen_baseline_recipe,
        "selection_contract": (
            "optimisation adequacy and human predictive increment are separate. Every learned "
            "state family requires non-zero first-step gradient and observed parameter movement "
            "in >=75% of cells, plus <=25% exhausted runs. Whether >=75% select a non-parent "
            "checkpoint with positive median INNER gain is reported separately as scientific "
            "inner_increment_supported and never relabelled as trainability. "
            "B_rate starts from a deterministic zero map. The transparent B_mark child always "
            "retains B_rate as an explicit parent candidate; "
            "its human gain is reported as an outcome, while first-step gradient, observed parameter "
            "movement, and <=25% exhausted runs establish optimisation adequacy. "
            "the dual family must satisfy optimisation adequacy separately for background and event state stages; "
            "then minimise mean within-cell INNER-loss rank"
        ),
        "selection_partition": "INNER", "selection_scores_used": False,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    path = root / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({m: d["selected_recipe"] for m, d in report["models"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
