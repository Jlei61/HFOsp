#!/usr/bin/env python3
"""Freeze the B_rate/B_mark recipe before any learned-state search starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from run_group_event_state_v037_optimizer_search import BASELINE_RECIPES
from src.topic5_group_event_state.v037.contracts import atomic_json


SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root
    cells: dict[str, list[dict]] = {}
    for recipe in BASELINE_RECIPES:
        rows = []
        for subject in SUBJECTS:
            for seed in SEEDS:
                path = root / "bmark" / recipe / subject / f"seed{seed}" / "card.json"
                if not path.exists():
                    continue
                stages = json.loads(path.read_text(encoding="utf-8"))["stages"]
                rate, mark = stages["q"], stages["bmark"]
                rows.append({
                    "subject": subject, "seed": seed,
                    "rate_gain": rate["initial_inner_loss"] - rate["best_inner_loss"],
                    "rate_selected_at_init": rate["selected_at_init"],
                    "rate_selected_at_budget_edge": rate["selected_at_budget_edge"],
                    "rate_training_budget_exhausted": rate["training_budget_exhausted"],
                    "rate_first_step_gradient_norm": rate["first_step_gradient_norm"],
                    "rate_peak_parameter_delta": rate["peak_parameter_delta_from_stage_start"],
                    "mark_best_inner_loss": mark["best_inner_loss"],
                    "mark_gain": mark["gain_over_parent"],
                    "mark_selected_at_init": mark["selected_at_init"],
                    "mark_selected_at_budget_edge": mark["selected_at_budget_edge"],
                    "mark_training_budget_exhausted": mark["training_budget_exhausted"],
                    "mark_first_step_gradient_norm": mark["first_step_gradient_norm"],
                    "mark_peak_parameter_delta": mark["peak_parameter_delta_from_stage_start"],
                })
        cells[recipe] = rows
    expected = len(SUBJECTS) * len(SEEDS)
    complete = {recipe: rows for recipe, rows in cells.items() if len(rows) == expected}
    ranks = {recipe: [] for recipe in complete}
    for subject in SUBJECTS:
        for seed in SEEDS:
            ordered = sorted(complete, key=lambda recipe: next(
                row["mark_best_inner_loss"] for row in complete[recipe]
                if row["subject"] == subject and row["seed"] == seed
            ))
            for rank, recipe in enumerate(ordered, 1):
                ranks[recipe].append(rank)
    recipes = {}
    for recipe, rows in complete.items():
        rate_non_init = statistics.fmean(not row["rate_selected_at_init"] for row in rows)
        rate_edge = statistics.fmean(row["rate_training_budget_exhausted"] for row in rows)
        rate_gain = statistics.median(row["rate_gain"] for row in rows)
        rate_pathway_explored = statistics.fmean(
            row["rate_first_step_gradient_norm"] > 0.0
            and row["rate_peak_parameter_delta"] > 1e-8
            for row in rows
        )
        mark_non_init = statistics.fmean(not row["mark_selected_at_init"] for row in rows)
        mark_edge = statistics.fmean(row["mark_training_budget_exhausted"] for row in rows)
        mark_gain = statistics.median(row["mark_gain"] for row in rows)
        mark_pathway_explored = statistics.fmean(
            row["mark_first_step_gradient_norm"] > 0.0
            and row["mark_peak_parameter_delta"] > 1e-8
            for row in rows
        )
        per_subject = {}
        for subject in SUBJECTS:
            subject_rows = [row for row in rows if row["subject"] == subject]
            per_subject[subject] = {
                "rate": (
                    statistics.fmean(
                        row["rate_first_step_gradient_norm"] > 0.0
                        and row["rate_peak_parameter_delta"] > 1e-8
                        for row in subject_rows
                    ) >= 0.75
                    and statistics.fmean(row["rate_training_budget_exhausted"] for row in subject_rows) <= 0.25
                ),
                "mark": (
                    statistics.fmean(
                        row["mark_first_step_gradient_norm"] > 0.0
                        and row["mark_peak_parameter_delta"] > 1e-8
                        for row in subject_rows
                    ) >= 0.75
                    and statistics.fmean(row["mark_training_budget_exhausted"] for row in subject_rows) <= 0.25
                ),
            }
        eligible = (
            rate_pathway_explored >= 0.75 and rate_edge <= 0.25
            and mark_pathway_explored >= 0.75 and mark_edge <= 0.25
            and all(node["rate"] and node["mark"] for node in per_subject.values())
        )
        recipes[recipe] = {
            "parameters": BASELINE_RECIPES[recipe],
            "rate_selected_after_training_fraction": rate_non_init,
            "rate_selected_at_budget_edge_fraction": rate_edge,
            "rate_median_inner_gain": rate_gain,
            "rate_pathway_explored_fraction": rate_pathway_explored,
            "mark_selected_after_training_fraction": mark_non_init,
            "mark_selected_at_budget_edge_fraction": mark_edge,
            "mark_median_inner_gain": mark_gain,
            "mark_pathway_explored_fraction": mark_pathway_explored,
            "mean_within_cell_rank": statistics.fmean(ranks[recipe]),
            "per_subject_stage_eligible": per_subject,
            "eligible": eligible, "cells": rows,
        }
    eligible = [recipe for recipe, node in recipes.items() if node["eligible"]]
    selected = min(eligible, key=lambda recipe: (
        recipes[recipe]["mean_within_cell_rank"],
        -recipes[recipe]["mark_median_inner_gain"], recipe,
    )) if eligible else None
    payload = {
        "format": "group_event_state_v0_3_7_frozen_baseline_recipe_v2",
        "status": "TRAINABLE_BASELINE_RECIPE_SELECTED" if selected else "NO_BASELINE_RECIPE_PASSED_TRAINABILITY",
        "selected_recipe": selected, "eligible_recipes": eligible,
        "incomplete_recipes": sorted(set(cells) - set(complete)), "recipes": recipes,
        "selection_partition": "INNER",
        "nested_control_contract": (
            "B_rate starts at a deterministic zero map and B_mark always retains "
            "fitted B_rate as the zero-increment parent; human rate/mark gains are "
            "outcomes, while non-zero first-step gradient, "
            "observed parameter movement, and non-exhausted optimisation establish "
            "that the transparent pathway was actually explored"
        ),
        "model_search_started_before_baseline_freeze": False,
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    atomic_json(root / "baseline_summary.json", payload)
    print(json.dumps({"status": payload["status"], "selected_recipe": selected}, indent=2))


if __name__ == "__main__":
    main()
