#!/usr/bin/env python3
"""Lock one dynamic-rate recipe from chronological FIT/INNER evidence only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS, OUTPUT_ROOT, V035_SUBJECTS, atomic_json  # noqa: E402


def main() -> None:
    root = OUTPUT_ROOT / "dynamic_rate_search"
    rows = []
    for path in sorted(root.glob("*/*/seed*/card.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        if card.get("selection", {}).get("status") != "HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH":
            raise RuntimeError(f"search card opened SELECTION targets: {path}")
        rows.append({
            "recipe": path.parents[2].name,
            "subject": card["subject"],
            "seed": int(card["seed"]),
            "inner_nll": float(card["stages"]["residual"]["best_inner_nll"]),
            "selected_step": int(card["stages"]["residual"]["selected_step"]),
            "card": str(path),
        })
    recipes = sorted({row["recipe"] for row in rows})
    subjects = list(V035_SUBJECTS)
    seeds = list(LOCKED_SEEDS[:3])
    by = {(r["recipe"], r["subject"], r["seed"]): r for r in rows}
    expected = {(recipe, subject, seed) for recipe in recipes for subject in subjects for seed in seeds}
    missing = sorted(expected - set(by))
    if not recipes or missing:
        raise RuntimeError(f"incomplete dynamic-rate search: recipes={recipes}, missing={missing[:10]}")
    medians = {
        (recipe, subject): float(np.median([by[(recipe, subject, seed)]["inner_nll"] for seed in seeds]))
        for recipe in recipes for subject in subjects
    }
    summary = {}
    for recipe in recipes:
        ranks = []
        for subject in subjects:
            ordered = sorted((medians[(r, subject)], r) for r in recipes)
            ranks.append(next(i for i, (_, r) in enumerate(ordered) if r == recipe))
        summary[recipe] = {
            "median_subject_rank": float(np.median(ranks)),
            "median_inner_nll": float(np.median([medians[(recipe, s)] for s in subjects])),
            "n_subjects": len(subjects),
            "n_units": len(subjects) * len(seeds),
        }
    chosen = min(recipes, key=lambda r: (summary[r]["median_subject_rank"], summary[r]["median_inner_nll"], r))
    payload = {
        "format": "group_event_state_v0_3_5_dynamic_rate_recipe_selection_v1",
        "selected_recipe": chosen,
        "selection_basis": "median subject rank on chronological INNER NLL; SELECTION targets held unread",
        "subjects": subjects,
        "seeds": seeds,
        "recipes": summary,
        "units": rows,
        "selection_targets_read": False,
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(root / "selected_recipe.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
