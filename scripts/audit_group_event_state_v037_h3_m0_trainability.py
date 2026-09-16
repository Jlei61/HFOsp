#!/usr/bin/env python3
"""Aggregate the targeted v0.3.7 M0 learning-rate sensitivity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("lr_*/*/seed*/card.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        training = card["training"]["M0_common_drive"]
        history = training["history"]
        initial = float(history[0]["inner"]["total"])
        best = min(float(item["inner"]["total"]) for item in history)
        learning_rate = float(path.parts[-4].removeprefix("lr_").replace("p", "."))
        rows.append({
            "subject": card["subject"],
            "seed": int(card["seed"]),
            "learning_rate": learning_rate,
            "selected_step": int(training["selected_step"]),
            "steps_run": int(training["steps_run"]),
            "inner_total_gain": initial - best,
            "count_to_future_background_gain": float(
                card["primary_contrasts"]["count_feedback_gain_on_future_background"]
            ),
            "card": str(path),
        })

    groups = []
    for subject in sorted({row["subject"] for row in rows}):
        for learning_rate in sorted({row["learning_rate"] for row in rows}):
            members = [
                row for row in rows
                if row["subject"] == subject and row["learning_rate"] == learning_rate
            ]
            if not members:
                continue
            groups.append({
                "subject": subject,
                "learning_rate": learning_rate,
                "n_seeds": len(members),
                "selected_non_origin": sum(row["selected_step"] > 0 for row in members),
                "selected_steps": [row["selected_step"] for row in members],
                "median_inner_total_gain": median(row["inner_total_gain"] for row in members),
                "median_count_to_future_background_gain": median(
                    row["count_to_future_background_gain"] for row in members
                ),
            })

    payload = {
        "format": "group_event_state_v0_3_7_h3_m0_trainability_sensitivity_v1",
        "purpose": "training-adequacy sensitivity for default-M0 origin selections only; not model selection",
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "n_cards": len(rows),
        "groups": groups,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_cards": len(rows), "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
