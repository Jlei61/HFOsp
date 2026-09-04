#!/usr/bin/env python3
"""Aggregate v0.3.4 S_P tuning cards without reading development outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json


ENDPOINTS = ("grammar", "subset", "continue", "extent", "lag", "total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/hfosp_group_event_state_v0_3_4/spatial_state_recalibrated/human"))
    parser.add_argument("--rung", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_subject = {}
    for subject_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        rows = []
        for path in sorted((subject_dir / f"rung{args.rung}").glob("*/training_card.json")):
            card = json.loads(path.read_text(encoding="utf-8"))
            if card.get("status") != "PASS":
                continue
            contract = card["contract"]
            if any(card.get(key) is not False for key in (
                "development_targets_read", "sealed_partition_opened", "seizure_outcomes_read"
            )):
                raise PermissionError(f"forbidden provenance in {path}")
            report = card["state_selection_full"]
            gains = {
                endpoint: float(report["train_mean_no_state"][endpoint])
                - float(report["learned_correct_time"][endpoint])
                for endpoint in ENDPOINTS
            }
            inner_gains = {
                endpoint: float(card["initial_inner"][endpoint])
                - float(card["selected_inner"][endpoint])
                for endpoint in ENDPOINTS
            }
            rows.append({
                "cell": path.parent.name,
                "path": str(path),
                "arch": contract["arch"],
                "optimizer": contract["optimizer"],
                "seed": int(contract["train"]["seed"]),
                "selected_step": int(card["selected_step"]),
                "steps_run": int(card["steps_run"]),
                "selection_gain": float(card["selection_gain"]),
                "inner_gain": float(card["inner_gain"]),
                "period_level_gain": float(card["period_level_gain"]),
                "beyond_period_gain": float(card["beyond_period_gain"]),
                "rolling_level_gain": float(card["rolling_level_gain"]),
                "wrong_time_cost": float(card["wrong_time_cost"]),
                "endpoint_gains": gains,
                "inner_endpoint_gains": inner_gains,
                "max_gradient_l2": float(card["max_gradient_l2"]),
                "peak_cuda_bytes": int(card["resources"]["peak_cuda_bytes"]),
            })
        if not rows:
            continue
        # Recipe ordering is TRAIN-inner only.  STATE_SELECTION values are
        # intentionally unavailable to this decision.
        rows.sort(key=lambda row: row["inner_gain"], reverse=True)
        by_subject[subject_dir.name] = {
            "n_cells": len(rows),
            "n_positive_gain": sum(row["selection_gain"] > 0 for row in rows),
            "n_positive_inner_gain": sum(row["inner_gain"] > 0 for row in rows),
            "n_selected_step_zero": sum(row["selected_step"] == 0 for row in rows),
            "median_selection_gain": statistics.median(row["selection_gain"] for row in rows),
            "median_inner_gain": statistics.median(row["inner_gain"] for row in rows),
            "median_beyond_period_gain": statistics.median(row["beyond_period_gain"] for row in rows),
            "median_wrong_time_cost": statistics.median(row["wrong_time_cost"] for row in rows),
            "endpoint_median_gain": {
                endpoint: statistics.median(row["endpoint_gains"][endpoint] for row in rows)
                for endpoint in ENDPOINTS
            },
            "top_cells": rows[:3],
            "cells": rows,
        }
    payload = {
        "format": "group_event_state_v0_3_4_spatial_search_summary_v1",
        "rung": args.rung,
        "by_subject": by_subject,
        "interpretation_boundary": (
            "Recipe ranking uses chronological TRAIN-inner only. Full STATE_SELECTION is report-only; "
            "period/rolling/correct-time controls classify L1-L3 but do not by themselves establish a cohort claim."
        ),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
