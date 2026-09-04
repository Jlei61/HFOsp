#!/usr/bin/env python3
"""Finalize the minimal dynamic/constant/null S_P instrument calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json, file_hash


KINDS = ("dynamic", "piecewise_constant", "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for path in args.card:
        card = json.loads(path.read_text(encoding="utf-8"))
        kind = str(card.get("synthetic_truth", {}).get("kind"))
        if kind not in KINDS or kind in rows:
            raise ValueError(f"unexpected or repeated synthetic truth kind: {kind}")
        if any(card.get(key) is not False for key in (
            "development_targets_read", "sealed_partition_opened", "seizure_outcomes_read",
        )):
            raise PermissionError(f"forbidden provenance in {path}")
        rows[kind] = {
            "path": str(path), "sha256": file_hash(path), "status": card.get("status"),
            "inner_gain": card.get("inner_gain"), "report_gain": card.get("selection_gain"),
            "period_level_gain": card.get("period_level_gain"),
            "beyond_period_gain": card.get("beyond_period_gain"),
            "wrong_time_cost": card.get("wrong_time_cost"),
            "functional_oracle": card.get("synthetic_truth", {}).get("functional_oracle"),
        }
    if set(rows) != set(KINDS):
        raise ValueError(f"synthetic suite incomplete: {sorted(rows)}")
    contrasts_pass = all(rows[k]["status"] == "PASS" for k in KINDS)
    end_to_end_pass = float(rows["dynamic"]["report_gain"]) > 0
    payload = {
        "format": "group_event_state_v0_3_4_spatial_synthetic_suite_v2",
        "status": (
            "PASS" if contrasts_pass and end_to_end_pass
            else "PARTIAL_PASS" if contrasts_pass
            else "FAIL"
        ),
        "contrast_calibration_pass": contrasts_pass,
        "dynamic_end_to_end_report_gain_pass": end_to_end_pass,
        "arms": rows,
        "interpretation": (
            "Engineering calibration only: dynamic truth must survive period and wrong-time controls; "
            "piecewise-constant truth must appear as a period-level signal; no-state truth must not yield a report gain. "
            "A full PASS additionally requires the learned dynamic model to beat the no-state baseline on its untouched report period."
        ),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
