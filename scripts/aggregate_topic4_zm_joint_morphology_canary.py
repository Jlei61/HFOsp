#!/usr/bin/env python3
"""Aggregate the fixed four-arm Joint Z/M morphology calibration."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BASE_I = 95.19851312666987
BASE_ETA = 0.007451594355587098
BASE_TAU_ADP = 500.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    for path in sorted(root.glob("ith*.json")):
        payload = json.loads(path.read_text())
        parameters = payload["parameters"]
        distance = (
            abs(math.log(float(parameters["I_th_EI"]) / BASE_I))
            + abs(math.log(float(parameters["eta_m"]) / BASE_ETA))
            + abs(math.log(float(parameters["tau_adp"]) / BASE_TAU_ADP))
        )
        etoi_dose = float(parameters.get("E_to_I_dose", 1.0))
        distance += abs(math.log(etoi_dose)) if etoi_dose > 0.0 else math.inf
        records.append({
            "path": str(path),
            "verdict": payload["verdict"],
            "parameters": parameters,
            "operational_onset_ms": payload.get("operational_onset_ms"),
            "final_joint_eligible": payload.get("final_joint_eligible", True),
            "distance_from_frozen_reference": distance,
            "morphology": payload.get("runaway_morphology"),
        })
    if len(records) != 4:
        raise RuntimeError(f"expected four calibration records, found {len(records)}")
    passed = [
        row for row in records
        if row["verdict"].endswith("CANARY_PASS")
        and row["final_joint_eligible"]
    ]
    selected = (
        min(passed, key=lambda row: row["distance_from_frozen_reference"])
        if passed else None
    )
    output = {
        "status": "ZM_JOINT_MORPHOLOGY_CALIBRATION_AGGREGATED",
        "selection_rule": (
            "among full morphology passes, choose the smallest log-distance "
            "from the frozen Z/M reference; no patient score or rendered figure used"),
        "records": records,
        "n_pass": len(passed),
        "selected": selected,
        "next_action": (
            "run three independent network seeds at the selected setting"
            if selected else
            "do not draw Figure 5A; expand the mechanistic calibration deliberately"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_pass": len(passed), "selected": selected}))


if __name__ == "__main__":
    main()
