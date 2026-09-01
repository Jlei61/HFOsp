#!/usr/bin/env python3
"""Aggregate frozen-setting Z/M morphology confirmation networks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PARAMETERS = (
    "I_th_EI", "tau_z", "tau_adp", "eta_m",
    "E_to_E_dose", "E_to_I_dose",
)


def summarize_confirmation(records: list[dict], frozen: dict) -> dict:
    if len(records) != 3:
        raise RuntimeError(f"expected three confirmation records, found {len(records)}")
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != 3:
        raise RuntimeError(f"confirmation seeds are not unique: {seeds}")
    for record in records:
        parameters = record["parameters"]
        for key in PARAMETERS:
            if not np.isclose(
                float(parameters[key]), float(frozen[key]),
                rtol=0.0, atol=1e-12,
            ):
                raise RuntimeError(
                    f"seed {record['seed']} parameter {key} drifted: "
                    f"{parameters[key]} != {frozen[key]}")
        if not bool(record.get("final_joint_eligible")):
            raise RuntimeError(f"seed {record['seed']} is not a full Joint arm")

    passed = [
        record for record in records
        if record["verdict"] == "JOINT_SUSTAINED_HIGH_OSCILLATORY_STATE_CANARY_PASS"
    ]
    representative = None
    if passed:
        onsets = np.asarray([
            float(record["runaway_morphology"]["scientific_onset_ms"])
            for record in passed
        ])
        median_onset = float(np.median(onsets))
        representative = min(
            passed,
            key=lambda record: (
                abs(float(record["runaway_morphology"]["scientific_onset_ms"])
                    - median_onset),
                int(record["seed"]),
            ),
        )
    accepted = len(passed) >= 2
    return {
        "status": "ZM_JOINT_MORPHOLOGY_CONFIRMATION_COMPLETE",
        "acceptance_rule": "at least two of three frozen, independent networks pass",
        "accepted": accepted,
        "n_pass": len(passed),
        "n_total": len(records),
        "frozen_parameters": {key: float(frozen[key]) for key in PARAMETERS},
        "seeds": seeds,
        "records": records,
        "representative_seed": (
            int(representative["seed"]) if representative is not None else None
        ),
        "next_action": (
            "render Figure 5A from the median-onset passing network"
            if accepted else
            "do not render a paper-facing Figure 5A from this setting"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    calibration = json.loads(Path(args.calibration_summary).read_text())
    selected = calibration.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("calibration has no frozen selected setting")
    frozen = selected["parameters"]
    records = [
        json.loads(path.read_text())
        for path in sorted(root.glob("ith080_s*.json"))
    ]
    output = summarize_confirmation(records, frozen)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accepted": output["accepted"],
        "n_pass": output["n_pass"],
        "representative_seed": output["representative_seed"],
    }))


if __name__ == "__main__":
    main()
