#!/usr/bin/env python3
"""Freeze the validation-only V3.0 to V3.1 human-execution handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_row(row: dict, route: str, primary_horizon: int) -> dict:
    if route == "goal2":
        return row.get("horizons", {}).get(str(primary_horizon), {})
    return row.get("combinations", {}).get(
        str(row.get("primary_exposure_events")), {}
    ).get(str(row.get("primary_horizon")), {})


def dataset_directions(rows: list[dict], route: str, primary_horizon: int) -> dict:
    output = {}
    for dataset in ("epilepsiae", "yuquan"):
        selected = []
        for row in rows:
            if not row.get("subject", "").startswith(dataset + "_"):
                continue
            primary = primary_row(row, route, primary_horizon)
            if row.get("eligible") and primary.get("n_validation_anchors", 0) >= 20:
                selected.append(primary)
        if route == "goal2":
            values = {
                "propagation_gain": [
                    item["observable"]["propagation_gain_standardized"]
                    for item in selected
                ],
                "true_minus_matched": [
                    item["true_minus_state_matched_null_gain"] for item in selected
                ],
                "future_minus_past": [
                    item["future_minus_past_state_gain"] for item in selected
                ],
            }
        else:
            values = {
                "cumulative_gain": [
                    item["observable"]["propagation_gain_standardized"]
                    for item in selected
                ],
                "true_minus_matched": [
                    item["true_minus_matched_cumulative_null_gain"]
                    for item in selected
                ],
                "alignment": [
                    item["dose_alignment"]["alignment_coefficient"]
                    for item in selected
                ],
            }
        output[dataset] = {
            "n_eligible": len(selected),
            **{
                name: {
                    "median": float(np.median(value)) if value else None,
                    "n_positive": int(np.sum(np.asarray(value) > 0)),
                }
                for name, value in values.items()
            },
        }
    return output


def build_state(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    local_path = ROOT / str(config["local_response_output_root"]) / "local_projection_state.json"
    cumulative_path = ROOT / str(config["cumulative_output_root"]) / "cumulative_response_state.json"
    innovation_path = ROOT / str(config["innovation_output_root"]) / "innovation_validity.json"
    synthetic_path = ROOT / str(config.get(
        "synthetic_output_root",
        "results/topic5_event_innovation_impulse_response/v3_0/synthetic_calibration",
    )) / "synthetic_identifiability_state.json"
    transition_path = ROOT / "results/topic5_event_innovation_state_space/v3_1/synthetic_calibration/synthetic_transition_acceptance_state.json"
    paths = (local_path, cumulative_path, innovation_path, synthetic_path, transition_path)
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"handoff input missing: {path}")
    local = load_json(local_path)
    cumulative = load_json(cumulative_path)
    innovation = load_json(innovation_path)
    synthetic = load_json(synthetic_path)
    transition = load_json(transition_path)
    if local.get("status") != "LOCAL_RESPONSE_VALIDATION_COMPLETE":
        raise RuntimeError("Goal 2 validation state is not complete")
    if cumulative.get("status") != "CUMULATIVE_RESPONSE_VALIDATION_COMPLETE":
        raise RuntimeError("Goal 3 validation state is not complete")
    if innovation.get("status") != "INNOVATION_VALIDITY_COMPLETE":
        raise RuntimeError("innovation validity is not complete")
    if synthetic.get("status") != "SYNTHETIC_IDENTIFIABILITY_COMPLETE":
        raise RuntimeError("V3.0 synthetic calibration is not complete")
    if transition.get("status") != "SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE":
        raise RuntimeError("V3.1 synthetic calibration is not complete")
    if any(value.get("human_test_outcomes_read") is not False for value in (local, cumulative, innovation)):
        raise RuntimeError("human test outcomes were read before the handoff freeze")
    goal2 = local["goal2_handoff"]
    goal3 = cumulative["goal3_handoff"]
    opened = goal2.get("status") == "OPEN" or goal3.get("status") == "OPEN"
    primary_horizon = int(config["primary_horizon"])
    return {
        "contract": "topic5_event_innovation_v3_0_to_v3_1_handoff",
        "status": "OPEN" if opened else "NOT_TRIGGERED",
        "v3_1_human_execution_allowed": bool(opened),
        "opening_rule": "Goal2 OR Goal3 using validation-only cohort medians",
        "goal2": goal2,
        "goal3": goal3,
        "minimum_validation_anchors": 20,
        "dataset_specific_directions": {
            "goal2": dataset_directions(local["patients"], "goal2", primary_horizon),
            "goal3": dataset_directions(cumulative["patients"], "goal3", primary_horizon),
        },
        "n_innovation_valid": innovation.get("n_innovation_valid", sum(
            row.get("status") == "INNOVATION_VALID" for row in innovation["patients"]
        )),
        "human_test_outcomes_read": False,
        "v2_7_outcome_is_release_condition": False,
        "capacity_rescue_allowed_if_closed": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "inputs_sha256": {path.name: sha256(path) for path in paths},
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic5_event_innovation_v3_0.yaml",
    )
    args = parser.parse_args()
    state = build_state(args.config.resolve())
    destination = ROOT / "results/topic5_event_innovation_impulse_response/v3_0/V3_1_HANDOFF_STATE.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
