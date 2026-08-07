#!/usr/bin/env python3
"""Parallel-safe patient workers and read-only aggregators for v3.0 responses.

The scientific computation remains in the frozen local/cumulative runners.
This file only schedules disjoint patients and aggregates their atomic outputs;
patient workers never write cohort states.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic5_event_innovation_v3_0_local_response as local  # noqa: E402
from scripts import run_topic5_event_innovation_v3_0_cumulative_response as cumulative  # noqa: E402
from src.topic5_resource_guard import pin_thread_environment  # noqa: E402


RUNNERS = {"local": local, "cumulative": cumulative}


def load_contract(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    innovation_root = ROOT / str(config["innovation_output_root"])
    innovation_path = innovation_root / "innovation_validity.json"
    innovation = json.loads(innovation_path.read_text(encoding="utf-8"))
    if (
        innovation.get("status") != "INNOVATION_VALIDITY_COMPLETE"
        or innovation.get("n_pass") != 34
        or innovation.get("human_test_outcomes_read") is not False
    ):
        raise RuntimeError("validation-only innovation contract is not complete")
    return config, innovation, innovation_path


def output_root(kind: str, config: dict) -> Path:
    key = "local_response_output_root" if kind == "local" else "cumulative_output_root"
    return ROOT / str(config[key])


def run_patients(kind, subjects, config, innovation, output):
    runner = RUNNERS[kind]
    cohort = {row["subject"] for row in innovation["patients"]}
    if set(subjects) - cohort:
        raise RuntimeError("worker requested a subject outside the frozen cohort")
    phase0_root = ROOT / str(config["output_root"])
    innovation_root = ROOT / str(config["innovation_output_root"])
    failures = []
    for subject in subjects:
        try:
            row = runner.run_subject(subject, config, phase0_root, innovation_root)
            runner.atomic_write_json(
                output / "per_subject" / f"{subject}.json", runner._jsonable(row)
            )
            print(subject, row["status"], flush=True)
        except Exception as exc:
            failures.append(
                {"subject": subject, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(subject, "FAIL", exc, flush=True)
    if failures:
        destination = output / "worker_failures" / f"worker_{os.getpid()}.json"
        runner.atomic_write_json(destination, failures)
        raise SystemExit(1)


def aggregate_local(config, innovation, innovation_path, output, config_path):
    rows = []
    failures = []
    for subject in [row["subject"] for row in innovation["patients"]]:
        path = output / "per_subject" / f"{subject}.json"
        if not path.exists():
            failures.append({"subject": subject, "error": "MissingArtifact"})
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    primary = int(config["primary_horizon"])
    handoff_rows = [
        row for row in rows
        if row.get("eligible")
        and row.get("horizons", {}).get(str(primary), {}).get(
            "n_validation_anchors", 0
        ) >= 20
    ]
    handoff = local.cohort_handoff(handoff_rows, primary) if rows else {
        "status": "GOAL2_NOT_OPEN", "reason": "no_patient_artifacts"
    }
    handoff["minimum_validation_anchors"] = 20
    handoff["n_excluded_below_anchor_minimum"] = int(
        sum(row.get("eligible", False) for row in rows) - len(handoff_rows)
    )
    summary = []
    for row in rows:
        primary_row = row.get("horizons", {}).get(str(primary), {})
        summary.append({
            "subject": row["subject"], "status": row["status"],
            "eligible": row.get("eligible", False),
            "dimension": row.get("dimension", np.nan),
            "propagation_gain": primary_row.get("observable", {}).get(
                "propagation_gain_standardized", np.nan
            ),
            "rank_gain": primary_row.get("observable", {}).get("rank_gain", np.nan),
            "precedence_gain": primary_row.get("observable", {}).get(
                "precedence_gain", np.nan
            ),
            "true_minus_matched": primary_row.get(
                "true_minus_state_matched_null_gain", np.nan
            ),
            "future_minus_past": primary_row.get(
                "future_minus_past_state_gain", np.nan
            ),
            "n_validation_anchors": primary_row.get("n_validation_anchors", 0),
        })
    if summary:
        local._atomic_csv(output / "patient_local_effects.csv", pd.DataFrame(summary))
    if failures:
        local._atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    state = {
        "contract": str(config["contract"]),
        "status": "LOCAL_RESPONSE_VALIDATION_COMPLETE" if not failures else "LOCAL_RESPONSE_VALIDATION_FAIL_CLOSED",
        "cohort_scope": "full_34_validation_only",
        "n_requested": 34, "n_completed": len(rows), "n_failed": len(failures),
        "n_innovation_valid": int(sum(row.get("eligible", False) for row in rows)),
        "goal2_handoff": handoff, "patients": rows, "failures": failures,
        "innovation_state_sha256": local.sha256(innovation_path),
        "config_sha256": local.sha256(config_path),
        "source_sha256": local.sha256(Path(local.__file__).resolve()),
        "scheduler_sha256": local.sha256(Path(__file__).resolve()),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    local.atomic_write_json(output / "local_projection_state.json", local._jsonable(state))
    local.atomic_write_json(output / "GOAL2_HANDOFF_STATE.json", local._jsonable(handoff))
    print(json.dumps({"status": state["status"], "handoff": handoff}, indent=2))
    if failures:
        raise SystemExit(1)


def aggregate_cumulative(config, innovation, innovation_path, output, config_path):
    rows = []
    failures = []
    for subject in [row["subject"] for row in innovation["patients"]]:
        path = output / "per_subject" / f"{subject}.json"
        if not path.exists():
            failures.append({"subject": subject, "error": "MissingArtifact"})
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    handoff_rows = []
    for row in rows:
        primary = row.get("combinations", {}).get(
            str(row.get("primary_exposure_events")), {}
        ).get(str(row.get("primary_horizon")), {})
        if row.get("eligible") and primary.get("n_validation_anchors", 0) >= 20:
            handoff_rows.append(row)
    handoff = cumulative.goal3_handoff(handoff_rows) if rows else {
        "status": "GOAL3_NOT_OPEN", "reason": "no_patient_artifacts"
    }
    handoff["minimum_validation_anchors"] = 20
    handoff["n_excluded_below_anchor_minimum"] = int(
        sum(row.get("eligible", False) for row in rows) - len(handoff_rows)
    )
    summary = []
    for row in rows:
        primary = row.get("combinations", {}).get(
            str(row.get("primary_exposure_events")), {}
        ).get(str(row.get("primary_horizon")), {})
        summary.append({
            "subject": row["subject"], "status": row["status"],
            "eligible": row.get("eligible", False),
            "propagation_gain": primary.get("observable", {}).get(
                "propagation_gain_standardized", np.nan
            ),
            "true_minus_matched": primary.get(
                "true_minus_matched_cumulative_null_gain", np.nan
            ),
            "dose_coefficient": primary.get("dose_alignment", {}).get(
                "dose_coefficient", np.nan
            ),
            "alignment_coefficient": primary.get("dose_alignment", {}).get(
                "alignment_coefficient", np.nan
            ),
            "n_validation_anchors": primary.get("n_validation_anchors", 0),
        })
    if summary:
        cumulative._atomic_csv(output / "dose_response.csv", pd.DataFrame(summary))
    if failures:
        cumulative._atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    state = {
        "contract": str(config["contract"]),
        "status": "CUMULATIVE_RESPONSE_VALIDATION_COMPLETE" if not failures else "CUMULATIVE_RESPONSE_VALIDATION_FAIL_CLOSED",
        "n_requested": 34, "n_completed": len(rows), "n_failed": len(failures),
        "goal3_handoff": handoff, "patients": rows, "failures": failures,
        "uniform_weight_order_null_revision": {
            "within_window_order_is_mathematically_invariant": True,
            "primary_null": "matched_complete_exposure_reassignment",
            "within_window_order_reserved_for_iei_decay_sensitivity": True,
        },
        "innovation_state_sha256": cumulative.sha256(innovation_path),
        "config_sha256": cumulative.sha256(config_path),
        "source_sha256": cumulative.sha256(Path(cumulative.__file__).resolve()),
        "scheduler_sha256": cumulative.sha256(Path(__file__).resolve()),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    cumulative.atomic_write_json(
        output / "cumulative_response_state.json", cumulative._jsonable(state)
    )
    cumulative.atomic_write_json(
        output / "GOAL3_HANDOFF_STATE.json", cumulative._jsonable(handoff)
    )
    cumulative.atomic_write_json(
        output / "iei_decay_sensitivity.json",
        cumulative._jsonable({
            "status": "VALIDATION_ONLY_IEI_SENSITIVITY_COMPLETE",
            "tau_seconds": list(map(float, config.get("iei_decay_tau_seconds", []))),
            "patients": [{
                "subject": row["subject"], "status": row["status"],
                "sensitivity": row.get("iei_decay_sensitivity", {}),
            } for row in rows],
            "human_test_outcomes_read": False,
            "biological_time_constant_claimed": False,
        }),
    )
    print(json.dumps({"status": state["status"], "handoff": handoff}, indent=2))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("local", "cumulative"), required=True)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--config", type=Path, default=local.DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config.resolve()
    config, innovation, innovation_path = load_contract(config_path)
    output = output_root(args.kind, config)
    if args.phase == "patients":
        if not args.subjects:
            raise ValueError("patients phase requires subjects")
        run_patients(args.kind, args.subjects, config, innovation, output)
    elif args.kind == "local":
        aggregate_local(config, innovation, innovation_path, output, config_path)
    else:
        aggregate_cumulative(config, innovation, innovation_path, output, config_path)


if __name__ == "__main__":
    main()
