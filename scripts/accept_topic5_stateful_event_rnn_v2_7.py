#!/usr/bin/env python3
"""Read-only derived acceptance for the repaired Topic 5 v2.7 observer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import accept_topic5_stateful_event_rnn_v2_6 as v26_accept  # noqa: E402
from scripts import run_topic5_stateful_event_rnn_v2_7_formal as formal  # noqa: E402


RESULT_ROOT = ROOT / "results/topic5_stateful_event_sequence_rnn/v2_7"
PARENT_ROOT = ROOT / "results/topic5_stateful_event_sequence_rnn/v2_6"
CONTRACT = "topic5_stateful_event_sequence_rnn_v2_7"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(result_root: Path) -> tuple[dict, list[dict]]:
    rows = []
    failures = []
    for path in sorted((result_root / "per_subject").glob("*.json")):
        if path.stem.endswith("_predictions"):
            continue
        record = load_json(path)
        subject = record.get("subject", path.stem)
        runs = record.get("recurrent_runs", [])
        reasons = []
        if record.get("contract") != CONTRACT:
            reasons.append("wrong_contract")
        if len(runs) != 3:
            reasons.append("not_three_seeds")
        if not all(record.get("contract_checks", {}).values()):
            reasons.append("data_contract_failure")
        for run in runs:
            trace = run.get("trace", {})
            scores = run.get("trained_test_score", {})
            if trace.get("finite") is not True:
                reasons.append("nonfinite_training")
            if int(trace.get("best_trained_epoch", -1)) < 0:
                reasons.append("missing_trained_checkpoint")
            if not all(np.isfinite(float(value)) for value in scores.values()):
                reasons.append("nonfinite_test_score")
        rows.append({
            "subject": subject,
            "n_runs": len(runs),
            "finite": not reasons,
            "reasons": sorted(set(reasons)),
        })
        if reasons:
            failures.append({"subject": subject, "reasons": sorted(set(reasons))})
    return {
        "n_patient_artifacts": len(rows),
        "n_runs": int(sum(row["n_runs"] for row in rows)),
        "n_failures": len(failures),
        "all_three_seed_runs_finite": bool(len(rows) == 34 and not failures),
        "trained_checkpoint_never_epoch_minus_one": bool(
            len(rows) == 34 and not failures
        ),
    }, failures


def paired_v26_v27(v27: pd.DataFrame) -> dict:
    v26 = v26_accept.load_patient_frame(PARENT_ROOT)
    merged = v27.merge(v26, on="subject", suffixes=("_v27", "_v26"))
    ewma_delta = (
        merged["trained_rnn_minus_ewma_propagation_v27"]
        - merged["trained_rnn_minus_ewma_propagation_v26"]
    )
    static_delta = (
        merged["trained_rnn_minus_static_propagation_v27"]
        - merged["trained_rnn_minus_static_propagation_v26"]
    )
    def paired_summary(values):
        array = np.asarray(values, dtype=float)
        if np.all(array == 0):
            return {
                "n": int(len(array)),
                "median": 0.0,
                "bootstrap_median_ci95": [0.0, 0.0],
                "n_favorable": 0,
                "n_non_ties": 0,
                "tie_excluded_sign_p": None,
                "wilcoxon_one_sided_p": 1.0,
                "wilcoxon_opposite_tail_p": 1.0,
                "exact_patientwise_identity": True,
            }
        result = v26_accept.directional_summary(values, favorable="negative")
        result["exact_patientwise_identity"] = False
        return result

    return {
        "n_paired": int(len(merged)),
        "v27_minus_v26_rnn_minus_ewma": paired_summary(ewma_delta),
        "v27_minus_v26_rnn_minus_static": paired_summary(static_delta),
    }


def build_state(result_root: Path = RESULT_ROOT) -> tuple[dict, pd.DataFrame]:
    frame = v26_accept.load_patient_frame(result_root)
    formal_state = load_json(result_root / "STATEFUL_TEST_STATE.json")
    run_state, run_failures = run_audit(result_root)
    required = {
        "formal": (result_root / "STATEFUL_TEST_STATE.json", "STATEFUL_34_PATIENT_TEST_COMPLETE"),
        "dense": (result_root / "dense_test_sensitivity/DENSE_TEST_STATE.json", "DENSE_TEST_34_COMPLETE"),
        "state_reset": (result_root / "state_reset_ablation/STATE_RESET_STATE.json", "STATE_RESET_34_COMPLETE"),
        "memory_curve": (result_root / "memory_curve/MEMORY_CURVE_STATE.json", "MEMORY_CURVE_34_COMPLETE"),
        "block_null": (result_root / "chronology_null/block_shuffle/BLOCK_NULL_STATE.json", "BLOCK_NULL_34_COMPLETE"),
        "reversal_null": (result_root / "chronology_null/time_reversal/TIME_REVERSAL_STATE.json", "TIME_REVERSAL_34_COMPLETE"),
        "h40": (result_root / "h40_sensitivity/H40_STATE.json", "H40_COHORT_AUDIT_COMPLETE"),
    }
    prerequisites = {}
    for name, (path, expected) in required.items():
        value = load_json(path) if path.exists() else {}
        prerequisites[name] = {
            "path": str(path),
            "expected_status": expected,
            "observed_status": value.get("status"),
            "complete": value.get("status") == expected,
            "sha256": formal.sha256(path) if path.exists() else None,
        }
    control_adapters = {}
    for name in (
        "dense", "state-reset", "memory-curve", "block-null",
        "reversal-null", "h40",
    ):
        path = result_root / "control_adapter" / f"{name}_aggregate.json"
        value = load_json(path) if path.exists() else {}
        control_adapters[name] = {
            "path": str(path),
            "status": value.get("status"),
            "contract": value.get("contract"),
            "checkpoint_contract_required": value.get(
                "checkpoint_contract_required"
            ),
            "complete": bool(
                value.get("status") == "CONTROL_ADAPTER_EXECUTION_COMPLETE"
                and value.get("contract") == CONTRACT
                and value.get("checkpoint_contract_required") == CONTRACT
            ),
            "sha256": formal.sha256(path) if path.exists() else None,
        }
    primary = v26_accept.directional_summary(
        frame["trained_rnn_minus_ewma_propagation"], favorable="negative"
    )
    reproduces = bool(
        len(frame) == 34
        and primary["median"]
        == formal_state["trained_primary_propagation"]["median_rnn_minus_ewma"]
        and primary["n_favorable"]
        == formal_state["trained_primary_propagation"]["n_rnn_better"]
    )
    complete = bool(
        len(frame) == 34
        and run_state["all_three_seed_runs_finite"]
        and reproduces
        and all(row["complete"] for row in prerequisites.values())
        and all(row["complete"] for row in control_adapters.values())
    )
    frozen = load_json(result_root / "validation_screen/FROZEN_VALIDATION_STATE.json")
    parent_unchanged = all(
        frozen["parent_v2_6"][key] == formal.sha256(PARENT_ROOT / relative)
        for key, relative in (
            ("frozen_validation_state_sha256", "validation_screen/FROZEN_VALIDATION_STATE.json"),
            ("primary_test_state_sha256", "STATEFUL_TEST_STATE.json"),
        )
    )
    state = {
        "contract": CONTRACT,
        "status": "DERIVED_ACCEPTANCE_COMPLETE" if complete and parent_unchanged else "INCOMPLETE",
        "derivation_only_no_training": True,
        "n_patients": int(len(frame)),
        "reproduces_frozen_primary_endpoint": reproduces,
        "run_audit": run_state,
        "run_failures": run_failures,
        "required_artifacts": prerequisites,
        "control_adapter_provenance": control_adapters,
        "parent_v2_6_unchanged": parent_unchanged,
        "comparisons": {
            "trained_rnn_minus_ewma_formal": primary,
            "trained_rnn_minus_static_formal": v26_accept.directional_summary(
                frame["trained_rnn_minus_static_propagation"], favorable="negative"
            ),
            "ewma_minus_static_formal": v26_accept.directional_summary(
                frame["ewma_minus_static_propagation"], favorable="negative"
            ),
            "trained_rnn_minus_ewma_dense": v26_accept.directional_summary(
                frame["dense_rnn_minus_ewma_propagation"], favorable="negative"
            ),
        },
        "support_strata": {
            "trained_rnn_minus_ewma_formal": v26_accept._stratified(
                frame, "trained_rnn_minus_ewma_propagation", favorable="negative"
            ),
            "trained_rnn_minus_static_formal": v26_accept._stratified(
                frame, "trained_rnn_minus_static_propagation", favorable="negative"
            ),
        },
        "chronology_nulls_both_tails": {
            "source_coherent_block_shuffle": v26_accept.directional_summary(
                frame["block_true_minus_null_propagation"], favorable="negative"
            ),
            "source_level_time_reversal": v26_accept.directional_summary(
                frame["reversal_true_minus_null_propagation"], favorable="negative"
            ),
        },
        "paired_v2_6_to_v2_7": paired_v26_v27(frame),
        "scientific_adjudication": {
            "status": "ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL" if complete else "PENDING",
            "allowed": [
                "event_history_state_tracking",
                "comparison_with_static_and_fixed_ewma_observers",
                "within_recording_memory_and_chronology_controls",
            ],
            "forbidden": [
                "event_driven_network_shaping",
                "evolving_graph_identification",
                "causal_plasticity",
                "within_event_next_rank_mechanism",
            ],
        },
        "old_heldout20_entered": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "runner_sha256": formal.sha256(Path(__file__).resolve()),
    }
    return state, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    state, frame = build_state(args.result_root.resolve())
    destination = args.result_root / "acceptance"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "patient_summary.csv", index=False)
    path = destination / "ACCEPTANCE_STATE.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(state, indent=2, sort_keys=True))
    if state["status"] != "DERIVED_ACCEPTANCE_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
