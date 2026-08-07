#!/usr/bin/env python3
"""Derive the frozen program-level acceptance for Topic 5 V3.0.

This layer is intentionally read-only with respect to model fitting.  It combines
the two independently aggregated human-test routes and applies the evidence
ladder frozen in the V3.0 specification.  It cannot reopen the pre-test V3.1
handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3_ROOT = ROOT / "results/topic5_event_innovation_impulse_response/v3_0"
HUMAN_ROOT = V3_ROOT / "human_exploratory"
RULE_STATE = HUMAN_ROOT / "ACCEPTANCE_RULE_STATE.json"
V27_ACCEPTANCE = (
    ROOT
    / "results/topic5_stateful_event_sequence_rnn/v2_7/acceptance/ACCEPTANCE_STATE.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_rule_state() -> dict:
    """Record the acceptance rule before either route is aggregated."""

    release_path = V3_ROOT / "HUMAN_TEST_RELEASE_STATE.json"
    release = load(release_path)
    if release.get("status") != "HUMAN_TEST_RELEASED":
        raise RuntimeError("human test was not formally released")
    aggregate_paths = (
        HUMAN_ROOT / "local/LOCAL_TEST_STATE.json",
        HUMAN_ROOT / "cumulative/CUMULATIVE_TEST_STATE.json",
    )
    if any(path.exists() for path in aggregate_paths):
        raise RuntimeError("route aggregate exists; acceptance rule cannot be newly frozen")
    state = {
        "contract": "topic5_event_innovation_impulse_response_v3_0_acceptance_rule",
        "status": "PRE_AGGREGATE_ACCEPTANCE_RULE_FROZEN",
        "rule": (
            "Level 2 requires all three frozen cohort medians to be positive in either "
            "Goal 2 or Goal 3 and the route's primary gain to have a patient-level "
            "two-sided Wilcoxon p-value <= 0.05; otherwise accepted V2.7 state tracking "
            "gives Level 1, else Level 0."
        ),
        "route_aggregate_outcomes_exist": False,
        "route_aggregate_outcomes_read": False,
        "patient_outputs_may_exist": True,
        "release_state_sha256": sha256(release_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(RULE_STATE, state)
    return state


def route_supports_level2(route: Mapping[str, Any]) -> bool:
    """Require a complete positive route plus inference on its primary gain."""

    inference = route["cohort_inference"]
    names = (
        ("propagation_gain", "true_minus_matched", "future_minus_past")
        if route.get("route") == "local"
        else ("cumulative_gain", "true_minus_matched", "alignment")
    )
    all_positive = all(
        inference.get(name, {}).get("n", 0) > 0
        and inference[name].get("median") is not None
        and float(inference[name]["median"]) > 0.0
        for name in names
    )
    primary = names[0]
    p_value = inference.get(primary, {}).get("wilcoxon_two_sided_p")
    return bool(
        all_positive
        and p_value is not None
        and float(p_value) <= 0.05
    )


def assign_evidence_level(
    local: Mapping[str, Any],
    cumulative: Mapping[str, Any],
    v2_7: Mapping[str, Any],
) -> tuple[int, list[str]]:
    """Return the highest level allowed by the predeclared V3.0 ladder."""

    supported = [
        name
        for name, route in (("goal2_local", local), ("goal3_cumulative", cumulative))
        if route_supports_level2(route)
    ]
    if supported:
        return 2, supported
    v2_status = v2_7.get("scientific_adjudication", {}).get("status")
    if v2_status == "ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL":
        return 1, []
    return 0, []


def patient_table(local: Mapping[str, Any], cumulative: Mapping[str, Any]) -> pd.DataFrame:
    local_rows = {row["subject"]: row for row in local["patients"]}
    cumulative_rows = {row["subject"]: row for row in cumulative["patients"]}
    if set(local_rows) != set(cumulative_rows):
        raise RuntimeError("local and cumulative patient sets differ")
    output = []
    for subject in sorted(local_rows):
        left = local_rows[subject]
        right = cumulative_rows[subject]
        record: dict[str, Any] = {
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "local_eligible": bool(left.get("eligible")),
            "cumulative_eligible": bool(right.get("eligible")),
            "local_status": left.get("status"),
            "cumulative_status": right.get("status"),
        }
        if left.get("eligible"):
            primary = left["horizons"]["20"]
            record.update({
                "local_propagation_gain": primary["observable"][
                    "propagation_gain_standardized"
                ],
                "local_true_minus_matched": primary[
                    "true_minus_state_matched_null_gain"
                ],
                "local_future_minus_past": primary["future_minus_past_state_gain"],
                "local_test_anchors": primary.get("n_test_anchors"),
            })
        if right.get("eligible"):
            primary = right["combinations"][
                str(right["primary_exposure_events"])
            ][str(right["primary_horizon"])]
            record.update({
                "cumulative_propagation_gain": primary["observable"][
                    "propagation_gain_standardized"
                ],
                "cumulative_true_minus_matched": primary[
                    "true_minus_matched_cumulative_null_gain"
                ],
                "cumulative_alignment": primary["dose_alignment"][
                    "alignment_coefficient"
                ],
                "cumulative_test_anchors": primary.get("n_test_anchors"),
            })
        output.append(record)
    return pd.DataFrame(output)


def build_acceptance() -> tuple[dict, dict, dict, pd.DataFrame]:
    paths = {
        "local": HUMAN_ROOT / "local/LOCAL_TEST_STATE.json",
        "cumulative": HUMAN_ROOT / "cumulative/CUMULATIVE_TEST_STATE.json",
        "release": V3_ROOT / "HUMAN_TEST_RELEASE_STATE.json",
        "handoff": V3_ROOT / "V3_1_HANDOFF_STATE.json",
        "v2_7": V27_ACCEPTANCE,
        "acceptance_rule": RULE_STATE,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"acceptance input missing: {missing}")
    records = {name: load(path) for name, path in paths.items()}
    local, cumulative = records["local"], records["cumulative"]
    if local.get("status") != "HUMAN_TEST_ROUTE_COMPLETE":
        raise RuntimeError("local human-test route is incomplete")
    if cumulative.get("status") != "HUMAN_TEST_ROUTE_COMPLETE":
        raise RuntimeError("cumulative human-test route is incomplete")
    if records["release"].get("status") != "HUMAN_TEST_RELEASED":
        raise RuntimeError("human test was not formally released")
    if records["handoff"].get("status") not in {"OPEN", "NOT_TRIGGERED"}:
        raise RuntimeError("V3.1 handoff is not frozen")
    if records["release"]["inputs_sha256"].get("handoff") != sha256(paths["handoff"]):
        raise RuntimeError("pre-test V3.1 handoff changed after release")
    if records["acceptance_rule"].get("status") != "PRE_AGGREGATE_ACCEPTANCE_RULE_FROZEN":
        raise RuntimeError("pre-aggregate acceptance rule is not frozen")
    if records["acceptance_rule"].get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("acceptance implementation changed after rule freeze")
    for route in (local, cumulative):
        if route.get("test_dependent_selection") is not False:
            raise RuntimeError("test-dependent selection flag is not false")
        if route.get("n_completed") != 34:
            raise RuntimeError("human-test route does not contain all 34 patients")
        if route.get("within_event_next_rank_model_fit") is not False:
            raise RuntimeError("scientific object drifted to within-event prediction")

    level, level2_routes = assign_evidence_level(local, cumulative, records["v2_7"])
    wording = {
        0: "Repeated events sample a stable patient-specific propagation repertoire.",
        1: (
            "Recent complete events help track the current repertoire state, but valid "
            "event innovations add no route-consistent future propagation information."
        ),
        2: (
            "Valid complete-event innovations predict later residual changes in the "
            "patient-specific rank/precedence state; this is a predictive association, "
            "not transition-mechanism identification."
        ),
    }[level]
    evidence = {
        "status": "EVIDENCE_LEVEL_FROZEN",
        "level": level,
        "level_name": {0: "stable_backbone", 1: "leaky_observer", 2: "innovation_predictive_association"}[level],
        "level2_supporting_routes": level2_routes,
        "decision_rule": (
            "Level 2 requires all three frozen cohort medians to be positive in either "
            "Goal 2 or Goal 3 and the route's primary gain to pass the patient-level "
            "two-sided Wilcoxon threshold of 0.05; otherwise accepted V2.7 state "
            "tracking gives Level 1."
        ),
        "allowed_wording": wording,
        "forbidden_wording": [
            "event-driven transition identified",
            "activity-dependent shaping",
            "causal plasticity",
            "within-event next-rank mechanism",
        ],
        "v3_1_handoff_status_frozen_before_test": records["handoff"]["status"],
        "v3_1_human_execution_allowed": records["handoff"].get(
            "v3_1_human_execution_allowed", False
        ),
        "human_test_cannot_reopen_v3_1": True,
    }
    inference = {
        "status": "PATIENT_FIRST_COHORT_INFERENCE_COMPLETE",
        "local": local["cohort_inference"],
        "cumulative": cumulative["cohort_inference"],
        "dataset_specific": {
            "local": local["dataset_specific"],
            "cumulative": cumulative["dataset_specific"],
        },
    }
    table = patient_table(local, cumulative)
    state = {
        "contract": "topic5_event_innovation_impulse_response_v3_0_acceptance",
        "status": "HUMAN_EXPLORATORY_COMPLETE",
        "n_patients": 34,
        "n_local_eligible": int(local["n_eligible"]),
        "n_cumulative_eligible": int(cumulative["n_eligible"]),
        "evidence_level": evidence,
        "cohort_inference": inference,
        "v3_1_handoff": {
            "status": records["handoff"]["status"],
            "human_execution_allowed": records["handoff"].get(
                "v3_1_human_execution_allowed", False
            ),
        },
        "inputs_sha256": {name: sha256(path) for name, path in paths.items()},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "human_test_outcomes_read": True,
        "test_dependent_selection": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    return state, inference, evidence, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-rule-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_rule_only:
        state = freeze_rule_state()
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    state, inference, evidence, table = build_acceptance()
    atomic_json(HUMAN_ROOT / "HUMAN_EXPLORATORY_STATE.json", state)
    atomic_json(HUMAN_ROOT / "cohort_inference.json", inference)
    atomic_json(HUMAN_ROOT / "evidence_level.json", evidence)
    table.to_csv(HUMAN_ROOT / "patient_summary.csv", index=False)
    print(json.dumps({
        "status": state["status"],
        "evidence_level": evidence["level"],
        "v3_1_handoff": state["v3_1_handoff"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
