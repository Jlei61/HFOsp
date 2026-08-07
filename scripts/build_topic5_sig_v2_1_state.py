#!/usr/bin/env python3
"""Build the fail-closed v2.1 identifiability state without rewriting v2 history."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import sha256_file  # noqa: E402


RESULT_ROOT = ROOT / "results/topic5_stable_interaction_graph/development"
SOURCES = {
    "identifiability_audit": (
        RESULT_ROOT / "identifiability_audit/cohort_identifiability.json"
    ),
    "human_graph_increment": (
        RESULT_ROOT
        / "human_graph_increment_pilot_v0_3_provenance"
        / "HUMAN_GRAPH_INCREMENT_PILOT.json"
    ),
    "matched_baseline_oracle_stress_test": (
        RESULT_ROOT
        / "human_matched_baseline_ladder_v0_2_training_adequacy"
        / "MATCHED_BASELINE_LADDER.json"
    ),
    "historical_v2_state": RESULT_ROOT / "SIG_V2_DEVELOPMENT_STATE.json",
    "d1_existing_artifact_audit": (
        RESULT_ROOT
        / "v2_1_existing_artifact_audit"
        / "D1_EXISTING_ARTIFACT_AUDIT.json"
    ),
    "d2_m2_operator_audit": (
        RESULT_ROOT
        / "v2_1_m2_operator_audit"
        / "D2_M2_OPERATOR_AUDIT.json"
    ),
    "d0_patient_matched_identifiability": (
        RESULT_ROOT
        / "v2_1_patient_matched_identifiability"
        / "D0_PATIENT_MATCHED_IDENTIFIABILITY.json"
    ),
    "d3_split_stability": (
        RESULT_ROOT / "v2_1_split_stability" / "D3_SPLIT_STABILITY.json"
    ),
    "d3_test_probe_rescore": (
        RESULT_ROOT
        / "v2_1_split_stability_test_probe_rescore"
        / "D3_TEST_PROBE_RESCORE.json"
    ),
    "d4_unseen_start": (
        RESULT_ROOT / "v2_1_unseen_start" / "D4_UNSEEN_START.json"
    ),
}


def _load(name: str) -> dict:
    path = SOURCES[name]
    if not path.is_file():
        raise RuntimeError(f"missing v2.1 source artifact: {path}")
    return json.loads(path.read_text())


def main() -> None:
    values = {name: _load(name) for name in SOURCES}
    audit = values["identifiability_audit"]
    increment = values["human_graph_increment"]
    ladder = values["matched_baseline_oracle_stress_test"]
    d1 = values["d1_existing_artifact_audit"]
    d2 = values["d2_m2_operator_audit"]
    d0 = values["d0_patient_matched_identifiability"]
    d3 = values["d3_split_stability"]
    d3_rescore = values["d3_test_probe_rescore"]
    d4 = values["d4_unseen_start"]
    checks = {
        "audit_complete": audit.get("status") == "COMPLETE",
        "feedback_increment_complete": increment.get("status") == "COMPLETE",
        "feedback_increment_is_not_g1": bool(increment.get("not_g1")),
        "feedback_increment_both_6_of_6": (
            increment.get("counts", {}).get("n_patients_both_better") == 6
        ),
        "historical_ladder_complete": ladder.get("status") == "COMPLETE",
        "historical_ladder_test_oracle_recognized": all(
            key in ladder.get("patient_rows", [{}])[0]
            for key in ("best_baseline_nll", "best_baseline_precedence_mae")
        ),
        "no_outer_heldout20_scored": not bool(
            ladder.get("old_heldout20_scored", True)
            or increment.get("old_heldout20_scored", True)
        ),
        "no_snn_inputs": not bool(
            ladder.get("snn_inputs_read", True)
            or increment.get("snn_inputs_read", True)
        ),
        "d1_complete": d1.get("status") == "COMPLETE_EXISTING_ARTIFACT_AUDIT",
        "d2_complete": d2.get("status") == "COMPLETE_EXISTING_M2_OPERATOR_AUDIT",
        "d0_complete": d0.get("status") == "COMPLETE_PATIENT_MATCHED_IDENTIFIABILITY",
        "d0_all_training_adequate": bool(d0.get("all_training_adequate")),
        "d3_complete": d3.get("status") == "COMPLETE_SPLIT_STABILITY_DEVELOPMENT",
        "d3_all_training_adequate": bool(d3.get("all_training_adequate")),
        "d3_test_probe_rescore_complete": (
            d3_rescore.get("status") == "COMPLETE_TEST_PROBE_SENSITIVITY"
        ),
        "d3_test_probe_rescore_no_outer_heldout20": not bool(
            d3_rescore.get("old_heldout20_scored", True)
        ),
        "d3_test_probe_rescore_no_snn": not bool(
            d3_rescore.get("snn_inputs_read", True)
        ),
        "d4_complete": d4.get("status") == "COMPLETE_UNSEEN_START_DEVELOPMENT",
        "d4_all_training_adequate": bool(d4.get("all_training_adequate")),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "v2.1 state source inconsistency: "
            + repr([name for name, passed in checks.items() if not passed])
        )

    payload = {
        "contract": "topic5_stable_interaction_identifiability_v2_1",
        "status": "COMPLETE_BOUNDED_SINGLE_GRAPH_DEVELOPMENT",
        "scientific_verdict": (
            "FEEDBACK_INCREMENT_PRESENT; CURRENT_SINGLE_FIXED_GRAPH_HAS_NO_"
            "REAL_OVER_MATCHED_NULL_STABILITY_SIGNAL_IN_FOUR_CALIBRATED_"
            "PATIENTS; TWO_PATIENTS_REMAIN_UNADJUDICATED"
        ),
        "gate_correction": {
            "historical_v2_predictive_rule": (
                "ENDPOINT_SPECIFIC_DEVELOPMENT_TEST_ORACLE_STRESS_TEST_ONLY"
            ),
            "historical_v2_state": "PRESERVED_AS_HISTORY_NOT_CURRENT_GATE",
            "structure_gates": "RUN_UNDER_V2_1_AND_BOUNDED_CLOSED",
        },
        "diagnostics": {
            "feedback_increment": "PRESENT_6_OF_6",
            "seen_distribution_predictive_dominance": (
                "NOT_ESTABLISHED_ORACLE_STRESS_TEST"
            ),
            "generation_diversity_adequacy": (
                "NO_COLLAPSE_MILD_OVERDISPERSION_IN_FIVE_OF_SIX"
            ),
            "d0_patient_matched_sensitivity_specificity": (
                "PASS_4_OF_6_TWO_UNADJUDICATED"
            ),
            "d1_baseline_envelope_diversity": "COMPLETE_DIAGNOSTIC_ONLY",
            "d2_m2_observable_operators": (
                "SEED_STABILITY_PRESENT_SPLIT_AND_NULL_OPEN"
            ),
            "d3_real_over_null_temporal_stability": (
                "NO_SIGNAL_0_OF_6; NO_SIGNAL_0_OF_4_CALIBRATED; "
                "UNCHANGED_ON_CHECKPOINT_INDEPENDENT_TEST_PROBE"
            ),
            "d4_unseen_start_compositional_generalization": (
                "MIXED_NLL_ONLY; BOTH_ENDPOINTS_2_OF_6"
            ),
            "d5_shared_backbone_modulation": (
                "NOT_AUTHORIZED_NO_STRUCTURE_SPECIFIC_SIGNAL"
            ),
            "full_cohort_or_replication": "LOCKED_NOT_RUN",
            "snn_gate": "ABSENT_BY_CONTRACT",
        },
        "key_counts": {
            "human_generation_eligible": audit[
                "n_generation_adequacy_eligible"
            ],
            "human_unseen_start_eligible": audit[
                "n_unseen_start_eligible"
            ],
            "feedback_increment_both_better": increment["counts"][
                "n_patients_both_better"
            ],
            "historical_oracle_nll_better": ladder["counts"][
                "sig1_nll_better_than_all_baselines"
            ],
            "historical_oracle_precedence_better": ladder["counts"][
                "sig1_precedence_better_than_all_baselines"
            ],
            "historical_oracle_both_better": ladder["counts"][
                "sig1_both_better_than_all_baselines"
            ],
            "validation_selected_baseline_both_better": d1[
                "validation_selected_baseline_counts"
            ]["both_better"],
            "future_schedule_route_increment_positive": d1[
                "future_schedule_route_audit"
            ]["n_positive_increment"],
            "d0_patient_matched_pass": d0["n_pass"],
            "d3_real_minus_null_positive": d3["real_minus_strongest_null"][
                "n_positive"
            ],
            "d3_test_probe_real_minus_null_positive": d3_rescore[
                "real_minus_strongest_null"
            ]["n_positive"],
            "d4_unseen_nll_better": d4["counts"][
                "sig1_unseen_nll_better"
            ],
            "d4_unseen_precedence_better": d4["counts"][
                "sig1_unseen_precedence_better"
            ],
            "d4_unseen_both_better": d4["counts"][
                "sig1_unseen_both_better"
            ],
        },
        "diagnostic_findings": {
            "future_schedule_route_balanced_accuracy_increment_median": d1[
                "future_schedule_route_audit"
            ]["median_balanced_accuracy_increment"],
            "m2_component_seed_stability_median": d2["cohort_summary"][
                "component_seed_stability_median"
            ],
            "m2_backbone_seed_stability_median": d2["cohort_summary"][
                "backbone_seed_stability_median"
            ],
            "d3_real_minus_strongest_null_median": d3[
                "real_minus_strongest_null"
            ]["median"],
            "d3_test_probe_real_minus_strongest_null_median": d3_rescore[
                "real_minus_strongest_null"
            ]["median"],
            "boundary": (
                "M2 seed stability did not translate into real-over-null "
                "temporal stability, including on an inner-test probe that "
                "did not select the saved checkpoints. It remains a "
                "descriptive lead, not authorization for a "
                "modulated-backbone model."
            ),
        },
        "safe_claim": (
            "Contact feedback improved conditional suffix generation relative "
            "to a matched no-feedback model. In four pilot patients where "
            "patient-matched calibration could distinguish a fixed graph from "
            "phase, mixture, and event-random negatives, real chronological "
            "influence stability did not exceed matched local/phase nulls. "
            "Two additional patients remained unadjudicated. Unseen-start NLL "
            "improved in five of six patients, but full precedence improved in "
            "only two, so the current single-fixed-graph formulation is closed "
            "without authorizing a higher-capacity modulation model."
        ),
        "forbidden_claims": [
            "stable graph failed",
            "shared interaction structure is absent or unnecessary",
            "mixture or template performance proves there is no network structure",
            "the historical predictive oracle is a structure gate",
            "SNN validates or gates the RNN",
            "all six patients lack stable structure",
            "the two D0-ineligible patients are negative",
            "M2 seed stability alone supports stable propagation regimes",
        ],
        "checks": checks,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in SOURCES.items()
        },
        "source_sha256": sha256_file(Path(__file__)),
    }
    target = RESULT_ROOT / "SIG_V2_1_IDENTIFIABILITY_STATE.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["diagnostics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
