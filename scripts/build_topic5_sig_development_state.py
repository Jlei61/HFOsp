#!/usr/bin/env python3
"""Build the fail-closed machine verdict for SIG-RNN v2 development."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import sha256_file  # noqa: E402


RESULT_ROOT = ROOT / "results/topic5_stable_interaction_graph/development"
SOURCES = {
    "identifiability_audit": (
        RESULT_ROOT / "identifiability_audit/cohort_identifiability.json"
    ),
    "g0a_initial": RESULT_ROOT / "synthetic_g0a/G0A_BENCHMARK.json",
    "g0a_failure_diagnostic": (
        RESULT_ROOT
        / "synthetic_g0a_diagnostics/round1_endpoint_decomposition"
        / "endpoint_decomposition.json"
    ),
    "g0a_learning_curve": (
        RESULT_ROOT
        / "synthetic_g0a_diagnostics/round2_nested_event_count"
        / "nested_event_count.json"
    ),
    "g0a2_independent": (
        RESULT_ROOT
        / "synthetic_g0a2_independent_confirmation_v0_2_training_adequacy"
        / "G0A2_CONFIRMATION.json"
    ),
    "human_graph_increment": (
        RESULT_ROOT
        / "human_graph_increment_pilot_v0_3_provenance"
        / "HUMAN_GRAPH_INCREMENT_PILOT.json"
    ),
    "matched_baseline_ladder": (
        RESULT_ROOT
        / "human_matched_baseline_ladder_v0_3_provenance"
        / "MATCHED_BASELINE_LADDER.json"
    ),
}
# The v0_2 runs produced the same numbers but recorded an aggregation-time
# source hash that did not describe their own fits, so they are kept as the
# original run and are no longer the state's source of truth.
SUPERSEDED = {
    "human_graph_increment": (
        RESULT_ROOT
        / "human_graph_increment_pilot_v0_2_training_adequacy"
        / "HUMAN_GRAPH_INCREMENT_PILOT.json"
    ),
    "matched_baseline_ladder": (
        RESULT_ROOT
        / "human_matched_baseline_ladder_v0_2_training_adequacy"
        / "MATCHED_BASELINE_LADDER.json"
    ),
}


def _load(name: str) -> dict:
    path = SOURCES[name]
    if not path.exists():
        raise RuntimeError(f"missing SIG development artifact: {path}")
    return json.loads(path.read_text())


def _run_artifact_checks(
    increment: dict, ladder: dict, g0a2: dict
) -> dict[str, bool]:
    increment_root = SOURCES["human_graph_increment"].parent / "per_run"
    ladder_root = SOURCES["matched_baseline_ladder"].parent / "per_run"
    increment_complete = True
    increment_optimizer = True
    for row in increment["run_rows"]:
        run = increment_root / row["subject"] / f"seed_{row['fit_seed']}"
        required = (
            run / "summary.json",
            run / "checkpoint.pt",
            run / "conditioned_generation.npz",
            run / "sig0_history.json",
            run / "sig1_history.json",
        )
        increment_complete &= all(path.is_file() for path in required)
        if (run / "checkpoint.pt").is_file():
            checkpoint = torch.load(
                run / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            increment_optimizer &= all(
                key in checkpoint
                for key in ("sig0_optimizer_state", "sig1_optimizer_state")
            )
    ladder_complete = True
    ladder_optimizer = True
    for row in ladder["run_rows"]:
        run = ladder_root / row["subject"] / f"seed_{row['fit_seed']}"
        required = (
            run / "summary.json",
            run / "checkpoint.pt",
            run / "conditioned_generation.npz",
        )
        ladder_complete &= all(path.is_file() for path in required)
        if (run / "checkpoint.pt").is_file():
            checkpoint = torch.load(
                run / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            models = checkpoint.get("models", {})
            ladder_optimizer &= bool(models) and all(
                value.get("optimizer_state") is not None
                for value in models.values()
            )
    g0a2_complete = True
    g0a2_optimizer = True
    g0a2_root = SOURCES["g0a2_independent"].parent / "per_run"
    for row in g0a2["runs"]:
        run = g0a2_root / f"seed_{row['fit_seed']}"
        required = (run / "summary.json", run / "checkpoint.pt")
        g0a2_complete &= all(path.is_file() for path in required)
        if (run / "checkpoint.pt").is_file():
            checkpoint = torch.load(
                run / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            g0a2_optimizer &= all(
                key in checkpoint
                for key in ("sig0_optimizer_state", "sig1_optimizer_state")
            )
    return {
        "human_increment_run_artifacts_complete": increment_complete,
        "human_increment_optimizer_states_present": increment_optimizer,
        "matched_ladder_run_artifacts_complete": ladder_complete,
        "matched_ladder_optimizer_states_present": ladder_optimizer,
        "g0a2_run_artifacts_complete": g0a2_complete,
        "g0a2_optimizer_states_present": g0a2_optimizer,
    }


def main() -> None:
    values = {name: _load(name) for name in SOURCES}
    audit = values["identifiability_audit"]
    g0a = values["g0a_initial"]
    diagnostic = values["g0a_failure_diagnostic"]
    curve = values["g0a_learning_curve"]
    g0a2 = values["g0a2_independent"]
    increment = values["human_graph_increment"]
    ladder = values["matched_baseline_ladder"]
    artifact_checks = _run_artifact_checks(increment, ladder, g0a2)

    required = {
        "audit_complete": audit.get("status") == "COMPLETE",
        "initial_g0a_preserved_fail": g0a.get("status") == "FAIL_CLOSED",
        "failure_diagnostic_did_not_regate": (
            diagnostic.get("g0a_status_unchanged") == "FAIL_CLOSED"
        ),
        "learning_curve_did_not_regate": (
            curve.get("g0a_status_unchanged") == "FAIL_CLOSED"
        ),
        "independent_g0a2_passed": g0a2.get("status") == "PASS",
        "independent_g0a2_kept_original_fail": (
            g0a2.get("original_g0a_status_unchanged") == "FAIL_CLOSED"
        ),
        "human_increment_complete": increment.get("status") == "COMPLETE",
        "human_increment_all_training_adequate": (
            bool(increment.get("all_training_adequate"))
            and int(increment.get("n_model_fits", -1)) == 36
        ),
        "human_increment_not_mislabeled_g1": bool(increment.get("not_g1")),
        "matched_ladder_complete": ladder.get("status") == "COMPLETE",
        "matched_ladder_all_training_adequate": (
            bool(ladder.get("all_training_adequate"))
            and int(ladder.get("n_model_fits", -1)) == 54
        ),
        "matched_ladder_estimator_contract_valid": bool(
            ladder.get("likelihood_estimator_contract_valid")
        ),
        "matched_ladder_m2_components_separated": (
            bool(ladder.get("m2_components_separated"))
            and float(
                ladder.get("m2_min_component_parameter_distance", 0.0)
            )
            > 1e-6
        ),
        "matched_ladder_stopped_structure_claim": (
            ladder.get("decision") == "STOP_BEFORE_STRUCTURE_CLAIM"
        ),
        "matched_ladder_stop_robust_to_rollout_resampling": (
            bool(ladder.get("stop_robust_to_rollout_resampling"))
            and int(ladder.get("max_possible_both_given_nll", 99)) < 4
        ),
        # Spec section 10: an aggregate whose recorded source does not
        # describe its own fits is not reproducible evidence.
        "human_artifacts_carry_fit_time_provenance": all(
            bool(value.get("fit_time_source_sha256"))
            and value["fit_time_source_sha256"].get("runner")
            == value.get("aggregation_source_sha256")
            for value in (increment, ladder)
        ),
        "matched_ladder_records_unevaluated_g1_clause": bool(
            ladder.get("g1_clauses_not_evaluated")
        ),
        "no_snn_inputs": all(
            not bool(value.get("snn_inputs_read", False))
            for value in values.values()
        ),
        "no_outer_heldout20_scored": all(
            not bool(value.get("old_heldout20_scored", False))
            for value in (increment, ladder)
        ),
        **artifact_checks,
    }
    if not all(required.values()):
        raise RuntimeError(
            "SIG development state is inconsistent: "
            + repr([name for name, passed in required.items() if not passed])
        )
    payload = {
        "contract": "topic5_stable_interaction_graph_rnn_v2",
        "status": "COMPLETE_BOUNDED_DEVELOPMENT",
        "scientific_verdict": (
            "FEEDBACK_GRAPH_HAS_INCREMENT_OVER_PHASE_ONLY_NOGRAPH_BUT_IS_NOT_"
            "SELECTED_OVER_THE_STRONGEST_PHASE_MATCHED_MIXTURE_OR_TEMPLATE"
        ),
        "gate_verdict": {
            "generic_synthetic_engineering_calibration": (
                "PASS_AT_N_MIN_9600_ON_INDEPENDENT_GRAPH"
            ),
            "human_graph_increment_screen": "PASS_6_OF_6",
            "g1_full_event_sufficiency": "NOT_PASSED_DEVELOPMENT",
            "g2_structure_stability": "LOCKED_NOT_RUN",
            "g3_one_structure_many_trajectories": "LOCKED_NOT_RUN",
            "g4_unseen_start_human": "LOCKED_NOT_RUN",
            "g5_full_cohort_or_replication": "LOCKED_NOT_RUN",
            "snn_gate": "ABSENT_BY_CONTRACT",
        },
        "key_counts": {
            "human_generation_eligible": audit[
                "n_generation_adequacy_eligible"
            ],
            "human_unseen_start_eligible": audit[
                "n_unseen_start_eligible"
            ],
            "sig1_vs_sig0_both_better": increment["counts"][
                "n_patients_both_better"
            ],
            "sig1_vs_strongest_baseline_nll_better": ladder["counts"][
                "sig1_nll_better_than_all_baselines"
            ],
            "sig1_vs_strongest_baseline_precedence_better": ladder["counts"][
                "sig1_precedence_better_than_all_baselines"
            ],
            "sig1_vs_strongest_baseline_both_better": ladder["counts"][
                "sig1_both_better_than_all_baselines"
            ],
        },
        "safe_claim": (
            "On six development patients, emitted-contact feedback through a "
            "shared contact-space graph improved likelihood and free-rollout "
            "precedence relative to an otherwise matched phase-only no-graph "
            "model. It did not consistently outperform the strongest "
            "phase-matched Markov mixture or latent time template, so the data "
            "do not establish that a shared stable graph is necessary."
        ),
        "forbidden_claims": [
            "a stable patient-specific interaction graph was identified",
            "one shared structure explains the full human repertoire",
            "human effective connectivity was recovered",
            "SNN validates or gates the RNN",
            "G1 was evaluated in full and failed",
        ],
        "g1_boundary": {
            "executed_rule": ladder["decision_rule_executed"],
            "rule_provenance": ladder["decision_rule_provenance"],
            "clauses_not_evaluated": ladder["g1_clauses_not_evaluated"],
            "reading_sensitivity": (
                "Under the literal spec section 8 reading (NLL non-inferior at "
                "0.01 nats/decision plus rollout better than M2-phase and M3) "
                "only 2 of 6 patients satisfy both endpoints, so the stop does "
                "not depend on the stricter executed rule."
            ),
        },
        "checks": required,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in SOURCES.items()
        },
        "superseded_artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "reason": (
                    "same numbers, but the aggregate recorded an "
                    "aggregation-time source hash that did not describe its "
                    "own fits; kept as the original run"
                ),
            }
            for name, path in SUPERSEDED.items()
            if path.exists()
        },
        "source_sha256": sha256_file(Path(__file__)),
    }
    target = RESULT_ROOT / "SIG_V2_DEVELOPMENT_STATE.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["gate_verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
