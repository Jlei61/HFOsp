#!/usr/bin/env python3
"""Build the machine-readable overall acceptance for the Topic 5 RNN line."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json"
)

SOURCES = {
    "path_mode": ROOT
    / (
        "results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0/"
        "analysis/formal_gate_summary.json"
    ),
    "competitive": ROOT
    / (
        "results/topic5_symmetric_axis_competitive_propagation_v2_3/formal/"
        "FORMAL_GATE_STATUS.json"
    ),
    "linear_symmetric_closeout": ROOT
    / (
        "results/topic5_symmetric_axis_propagation_state_v2_2/"
        "closeout_v2_2_1/CLOSEOUT_STATUS.json"
    ),
    "transition_decomposition": ROOT
    / (
        "results/topic5_interictal_transition_decomposition_v0_1/"
        "DECOMPOSITION_STATUS.json"
    ),
    "axis_selection": ROOT
    / (
        "results/topic5_rnn_axis_positive_static_transfer_v2_4/formal/"
        "AXIS_SELECTION_GATE_STATUS.json"
    ),
    "axis_static_readout": ROOT
    / (
        "results/topic5_rnn_axis_positive_static_transfer_v2_4/static_readout/"
        "STATIC_READOUT_GATE_STATUS.json"
    ),
    "internal_state": ROOT
    / "results/topic5_rnn_internal_state_reduction/FINAL_STATUS.json",
    "fixed_static": ROOT
    / (
        "results/topic5_static_scaffold_fixed_readout_validation/"
        "FINAL_ACCEPTANCE.json"
    ),
    "static_reliability": ROOT
    / (
        "results/topic5_interictal_scaffold_reliability_history_necessity/"
        "static_reliability_v0_1/summary.json"
    ),
    "history_depth": ROOT
    / (
        "results/topic5_interictal_scaffold_reliability_history_necessity/"
        "history_runs_v0_1/history_necessity_summary.json"
    ),
    "matched_h3_shuffle": ROOT
    / (
        "results/topic5_interictal_scaffold_reliability_history_necessity/"
        "history3_rank_shuffle_runs_v0_1/"
        "history3_rank_shuffle_summary.json"
    ),
    "full_rank_reference": ROOT
    / (
        "results/topic5_interictal_rank_distribution/"
        "full_rank_reference_analysis/full_rank_reference_summary.json"
    ),
    "low_rank_sweep": ROOT
    / (
        "results/topic5_low_rank_dynamics/analysis/"
        "structured_rank_sweep_v1/analysis_summary.json"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    data = {name: _load(path) for name, path in SOURCES.items()}

    path_mode = data["path_mode"]
    competitive = data["competitive"]
    linear_closeout = data["linear_symmetric_closeout"]
    decomposition = data["transition_decomposition"]
    axis_selection = data["axis_selection"]
    axis_static = data["axis_static_readout"]
    internal = data["internal_state"]
    fixed = data["fixed_static"]
    static = data["static_reliability"]
    history = data["history_depth"]
    matched = data["matched_h3_shuffle"]
    full_rank = data["full_rank_reference"]
    low_rank = data["low_rank_sweep"]

    _require(
        path_mode["status"] == "complete"
        and path_mode["n_patients"] == 34
        and path_mode["n_runs"] == 510
        and path_mode["ictal_target_read"] is False,
        "persistent path-mode formal artifact is incomplete or unsealed",
    )
    _require(
        competitive["status"] == "COMPLETE"
        and competitive["n_physical_axis_patients"] == 22
        and competitive["target_values_read"] is False,
        "competitive structured-RNN artifact is incomplete or unsealed",
    )
    _require(
        linear_closeout["status"] == "COMPLETE"
        and linear_closeout["target_values_read"] is False,
        "linear symmetric-axis closeout is incomplete or unsealed",
    )
    _require(
        decomposition["status"] == "COMPLETE"
        and decomposition["target_values_read"] is False,
        "transition decomposition is incomplete or unsealed",
    )
    _require(
        axis_selection["status"] == "COMPLETE"
        and axis_selection["target_values_read"] is False,
        "axis-selection artifact is incomplete or read the target",
    )
    _require(
        axis_static["status"] == "COMPLETE"
        and axis_static["target_values_read"] is True,
        "axis static-readout target status is inconsistent",
    )
    _require(
        internal["status"] == "COMPLETE"
        and internal["audit"]["status"] == "PASS",
        "internal-state reduction is not complete",
    )
    _require(
        fixed["status"] == "PASS_WITH_BOUNDED_STATIC_CONCLUSION",
        "fixed static-scaffold acceptance is not complete",
    )
    _require(
        static["status"] == "complete"
        and static["n_patients"] == 34
        and static["ictal_target_read"] is False,
        "static reliability artifact is incomplete or unsealed",
    )
    _require(
        history["status"] == "complete"
        and history["n_patients"] == 34
        and history["n_formal_folds"] == 102
        and history["ictal_target_read"] is False,
        "finite-history formal artifact is incomplete or unsealed",
    )
    _require(
        matched["status"] == "complete"
        and matched["n_patients"] == 34
        and matched["n_folds"] == 102
        and matched["ictal_target_read"] is False,
        "matched H3 rank-shuffle artifact is incomplete or unsealed",
    )
    _require(
        full_rank["status"] == "complete"
        and full_rank["n_patients"] == 34
        and full_rank["ictal_target_read"] is False,
        "full-rank reference is incomplete or unsealed",
    )
    _require(
        low_rank["status"] == "complete"
        and low_rank["n_patients"] == 34
        and low_rank["ictal_target_read"] is False,
        "low-rank sweep is incomplete or unsealed",
    )

    history_contrasts = history["contrasts"]
    fixed_science = fixed["scientific_acceptance"]
    result = {
        "contract": "topic5_rnn_overall_acceptance_v1_0",
        "status": "ACCEPTED_AS_BOUNDED_SUPPLEMENTARY_COMPUTATIONAL_RESULT",
        "date": "2026-07-28",
        "scope": (
            "interictal rank-sequence self-supervision, structured-RNN "
            "falsification, and reused-target early-ictal static readout"
        ),
        "execution_integrity": {
            "status": "PASS",
            "components": {
                "persistent_path_mode": {
                    "status": "COMPLETE_BOUNDED_NEGATIVE",
                    "n_patients": path_mode["n_patients"],
                    "n_seeds": path_mode["n_seeds"],
                    "n_runs": path_mode["n_runs"],
                    "ictal_target_read": path_mode["ictal_target_read"],
                },
                "competitive_structured_rnn": {
                    "status": "COMPLETE_TO_PREREGISTERED_STOP",
                    "n_patients": competitive["n_physical_axis_patients"],
                    "n_seeds": competitive["n_seeds"],
                    "target_values_read": competitive["target_values_read"],
                },
                "linear_symmetric_observation_model": {
                    "status": "COMPLETE_BOUNDED_NEGATIVE",
                    "n_patients": linear_closeout["cohort_endpoints"][
                        "full_benefit_over_node"
                    ]["n_patients"],
                    "target_values_read": linear_closeout[
                        "target_values_read"
                    ],
                },
                "transition_signal_decomposition": {
                    "status": decomposition["status"],
                    "decision_at_time": decomposition["decision"],
                    "coordinate_free_patients": decomposition[
                        "coordinate_free_patients"
                    ],
                    "physical_axis_patients": decomposition[
                        "physical_axis_patients"
                    ],
                    "target_values_read": decomposition[
                        "target_values_read"
                    ],
                },
                "axis_selection_and_static_readout": {
                    "axis_selection_status": axis_selection[
                        "gate_a_axis_positive_construct_validity"
                    ],
                    "axis_selection_target_read": axis_selection[
                        "target_values_read"
                    ],
                    "static_readout_status": axis_static["status"],
                    "static_readout_target_read": axis_static[
                        "target_values_read"
                    ],
                },
                "internal_state_reduction": {
                    "status": internal["postreview_status"],
                    "n_interictal_patients": internal["audit"]["completeness"][
                        "subject_analyses"
                    ]["complete"],
                    "n_early_ictal_patients": internal["audit"][
                        "strict_early_ictal"
                    ]["n_patients"],
                    "n_seizures": internal["audit"]["strict_early_ictal"][
                        "n_seizures"
                    ],
                },
                "fixed_static_readout": {
                    "status": fixed["status"],
                    "n_patients": fixed["execution"]["n_patients"],
                    "n_seizures": fixed["execution"]["n_seizures"],
                    "target_reused_not_independent_confirmation": fixed[
                        "execution"
                    ]["target_reused_not_independent_confirmation"],
                },
                "finite_history": {
                    "status": "COMPLETE",
                    "n_patients": history["n_patients"],
                    "n_seeds": history["n_seeds"],
                    "n_new_models": history["n_new_models"]
                    + matched["n_folds"],
                    "ictal_target_read": history["ictal_target_read"],
                },
                "unconstrained_full_rank_reference": {
                    "status": full_rank["status"],
                    "n_patients": full_rank["n_patients"],
                    "n_seeds": full_rank["n_seeds"],
                    "ictal_target_read": full_rank["ictal_target_read"],
                },
                "low_rank_sweep": {
                    "status": low_rank["status"],
                    "n_patients": low_rank["n_patients"],
                    "n_seeds": low_rank["n_seeds"],
                    "ranks": low_rank["ranks"],
                    "ictal_target_read": low_rank["ictal_target_read"],
                },
            },
        },
        "scientific_acceptance": {
            "static_interictal_contact_scaffold": {
                "status": "SUPPORTED_TARGET_BLIND",
                "estimand": "train80_vs_heldout20_contact_participation_spearman",
                "n": static["n_patients"],
                "median": static["primary"]["median"],
                "ci95": static["primary"]["ci95"],
                "n_positive": static["primary"]["n_positive"],
                "wilcoxon_greater_p": static["primary"][
                    "wilcoxon_greater_p"
                ],
                "boundary": (
                    "a reproducible participation topography, not a full rank "
                    "distribution, physical axis, or propagation direction"
                ),
            },
            "short_ordered_history": {
                "status": "SUPPORTED_TARGET_BLIND",
                "history2_over_history1": history_contrasts[
                    "gain_history2_over_history1"
                ],
                "history3_over_history2": history_contrasts[
                    "gain_history3_over_history2"
                ],
                "ordered_h3_over_matched_shuffle": {
                    "median": matched["median_ordered_gain"],
                    "ci95": matched["ordered_gain_ci95"],
                    "n_positive": matched["n_positive"],
                    "n_patients": matched["n_patients"],
                    "wilcoxon_two_sided_p": matched[
                        "wilcoxon_two_sided_p"
                    ],
                },
                "boundary": (
                    "the useful sequence memory is concentrated in the latest "
                    "two to three rank sets"
                ),
            },
            "unbounded_full_history": {
                "status": "NOT_SUPPORTED",
                "full_over_history3": history_contrasts[
                    "gain_full_over_history3"
                ],
                "full_over_strongest_nonrecurrent": fixed_science[
                    "formal_ordered_history_gain"
                ]["full_vs_strongest_nonrecurrent_heldout_nll"],
                "boundary": (
                    "full-history GRU is not required and is not the accepted "
                    "mechanistic object"
                ),
            },
            "positive_low_rank_recurrent_modes": {
                "status": "NOT_SUPPORTED_BY_TESTED_PARAMETERIZATION",
                "minimum_distribution_sufficient_rank": low_rank[
                    "pre_registered_distribution_sufficient_rank"
                ],
                "positive_mode_behavioral_support": low_rank[
                    "positive_low_rank_mode_behavioral_support"
                ],
                "rank1_loading_seed_similarity": low_rank[
                    "positive_mode_diagnostics"
                ][
                    "rank1_u_loading_median_chance_adjusted_seed_similarity"
                ],
                "boundary": (
                    "rank-0 retained diagonal recurrent memory, so this sweep "
                    "is a sensitivity analysis and not a clean low-rank "
                    "mechanism test"
                ),
            },
            "structured_path_or_axis_mechanism": {
                "status": "NOT_SUPPORTED_BY_CURRENT_MODEL_FAMILIES",
                "linear_symmetric_predictive_adequacy": linear_closeout[
                    "claim1_predictive_adequacy"
                ],
                "persistent_path_comparison_gate": path_mode[
                    "comparison_gate_pass"
                ],
                "persistent_path_structure_gate": path_mode[
                    "structure_gate_pass"
                ],
                "competition_state": competitive[
                    "claim_B_competition_vs_one_state"
                ],
                "physical_axis": competitive[
                    "claim_C_matched_axis_increment"
                ],
                "source_conditioned_direction": competitive[
                    "claim_D_source_conditioned_direction"
                ],
                "axis_readback_construct_validity": axis_selection[
                    "gate_a_axis_positive_construct_validity"
                ],
                "boundary": (
                    "failure applies to the tested observation mappings; it "
                    "does not prove that patients lack a pathological axis"
                ),
            },
            "empirical_transition_signal": {
                "status": "SUPPORTED_AS_TARGET_BLIND_DESIGN_DIAGNOSTIC",
                "decomposition_decision_at_time": decomposition["decision"],
                "go_conditions": decomposition["go_conditions"],
                "later_structured_model_outcome": (
                    "v2.3 predictive history passed, but competition, matched "
                    "axis, and source-conditioned direction failed"
                ),
                "boundary": (
                    "the decomposition justified testing v2.3; it is not "
                    "independent evidence that the v2.3 mechanism is correct"
                ),
            },
            "early_ictal_static_contact_morphology": {
                "status": "SUPPORTED_WITHIN_REUSED_TARGET_DATASET",
                "target": (
                    "clinical onset [0,10] s, 1-150 Hz baseline-normalized "
                    "contact energy"
                ),
                "n_patients": fixed["execution"]["n_patients"],
                "n_seizures": fixed["execution"]["n_seizures"],
                "full_gru_sign_free": fixed_science[
                    "static_sign_free_morphology"
                ]["full_gru_all_contact"],
                "raw_train80_sign_free": fixed_science[
                    "static_sign_free_morphology"
                ]["raw_train80_all_contact"],
                "boundary": (
                    "internal same-dataset sign-free contact morphology, not "
                    "a fixed positive field direction or independent transfer "
                    "confirmation"
                ),
            },
            "fixed_positive_cross_state_direction": {
                "status": "NOT_ESTABLISHED",
                "full_gru_all_contact": fixed_science[
                    "static_signed_correspondence"
                ]["full_gru_all_contact"],
            },
            "gru_specific_static_transfer": {
                "status": "NOT_ESTABLISHED",
                "full_minus_best_regularized": fixed_science[
                    "gru_specific_static_increment"
                ]["full_minus_best_regularized"],
                "full_minus_rank_shuffle": fixed_science[
                    "gru_specific_static_increment"
                ]["full_minus_rank_shuffle"],
            },
            "dynamic_seizure_prediction_or_replay": {
                "status": "NOT_TESTED_AND_NOT_ALLOWED",
                "reason": (
                    "the accepted target is a static early-ictal contact field; "
                    "exact per-seizure onset-source metadata are unavailable "
                    "for source-conditioned dynamic transfer"
                ),
            },
        },
        "paper_acceptance": {
            "tier": "SUPPLEMENTARY_BOUNDED_COMPUTATIONAL_RESULT",
            "canonical_claim": (
                "Interictal group events define a reproducible patient-specific "
                "contact scaffold and contain order-dependent information over "
                "the latest two to three rank sets. The same dataset shows "
                "orientation-free static correspondence with early-ictal "
                "contact energy, but this correspondence is not specific to a "
                "GRU and the tested path, axis, competition, and source "
                "mechanisms are unsupported."
            ),
            "canonical_report": (
                "docs/archive/topic5/"
                "rnn_overall_integrated_acceptance_2026-07-28.md"
            ),
            "canonical_manuscript_source": (
                "docs/paper-draft/"
                "figure6_static_contact_topography_bounded_result.md"
            ),
            "figures": [
                (
                    "results/topic5_interictal_scaffold_reliability_history_"
                    "necessity/figures/"
                    "topic5_scaffold_reliability_history_necessity_v0_1.png"
                ),
                (
                    "results/paper-ready-figure/"
                    "fig6_static_contact_topography/figures/"
                    "fig6_static_contact_topography.png"
                ),
            ],
        },
        "next_action": {
            "freeze_current_rnn_families": True,
            "do_not_tune": [
                "full-history GRU",
                "persistent path-mode graph",
                "symmetric-axis kernel",
                "competition trace",
                "source-direction term",
            ],
            "accepted_reference_model": "history_3_gru",
            "independent_confirmation_required_for": [
                "early-ictal static contact correspondence",
                "ordered-state early-ictal readback",
            ],
            "do_not_start": (
                "new early-ictal dynamic RNN or seizure predictor on the reused "
                "target cohort"
            ),
        },
        "source_artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in SOURCES.items()
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
