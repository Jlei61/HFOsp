#!/usr/bin/env python3
"""Build the machine-readable verdict for the SPF-RNN multiround review."""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import sha256_file  # noqa: E402

REVIEW_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development"
    / "multiround_review_2026-07-31"
)
PILOT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4"
)
SNN_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/snn_positive_control"
    / "existing_artifact_system_identification"
)
SNN_INPUT_ROOT = SNN_ROOT.parent / "existing_artifact_reuse"
CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"


def _state(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete round state: {path}")
    return value


def _comparison(
    patient: pd.DataFrame,
    left: str,
    right: str,
    value: str,
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    wide = patient.pivot(index="subject", columns="model", values=value)
    delta = wide[left] - wide[right]
    left_better = delta > 0.0 if higher_is_better else delta < 0.0
    right_better = delta < 0.0 if higher_is_better else delta > 0.0
    return {
        "left": left,
        "right": right,
        "metric": value,
        "higher_is_better": bool(higher_is_better),
        "n_patients": int(len(delta)),
        "median_left_minus_right": float(np.median(delta)),
        "left_better": int(np.sum(left_better)),
        "right_better": int(np.sum(right_better)),
        "ties": int(np.sum(delta == 0.0)),
        "per_patient": {
            str(subject): float(value) for subject, value in delta.items()
        },
    }


def _assert_adequate() -> dict[str, int]:
    counts = {}
    patterns = {
        "round3": REVIEW_ROOT / "round3_nested_learning_curve/per_run/*.json",
        "round5": SNN_ROOT / "per_run/seed*.json",
        "round6": (
            REVIEW_ROOT
            / "round6_latent_dimension_sensitivity/per_run/*.json"
        ),
    }
    for label, pattern in patterns.items():
        paths = [Path(value) for value in glob.glob(str(pattern))]
        if not paths:
            raise RuntimeError(f"missing {label} per-run outputs")
        invalid = []
        n_models = 0
        for path in paths:
            payload = json.loads(path.read_text())
            for name, model in payload["models"].items():
                n_models += 1
                adequacy = model["training_adequacy"]
                if not bool(adequacy["converged"]) and adequacy[
                    "verdict"
                ] != "NO_FREE_PARAMETERS":
                    invalid.append((path.name, name, adequacy["verdict"]))
        if invalid:
            raise RuntimeError(f"{label} contains inadequate fits: {invalid}")
        counts[label] = n_models
    return counts


def _assert_round_sources(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    single_sources = {
        "round1": "scripts/analyze_topic5_spf_multiround.py",
        "round2": "scripts/analyze_topic5_spf_multiround.py",
        "round3": "scripts/run_topic5_spf_nested_learning_curve.py",
        "round4": "scripts/analyze_topic5_spf_multiround.py",
        "round5": "scripts/run_topic5_spf_existing_snn_identifiability.py",
        "round6": "scripts/run_topic5_spf_latent_dimension_sensitivity.py",
    }
    checked: dict[str, Any] = {}
    for label, relative in single_sources.items():
        expected = states[label].get("source_sha256")
        actual = sha256_file(ROOT / relative)
        if expected != actual:
            raise RuntimeError(
                f"{label} source drift: {relative} expected {expected}, "
                f"found {actual}"
            )
        checked[label] = {relative: actual}
    for label in ("round4b", "round7"):
        expected = states[label].get("source_sha256")
        if not isinstance(expected, dict) or not expected:
            raise RuntimeError(f"{label} lacks dependency-level source hashes")
        checked[label] = {}
        for relative, digest in expected.items():
            actual = sha256_file(ROOT / relative)
            if digest != actual:
                raise RuntimeError(
                    f"{label} source drift: {relative} expected {digest}, "
                    f"found {actual}"
                )
            checked[label][relative] = actual
    return checked


def main() -> None:
    base_state = _state(PILOT_ROOT / "LADDER_PILOT_STATE.json")
    expected_base = {"n_subjects": 6, "n_seeds": 3, "n_models": 8}
    for key, expected in expected_base.items():
        if int(base_state.get(key, -1)) != expected:
            raise RuntimeError(
                f"frozen v0.4 pilot {key} drift: "
                f"{base_state.get(key)} != {expected}"
            )
    if base_state.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("frozen v0.4 pilot/config fingerprint drift")
    states = {
        "round1": _state(
            REVIEW_ROOT / "round1_likelihood_calibration/ROUND_STATE.json"
        ),
        "round2": _state(
            REVIEW_ROOT
            / "round2_length_progress_decomposition/ROUND_STATE.json"
        ),
        "round3": _state(
            REVIEW_ROOT / "round3_nested_learning_curve/ROUND_STATE.json"
        ),
        "round4": _state(
            REVIEW_ROOT / "round4_observable_seed_stability/ROUND_STATE.json"
        ),
        "round4b": _state(
            REVIEW_ROOT
            / "round4b_dynamic_residual_seed_stability/ROUND_STATE.json"
        ),
        "round5": _state(SNN_ROOT / "ROUND_STATE.json"),
        "round6": _state(
            REVIEW_ROOT
            / "round6_latent_dimension_sensitivity/ROUND_STATE.json"
        ),
        "round7": _state(
            REVIEW_ROOT / "round7_field_utilization/ROUND_STATE.json"
        ),
    }
    source_provenance = _assert_round_sources(states)
    adequacy = _assert_adequate()

    base = pd.read_csv(PILOT_ROOT / "ladder_per_patient.csv")
    base_nll = base[
        [
            "subject",
            "model",
            "test_nll_per_decision",
        ]
    ]
    base_comparisons = {
        "m4_vs_m0": _comparison(
            base_nll, "m4_field", "m0_static", "test_nll_per_decision"
        ),
        "m4_vs_m1": _comparison(
            base_nll, "m4_field", "m1_markov", "test_nll_per_decision"
        ),
        "m4_vs_m1phase": _comparison(
            base_nll,
            "m4_field",
            "m1_markov_phase",
            "test_nll_per_decision",
        ),
        "m4_vs_m2phase": _comparison(
            base_nll,
            "m4_field",
            "m2_markov_mixture_phase",
            "test_nll_per_decision",
        ),
        "m4_vs_m3": _comparison(
            base_nll, "m4_field", "m3_template", "test_nll_per_decision"
        ),
        "m4phase_vs_m3": _comparison(
            base_nll,
            "m4_field_phase",
            "m3_template",
            "test_nll_per_decision",
        ),
    }
    base_repertoire = base[
        [
            "subject",
            "model",
            "precedence_correlation",
        ]
    ]
    base_repertoire_comparisons = {
        "m4_vs_m2phase": _comparison(
            base_repertoire,
            "m4_field",
            "m2_markov_mixture_phase",
            "precedence_correlation",
            higher_is_better=True,
        ),
        "m4_vs_m3": _comparison(
            base_repertoire,
            "m4_field",
            "m3_template",
            "precedence_correlation",
            higher_is_better=True,
        ),
        "m4phase_vs_m3": _comparison(
            base_repertoire,
            "m4_field_phase",
            "m3_template",
            "precedence_correlation",
            higher_is_better=True,
        ),
    }
    parameter_counts = {
        str(model): int(np.median(values["n_trainable_parameters"]))
        for model, values in base.groupby("model")
    }

    calibration = pd.read_csv(
        REVIEW_ROOT
        / "round1_likelihood_calibration/likelihood_calibration_runs.csv"
    )
    max_samples = int(calibration["samples"].max())
    max_calibration = calibration[calibration["samples"] == max_samples]
    round1_patient = (
        max_calibration.groupby(
            ["subject", "model", "proposal"], as_index=False
        )
        .mean(numeric_only=True)
    )
    round1 = {"max_samples": max_samples}
    for proposal in ("importance", "prior"):
        selected = round1_patient[
            round1_patient["proposal"] == proposal
        ][["subject", "model", "nll_per_decision"]]
        round1[f"m4_vs_m3_{proposal}"] = _comparison(
            selected, "m4_field", "m3_template", "nll_per_decision"
        )
        round1[f"m4phase_vs_m3_{proposal}"] = _comparison(
            selected, "m4_field_phase", "m3_template", "nll_per_decision"
        )
    importance = max_calibration[
        max_calibration["proposal"] == "importance"
    ]
    round1["importance_ess_fraction_median"] = float(
        importance["ess_fraction_median"].median()
    )
    round1["importance_ess_fraction_q10_min"] = float(
        importance["ess_fraction_q10"].min()
    )
    samples = sorted(calibration["samples"].unique())
    previous = int(samples[-2])
    convergence = calibration.pivot_table(
        index=["subject", "seed", "model", "proposal"],
        columns="samples",
        values="nll_per_decision",
    )
    change = np.abs(convergence[max_samples] - convergence[previous])
    round1["absolute_change_previous_to_max_median"] = float(
        np.median(change)
    )
    round1["absolute_change_previous_to_max_maximum"] = float(np.max(change))

    gap = pd.read_csv(
        REVIEW_ROOT
        / "round2_length_progress_decomposition"
        / "event_length_gap_correlations.csv"
    )
    gap_patient = (
        gap.groupby(["subject", "comparison"], as_index=False)
        .mean(numeric_only=True)
    )
    round2 = {}
    for comparison, values in gap_patient.groupby("comparison"):
        round2[str(comparison)] = {
            "median_mean_delta_nll_per_decision": float(
                values["mean_delta_nll_per_decision"].median()
            ),
            "median_within_patient_spearman_delta_vs_group_count": float(
                values["spearman_delta_vs_group_count"].median()
            ),
            "left_better_patients": int(
                np.sum(values["mean_delta_nll_per_decision"] < 0.0)
            ),
            "n_patients": int(len(values)),
        }

    learning = pd.read_csv(
        REVIEW_ROOT
        / "round3_nested_learning_curve/learning_curve_contrasts.csv"
    )
    round3 = []
    for (fraction, comparison), values in learning.groupby(
        ["fraction", "comparison"]
    ):
        round3.append(
            {
                "fraction_of_fixed_maximum_training_budget": float(fraction),
                "comparison": str(comparison),
                "median_delta_nll_per_decision": float(
                    values["delta_nll_per_decision"].median()
                ),
                "left_better_patients": int(
                    np.sum(values["delta_nll_per_decision"] < 0.0)
                ),
                "median_delta_prior_predictive_nll_per_decision": float(
                    values[
                        "delta_prior_predictive_nll_per_decision"
                    ].median()
                ),
                "left_better_patients_prior_predictive": int(
                    np.sum(
                        values[
                            "delta_prior_predictive_nll_per_decision"
                        ]
                        < 0.0
                    )
                ),
                "n_patients": int(len(values)),
            }
        )

    stability = pd.read_csv(
        REVIEW_ROOT
        / "round4_observable_seed_stability/response_seed_pair_stability.csv"
    )
    residual = pd.read_csv(
        REVIEW_ROOT
        / "round4b_dynamic_residual_seed_stability"
        / "dynamic_residual_seed_pairs.csv"
    )
    response_fidelity = pd.read_csv(
        REVIEW_ROOT
        / "round4_observable_seed_stability/response_fidelity_by_stratum.csv"
    )
    residual_fidelity = pd.read_csv(
        REVIEW_ROOT
        / "round4b_dynamic_residual_seed_stability"
        / "dynamic_residual_fidelity.csv"
    )
    round4 = {}
    for model, values in stability.groupby("model"):
        patient = values.groupby("subject")[
            "mean_observable_response_correlation"
        ].mean()
        round4.setdefault(str(model), {})[
            "full_response_seed_correlation_patient_median"
        ] = float(patient.median())
    for model, values in residual.groupby("model"):
        patient = values.groupby("subject")[
            "mean_dynamic_residual_correlation"
        ].mean()
        round4.setdefault(str(model), {})[
            "m0_subtracted_seed_correlation_patient_median"
        ] = float(patient.median())
    for model, values in response_fidelity.groupby("model"):
        patient = values.groupby("subject").mean(numeric_only=True)
        round4.setdefault(str(model), {}).update(
            {
                "observable_fidelity_patient_median": float(
                    patient["response_correlation_to_observed"].median()
                ),
                "generated_to_observed_entropy_ratio_patient_median": float(
                    patient[
                        "entropy_ratio_generated_to_observed"
                    ].median()
                ),
            }
        )
    for model, values in residual_fidelity.groupby("model"):
        patient = values.groupby("subject").mean(numeric_only=True)
        round4.setdefault(str(model), {})[
            "m0_subtracted_fidelity_patient_median"
        ] = float(
            patient["dynamic_residual_correlation_to_observed"].median()
        )

    snn = pd.read_csv(SNN_ROOT / "snn_system_identification_summary.csv")
    snn_inventory = json.loads(
        (SNN_INPUT_ROOT / "existing_snn_artifact_inventory.json").read_text()
    )
    paired = snn[snn["evaluation_family"] == "paired_source_sink"].set_index(
        "model"
    )
    round5 = {
        "simulator_called": False,
        "input_rank_event_sha256": {
            str(family["family"]): str(family["rank_event_sha256"])
            for family in snn_inventory["families"]
        },
        "paired_test_nll_per_decision": {
            str(model): float(value)
            for model, value in paired[
                "paired_test_nll_per_decision_mean"
            ].items()
        },
        "paired_direction_brier": {
            str(model): float(value)
            for model, value in paired["event_direction_brier_mean"].items()
        },
        "conditional_source_forward_fraction": {
            str(row.model): float(row.generated_forward_fraction_mean)
            for row in snn[
                snn["evaluation_family"] == "source_only"
            ].itertuples()
        },
        "conditional_sink_forward_fraction": {
            str(row.model): float(row.generated_forward_fraction_mean)
            for row in snn[
                snn["evaluation_family"] == "sink_only"
            ].itertuples()
        },
    }
    snn_runs = sorted(SNN_ROOT.glob("per_run/seed*.json"))
    shortcut = json.loads(snn_runs[0].read_text())[
        "first_rank_direction_shortcut"
    ]
    round5["first_rank_direction_shortcut"] = shortcut

    dimension = pd.read_csv(
        REVIEW_ROOT
        / "round6_latent_dimension_sensitivity/dimension_contrasts.csv"
    )
    round6 = []
    for (latent_dim, comparison), values in dimension.groupby(
        ["latent_dim", "comparison"]
    ):
        round6.append(
            {
                "latent_dim": int(latent_dim),
                "comparison": str(comparison),
                "median_delta_nll_per_decision": float(
                    values["delta_nll_per_decision"].median()
                ),
                "left_better_patients": int(
                    np.sum(values["delta_nll_per_decision"] < 0.0)
                ),
                "median_delta_prior_predictive_nll_per_decision": float(
                    values[
                        "delta_prior_predictive_nll_per_decision"
                    ].median()
                ),
                "left_better_patients_prior_predictive": int(
                    np.sum(
                        values[
                            "delta_prior_predictive_nll_per_decision"
                        ]
                        < 0.0
                    )
                ),
                "n_patients": int(len(values)),
            }
        )
    utilization = pd.read_csv(
        REVIEW_ROOT / "round7_field_utilization/field_utilization_summary.csv"
    )
    round7 = {
        str(row.model): {
            "n_runs": int(row.n_runs),
            "raw_kl_per_event_median": float(
                row.best_epoch_raw_kl_per_event_median
            ),
            "total_state_displacement_median": float(
                row.prior_mean_total_state_displacement_median
            ),
            "temporal_logit_sd_median": float(
                row.prior_mean_temporal_logit_sd_median
            ),
            "alpha_median": float(row.alpha_median),
        }
        for row in utilization.itertuples()
    }

    human_g1 = (
        "STOP_RULE_REACHED_NOT_SELECTED"
        if base_comparisons["m4_vs_m2phase"]["left_better"] < 6
        or base_comparisons["m4_vs_m3"]["left_better"] < 6
        else "UNEXPECTED_REVIEW_REQUIRED"
    )
    # Round 5 is retained as an exploratory compatibility analysis, not a
    # Gate.  The pooled legacy SNN artifacts do not satisfy the original G0
    # same-condition/N_min contract, and first-rank lookup already solves most
    # of the direction task.  Therefore neither the NLL ordering nor the
    # direction-transfer ordering can adjudicate SNN identifiability.
    g0 = "REMOVED_FROM_RNN_GATE_NOT_EVALUABLE_FROM_ROUND5"
    payload = {
        "contract": "topic5_spf_multiround_review_v0_1",
        "status": "COMPLETE",
        "development_only": True,
        "old_heldout20_scored": False,
        "snn_simulator_called": False,
        "n_human_patients": 6,
        "n_human_fit_seeds": 3,
        "base_v0_4_state": base_state,
        "fit_adequacy_counts": adequacy,
        "round_source_provenance": source_provenance,
        "base_v0_4": base_comparisons,
        "base_v0_4_repertoire_precedence": base_repertoire_comparisons,
        "base_v0_4_trainable_parameter_count_patient_median": parameter_counts,
        "round1_likelihood_calibration": round1,
        "round2_length_progress_decomposition": round2,
        "round3_nested_learning_curve": round3,
        "round4_observable_seed_stability": round4,
        "round5_existing_snn_system_identification": round5,
        "round6_latent_dimension_sensitivity": round6,
        "round7_field_utilization": round7,
        "gate_verdict": {
            "g0_snn_identifiability_historical": g0,
            "g1_v0_1_autonomous_latent_trajectory": human_g1,
            "g2_stable_dynamic_structure": (
                "NOT_OPENED_AS_STRUCTURE_GATE; v0.1 has no emitted-contact "
                "feedback or executable contact-intervention object"
            ),
            "g3_one_structure_many_trajectories": (
                "OUT_OF_SCOPE_FOR_V0_1_AUTONOMOUS_LATENT_TRAJECTORY_NULL"
            ),
            "expand_to_34_patient_cohort": False,
        },
        "verdict_revision": {
            "date": "2026-07-31",
            "reason": (
                "The prior summary inflated an explicitly open, shortcut-"
                "confounded legacy-SNN compatibility analysis into a negative "
                "G0 verdict. Round 5 is now removed from all RNN gates."
            ),
            "round5_status": "EXPLORATORY_COMPATIBILITY_CHECK_ONLY",
            "m4phase_vs_m3_status": (
                "NON_SELECTION_TIE; 2/6 patients, median delta about "
                "+0.010 NLL/decision"
            ),
        },
        "safe_claim": (
            "Given the first rank, event length, and rank cardinalities, "
            "complete suffix rank events contain organization beyond a static "
            "scaffold and a stationary first-order model. The six-patient "
            "development data did not select the v0.1 deterministic "
            "autonomous latent-trajectory model, whose emitted contacts do "
            "not feed back into its latent state. This result does not test or "
            "exclude an identifiable stable contact-interaction structure."
        ),
        "paper_role": (
            "bounded null-model characterization / Extended Data or methods "
            "audit; RNN and SNN are developed independently"
        ),
        "interpretation_boundary": (
            "The stop rule applies only to a low-dimensional initial-state-"
            "driven deterministic trajectory conditioned on the first rank "
            "and future event envelope, without emitted-contact feedback. It "
            "does not test a contact-space recurrent graph, process noise, or "
            "continuous-latency dynamics and says nothing about whether the "
            "patient or SNN has a stable physical network."
        ),
        "round5_interpretation_boundary": {
            "g0_adjudication": "NOT_EVALUABLE",
            "legacy_artifact_pooling": (
                "does not establish a same-condition nested event-count curve "
                "or N_min"
            ),
            "first_rank_lookup_shortcut": (
                "100% direction accuracy in source-only and sink-only, 78.4% "
                "in paired; direction transfer cannot identify structure"
            ),
            "endpoint_conflict": (
                "autonomous models lead the perturbation-direction Brier "
                "endpoint but trail NLL; neither ordering is promoted to a "
                "Gate under the shortcut-confounded design"
            ),
        },
        "diagnostic_independence_limit": (
            "Human rounds 1-4 and 6-7 reuse the same six-patient development "
            "pool and development-test partition. They address different "
            "alternative explanations but are not independent statistical "
            "replications."
        ),
        "next_action": (
            "Archive v0.1 without further architecture rescue. Open a new, "
            "SNN-independent Stable Interaction Graph contract in which "
            "generated contacts feed back through a shared contact-space "
            "interaction, and calibrate it first on a generic synthetic graph."
        ),
        "round_states": {
            key: str(path)
            for key, path in {
                "round1": (
                    REVIEW_ROOT
                    / "round1_likelihood_calibration/ROUND_STATE.json"
                ),
                "round2": (
                    REVIEW_ROOT
                    / "round2_length_progress_decomposition/ROUND_STATE.json"
                ),
                "round3": (
                    REVIEW_ROOT
                    / "round3_nested_learning_curve/ROUND_STATE.json"
                ),
                "round4": (
                    REVIEW_ROOT
                    / "round4_observable_seed_stability/ROUND_STATE.json"
                ),
                "round4b": (
                    REVIEW_ROOT
                    / "round4b_dynamic_residual_seed_stability/ROUND_STATE.json"
                ),
                "round5": SNN_ROOT / "ROUND_STATE.json",
                "round6": (
                    REVIEW_ROOT
                    / "round6_latent_dimension_sensitivity/ROUND_STATE.json"
                ),
                "round7": (
                    REVIEW_ROOT / "round7_field_utilization/ROUND_STATE.json"
                ),
            }.items()
        },
        "source_sha256": sha256_file(Path(__file__)),
    }
    target = REVIEW_ROOT / "MULTIROUND_VERDICT.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["gate_verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
