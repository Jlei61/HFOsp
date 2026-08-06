#!/usr/bin/env python3
"""Assemble the machine-readable acceptance for HistoryRNN direct transfer v0.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"required closeout artifact missing: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/topic5_history_rnn_direct_early_ictal_transfer_v0_2"),
    )
    parser.add_argument(
        "--g1-closeout",
        type=Path,
        default=Path("results/topic5_history_rnn_early_ictal_field/FINAL_CLOSEOUT.json"),
    )
    parser.add_argument(
        "--budget-comparison",
        type=Path,
        default=None,
        help="c10-to-c30 direct-transfer comparison JSON; defaults under --root.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    direct = _read(root / "DIRECT_TRANSFER_SUMMARY.json")
    synthetic = _read(
        root / "g1_diagnostics/synthetic_v2/SYNTHETIC_RECOVERABILITY.json"
    )
    utilization = _read(
        root / "g1_diagnostics/checkpoint_utilization/CHECKPOINT_UTILIZATION_SUMMARY.json"
    )
    convergence = _read(
        root / "g1_diagnostics/real_convergence/REAL_CONVERGENCE_SUMMARY.json"
    )
    budget_path = (
        args.budget_comparison.resolve()
        if args.budget_comparison is not None
        else root
        / "training_budget_comparison_c10_to_c30"
        / "DIRECT_TRAINING_BUDGET_COMPARISON.json"
    )
    budget = _read(budget_path)
    g1 = _read(args.g1_closeout.resolve())
    if not direct.get("target_values_read"):
        raise RuntimeError("direct transfer did not record target access")
    if synthetic.get("status") != "PASS":
        raise RuntimeError("synthetic recoverability did not pass")
    if int(utilization.get("n_checkpoints", 0)) != 16:
        raise RuntimeError("final c30 state-utilization audit is incomplete")
    if int(utilization.get("history_checkpoint_cycles", 0)) != 30:
        raise RuntimeError("state-utilization audit did not inspect c30 checkpoints")
    if int(direct.get("n_completed_folds", 0)) != 16:
        raise RuntimeError("direct transfer fold count is incomplete")
    if int(direct.get("history_checkpoint_cycles", 0)) != 30:
        raise RuntimeError("final direct transfer is not based on the c30 checkpoint budget")
    channel_null = direct.get("all_contact_channel_shuffle") or {}
    if int(channel_null.get("n_draws_per_patient", 0)) != 5000:
        raise RuntimeError("all-contact channel-shuffle audit is missing or incomplete")
    if channel_null.get("patient_fold") != "median_across_seizures":
        raise RuntimeError("patient-first channel-null folding contract drifted")

    direct_status = str(direct.get("status"))
    if direct_status == "R2_RELATIVE_INCREMENT_ONLY_ABSOLUTE_NOT_SUPPORTED":
        acceptance_status = (
            "ACCEPTED_SUPPLEMENTARY_RELATIVE_INCREMENT_WITH_ABSOLUTE_BOUNDARY"
        )
        interpretation = (
            "The HistoryRNN state improved contact-wise early-ictal field ranking "
            "relative to the frozen static-plus-unordered baseline, and the increment "
            "was reduced by order shuffle and exact zero-state controls. However, "
            "absolute held-out field prediction and within-patient seizure-specific "
            "pairing were not supported. The result therefore identifies a small "
            "cross-state spatial increment, not a usable seizure-field predictor."
        )
    elif direct_status in {
        "DIRECT_R2_SIGNAL_SUPPORTED_BUT_NOT_SEIZURE_SPECIFIC",
        "SEIZURE_CONDITIONED_HISTORY_SIGNAL_SUPPORTED",
    }:
        acceptance_status = "ACCEPTED_SUPPLEMENTARY_DIRECT_SIGNAL"
        interpretation = (
            "The HistoryRNN provided a positive absolute held-out early-ictal field "
            "association beyond the static-plus-unordered baseline. Seizure-specific "
            "interpretation is allowed only when the correct-versus-wrong pairing "
            "control also passes."
        )
    else:
        acceptance_status = "ACCEPTED_SUPPLEMENTARY_BOUNDED_NEGATIVE"
        interpretation = (
            "The current HistoryRNN did not provide a held-out early-ictal field "
            "increment beyond the static-plus-unordered baseline. This bounded result "
            "does not negate the separately established sign-free static morphology."
        )

    budget_robust = budget.get("status") == "ROBUST_SCIENTIFIC_VERDICT_ACROSS_BUDGETS"
    if not budget_robust:
        acceptance_status = "ACCEPTED_SUPPLEMENTARY_TRAINING_SENSITIVE_BOUNDARY"
        interpretation += (
            " The c10 and c30 scientific flags were not identical, so the direct "
            "transfer verdict remains training-budget sensitive and cannot support "
            "a stable recurrent-state claim."
        )

    result = {
        "status": acceptance_status,
        "contract": "topic5_history_rnn_direct_early_ictal_transfer_v0_2_closeout",
        "engineering_execution": "PASS",
        "g1_next_event_proxy": g1.get("status"),
        "synthetic_recoverability": synthetic.get("status"),
        "real_training_sufficiency": convergence.get("status"),
        "direct_training_budget_robustness": budget.get("status"),
        "state_branch_utilization": utilization.get("status"),
        "latent_state_to_early_ictal_field": direct.get("status"),
        "activity_integrator_to_early_ictal_field": direct.get(
            "activity_integrator_status"
        ),
        "seizure_specific_state": (
            "SUPPORTED_PREDICTIVE_ASSOCIATION"
            if direct_status == "SEIZURE_CONDITIONED_HISTORY_SIGNAL_SUPPORTED"
            else "NOT_SUPPORTED"
        ),
        "history_dependent_network_reconfiguration": "NOT_SUPPORTED_BY_CURRENT_READOUT",
        "causal_network_shaping": "NOT_ESTABLISHED",
        "target_values_read": True,
        "target_reuse_tier": "INTERNAL_VALIDATION_NOT_INDEPENDENT_CONFIRMATION",
        "all_contact_channel_shuffle_draws": int(
            channel_null["n_draws_per_patient"]
        ),
        "patient_fold": channel_null["patient_fold"],
        "primary_cohort": direct.get("primary_cohort"),
        "supportive_cohort": direct.get("supportive_cohort"),
        "scientific_interpretation": interpretation,
        "claim_boundary": (
            "Predictive association only; no causal shaping, biological time constant, "
            "or prospective seizure forecasting claim is allowed."
        ),
        "evidence": {
            "direct_summary": "DIRECT_TRANSFER_SUMMARY.json",
            "synthetic": "g1_diagnostics/synthetic_v2/SYNTHETIC_RECOVERABILITY.json",
            "utilization": "g1_diagnostics/checkpoint_utilization/CHECKPOINT_UTILIZATION_SUMMARY.json",
            "convergence": "g1_diagnostics/real_convergence/REAL_CONVERGENCE_SUMMARY.json",
            "budget_comparison": str(budget_path),
            "channel_null": "direct_transfer_channel_null_patient_metrics.csv",
            "figure": "figures/topic5_history_to_early_ictal_direct_transfer_v0_2.png",
        },
    }
    path = root / "FINAL_ACCEPTANCE.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
