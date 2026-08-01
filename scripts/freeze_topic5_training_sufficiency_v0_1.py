#!/usr/bin/env python3
"""Write the Phase B and Phase C freeze records.

The freeze records are the audit trail that Phase D may finally read the outer
heldout 20%: they state exactly which data ranges the selection consumed and
assert that neither the outer heldout nor any ictal target was opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_training_sufficiency import run_environment  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_rnn_training_sufficiency_v0_1"
ANALYSIS = RESULT_ROOT / "analysis"
DEVELOPMENT = RESULT_ROOT / "development"


def _read(path: Path):
    if not path.is_file():
        raise RuntimeError(f"required evidence is missing: {path}")
    return json.loads(path.read_text())


def _assert_sealed(root: Path) -> dict:
    """Every completed development cell must certify the seal."""
    n_cells = 0
    for summary_path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("ictal_target_read") is not False:
            raise RuntimeError(f"{summary_path}: ictal target seal broken")
        if summary.get("outer_heldout_read") is not False:
            raise RuntimeError(f"{summary_path}: outer heldout was read")
        n_cells += 1
    return {"n_cells_checked": n_cells, "all_sealed": True}


def freeze_hyperparameters(args) -> dict:
    b1 = _read(ANALYSIS / "b1_selection.json")
    extended_path = ANALYSIS / "b1x_selection.json"
    # the extended run supersedes the 4-cycle screen for the budget itself
    budget_selection = (
        _read(extended_path) if extended_path.is_file() else b1
    )
    b2 = _read(ANALYSIS / "b2_selection.json")
    b3 = _read(ANALYSIS / "b3_chunk_parity.json")
    b1c = _read(ANALYSIS / "b1c_paired_tests.json")
    seal = _assert_sealed(DEVELOPMENT)

    best = b1c["best_arm"]
    selected = {
        "model": "LinearStateSequenceRNN",
        "hidden_size": int(budget_selection["selected"]["hidden_size"]),
        "shared_coverage_cycles": int(best["cycles"]),
        "updates_per_patient": int(best["updates_per_patient"]),
        "heldout_offset_calibration_cycles": int(best["offset_cycles"]),
        "learning_rate": float(b2["selected"]["learning_rate"]),
        "optimizer": str(b2["selected"]["optimizer"]),
        "weight_decay": float(b2["selected"]["weight_decay"]),
        "gradient_clip": 1.0,
        "memory_chunk_size": int(b2["selected"]["batch_size"]),
        "local_offset_dim": 4,
        "objective": "teacher_forced_one_step",
    }
    return {
        "status": "FROZEN",
        "contract": "topic5_rnn_training_sufficiency_v0_1_hyperparameter_freeze",
        "selected": selected,
        "evidence": {
            "b1_training_budget": b1,
            "b1x_extended_budget": (
                budget_selection if extended_path.is_file() else None
            ),
            "b2_learning_rate_and_optimizer": b2,
            "b3_chunk_parity": b3,
            "b1c_loso_development_confirmation": {
                "best_arm": b1c["best_arm"],
                "arms": b1c["arms"],
                "paired_vs_best": b1c["paired_vs_best"],
            },
        },
        "data_ranges_read_for_selection": {
            "cohort": "all 34 sealed patients",
            "events": (
                "train80 only, split into a chronological inner-training first "
                "90% and an inner-validation last 10%"
            ),
            "outer_heldout20_read": False,
            "ictal_target_read": False,
            "ab_labels_read": False,
            "physical_axis_read": False,
            "soz_read": False,
            "inter_event_interval_read": False,
            "seizure_labels_read": False,
        },
        "seal_audit": seal,
        "environment": run_environment(),
    }


def freeze_objective(args) -> dict:
    payload = _read(ANALYSIS / "c_paired_tests.json")
    seal = _assert_sealed(DEVELOPMENT)
    frozen = _read(DEVELOPMENT / "HYPERPARAMETER_FREEZE.json")

    summary = pd.DataFrame(payload["descriptive"])
    rollout = summary[summary.rollout_condition == "full_constructive"].copy()
    reference = payload["reference_condition"]
    guard_column = "likelihood_contact_choice_nll__patient_median"
    likelihood = summary[summary.rollout_condition == "none"].set_index("condition")
    reference_nll = float(likelihood.loc[reference, guard_column])

    #: generation endpoints that decide the rollout-aware winner
    ranking = [
        ("transition_correlation", True),
        ("suffix_rank_wasserstein", False),
        ("suffix_precedence_correlation", True),
        ("suffix_precedence_mae", False),
        ("suffix_participation_mae", False),
        ("event_length_wasserstein", False),
    ]
    scores = {}
    for _, row in rollout.iterrows():
        condition = str(row.condition)
        if condition == reference:
            continue
        wins = 0
        detail = {}
        for endpoint, higher in ranking:
            column = f"{endpoint}__patient_median"
            if column not in rollout.columns:
                continue
            candidate = float(row[column])
            baseline = float(
                rollout.loc[rollout.condition == reference, column].iloc[0]
            )
            better = candidate > baseline if higher else candidate < baseline
            detail[endpoint] = {
                "condition": candidate,
                "reference": baseline,
                "better": bool(better),
            }
            wins += int(better)
        guard = float(likelihood.loc[condition, guard_column])
        scores[condition] = {
            "n_generation_endpoints_better": wins,
            "n_generation_endpoints": len(detail),
            "detail": detail,
            "one_step_contact_choice_nll": guard,
            "reference_one_step_contact_choice_nll": reference_nll,
            "one_step_not_degraded": bool(guard <= reference_nll + 0.002),
        }
    eligible = {
        condition: value
        for condition, value in scores.items()
        if value["one_step_not_degraded"]
    }
    pool = eligible or scores
    winner = max(
        pool,
        key=lambda condition: (
            pool[condition]["n_generation_endpoints_better"],
            -pool[condition]["one_step_contact_choice_nll"],
        ),
    )
    return {
        "status": "FROZEN",
        "contract": "topic5_rnn_training_sufficiency_v0_1_objective_freeze",
        "reference_condition": reference,
        "selected_rollout_aware_objective": winner.replace("objective_", ""),
        "selection_rule": (
            "most generation endpoints improved over teacher forcing among "
            "objectives whose one-step contact-choice NLL is not degraded by "
            "more than 0.002 nats/decision"
        ),
        "any_objective_passed_the_one_step_guard": bool(eligible),
        "scores": scores,
        "frozen_training_budget": frozen["selected"],
        "data_ranges_read_for_selection": {
            "events": "train80 inner training and inner validation only",
            "outer_heldout20_read": False,
            "ictal_target_read": False,
            "rollout_loss_weights_tuned_on_heldout_suffix": False,
            "ab_or_axis_used": False,
        },
        "seal_audit": seal,
        "environment": run_environment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hyperparameters", "objective"), required=True)
    args = parser.parse_args()

    if args.kind == "hyperparameters":
        payload = freeze_hyperparameters(args)
        out = DEVELOPMENT / "HYPERPARAMETER_FREEZE.json"
    else:
        payload = freeze_objective(args)
        out = DEVELOPMENT / "OBJECTIVE_FREEZE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"written": str(out.relative_to(ROOT)), "status": payload["status"]}))


if __name__ == "__main__":
    main()
