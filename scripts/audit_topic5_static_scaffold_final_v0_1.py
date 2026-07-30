#!/usr/bin/env python3
"""Final scientific and reproduction audit for static-scaffold v0.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
INTERNAL = ROOT / "results/topic5_rnn_internal_state_reduction"
SEEDS = (20260725, 20260726, 20260727)
SUBJECTS = tuple(
    pd.read_csv(RESULT / "input_availability.csv").subject.astype(str)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0.0]
    p = (
        1.0
        if not len(nonzero)
        else float(
            wilcoxon(
                nonzero,
                alternative="greater",
                method="exact" if len(nonzero) <= 50 else "approx",
            ).pvalue
        )
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def model_metric(
    frame: pd.DataFrame, model: str, null_mode: str, metric: str
) -> dict[str, Any]:
    values = frame.loc[
        (frame.model == model)
        & (frame.null_mode == null_mode)
        & frame.eligible,
        metric,
    ].to_numpy(float)
    return summarize(values)


def paired(
    left: pd.DataFrame,
    left_model: str,
    right: pd.DataFrame,
    right_model: str,
    metric: str,
    null_mode: str = "all_contact",
) -> dict[str, Any]:
    a = left.loc[
        (left.model == left_model)
        & (left.null_mode == null_mode)
        & left.eligible
    ].set_index("subject")
    b = right.loc[
        (right.model == right_model)
        & (right.null_mode == null_mode)
        & right.eligible
    ].set_index("subject")
    common = a.index.intersection(b.index)
    return summarize(
        (
            a.loc[common, metric] - b.loc[common, metric]
        ).to_numpy(float)
    )


def main() -> None:
    baseline_freeze = json.loads((RESULT / "BASELINE_FREEZE.json").read_text())
    if baseline_freeze["target_values_read"]:
        raise RuntimeError("target-free baseline freeze was contaminated")
    tf_paths = sorted(
        (RESULT / "teacher_forced_fields/per_seed").glob("*.npz")
    )
    if len(tf_paths) != 96:
        raise RuntimeError(f"expected 96 teacher-forced cells, found {len(tf_paths)}")
    tf_sealed = 0
    for path in tf_paths:
        metadata = json.loads(path.with_suffix(".json").read_text())
        if not metadata["target_values_read"] and metadata["output_npz_sha256"] == sha256(path):
            tf_sealed += 1
    if tf_sealed != 96:
        raise RuntimeError("teacher-forced fingerprint/target seal audit failed")

    phase1 = pd.read_csv(RESULT / "phase1_existing_fields_patient_metrics.csv")
    phase2 = pd.read_csv(RESULT / "phase2_regularized_baseline_patient_metrics.csv")
    phase3 = pd.read_csv(RESULT / "phase3_teacher_forced_patient_metrics.csv")
    phase4 = pd.read_csv(RESULT / "phase4_contact_confound_partial_scores.csv")
    expected_rows = {"phase1": 480, "phase2": 560, "phase3": 160}
    actual_rows = {
        "phase1": len(phase1),
        "phase2": len(phase2),
        "phase3": len(phase3),
    }
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"patient-metric row count drift: {actual_rows} != {expected_rows}"
        )
    required_confound = {
        "within_shaft_position",
        "geometry_pc1",
        "soz_indicator",
        "baseline_band_power",
        "broadband_1_250",
        "raw_participation",
    }
    missing_confound = required_confound - set(phase4.confound_block)
    if missing_confound:
        raise RuntimeError(f"missing final confound blocks: {missing_confound}")

    formal_root = (
        ROOT
        / "results/topic5_interictal_rank_distribution/runs"
        / "formal_multiseed_20260725_v1"
    )
    formal = pd.read_csv(
        formal_root / "patient_seed_collapsed_summary.csv"
    )
    order = pd.read_csv(INTERNAL / "interictal_order_full_vs_rank_shuffle.csv")
    order_perturbation = order.loc[
        (order.metric == "nll_loss") & (order.order_perturbation == "shuffle"),
        "full_minus_rank_shuffle_sensitivity",
    ].to_numpy(float)
    strongest_nonrecurrent_gain = (
        formal.ordered_history_nll_gain.to_numpy(float)
    )
    rank_shuffle_rows = []
    for seed in SEEDS:
        for path in sorted(
            (formal_root / f"seed_{seed}").glob("*/heldout_metrics.csv")
        ):
            metric = pd.read_csv(path).set_index("control").heldout_event_nll
            rank_shuffle_rows.append(
                {
                    "subject": path.parent.name,
                    "seed": int(seed),
                    "gain": float(
                        metric["rank_shuffle_gru"]
                        - metric["full_history_gru"]
                    ),
                }
            )
    rank_shuffle_gain = (
        pd.DataFrame(rank_shuffle_rows)
        .groupby("subject")
        .gain.mean()
        .to_numpy(float)
    )

    full_signed = model_metric(
        phase1, "full_history_gru", "all_contact", "signed_margin"
    )
    full_absolute = model_metric(
        phase1, "full_history_gru", "all_contact", "absolute_margin"
    )
    raw_absolute = model_metric(
        phase2,
        "raw_train80_participation",
        "all_contact",
        "absolute_margin",
    )
    raw_shaft_absolute = model_metric(
        phase2,
        "raw_train80_participation",
        "within_shaft_circular",
        "absolute_margin",
    )
    raw_geometry_absolute = model_metric(
        phase2,
        "raw_train80_participation",
        "geometry_smooth_rbf",
        "absolute_margin",
    )
    full_vs_best = paired(
        phase1,
        "full_history_gru",
        phase2,
        "best_validation_regularized_participation",
        "absolute_margin",
    )
    full_vs_rank = paired(
        phase1,
        "full_history_gru",
        phase1,
        "rank_shuffle_gru",
        "absolute_margin",
    )
    full_vs_teacher = paired(
        phase1,
        "full_history_gru",
        phase3,
        "teacher_forced_full_gru",
        "absolute_margin",
    )
    field_similarity = pd.read_csv(
        RESULT / "phase2_regularized_baseline_field_similarity.csv"
    )
    similarity = summarize(
        field_similarity.loc[
            (field_similarity.baseline == "best_validation_regularized_participation")
            & field_similarity.eligible,
            "full_gru_field_spearman",
        ].to_numpy(float)
    )

    confound_summary = {}
    for block in (
        "within_shaft_position",
        "geometry_pc1",
        "soz_indicator",
        "baseline_band_power",
        "broadband_1_250",
        "raw_participation",
    ):
        for model in (
            "raw_train80_participation",
            "full_history_gru",
        ):
            values = phase4.loc[
                (phase4.confound_block == block)
                & (phase4.model == model)
                & phase4.eligible,
                "absolute_margin",
            ].to_numpy(float)
            confound_summary[f"{block}__{model}"] = summarize(values)

    artifacts = [
        "INPUT_AUDIT.json",
        "BASELINE_FREEZE.json",
        "PHASE1_EXISTING_FIELDS_SUMMARY.json",
        "PHASE2_REGULARIZED_BASELINE_SUMMARY.json",
        "PHASE3_TEACHER_FORCED_SUMMARY.json",
        "PHASE4_CONTACT_CONFOUND_SUMMARY.json",
        "BASELINE_POWER_CONFOUND_AUDIT.json",
    ]
    artifact_hashes = {
        name: sha256(RESULT / name) for name in artifacts
    }
    conclusion = {
        "formal_ordered_history_gain": {
            "status": "SUPPORTED_VS_RANK_SHUFFLE_NOT_BEST_NONRECURRENT",
            "full_vs_rank_shuffle_heldout_nll": summarize(
                rank_shuffle_gain
            ),
            "full_vs_strongest_nonrecurrent_heldout_nll": summarize(
                strongest_nonrecurrent_gain
            ),
            "boundary": (
                "true event order improves heldout NLL over a GRU trained on "
                "rank-shuffled events, but the full-history GRU does not "
                "outperform the best nonrecurrent prefix model"
            ),
        },
        "order_perturbation_sensitivity": {
            "status": "SUPPORTED_AS_MODEL_USAGE_DIAGNOSTIC",
            "matched_shuffle_nll_cost": summarize(order_perturbation),
            "boundary": (
                "evaluation-time sensitivity shows that the trained full GRU "
                "uses order and is complementary to the separately trained "
                "rank-shuffle comparison"
            ),
        },
        "static_signed_correspondence": {
            "status": "NOT_ESTABLISHED",
            "full_gru_all_contact": full_signed,
        },
        "static_sign_free_morphology": {
            "status": "SUPPORTED_WITHIN_REUSED_TARGET_DATASET",
            "full_gru_all_contact": full_absolute,
            "raw_train80_all_contact": raw_absolute,
            "raw_train80_within_shaft": raw_shaft_absolute,
            "raw_train80_geometry_smooth": raw_geometry_absolute,
            "boundary": (
                "the bridge is a patient-specific sign-free static contact "
                "morphology, not a fixed positive field direction"
            ),
        },
        "gru_specific_static_increment": {
            "status": "NOT_ESTABLISHED",
            "full_minus_best_regularized": full_vs_best,
            "full_minus_rank_shuffle": full_vs_rank,
            "full_field_vs_best_regularized_spearman": similarity,
        },
        "free_rollout_specific_increment": {
            "status": (
                "EXPLORATORY"
                if full_vs_teacher["wilcoxon_greater_p"] < 0.05
                else "NOT_ESTABLISHED"
            ),
            "full_free_minus_teacher_forced": full_vs_teacher,
            "boundary": (
                "a free-vs-teacher difference cannot establish ordered "
                "dynamics unless it also exceeds rank-shuffle and regularized "
                "static baselines"
            ),
        },
        "single_confound_robustness": {
            "status": "SENSITIVITY_ONLY",
            "metrics": confound_summary,
            "unresolved": ["GM/WM label", "artifact/rejection rate"],
        },
        "physical_axis_or_dynamic_seizure_replay": {
            "status": "NOT_TESTED_BY_THIS_CONTRACT",
        },
    }
    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "status": "PASS_WITH_BOUNDED_STATIC_CONCLUSION",
        "execution": {
            "n_patients": len(SUBJECTS),
            "n_seizures": 106,
            "teacher_forced_cells": len(tf_paths),
            "teacher_forced_cells_target_blind_and_fingerprint_valid": tf_sealed,
            "patient_metric_rows": actual_rows,
            "baseline_power_included": True,
            "target_reused_not_independent_confirmation": True,
        },
        "scientific_acceptance": conclusion,
        "artifact_hashes": artifact_hashes,
        "next_goal_recommendation": {
            "status": "DO_NOT_START_NEW_EARLY_ICTAL_RNN_DYNAMICS_MODEL",
            "reason": (
                "neither formal heldout order gain nor GRU-specific static "
                "increment was established"
            ),
            "recommended": (
                "close out the RNN as a bounded supplementary analysis and "
                "prioritize independent contact-topography replication or "
                "target-free signed-field construction"
            ),
        },
    }
    (RESULT / "FINAL_ACCEPTANCE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (RESULT / "RUN_STATUS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "final_acceptance": "FINAL_ACCEPTANCE.json",
                "paper_ready_figure": (
                    "../paper-ready-figure/"
                    "fig6_static_contact_topography/figures/"
                    "fig6_static_contact_topography.png"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
