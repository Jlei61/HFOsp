#!/usr/bin/env python3
"""Analyze the 34-subject persistent path-mode formal experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records
from src.topic5_rank_distribution import contact_rank_distribution


SEEDS = (20260726, 20260727, 20260728)
DEVELOPMENT_SUBJECTS = {
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
}
SPECS = (
    (0, "no_history"),
    (1, "merged_path"),
    (2, "intact"),
    (2, "weight_shuffle"),
    (2, "mode_shuffle"),
)
BASELINES = ("no_history", "merged_path", "weight_shuffle", "mode_shuffle")
PRIMARY = ("participation_mae", "rank_wasserstein")
SECONDARY = (
    "heldout_event_nll",
    "precedence_mae",
    "path_sliced_wasserstein",
)
LESIONS = (
    "graph",
    "inhibition",
    "drop_forward",
    "drop_reverse",
    "mode_collapse",
    "drop_dominant_mode",
)


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    result = np.full_like(values, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def _directional_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _subjects() -> list[str]:
    frame = pd.read_csv(
        ROOT
        / "results/topic5_interictal_rank_distribution/"
        "dataset_v0_4/subject_audit.csv"
    )
    subjects = sorted(frame.loc[frame.status.eq("ok"), "subject"].astype(str))
    if len(subjects) != 34:
        raise RuntimeError(f"expected 34 subjects, found {len(subjects)}")
    return subjects


def load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    inventory = []
    all_subjects = set(_subjects())
    required_metric_columns = {
        "subject",
        "seed",
        "mode_count",
        "control",
        "lesion",
        *PRIMARY,
        *SECONDARY,
    }
    for seed in SEEDS:
        for subject in _subjects():
            for mode_count, control in SPECS:
                run_dir = (
                    root
                    / f"seed_{seed}"
                    / f"k_{mode_count}"
                    / control
                    / subject
                )
                state_path = run_dir / "run_state.json"
                summary_path = run_dir / "summary.json"
                metric_path = run_dir / "heldout_metrics.csv"
                state = (
                    json.loads(state_path.read_text())
                    if state_path.exists()
                    else {}
                )
                summary = (
                    json.loads(summary_path.read_text())
                    if summary_path.exists()
                    else {}
                )
                shared_coverage = summary.get("shared_coverage", {})
                exact_shared = (
                    bool(summary.get("formal_coverage", False))
                    and set(shared_coverage) == all_subjects - {subject}
                    and all(
                        int(value.get("completed_cycles", 0)) == 2
                        and float(
                            value.get("fraction_of_first_cycle", 0.0)
                        )
                        == 1.0
                        for value in shared_coverage.values()
                    )
                )
                calibration = summary.get("calibration_coverage", {})
                exact_calibration = (
                    int(calibration.get("completed_cycles", 0)) == 4
                    and float(calibration.get("fraction_of_first_cycle", 0.0))
                    == 1.0
                )
                sealed = (
                    state.get("ictal_target_read") is False
                    and summary.get("ictal_target_read") is False
                )
                run_metrics = (
                    pd.read_csv(metric_path)
                    if metric_path.exists()
                    else pd.DataFrame()
                )
                expected_lesions = (
                    {"none", *LESIONS}
                    if mode_count == 2 and control == "intact"
                    else {"none"}
                )
                metric_schema_valid = bool(
                    required_metric_columns.issubset(run_metrics.columns)
                    and len(run_metrics) == len(expected_lesions)
                    and set(run_metrics.lesion.astype(str))
                    == expected_lesions
                    and run_metrics.subject.astype(str).eq(subject).all()
                    and run_metrics.seed.astype(int).eq(seed).all()
                    and run_metrics.mode_count.astype(int).eq(mode_count).all()
                    and run_metrics.control.astype(str).eq(control).all()
                )
                summary_identity_valid = bool(
                    summary.get("subject") == subject
                    and int(summary.get("seed", -1)) == seed
                    and int(summary.get("mode_count", -1)) == mode_count
                    and summary.get("control") == control
                    and int(summary.get("rollouts", -1)) == 5000
                    and set(summary.get("input_fingerprints", {}))
                    == all_subjects
                )
                inventory.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "mode_count": mode_count,
                        "control": control,
                        "status": state.get("status", "MISSING"),
                        "metrics_exists": metric_path.exists(),
                        "checkpoint_exists": (
                            run_dir / "checkpoint.pt"
                        ).exists(),
                        "rollouts_exists": (
                            run_dir / "free_rollouts.npz"
                        ).exists(),
                        "exact_shared_coverage": exact_shared,
                        "exact_calibration_coverage": exact_calibration,
                        "metric_schema_valid": metric_schema_valid,
                        "summary_identity_valid": summary_identity_valid,
                        "ictal_target_sealed": sealed,
                        "peak_gpu_memory_mb": summary.get(
                            "peak_gpu_memory_mb", np.nan
                        ),
                        "elapsed_seconds": summary.get(
                            "elapsed_seconds", np.nan
                        ),
                        "run_dir": str(run_dir),
                    }
                )
                if (
                    state.get("status") != "COMPLETE"
                    or not metric_path.exists()
                    or not exact_shared
                    or not exact_calibration
                    or not metric_schema_valid
                    or not summary_identity_valid
                    or not sealed
                ):
                    continue
                run_metrics["run_dir"] = str(run_dir)
                metric_rows.append(run_metrics)
    inventory_frame = pd.DataFrame(inventory)
    metrics = (
        pd.concat(metric_rows, ignore_index=True)
        if metric_rows
        else pd.DataFrame()
    )
    return metrics, inventory_frame


def _reference(
    primary: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    if baseline == "no_history":
        return primary[
            (primary.mode_count == 0) & (primary.control == baseline)
        ]
    if baseline == "merged_path":
        return primary[
            (primary.mode_count == 1) & (primary.control == baseline)
        ]
    return primary[
        (primary.mode_count == 2) & (primary.control == baseline)
    ]


def comparison_benefits(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.lesion.astype(str).eq("none")].copy()
    intact = primary[
        (primary.mode_count == 2) & (primary.control == "intact")
    ].set_index(["subject", "seed"])
    rows = []
    for baseline in BASELINES:
        reference = _reference(primary, baseline).set_index(["subject", "seed"])
        for metric in (*PRIMARY, *SECONDARY):
            left, right = intact[metric].align(reference[metric], join="inner")
            for (subject, seed), value in (right - left).items():
                rows.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "subject": subject,
                        "seed": int(seed),
                        "benefit": float(value),
                    }
                )
    return pd.DataFrame(rows)


def lesion_benefits(metrics: pd.DataFrame) -> pd.DataFrame:
    current = metrics[
        (metrics.mode_count == 2) & (metrics.control == "intact")
    ]
    intact = current[current.lesion.astype(str).eq("none")].set_index(
        ["subject", "seed"]
    )
    rows = []
    for lesion in LESIONS:
        altered = current[current.lesion == lesion].set_index(
            ["subject", "seed"]
        )
        for metric in (*PRIMARY, *SECONDARY):
            left, right = intact[metric].align(altered[metric], join="inner")
            for (subject, seed), value in (right - left).items():
                rows.append(
                    {
                        "lesion": lesion,
                        "metric": metric,
                        "subject": subject,
                        "seed": int(seed),
                        "benefit": float(value),
                    }
                )
    return pd.DataFrame(rows)


def patient_statistics(
    benefits: pd.DataFrame,
    *,
    group_column: str,
    primary_only: bool,
    expected_patients: int = 34,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = (
        benefits.groupby([group_column, "metric", "subject"], as_index=False)
        .benefit.median()
        .rename(columns={"benefit": "seed_median_benefit"})
    )
    if primary_only:
        patient = patient[patient.metric.isin(PRIMARY)].copy()
    rows = []
    for keys, frame in patient.groupby([group_column, "metric"], sort=True):
        values = frame.seed_median_benefit.to_numpy(float)
        rows.append(
            {
                group_column: keys[0],
                "metric": keys[1],
                "n_patients": int(len(values)),
                "median_benefit": float(np.median(values)),
                "n_patients_better": int(np.sum(values > 0)),
                "wilcoxon_p_greater": _directional_wilcoxon(values),
            }
        )
    stats = pd.DataFrame(rows)
    stats["wilcoxon_q_bh"] = benjamini_hochberg(
        stats.wilcoxon_p_greater.to_numpy(float)
    )
    stats["pass"] = (
        (stats.n_patients == int(expected_patients))
        & (stats.median_benefit > 0)
        & (stats.n_patients_better > int(expected_patients) / 2)
        & (stats.wilcoxon_q_bh < 0.05)
    )
    return patient, stats


def node_distributions(root: Path) -> pd.DataFrame:
    records = load_records(
        ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    )
    rows = []
    for seed in SEEDS:
        for subject in _subjects():
            record = records[subject]
            run_dir = (
                root
                / f"seed_{seed}"
                / "k_2"
                / "intact"
                / subject
            )
            with np.load(run_dir / "free_rollouts.npz", allow_pickle=False) as z:
                generated_groups = np.asarray(z["event_group_ids"], np.int16)
                generated_count = np.asarray(z["event_group_count"], np.int16)
                if bool(z["ictal_target_read"]):
                    raise RuntimeError(f"{subject}: ictal target entered rollout")
            observed_groups = record.group_ids[record.eval_indices]
            observed_count = record.group_count[record.eval_indices]
            generated = contact_rank_distribution(
                generated_groups, generated_count, bins=10
            )
            observed = contact_rank_distribution(
                observed_groups, observed_count, bins=10
            )
            for contact, name in enumerate(record.contact_names):
                row = {
                    "subject": subject,
                    "dataset": record.dataset,
                    "seed": seed,
                    "contact_index": contact,
                    "contact_name": str(name),
                    "observed_participation": float(
                        observed["participation_probability"][contact]
                    ),
                    "generated_participation": float(
                        generated["participation_probability"][contact]
                    ),
                    "observed_mean_rank": float(
                        observed["mean_rank"][contact]
                    ),
                    "generated_mean_rank": float(
                        generated["mean_rank"][contact]
                    ),
                }
                for bin_index in range(10):
                    row[f"observed_rank_bin_{bin_index}"] = float(
                        observed["rank_histogram"][contact, bin_index]
                    )
                    row[f"generated_rank_bin_{bin_index}"] = float(
                        generated["rank_histogram"][contact, bin_index]
                    )
                row["observed_early_probability"] = float(
                    np.sum(observed["rank_histogram"][contact, :3])
                )
                row["generated_early_probability"] = float(
                    np.sum(generated["rank_histogram"][contact, :3])
                )
                row["observed_middle_probability"] = float(
                    np.sum(observed["rank_histogram"][contact, 3:7])
                )
                row["generated_middle_probability"] = float(
                    np.sum(generated["rank_histogram"][contact, 3:7])
                )
                row["observed_late_probability"] = float(
                    np.sum(observed["rank_histogram"][contact, 7:])
                )
                row["generated_late_probability"] = float(
                    np.sum(generated["rank_histogram"][contact, 7:])
                )
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    metrics, inventory = load_runs(root)
    inventory.to_csv(analysis / "run_inventory.csv", index=False)
    if (
        len(inventory) != 510
        or not inventory.status.eq("COMPLETE").all()
        or not inventory.metrics_exists.all()
        or not inventory.checkpoint_exists.all()
        or not inventory.rollouts_exists.all()
        or not inventory.exact_shared_coverage.all()
        or not inventory.exact_calibration_coverage.all()
        or not inventory.metric_schema_valid.all()
        or not inventory.summary_identity_valid.all()
        or not inventory.ictal_target_sealed.all()
    ):
        raise RuntimeError("formal inventory is incomplete or violates seals")
    metrics.to_csv(analysis / "all_metrics.csv", index=False)
    if len(metrics) != 34 * 3 * (4 + 7):
        raise RuntimeError(
            f"unexpected formal metric row count: {len(metrics)}"
        )

    comparisons = comparison_benefits(metrics)
    comparisons.to_csv(analysis / "comparison_benefits.csv", index=False)
    comparison_patient, comparison_stats = patient_statistics(
        comparisons, group_column="baseline", primary_only=True
    )
    comparison_patient.to_csv(
        analysis / "comparison_patient_seed_medians.csv", index=False
    )
    comparison_stats.to_csv(
        analysis / "comparison_primary_statistics.csv", index=False
    )
    _, comparison_all_stats = patient_statistics(
        comparisons,
        group_column="baseline",
        primary_only=False,
    )
    comparison_all_stats["inference_role"] = np.where(
        comparison_all_stats.metric.isin(PRIMARY),
        "primary_duplicate_for_complete_table",
        "secondary_diagnostic",
    )
    comparison_all_stats.to_csv(
        analysis / "comparison_all_metric_statistics.csv", index=False
    )

    lesions = lesion_benefits(metrics)
    lesions.to_csv(analysis / "lesion_benefits.csv", index=False)
    lesion_patient, lesion_stats = patient_statistics(
        lesions, group_column="lesion", primary_only=True
    )
    lesion_patient.to_csv(
        analysis / "lesion_patient_seed_medians.csv", index=False
    )
    lesion_stats.to_csv(
        analysis / "lesion_primary_statistics.csv", index=False
    )
    _, lesion_all_stats = patient_statistics(
        lesions,
        group_column="lesion",
        primary_only=False,
    )
    lesion_all_stats["inference_role"] = np.where(
        lesion_all_stats.metric.isin(PRIMARY),
        "primary_duplicate_for_complete_table",
        "secondary_diagnostic",
    )
    lesion_all_stats.to_csv(
        analysis / "lesion_all_metric_statistics.csv", index=False
    )

    comparison_confirm = comparisons[
        ~comparisons.subject.isin(DEVELOPMENT_SUBJECTS)
    ]
    comparison_confirm_patient, comparison_confirm_stats = patient_statistics(
        comparison_confirm,
        group_column="baseline",
        primary_only=True,
        expected_patients=31,
    )
    comparison_confirm_patient.to_csv(
        analysis
        / "comparison_patient_seed_medians_development_excluded.csv",
        index=False,
    )
    comparison_confirm_stats.to_csv(
        analysis
        / "comparison_primary_statistics_development_excluded.csv",
        index=False,
    )
    lesion_confirm = lesions[~lesions.subject.isin(DEVELOPMENT_SUBJECTS)]
    lesion_confirm_patient, lesion_confirm_stats = patient_statistics(
        lesion_confirm,
        group_column="lesion",
        primary_only=True,
        expected_patients=31,
    )
    lesion_confirm_patient.to_csv(
        analysis / "lesion_patient_seed_medians_development_excluded.csv",
        index=False,
    )
    lesion_confirm_stats.to_csv(
        analysis / "lesion_primary_statistics_development_excluded.csv",
        index=False,
    )

    nodes = node_distributions(root)
    nodes.to_csv(analysis / "intact_k2_contact_distributions.csv", index=False)

    comparison_gate = bool(comparison_stats["pass"].all())
    structure_options = {}
    for lesion in ("graph", "mode_collapse"):
        selected = lesion_stats[lesion_stats.lesion == lesion]
        structure_options[lesion] = bool(
            len(selected) == len(PRIMARY) and selected["pass"].all()
        )
    structure_gate = any(structure_options.values())
    formal_gate = comparison_gate and structure_gate
    confirm_comparison_gate = bool(comparison_confirm_stats["pass"].all())
    confirm_structure_options = {}
    for lesion in ("graph", "mode_collapse"):
        selected = lesion_confirm_stats[
            lesion_confirm_stats.lesion == lesion
        ]
        confirm_structure_options[lesion] = bool(
            len(selected) == len(PRIMARY) and selected["pass"].all()
        )
    summary = {
        "status": "complete",
        "contract": "topic5_persistent_path_mode_rnn_v1_0",
        "n_patients": 34,
        "n_seeds": 3,
        "n_runs": 510,
        "mode_count": 2,
        "primary_metrics": list(PRIMARY),
        "comparison_gate_pass": comparison_gate,
        "structure_gate_options": structure_options,
        "structure_gate_pass": structure_gate,
        "formal_interictal_gate_pass": formal_gate,
        "development_subjects": sorted(DEVELOPMENT_SUBJECTS),
        "development_excluded_sensitivity": {
            "n_patients": 31,
            "comparison_gate_pass": confirm_comparison_gate,
            "structure_gate_options": confirm_structure_options,
            "structure_gate_pass": any(confirm_structure_options.values()),
            "role": "sensitivity_not_formal_hard_gate",
        },
        "ictal_target_read": False,
        "next_action": (
            "frozen_ictal_static_readout"
            if formal_gate
            else "bounded_negative_stop_no_ictal"
        ),
        "peak_gpu_memory_mb": float(
            inventory.peak_gpu_memory_mb.max()
        ),
        "total_run_hours": float(
            inventory.elapsed_seconds.sum() / 3600.0
        ),
    }
    (analysis / "formal_gate_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    manifest_path = root / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "COMPLETE",
            "formal_interictal_gate_pass": formal_gate,
            "ictal_target_read": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
