#!/usr/bin/env python3
"""Analyze the 34-patient structured transition-skeleton cohort screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records

SEEDS = [20260726, 20260727, 20260728]
CONTROLS = {
    "rank0": ("rank_0", "intact"),
    "patient_paths": ("rank_1", "intact"),
    "shuffled_paths": ("rank_1", "weight_shuffle"),
}
LABELS = {
    "rank0": "No history",
    "shuffled_paths": "Shuffled paths",
    "patient_paths": "Patient paths",
}
METRICS = [
    "heldout_event_nll",
    "participation_mae",
    "rank_wasserstein",
    "precedence_mae",
    "precedence_correlation",
    "path_sliced_wasserstein",
    "axis_rho_wasserstein",
]
LOWER_IS_BETTER = {
    metric: metric != "precedence_correlation" for metric in METRICS
}
DATASET_COLOR = {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}


def _contact_rank_uniqueness_audit() -> pd.DataFrame:
    records = load_records(
        ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    )
    rows = []
    for subject, record in records.items():
        row = {"subject": subject, "dataset": record.dataset}
        for split, indices in (
            ("train80", record.train_indices),
            ("heldout20", record.eval_indices),
        ):
            groups = np.asarray(record.group_ids[indices], int)
            counts = np.asarray(record.group_count[indices], int)
            participants = np.sum(groups >= 0, axis=1)
            unique = np.asarray(
                [
                    len(np.unique(event[event >= 0]))
                    for event in groups
                ],
                int,
            )
            row[f"{split}_tied_rank_fraction"] = float(
                np.mean(unique < participants)
            )
            row[f"{split}_count_mismatch_fraction"] = float(
                np.mean(counts != unique)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bh_fdr(values: pd.Series) -> pd.Series:
    x = values.to_numpy(float)
    out = np.full_like(x, np.nan)
    valid = np.flatnonzero(np.isfinite(x))
    if not len(valid):
        return pd.Series(out, index=values.index)
    order = valid[np.argsort(x[valid])]
    adjusted = x[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def _wilcoxon(values: np.ndarray, alternative: str) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative=alternative).pvalue)


def _load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    inventory = []
    for control, (rank_dir, prior_dir) in CONTROLS.items():
        for seed in SEEDS:
            control_root = (
                root / f"seed_{seed}" / rank_dir / prior_dir
            )
            for path in sorted(control_root.glob("*/heldout_metrics.csv")):
                frame = pd.read_csv(path)
                primary = frame.loc[frame.lesion.astype(str) == "none"]
                if len(primary) != 1:
                    raise RuntimeError(f"{path}: expected one primary row")
                row = primary.iloc[0].to_dict()
                row["control"] = control
                row["run_dir"] = str(path.parent)
                summary_path = path.parent / "summary.json"
                state_path = path.parent / "run_state.json"
                summary = json.loads(summary_path.read_text())
                state = json.loads(state_path.read_text())
                for metric in (
                    "participation_mae",
                    "rank_wasserstein",
                    "precedence_mae",
                ):
                    row[f"{metric}_empirical_excess"] = (
                        float(row[metric])
                        - float(summary["empirical_distribution_errors"][metric])
                        - float(
                            summary["split_half_distribution_errors"][metric]
                        )
                    )
                row["path_empirical_excess"] = (
                    float(row["path_sliced_wasserstein"])
                    - float(row["path_empirical_distance"])
                    - float(row["path_split_half_distance"])
                )
                rows.append(row)
                inventory.append(
                    {
                        "subject": row["subject"],
                        "dataset": row["dataset"],
                        "seed": int(row["seed"]),
                        "control": control,
                        "status": state["status"],
                        "ictal_target_read": bool(
                            summary["ictal_target_read"]
                        ),
                        "run_dir": str(path.parent),
                        "summary_exists": summary_path.exists(),
                        "checkpoint_exists": (
                            path.parent / "checkpoint.pt"
                        ).exists(),
                        "rollout_exists": (
                            path.parent / "free_rollouts.npz"
                        ).exists(),
                        "training_log_exists": (
                            path.parent / "training_log.csv"
                        ).exists(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(inventory)


def _paired_statistics(patient: pd.DataFrame) -> pd.DataFrame:
    indexed = patient.set_index(["subject", "dataset", "control"])
    rows = []
    for baseline in ("rank0", "shuffled_paths"):
        for metric in METRICS:
            intact = indexed.xs("patient_paths", level="control")[metric]
            reference = indexed.xs(baseline, level="control")[metric]
            intact, reference = intact.align(reference, join="inner")
            benefit = (
                reference - intact
                if LOWER_IS_BETTER[metric]
                else intact - reference
            )
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n_patients": int(len(benefit)),
                    "median_benefit": float(np.nanmedian(benefit)),
                    "n_patient_paths_better": int(np.sum(benefit > 0)),
                    "wilcoxon_p_directional": _wilcoxon(
                        benefit.to_numpy(), "greater"
                    ),
                    "wilcoxon_p_two_sided": _wilcoxon(
                        benefit.to_numpy(), "two-sided"
                    ),
                }
            )
    stats = pd.DataFrame(rows)
    stats["wilcoxon_q_directional"] = _bh_fdr(
        stats.wilcoxon_p_directional
    )
    stats["wilcoxon_q_two_sided"] = _bh_fdr(
        stats.wilcoxon_p_two_sided
    )
    return stats


def _seed_stability(all_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control in CONTROLS:
        frame = all_seed.loc[all_seed.control == control]
        for metric in METRICS:
            pivot = frame.pivot(index="subject", columns="seed", values=metric)
            for left_index, left in enumerate(SEEDS):
                for right in SEEDS[left_index + 1 :]:
                    rows.append(
                        {
                            "control": control,
                            "metric": metric,
                            "seed_left": left,
                            "seed_right": right,
                            "spearman": float(
                                spearmanr(
                                    pivot[left], pivot[right]
                                ).statistic
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _plot_paired_axis(
    axis: plt.Axes,
    patient: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    order = ["rank0", "shuffled_paths", "patient_paths"]
    pivot = patient.pivot(index="subject", columns="control", values=metric)
    for subject, values in pivot.iterrows():
        dataset = patient.loc[
            patient.subject == subject, "dataset"
        ].iloc[0]
        axis.plot(
            range(3),
            [values[item] for item in order],
            color=DATASET_COLOR[dataset],
            alpha=0.24,
            lw=0.7,
            zorder=1,
        )
        axis.scatter(
            range(3),
            [values[item] for item in order],
            color=DATASET_COLOR[dataset],
            alpha=0.62,
            s=11,
            edgecolor="none",
            zorder=2,
        )
    medians = [pivot[item].median() for item in order]
    axis.plot(range(3), medians, color="#222222", lw=1.8, zorder=3)
    axis.scatter(
        range(3),
        medians,
        color="#222222",
        s=27,
        zorder=4,
    )
    axis.set_xticks(range(3), [LABELS[item] for item in order], rotation=22)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)


def _make_figures(
    patient: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75))
    _plot_paired_axis(
        axes[0], patient, "heldout_event_nll", "Held-out next-set NLL"
    )
    _plot_paired_axis(
        axes[1], patient, "precedence_mae", "Pairwise precedence MAE"
    )
    _plot_paired_axis(
        axes[2],
        patient,
        "path_sliced_wasserstein",
        "Whole-path distance",
    )
    for label, axis in zip("ABC", axes):
        axis.text(
            -0.18,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout(w_pad=1.1)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"cohort_structure_gate.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    pivot = patient.pivot(
        index=["subject", "dataset"], columns="control", values=METRICS
    )
    x = (
        pivot["precedence_mae"]["shuffled_paths"]
        - pivot["precedence_mae"]["patient_paths"]
    )
    y = (
        pivot["path_sliced_wasserstein"]["shuffled_paths"]
        - pivot["path_sliced_wasserstein"]["patient_paths"]
    )
    fig, axis = plt.subplots(figsize=(3.35, 3.15))
    axis.axhline(0, color="#999999", lw=0.8)
    axis.axvline(0, color="#999999", lw=0.8)
    for dataset in ("epilepsiae", "yuquan"):
        use = pivot.index.get_level_values("dataset") == dataset
        axis.scatter(
            x[use],
            y[use],
            color=DATASET_COLOR[dataset],
            s=26,
            alpha=0.8,
            label=dataset.capitalize(),
        )
    axis.set(
        xlabel="Precedence benefit vs shuffled paths",
        ylabel="Whole-path benefit vs shuffled paths",
    )
    axis.legend(frameon=False, fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"patient_path_benefit_joint.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    (figure_dir / "README.md").write_text(
        """### cohort_structure_gate.png

三联图逐项检验患者特异多路径图是否优于无历史模型和等密度边权重排。A 是 held-out next-set NLL，B 是 pairwise precedence 误差，C 是 label-free whole-path distance；细线为患者，黑线和黑点为 cohort median。

**关注点**：患者特异路径必须在 B/C 同时低于两个对照，才通过间期结构门；A 只证明训练可用。

### patient_path_benefit_joint.png

每个点是一位患者，横轴为真实路径相对边权重排的 precedence 改善，纵轴为 whole-path 改善。右上象限表示两个承重传播指标同时支持患者特异路径。

**关注点**：判断证据是否由多数患者共同贡献，还是仅由少数个体驱动。
"""
    )
    metadata = {
        "n_patients": int(patient.subject.nunique()),
        "seeds": SEEDS,
        "controls": LABELS,
        "primary_metrics": [
            "precedence_mae",
            "path_sliced_wasserstein",
        ],
        "statistics_file": "paired_statistics.csv",
        "ictal_target_read": False,
    }
    (figure_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/"
        "cohort_screen_transition_skeleton_v0_7_1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/"
        "cohort_screen_transition_skeleton_v0_7_1/analysis",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_seed, inventory = _load_runs(args.run_root)
    rank_audit = _contact_rank_uniqueness_audit()
    expected = 34 * len(SEEDS) * len(CONTROLS)
    if len(all_seed) != expected:
        raise RuntimeError(
            f"expected {expected} primary rows, found {len(all_seed)}"
        )
    if (
        inventory.status.ne("COMPLETE").any()
        or inventory.ictal_target_read.any()
        or not inventory[
            [
                "summary_exists",
                "checkpoint_exists",
                "rollout_exists",
                "training_log_exists",
            ]
        ].all().all()
    ):
        raise RuntimeError("run inventory failed completeness or target seal")
    exact_zero_columns = [
        "train80_count_mismatch_fraction",
        "heldout20_tied_rank_fraction",
        "heldout20_count_mismatch_fraction",
    ]
    if (
        len(rank_audit) != 34
        or rank_audit[exact_zero_columns].ne(0).any().any()
        or rank_audit.train80_tied_rank_fraction.max() > 0.001
    ):
        raise RuntimeError("contact-rank uniqueness contract failed")
    all_seed.to_csv(args.output_dir / "all_seed_metrics.csv", index=False)
    inventory.to_csv(args.output_dir / "run_inventory.csv", index=False)
    rank_audit.to_csv(
        args.output_dir / "contact_rank_uniqueness_audit.csv", index=False
    )
    patient = (
        all_seed.groupby(["subject", "dataset", "control"], as_index=False)
        .median(numeric_only=True)
    )
    patient.to_csv(
        args.output_dir / "patient_median_metrics.csv", index=False
    )
    stats = _paired_statistics(patient)
    stats.to_csv(args.output_dir / "paired_statistics.csv", index=False)
    stability = _seed_stability(all_seed)
    stability.to_csv(args.output_dir / "seed_stability.csv", index=False)

    primary = stats.loc[
        stats.metric.isin(
            ["precedence_mae", "path_sliced_wasserstein"]
        )
    ].copy()
    primary["passes"] = (
        (primary.median_benefit > 0)
        & (primary.wilcoxon_q_directional < 0.05)
    )
    empirical_columns = [
        "participation_mae_empirical_excess",
        "rank_wasserstein_empirical_excess",
        "precedence_mae_empirical_excess",
        "path_empirical_excess",
    ]
    empirical_gate = (
        patient.loc[patient.control == "patient_paths", empirical_columns]
        <= 0
    )
    summary = {
        "status": "complete",
        "n_runs": int(len(all_seed)),
        "n_patients": int(patient.subject.nunique()),
        "n_seeds": len(SEEDS),
        "controls": list(CONTROLS),
        "heldout_contact_rank_unique_all34": True,
        "train80_tied_rank_fraction_max": float(
            rank_audit.train80_tied_rank_fraction.max()
        ),
        "primary_gate_rows": primary.to_dict(orient="records"),
        "primary_gate_pass": bool(primary.passes.all()),
        "n_patient_paths_within_empirical_variability": {
            column: int(empirical_gate[column].sum())
            for column in empirical_columns
        },
        "seed_stability_median_spearman": {
            f"{control}:{metric}": float(
                stability.loc[
                    (stability.control == control)
                    & (stability.metric == metric),
                    "spearman",
                ].median()
            )
            for control in CONTROLS
            for metric in METRICS
        },
        "ictal_target_read": False,
        "ictal_read_authorized": bool(primary.passes.all()),
    }
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    _make_figures(patient, stats, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
