#!/usr/bin/env python3
"""Summarize the target-sealed v0.8 set-valued structured-RNN pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records

SEEDS = [20260726, 20260727, 20260728]
SUBJECTS = ["epilepsiae_1073", "epilepsiae_1146", "yuquan_chenziyang"]
SUBJECT_LABEL = {
    "epilepsiae_1073": "E1073",
    "epilepsiae_1146": "E1146",
    "yuquan_chenziyang": "Y-Chenziyang",
}
SUBJECT_COLOR = {
    "epilepsiae_1073": "#2166AC",
    "epilepsiae_1146": "#67A9CF",
    "yuquan_chenziyang": "#B66A2B",
}
CONDITIONS = {
    "rank0": (0, "intact"),
    "rank1": (1, "intact"),
    "rank2": (2, "intact"),
    "rank4": (4, "intact"),
    "rank1_shuffle": (1, "weight_shuffle"),
    "rank2_shuffle": (2, "weight_shuffle"),
}
METRICS = [
    "heldout_event_nll",
    "participation_mae",
    "rank_wasserstein",
    "precedence_mae",
    "precedence_correlation",
    "path_sliced_wasserstein",
]
HIGHER_IS_BETTER = {"precedence_correlation"}


def _load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    inventory = []
    for condition, (rank, prior) in CONDITIONS.items():
        for path in sorted(
            root.glob(
                f"seed_*/rank_{rank}/{prior}/*/heldout_metrics.csv"
            )
        ):
            frame = pd.read_csv(path)
            primary = frame.loc[frame.lesion.astype(str) == "none"]
            if len(primary) != 1:
                raise RuntimeError(f"{path}: expected one primary row")
            row = primary.iloc[0].to_dict()
            row["condition"] = condition
            rows.append(row)
            summary = json.loads((path.parent / "summary.json").read_text())
            state = json.loads((path.parent / "run_state.json").read_text())
            with np.load(path.parent / "free_rollouts.npz") as z:
                groups = np.asarray(z["event_group_ids"])
                counts = np.asarray(z["event_group_count"])
                participants = np.sum(groups >= 0, axis=1)
            inventory.append(
                {
                    "subject": row["subject"],
                    "dataset": row["dataset"],
                    "seed": int(row["seed"]),
                    "condition": condition,
                    "status": state["status"],
                    "ictal_target_read": bool(summary["ictal_target_read"]),
                    "generated_tied_rank_fraction": float(
                        np.mean(participants > counts)
                    ),
                    "generated_participant_count_mean": float(
                        np.mean(participants)
                    ),
                    "generated_rank_set_count_mean": float(np.mean(counts)),
                    "run_dir": str(path.parent),
                }
            )
    metrics = pd.DataFrame(rows)
    inventory_frame = pd.DataFrame(inventory)
    expected = len(SUBJECTS) * len(SEEDS) * len(CONDITIONS)
    if len(metrics) != expected:
        raise RuntimeError(f"expected {expected} runs, found {len(metrics)}")
    if (
        inventory_frame.status.ne("COMPLETE").any()
        or inventory_frame.ictal_target_read.any()
    ):
        raise RuntimeError("run completeness or target seal failed")
    return metrics, inventory_frame


def _observed_tied_fraction(config_path: Path) -> pd.DataFrame:
    config = yaml.safe_load(config_path.read_text())
    records = load_records(ROOT / config["inputs"]["dataset"])
    rows = []
    for subject in SUBJECTS:
        record = records[subject]
        groups = record.group_ids[record.eval_indices]
        counts = record.group_count[record.eval_indices]
        participants = np.sum(groups >= 0, axis=1)
        rows.append(
            {
                "subject": subject,
                "observed_tied_rank_fraction": float(
                    np.mean(participants > counts)
                ),
            }
        )
    return pd.DataFrame(rows)


def _comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["subject", "seed", "condition"])
    pairs = [
        ("rank1", "rank0"),
        ("rank2", "rank0"),
        ("rank4", "rank0"),
        ("rank1", "rank1_shuffle"),
        ("rank2", "rank2_shuffle"),
    ]
    rows = []
    for model, reference in pairs:
        for metric in METRICS:
            left = indexed.xs(model, level="condition")[metric]
            right = indexed.xs(reference, level="condition")[metric]
            left, right = left.align(right, join="inner")
            benefit = (
                left - right
                if metric in HIGHER_IS_BETTER
                else right - left
            )
            per_subject = benefit.groupby(level="subject").median()
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "n_patient_seed": int(len(benefit)),
                    "n_patient_seed_better": int(np.sum(benefit > 0)),
                    "median_patient_seed_benefit": float(
                        np.median(benefit)
                    ),
                    "n_subjects": int(len(per_subject)),
                    "n_subjects_better": int(np.sum(per_subject > 0)),
                    "median_subject_benefit": float(
                        np.median(per_subject)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _condition_subject_median(
    metrics: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    metric_median = (
        metrics.groupby(["subject", "dataset", "condition"], as_index=False)
        .median(numeric_only=True)
    )
    tied = (
        inventory.groupby(["subject", "dataset", "condition"], as_index=False)
        .generated_tied_rank_fraction.median()
    )
    return metric_median.merge(
        tied, on=["subject", "dataset", "condition"], validate="one_to_one"
    )


def _paired_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    conditions: list[str],
    labels: list[str],
    metric: str,
    ylabel: str,
) -> None:
    pivot = frame.pivot(index="subject", columns="condition", values=metric)
    x = np.arange(len(conditions))
    for subject in SUBJECTS:
        values = [pivot.loc[subject, condition] for condition in conditions]
        axis.plot(
            x,
            values,
            color=SUBJECT_COLOR[subject],
            lw=1.0,
            alpha=0.85,
            marker="o",
            ms=3.6,
            label=SUBJECT_LABEL[subject],
        )
    median = [pivot[condition].median() for condition in conditions]
    axis.plot(x, median, color="#222222", lw=1.8, marker="o", ms=4.2)
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", rotation=20)


def _plot(
    subject_median: pd.DataFrame,
    observed: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    observed_map = observed.set_index("subject").observed_tied_rank_fraction
    tied_rows = []
    for subject in SUBJECTS:
        tied_rows.append(
            {
                "subject": subject,
                "dataset": (
                    "yuquan" if subject.startswith("yuquan") else "epilepsiae"
                ),
                "condition": "observed",
                "tied_fraction": observed_map[subject],
            }
        )
    generated = subject_median.loc[
        subject_median.condition.isin(["rank0", "rank1", "rank2", "rank4"]),
        ["subject", "dataset", "condition", "generated_tied_rank_fraction"],
    ].rename(columns={"generated_tied_rank_fraction": "tied_fraction"})
    tied = pd.concat([pd.DataFrame(tied_rows), generated], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.25))
    _paired_panel(
        axes[0, 0],
        tied,
        ["observed", "rank0", "rank1", "rank2", "rank4"],
        ["Data", "R0", "R1", "R2", "R4"],
        "tied_fraction",
        "Events with tied ranks",
    )
    _paired_panel(
        axes[0, 1],
        subject_median,
        ["rank0", "rank1", "rank2", "rank4"],
        ["R0", "R1", "R2", "R4"],
        "heldout_event_nll",
        "Held-out set NLL",
    )
    _paired_panel(
        axes[0, 2],
        subject_median,
        ["rank0", "rank1", "rank2", "rank4"],
        ["R0", "R1", "R2", "R4"],
        "participation_mae",
        "Participation MAE",
    )
    _paired_panel(
        axes[1, 0],
        subject_median,
        ["rank0", "rank1", "rank2", "rank4"],
        ["R0", "R1", "R2", "R4"],
        "rank_wasserstein",
        "Rank-distribution distance",
    )
    _paired_panel(
        axes[1, 1],
        subject_median,
        ["rank0", "rank2_shuffle", "rank2"],
        ["No history", "Shuffled", "Patient paths"],
        "precedence_mae",
        "Pairwise precedence MAE",
    )
    _paired_panel(
        axes[1, 2],
        subject_median,
        ["rank0", "rank2_shuffle", "rank2"],
        ["No history", "Shuffled", "Patient paths"],
        "path_sliced_wasserstein",
        "Whole-path distance",
    )
    for label, axis in zip("ABCDEF", axes.flat):
        axis.text(
            -0.18,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.52, 1.015),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.3, h_pad=1.1)
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"transition_set_pilot_gate.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    (figures / "README.md").write_text(
        """### transition_set_pilot_gate.png

六个 panel 依次检查：生成器能否保留同 rank 多触点（A），不同结构 rank 的 held-out set loss（B）、参与概率（C）和完整 contact-rank distribution（D），以及最佳候选 rank 2 相对无历史和同密度路径乱序对照的传播先后（E）与完整路径（F）。彩色线是三位预先固定患者的三 seed 中位数，黑线是三位患者的中位数。

**关注点**：A 通过说明动作单位已修复；B/C 的改善不能替代 E/F。当前 E 未通过，因此本图是负向 gate 诊断，不是 cohort 主结果。
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT
        / "results/topic5_structured_axis_graph/"
        "screen_transition_set_v0_8_pilot_1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_transition_set_graph_rnn_v0_8.yaml",
    )
    args = parser.parse_args()
    output_dir = args.run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, inventory = _load_runs(args.run_root)
    observed = _observed_tied_fraction(args.config)
    comparisons = _comparisons(metrics)
    subject_median = _condition_subject_median(metrics, inventory)
    metrics.to_csv(output_dir / "all_run_metrics.csv", index=False)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    observed.to_csv(output_dir / "observed_tied_rank_fraction.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    subject_median.to_csv(
        output_dir / "subject_median_metrics.csv", index=False
    )
    primary = comparisons.loc[
        (comparisons.model == "rank2")
        & comparisons.reference.isin(["rank0", "rank2_shuffle"])
        & comparisons.metric.isin(
            ["precedence_mae", "path_sliced_wasserstein"]
        )
    ].copy()
    primary["passes"] = (
        (primary.n_patient_seed_better >= 6)
        & (primary.n_subjects_better >= 2)
        & (primary.median_patient_seed_benefit > 0)
        & (primary.median_subject_benefit > 0)
    )
    summary = {
        "status": "complete",
        "contract": "topic5_transition_set_graph_rnn_v0_8",
        "n_subjects": len(SUBJECTS),
        "n_seeds": len(SEEDS),
        "n_runs": int(len(metrics)),
        "generated_multi_contact_rank_verified": bool(
            inventory.generated_tied_rank_fraction.gt(0).all()
        ),
        "primary_gate_rows": primary.to_dict(orient="records"),
        "interictal_structure_gate_pass": bool(primary.passes.all()),
        "ictal_target_read": False,
        "ictal_read_authorized": False,
        "cohort_run_authorized": bool(primary.passes.all()),
        "decision": (
            "advance_to_34_patient_cohort"
            if primary.passes.all()
            else "stop_before_cohort_and_ictal"
        ),
    }
    (output_dir / "gate_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    _plot(subject_median, observed, output_dir)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
