#!/usr/bin/env python3
"""Apply the frozen hard gate to the persistent path-mode pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBJECTS = (
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
)
SEEDS = (20260726, 20260727, 20260728)
SPECS = (
    (0, "no_history"),
    (1, "merged_path"),
    (1, "intact"),
    (1, "weight_shuffle"),
    (2, "intact"),
    (2, "weight_shuffle"),
    (2, "mode_shuffle"),
    (3, "intact"),
    (3, "weight_shuffle"),
    (3, "mode_shuffle"),
    (4, "intact"),
    (4, "weight_shuffle"),
    (4, "mode_shuffle"),
)
METRICS = ("precedence_mae", "path_sliced_wasserstein")
BASELINES = ("no_history", "merged_path", "weight_shuffle", "mode_shuffle")
COLORS = {
    "no_history": "#777777",
    "merged_path": "#9E7C23",
    "weight_shuffle": "#B04A3F",
    "mode_shuffle": "#4169A1",
}


def _load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    inventory = []
    for seed in SEEDS:
        for subject in SUBJECTS:
            for mode_count, control in SPECS:
                run_dir = (
                    root
                    / f"seed_{seed}"
                    / f"k_{mode_count}"
                    / control
                    / subject
                )
                state_path = run_dir / "run_state.json"
                metric_path = run_dir / "heldout_metrics.csv"
                summary_path = run_dir / "summary.json"
                status = "MISSING"
                sealed = False
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    status = str(state.get("status", "UNKNOWN"))
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    sealed = not bool(summary.get("ictal_target_read", True))
                inventory.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "mode_count": mode_count,
                        "control": control,
                        "status": status,
                        "ictal_target_sealed": sealed,
                        "metrics_exists": metric_path.exists(),
                        "checkpoint_exists": (
                            run_dir / "checkpoint.pt"
                        ).exists(),
                        "training_log_exists": (
                            run_dir / "training_log.csv"
                        ).exists(),
                        "rollout_exists": (
                            run_dir / "free_rollouts.npz"
                        ).exists(),
                        "run_dir": str(run_dir),
                    }
                )
                if status != "COMPLETE" or not metric_path.exists() or not sealed:
                    continue
                frame = pd.read_csv(metric_path)
                frame["run_dir"] = str(run_dir)
                rows.append(frame)
    inventory_frame = pd.DataFrame(inventory)
    metric_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return metric_frame, inventory_frame


def _reference_rows(primary: pd.DataFrame, mode_count: int, baseline: str):
    if baseline == "no_history":
        return primary[
            (primary.mode_count == 0) & (primary.control == "no_history")
        ]
    if baseline == "merged_path":
        return primary[
            (primary.mode_count == 1) & (primary.control == "merged_path")
        ]
    return primary[
        (primary.mode_count == mode_count) & (primary.control == baseline)
    ]


def _paired_benefits(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.lesion.astype(str) == "none"].copy()
    rows = []
    for mode_count in range(1, 5):
        intact = primary[
            (primary.mode_count == mode_count)
            & (primary.control == "intact")
        ].set_index(["subject", "seed"])
        baselines = BASELINES if mode_count >= 2 else BASELINES[:-1]
        for baseline in baselines:
            reference = _reference_rows(primary, mode_count, baseline).set_index(
                ["subject", "seed"]
            )
            for metric in METRICS:
                left, right = intact[metric].align(
                    reference[metric], join="inner"
                )
                for (subject, seed), value in (right - left).items():
                    rows.append(
                        {
                            "mode_count": mode_count,
                            "baseline": baseline,
                            "metric": metric,
                            "subject": subject,
                            "seed": int(seed),
                            "benefit": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def _count_gate(
    frame: pd.DataFrame,
    *,
    min_patient_seed: int,
    min_subjects: int,
) -> dict:
    subject_median = frame.groupby("subject").benefit.median()
    return {
        "n_patient_seed": int(len(frame)),
        "n_patient_seed_better": int((frame.benefit > 0).sum()),
        "median_benefit": float(frame.benefit.median()),
        "n_subject_median_better": int((subject_median > 0).sum()),
        "pass": bool(
            int((frame.benefit > 0).sum()) >= int(min_patient_seed)
            and int((subject_median > 0).sum()) >= int(min_subjects)
        ),
    }


def _comparison_checks(benefits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for keys, frame in benefits.groupby(
        ["mode_count", "baseline", "metric"], sort=True
    ):
        result = _count_gate(
            frame,
            min_patient_seed=cfg["evaluation"][
                "pilot_min_patient_seed_better"
            ],
            min_subjects=cfg["evaluation"]["pilot_min_subjects_better"],
        )
        rows.append(
            {
                "mode_count": int(keys[0]),
                "baseline": keys[1],
                "metric": keys[2],
                **result,
            }
        )
    return pd.DataFrame(rows)


def _seed_stability(metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    primary = metrics[
        (metrics.lesion.astype(str) == "none")
        & (metrics.control == "intact")
    ]
    rows = []
    for metric in METRICS:
        values = []
        for subject in SUBJECTS:
            pivot = primary[primary.subject == subject].pivot(
                index="mode_count", columns="seed", values=metric
            )
            for left_index, left in enumerate(SEEDS):
                for right in SEEDS[left_index + 1 :]:
                    statistic = spearmanr(pivot[left], pivot[right]).statistic
                    values.append(float(statistic))
                    rows.append(
                        {
                            "metric": metric,
                            "subject": subject,
                            "seed_left": left,
                            "seed_right": right,
                            "spearman": float(statistic),
                            "median_for_gate": np.nan,
                            "pass": False,
                        }
                    )
        median = float(np.nanmedian(values))
        for row in rows:
            if row["metric"] == metric:
                row["median_for_gate"] = median
                row["pass"] = bool(median >= threshold)
    return pd.DataFrame(rows)


def _lesion_benefits(metrics: pd.DataFrame) -> pd.DataFrame:
    intact = metrics[
        (metrics.control == "intact")
        & (metrics.lesion.astype(str) == "none")
    ].set_index(["mode_count", "subject", "seed"])
    rows = []
    for mode_count in range(1, 5):
        current = metrics[
            (metrics.control == "intact")
            & (metrics.mode_count == mode_count)
        ]
        lesion_names = [
            "graph",
            "inhibition",
            "drop_forward",
            "drop_reverse",
            "mode_collapse",
        ]
        if mode_count >= 2:
            lesion_names.append("drop_dominant_mode")
        for lesion in lesion_names:
            lesion_frame = current[current.lesion == lesion].set_index(
                ["mode_count", "subject", "seed"]
            )
            for metric in METRICS:
                left, right = intact[metric].align(
                    lesion_frame[metric], join="inner"
                )
                for (k, subject, seed), value in (right - left).items():
                    rows.append(
                        {
                            "mode_count": int(k),
                            "lesion": lesion,
                            "metric": metric,
                            "subject": subject,
                            "seed": int(seed),
                            "benefit": float(value),
                        }
                    )
        if mode_count == 1:
            direction = current[
                current.lesion.isin(["drop_forward", "drop_reverse"])
            ].groupby(["subject", "seed"])[list(METRICS)].mean()
            for metric in METRICS:
                reference = intact.xs(mode_count, level="mode_count")[metric]
                left, right = reference.align(direction[metric], join="inner")
                for (subject, seed), value in (right - left).items():
                    rows.append(
                        {
                            "mode_count": 1,
                            "lesion": "direction_removal_mean",
                            "metric": metric,
                            "subject": subject,
                            "seed": int(seed),
                            "benefit": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def _lesion_checks(benefits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for keys, frame in benefits.groupby(
        ["mode_count", "lesion", "metric"], sort=True
    ):
        result = _count_gate(
            frame,
            min_patient_seed=cfg["evaluation"][
                "pilot_min_patient_seed_better"
            ],
            min_subjects=cfg["evaluation"]["pilot_min_subjects_better"],
        )
        rows.append(
            {
                "mode_count": int(keys[0]),
                "lesion": keys[1],
                "metric": keys[2],
                **result,
            }
        )
    return pd.DataFrame(rows)


def _mode_gate(
    mode_count: int,
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    lesion: pd.DataFrame,
) -> dict:
    required_baselines = list(BASELINES)
    if mode_count == 1:
        required_baselines.remove("mode_shuffle")
    comparison_use = comparison[
        (comparison.mode_count == mode_count)
        & comparison.baseline.isin(required_baselines)
        & comparison.metric.isin(METRICS)
    ]
    comparison_pass = bool(
        len(comparison_use) == len(required_baselines) * len(METRICS)
        and comparison_use["pass"].all()
    )
    stability_metric = (
        stability.groupby("metric")["pass"].first().reindex(METRICS)
    )
    stability_pass = bool(
        stability_metric.notna().all() and stability_metric.all()
    )
    if mode_count == 1:
        candidates = ("direction_removal_mean",)
    else:
        candidates = ("mode_collapse", "drop_dominant_mode")
    lesion_options = {}
    for candidate in candidates:
        use = lesion[
            (lesion.mode_count == mode_count)
            & (lesion.lesion == candidate)
            & lesion.metric.isin(METRICS)
        ]
        lesion_options[candidate] = bool(
            len(use) == len(METRICS) and use["pass"].all()
        )
    lesion_pass = bool(any(lesion_options.values()))
    return {
        "mode_count": mode_count,
        "comparison_pass": comparison_pass,
        "stability_pass": stability_pass,
        "lesion_pass": lesion_pass,
        "lesion_options": lesion_options,
        "hard_gate_pass": bool(
            comparison_pass and stability_pass and lesion_pass
        ),
    }


def _plot(
    benefits: pd.DataFrame,
    lesion_benefits: pd.DataFrame,
    prior_audit: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6))
    prior = prior_audit.groupby("mode_count").agg(
        stability=("split_half_aligned_mode_cosine_median", "median"),
        reconstruction=("heldout_soft_reconstruction_cosine_median", "median"),
    )
    axes[0, 0].plot(prior.index, prior.stability, "o-", label="Split-half")
    axes[0, 0].plot(
        prior.index, prior.reconstruction, "s-", label="Held-out fit"
    )
    axes[0, 0].set(
        xlabel="Number of path modes (K)",
        ylabel="Cosine similarity",
        xticks=range(1, 5),
        ylim=(0, 1.03),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis, metric, title in (
        (axes[0, 1], "precedence_mae", "Precedence"),
        (axes[1, 0], "path_sliced_wasserstein", "Whole path"),
    ):
        use = benefits[benefits.metric == metric]
        for baseline in BASELINES:
            frame = use[use.baseline == baseline]
            if frame.empty:
                continue
            med = frame.groupby("mode_count").benefit.median()
            axis.plot(
                med.index,
                med.values,
                "o-",
                color=COLORS[baseline],
                label=baseline.replace("_", " "),
            )
        axis.axhline(0, color="#999999", lw=0.8)
        axis.set(
            xlabel="Number of path modes (K)",
            ylabel=f"{title} benefit",
            xticks=range(1, 5),
        )
        axis.legend(frameon=False, fontsize=6)
    mode_lesions = lesion_benefits[
        lesion_benefits.lesion.isin(
            ["direction_removal_mean", "mode_collapse", "drop_dominant_mode"]
        )
    ]
    for (lesion_name, metric), frame in mode_lesions.groupby(
        ["lesion", "metric"]
    ):
        med = frame.groupby("mode_count").benefit.median()
        axes[1, 1].plot(
            med.index,
            med.values,
            "o-",
            label=f"{lesion_name}: {metric.split('_')[0]}",
        )
    axes[1, 1].axhline(0, color="#999999", lw=0.8)
    axes[1, 1].set(
        xlabel="Number of path modes (K)",
        ylabel="Lesion cost",
        xticks=range(1, 5),
    )
    axes[1, 1].legend(frameon=False, fontsize=5.7)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.17,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"persistent_path_pilot_gate.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    (figure_dir / "README.md").write_text(
        """### persistent_path_pilot_gate.png

A 检查 K=1–4 路径模式能否在 train80 分半中复现，并报告对 heldout20 的无监督重建。
B、C 分别显示 intact path-mode RNN 相对各对照的 precedence 和完整路径收益，正值更好。
D 显示去掉方向或路径模式后误差是否上升。

**关注点**：只有 B、C 对全部预注册对照同时为正，且 D 的结构损伤确实破坏结果，才允许进入 34 人正式运行。
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v0_9.yaml",
    )
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    input_root = (
        args.input_root
        if args.input_root is not None
        else ROOT / cfg["outputs"]["pilot"]
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else input_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, inventory = _load_runs(input_root)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    expected = len(SUBJECTS) * len(SEEDS) * len(SPECS)
    complete = bool(
        len(inventory) == expected
        and (inventory.status == "COMPLETE").all()
        and inventory.ictal_target_sealed.all()
        and inventory[
            [
                "metrics_exists",
                "checkpoint_exists",
                "training_log_exists",
                "rollout_exists",
            ]
        ].all().all()
    )
    if not complete:
        raise RuntimeError(
            f"pilot incomplete: expected {expected}, "
            f"complete={(inventory.status == 'COMPLETE').sum()}"
        )
    metrics.to_csv(output_dir / "all_seed_metrics.csv", index=False)
    benefits = _paired_benefits(metrics)
    benefits.to_csv(output_dir / "paired_benefits.csv", index=False)
    comparison = _comparison_checks(benefits, cfg)
    comparison.to_csv(output_dir / "comparison_gate_checks.csv", index=False)
    stability = _seed_stability(
        metrics,
        float(cfg["evaluation"]["pilot_seed_rank_stability_min"]),
    )
    stability.to_csv(output_dir / "seed_rank_stability.csv", index=False)
    lesion_benefits = _lesion_benefits(metrics)
    lesion_benefits.to_csv(
        output_dir / "lesion_benefits.csv", index=False
    )
    lesion = _lesion_checks(lesion_benefits, cfg)
    lesion.to_csv(output_dir / "lesion_gate_checks.csv", index=False)
    mode_gates = [
        _mode_gate(mode_count, comparison, stability, lesion)
        for mode_count in range(1, 5)
    ]
    passing = [
        item["mode_count"] for item in mode_gates if item["hard_gate_pass"]
    ]
    summary = {
        "status": "complete",
        "pilot_complete": True,
        "n_expected_runs": expected,
        "n_complete_runs": expected,
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "mode_gates": mode_gates,
        "hard_gate_pass": bool(passing),
        "selected_mode_count": min(passing) if passing else None,
        "next_action": (
            "freeze_selected_K_and_launch_34x3_exact_coverage"
            if passing
            else "bounded_negative_stop_no_ictal_read"
        ),
        "ictal_target_read": False,
    }
    (output_dir / "pilot_gate_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    prior_audit = pd.read_csv(
        ROOT
        / cfg["outputs"]["prior"]
        / "path_mode_prior_audit.csv"
    )
    _plot(benefits, lesion_benefits, prior_audit, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
