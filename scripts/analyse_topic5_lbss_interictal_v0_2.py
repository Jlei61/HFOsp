#!/usr/bin/env python3
"""Audit and aggregate the 465 target-free LBSS units at patient level."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
COLORS = {
    "L0_LOCAL_ONLY": "#7f8b94",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL": "#3d78a6",
    "L2_LOCAL_PLUS_RANDOM_LR": "#9a72b0",
    "L3_LOCAL_PLUS_LEARNED_LR": "#c83e32",
    "C_L3_ORDER_SHUFFLED": "#b5b5b5",
}
OLD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-rnn-motif-cross-state-v0-4/"
    "results/topic5_rnn_motif_cross_state_benchmark_v0_4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def paired_test(values: np.ndarray, tolerance: float = 1e-9) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    tied = np.abs(values) <= tolerance
    nonzero = values[~tied]
    if nonzero.size == 0:
        p = 1.0
    else:
        p = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else float("nan"),
        "n_positive": int((values > tolerance).sum()),
        "n_negative": int((values < -tolerance).sum()),
        "n_tied": int(tied.sum()),
        "wilcoxon_p_two_sided": p,
    }


def load_distance(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text()))
    return frame[["event_index", "rank_index", "frontier_distance_mm", "contact_nll", "top1"]]


def old_no_rec_metrics(fit_id: str, seed: int) -> dict:
    path = OLD_ROOT / "per_subject" / fit_id / "M0_NO_REC__rnn" / f"seed{seed}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def aggregate_patient(fit_table: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    seed_fit = fit_table.groupby(["subject", "fit_id", "arm"], sort=False)[value_columns].median().reset_index()
    return seed_fit.groupby(["subject", "arm"], sort=False)[value_columns].mean().reset_index()


def plot_summary(patient: pd.DataFrame, summary: dict, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    pivot = patient.pivot(index="subject", columns="arm")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.15))

    comparisons = [
        ("L0_LOCAL_ONLY", "L3_LOCAL_PLUS_LEARNED_LR", "L3 - local"),
        ("L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L3_LOCAL_PLUS_LEARNED_LR", "L3 - extra local"),
        ("L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR", "L3 - random LR"),
        ("C_L3_ORDER_SHUFFLED", "L3_LOCAL_PLUS_LEARNED_LR", "True - shuffle"),
    ]
    ax = axes[0]
    for index, (reference, target, _) in enumerate(comparisons):
        values = pivot["test_contact_nll"][reference] - pivot["test_contact_nll"][target]
        ax.scatter(np.full(len(values), index) + np.linspace(-0.10, 0.10, len(values)), values,
                   s=16, color="#8e969d", alpha=0.72)
        ax.scatter(index, np.median(values), s=46, color=COLORS[target], edgecolor="white", lw=0.5)
    ax.axhline(0, color="#202020", lw=0.8)
    ax.set_xticks(range(4), [item[2] for item in comparisons], rotation=28, ha="right")
    ax.set_ylabel("Held-out NLL benefit")

    ax = axes[1]
    for index, (reference, target, _) in enumerate(comparisons[:3]):
        values = pivot["distal_contact_nll"][reference] - pivot["distal_contact_nll"][target]
        ax.scatter(np.full(len(values), index) + np.linspace(-0.10, 0.10, len(values)), values,
                   s=16, color="#8e969d", alpha=0.72)
        ax.scatter(index, np.nanmedian(values), s=46, color="#c83e32", edgecolor="white", lw=0.5)
    ax.axhline(0, color="#202020", lw=0.8)
    ax.set_xticks(range(3), ["vs local", "vs extra local", "vs random LR"], rotation=28, ha="right")
    ax.set_ylabel("Distal NLL benefit")

    ax = axes[2]
    shown = ARMS
    for patient_name in pivot.index:
        ax.plot(range(len(shown)), [pivot["rollout_spearman"].loc[patient_name, arm] for arm in shown],
                color="#c5c9cd", lw=0.55, alpha=0.55)
    medians = [pivot["rollout_spearman"][arm].median() for arm in shown]
    ax.scatter(range(len(shown)), medians, s=42, color=[COLORS[arm] for arm in shown], zorder=3)
    ax.set_xticks(range(len(shown)), ["Local", "+extra", "+random", "+learned", "Shuffle"],
                  rotation=28, ha="right")
    ax.set_ylabel("Free-rollout rank correlation")

    for label, ax in zip("ABC", axes):
        ax.text(-0.17, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"stage_d_interictal_lbss_summary.{suffix}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### stage_d_interictal_lbss_summary.png\n\n"
        "A 比较 L3 在全部 held-out next-rank decisions 上相对三种 matched arm 及顺序打乱的 NLL 增益。"
        "B 只看由真实训练事件冻结的 distal transitions；C 展示只给第一 rank 后自由生成的患者级传播排序一致性。\n\n"
        "**关注点**：患者是统计单位，正值表示 L3 更好；该图只使用间期数据。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    args = parser.parse_args()
    out = args.out_root.resolve()
    if not (out / "FORMAL_TRAINING_COMPLETE.json").exists():
        raise RuntimeError("formal training is not complete")
    input_manifest = json.loads((out / "INPUT_CACHE_MANIFEST.json").read_text())
    fit_ids = sorted({row["fit_id"] for row in input_manifest["files"]})
    snapshot = out / "run_snapshot"
    expected_trainer = sha256(snapshot / "scripts/train_topic5_lbss_unit_v0_2.py")
    expected_model = sha256(snapshot / "src/topic5_lbss_rnn_v0_2.py")
    rows = []
    audit_errors = []
    for fit_id in fit_ids:
        per_seed_reference: dict[int, tuple[str, float, float]] = {}
        for seed in (0, 1, 2):
            initial_l2 = None
            initial_l3 = None
            local_reference = None
            for arm in ARMS:
                directory = out / "per_fit" / fit_id / arm / f"seed{seed}"
                metrics_path = directory / "metrics.json"
                graph_path = directory / "graph.npz"
                if not metrics_path.exists() or not graph_path.exists():
                    audit_errors.append(f"missing:{fit_id}:{arm}:seed{seed}")
                    continue
                metrics = json.loads(metrics_path.read_text())
                if not metrics.get("converged") or not metrics.get("best_checkpoint_eligible"):
                    audit_errors.append(f"invalid_checkpoint:{fit_id}:{arm}:seed{seed}")
                if metrics.get("target_values_read") is not False:
                    audit_errors.append(f"target_read:{fit_id}:{arm}:seed{seed}")
                if metrics["producer_hashes"]["trainer"] != expected_trainer:
                    audit_errors.append(f"trainer_hash:{fit_id}:{arm}:seed{seed}")
                if metrics["producer_hashes"]["model"] != expected_model:
                    audit_errors.append(f"model_hash:{fit_id}:{arm}:seed{seed}")
                if not finite_tree({key: value for key, value in metrics.items() if key != "config"}):
                    audit_errors.append(f"nonfinite:{fit_id}:{arm}:seed{seed}")
                reference = (
                    metrics["distance_bin_reference_sha256"],
                    metrics["distance_thresholds_mm"]["q50"],
                    metrics["distance_thresholds_mm"]["q80"],
                )
                if seed in per_seed_reference and per_seed_reference[seed] != reference:
                    audit_errors.append(f"distance_reference_mismatch:{fit_id}:seed{seed}")
                per_seed_reference[seed] = reference
                if arm == "C_L3_ORDER_SHUFFLED":
                    shuffle = metrics["shuffle_audit"]
                    if not shuffle or not shuffle["heldout_test_unchanged"]:
                        audit_errors.append(f"shuffle_test_changed:{fit_id}:seed{seed}")
                graph = np.load(graph_path, allow_pickle=False)
                if local_reference is None:
                    local_reference = graph["local_mask"]
                elif not np.array_equal(local_reference, graph["local_mask"]):
                    audit_errors.append(f"local_mask_mismatch:{fit_id}:seed{seed}:{arm}")
                if arm == "L2_LOCAL_PLUS_RANDOM_LR":
                    initial_l2 = graph["initial_added_mask"]
                if arm == "L3_LOCAL_PLUS_LEARNED_LR":
                    initial_l3 = graph["initial_added_mask"]

                old = old_no_rec_metrics(fit_id, seed)
                rows.append({
                    "fit_id": fit_id,
                    "subject": metrics["subject"],
                    "scope": metrics["scope"],
                    "arm": arm,
                    "seed": seed,
                    "test_contact_nll": metrics["test"]["contact_nll"],
                    "test_top1": metrics["test"]["top1"],
                    "local_contact_nll": metrics["distance_bins"]["local"]["contact_nll"],
                    "local_top1": metrics["distance_bins"]["local"]["top1"],
                    "local_n": metrics["distance_bins"]["local"]["n"],
                    "local_distance_median_mm": metrics["distance_bins"]["local"]["distance_median_mm"],
                    "intermediate_contact_nll": metrics["distance_bins"]["intermediate"]["contact_nll"],
                    "intermediate_top1": metrics["distance_bins"]["intermediate"]["top1"],
                    "intermediate_n": metrics["distance_bins"]["intermediate"]["n"],
                    "intermediate_distance_median_mm": metrics["distance_bins"]["intermediate"]["distance_median_mm"],
                    "distal_contact_nll": metrics["distance_bins"]["distal"]["contact_nll"],
                    "distal_top1": metrics["distance_bins"]["distal"]["top1"],
                    "distal_n": metrics["distance_bins"]["distal"]["n"],
                    "distal_distance_median_mm": metrics["distance_bins"]["distal"]["distance_median_mm"],
                    "rollout_spearman": metrics["rollout"]["seed_removed_spearman_median"],
                    "rollout_length_ratio": metrics["rollout"]["length_ratio_median"],
                    "no_rec_contact_nll": old["test"]["contact_nll"],
                    "n_epochs": metrics["n_epochs"],
                    "seconds": metrics["seconds"],
                })
            if initial_l2 is None or initial_l3 is None or not np.array_equal(initial_l2, initial_l3):
                audit_errors.append(f"L2_L3_initial_mask_mismatch:{fit_id}:seed{seed}")

    if audit_errors:
        (out / "INTERICTAL_AGGREGATION_BLOCKED.json").write_text(json.dumps({
            "status": "BLOCKED", "errors": audit_errors
        }, indent=2) + "\n")
        raise RuntimeError(f"interictal audit failed with {len(audit_errors)} errors")
    table = pd.DataFrame(rows)

    # Continuous distance-gain slope on identical held-out decisions.
    slopes = []
    for fit_id in fit_ids:
        for seed in (0, 1, 2):
            frames = {
                arm: load_distance(out / "per_fit" / fit_id / arm / f"seed{seed}" / "distance_decisions.json")
                for arm in ARMS
            }
            l3 = frames["L3_LOCAL_PLUS_LEARNED_LR"].rename(columns={"contact_nll": "l3_nll"})
            for reference in ("L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L2_LOCAL_PLUS_RANDOM_LR"):
                merged = l3.merge(
                    frames[reference][["event_index", "rank_index", "contact_nll"]],
                    on=["event_index", "rank_index"], how="inner", validate="one_to_one"
                )
                gain = merged["contact_nll"] - merged["l3_nll"]
                distance = merged["frontier_distance_mm"]
                slope = float(np.polyfit(distance, gain, 1)[0]) if len(merged) >= 3 else float("nan")
                slopes.append({"fit_id": fit_id, "seed": seed, "reference_arm": reference,
                               "distance_gain_slope_per_mm": slope, "n": len(merged)})
    pd.DataFrame(slopes).to_csv(out / "interictal_distance_gain_slopes.csv", index=False)
    table.to_csv(out / "interictal_per_fit_seed.csv", index=False)
    value_columns = [
        "test_contact_nll", "test_top1",
        "local_contact_nll", "local_top1", "local_n", "local_distance_median_mm",
        "intermediate_contact_nll", "intermediate_top1", "intermediate_n", "intermediate_distance_median_mm",
        "distal_contact_nll", "distal_top1", "distal_n", "distal_distance_median_mm",
        "rollout_spearman", "rollout_length_ratio", "no_rec_contact_nll", "n_epochs", "seconds",
    ]
    patient = aggregate_patient(table, value_columns)
    patient.to_csv(out / "interictal_per_patient.csv", index=False)
    pivot = patient.pivot(index="subject", columns="arm")
    comparisons = {
        "L0_vs_no_rec": pivot["no_rec_contact_nll"]["L0_LOCAL_ONLY"] - pivot["test_contact_nll"]["L0_LOCAL_ONLY"],
        "L3_vs_L0_all": pivot["test_contact_nll"]["L0_LOCAL_ONLY"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_all": pivot["test_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2_all": pivot["test_contact_nll"]["L2_LOCAL_PLUS_RANDOM_LR"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_shuffle_all": pivot["test_contact_nll"]["C_L3_ORDER_SHUFFLED"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L0_distal": pivot["distal_contact_nll"]["L0_LOCAL_ONLY"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_distal": pivot["distal_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2_distal": pivot["distal_contact_nll"]["L2_LOCAL_PLUS_RANDOM_LR"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
    }
    summary = {
        "contract": "topic5_lbss_interictal_summary_v0_2",
        "n_patients": int(patient.subject.nunique()),
        "n_fits": len(fit_ids),
        "n_units": len(table),
        "comparisons": {name: paired_test(values.to_numpy()) for name, values in comparisons.items()},
        "target_values_read": False,
    }
    (out / "INTERICTAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "INTERICTAL_AUDIT.json").write_text(json.dumps({
        "status": "PASS",
        "expected_units": 465,
        "actual_units": len(table),
        "all_checkpoints_converged_and_eligible": True,
        "all_test_targets_shared": True,
        "all_distance_bins_shared": True,
        "all_L2_L3_initial_masks_shared": True,
        "target_values_read": False,
    }, indent=2) + "\n")
    plot_summary(patient, summary, out)
    (out / "INTERICTAL_ANALYSIS_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "n_units": len(table), "target_values_read": False
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
