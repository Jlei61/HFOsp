#!/usr/bin/env python3
"""
Refine-SOZ validation figure for Yuquan dataset (Figure 1 equivalent).

Reproduces the core panels from the legacy pipeline:
- plotting_fig1_hfoHist.py  (per-subject raw/refined bar charts with SOZ overlay)
- plotting_fig2_AUC_groupingComp.py  (cohort AUC ROC + Raw vs Refined comparison)

Usage:
    python scripts/plot_refine_soz_validation.py [--out-dir results/refine_soz_validation]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_rel, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.event_periodicity import match_bipolar_soz, _normalize_channel_name


SOZ_COLOR = np.array([5, 88, 173]) / 255
NON_SOZ_COLOR = "gray"
THRESH_COLOR = "tab:blue"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def load_soz_channels(soz_json: Path) -> Dict[str, List[str]]:
    with open(soz_json) as f:
        return json.load(f)


def load_subject_params(params_json: Path) -> dict:
    with open(params_json) as f:
        return json.load(f)


def get_pick_k(subject_params: dict, sub_name: str, dataset: str = "yuquan") -> float:
    defaults = subject_params.get(dataset, {}).get("_defaults", {})
    sub_cfg = subject_params.get(dataset, {}).get(sub_name, {})
    return sub_cfg.get("pick_k", defaults.get("pick_k", 1.0))


def classify_channels_soz(
    ch_names: np.ndarray, soz_list: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    soz_set = {_normalize_channel_name(c) for c in soz_list}
    in_soz = np.array(
        [match_bipolar_soz(ch, soz_set) == "soz" for ch in ch_names], dtype=bool
    )
    return np.where(in_soz)[0], np.where(~in_soz)[0]


def load_raw_counts(sub_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Sum per-record gpu.npz event counts → total raw counts per channel."""
    counts_list = []
    ch_names = None
    for fname in sorted(os.listdir(sub_dir)):
        if fname.endswith("_gpu.npz") and not fname.startswith("_"):
            data = np.load(sub_dir / fname, allow_pickle=True)
            counts_list.append(data["events_count"])
            if ch_names is None:
                ch_names = data["chns_names"]
    if not counts_list:
        return None, None
    return np.sum(counts_list, axis=0), ch_names


def load_refine_counts(sub_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    refine_path = sub_dir / "_refineGpu.npz"
    if not refine_path.exists():
        return None, None
    data = np.load(refine_path, allow_pickle=True)
    return data["events_count"], data["chns_names"]


def plot_channel_bar(
    ax: plt.Axes,
    counts: np.ndarray,
    ch_names: np.ndarray,
    soz_idx: np.ndarray,
    non_soz_idx: np.ndarray,
    title: str,
    pick_k: float = 1.0,
    show_thresh: bool = True,
):
    if len(soz_idx) > 0:
        ax.bar(soz_idx, counts[soz_idx], color=SOZ_COLOR, edgecolor=SOZ_COLOR, label="SOZ")
    if len(non_soz_idx) > 0:
        ax.bar(non_soz_idx, counts[non_soz_idx], color=NON_SOZ_COLOR, edgecolor=NON_SOZ_COLOR)

    if show_thresh:
        mean_c = np.mean(counts)
        std_c = np.std(counts)
        thresh = mean_c + pick_k * std_c
        ax.axhline(thresh, color=THRESH_COLOR, linestyle="--",
                   label=f"mean+{pick_k:.1f}*std", linewidth=1)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Channels")
    ax.set_ylabel("Events")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_auc(counts: np.ndarray, soz_idx: np.ndarray, n_channels: int) -> Tuple[float, np.ndarray, np.ndarray]:
    y_true = np.zeros(n_channels, dtype=int)
    y_true[soz_idx] = 1
    if y_true.sum() == 0 or y_true.sum() == n_channels:
        return np.nan, np.array([]), np.array([])
    fpr, tpr, _ = roc_curve(y_true, counts, pos_label=1)
    return auc(fpr, tpr), fpr, tpr


def run(
    data_root: Path,
    soz_json: Path,
    params_json: Path,
    out_dir: Path,
    dataset: str = "yuquan",
):
    soz_all = load_soz_channels(soz_json)
    subject_params = load_subject_params(params_json)
    dataset_label = dataset.capitalize()

    subjects = sorted(os.listdir(data_root))
    subjects = [s for s in subjects if (data_root / s).is_dir()]

    ds_params = subject_params.get(dataset, {})
    ds_subject_ids = {k for k in ds_params if not k.startswith("_")}
    if ds_subject_ids:
        subjects = [s for s in subjects if s in ds_subject_ids]

    results = {}

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_sub_dir = fig_dir / "per_subject"
    per_sub_dir.mkdir(exist_ok=True)

    for sub in subjects:
        sub_dir = data_root / sub
        raw_counts, raw_chns = load_raw_counts(sub_dir)
        ref_counts, ref_chns = load_refine_counts(sub_dir)

        if raw_counts is None:
            print(f"  {sub}: skip (no gpu.npz)")
            continue

        soz_list = soz_all.get(sub, [])
        if not soz_list:
            print(f"  {sub}: skip (no SOZ info)")
            continue

        pick_k = get_pick_k(subject_params, sub, dataset=dataset)

        soz_idx_raw, non_soz_idx_raw = classify_channels_soz(raw_chns, soz_list)

        raw_auc, raw_fpr, raw_tpr = compute_auc(raw_counts, soz_idx_raw, len(raw_chns))

        ref_auc = np.nan
        ref_fpr, ref_tpr = np.array([]), np.array([])
        has_refine = ref_counts is not None
        if has_refine:
            soz_idx_ref, non_soz_idx_ref = classify_channels_soz(ref_chns, soz_list)
            ref_auc, ref_fpr, ref_tpr = compute_auc(ref_counts, soz_idx_ref, len(ref_chns))

        results[sub] = {
            "raw_auc": raw_auc, "ref_auc": ref_auc,
            "raw_fpr": raw_fpr, "raw_tpr": raw_tpr,
            "ref_fpr": ref_fpr, "ref_tpr": ref_tpr,
            "n_soz": int(len(soz_idx_raw)),
            "n_channels": int(len(raw_chns)),
            "pick_k": pick_k,
        }

        n_panels = 2 if has_refine else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
        if n_panels == 1:
            axes = [axes]

        plot_channel_bar(
            axes[0], raw_counts, raw_chns, soz_idx_raw, non_soz_idx_raw,
            f"{sub} — Raw HFO Counts (pre-refine)", pick_k=pick_k,
        )

        if has_refine:
            plot_channel_bar(
                axes[1], ref_counts, ref_chns, soz_idx_ref, non_soz_idx_ref,
                f"{sub} — Refined HFO Counts (post-refine)", pick_k=pick_k,
            )

        fig.tight_layout()
        fig.savefig(per_sub_dir / f"{sub}_refine_soz.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {sub}: raw_AUC={raw_auc:.3f}, ref_AUC={ref_auc:.3f}, SOZ={len(soz_idx_raw)}/{len(raw_chns)}")

    valid_subs = [s for s in results if not np.isnan(results[s]["raw_auc"])]
    if not valid_subs:
        print("No subjects with valid AUC. Aborting cohort plot.")
        return

    # ── Cohort Figure: 2×2 layout ──
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Raw AUC ROC curves
    ax_raw_roc = fig.add_subplot(gs[0, 0])
    for sub in valid_subs:
        r = results[sub]
        if len(r["raw_fpr"]) > 0:
            ax_raw_roc.plot(r["raw_fpr"], r["raw_tpr"],
                           label=f"{sub}: {r['raw_auc']:.2f}", linewidth=1)
    ax_raw_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    raw_aucs = [results[s]["raw_auc"] for s in valid_subs if not np.isnan(results[s]["raw_auc"])]
    ax_raw_roc.set_title(f"Raw HFO — ROC (mean AUC={np.mean(raw_aucs):.3f})", fontsize=12)
    ax_raw_roc.set_xlabel("FPR", fontsize=11)
    ax_raw_roc.set_ylabel("TPR", fontsize=11)
    ax_raw_roc.legend(fontsize=6, loc="lower right", ncol=2)
    ax_raw_roc.spines["top"].set_visible(False)
    ax_raw_roc.spines["right"].set_visible(False)

    # Panel B: Refined AUC ROC curves
    ax_ref_roc = fig.add_subplot(gs[0, 1])
    ref_valid = [s for s in valid_subs if not np.isnan(results[s]["ref_auc"])]
    for sub in ref_valid:
        r = results[sub]
        if len(r["ref_fpr"]) > 0:
            ax_ref_roc.plot(r["ref_fpr"], r["ref_tpr"],
                           label=f"{sub}: {r['ref_auc']:.2f}", linewidth=1)
    ax_ref_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ref_aucs = [results[s]["ref_auc"] for s in ref_valid]
    if ref_aucs:
        ax_ref_roc.set_title(f"Refined HFO — ROC (mean AUC={np.mean(ref_aucs):.3f})", fontsize=12)
    else:
        ax_ref_roc.set_title("Refined HFO — ROC (no data)", fontsize=12)
    ax_ref_roc.set_xlabel("FPR", fontsize=11)
    ax_ref_roc.set_ylabel("TPR", fontsize=11)
    ax_ref_roc.legend(fontsize=6, loc="lower right", ncol=2)
    ax_ref_roc.spines["top"].set_visible(False)
    ax_ref_roc.spines["right"].set_visible(False)

    # Panel C: AUC boxplot comparison (Raw vs Refined) — paired
    ax_box = fig.add_subplot(gs[1, 0])
    paired_subs = [s for s in ref_valid if not np.isnan(results[s]["raw_auc"])]
    if paired_subs:
        raw_paired = np.array([results[s]["raw_auc"] for s in paired_subs])
        ref_paired = np.array([results[s]["ref_auc"] for s in paired_subs])

        jitter = 0.04 * np.random.randn(len(paired_subs))
        ax_box.scatter(np.zeros(len(paired_subs)) + jitter, raw_paired,
                      s=60, color="tab:red", edgecolor="w", alpha=0.7, zorder=3)
        ax_box.scatter(np.ones(len(paired_subs)) + jitter, ref_paired,
                      s=60, color="tab:red", edgecolor="k", linewidth=1.2, alpha=0.7, zorder=3)
        for i in range(len(paired_subs)):
            ax_box.plot([jitter[i], 1 + jitter[i]], [raw_paired[i], ref_paired[i]],
                       "k--", linewidth=0.4, alpha=0.5, zorder=1)

        bp = ax_box.boxplot([raw_paired, ref_paired], positions=[0, 1],
                           showfliers=False, showcaps=False, widths=0.35,
                           medianprops=dict(linewidth=3, color="gray"),
                           whiskerprops=dict(linewidth=1.5, color="gray"),
                           boxprops=dict(linewidth=1.5, color="gray"))

        if len(paired_subs) >= 3:
            try:
                _, p_val = wilcoxon(raw_paired, ref_paired)
            except Exception:
                _, p_val = ttest_rel(raw_paired, ref_paired)
        else:
            p_val = np.nan
        ymax = max(raw_paired.max(), ref_paired.max())
        tick_y = ymax + 0.05
        ax_box.plot([0, 0, 1, 1], [tick_y - 0.01, tick_y, tick_y, tick_y - 0.01], "k-", linewidth=1)
        p_str = f"p={p_val:.3f}" if not np.isnan(p_val) else "n<3"
        ax_box.text(0.5, tick_y + 0.01, p_str, ha="center", va="bottom", fontsize=10)

        raw_mean, ref_mean = raw_paired.mean(), ref_paired.mean()
        ax_box.scatter(0, raw_mean, s=100, marker="^", facecolor="none", edgecolor="k", zorder=5)
        ax_box.scatter(1, ref_mean, s=100, marker="^", facecolor="none", edgecolor="k", zorder=5)
        ax_box.text(-0.15, raw_mean, f"μ={raw_mean:.2f}", fontsize=8, va="center", ha="right",
                   bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.1))
        ax_box.text(1.15, ref_mean, f"μ={ref_mean:.2f}", fontsize=8, va="center", ha="left",
                   bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.1))

    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(["Raw Counts", "Refined Counts"], fontsize=11)
    ax_box.set_ylabel("AUC", fontsize=12, fontweight="bold")
    ax_box.set_title(f"Raw vs Refined AUC ({dataset_label})", fontsize=12)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)

    # Panel D: Per-subject AUC summary bar chart
    ax_bar = fig.add_subplot(gs[1, 1])
    bar_subs = [s for s in valid_subs]
    x = np.arange(len(bar_subs))
    width = 0.35
    raw_bars = [results[s]["raw_auc"] for s in bar_subs]
    ref_bars = [results[s].get("ref_auc", np.nan) for s in bar_subs]
    ax_bar.bar(x - width / 2, raw_bars, width, label="Raw", color="lightcoral", edgecolor="gray")
    ref_mask = ~np.isnan(ref_bars)
    if ref_mask.any():
        ax_bar.bar(x[ref_mask] + width / 2, np.array(ref_bars)[ref_mask], width,
                  label="Refined", color=SOZ_COLOR, edgecolor="gray")
    ax_bar.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s[:8] for s in bar_subs], rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("AUC", fontsize=11)
    ax_bar.set_title("Per-subject AUC", fontsize=12)
    ax_bar.legend(fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.suptitle(f"{dataset_label} HFO Detection — Refine vs SOZ Validation", fontsize=14, fontweight="bold")
    fig.savefig(fig_dir / f"{dataset}_refine_soz_cohort.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCohort figure saved: {fig_dir / f'{dataset}_refine_soz_cohort.png'}")

    # ── Summary JSON ──
    summary = {
        "n_subjects_total": len(subjects),
        "n_subjects_with_gpu": sum(1 for s in subjects if (data_root / s / "_refineGpu.npz").exists() or
                                   any(f.endswith("_gpu.npz") for f in os.listdir(data_root / s)
                                       if not f.startswith("_"))),
        "n_subjects_with_refine_and_soz": len(paired_subs) if paired_subs else 0,
        "raw_auc_mean": float(np.mean(raw_aucs)) if raw_aucs else None,
        "raw_auc_median": float(np.median(raw_aucs)) if raw_aucs else None,
        "ref_auc_mean": float(np.mean(ref_aucs)) if ref_aucs else None,
        "ref_auc_median": float(np.median(ref_aucs)) if ref_aucs else None,
        "per_subject": {
            s: {"raw_auc": float(results[s]["raw_auc"]),
                "ref_auc": float(results[s]["ref_auc"]),
                "n_soz": results[s]["n_soz"],
                "n_channels": results[s]["n_channels"],
                "pick_k": results[s]["pick_k"]}
            for s in valid_subs
        },
    }
    summary_path = out_dir / "refine_soz_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Refine-SOZ validation figure")
    parser.add_argument("--dataset", default="yuquan", choices=["yuquan", "epilepsiae"],
                       help="Dataset to validate")
    parser.add_argument("--data-root", default=None,
                       help="Data root directory (auto-detected from dataset)")
    parser.add_argument("--soz-json", default=None,
                       help="SOZ channels JSON (auto-detected from dataset)")
    parser.add_argument("--params-json", default=str(PROJECT_ROOT / "config" / "subject_params.json"),
                       help="Subject parameters JSON")
    parser.add_argument("--out-dir", default=None,
                       help="Output directory (auto-detected from dataset)")
    args = parser.parse_args()

    if args.data_root is None:
        if args.dataset == "yuquan":
            args.data_root = str(PROJECT_ROOT / "results" / "hfo_detection")
        else:
            args.data_root = str(PROJECT_ROOT / "results" / "hfo_detection")

    if args.soz_json is None:
        if args.dataset == "yuquan":
            args.soz_json = str(PROJECT_ROOT / "results" / "yuquan_soz_core_channels.json")
        else:
            args.soz_json = str(PROJECT_ROOT / "results" / "epilepsiae_soz_core_channels.json")

    if args.out_dir is None:
        args.out_dir = str(PROJECT_ROOT / "results" / "refine_soz_validation")

    run(
        data_root=Path(args.data_root),
        soz_json=Path(args.soz_json),
        params_json=Path(args.params_json),
        out_dir=Path(args.out_dir),
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
