#!/usr/bin/env python3
"""Interictal propagation PR-1 visualization.

Generates two figures:
  1. pr1_propagation_heatmap_examples.png — Figure-2-style lagPatRank heatmaps
     for 3 representative subjects (original + clustered order + per-channel
     rank histograms).
  2. pr1_propagation_cohort_summary.png — 6-panel cohort statistics covering
     mixture screen, cluster-aware stereotypy, legacy MI, bias fraction,
     n_participating stratification, and SOZ comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

RESULTS_DIR = Path("results/interictal_propagation")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_lagpat(subject_dir: Path) -> Dict[str, Any]:
    """Minimal loader for heatmap plotting (no alignment needed for single-subject viz)."""
    from src.interictal_propagation import load_subject_propagation_patterns
    return load_subject_propagation_patterns(subject_dir)


def _pick_representatives(subjects: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick 3 subjects: highest within-cluster tau, most negative inter-cluster corr, median."""
    valid = [
        rec for rec in subjects.values()
        if isinstance(rec, dict)
        and "error" not in rec
        and rec.get("cluster")
        and "error" not in rec.get("cluster", {})
    ]
    if len(valid) < 3:
        return valid[:3]
    valid = sorted(valid, key=lambda r: r["cluster"].get("within_cluster_tau_mean", 0), reverse=True)
    best = valid[0]
    by_corr = sorted(valid, key=lambda r: r["cluster"].get("inter_cluster_corr", 0))
    most_anti = by_corr[0]
    mid_idx = len(valid) // 2
    median_sub = valid[mid_idx] if valid[mid_idx] not in (best, most_anti) else valid[mid_idx - 1]
    seen = set()
    result = []
    for cand in [best, most_anti, median_sub]:
        key = (cand["dataset"], cand["subject"])
        if key not in seen:
            seen.add(key)
            result.append(cand)
    while len(result) < 3 and valid:
        cand = valid.pop()
        key = (cand["dataset"], cand["subject"])
        if key not in seen:
            seen.add(key)
            result.append(cand)
    return result[:3]


def plot_heatmap_examples(subjects: Dict[str, Dict[str, Any]]) -> None:
    """Figure-2-style heatmaps for representative subjects."""
    reps = _pick_representatives(subjects)
    if not reps:
        return

    n_reps = len(reps)
    fig = plt.figure(figsize=(18, 4.5 * n_reps))
    outer = gridspec.GridSpec(n_reps, 1, hspace=0.35)

    for row_idx, rec in enumerate(reps):
        ds = rec["dataset"]
        sub = rec["subject"]
        if ds == "yuquan":
            subject_dir = YUQUAN_ROOT / sub
        else:
            subject_dir = EPILEPSIAE_ROOT / sub / "all_recs"

        try:
            loaded = _load_lagpat(subject_dir)
        except Exception:
            continue
        ranks = loaded["ranks"]
        bools = loaded["bools"]
        ch_names = loaded["channel_names"]
        n_ch, n_ev = ranks.shape
        if n_ev == 0:
            continue

        cluster_info = rec.get("cluster", {})
        labels = np.array(cluster_info.get("labels", []))

        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row_idx],
                                                 width_ratios=[2, 2, 1.2], wspace=0.15)

        valid_events = np.where(np.sum(bools > 0, axis=0) >= 3)[0]
        display_ranks = ranks[:, valid_events] if valid_events.size > 0 else ranks

        ax_ori = fig.add_subplot(inner[0])
        im = ax_ori.pcolormesh(display_ranks, rasterized=True, cmap="viridis")
        ax_ori.set_yticks(np.arange(n_ch) + 0.5)
        ax_ori.set_yticklabels(ch_names, fontsize=6)
        ax_ori.set_xlabel("Pop Events (time order)", fontsize=9)
        n_ev_disp = display_ranks.shape[1]
        clust_info_str = ""
        if cluster_info.get("clusters"):
            c0 = cluster_info["clusters"][0]
            c1 = cluster_info["clusters"][1] if len(cluster_info["clusters"]) > 1 else {}
            clust_info_str = (
                f"  |  within-τ: {c0.get('raw_tau', 0):.3f} / {c1.get('raw_tau', 0):.3f}"
                f"  |  inter-corr: {cluster_info.get('inter_cluster_corr', 0):.2f}"
            )
        mi_info = rec.get("legacy_mi", {})
        mi_str = f"  |  MI={mi_info.get('mi_mean', 0):.3f}, p={mi_info.get('p_value', 1):.3f}" if mi_info else ""
        ax_ori.set_title(
            f"{ds}:{sub}  (n={n_ev_disp}, τ={rec['propagation_stereotypy']['all']['mean_tau']:.3f}){mi_str}",
            fontsize=9,
        )

        ax_clust = fig.add_subplot(inner[1])
        if labels.size == valid_events.size and labels.size > 0:
            order = np.argsort(labels)
            clustered_ranks = display_ranks[:, order]
            clustered_labels = labels[order]
            ax_clust.pcolormesh(clustered_ranks, rasterized=True, cmap="viridis")
            boundary = int(np.sum(clustered_labels == 0))
            ax_clust.axvline(boundary, color="red", lw=1.5, ls="--")
            ax_clust.text(boundary / 2, n_ch + 0.3,
                          f"C0 (n={boundary})", ha="center", fontsize=7, color="red")
            ax_clust.text((boundary + n_ev_disp) / 2, n_ch + 0.3,
                          f"C1 (n={n_ev_disp - boundary})", ha="center", fontsize=7, color="red")
        else:
            ax_clust.pcolormesh(display_ranks, rasterized=True, cmap="viridis")
        ax_clust.set_yticks([])
        ax_clust.set_xlabel("Pop Events (clustered)", fontsize=9)
        ax_clust.set_title(f"KMeans k=2{clust_info_str}", fontsize=9)

        ax_hist = fig.add_subplot(inner[2])
        overlap = 0.85
        for ci in range(n_ch):
            vals = ranks[ci, bools[ci] > 0] if bools.size else np.array([])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            hist, _ = np.histogram(vals, bins=np.arange(0, n_ch + 1) - 0.5)
            hist_norm = hist / max(1, vals.size)
            y_base = ci * (1.0 - overlap)
            ax_hist.bar(np.arange(n_ch), hist_norm, bottom=y_base, width=1.0,
                        color=plt.cm.viridis(ci / max(1, n_ch - 1)),
                        alpha=0.75, linewidth=0, rasterized=True)
            ax_hist.axhline(y_base, color="0.8", lw=0.3)
        ax_hist.set_yticks(np.arange(n_ch) * (1.0 - overlap))
        ax_hist.set_yticklabels(ch_names, fontsize=6)
        ax_hist.set_xlabel("Rank", fontsize=9)
        ax_hist.set_title("Per-channel rank distribution", fontsize=9)
        ax_hist.set_xlim(-0.5, n_ch - 0.5)

    fig.suptitle("PR-1: Propagation Pattern Heatmaps (Figure 2 style)", fontsize=13, y=0.99)
    out = FIG_DIR / "pr1_propagation_heatmap_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_cohort_summary(subjects: Dict[str, Dict[str, Any]], cohort: Dict[str, Any]) -> None:
    """6-panel cohort summary."""
    valid = [
        rec for rec in subjects.values()
        if isinstance(rec, dict) and "error" not in rec and rec.get("propagation_stereotypy")
    ]
    if not valid:
        return
    valid = sorted(valid, key=lambda r: (r["dataset"], r["subject"]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel A: Cluster-aware stereotypy (the key panel)
    ax = axes[0, 0]
    overall_vals = []
    within_vals = []
    for rec in valid:
        cl = rec.get("cluster", {})
        ov = cl.get("overall_tau", np.nan)
        wi = cl.get("within_cluster_tau_mean", np.nan)
        if np.isfinite(ov) and np.isfinite(wi):
            overall_vals.append(ov)
            within_vals.append(wi)
    if overall_vals:
        ov_arr = np.array(overall_vals)
        wi_arr = np.array(within_vals)
        colors_a = ["#2166AC" if rec["dataset"] == "yuquan" else "#E08214"
                     for rec in valid
                     if rec.get("cluster") and np.isfinite(rec["cluster"].get("overall_tau", np.nan))
                     and np.isfinite(rec["cluster"].get("within_cluster_tau_mean", np.nan))]
        ax.scatter(ov_arr, wi_arr, c=colors_a[:len(ov_arr)], s=55, edgecolors="white", linewidths=0.6, zorder=2)
        lim = max(np.max(ov_arr), np.max(wi_arr)) + 0.03
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-0.01, lim)
        ax.set_ylim(-0.01, lim)
    ca = cohort.get("cluster_analysis", {})
    ax.set_xlabel("Overall mean τ (all events)", fontsize=10)
    ax.set_ylabel("Within-cluster mean τ (k=2)", fontsize=10)
    ax.set_title(
        f"A: Cluster-aware stereotypy\n"
        f"within τ median={ca.get('within_cluster_tau_median', np.nan):.3f}, "
        f"uplift median={ca.get('uplift_median', np.nan):.3f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)

    # Panel B: Inter-cluster correlation distribution
    ax = axes[0, 1]
    inter_corrs = [
        rec["cluster"]["inter_cluster_corr"]
        for rec in valid
        if rec.get("cluster") and np.isfinite(rec["cluster"].get("inter_cluster_corr", np.nan))
    ]
    if inter_corrs:
        ax.hist(inter_corrs, bins=15, color="#4393C3", edgecolor="white", alpha=0.85)
        ax.axvline(-0.5, color="red", ls="--", lw=1, alpha=0.6, label="r < −0.5 (forward/reverse)")
        ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.4)
        ax.legend(fontsize=8)
    n_anti = ca.get("n_anticorrelated", 0)
    ax.set_xlabel("Inter-cluster Spearman r", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"B: Inter-cluster pattern correlation\n"
        f"median={ca.get('inter_cluster_corr_median', np.nan):.2f}, "
        f"n(r<−0.5)={n_anti} (forward/reverse)",
        fontsize=10,
    )

    # Panel C: Legacy MI
    ax = axes[0, 2]
    mi_means = []
    mi_colors = []
    for rec in valid:
        mi = rec.get("legacy_mi", {})
        m = mi.get("mi_mean", np.nan)
        if np.isfinite(m):
            mi_means.append(m)
            mi_colors.append("#B2182B" if mi.get("significant", False) else "#4393C3")
    if mi_means:
        x_mi = np.arange(len(mi_means))
        ax.bar(x_mi, mi_means, color=mi_colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.5)
    lmi = cohort.get("legacy_mi", {})
    ax.set_xlabel("Subject", fontsize=10)
    ax.set_ylabel("Mean MI", fontsize=10)
    ax.set_title(
        f"C: Legacy MI (red=significant)\n"
        f"significant: {lmi.get('n_significant', 0)}/{lmi.get('n_tested', 0)}, "
        f"median MI={lmi.get('mi_mean_median', np.nan):.3f}",
        fontsize=10,
    )
    ax.set_xticks(np.arange(len(mi_means)))
    ax.set_xticklabels(
        [f"{r['dataset'][:3]}:{r['subject']}" for r in valid if r.get("legacy_mi") and np.isfinite(r["legacy_mi"].get("mi_mean", np.nan))],
        rotation=70, ha="right", fontsize=6,
    )

    # Panel D: Raw vs Centered tau (identity bias)
    ax = axes[1, 0]
    raw_list = []
    cen_list = []
    sc_colors = []
    for rec in valid:
        c = rec["centered_rank"]
        if np.isfinite(c.get("raw_tau", np.nan)) and np.isfinite(c.get("centered_tau", np.nan)):
            raw_list.append(float(c["raw_tau"]))
            cen_list.append(float(c["centered_tau"]))
            sc_colors.append("#B2182B" if rec["source_diagnostic"].get("soz_source_erased", False) else "#2166AC")
    if raw_list:
        r_arr = np.array(raw_list)
        c_arr = np.array(cen_list)
        lo = min(np.min(r_arr), np.min(c_arr)) - 0.02
        hi = max(np.max(r_arr), np.max(c_arr)) + 0.02
        ax.scatter(r_arr, c_arr, c=sc_colors, s=50, edgecolors="white", linewidths=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Raw mean τ", fontsize=10)
    ax.set_ylabel("Centered mean τ", fontsize=10)
    ax.set_title(
        f"D: Identity-bias diagnosis\n"
        f"bias fraction median={cohort.get('bias_fraction_median', np.nan):.3f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)

    # Panel E: tau by n_participating
    ax = axes[1, 1]
    labels_e = [b["bin_label"] for b in valid[0]["by_nparticipating"]]
    for rec in valid:
        vals = [b.get("mean_tau", np.nan) for b in rec["by_nparticipating"]]
        ax.plot(labels_e, vals, color="0.8", lw=0.8, alpha=0.7)
    median_curve = []
    for idx in range(len(labels_e)):
        vals = [
            rec["by_nparticipating"][idx].get("mean_tau", np.nan)
            for rec in valid if idx < len(rec["by_nparticipating"])
        ]
        vals = [v for v in vals if np.isfinite(v)]
        median_curve.append(float(np.median(vals)) if vals else np.nan)
    ax.plot(labels_e, median_curve, color="black", lw=2.2, marker="o")
    ax.set_ylabel("Mean τ", fontsize=10)
    ax.set_title("E: Stereotypy by n_participating", fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    # Panel F: SOZ vs nonSOZ
    ax = axes[1, 2]
    x_vals = []
    y_vals = []
    xe = [[], []]
    ye = [[], []]
    pt_colors = []
    for rec in valid:
        prop = rec["propagation_stereotypy"]
        soz = prop["soz"]
        nonsoz = prop["nonsoz"]
        if not (np.isfinite(soz.get("mean_tau", np.nan)) and np.isfinite(nonsoz.get("mean_tau", np.nan))):
            continue
        x_vals.append(float(nonsoz["mean_tau"]))
        y_vals.append(float(soz["mean_tau"]))
        xe[0].append(max(0.0, float(nonsoz["mean_tau"] - nonsoz["tau_ci_lo"])))
        xe[1].append(max(0.0, float(nonsoz["tau_ci_hi"] - nonsoz["mean_tau"])))
        ye[0].append(max(0.0, float(soz["mean_tau"] - soz["tau_ci_lo"])))
        ye[1].append(max(0.0, float(soz["tau_ci_hi"] - soz["mean_tau"])))
        pt_colors.append("#2166AC" if rec["dataset"] == "yuquan" else "#E08214")
    if x_vals:
        x_a = np.array(x_vals)
        y_a = np.array(y_vals)
        ax.errorbar(x_a, y_a, xerr=np.array(xe), yerr=np.array(ye),
                     fmt="none", ecolor="0.7", elinewidth=0.8, alpha=0.8, zorder=1)
        ax.scatter(x_a, y_a, c=pt_colors, s=55, edgecolors="white", linewidths=0.6, zorder=2)
        lo = min(np.min(x_a), np.min(y_a)) - 0.02
        hi = max(np.max(x_a), np.max(y_a)) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    svn = cohort.get("soz_vs_nonsoz", {})
    ax.set_xlabel("nonSOZ mean τ", fontsize=10)
    ax.set_ylabel("SOZ mean τ", fontsize=10)
    ax.set_title(
        f"F: SOZ vs nonSOZ\n"
        f"pairs={svn.get('n_pairs', 0)}, p={svn.get('wilcoxon_greater_p', np.nan):.3g}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)

    fig.suptitle("Interictal Group-Event Internal Propagation — PR-1 Cohort Summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "pr1_propagation_cohort_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    subjects = _load("pr1_subject_summary.json")
    cohort = _load("pr1_cohort_summary.json")
    if not subjects:
        return
    plot_heatmap_examples(subjects)
    plot_cohort_summary(subjects, cohort)


if __name__ == "__main__":
    main()
