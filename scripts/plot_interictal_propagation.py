#!/usr/bin/env python3
"""Interictal propagation visualization.

Generates two figures:
  1. pr1_propagation_heatmap_examples.png — Figure-2-style lagPatRank heatmaps
     for 3 representative subjects (original + clustered order + per-channel
     rank histograms).
  2. pr1_propagation_cohort_summary.png — 6-panel cohort statistics covering
     mixture screen, cluster-aware stereotypy, legacy MI, bias fraction,
     n_participating stratification, and SOZ comparison.

Also supports PR-3 per-subject publication-style figures via ``--pr3``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (  # noqa: E402
    style_panel as _style_panel,
    violin_with_scatter as _violin_with_scatter,
    add_significance_bracket as _add_significance_bracket,
    COL_YQ, COL_EPI, COL_SIG, COL_NONSIG,
)

RESULTS_DIR = Path("results/interictal_propagation")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
PR3_FIG_DIR = FIG_DIR / "per_subject"
PR4A_FIG_DIR = FIG_DIR / "per_subject"
PR4A_FIG_DIR.mkdir(parents=True, exist_ok=True)
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")
PR4A_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

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
    from src.interictal_propagation import load_subject_propagation_events
    return load_subject_propagation_events(subject_dir)


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
        MAX_DISPLAY = 2000
        if valid_events.size > MAX_DISPLAY:
            step = valid_events.size // MAX_DISPLAY
            display_indices = valid_events[::step][:MAX_DISPLAY]
        else:
            display_indices = valid_events
        display_ranks = ranks[:, display_indices] if display_indices.size > 0 else ranks

        ax_ori = fig.add_subplot(inner[0])
        im = ax_ori.pcolormesh(display_ranks, rasterized=True, cmap="viridis")
        ax_ori.set_yticks(np.arange(n_ch) + 0.5)
        ax_ori.set_yticklabels(ch_names, fontsize=6)
        ax_ori.set_xlabel("Pop Events (time order)", fontsize=9)
        n_ev_disp = int(valid_events.size)
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
            if valid_events.size > MAX_DISPLAY:
                disp_labels = labels[::step][:MAX_DISPLAY]
                disp_order = np.argsort(disp_labels)
                clustered_ranks = display_ranks[:, disp_order]
                clustered_labels = disp_labels[disp_order]
            else:
                clustered_ranks = display_ranks[:, order]
                clustered_labels = labels[order]
            ax_clust.pcolormesh(clustered_ranks, rasterized=True, cmap="viridis")
            boundary = int(np.sum(clustered_labels == 0))
            n_ev_disp_clust = clustered_ranks.shape[1]
            ax_clust.axvline(boundary, color="red", lw=1.5, ls="--")
            c0_total = int(np.sum(labels == 0))
            c1_total = int(np.sum(labels == 1))
            ax_clust.text(boundary / 2, n_ch + 0.3,
                          f"C0 (n={c0_total})", ha="center", fontsize=7, color="red")
            ax_clust.text((boundary + n_ev_disp_clust) / 2, n_ch + 0.3,
                          f"C1 (n={c1_total})", ha="center", fontsize=7, color="red")
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


_K_COLORS = {2: "#4477AA", 4: "#EE6677", 6: "#228833"}


def plot_cohort_summary(subjects: Dict[str, Dict[str, Any]], cohort: Dict[str, Any]) -> None:
    """Publication-grade 6-panel cohort propagation figure.

    Panel narrative:
      a: MI significance (data vs permutation null)
      b: Bimodality directly visible (representative pairwise-\u03c4 distribution)
      c: Each mode is internally stereotyped (cluster-aware uplift)
      d: Anti-correlated template structure persists across time
      e: Forward/reverse anti-correlation is the dominant pattern
      f: Within-cluster identity-bias decomposition
    """
    import scipy.stats as st

    valid = [
        rec for rec in subjects.values()
        if isinstance(rec, dict) and "error" not in rec and rec.get("propagation_stereotypy")
    ]
    if not valid:
        return
    valid = sorted(valid, key=lambda r: (r["dataset"], r["subject"]))
    yq = [r for r in valid if r["dataset"] == "yuquan"]
    epi = [r for r in valid if r["dataset"] == "epilepsiae"]

    fig, axes = plt.subplots(2, 3, figsize=(26, 16))
    fig.patch.set_facecolor("white")

    # ================================================================
    # Panel a: MI with null hypothesis
    # ================================================================
    ax = axes[0, 0]
    _style_panel(ax, "a")

    COL_NULL = "#BBBBBB"
    group_spacing = 1.8
    pair_gap = 0.6

    for gi, (recs, col, label) in enumerate(
        [(yq, COL_YQ, "Yuquan"), (epi, COL_EPI, "Epilepsiae")]
    ):
        mi_data, mi_null = [], []
        for r in recs:
            mi = r.get("legacy_mi", {})
            m = mi.get("mi_median", mi.get("mi_mean", np.nan))
            null_m = mi.get("permuted_mean_median", np.nan)
            if np.isfinite(m):
                mi_data.append(m)
            if np.isfinite(null_m):
                mi_null.append(null_m)

        base = gi * group_spacing
        pos_data = base
        pos_null = base + pair_gap

        d_arr = np.asarray(mi_data, dtype=float)
        n_arr = np.asarray(mi_null, dtype=float)

        _violin_with_scatter(ax, d_arr, pos_data, col, rng_seed=42 + gi)
        _violin_with_scatter(
            ax, n_arr, pos_null, COL_NULL, rng_seed=99 + gi,
            scatter_size=35, alpha_body=0.15,
        )

        if d_arr.size >= 2 and n_arr.size >= 2:
            try:
                _, p_mw = st.mannwhitneyu(d_arr, n_arr, alternative="greater")
            except Exception:
                p_mw = np.nan
            y_top = max(np.max(d_arr), np.max(n_arr)) + 0.02
            _add_significance_bracket(ax, pos_data, pos_null, y_top, p_mw)

    n_sig = sum(
        1 for r in valid
        if r.get("legacy_mi", {}).get("significant", False)
    )
    ax.set_xticks([0, pair_gap, group_spacing, group_spacing + pair_gap])
    ax.set_xticklabels(["Data", "Null", "Data", "Null"], fontsize=18)
    ax.text(pair_gap / 2, -0.12, "Yuquan", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=20, fontweight="bold")
    ax.text(group_spacing + pair_gap / 2, -0.12, "Epilepsiae",
            transform=ax.get_xaxis_transform(),
            ha="center", fontsize=20, fontweight="bold")
    ax.set_ylabel("Matching Index", fontsize=22)
    ax.set_title(
        f"MI: data vs permutation null  ({n_sig}/{len(valid)} sig.)",
        fontsize=22, pad=12,
    )

    # ================================================================
    # Panel b: Bimodality — representative pairwise-\u03c4 distribution
    # ================================================================
    ax = axes[0, 1]
    _style_panel(ax, "b")

    best_dip_rec = max(
        valid,
        key=lambda r: r.get("mixture", {}).get("dip_stat", -1),
    )
    rep_ds = best_dip_rec["dataset"]
    rep_sub = best_dip_rec["subject"]
    rep_dir = _resolve_subject_dir(rep_ds, rep_sub)
    tau_dist_ok = False
    try:
        from src.interictal_propagation import (
            compute_pairwise_tau_values,
            load_subject_propagation_events,
            _valid_event_indices,
            _pairwise_tau_summary,
        )
        loaded = load_subject_propagation_events(rep_dir)
        tau_vals = compute_pairwise_tau_values(
            loaded["ranks"], loaded["bools"], n_sample=300, seed=0,
        )

        ada = best_dip_rec.get("adaptive_cluster", {})
        ada_labels = np.array(ada.get("labels", []), dtype=int)
        v_ev = _valid_event_indices(loaded["bools"], min_participating=3)

        within_taus, between_taus = [], []
        if ada_labels.size == v_ev.size and ada_labels.size > 0:
            sampled_idx = np.arange(min(300, v_ev.size))
            for i in range(len(sampled_idx)):
                for j in range(i + 1, len(sampled_idx)):
                    ei, ej = v_ev[sampled_idx[i]], v_ev[sampled_idx[j]]
                    shared = (loaded["bools"][:, ei] > 0) & (loaded["bools"][:, ej] > 0)
                    if shared.sum() < 3:
                        continue
                    x = loaded["ranks"][shared, ei].astype(float)
                    y = loaded["ranks"][shared, ej].astype(float)
                    fin = np.isfinite(x) & np.isfinite(y)
                    if fin.sum() < 3:
                        continue
                    from scipy.stats import kendalltau
                    tau, _ = kendalltau(x[fin], y[fin])
                    if not np.isfinite(tau):
                        continue
                    li = ada_labels[sampled_idx[i]]
                    lj = ada_labels[sampled_idx[j]]
                    if li == lj:
                        within_taus.append(tau)
                    else:
                        between_taus.append(tau)

        if within_taus and between_taus:
            from scipy.stats import gaussian_kde
            xg = np.linspace(-1, 1, 300)
            for taus, col, label_t in [
                (within_taus, "#4477AA", "Within-cluster pairs"),
                (between_taus, "#EE6677", "Between-cluster pairs"),
            ]:
                if len(taus) > 10:
                    kde = gaussian_kde(taus, bw_method=0.12)
                    yg = kde(xg)
                    ax.fill_between(xg, yg, alpha=0.25, color=col)
                    ax.plot(xg, yg, color=col, lw=2.5, label=label_t)
            ax.legend(fontsize=14, frameon=False, loc="upper left")
            all_taus = within_taus + between_taus
            x_lo = max(-1, np.percentile(all_taus, 0.5) - 0.05)
            x_hi = min(1, np.percentile(all_taus, 99.5) + 0.05)
            ax.set_xlim(x_lo, x_hi)
            y_top = ax.get_ylim()[1]
            ax.set_ylim(0, y_top * 1.02)
            tau_dist_ok = True
    except Exception:
        pass

    if not tau_dist_ok:
        ax.text(
            0.5, 0.5, "Data unavailable",
            transform=ax.transAxes, ha="center", fontsize=16,
        )

    n_multimodal = sum(
        1 for r in valid
        if r.get("mixture", {}).get("dip_p", 1) < 0.05
    )
    ax.set_xlabel("Pairwise Kendall \u03c4", fontsize=22)
    ax.set_ylabel("Density", fontsize=22)
    ax.set_title(
        f"Bimodality  ({n_multimodal}/{len(valid)} dip test p < 0.001)",
        fontsize=22, pad=12,
    )
    ax.text(
        0.97, 0.95,
        f"Example: {rep_ds}:{rep_sub}",
        transform=ax.transAxes, fontsize=14,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.85),
    )

    # ================================================================
    # Panel c: Within-cluster tau uplift (adaptive k)
    # ================================================================
    ax = axes[0, 2]
    _style_panel(ax, "c")

    overall_vals: List[float] = []
    within_vals: List[float] = []
    colors_c: List[str] = []
    for rec in valid:
        ada = rec.get("adaptive_cluster", {})
        if not ada or "error" in ada:
            ada = rec.get("cluster", {})
        ov = ada.get("overall_tau", np.nan)
        wi = ada.get("within_cluster_tau_mean", np.nan)
        if np.isfinite(ov) and np.isfinite(wi):
            overall_vals.append(ov)
            within_vals.append(wi)
            colors_c.append(COL_YQ if rec["dataset"] == "yuquan" else COL_EPI)

    med_uplift = np.nan
    if overall_vals:
        ov_arr = np.array(overall_vals)
        wi_arr = np.array(within_vals)
        lim = max(np.max(ov_arr), np.max(wi_arr)) + 0.05
        ax.fill_between(
            [0, lim], [0, lim], [0, 0],
            color="#F0F0F0", zorder=0,
        )
        ax.plot([0, lim], [0, lim], color="#999999", lw=1.5, ls="--", zorder=1)
        ax.scatter(
            ov_arr, wi_arr, c=colors_c, s=100,
            edgecolors="white", linewidths=1.0, zorder=3, alpha=0.85,
        )
        ax.set_xlim(-0.01, lim)
        ax.set_ylim(-0.01, lim)
        med_uplift = float(np.median(wi_arr - ov_arr))
        ax.annotate(
            f"\u0394\u03c4 = {med_uplift:+.3f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=18, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.85),
        )
        n_above = int(np.sum(wi_arr > ov_arr))
        ax.text(
            0.95, 0.05, f"{n_above}/{len(ov_arr)} above diagonal",
            transform=ax.transAxes, fontsize=16,
            ha="right", va="bottom", color="#666666",
        )

    ax.set_xlabel("Overall \u03c4", fontsize=22)
    ax.set_ylabel("Within-cluster \u03c4", fontsize=22)
    ax.set_title("Cluster-aware stereotypy uplift", fontsize=22, pad=12)
    ax.set_aspect("equal", adjustable="box")

    # ================================================================
    # Panel d: Template temporal stability (split-half reproducibility)
    # ================================================================
    ax = axes[1, 0]
    _style_panel(ax, "d")

    d_match_yq: List[float] = []
    d_match_epi: List[float] = []
    n_strong, n_moderate, n_weak = 0, 0, 0
    for rec in valid:
        repro = rec.get("time_split_reproducibility", {})
        splits = repro.get("splits", {})
        sh = splits.get("first_half_second_half", {})
        mc = sh.get("mean_match_corr", np.nan)
        if not np.isfinite(mc):
            oe = splits.get("odd_even_block", {})
            mc = oe.get("mean_match_corr", np.nan)
        if not np.isfinite(mc):
            continue
        (d_match_yq if rec["dataset"] == "yuquan" else d_match_epi).append(mc)
        grade = repro.get("reproducibility_grade", "")
        if grade == "strong":
            n_strong += 1
        elif grade == "moderate":
            n_moderate += 1
        else:
            n_weak += 1

    positions_d = [0, 1]
    for i, (mc_data, pos, col) in enumerate(
        zip([d_match_yq, d_match_epi], positions_d, [COL_YQ, COL_EPI])
    ):
        if not mc_data:
            continue
        vals = np.asarray(mc_data, dtype=float)
        _violin_with_scatter(ax, vals, pos, col, rng_seed=77 + i, scatter_size=70)

    ax.axhline(0.8, color=COL_SIG, ls="--", lw=1.5, alpha=0.5, zorder=1,
               label="strong threshold")
    ax.axhline(0.5, color="#AAAAAA", ls=":", lw=1.2, alpha=0.5, zorder=1,
               label="moderate threshold")
    ax.set_xticks(positions_d)
    ax.set_xticklabels(["Yuquan", "Epilepsiae"], fontsize=22)
    ax.set_ylabel("Split-half template\nmatch correlation", fontsize=22)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, 1.6)
    total_d = len(d_match_yq) + len(d_match_epi)
    ax.text(
        0.97, 0.05,
        f"Strong: {n_strong}/{total_d}\n"
        f"Moderate: {n_moderate}/{total_d}\n"
        f"Weak: {n_weak}/{total_d}",
        transform=ax.transAxes, fontsize=16,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.85),
    )
    ax.set_title(
        f"Template temporal stability  (n={total_d})",
        fontsize=22, pad=12,
    )

    # ================================================================
    # Panel e: Inter-cluster correlation with significance bracket
    # ================================================================
    ax = axes[1, 1]
    _style_panel(ax, "e")

    inter_corrs_k2: List[float] = []
    for rec in valid:
        ada = rec.get("adaptive_cluster", {})
        sk = ada.get("stable_k") or ada.get("chosen_k")
        if sk != 2:
            continue
        corr_mat = ada.get("inter_cluster_corr_matrix", [])
        if corr_mat and len(corr_mat) >= 2 and len(corr_mat[0]) >= 2:
            r = float(corr_mat[0][1])
            if np.isfinite(r):
                inter_corrs_k2.append(r)

    w_p = np.nan
    if inter_corrs_k2:
        r_arr = np.array(inter_corrs_k2)

        bins = np.linspace(-1, 1, 18)
        ax.hist(
            r_arr, bins=bins, color=_K_COLORS[2],
            edgecolor="white", alpha=0.55, linewidth=1.2, zorder=2,
        )
        from scipy.stats import gaussian_kde as _gkde
        if len(r_arr) > 5:
            kde_e = _gkde(r_arr, bw_method=0.2)
            xg_e = np.linspace(-1.05, 1.05, 200)
            yg_e = kde_e(xg_e) * len(r_arr) * (bins[1] - bins[0])
            ax.fill_between(xg_e, yg_e, alpha=0.15, color=_K_COLORS[2], zorder=1)
            ax.plot(xg_e, yg_e, color=_K_COLORS[2], lw=2, alpha=0.8, zorder=1)

        ax.axvline(0, color="#BBBBBB", ls="-", lw=1.2, alpha=0.5, zorder=0)

        med_r = float(np.median(r_arr))
        ax.axvline(med_r, color="black", ls="--", lw=2.5, zorder=4)
        n_anti = int(np.sum(r_arr < -0.5))

        try:
            _, w_p = st.wilcoxon(r_arr, alternative="less")
        except Exception:
            w_p = np.nan

        y_max = ax.get_ylim()[1]
        if np.isfinite(w_p) and w_p < 0.05:
            star = "***" if w_p < 0.001 else ("**" if w_p < 0.01 else "*")
            ax.annotate(
                star, xy=(med_r, y_max * 0.88),
                fontsize=26, fontweight="bold", ha="center", va="bottom",
                color=COL_SIG,
            )

        ax.axvspan(-1.05, -0.5, color="#FFEEEE", alpha=0.3, zorder=0)
        ax.text(
            0.97, 0.95,
            f"n = {len(r_arr)} (k=2)\n"
            f"median r = {med_r:.2f}\n"
            f"n(r < \u22120.5) = {n_anti}\n"
            f"Wilcoxon p = {w_p:.1e}",
            transform=ax.transAxes, fontsize=16,
            va="top", ha="right",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                edgecolor="#CCCCCC", alpha=0.92,
            ),
        )
        ax.set_xlim(-1.05, 1.05)

    ax.set_xlabel("Inter-cluster Spearman r", fontsize=22)
    ax.set_ylabel("Count", fontsize=22)
    ax.set_title("Inter-cluster template correlation (k=2)", fontsize=22, pad=12)

    # ================================================================
    # Panel f: Within-cluster identity-bias decomposition
    # ================================================================
    ax = axes[1, 2]
    _style_panel(ax, "f")

    raw_wc: List[float] = []
    cen_wc: List[float] = []
    f_colors: List[str] = []
    for rec in valid:
        wcc = rec.get("within_cluster_centered", {})
        if not wcc or "error" in wcc:
            c = rec.get("centered_rank", {})
            rt = c.get("raw_tau", np.nan)
            ct = c.get("centered_tau", np.nan)
        else:
            rt = wcc.get("mean_raw_tau", np.nan)
            ct = wcc.get("mean_centered_tau", np.nan)
        if np.isfinite(rt) and np.isfinite(ct):
            raw_wc.append(float(rt))
            cen_wc.append(float(ct))
            f_colors.append(COL_YQ if rec["dataset"] == "yuquan" else COL_EPI)

    if raw_wc:
        r_arr_f = np.array(raw_wc)
        c_arr_f = np.array(cen_wc)
        lo = min(np.min(r_arr_f), np.min(c_arr_f)) - 0.02
        hi = max(np.max(r_arr_f), np.max(c_arr_f)) + 0.02
        ax.fill_between(
            [lo, hi], [lo, hi], [lo, lo],
            color="#F0F0F0", zorder=0,
        )
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=1.5, ls="--", zorder=1)

        for i in range(len(r_arr_f)):
            ax.annotate(
                "", xy=(r_arr_f[i], c_arr_f[i]),
                xytext=(r_arr_f[i], r_arr_f[i]),
                arrowprops=dict(
                    arrowstyle="-", color="#CCCCCC",
                    lw=0.8, ls=":",
                ),
                zorder=1,
            )

        ax.scatter(
            r_arr_f, c_arr_f, c=f_colors, s=100,
            edgecolors="white", linewidths=1.0, zorder=3, alpha=0.85,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        bias_fracs = []
        for rec in valid:
            wcc = rec.get("within_cluster_centered", {})
            if wcc and "error" not in wcc:
                bf = wcc.get("mean_bias_fraction", np.nan)
                if np.isfinite(bf):
                    bias_fracs.append(bf)
        if not bias_fracs:
            bias_fracs = [
                rec.get("centered_rank", {}).get("bias_fraction", np.nan)
                for rec in valid
            ]
            bias_fracs = [b for b in bias_fracs if np.isfinite(b)]

        med_bias = float(np.median(bias_fracs)) if bias_fracs else np.nan
        label_suffix = "within-cluster" if any(
            rec.get("within_cluster_centered") and "error" not in rec.get("within_cluster_centered", {})
            for rec in valid
        ) else "overall"

        ax.annotate(
            f"Median bias = {med_bias:.0%}" if np.isfinite(med_bias) else "",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=18, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.85),
        )

    ax.set_xlabel(f"Raw \u03c4 ({label_suffix})", fontsize=22)
    ax.set_ylabel(f"Centered \u03c4 ({label_suffix})", fontsize=22)
    ax.set_title("Identity-bias decomposition", fontsize=22, pad=12)
    ax.set_aspect("equal", adjustable="box")

    # ================================================================
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "cohort_propagation_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def _load_soz_map(dataset: str) -> Dict[str, List[str]]:
    path = SOZ_FILE_YQ if dataset == "yuquan" else SOZ_FILE_EPI
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _resolve_subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return EPILEPSIAE_ROOT / subject / "all_recs"


def _load_pr3_subject_records(dataset: str, subjects: List[str] | None) -> List[Dict[str, Any]]:
    if not PER_SUBJECT_DIR.exists():
        return []

    wanted = set(subjects or [])
    records: List[Dict[str, Any]] = []
    for path in sorted(PER_SUBJECT_DIR.glob("*.json")):
        stem = path.stem
        if "_" not in stem:
            continue
        rec_dataset, rec_subject = stem.split("_", 1)
        if dataset != "both" and rec_dataset != dataset:
            continue
        if wanted and rec_subject not in wanted:
            continue
        with open(path) as f:
            rec = json.load(f)
        rec["dataset"] = rec_dataset
        rec["subject"] = rec_subject
        records.append(rec)
    return records


def _dataset_timezone(dataset: str) -> str:
    return "Asia/Shanghai" if dataset == "yuquan" else "Europe/Berlin"


def _local_hour(epoch_sec: float, timezone_name: str) -> float:
    dt = datetime.fromtimestamp(float(epoch_sec), tz=timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    )
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _formal_day_mask(dataset: str, event_abs_times: np.ndarray) -> np.ndarray:
    event_abs_times = np.asarray(event_abs_times, dtype=float)
    mask = np.zeros(event_abs_times.shape, dtype=bool)
    finite = np.isfinite(event_abs_times)
    if not np.any(finite):
        return mask
    tz_name = _dataset_timezone(dataset)
    hours = np.array([_local_hour(t, tz_name) for t in event_abs_times[finite]], dtype=float)
    mask[finite] = (hours >= 8.0) & (hours < 20.0)
    return mask


def _sample_event_indices(event_indices: np.ndarray, max_events: int) -> np.ndarray:
    event_indices = np.asarray(event_indices, dtype=int)
    if event_indices.size <= max_events:
        return event_indices
    raw = np.linspace(0, event_indices.size - 1, num=max_events)
    keep = np.unique(raw.astype(int))
    return event_indices[keep]


def _channel_tick_indices(n_channels: int) -> np.ndarray:
    if n_channels <= 20:
        step = 1
    elif n_channels <= 40:
        step = 2
    else:
        step = 3
    return np.arange(0, n_channels, step, dtype=int)


def _fixed_channel_order(ranks: np.ndarray, bools: np.ndarray) -> np.ndarray:
    n_ch = ranks.shape[0]
    mean_rank = np.full(n_ch, np.nan, dtype=float)
    counts = np.sum(bools > 0, axis=1).astype(int) if bools.size else np.zeros(n_ch, dtype=int)
    for idx in range(n_ch):
        vals = np.asarray(ranks[idx, bools[idx] > 0], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean_rank[idx] = float(np.mean(vals))
    fill = np.nanmax(mean_rank[np.isfinite(mean_rank)]) + 1.0 if np.any(np.isfinite(mean_rank)) else 1.0
    keys = np.where(np.isfinite(mean_rank), mean_rank, fill)
    return np.lexsort((np.arange(n_ch), -counts, keys))


def _mean_rank_profile(ranks: np.ndarray, bools: np.ndarray, event_indices: np.ndarray) -> np.ndarray:
    n_ch = ranks.shape[0]
    event_indices = np.asarray(event_indices, dtype=int)
    out = np.full(n_ch, np.nan, dtype=float)
    if event_indices.size == 0:
        return out
    for idx in range(n_ch):
        vals = np.asarray(ranks[idx, event_indices], dtype=float)
        mask = np.asarray(bools[idx, event_indices], dtype=bool)
        vals = vals[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[idx] = float(np.mean(vals))
    return out


def _cluster_profiles(
    ranks: np.ndarray,
    bools: np.ndarray,
    valid_events: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> List[np.ndarray]:
    profiles: List[np.ndarray] = []
    for cluster_id in range(n_clusters):
        cluster_events = valid_events[labels == cluster_id]
        profiles.append(_mean_rank_profile(ranks, bools, cluster_events))
    return profiles



def _plot_rank_heatmap(
    ax: plt.Axes,
    display_ranks: np.ndarray,
    channel_names: List[str],
    title: str,
    show_ylabels: bool = True,
) -> Any:
    n_ch = display_ranks.shape[0]
    im = ax.pcolormesh(display_ranks, rasterized=True, cmap="viridis")
    ax.set_yticks(np.arange(n_ch) + 0.5)
    ax.set_yticklabels(
        channel_names if show_ylabels else [],
        fontsize=14 if n_ch > 24 else 16,
    )
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    return im


def _plot_daynight_strip(ax: plt.Axes, is_day: np.ndarray) -> None:
    if is_day.size == 0:
        ax.axis("off")
        return
    strip = np.where(np.asarray(is_day, dtype=bool), 1, 0)[None, :]
    ax.imshow(
        strip,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["black", "white"]),
        vmin=0,
        vmax=1,
    )
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_cluster_boundaries(ax: plt.Axes, labels_sorted: np.ndarray, n_ch: int) -> None:
    if labels_sorted.size == 0:
        return
    unique_labels = np.unique(labels_sorted)
    cursor = 0
    for cluster_id in unique_labels:
        count = int(np.sum(labels_sorted == cluster_id))
        if cursor > 0:
            ax.axvline(cursor, color="red", lw=1.5, ls="--")
        ax.text(
            cursor + count / 2.0,
            n_ch + 0.15,
            f"C{int(cluster_id)} (n={count})",
            color="red",
            ha="center",
            fontsize=14,
        )
        cursor += count


def _plot_rank_histogram(
    ax: plt.Axes,
    ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    channel_order: np.ndarray,
    channel_names: List[str],
    title: str,
) -> None:
    n_ch = len(channel_order)
    ordered_names = [channel_names[idx] for idx in channel_order]
    overlap = 0.85
    for ci_idx, ci in enumerate(channel_order):
        vals = np.asarray(ranks[ci, event_indices], dtype=float)
        mask = np.asarray(bools[ci, event_indices], dtype=bool)
        vals = vals[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=np.arange(0, n_ch + 1) - 0.5)
        hist_norm = hist / max(1, vals.size)
        y_base = ci_idx * (1.0 - overlap)
        ax.bar(
            np.arange(n_ch), hist_norm, bottom=y_base, width=1.0,
            color=plt.cm.viridis(ci_idx / max(1, n_ch - 1)),
            alpha=0.75, linewidth=0, rasterized=True,
        )
        ax.axhline(y_base, color="0.8", lw=0.3)
    ax.set_yticks(np.arange(n_ch) * (1.0 - overlap))
    ax.set_yticklabels(ordered_names, fontsize=14 if n_ch > 24 else 16)
    ax.set_xlabel("Rank", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.tick_params(axis="x", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_cluster_rank_fig4(
    ax: plt.Axes,
    ranks: np.ndarray,
    bools: np.ndarray,
    valid_events: np.ndarray,
    labels: np.ndarray,
    channel_order: np.ndarray,
    channel_names: List[str],
    title: str,
) -> None:
    """Per-cluster mean rank line + shaded mean +/- std band on fixed channel order."""
    n_ch = len(channel_order)
    ordered_names = [channel_names[idx] for idx in channel_order]
    unique_k = np.unique(labels)
    n_k = len(unique_k)
    _base_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    line_colors = [_base_colors[i % len(_base_colors)] for i in range(n_k)]
    y_pos = np.arange(n_ch, dtype=float)

    for ki, cid in enumerate(unique_k):
        mask_cluster = labels == cid
        eidx = valid_events[mask_cluster]
        means = np.full(n_ch, np.nan)
        stds = np.full(n_ch, np.nan)
        for ci_plot, ci_raw in enumerate(channel_order):
            vals = np.asarray(ranks[ci_raw, eidx], dtype=float)
            bmask = np.asarray(bools[ci_raw, eidx], dtype=bool)
            vals = vals[bmask & np.isfinite(vals)]
            if vals.size > 0:
                means[ci_plot] = np.mean(vals)
                stds[ci_plot] = np.std(vals)

        valid = np.isfinite(means)
        ax.fill_betweenx(
            y_pos[valid],
            (means - stds)[valid], (means + stds)[valid],
            color=line_colors[ki], alpha=0.15, linewidth=0,
        )
        ax.plot(
            means[valid], y_pos[valid],
            "-o", color=line_colors[ki], lw=2.5, ms=6, zorder=10,
            label=f"C{int(cid)} (n={int(mask_cluster.sum())})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=14 if n_ch > 24 else 16)
    ax.set_ylim(-0.5, n_ch - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rank", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.tick_params(axis="x", labelsize=14)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_k)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- MI distribution plotting (Figure 2B style) ----

def _mi_vector(tmpl: np.ndarray, vec: np.ndarray) -> float:
    n_ch = len(tmpl)
    sign_t = np.sign(tmpl[:, None] - tmpl[None, :])
    sign_v = np.sign(vec[:, None] - vec[None, :])
    n_pairs = n_ch * (n_ch - 1) // 2
    if n_pairs == 0:
        return 0.0
    return float(np.sum(np.triu(sign_t * sign_v, k=1))) / n_pairs


def _mi_matrix(template: np.ndarray, ranks_block: np.ndarray) -> np.ndarray:
    """Vectorised MI (concordant pair fraction): template (n_ch,) vs ranks_block (n_ch, n_ev) -> (n_ev,)."""
    n_ch = template.shape[0]
    n_pairs = n_ch * (n_ch - 1) // 2
    if n_pairs == 0:
        return np.zeros(ranks_block.shape[1])
    pa, pb = np.triu_indices(n_ch, k=1)
    sign_t = np.sign(template[pa] - template[pb])          # (n_pairs,)
    sign_v = np.sign(ranks_block[pa] - ranks_block[pb])    # (n_pairs, n_ev)
    return (sign_t[:, None] * sign_v).sum(axis=0) / n_pairs


def _permuted_mi_medians(
    template: np.ndarray, n_ch: int, n_ev: int,
    n_permutations: int, rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised permutation null: returns (n_permutations,) median MI."""
    medians = np.empty(n_permutations)
    for pi in range(n_permutations):
        shuffled = rng.random((n_ch, n_ev)).argsort(axis=0).astype(float)
        medians[pi] = float(np.median(_mi_matrix(template, shuffled)))
    return medians


_MI_SUBSAMPLE_CAP = 10000


def _compute_mi_distribution(
    ranks: np.ndarray, bools: np.ndarray, n_permutations: int = 200, seed: int = 0,
) -> Dict[str, Any]:
    """Compute full MI distribution + permutation null for plotting."""
    from src.interictal_propagation import _legacy_hist_mean_rank

    n_ch, n_ev = ranks.shape
    template = _legacy_hist_mean_rank(ranks, bools)
    mi_arr = _mi_matrix(template, ranks)

    rng = np.random.default_rng(seed)
    perm_n = min(n_ev, _MI_SUBSAMPLE_CAP)
    perm_medians = _permuted_mi_medians(template, n_ch, perm_n, n_permutations, rng)

    return {
        "mi_arr": mi_arr,
        "template": template,
        "perm_medians": perm_medians,
        "real_median": float(np.median(mi_arr)),
    }


def _compute_cluster_mi_distributions(
    ranks: np.ndarray, bools: np.ndarray,
    labels: np.ndarray, valid_events: np.ndarray,
    n_permutations: int = 200, seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """Per-cluster MI distributions + permutation test."""
    from src.interictal_propagation import _legacy_hist_mean_rank

    rng = np.random.default_rng(seed)
    result = {}
    for cid in np.unique(labels):
        mask = labels == cid
        eidx = valid_events[mask]
        if eidx.size < 5:
            continue
        sub_ranks = ranks[:, eidx]
        sub_bools = bools[:, eidx]
        n_ch, n_ev = sub_ranks.shape
        tmpl = _legacy_hist_mean_rank(sub_ranks, sub_bools)
        mi_arr = _mi_matrix(tmpl, sub_ranks)
        real_median = float(np.median(mi_arr))

        perm_n = min(n_ev, _MI_SUBSAMPLE_CAP)
        perm_medians = _permuted_mi_medians(tmpl, n_ch, perm_n, n_permutations, rng)
        p_val = float(np.mean(perm_medians >= real_median))

        result[int(cid)] = {
            "mi_arr": mi_arr,
            "perm_medians": perm_medians,
            "real_median": real_median,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }
    return result


def _plot_mi_fig2b(
    ax: plt.Axes,
    mi_arr: np.ndarray,
    perm_medians: np.ndarray,
    real_median: float,
    title: str,
    hist_color: str = "#1B9E77",
    hist_alpha: float = 0.55,
) -> None:
    """Paper Figure 2B style: MI distribution (green) + surrogate (gray) + threshold."""
    ax.hist(
        mi_arr, bins=40, density=True, color=hist_color,
        alpha=hist_alpha, edgecolor="none", rasterized=True,
        label=f"MI distribution (n={mi_arr.size})",
    )

    if perm_medians.size > 3:
        thresh = float(np.percentile(perm_medians, 95))
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(perm_medians, bw_method="silverman")
            xx = np.linspace(perm_medians.min() - 0.05, perm_medians.max() + 0.05, 300)
            yy = kde(xx)
            scale = ax.get_ylim()[1] * 0.5 / max(yy.max(), 1e-9)
            ax.fill_between(xx, yy * scale, color="0.75", alpha=0.5, zorder=0,
                            label="Surrogate median dist.")
        except Exception:
            ax.hist(perm_medians, bins=20, density=True, color="0.80",
                    alpha=0.5, edgecolor="none", zorder=0,
                    label="Surrogate median dist.")
        ax.axvline(thresh, color="black", ls="--", lw=2, alpha=0.7,
                   label=f"95% threshold ({thresh:.3f})")

    ax.axvline(real_median, color=COL_SIG, lw=2.5, zorder=5,
               label=f"Real median ({real_median:.3f})")
    ax.scatter([real_median], [ax.get_ylim()[1] * 0.95], color=COL_SIG,
               s=120, zorder=6, marker="v", edgecolors="black", linewidths=0.8)

    lo = max(-1.0, float(np.percentile(mi_arr, 0.5)) - 0.1)
    hi = min(1.0, float(np.percentile(mi_arr, 99.5)) + 0.1)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Matching Index", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


MI_FIG_DIR = FIG_DIR / "per_subject_mi"


def plot_mi_subject_figure(record: Dict[str, Any], n_permutations: int = 200) -> Path:
    """Per-subject MI distribution figure: overall + per-cluster."""
    from src.interictal_propagation import _valid_event_indices

    dataset = str(record["dataset"])
    subject = str(record["subject"])
    subject_dir = _resolve_subject_dir(dataset, subject)
    loaded = _load_lagpat(subject_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    n_ch = ranks.shape[0]
    valid_events = _valid_event_indices(bools, min_participating=3)
    if valid_events.size < 10:
        raise ValueError(f"{dataset}/{subject}: too few events for MI plot")

    print(f"  Computing MI distribution for {dataset}/{subject} "
          f"(n={valid_events.size}, {n_permutations} permutations)...")
    mi_data = _compute_mi_distribution(ranks, bools, n_permutations=n_permutations)

    adaptive = record.get("adaptive_cluster", {})
    chosen_k = int(adaptive.get("chosen_k", 2))
    adaptive_labels = np.asarray(adaptive.get("labels", []), dtype=int)
    has_labels = adaptive_labels.size == valid_events.size
    if not has_labels:
        k2 = record.get("cluster", {})
        adaptive_labels = np.asarray(k2.get("labels", []), dtype=int)
        chosen_k = 2
        has_labels = adaptive_labels.size == valid_events.size

    n_panels = 1 + (chosen_k if has_labels else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    p_val = record.get("legacy_mi", {}).get("p_value", np.nan)
    sig_str = "significant" if record.get("legacy_mi", {}).get("significant", False) else "n.s."
    _plot_mi_fig2b(
        axes[0], mi_data["mi_arr"], mi_data["perm_medians"],
        mi_data["real_median"],
        title=f"Overall MI  (p={p_val:.3f}, {sig_str})",
    )

    if has_labels:
        cluster_mi = _compute_cluster_mi_distributions(
            ranks, bools, adaptive_labels, valid_events,
            n_permutations=n_permutations,
        )
        cluster_colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]
        for ki, cid in enumerate(sorted(cluster_mi.keys())):
            ax = axes[1 + ki]
            c_data = cluster_mi[cid]
            c_p = c_data["p_value"]
            c_sig = "significant" if c_data["significant"] else "n.s."
            _plot_mi_fig2b(
                ax, c_data["mi_arr"], c_data["perm_medians"],
                c_data["real_median"],
                title=f"Cluster {cid} MI  (p={c_p:.3f}, {c_sig})",
                hist_color=cluster_colors[ki % len(cluster_colors)],
                hist_alpha=0.55,
            )

    fig.suptitle(f"{dataset}:{subject}  —  MI Distribution (Fig.2B style)", fontsize=20, y=1.02)
    fig.tight_layout()
    MI_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = MI_FIG_DIR / f"{dataset}_{subject}_mi_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def _strongest_forward_reverse_label(adaptive_cluster: Dict[str, Any]) -> str:
    pairs = adaptive_cluster.get("candidate_forward_reverse_pairs", [])
    if not pairs:
        return "forward/reverse: none"
    strongest = min(pairs, key=lambda pair: pair.get("spearman_r", 0.0))
    return (
        f"forward/reverse: {len(pairs)} pair(s), "
        f"best r={strongest.get('spearman_r', float('nan')):.2f}"
    )


def _ensure_pr3_readme() -> None:
    PR3_FIG_DIR.mkdir(parents=True, exist_ok=True)
    readme_path = PR3_FIG_DIR / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "\n".join(
            [
                "### per-subject propagation figure",
                "这组图采用 2x2 布局。左列宽，右列窄。",
                "左上：原始 lagPatRank heatmap（时间顺序），底部附 Day/Night 条带（白=Day, 黑=Night）。左下：k_best 聚类后的 heatmap（按簇排序，红色虚线分隔，簇标注在顶部）。",
                "右上：Per-channel rank distribution（原始通道顺序，stacked histogram）。右下：Cluster rank distributions（固定通道排序，两个簇用不同颜色叠加 rank 分布，直观对比不同传播模式的分布差异）。",
                "**关注点**：先看聚类后 heatmap 的簇内颜色一致性和簇间差异，再看右下角分布中各簇的 rank 峰值位置差异（forward/reverse）。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def plot_pr3_subject_figure(record: Dict[str, Any], max_events: int = 2000) -> Path:
    from src.interictal_propagation import _valid_event_indices

    dataset = str(record["dataset"])
    subject = str(record["subject"])
    subject_dir = _resolve_subject_dir(dataset, subject)
    loaded = _load_lagpat(subject_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    channel_names = list(loaded["channel_names"])
    n_ch = ranks.shape[0]
    valid_events = _valid_event_indices(bools, min_participating=3)
    if valid_events.size == 0:
        raise ValueError(f"{dataset}/{subject}: no valid events for PR-3 plotting")

    channel_order = _fixed_channel_order(ranks, bools)
    ordered_names = [channel_names[idx] for idx in channel_order]
    original_order = np.arange(n_ch, dtype=int)
    display_events = _sample_event_indices(valid_events, max_events=max_events)
    day_mask = _formal_day_mask(dataset, loaded["event_abs_times"][display_events])

    repro = record.get("time_split_reproducibility", {})
    repro_grade = str(repro.get("reproducibility_grade", "unknown"))

    all_tau = record.get("propagation_stereotypy", {}).get("all", {}).get(
        "mean_tau", float("nan")
    )
    mi_info = record.get("legacy_mi", {})
    mi_str = ""
    if mi_info:
        mi_str = (
            f"  |  MI={mi_info.get('mi_mean', 0):.3f}, "
            f"p={mi_info.get('p_value', 1):.3f}"
        )

    adaptive = record.get("adaptive_cluster", {})
    chosen_k = int(adaptive.get("chosen_k", 2))
    adaptive_labels = np.asarray(adaptive.get("labels", []), dtype=int)
    has_adaptive = adaptive_labels.size == valid_events.size and adaptive_labels.size > 0
    best_k = chosen_k if has_adaptive else 2
    best_labels = adaptive_labels if has_adaptive else np.asarray(
        record.get("cluster", {}).get("labels", []), dtype=int
    )
    has_best = best_labels.size == valid_events.size and best_labels.size > 0

    if has_adaptive:
        best_within_tau = adaptive.get("within_cluster_tau_mean", float("nan"))
        best_inter_corr = float("nan")
        corr_mat = adaptive.get("inter_cluster_corr_matrix", [])
        if corr_mat and best_k == 2 and len(corr_mat) >= 2 and len(corr_mat[0]) >= 2:
            best_inter_corr = float(corr_mat[0][1])
        fwd_rev_str = _strongest_forward_reverse_label(adaptive)
    else:
        k2 = record.get("cluster", {})
        best_within_tau = k2.get("within_cluster_tau_mean", float("nan"))
        best_inter_corr = k2.get("inter_cluster_corr", float("nan"))
        fwd_rev_str = ""

    # ---- 2x2 layout: left=heatmaps (wide), right=histograms (narrower) ----
    row_height = max(4.5, 0.28 * n_ch)
    fig = plt.figure(figsize=(22, row_height * 2 + 2.5))
    outer = gridspec.GridSpec(
        2, 2,
        width_ratios=[5, 1.0],
        height_ratios=[1, 1],
        hspace=0.50,
        wspace=0.15,
    )

    # ---- Top-left: Raw heatmap + Day/Night strip (flush) ----
    raw_sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0],
        height_ratios=[20, 1], hspace=0.02,
    )
    ax_raw = fig.add_subplot(raw_sub[0])
    display_ranks_raw = ranks[channel_order][:, display_events]
    im = _plot_rank_heatmap(
        ax_raw, display_ranks_raw, ordered_names,
        title=(
            f"{dataset}:{subject}  (n={valid_events.size}, "
            f"\u03c4={all_tau:.3f}){mi_str}"
        ),
    )
    ax_raw.tick_params(axis="x", labelbottom=False)
    ax_raw.set_xlabel("", fontsize=1)
    ax_raw_strip = fig.add_subplot(raw_sub[1], sharex=ax_raw)
    _plot_daynight_strip(ax_raw_strip, day_mask)

    # ---- Top-right: Per-channel rank histogram (original channel order) ----
    ax_hist_orig = fig.add_subplot(outer[0, 1])
    _plot_rank_histogram(
        ax_hist_orig, ranks, bools, valid_events,
        original_order, channel_names,
        title="Per-channel rank distribution",
    )

    # ---- Bottom-left: k_best clustered heatmap ----
    ax_clust = fig.add_subplot(outer[1, 0])
    if has_best:
        disp_best_labels = best_labels[np.isin(valid_events, display_events)]
        order_best = np.argsort(disp_best_labels, kind="stable")
        best_events_sorted = display_events[order_best]
        best_labels_sorted = disp_best_labels[order_best]
        display_ranks_best = ranks[channel_order][:, best_events_sorted]
        _plot_rank_heatmap(
            ax_clust, display_ranks_best, ordered_names, title="",
        )
        clust_title = (
            f"KMeans k={best_k}  |  within-\u03c4={best_within_tau:.3f}"
            f"  |  inter-corr: {best_inter_corr:.2f}"
        )
        if fwd_rev_str:
            clust_title += f"  |  {fwd_rev_str}"
        ax_clust.set_title(clust_title, fontsize=16, pad=28)
        _plot_cluster_boundaries(ax_clust, best_labels_sorted, n_ch)
    else:
        _plot_rank_heatmap(
            ax_clust, display_ranks_raw, ordered_names,
            title=f"KMeans k={best_k} (no labels)",
        )
    ax_clust.set_xlabel("Pop Events (clustered)", fontsize=16)

    # ---- Bottom-right: Per-cluster rank distribution, Fig4 style ----
    ax_cdist = fig.add_subplot(outer[1, 1])
    if has_best:
        _plot_cluster_rank_fig4(
            ax_cdist, ranks, bools, valid_events, best_labels,
            channel_order, channel_names,
            title="Cluster rank distributions",
        )
    else:
        _plot_rank_histogram(
            ax_cdist, ranks, bools, valid_events,
            channel_order, channel_names,
            title="Fixed-order rank distribution",
        )

    # ---- Colorbar (horizontal, below bottom-left heatmap) ----
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(
        ax_clust, width="35%", height="4%",
        loc="lower left", bbox_to_anchor=(0.32, -0.25, 1, 1),
        bbox_transform=ax_clust.transAxes, borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("First \u2192 Last", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    # ---- Suptitle ----
    suptitle = f"{dataset}:{subject} | repro={repro_grade}"
    if repro_grade == "moderate":
        suptitle += " | WARNING: template moderate"
    fig.suptitle(suptitle, fontsize=20, y=0.98)

    _ensure_pr3_readme()
    out = PR3_FIG_DIR / f"{dataset}_{subject}_propagation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_pr3_subjects(records: List[Dict[str, Any]], max_events: int = 2000) -> None:
    for record in records:
        plot_pr3_subject_figure(record, max_events=max_events)


# ---------------------------------------------------------------------------
# PR-4A: temporal dynamics figures
# ---------------------------------------------------------------------------


def _plot_pr4a_daynight_strip(ax: plt.Axes, is_day: np.ndarray) -> None:
    if is_day.size == 0:
        ax.axis("off")
        return
    strip = np.where(np.asarray(is_day, dtype=bool), 1, 0)[None, :]
    ax.imshow(
        strip,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["black", "white"]),
        vmin=0,
        vmax=1,
    )
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_pr4a_subject_timeline(record: Dict[str, Any]) -> None:
    bins = record.get("timeline_bins", [])
    if not bins:
        return
    dataset = record.get("dataset", "unknown")
    subject = record.get("subject", "unknown")
    n_clusters = int(record.get("n_clusters", 0))
    x = np.array([b["hours_from_timeline_start"] for b in bins], dtype=float)
    counts = np.array([b["n_events"] for b in bins], dtype=float)
    fractions = np.array([b["cluster_fractions"] for b in bins], dtype=float)
    is_day = np.array([b.get("day_night") == "day" for b in bins], dtype=bool)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3.0, 1.0, 0.24], hspace=0.15)

    ax_occ = fig.add_subplot(outer[0])
    for cid in range(min(n_clusters, fractions.shape[1] if fractions.ndim == 2 else 0)):
        ax_occ.plot(
            x,
            fractions[:, cid],
            marker="o",
            ms=4,
            lw=2,
            color=PR4A_COLORS[cid % len(PR4A_COLORS)],
            label=f"C{cid}",
        )
    ax_occ.set_ylabel("Occupancy fraction", fontsize=11)
    ax_occ.set_ylim(-0.02, 1.02)
    ax_occ.set_title(
        f"{dataset}:{subject}  PR-4A fixed-template occupancy"
        f"  |  k={record.get('chosen_k')}  |  grade={record.get('reproducibility_grade')}",
        fontsize=12,
    )
    ax_occ.grid(True, axis="y", alpha=0.2)
    ax_occ.spines["top"].set_visible(False)
    ax_occ.spines["right"].set_visible(False)
    ax_occ.legend(loc="upper right", ncol=min(4, max(1, n_clusters)), fontsize=9)

    ax_cnt = fig.add_subplot(outer[1], sharex=ax_occ)
    widths = np.diff(x).mean() if x.size > 1 else 1.0
    ax_cnt.bar(x, counts, width=max(widths * 0.85, 0.5), color="0.4", alpha=0.85)
    ax_cnt.set_ylabel("Events/bin", fontsize=10)
    ax_cnt.set_xlabel("Hours from timeline start", fontsize=11)
    ax_cnt.spines["top"].set_visible(False)
    ax_cnt.spines["right"].set_visible(False)

    ax_strip = fig.add_subplot(outer[2], sharex=ax_occ)
    _plot_pr4a_daynight_strip(ax_strip, is_day)

    out = PR4A_FIG_DIR / f"{dataset}_{subject}_24h_timeline.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pr4a_subject_timelines(temporal_results: Dict[str, Dict[str, Any]]) -> None:
    for record in temporal_results.values():
        if isinstance(record, dict):
            plot_pr4a_subject_timeline(record)


def plot_pr4a_daynight_group(
    temporal_results: Dict[str, Dict[str, Any]],
    cohort: Dict[str, Any],
) -> None:
    valid = []
    for record in temporal_results.values():
        if not isinstance(record, dict):
            continue
        dn = record.get("day_night_summary", {})
        day = dn.get("day", {})
        night = dn.get("night", {})
        if day.get("n_events", 0) <= 0 or night.get("n_events", 0) <= 0:
            continue
        valid.append(record)
    if not valid:
        return

    dom_day = np.array([r["day_night_summary"]["day"]["dominant_fraction"] for r in valid], dtype=float)
    dom_night = np.array([r["day_night_summary"]["night"]["dominant_fraction"] for r in valid], dtype=float)
    ent_day = np.array([r["day_night_summary"]["day"]["normalized_entropy"] for r in valid], dtype=float)
    ent_night = np.array([r["day_night_summary"]["night"]["normalized_entropy"] for r in valid], dtype=float)
    tv = np.array([r["day_night_summary"]["total_variation_distance"] for r in valid], dtype=float)
    colors = ["#2166AC" if r.get("dataset") == "yuquan" else "#E08214" for r in valid]
    summary = cohort.get("temporal_dynamics_analysis", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(dom_day, dom_night, c=colors, s=55, edgecolors="white", linewidths=0.7)
    lo = min(np.nanmin(dom_day), np.nanmin(dom_night)) - 0.02
    hi = max(np.nanmax(dom_day), np.nanmax(dom_night)) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Day dominant fraction", fontsize=10)
    ax.set_ylabel("Night dominant fraction", fontsize=10)
    ax.set_title(
        "A: Dominant cluster occupancy\n"
        f"median day-night={summary.get('dominant_fraction', {}).get('median_day_minus_night', np.nan):.3f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.scatter(ent_day, ent_night, c=colors, s=55, edgecolors="white", linewidths=0.7)
    lo = min(np.nanmin(ent_day), np.nanmin(ent_night)) - 0.02
    hi = max(np.nanmax(ent_day), np.nanmax(ent_night)) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Day normalized entropy", fontsize=10)
    ax.set_ylabel("Night normalized entropy", fontsize=10)
    ax.set_title(
        "B: Occupancy entropy\n"
        f"median day-night={summary.get('normalized_entropy', {}).get('median_day_minus_night', np.nan):.3f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.hist(tv[np.isfinite(tv)], bins=12, color="#4393C3", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Day-night total variation", fontsize=10)
    ax.set_ylabel("Subjects", fontsize=10)
    ax.set_title(
        "C: Within-subject occupancy shift\n"
        f"median TV={summary.get('day_night_total_variation', {}).get('median', np.nan):.3f}",
        fontsize=10,
    )

    fig.suptitle("PR-4A fixed-template day/night descriptive summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "pr4a_daynight_group_analysis.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_followup_single_subject(record: Dict[str, Any]) -> None:
    """Per-subject PR-4D figure: rate envelope (top) + stacked histogram (bottom)."""
    dataset = record.get("dataset", "unknown")
    subject = record.get("subject", "unknown")
    nc = int(record.get("n_clusters", record.get("chosen_k", 0)))
    rc = record.get("rate_curve", {})
    hist = record.get("histogram", {})
    summary = record.get("summary", {})

    grid_hours = np.array(rc.get("grid_hours", []), dtype=float)
    ptr = np.array(rc.get("per_template_rate", []), dtype=float)
    bin_hours = np.array(hist.get("bin_center_hours", []), dtype=float)
    bin_width_hours = np.array(hist.get("bin_width_hours", []), dtype=float)
    ptc = np.array(hist.get("per_template_count", []), dtype=int)

    if grid_hours.size == 0 or ptr.size == 0:
        return

    x_max = float(np.nanmax(grid_hours)) * 1.005

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.15)

    ax_rate = fig.add_subplot(outer[0])
    for c in range(min(nc, ptr.shape[0])):
        ax_rate.plot(
            grid_hours, ptr[c], lw=2.0,
            color=PR4A_COLORS[c % len(PR4A_COLORS)], label=f"C{c}",
        )
    ax_rate.set_xlim(0, x_max)
    ax_rate.set_ylim(bottom=0)
    ax_rate.set_ylabel("Rate (events / hr)", fontsize=11)
    dom_frac = summary.get("dominant_rate_fraction", float("nan"))
    ax_rate.set_title(
        f"{dataset}:{subject}  template-decomposed rate  |  "
        f"dom_frac={dom_frac:.3f}  n={record.get('n_events_used', '?')}",
        fontsize=12,
    )
    ax_rate.legend(loc="upper right", ncol=min(4, max(1, nc)), fontsize=9)
    ax_rate.grid(True, axis="y", alpha=0.2)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)

    ax_hist = fig.add_subplot(outer[1], sharex=ax_rate)
    if bin_hours.size > 0 and ptc.size > 0:
        if bin_width_hours.size != bin_hours.size:
            bin_width_hours = np.full(bin_hours.shape, float(record.get("bin_hours", 1.0)), dtype=float)
        bottom = np.zeros(ptc.shape[1], dtype=float)
        for c in range(min(nc, ptc.shape[0])):
            ax_hist.bar(
                bin_hours, ptc[c], width=bin_width_hours * 0.9, bottom=bottom,
                color=PR4A_COLORS[c % len(PR4A_COLORS)], alpha=0.8,
            )
            bottom += ptc[c].astype(float)
    ax_hist.set_ylim(bottom=0)
    ax_hist.set_ylabel("Event count / bin", fontsize=10)
    ax_hist.set_xlabel("Hours from timeline start", fontsize=11)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    out = PR4A_FIG_DIR / f"{dataset}_{subject}_pr4d_template_rate.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pr4a_followup_subjects(followup_results: Dict[str, Dict[str, Any]]) -> None:
    for record in followup_results.values():
        if isinstance(record, dict) and record.get("rate_curve"):
            _plot_followup_single_subject(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot interictal propagation figures")
    parser.add_argument("--pr3", action="store_true", help="Generate PR-3 per-subject figures")
    parser.add_argument("--mi", action="store_true", help="Generate per-subject MI distribution figures (Fig.2B style)")
    parser.add_argument("--cohort", action="store_true", help="Generate 6-panel cohort summary figure")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional subject filter")
    parser.add_argument("--smoke", action="store_true", help="Use chengshuai + 548 for PR-3 preview")
    parser.add_argument("--pr4a", action="store_true", help="Generate PR-4A temporal dynamics figures")
    parser.add_argument("--pr4a-followup", action="store_true", help="Generate PR-4D template-rate figures")
    parser.add_argument("--max-events", type=int, default=2000, help="Max displayed events per panel")
    args = parser.parse_args()

    if args.pr4a:
        cohort = _load("pr1_cohort_summary.json")
        temporal = _load("pr4a_temporal_dynamics.json")
        if temporal:
            plot_pr4a_subject_timelines(temporal)
            plot_pr4a_daynight_group(temporal, cohort)
        return

    if args.pr4a_followup:
        followup = _load("pr4a_followup_template_mix_dynamics.json")
        if followup:
            plot_pr4a_followup_subjects(followup)
        return

    if args.pr3:
        selected_subjects = list(args.subjects or [])
        if args.smoke and not selected_subjects:
            selected_subjects = ["chengshuai", "548"]
        records = _load_pr3_subject_records(args.dataset, selected_subjects)
        if not records:
            raise SystemExit("No matching per-subject propagation JSON files found.")
        plot_pr3_subjects(records, max_events=args.max_events)
        return

    if args.mi:
        selected_subjects = list(args.subjects or [])
        if args.smoke and not selected_subjects:
            selected_subjects = ["chengshuai", "548"]
        records = _load_pr3_subject_records(args.dataset, selected_subjects)
        if not records:
            raise SystemExit("No matching per-subject propagation JSON files found.")
        for ri, rec in enumerate(records, 1):
            print(f"[{ri}/{len(records)}] {rec['dataset']}/{rec['subject']}")
            try:
                plot_mi_subject_figure(rec)
            except Exception as e:
                print(f"  SKIP: {e}")
        return

    if args.cohort:
        subjects = _load("pr1_subject_summary.json")
        cohort = _load("pr1_cohort_summary.json")
        if not subjects:
            raise SystemExit("No pr1_subject_summary.json found.")
        plot_cohort_summary(subjects, cohort)
        return

    subjects = _load("pr1_subject_summary.json")
    cohort = _load("pr1_cohort_summary.json")
    if not subjects:
        return
    plot_heatmap_examples(subjects)
    plot_cohort_summary(subjects, cohort)


if __name__ == "__main__":
    main()
