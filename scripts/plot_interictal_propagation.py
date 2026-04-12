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

RESULTS_DIR = Path("results/interictal_propagation")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
PR3_FIG_DIR = FIG_DIR / "per_subject"
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")

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


def _style_panel(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)


COL_YQ = "#2166AC"
COL_EPI = "#E08214"
COL_SIG = "#B2182B"
COL_NONSIG = "#999999"


def plot_cohort_summary(subjects: Dict[str, Dict[str, Any]], cohort: Dict[str, Any]) -> None:
    """Publication-grade 6-panel cohort propagation stereotypy figure."""
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

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ================================================================
    # Panel A: MI Violin (improved paper Fig.2c)
    # ================================================================
    ax = axes[0, 0]
    _style_panel(ax)

    def _mi_medians(recs):
        out = []
        for r in recs:
            mi = r.get("legacy_mi", {})
            m = mi.get("mi_median", mi.get("mi_mean", np.nan))
            if np.isfinite(m):
                out.append((m, mi.get("significant", False)))
        return out

    yq_mi = _mi_medians(yq)
    epi_mi = _mi_medians(epi)

    positions = [0, 1]
    datasets_mi = [yq_mi, epi_mi]
    box_colors = ["#30368b", "#5c2366"]
    jitter = 0.04

    for i, (mi_data, pos) in enumerate(zip(datasets_mi, positions)):
        if not mi_data:
            continue
        vals = [v for v, _ in mi_data]
        sigs = [s for _, s in mi_data]

        violins = ax.violinplot(
            vals, positions=[pos], widths=0.45,
            bw_method="silverman",
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in violins["bodies"]:
            pc.set_facecolor("none")
            pc.set_edgecolor("black")
            pc.set_linewidth(2.0)

        bp = ax.boxplot(
            vals, positions=[pos], widths=0.2,
            patch_artist=True, showfliers=False, showcaps=False,
            medianprops=dict(linewidth=3, color="black", solid_capstyle="butt"),
            whiskerprops=dict(linewidth=2, color="black"),
            zorder=-3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(box_colors[i])
            patch.set_edgecolor("black")
            patch.set_linewidth(2)

        x_jit = np.full(len(vals), pos) + st.t(df=6, scale=jitter).rvs(len(vals))
        v_arr = np.array(vals)
        s_arr = np.array(sigs)
        if s_arr.any():
            ax.scatter(x_jit[s_arr], v_arr[s_arr], s=80,
                       color=COL_SIG, edgecolor="white", linewidths=0.5, zorder=3)
            ax.scatter(x_jit[s_arr], v_arr[s_arr], s=80,
                       marker="+", color="black", linewidths=1.5, zorder=4)
        if (~s_arr).any():
            ax.scatter(x_jit[~s_arr], v_arr[~s_arr], s=60,
                       facecolors="none", edgecolors=COL_NONSIG, linewidths=1.2, zorder=3)

    lmi = cohort.get("legacy_mi", {})
    n_sig = lmi.get("n_significant", 0)
    n_tested = lmi.get("n_tested", 0)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Yuquan", "Epilepsiae"], fontsize=16)
    ax.set_ylabel("Median MI", fontsize=16)
    ax.set_title(f"A: Matching Index  (significant: {n_sig}/{n_tested})", fontsize=18)
    ax.set_xlim(-0.6, 1.6)

    # ================================================================
    # Panel B: Cluster-aware tau uplift scatter
    # ================================================================
    ax = axes[0, 1]
    _style_panel(ax)
    overall_vals, within_vals, colors_b = [], [], []
    for rec in valid:
        cl = rec.get("cluster", {})
        ov = cl.get("overall_tau", np.nan)
        wi = cl.get("within_cluster_tau_mean", np.nan)
        if np.isfinite(ov) and np.isfinite(wi):
            overall_vals.append(ov)
            within_vals.append(wi)
            colors_b.append(COL_YQ if rec["dataset"] == "yuquan" else COL_EPI)
    if overall_vals:
        ov_arr = np.array(overall_vals)
        wi_arr = np.array(within_vals)
        lim = max(np.max(ov_arr), np.max(wi_arr)) + 0.03
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4)
        ax.scatter(ov_arr, wi_arr, c=colors_b, s=70, edgecolors="white", linewidths=0.8, zorder=2)
        ax.set_xlim(-0.01, lim)
        ax.set_ylim(-0.01, lim)
    ca = cohort.get("cluster_analysis", {})
    ax.set_xlabel("Overall mean \u03c4", fontsize=16)
    ax.set_ylabel("Within-cluster mean \u03c4", fontsize=16)
    ax.set_title(
        f"B: Cluster-aware stereotypy\n"
        f"uplift median={ca.get('uplift_median', np.nan):.3f}",
        fontsize=18,
    )
    ax.grid(True, alpha=0.15)

    # ================================================================
    # Panel C: Inter-cluster correlation histogram
    # ================================================================
    ax = axes[0, 2]
    _style_panel(ax)
    inter_corrs = [
        rec["cluster"]["inter_cluster_corr"]
        for rec in valid
        if rec.get("cluster") and np.isfinite(rec["cluster"].get("inter_cluster_corr", np.nan))
    ]
    if inter_corrs:
        ax.hist(inter_corrs, bins=15, color="#4393C3", edgecolor="white", alpha=0.85)
        ax.axvline(-0.5, color="red", ls="--", lw=1.5, alpha=0.7,
                   label="r < \u22120.5 (forward/reverse)")
        ax.axvline(0, color="black", ls="-", lw=1, alpha=0.3)
        ax.legend(fontsize=13)
    n_anti = ca.get("n_anticorrelated", 0)
    ax.set_xlabel("Inter-cluster Spearman r", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title(
        f"C: Inter-cluster correlation\n"
        f"median={ca.get('inter_cluster_corr_median', np.nan):.2f}, "
        f"n(r<\u22120.5)={n_anti}",
        fontsize=18,
    )

    # ================================================================
    # Panel D: Reproducibility grade stacked bar
    # ================================================================
    ax = axes[1, 0]
    _style_panel(ax)
    grade_colors = {"strong": "#2ca02c", "moderate": "#f0ad4e", "weak": "#d9534f"}
    grade_order = ["strong", "moderate", "weak"]

    def _count_grades(recs):
        counts = {"strong": 0, "moderate": 0, "weak": 0}
        for r in recs:
            g = r.get("time_split_reproducibility", {}).get("reproducibility_grade", "")
            if g in counts:
                counts[g] += 1
        return counts

    yq_grades = _count_grades(yq)
    epi_grades = _count_grades(epi)
    x_pos = np.array([0, 1])
    bottom_yq, bottom_epi = 0, 0
    for grade in grade_order:
        heights = [yq_grades[grade], epi_grades[grade]]
        bars = ax.bar(x_pos, heights, bottom=[bottom_yq, bottom_epi],
                      width=0.5, color=grade_colors[grade], edgecolor="white",
                      linewidth=0.8, label=grade)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + h / 2.0, str(h),
                        ha="center", va="center", fontsize=14, fontweight="bold")
        bottom_yq += yq_grades[grade]
        bottom_epi += epi_grades[grade]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Yuquan", "Epilepsiae"], fontsize=16)
    ax.set_ylabel("Subject count", fontsize=16)
    total_s = yq_grades["strong"] + epi_grades["strong"]
    total_m = yq_grades["moderate"] + epi_grades["moderate"]
    total_w = yq_grades["weak"] + epi_grades["weak"]
    ax.set_title(
        f"D: Template reproducibility\n"
        f"{total_s} strong / {total_m} moderate / {total_w} weak",
        fontsize=18,
    )
    ax.legend(fontsize=13, loc="upper right")

    # ================================================================
    # Panel E: stable_k distribution
    # ================================================================
    ax = axes[1, 1]
    _style_panel(ax)
    k_vals_yq = []
    k_vals_epi = []
    for r in yq:
        sk = r.get("adaptive_cluster", {}).get("stable_k")
        if sk is not None:
            k_vals_yq.append(int(sk))
    for r in epi:
        sk = r.get("adaptive_cluster", {}).get("stable_k")
        if sk is not None:
            k_vals_epi.append(int(sk))
    all_ks = sorted(set(k_vals_yq + k_vals_epi))
    if not all_ks:
        all_ks = [2]
    yq_counts = [k_vals_yq.count(k) for k in all_ks]
    epi_counts = [k_vals_epi.count(k) for k in all_ks]
    x_k = np.arange(len(all_ks))
    bar_w = 0.35
    ax.bar(x_k - bar_w / 2, yq_counts, bar_w, color=COL_YQ, edgecolor="white",
           linewidth=0.8, label="Yuquan")
    ax.bar(x_k + bar_w / 2, epi_counts, bar_w, color=COL_EPI, edgecolor="white",
           linewidth=0.8, label="Epilepsiae")
    ax.set_xticks(x_k)
    ax.set_xticklabels([f"k={k}" for k in all_ks], fontsize=14)
    ax.set_ylabel("Subject count", fontsize=16)
    mode_k = max(all_ks, key=lambda k: k_vals_yq.count(k) + k_vals_epi.count(k))
    mode_n = k_vals_yq.count(mode_k) + k_vals_epi.count(mode_k)
    ax.set_title(
        f"E: Adaptive stable_k distribution\n"
        f"mode: k={mode_k} ({mode_n}/{len(k_vals_yq) + len(k_vals_epi)} subjects)",
        fontsize=18,
    )
    ax.legend(fontsize=13)

    # ================================================================
    # Panel F: Raw vs Centered tau scatter
    # ================================================================
    ax = axes[1, 2]
    _style_panel(ax)
    raw_list, cen_list, sc_colors = [], [], []
    for rec in valid:
        c = rec.get("centered_rank", {})
        if np.isfinite(c.get("raw_tau", np.nan)) and np.isfinite(c.get("centered_tau", np.nan)):
            raw_list.append(float(c["raw_tau"]))
            cen_list.append(float(c["centered_tau"]))
            erased = rec.get("source_diagnostic", {}).get("soz_source_erased", False)
            sc_colors.append(COL_SIG if erased else COL_YQ)
    if raw_list:
        r_arr = np.array(raw_list)
        c_arr = np.array(cen_list)
        lo = min(np.min(r_arr), np.min(c_arr)) - 0.02
        hi = max(np.max(r_arr), np.max(c_arr)) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
        ax.scatter(r_arr, c_arr, c=sc_colors, s=70, edgecolors="white", linewidths=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Raw mean \u03c4", fontsize=16)
    ax.set_ylabel("Centered mean \u03c4", fontsize=16)
    ax.set_title(
        f"F: Identity-bias diagnosis\n"
        f"bias fraction median={cohort.get('bias_fraction_median', np.nan):.3f}",
        fontsize=18,
    )
    ax.grid(True, alpha=0.15)

    # ================================================================
    fig.suptitle(
        "Interictal Propagation Stereotypy \u2014 Cohort Summary",
        fontsize=20, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "cohort_propagation_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot interictal propagation figures")
    parser.add_argument("--pr3", action="store_true", help="Generate PR-3 per-subject figures")
    parser.add_argument("--mi", action="store_true", help="Generate per-subject MI distribution figures (Fig.2B style)")
    parser.add_argument("--cohort", action="store_true", help="Generate 6-panel cohort summary figure")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional subject filter")
    parser.add_argument("--smoke", action="store_true", help="Use chengshuai + 548 for PR-3 preview")
    parser.add_argument("--max-events", type=int, default=2000, help="Max displayed events per panel")
    args = parser.parse_args()

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
