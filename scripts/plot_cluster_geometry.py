#!/usr/bin/env python3
"""Topic 1 cluster geometry visualization — figures.

Per-subject 3-panel + cohort 2x2 + showcase 3 figures, all under the
template-matching metric (see archive plan §3.3 / §5).

Design doc: docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (  # noqa: E402
    COL_EPILEPSIAE,
    COL_NEUTRAL,
    COL_NIGHT,
    COL_NONSIG,
    COL_SIG,
    COL_YUQUAN,
    DPI_PUB,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    add_significance_bracket,
    dataset_color,
    new_figure,
    savefig_pub,
    style_panel,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("plot_cluster_geometry")


GEOMETRY_DIR = Path("results/interictal_propagation/cluster_geometry")
PER_SUBJECT_DIR = GEOMETRY_DIR / "per_subject"
FIGURES_DIR = GEOMETRY_DIR / "figures"
PER_SUBJECT_FIG_DIR = FIGURES_DIR / "per_subject"
SHOWCASE_FIG_DIR = FIGURES_DIR / "showcase"


# Cluster colors: distinct, Morandi-flavored
CLUSTER_PALETTE = [
    "#6F8FA8",  # Morandi blue
    "#A35E48",  # Morandi rust
    "#9DAA90",  # Morandi sage
    "#C9A86A",  # Morandi mustard
    "#7E6E84",  # Morandi plum
    "#B07A6E",  # Morandi terracotta
]
TEMPLATE_MARKER_SIZE = 360  # large stars
EVENT_MARKER_SIZE = 18
BOUNDARY_MARKER_SIZE = 36


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_subject_geometry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_cohort_summary() -> Optional[Dict[str, Any]]:
    p = GEOMETRY_DIR / "cohort_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _list_per_subject_files() -> List[Path]:
    if not PER_SUBJECT_DIR.exists():
        return []
    return sorted(PER_SUBJECT_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Per-subject 3-panel figure
# ---------------------------------------------------------------------------


def _scatter_metric_panel(
    ax: plt.Axes,
    events: List[Dict[str, Any]],
    templates: List[Dict[str, Any]],
    chosen_k: int,
    x_field: str,
    y_field: str,
    bin_decimals: int = 3,
) -> Tuple[int, int]:
    """Render a 2D density-aware scatter (PCA or MDS) of events + templates.

    Identical (x, y) coordinates within ``bin_decimals`` rounding precision
    are aggregated into a single marker whose size scales with the count
    (sqrt scaling so a 100x denser bin is 10x the area). This is essential
    for low-n_ch subjects whose discrete rank vectors collide heavily
    (e.g. 818 has 120 unique PCA coords for 11337 events).

    Returns (n_events_visible, n_unique_bins).
    """
    bin_key = lambda x, y: (round(x, bin_decimals), round(y, bin_decimals))

    n_visible = 0
    n_unique_bins = 0
    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        agree_counts: Dict[Tuple[float, float], int] = {}
        bound_counts: Dict[Tuple[float, float], int] = {}
        for ev in events:
            if ev["kmeans_label"] != k:
                continue
            x = ev.get(x_field)
            y = ev.get(y_field)
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue
            n_visible += 1
            key = bin_key(x, y)
            if ev.get("agreement", True):
                agree_counts[key] = agree_counts.get(key, 0) + 1
            else:
                bound_counts[key] = bound_counts.get(key, 0) + 1

        n_unique_bins += len(agree_counts) + len(bound_counts)

        if agree_counts:
            xs = [k[0] for k in agree_counts.keys()]
            ys = [k[1] for k in agree_counts.keys()]
            counts = np.array(list(agree_counts.values()), dtype=float)
            sizes = EVENT_MARKER_SIZE * np.sqrt(counts)
            ax.scatter(xs, ys, s=sizes, c=col, alpha=0.55,
                       edgecolors="none", zorder=2)
        if bound_counts:
            xs = [k[0] for k in bound_counts.keys()]
            ys = [k[1] for k in bound_counts.keys()]
            counts = np.array(list(bound_counts.values()), dtype=float)
            sizes = BOUNDARY_MARKER_SIZE * np.sqrt(counts)
            ax.scatter(xs, ys, s=sizes, facecolors="none",
                       edgecolors=col, linewidths=1.3, alpha=0.9, zorder=3)

    for k, tmpl in enumerate(templates):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        tx = tmpl.get(x_field)
        ty = tmpl.get(y_field)
        if tx is None or ty is None or not np.isfinite(tx) or not np.isfinite(ty):
            continue
        ax.scatter([tx], [ty], marker="*", s=TEMPLATE_MARKER_SIZE,
                   c=col, edgecolors="black", linewidths=1.5, zorder=4)
    return n_visible, n_unique_bins


def _load_pr2_validity(dataset: str, subject: str) -> Dict[str, Any]:
    """Pull cluster-validity numbers from PR-2's pr1_subject_summary.json
    for the figure's stats annotation. Returns {} if not available.
    """
    pr2_path = Path("results/interictal_propagation/per_subject") / f"{dataset}_{subject}.json"
    if not pr2_path.exists():
        return {}
    try:
        with pr2_path.open() as f:
            d = json.load(f)
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    mix = d.get("mixture", {})
    out["dip_p"] = mix.get("dip_p")
    out["silhouette_k2_diptest"] = mix.get("silhouette_k2")
    ac = d.get("adaptive_cluster", {})
    out["overall_tau"] = ac.get("overall_tau")
    out["within_cluster_tau_mean"] = ac.get("within_cluster_tau_mean")
    out["uplift"] = ac.get("uplift")
    icc = ac.get("inter_cluster_corr_matrix")
    if icc:
        arr = np.array(icc, dtype=float)
        n = arr.shape[0]
        off = [arr[i, j] for i in range(n) for j in range(i + 1, n) if np.isfinite(arr[i, j])]
        out["inter_cluster_min_r"] = float(np.min(off)) if off else None
        out["inter_cluster_max_r"] = float(np.max(off)) if off else None
    out["adaptive_ami_at_chosen_k"] = next(
        (s.get("median_ami") for s in ac.get("scan", []) if s.get("k") == ac.get("chosen_k")),
        None,
    )
    out["adaptive_silhouette_at_chosen_k"] = next(
        (s.get("median_silhouette") for s in ac.get("scan", []) if s.get("k") == ac.get("chosen_k")),
        None,
    )
    rp = d.get("time_split_reproducibility", {})
    out["pr25_grade"] = rp.get("reproducibility_grade")
    splits = rp.get("splits", {})
    halfsplit = splits.get("first_half_second_half", {})
    out["pr25_template_corr_first_second"] = halfsplit.get("mean_match_corr")
    out["pr25_assignment_first_second"] = halfsplit.get("assignment_agreement")
    odd = splits.get("odd_even_block", {})
    out["pr25_template_corr_odd_even"] = odd.get("mean_match_corr")
    mi = d.get("legacy_mi", {})
    out["mi_permutation_p"] = mi.get("p_value")
    return out


def _validity_subtitle(v: Dict[str, Any]) -> str:
    """One-line summary of PR-2 cluster validity numbers for figure suptitle."""
    if not v:
        return ""
    parts = []
    dip_p = v.get("dip_p")
    if dip_p is not None:
        parts.append(f"dip p<{1e-3:.0e}" if dip_p < 1e-3 else f"dip p={dip_p:.3g}")
    ami = v.get("adaptive_ami_at_chosen_k")
    if ami is not None and np.isfinite(ami):
        parts.append(f"AMI={ami:.2f}")
    sil = v.get("adaptive_silhouette_at_chosen_k")
    if sil is not None and np.isfinite(sil):
        parts.append(f"sil(KMeans)={sil:.2f}")
    icr_min = v.get("inter_cluster_min_r")
    icr_max = v.get("inter_cluster_max_r")
    if icr_min is not None and icr_max is not None:
        if abs(icr_min) >= abs(icr_max):
            parts.append(f"inter-cluster r={icr_min:.2f}")
        else:
            parts.append(f"inter-cluster r={icr_max:.2f}")
    grade = v.get("pr25_grade")
    if grade:
        parts.append(f"PR-2.5 {grade}")
    mi_p = v.get("mi_permutation_p")
    if mi_p is not None:
        if mi_p < 1e-3:
            parts.append(f"MI perm p<0.001")
        else:
            parts.append(f"MI perm p={mi_p:.3g}")
    uplift = v.get("uplift")
    if uplift is not None and np.isfinite(uplift):
        parts.append(f"τ uplift +{uplift:.2f}")
    return " · ".join(parts)


def _select_axis_cluster_pair(
    data: Dict[str, Any],
) -> Tuple[int, int]:
    """Pick (a, b) cluster IDs whose templates form the figure's x/y axes.

    Strategy: choose the most anti-correlated template pair (lowest
    inter-cluster Spearman r). If no inter-cluster matrix or only k=2,
    fall back to clusters 0 and 1 (or 0 and the next largest).
    """
    chosen_k = int(data["chosen_k"])
    if chosen_k < 2:
        return 0, 0  # degenerate

    icc = data.get("inter_cluster_corr_matrix")
    if icc is not None:
        arr = np.array(icc, dtype=float)
        n = arr.shape[0]
        best_r = float("inf")
        best_pair = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                r = arr[i, j]
                if r is not None and np.isfinite(r) and r < best_r:
                    best_r = r
                    best_pair = (i, j)
        if np.isfinite(best_r):
            return best_pair

    # Fallback: largest two clusters
    counts = [(int(t.get("cluster_id", k)), 0) for k, t in enumerate(data.get("templates", []))]
    return 0, 1


def plot_template_distance_plane(
    data: Dict[str, Any],
    output_path: Path,
    is_showcase: bool = False,
) -> Path:
    """Single-panel per-subject figure in fixed template-distance coords.

    x = d(event, T_a), y = d(event, T_b) where (a, b) is the most
    anti-correlated template pair (Spearman r). Axes have universal
    semantics across all subjects with k>=2:
      - origin = "event matches both T_a and T_b" (impossible unless T_a==T_b)
      - on x-axis = "event matches T_b exactly"
      - on y-axis = "event matches T_a exactly"
      - diagonal y=x = decision boundary (equidistant)
      - far from origin = "event is dissimilar to both" (off-mode)

    Templates land at fixed positions:
      - T_a at (0, d(T_a, T_b))
      - T_b at (d(T_a, T_b), 0)
      - T_other at (d(T_a, T_other), d(T_b, T_other))

    Density-aware: marker size proportional to count of identical
    (x, y) within rounding precision.
    """
    if data.get("status") != "ok":
        logger.warning("Skip %s: status=%s", data.get("subject"), data.get("status"))
        return output_path

    dataset = data.get("dataset", "")
    subject = data.get("subject", "")
    chosen_k = int(data["chosen_k"])
    events = data["events"]
    templates = data["templates"]
    D_tt = np.array(data.get("template_template_distance", [[]]), dtype=float)

    ax_a, ax_b = _select_axis_cluster_pair(data)
    if ax_a == ax_b:
        logger.warning("Skip %s: degenerate axis pair (k=%d)", subject, chosen_k)
        return output_path

    icc = np.array(data.get("inter_cluster_corr_matrix", [[1.0]]), dtype=float)
    pair_r = float(icc[ax_a, ax_b]) if ax_a < icc.shape[0] and ax_b < icc.shape[1] else float("nan")

    figsize = (8.5, 8.0) if not is_showcase else (11, 10)
    fig, ax = new_figure(nrows=1, ncols=1, figsize=figsize)
    style_panel(ax, label="", label_x=-0.10, label_y=1.04)
    ax.set_facecolor("#FAFAFA")

    # Color events by **matching_label** so that, by construction, all
    # blue events have d_to_T_a < d_to_T_b (above diagonal y>x) and all
    # rust events have d_to_T_b < d_to_T_a (below). y=x then is the strict
    # decision boundary for matching's argmin rule (k=2 case; for k>2 it
    # is the T_a-vs-T_b slice, with off-axis matching clusters falling on
    # either side depending on which axis their template is closer to).
    # Open circles flag events where KMeans's full-rank decision disagrees
    # with matching's masked-distance decision (metric drift).
    bin_decimals = 3
    n_visible = 0
    n_unique_bins = 0
    n_unassigned = 0
    n_disagree = 0

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        agree_counts: Dict[Tuple[float, float], int] = {}
        bound_counts: Dict[Tuple[float, float], int] = {}
        for ev in events:
            mlabel = ev.get("matching_label", -1)
            if mlabel != k:
                continue
            d_each = ev.get("d_to_each_template")
            if d_each is None or len(d_each) <= max(ax_a, ax_b):
                continue
            xv = d_each[ax_a]
            yv = d_each[ax_b]
            if xv is None or yv is None or not np.isfinite(xv) or not np.isfinite(yv):
                continue
            n_visible += 1
            key = (round(xv, bin_decimals), round(yv, bin_decimals))
            if ev.get("agreement", True):
                agree_counts[key] = agree_counts.get(key, 0) + 1
            else:
                bound_counts[key] = bound_counts.get(key, 0) + 1
                n_disagree += 1
        n_unique_bins += len(agree_counts) + len(bound_counts)

        if agree_counts:
            xs = [p[0] for p in agree_counts.keys()]
            ys = [p[1] for p in agree_counts.keys()]
            counts = np.array(list(agree_counts.values()), dtype=float)
            sizes = EVENT_MARKER_SIZE * np.sqrt(counts)
            ax.scatter(xs, ys, s=sizes, c=col, alpha=0.55,
                       edgecolors="none", zorder=2,
                       label=f"matching cluster {k}"
                             if k in (ax_a, ax_b)
                             else f"matching cluster {k} (off-axis)")
        if bound_counts:
            xs = [p[0] for p in bound_counts.keys()]
            ys = [p[1] for p in bound_counts.keys()]
            counts = np.array(list(bound_counts.values()), dtype=float)
            sizes = BOUNDARY_MARKER_SIZE * np.sqrt(counts)
            ax.scatter(xs, ys, s=sizes, facecolors="none",
                       edgecolors=col, linewidths=1.3, alpha=0.85, zorder=3)

    # Plot unassigned (matching_label = -1) events in neutral gray for
    # honesty — they cannot be argmin'd because no template has enough
    # shared channels.
    unassigned_counts: Dict[Tuple[float, float], int] = {}
    for ev in events:
        if ev.get("matching_label", -1) != -1:
            continue
        d_each = ev.get("d_to_each_template")
        if d_each is None:
            continue
        xv = d_each[ax_a] if len(d_each) > ax_a else None
        yv = d_each[ax_b] if len(d_each) > ax_b else None
        if xv is None or yv is None or not np.isfinite(xv) or not np.isfinite(yv):
            continue
        n_visible += 1
        n_unassigned += 1
        key = (round(xv, bin_decimals), round(yv, bin_decimals))
        unassigned_counts[key] = unassigned_counts.get(key, 0) + 1
    if unassigned_counts:
        xs = [p[0] for p in unassigned_counts.keys()]
        ys = [p[1] for p in unassigned_counts.keys()]
        counts = np.array(list(unassigned_counts.values()), dtype=float)
        sizes = EVENT_MARKER_SIZE * np.sqrt(counts)
        ax.scatter(xs, ys, s=sizes, c="#BDBDBD", alpha=0.55,
                   edgecolors="none", zorder=2,
                   label=f"unassigned (n={n_unassigned})")

    # Plot all templates at their fixed positions in this plane
    if D_tt.size > 0:
        for k in range(chosen_k):
            col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
            tx = D_tt[ax_a, k] if ax_a < D_tt.shape[0] and k < D_tt.shape[1] else float("nan")
            ty = D_tt[ax_b, k] if ax_b < D_tt.shape[0] and k < D_tt.shape[1] else float("nan")
            if np.isfinite(tx) and np.isfinite(ty):
                ax.scatter([tx], [ty], marker="*", s=TEMPLATE_MARKER_SIZE,
                           c=col, edgecolors="black", linewidths=1.5, zorder=5)
                ax.annotate(f"T{k}", (tx, ty), xytext=(8, 8),
                            textcoords="offset points", fontsize=FS_TICK,
                            fontweight="bold")

    # Decision boundary y=x (light line)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    diag_lo = max(0.0, min(xmin, ymin))
    diag_hi = max(xmax, ymax)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#999999",
            lw=1.0, ls="--", alpha=0.6, zorder=1, label="y=x decision boundary")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    ax.set_xlabel(f"d(event, T{ax_a})  [shared-channel RMS rank deviation]", fontsize=FS_LABEL)
    ax.set_ylabel(f"d(event, T{ax_b})", fontsize=FS_LABEL)

    n_total = data["n_events_total"]
    agreement = data["agreement_overall"]
    sil_med = data["silhouette_median"]
    extra_clusters = chosen_k - 2
    extra_note = f" + {extra_clusters} off-axis matching cluster(s)" if extra_clusters > 0 else ""
    boundary_kind = (
        "y=x is the strict matching argmin boundary"
        if chosen_k == 2
        else f"y=x is the T{ax_a} vs T{ax_b} slice (off-axis clusters not bound by it)"
    )
    panel_title = (
        f"Template-distance plane: T{ax_a} vs T{ax_b}  "
        f"(inter-cluster r={pair_r:.2f})\n"
        f"{n_visible:,} events on {n_unique_bins} unique rank vectors "
        f"(k={chosen_k}{extra_note}, agreement={agreement:.3f}, sil={sil_med:.3f})\n"
        f"{boundary_kind}; "
        f"open circles = {n_disagree:,} events ({n_disagree / max(n_visible,1):.1%}) where KMeans disagrees with matching"
    )
    ax.set_title(panel_title, fontsize=FS_TITLE - 4, loc="left")
    ax.legend(loc="upper right", fontsize=FS_TICK - 2, frameon=False)

    # Suptitle: subject id + validity stats
    validity = _load_pr2_validity(dataset, subject)
    validity_subtitle = _validity_subtitle(validity)
    fig.suptitle(
        f"{dataset.capitalize()} · {subject} — fixed template-distance plane\n"
        f"{validity_subtitle}",
        fontsize=FS_TITLE,
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return savefig_pub(fig, output_path)


def plot_per_subject_geometry(
    data: Dict[str, Any],
    output_path: Path,
    is_showcase: bool = False,
) -> Path:
    """Produce the 4-panel (2x2) per-subject figure.

    Panel a: PCA scatter on KMeans feature matrix — all events, KMeans-native view
    Panel b: per-event silhouette ranked (template-matching metric, audit)
    Panel c: cluster template profile (channel-rank lines + IQR band)
    Panel d: MDS scatter under template-matching distance — subsample audit
    """
    if data.get("status") != "ok":
        logger.warning("Skip %s: status=%s", data.get("subject"), data.get("status"))
        return output_path

    dataset = data.get("dataset", "")
    subject = data.get("subject", "")
    chosen_k = int(data["chosen_k"])
    events = data["events"]
    templates = data["templates"]
    channel_names = data.get("channel_names", [])
    n_ch = len(channel_names)

    figsize = (16, 11) if not is_showcase else (20, 13)
    fig, axes = new_figure(nrows=2, ncols=2, figsize=figsize)

    # -----------------------------------------------------------------------
    # Panel a: PCA scatter — KMeans-native, ALL events
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    style_panel(ax, label="a", label_x=-0.10, label_y=1.04)
    ax.set_facecolor("#FAFAFA")
    n_pca_visible, n_pca_bins = _scatter_metric_panel(
        ax, events, templates, chosen_k, x_field="pca_x", y_field="pca_y",
    )
    evr = data.get("pca_explained_variance_ratio", [None, None])
    evr_str = ""
    if evr and all(v is not None for v in evr[:2]):
        evr_str = f"  EVR={evr[0]:.2f},{evr[1]:.2f}"
    ax.set_xlabel(f"PC1 (KMeans feature space){evr_str}", fontsize=FS_LABEL)
    ax.set_ylabel("PC2", fontsize=FS_LABEL)

    n_total = data["n_events_total"]
    n_used = data["n_events_used_for_mds"]
    sil_med = data["silhouette_median"]
    agreement = data["agreement_overall"]
    stress = data["stress"]
    title_a = (
        f"PCA · {n_pca_visible:,} events on {n_pca_bins} unique rank vectors  "
        f"(k={chosen_k}  agreement={agreement:.3f})"
    )
    ax.set_title(title_a, fontsize=FS_TITLE - 4, loc="left")

    # -----------------------------------------------------------------------
    # Panel b: per-event silhouette ranked bar
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    style_panel(ax, label="b", label_x=-0.10, label_y=1.04)

    # Group by cluster, sort sils within cluster (desc); plot as filled curves
    cluster_block_centers: List[Tuple[int, float]] = []
    cur = 0
    n_total_events = sum(
        1 for ev in events if ev["silhouette"] is not None and np.isfinite(ev["silhouette"])
    )
    gap = max(int(n_total_events * 0.01), 50)
    n_neg_total = 0

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        sil_in_k = np.array(
            sorted(
                [
                    ev["silhouette"]
                    for ev in events
                    if ev["kmeans_label"] == k
                    and ev["silhouette"] is not None
                    and np.isfinite(ev["silhouette"])
                ],
                reverse=True,
            ),
            dtype=float,
        )
        n_in_k = sil_in_k.size
        if n_in_k == 0:
            continue
        block_start = cur
        x = np.arange(n_in_k) + cur
        # Positive portion fill
        pos_mask = sil_in_k >= 0
        neg_mask = ~pos_mask
        n_neg_total += int(neg_mask.sum())
        # Plot positive band in cluster color
        if pos_mask.any():
            ax.fill_between(
                x, np.zeros(n_in_k), np.where(pos_mask, sil_in_k, 0.0),
                color=col, alpha=0.55, linewidth=0,
            )
        # Plot negative band in rust (boundary events)
        if neg_mask.any():
            ax.fill_between(
                x, np.where(neg_mask, sil_in_k, 0.0), np.zeros(n_in_k),
                color=COL_SIG, alpha=0.75, linewidth=0,
            )
        # Outline curve
        ax.plot(x, sil_in_k, color=col, lw=1.0, alpha=0.9)
        cluster_block_centers.append((k, (block_start + cur + n_in_k) / 2))
        cur += n_in_k + gap

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("event (ordered by cluster, then silhouette desc)", fontsize=FS_LABEL)
    ax.set_ylabel("per-event silhouette", fontsize=FS_LABEL)
    if cluster_block_centers:
        ax.set_xticks([c[1] for c in cluster_block_centers])
        ax.set_xticklabels(
            [f"cluster {c[0]}" for c in cluster_block_centers], fontsize=FS_TICK
        )
    ax.set_ylim(-1.05, 1.05)
    neg_frac = n_neg_total / max(n_total_events, 1)
    ax.set_title(
        f"per-event silhouette\nnegative {n_neg_total:,}/{n_total_events:,} ({neg_frac:.1%})",
        fontsize=FS_TITLE - 4, loc="left",
    )

    # -----------------------------------------------------------------------
    # Panel c: cluster template profile
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    style_panel(ax, label="c", label_x=-0.10, label_y=1.04)

    templates_real = np.asarray(data["templates_real"], dtype=float)  # (k, n_ch)

    # Choose ordering channel: dominant cluster (largest n_events)
    cluster_n_events = [
        sum(1 for ev in events if ev["kmeans_label"] == k) for k in range(chosen_k)
    ]
    dominant_k = int(np.argmax(cluster_n_events))
    dom_template = templates_real[dominant_k]
    finite_mask = np.isfinite(dom_template)
    if not finite_mask.any():
        # Fallback: pick first cluster with finite values
        for k in range(chosen_k):
            if np.any(np.isfinite(templates_real[k])):
                dominant_k = k
                dom_template = templates_real[k]
                finite_mask = np.isfinite(dom_template)
                break

    # Build channel order: finite channels of dominant template sorted by
    # template rank ascending; non-finite channels appended at end
    finite_idx = np.where(finite_mask)[0]
    sorted_finite = finite_idx[np.argsort(dom_template[finite_idx])]
    nonfinite_idx = np.where(~finite_mask)[0]
    channel_order = np.concatenate([sorted_finite, nonfinite_idx])

    # Per-cluster IQR computed on raw rank vectors of events in that cluster
    n_show = len(sorted_finite)  # only show finite channels of dominant template
    x_pos = np.arange(n_show)

    iqr_low = {
        int(t["cluster_id"]): np.asarray(t.get("template_iqr_low", []), dtype=float)
        for t in templates
    }
    iqr_high = {
        int(t["cluster_id"]): np.asarray(t.get("template_iqr_high", []), dtype=float)
        for t in templates
    }

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        tline = templates_real[k, channel_order[:n_show]]
        finite_line = np.isfinite(tline)
        # IQR band when present
        lo = iqr_low.get(k)
        hi = iqr_high.get(k)
        if lo is not None and hi is not None and lo.size == n_ch and hi.size == n_ch:
            lo_ord = lo[channel_order[:n_show]]
            hi_ord = hi[channel_order[:n_show]]
            band_finite = np.isfinite(lo_ord) & np.isfinite(hi_ord)
            if band_finite.any():
                ax.fill_between(
                    x_pos[band_finite], lo_ord[band_finite], hi_ord[band_finite],
                    color=col, alpha=0.18, linewidth=0, zorder=1,
                )
        ax.plot(
            x_pos[finite_line],
            tline[finite_line],
            color=col, lw=2.2,
            marker="o", markersize=5,
            label=f"cluster {k}",
            zorder=3,
        )

    ax.set_xlabel("channel (ordered by dominant cluster rank)", fontsize=FS_LABEL)
    ax.set_ylabel("template mean rank", fontsize=FS_LABEL)
    ax.set_title("cluster template profile", fontsize=FS_TITLE - 4, loc="left")
    ax.legend(loc="best", fontsize=FS_TICK - 2, frameon=False)

    # Channel name ticks (sparse if many)
    if n_show <= 12:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [channel_names[ci] for ci in channel_order[:n_show]],
            rotation=45, ha="right", fontsize=FS_TICK - 2,
        )
    else:
        step = max(1, n_show // 10)
        sel = np.arange(0, n_show, step)
        ax.set_xticks(sel)
        ax.set_xticklabels(
            [channel_names[channel_order[i]] for i in sel],
            rotation=45, ha="right", fontsize=FS_TICK - 2,
        )

    # -----------------------------------------------------------------------
    # Panel d: MDS scatter — template-matching audit (subsample)
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    style_panel(ax, label="d", label_x=-0.10, label_y=1.04)
    ax.set_facecolor("#FAFAFA")
    n_mds_visible, n_mds_bins = _scatter_metric_panel(
        ax, events, templates, chosen_k, x_field="mds_x", y_field="mds_y",
    )
    ax.set_xlabel("MDS-1 (template-matching metric)", fontsize=FS_LABEL)
    ax.set_ylabel("MDS-2", fontsize=FS_LABEL)
    title_d = (
        f"MDS · {n_mds_visible:,} subsample events  "
        f"(sil_med={sil_med:.3f}  stress={stress:.2f})"
    )
    if data.get("stress_warning") or data.get("imputation_warning"):
        title_d += "  ⚠"
    if data.get("subsampled"):
        title_d += f"\n(subsampled from {n_total:,})"
    ax.set_title(title_d, fontsize=FS_TITLE - 4, loc="left")

    validity = _load_pr2_validity(dataset, subject)
    validity_subtitle = _validity_subtitle(validity)
    fig.suptitle(
        f"{dataset.capitalize()} · {subject} — cluster geometry "
        f"(KMeans-native PCA + template-matching MDS audit)\n"
        f"{validity_subtitle}",
        fontsize=FS_TITLE,
        y=0.997,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return savefig_pub(fig, output_path)


# ---------------------------------------------------------------------------
# Cohort 2x2 summary figure
# ---------------------------------------------------------------------------


def plot_cohort_summary(
    cohort: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Produce the 2x2 cohort summary described in §5.2."""
    per_subject = cohort.get("per_subject", {})
    if not per_subject:
        logger.warning("Cohort summary has no included subjects; skip figure")
        return output_path

    # Build per-subject record list
    records: List[Dict[str, Any]] = []
    for key, v in per_subject.items():
        if "/" in key:
            ds, sub = key.split("/", 1)
        else:
            ds, sub = "unknown", key
        records.append(
            {
                "key": key,
                "dataset": ds,
                "subject": sub,
                "silhouette": float(v["silhouette_median"]) if v["silhouette_median"] is not None else float("nan"),
                "agreement": float(v["agreement_overall"]) if v["agreement_overall"] is not None else float("nan"),
                "stable_k": int(v["stable_k"]),
                "n_events": int(v["n_events"]),
                "stress": float(v["stress"]) if v["stress"] is not None else float("nan"),
            }
        )

    fig, axes = new_figure(nrows=2, ncols=2, figsize=(16, 12))

    # ------- Panel a: per-subject silhouette ranked bar ---------------
    ax = axes[0, 0]
    style_panel(ax, label="a", label_x=-0.10, label_y=1.04)
    rec_a = sorted(records, key=lambda r: -r["silhouette"] if np.isfinite(r["silhouette"]) else 0.0)
    x = np.arange(len(rec_a))
    cols = [dataset_color(r["dataset"]) for r in rec_a]
    ax.bar(x, [r["silhouette"] for r in rec_a], color=cols, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset'][:1].upper()}:{r['subject'][:8]}" for r in rec_a],
                       rotation=70, ha="right", fontsize=FS_TICK - 4)
    ax.set_ylabel("silhouette median (template-matching)", fontsize=FS_LABEL)
    ax.set_title("per-subject cluster validity", fontsize=FS_TITLE - 1)
    ax.set_ylim(-0.1, 1.0)

    # ------- Panel b: per-subject agreement ranked bar ---------------
    ax = axes[0, 1]
    style_panel(ax, label="b", label_x=-0.10, label_y=1.04)
    rec_b = sorted(records, key=lambda r: -r["agreement"] if np.isfinite(r["agreement"]) else 0.0)
    x = np.arange(len(rec_b))
    cols = [
        COL_SIG if r["agreement"] < 0.85 else dataset_color(r["dataset"])
        for r in rec_b
    ]
    ax.bar(x, [r["agreement"] for r in rec_b], color=cols, edgecolor="white", linewidth=0.5)
    ax.axhline(0.85, color=COL_SIG, lw=1.0, ls="--", alpha=0.6,
               label="< 0.85 highlighted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset'][:1].upper()}:{r['subject'][:8]}" for r in rec_b],
                       rotation=70, ha="right", fontsize=FS_TICK - 4)
    ax.set_ylabel("KMeans-vs-template-matching agreement", fontsize=FS_LABEL)
    ax.set_title("audit: label agreement under unified metric", fontsize=FS_TITLE - 1)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=FS_TICK - 2, frameon=False)

    # ------- Panel c: joint silhouette vs agreement scatter -----------
    ax = axes[1, 0]
    style_panel(ax, label="c", label_x=-0.10, label_y=1.04)
    sil_arr = np.array([r["silhouette"] for r in records])
    agr_arr = np.array([r["agreement"] for r in records])
    valid = np.isfinite(sil_arr) & np.isfinite(agr_arr)
    if int(valid.sum()) >= 3:
        rho, pval = spearmanr(sil_arr[valid], agr_arr[valid])
    else:
        rho, pval = float("nan"), float("nan")

    marker_for_k = {2: "o", 4: "^", 6: "s"}
    for r in records:
        marker = marker_for_k.get(r["stable_k"], "D")
        col = dataset_color(r["dataset"])
        ax.scatter(r["silhouette"], r["agreement"],
                   s=110, marker=marker, c=col,
                   edgecolors="black", linewidths=0.8, alpha=0.85)

    ax.set_xlabel("silhouette median", fontsize=FS_LABEL)
    ax.set_ylabel("KMeans-matching agreement", fontsize=FS_LABEL)
    ax.set_title(
        f"joint relationship  Spearman ρ = {rho:.3f} (p = {pval:.3g})",
        fontsize=FS_TITLE - 1,
    )

    # Legend for shape
    handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="",
                   markersize=8, label="k=2"),
        plt.Line2D([0], [0], marker="^", color="black", linestyle="",
                   markersize=8, label="k=4"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="",
                   markersize=8, label="k=6"),
        plt.Line2D([0], [0], marker="o", color=COL_YUQUAN, linestyle="",
                   markersize=8, label="Yuquan"),
        plt.Line2D([0], [0], marker="o", color=COL_EPILEPSIAE, linestyle="",
                   markersize=8, label="Epilepsiae"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=FS_TICK - 2, frameon=False, ncol=2)

    # ------- Panel d: boundary fraction by n_participating bin --------
    ax = axes[1, 1]
    style_panel(ax, label="d", label_x=-0.10, label_y=1.04)
    bin_keys = ["3-4", "5-8", "9+"]
    boundary_data = cohort.get("boundary_fraction_by_nparticipating", {})
    bin_vals: List[List[float]] = [
        [v for v in boundary_data.get(b, []) if v is not None and np.isfinite(v)]
        for b in bin_keys
    ]
    positions = np.arange(len(bin_keys))
    rng = np.random.default_rng(42)
    for i, vals in enumerate(bin_vals):
        if not vals:
            continue
        vals_arr = np.array(vals)
        # Box / scatter
        bp = ax.boxplot(
            vals_arr, positions=[positions[i]], widths=0.5,
            patch_artist=True, showfliers=False, showcaps=False,
            medianprops=dict(linewidth=2.2, color="black"),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COL_NEUTRAL)
            patch.set_alpha(0.4)
            patch.set_edgecolor("black")
        jit = positions[i] + rng.normal(0, 0.06, vals_arr.size)
        ax.scatter(jit, vals_arr, s=55, c=COL_NIGHT, edgecolors="white",
                   linewidths=0.8, alpha=0.85, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"n_part {b}" for b in bin_keys], fontsize=FS_TICK)
    ax.set_ylabel("boundary-event fraction (per subject)", fontsize=FS_LABEL)
    ax.set_title("metric drift by n_participating", fontsize=FS_TITLE - 1)
    ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))

    fig.suptitle(
        f"Cluster geometry cohort summary  "
        f"(included={cohort.get('n_subjects_included', '?')}, "
        f"excluded={cohort.get('n_subjects_excluded', '?')})",
        fontsize=FS_TITLE + 1, y=0.998,
    )
    fig.tight_layout()
    return savefig_pub(fig, output_path)


# ---------------------------------------------------------------------------
# Showcase selector
# ---------------------------------------------------------------------------


def _select_low_agreement_showcase(
    cohort: Dict[str, Any],
    min_n_events: int = 200,
    target_subjects_to_avoid: Optional[List[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Pick a non-trivial low-agreement subject for the showcase trio."""
    target_subjects_to_avoid = set(target_subjects_to_avoid or [])
    candidates: List[Tuple[float, str, str]] = []
    for key, v in cohort.get("per_subject", {}).items():
        if "/" not in key:
            continue
        ds, sub = key.split("/", 1)
        if sub in target_subjects_to_avoid:
            continue
        if v.get("n_events", 0) < min_n_events:
            continue
        agr = v.get("agreement_overall", 1.0)
        if agr is None or not np.isfinite(agr):
            continue
        if agr >= 0.85:
            continue
        sil = v.get("silhouette_median", 0.0) or 0.0
        candidates.append((sil, ds, sub))
    if not candidates:
        return None
    candidates.sort()  # ascending silhouette
    return (candidates[0][1], candidates[0][2])


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------


def _write_figures_readme(showcase_picks: Optional[List[Tuple[str, str]]] = None) -> None:
    readme = FIGURES_DIR / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)

    showcase_lines = []
    for ds, sub in (showcase_picks or []):
        showcase_lines.append(f"- `{sub}_geometry_showcase.png` ({ds})")
    showcase_block = "\n".join(showcase_lines) if showcase_lines else "（cohort 跑完后由 plot 脚本动态决定）"

    text = f"""# 间期事件 cluster geometry 可视化

> 设计文档：`docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md`
> 结果文档：`docs/archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md`
>
> 每张 per-subject 图同时展示两种距离 metric 下的几何：
> - **PCA**（panel a）= sklearn PCA on KMeans 训练用的全 rank feature matrix；KMeans 的"原生"距离视图，**所有事件**展示
> - **MDS**（panel d）= classical MDS on `assign_events_to_templates` 的 shared-channel mean squared deviation；template-matching 距离的 audit 视图，subsample 到 8000 events
>
> 注意：lagPatRank 在非参与通道上**不是 NaN/0**，而是 legacy `return_massCenterPat` 给出的有限 fallback rank。KMeans 用全 rank 向量算欧氏距离；matching 只用共享通道。

## per_subject/

每 subject 一张 2×2 图（status=ok 的 subject 才出图）。

### `<dataset>_<subject>_geometry.png`

- **Panel a (top-left, KMeans-native PCA)**：PCA(2) on 全 rank feature matrix；事件按 PR-2 KMeans label 上色；模板用大星黑边突出；KMeans label 与 template-matching reassign 不一致的事件叠加空心圆。**所有事件**展示，title 写 `EVR=...,...`（前两 PC 解释方差比）。
- **Panel b (top-right, silhouette ranked)**：每事件 silhouette（template-matching metric 下）排序条。按 cluster 分块、cluster 内按 silhouette 降序。silhouette < 0 的事件用 Morandi rust 高亮——KMeans 归到本簇，但在 template-matching metric 下离对家更近。
- **Panel c (bottom-left, template profile)**：横轴 = 通道（按 dominant cluster 的 template rank 升序），纵轴 = template mean rank；每 cluster 一条折线 + 半透明 IQR 带。直接展示每个传播模式的"哪些通道早 / 哪些通道晚"，是 panel a/d 的 2D 视图无法替代的结构层信息。
- **Panel d (bottom-right, template-matching MDS audit)**：MDS scatter，subsample 到 8000 events + 全部模板；事件颜色与 panel a 一致；title 写 `sil_med=... stress=...`。stress > 0.30 或 imputation > 0.20 挂 ⚠️。

**关注点（per-subject）**：
- Panel a / d 的事件云分离形态在 KMeans-native vs matching 两种 metric 下是否一致
- Panel b 的 silhouette 分布是否大部分为正；负 silhouette 比例 = boundary-event 比例
- Panel c 两条 template 折线是否反向（forward/reverse）或近重合（identity-dominated）
- Panel d 的 stress：> 0.30 表示 2D 不能完全保留 metric 距离结构（k≥3 多 cluster 时常见，不是 cluster validity 问题）

## showcase/

3 张精修大图，与 per_subject 同 4-panel 结构、figsize 加大：

{showcase_block}

> 第一张通常是 `958`（k=2 forward/reverse 案例）；第二张是多模态例子（按 fallback chain 从 huangwanling 开始挑第一个 status=ok 的）；第三张由 plot 脚本从 cohort 中 auto-pick `agreement < 0.85 ∩ n_events ≥ 200` 的 silhouette-最低 subject。

**关注点（showcase）**：每张图通过 panel c 直接看该 subject 的"传播模式骨架"，通过 panel a vs d 看两种 metric 的几何是否一致。

## cohort_geometry_summary.png

一张 2×2 大图。

- **Panel a**：per-subject silhouette median 排序条；YQ 蓝、EPI 赭，stable_k 用形状区分（圆 / 三角 / 方）
- **Panel b**：per-subject KMeans-matching agreement 排序条；agreement < 0.85 用 Morandi rust 高亮，0.85 dashed line
- **Panel c (consistency check, 不是独立科学发现)**：silhouette median vs agreement joint scatter；Spearman ρ + p 在 title；marker shape = stable_k，color = dataset。**注**：silhouette 与 agreement 都源自同一组 d_within / d_min_other 距离差，cohort 上 ρ 强正本来就是定义上预期，**不**当作独立 finding，仅作 sanity check（反向相关 / 无关才需要担心 pipeline）。
- **Panel d**：boundary-event fraction（agreement = False 比例）按 n_participating bin（3-4 / 5-8 / 9+）做 box + scatter；cohort 每 subject 一个数据点

**关注点（cohort）**：
- Panel a：cohort silhouette 范围与中位（大致衡量 cluster 在 matching metric 下的分离强度）
- Panel b：低 agreement subject 数量（caveat 候选）
- Panel c：sanity（ρ 应正）
- Panel d：箱体若 3-4 显著高、9+ 显著低，metric drift 主要源于低 n_participating 事件——主要的机制候选
"""
    readme.write_text(text)
    logger.info("Wrote %s", readme)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-subject", action="store_true", default=False,
                        help="Plot per-subject 3-panel figures.")
    parser.add_argument("--cohort", action="store_true", default=False,
                        help="Plot cohort 2x2 summary.")
    parser.add_argument("--showcase", action="store_true", default=False,
                        help="Plot showcase trio (958, huangwanling, low-agreement).")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Equivalent to --per-subject --cohort --showcase.")
    parser.add_argument("--subjects", action="append", default=None,
                        help="Restrict per-subject plots to specific subjects.")
    args = parser.parse_args()

    if args.all:
        args.per_subject = True
        args.cohort = True
        args.showcase = True

    if not (args.per_subject or args.cohort or args.showcase):
        parser.error("must specify at least one of --per-subject / --cohort / --showcase / --all")

    PER_SUBJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SHOWCASE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    showcase_picks: List[Tuple[str, str]] = []

    files = _list_per_subject_files()
    if not files:
        logger.warning("No per-subject geometry JSONs found in %s — run scripts/run_cluster_geometry.py first", PER_SUBJECT_DIR)
        return

    if args.per_subject:
        for path in files:
            data = _load_subject_geometry(path)
            if data is None or data.get("status") != "ok":
                continue
            ds = data.get("dataset", "x")
            sub = data.get("subject", path.stem)
            if args.subjects and sub not in args.subjects:
                continue
            # Single-panel template-distance plane (default per user 2026-05-06 review)
            out = PER_SUBJECT_FIG_DIR / f"{ds}_{sub}_geometry.png"
            plot_template_distance_plane(data, out, is_showcase=False)

    if args.showcase:
        cohort = _load_cohort_summary()
        # Fixed showcase candidates by priority. Yuquan huangwanling is the
        # original multimodal (k=4) showcase but is currently excluded by
        # data-freshness; fall back to Epilepsiae 818 (also k=4 in archive).
        fixed_showcase: List[Tuple[str, str]] = [("epilepsiae", "958")]
        for cand in [("yuquan", "huangwanling"), ("epilepsiae", "818"),
                     ("epilepsiae", "1077"), ("epilepsiae", "zhangjinhan")]:
            json_path = PER_SUBJECT_DIR / f"{cand[0]}_{cand[1]}.json"
            if json_path.exists():
                data = _load_subject_geometry(json_path)
                if data and data.get("status") == "ok":
                    fixed_showcase.append(cand)
                    break

        if cohort:
            avoid = {sub for _, sub in fixed_showcase}
            low_pick = _select_low_agreement_showcase(
                cohort, target_subjects_to_avoid=list(avoid)
            )
            if low_pick:
                fixed_showcase.append(low_pick)

        for ds, sub in fixed_showcase:
            json_path = PER_SUBJECT_DIR / f"{ds}_{sub}.json"
            data = _load_subject_geometry(json_path)
            if data is None or data.get("status") != "ok":
                logger.warning("Showcase skip %s/%s: no geometry data", ds, sub)
                continue
            out = SHOWCASE_FIG_DIR / f"{sub}_geometry_showcase.png"
            plot_template_distance_plane(data, out, is_showcase=True)
            showcase_picks.append((ds, sub))

    if args.cohort:
        cohort = _load_cohort_summary()
        if cohort is None:
            logger.warning("No cohort_summary.json found at %s", GEOMETRY_DIR)
        else:
            out = FIGURES_DIR / "cohort_geometry_summary.png"
            plot_cohort_summary(cohort, out)

    _write_figures_readme(showcase_picks=showcase_picks)


if __name__ == "__main__":
    main()
