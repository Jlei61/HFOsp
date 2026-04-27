#!/usr/bin/env python3
"""PR-6 Template Pair Geometry — publication-quality figures.

Main figure (6 panels):
  A. Per-subject schematic of T0/T1 source/sink + node classes (one fwd/rev case)
  B. Split-half + odd-even endpoint Jaccard scatter (per subject)
  C. n_swap_node vs n_same_side_node scatter (subject-level paired test)
  D. Per-subject stacked bar of node anatomy (swap / same / specific)
  E. Subgroup comparison (forward/reverse vs non-fwdrev) on swap / same-side / Spearman
  F. SOZ frac in template_specific vs swap_node scatter (subject-level NULL)

Supplements:
  - Coreness sensitivity paired scatter
  - Forward/reverse small multiples (per-subject T0/T1 source-sink role view)
  - Per-subject T0 ∩ T1 endpoint Jaccard bar

Outputs to ``results/interictal_propagation/template_anchoring/figures/``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (  # noqa: E402
    style_panel,
    add_significance_bracket,
    savefig_pub,
    COL_YUQUAN,
    COL_EPILEPSIAE,
    COL_SIG,
    COL_NONSIG,
    COL_SOZ,
    COL_NONSOZ,
    COL_NEUTRAL,
    COL_DAY,
    COL_NIGHT,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
)

RESULTS_DIR = Path("results/interictal_propagation/template_anchoring")
FIG_DIR = RESULTS_DIR / "figures"
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"

# Node class colors
COL_SWAP = COL_SIG                # rust — the headline
COL_SAME = "#7E6E84"              # plum (COL_NIGHT) — competing geometry
COL_TSPEC = "#C9A86A"             # mustard — template-specific
COL_SHARED_UN = "#B5A99B"         # warm dust
COL_NONEP = "#D6D2CC"             # very light gray

# Subject-level paired-test color
COL_FWDREV = "#A35E48"            # rust accent (matches COL_SWAP)
COL_NONFWD = "#6F8FA8"            # blue (matches COL_YUQUAN)

NODE_CLASS_LABELS = {
    "swap_node": "swap",
    "same_side_node": "same-side",
    "template_specific_endpoint": "template-specific",
    "shared_endpoint_unassigned": "shared (unassigned)",
    "non_endpoint": "non-endpoint",
}

NODE_CLASS_COLORS = {
    "swap_node": COL_SWAP,
    "same_side_node": COL_SAME,
    "template_specific_endpoint": COL_TSPEC,
    "shared_endpoint_unassigned": COL_SHARED_UN,
    "non_endpoint": COL_NONEP,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cohort() -> Dict[str, Any]:
    path = RESULTS_DIR / "cohort_summary.json"
    with open(path) as f:
        return json.load(f)


def load_subject(subject_id: str, dataset: str) -> Optional[Dict[str, Any]]:
    fp = PER_SUBJECT_DIR / f"{dataset}_{subject_id}.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def _short_subject_id(subj: str) -> str:
    if len(subj) <= 8:
        return subj
    return subj[:7] + "."


# ---------------------------------------------------------------------------
# Panel A — schematic
# ---------------------------------------------------------------------------

def panel_a_schematic(
    ax: plt.Axes,
    cohort: Dict[str, Any],
    example_subject_id: str = "635",
    example_dataset: str = "epilepsiae",
) -> None:
    rec = load_subject(example_subject_id, example_dataset)
    if rec is None:
        ax.text(0.5, 0.5, f"per-subject JSON not found:\n{example_dataset}_{example_subject_id}",
                transform=ax.transAxes, ha="center", va="center", fontsize=FS_LABEL)
        style_panel(ax, label="a")
        return

    pt = rec["per_template"]
    if len(pt) < 2:
        ax.text(0.5, 0.5, "needs 2 templates", transform=ax.transAxes, ha="center", va="center")
        style_panel(ax, label="a")
        return

    classification = rec["node_anatomy"]["classification"]["per_channel"]
    cls_by_ch = {c["channel"]: c["class"] for c in classification}

    # union of channels in either source/sink/middle
    chs: List[str] = []
    for r in pt[:2]:
        for c in (r.get("source") or []) + (r.get("sink") or []) + (r.get("middle") or []):
            if c not in chs:
                chs.append(c)

    def role(ch: str, tmpl: Dict[str, Any]) -> str:
        if ch in (tmpl.get("source") or []):
            return "source"
        if ch in (tmpl.get("sink") or []):
            return "sink"
        if ch in (tmpl.get("middle") or []):
            return "middle"
        return "absent"

    # sort by node class so swap nodes come first
    class_order = ["swap_node", "same_side_node", "template_specific_endpoint",
                   "shared_endpoint_unassigned", "non_endpoint"]
    chs.sort(key=lambda c: (class_order.index(cls_by_ch.get(c, "non_endpoint")), c))

    n = len(chs)

    role_colors = {"source": "#C97B5F", "sink": "#5A7E9F",
                   "middle": "#D6D2CC", "absent": "#F2F0EE"}

    cell_w = 0.55
    x_t0 = 0.0
    x_t1 = 1.4

    for i, ch in enumerate(chs):
        for x_pos, tmpl in [(x_t0, pt[0]), (x_t1, pt[1])]:
            r = role(ch, tmpl)
            rect = mpatches.FancyBboxPatch(
                (x_pos - cell_w / 2, n - 1 - i - 0.36),
                cell_w, 0.72,
                boxstyle="round,pad=0.005,rounding_size=0.05",
                linewidth=0.8, edgecolor="#777777",
                facecolor=role_colors[r],
            )
            ax.add_patch(rect)

        cls = cls_by_ch.get(ch, "non_endpoint")
        ax.plot([x_t0 + cell_w / 2, x_t1 - cell_w / 2],
                [n - 1 - i, n - 1 - i],
                color=NODE_CLASS_COLORS[cls],
                lw=2.6, alpha=0.9, zorder=1, solid_capstyle="round")

        ax.text(-0.55, n - 1 - i, ch, ha="right", va="center",
                fontsize=10, color="#333333", family="monospace")

    ax.text(x_t0, n + 0.05, "T0", ha="center", va="bottom",
            fontsize=FS_LABEL, fontweight="bold")
    ax.text(x_t1, n + 0.05, "T1", ha="center", va="bottom",
            fontsize=FS_LABEL, fontweight="bold")

    role_legend = [
        mpatches.Patch(facecolor=role_colors["source"], edgecolor="#777", label="source"),
        mpatches.Patch(facecolor=role_colors["sink"], edgecolor="#777", label="sink"),
        mpatches.Patch(facecolor=role_colors["middle"], edgecolor="#777", label="middle"),
    ]
    cls_legend = [
        Line2D([0], [0], color=NODE_CLASS_COLORS[c], lw=3, label=NODE_CLASS_LABELS[c])
        for c in ["swap_node", "template_specific_endpoint", "non_endpoint"]
    ]
    leg1 = ax.legend(handles=role_legend, loc="upper right",
                     bbox_to_anchor=(0.99, 0.99), fontsize=FS_TICK - 2,
                     frameon=False, title="role",
                     title_fontsize=FS_TICK - 2,
                     handlelength=1.2, handleheight=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cls_legend, loc="lower right",
              bbox_to_anchor=(0.99, 0.01), fontsize=FS_TICK - 2,
              frameon=False, title="node class",
              title_fontsize=FS_TICK - 2,
              handlelength=2.0)

    ax.set_xlim(-2.0, 3.0)
    ax.set_ylim(-0.7, n + 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        f"example: {example_dataset}:{example_subject_id} (fwd/rev reproduced)",
        fontsize=FS_LABEL, pad=8,
    )
    style_panel(ax, label="a", label_x=-0.05, label_y=1.03)


# ---------------------------------------------------------------------------
# Panel B — split-half + odd-even endpoint Jaccard
# ---------------------------------------------------------------------------

def panel_b_jaccard(ax: plt.Axes, cohort: Dict[str, Any]) -> None:
    sh = cohort["split_half_endpoint_robustness"]
    fh_rows = {r["subject_id"]: r for r in sh["first_half_second_half"]["per_subject"]}
    oe_rows = {r["subject_id"]: r for r in sh["odd_even_block"]["per_subject"]}

    # use template_pair_geometry per_subject for h1_eligible + fwdrev flags
    geom = {r["subject_id"]: r for r in cohort["template_pair_geometry"]["per_subject"]}

    common = sorted(set(fh_rows) & set(oe_rows))
    fh_med = sh["first_half_second_half"]["median_jaccard_endpoint"]
    oe_med = sh["odd_even_block"]["median_jaccard_endpoint"]

    fwd_x, fwd_y, non_x, non_y = [], [], [], []
    for s in common:
        x = fh_rows[s].get("mean_jaccard_endpoint")
        y = oe_rows[s].get("mean_jaccard_endpoint")
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        is_fwd = bool(geom.get(s, {}).get("forward_reverse_reproduced"))
        if is_fwd:
            fwd_x.append(x); fwd_y.append(y)
        else:
            non_x.append(x); non_y.append(y)

    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.4, color="#cccccc", lw=0.8, ls=":", zorder=1)
    ax.axvline(0.4, color="#cccccc", lw=0.8, ls=":", zorder=1)

    ax.scatter(non_x, non_y, s=70, c=COL_NONFWD, edgecolors="white",
               linewidths=1.0, alpha=0.85, label="non-fwd/rev", zorder=3)
    ax.scatter(fwd_x, fwd_y, s=110, c=COL_FWDREV, edgecolors="black",
               linewidths=1.0, alpha=0.95, marker="D", label="fwd/rev", zorder=4)

    ax.axvline(fh_med, color=COL_NEUTRAL, lw=1.0, ls="-", alpha=0.55)
    ax.axhline(oe_med, color=COL_NEUTRAL, lw=1.0, ls="-", alpha=0.55)
    ax.text(0.04, 0.96,
            f"split-half median {fh_med:.2f}\nodd-even median {oe_med:.2f}",
            transform=ax.transAxes,
            fontsize=FS_TICK - 2, color="#555", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("split-half endpoint Jaccard", fontsize=FS_LABEL)
    ax.set_ylabel("odd-even endpoint Jaccard", fontsize=FS_LABEL)
    ax.set_title("endpoint set is stable in time", fontsize=FS_LABEL, pad=4)
    ax.legend(loc="lower right", fontsize=FS_TICK - 2, frameon=False)
    style_panel(ax, label="b")


# ---------------------------------------------------------------------------
# Panel C — n_swap vs n_same_side
# ---------------------------------------------------------------------------

def panel_c_swap_vs_same(ax: plt.Axes, cohort: Dict[str, Any]) -> None:
    rows = cohort["node_anatomy"]["per_subject"]
    test = cohort["node_anatomy"]["h1_eligible"]["subject_level_swap_minus_same"]

    fwd_x, fwd_y, non_x, non_y = [], [], [], []
    for r in rows:
        if not r.get("h1_eligible"):
            continue
        x = int(r["n_same_side_node"])
        y = int(r["n_swap_node"])
        if r.get("forward_reverse_reproduced"):
            fwd_x.append(x); fwd_y.append(y)
        else:
            non_x.append(x); non_y.append(y)

    all_x = fwd_x + non_x
    all_y = fwd_y + non_y
    if not all_x:
        return
    lim = max(max(all_x), max(all_y)) + 1
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=1.0, ls="--", zorder=1)

    # jitter same coordinates so overlapping points show
    rng = np.random.default_rng(0)
    def jit(arr): return np.array(arr) + rng.normal(0, 0.12, len(arr))

    ax.scatter(jit(non_x), jit(non_y), s=80, c=COL_NONFWD, edgecolors="white",
               linewidths=1.0, alpha=0.85, label="non-fwd/rev", zorder=3)
    ax.scatter(jit(fwd_x), jit(fwd_y), s=120, c=COL_FWDREV, edgecolors="black",
               linewidths=1.0, marker="D", alpha=0.95, label="fwd/rev", zorder=4)

    p_w = test["wilcoxon_greater"]["p_value"]
    p_s = test["sign_test"]["p_value"]
    np_, nn_, nz_ = test["n_positive"], test["n_negative"], test["n_zero"]

    txt = (f"H1-eligible (n={test['n']})\n"
           f"swap > same: {np_}p / {nn_}n / {nz_}z\n"
           f"Wilcoxon p={p_w:.3f}\n"
           f"sign-test p={p_s:.3f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=FS_TICK - 2, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92))

    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_xlabel("# same-side node", fontsize=FS_LABEL)
    ax.set_ylabel("# swap node", fontsize=FS_LABEL)
    ax.set_title("swap vs same-side node count", fontsize=FS_LABEL, pad=4)
    ax.legend(loc="lower right", fontsize=FS_TICK - 2, frameon=False)
    style_panel(ax, label="c")


# ---------------------------------------------------------------------------
# Panel D — stacked bar per subject
# ---------------------------------------------------------------------------

def panel_d_stacked(ax: plt.Axes, cohort: Dict[str, Any]) -> None:
    rows = [r for r in cohort["node_anatomy"]["per_subject"] if r.get("h1_eligible")]
    if not rows:
        return

    # sort by swap_minus_same descending
    rows.sort(key=lambda r: int(r["swap_minus_same_side_count"]), reverse=True)

    n = len(rows)
    x = np.arange(n)

    classes = ["swap_node", "same_side_node", "template_specific_endpoint",
               "shared_endpoint_unassigned"]
    colors = [NODE_CLASS_COLORS[c] for c in classes]
    labels = [NODE_CLASS_LABELS[c] for c in classes]

    bottom = np.zeros(n)
    for cls, color, label in zip(classes, colors, labels):
        vals = np.array([int(r.get(f"n_{cls}", 0)) for r in rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white",
               linewidth=0.6, label=label, zorder=2)
        bottom += vals

    # forward/reverse markers above bars
    for i, r in enumerate(rows):
        if r.get("forward_reverse_reproduced"):
            ax.text(i, bottom[i] + 0.15, "F/R", ha="center", va="bottom",
                    fontsize=FS_TICK - 4, color=COL_FWDREV, fontweight="bold")

    # x-tick labels
    labels_x = [f"{r['dataset'][0].upper()}:{_short_subject_id(r['subject_id'])}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=55, ha="right", fontsize=FS_TICK - 3)

    ax.set_ylabel("# channels", fontsize=FS_LABEL)
    ax.set_title("per-subject node anatomy (sorted by n_swap − n_same_side)",
                 fontsize=FS_LABEL, pad=6)
    ymax = max(bottom) * 1.30
    ax.legend(loc="upper left", fontsize=FS_TICK - 2,
              frameon=True, fancybox=True, framealpha=0.92,
              edgecolor="#cccccc", ncol=2,
              handlelength=1.4, columnspacing=1.2,
              borderpad=0.4)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)
    style_panel(ax, label="d")


# ---------------------------------------------------------------------------
# Panel E — subgroup violin
# ---------------------------------------------------------------------------

def _violin_pair(ax: plt.Axes, vals_a: np.ndarray, vals_b: np.ndarray,
                 pos_a: float, pos_b: float, color_a: str, color_b: str,
                 width: float = 0.6) -> None:
    for vals, pos, color in [(vals_a, pos_a, color_a), (vals_b, pos_b, color_b)]:
        if vals.size == 0:
            continue
        if vals.size >= 2 and np.std(vals) > 1e-12:
            vp = ax.violinplot(vals, positions=[pos], widths=width,
                               showmeans=False, showmedians=False,
                               showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.22)
                pc.set_edgecolor(color)
                pc.set_linewidth(1.0)
        # boxplot
        bp = ax.boxplot(vals, positions=[pos], widths=width * 0.4,
                        patch_artist=True, showfliers=False, showcaps=False,
                        medianprops=dict(linewidth=2.2, color="black"),
                        whiskerprops=dict(linewidth=1.0, color="black"),
                        zorder=2)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.9)
        rng = np.random.default_rng(abs(int(pos * 11)) + 1)
        jit = np.full(vals.size, pos) + rng.normal(0, 0.05, vals.size)
        ax.scatter(jit, vals, s=42, c=color, edgecolors="white",
                   linewidths=0.7, zorder=3, alpha=0.9)


def panel_e_subgroup(ax: plt.Axes, cohort: Dict[str, Any]) -> None:
    rows = [r for r in cohort["template_pair_geometry"]["per_subject"]
            if r.get("h1_eligible")]
    if not rows:
        return

    fwd = [r for r in rows if r.get("forward_reverse_reproduced")]
    non = [r for r in rows if not r.get("forward_reverse_reproduced")]

    metrics = [
        ("swap_score", "swap score"),
        ("same_side_score", "same-side score"),
        ("spearman_rank_pair", "Spearman ρ(T0,T1)"),
    ]

    base_x = np.arange(len(metrics)) * 3.0

    for i, (key, _) in enumerate(metrics):
        v_fwd = np.array([r[key] for r in fwd if r.get(key) is not None
                         and not (isinstance(r[key], float) and math.isnan(r[key]))])
        v_non = np.array([r[key] for r in non if r.get(key) is not None
                         and not (isinstance(r[key], float) and math.isnan(r[key]))])
        _violin_pair(ax, v_fwd, v_non,
                     base_x[i] - 0.55, base_x[i] + 0.55,
                     COL_FWDREV, COL_NONFWD, width=1.0)

    ax.axhline(0.0, color="#cccccc", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(base_x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=FS_TICK - 1)
    ax.set_ylabel("score", fontsize=FS_LABEL)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"forward/reverse (n={len(fwd)}) vs non-fwd/rev (n={len(non)})",
                 fontsize=FS_LABEL, pad=4)

    legend = [
        mpatches.Patch(color=COL_FWDREV, alpha=0.7, label="forward/reverse"),
        mpatches.Patch(color=COL_NONFWD, alpha=0.7, label="non-forward/reverse"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=FS_TICK - 2, frameon=False)
    style_panel(ax, label="e")


# ---------------------------------------------------------------------------
# Panel F — SOZ frac scatter
# ---------------------------------------------------------------------------

def panel_f_soz(ax: plt.Axes, cohort: Dict[str, Any]) -> None:
    paired = cohort["node_anatomy"]["h1_eligible"]["subject_level_soz_swap_vs_template_specific"]
    per = paired.get("per_subject") or []

    geom = {r["subject_id"]: r for r in cohort["template_pair_geometry"]["per_subject"]}

    fwd_x, fwd_y, non_x, non_y = [], [], [], []
    for r in per:
        x = r["frac_soz_template_specific"]
        y = r["frac_soz_swap"]
        if (x is None or y is None
                or (isinstance(x, float) and math.isnan(x))
                or (isinstance(y, float) and math.isnan(y))):
            continue
        sid = r["subject_id"]
        is_fwd = bool(geom.get(sid, {}).get("forward_reverse_reproduced"))
        if is_fwd:
            fwd_x.append(x); fwd_y.append(y)
        else:
            non_x.append(x); non_y.append(y)

    rng = np.random.default_rng(1)
    def jit(arr, scale=0.012): return np.array(arr) + rng.normal(0, scale, len(arr))

    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls="--", zorder=1)

    ax.scatter(jit(non_x), jit(non_y), s=80, c=COL_NONFWD, edgecolors="white",
               linewidths=1.0, alpha=0.85, label="non-fwd/rev", zorder=3)
    ax.scatter(jit(fwd_x), jit(fwd_y), s=120, c=COL_FWDREV, edgecolors="black",
               linewidths=1.0, marker="D", alpha=0.95, label="fwd/rev", zorder=4)

    n = paired["n"]
    med = paired["median_delta"]
    p_w = paired["wilcoxon_greater"]["p_value"]
    p_s = paired["sign_test"]["p_value"]
    np_, nn_, nz_ = paired["n_positive"], paired["n_negative"], paired["n_zero"]
    txt = (f"H1-eligible (n={n})\n"
           f"swap > t-spec: {np_}p / {nn_}n / {nz_}tie\n"
           f"median Δ = {med:+.2f}\n"
           f"Wilcoxon p={p_w:.2f}\n"
           f"sign-test p={p_s:.2f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=FS_TICK - 2, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("SOZ frac in template-specific endpoint", fontsize=FS_LABEL)
    ax.set_ylabel("SOZ frac in swap node", fontsize=FS_LABEL)
    ax.set_title("SOZ does not explain swap geometry", fontsize=FS_LABEL, pad=4)
    ax.legend(loc="lower right", fontsize=FS_TICK - 2, frameon=False)
    style_panel(ax, label="f")


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def make_main_figure(cohort: Dict[str, Any], out_path: Path,
                     example_subject: str = "635",
                     example_dataset: str = "epilepsiae") -> Path:
    fig = plt.figure(figsize=(15, 22))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[1.10, 1.00, 1.05, 1.30],
        width_ratios=[1.0, 1.0],
        hspace=0.72, wspace=0.32,
        left=0.07, right=0.96, top=0.96, bottom=0.045,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])
    ax_e = fig.add_subplot(gs[3, :])

    panel_a_schematic(ax_a, cohort, example_subject, example_dataset)
    panel_b_jaccard(ax_b, cohort)
    panel_c_swap_vs_same(ax_c, cohort)
    panel_f_soz(ax_f, cohort)
    panel_d_stacked(ax_d, cohort)
    panel_e_subgroup(ax_e, cohort)

    fig.suptitle(
        "PR-6 Template Pair Geometry: stable templates have structured but "
        "SOZ-independent role swapping",
        fontsize=FS_TITLE + 1, y=0.985, fontweight="bold",
    )

    return savefig_pub(fig, out_path)


# ---------------------------------------------------------------------------
# Supplements
# ---------------------------------------------------------------------------

def make_coreness_sensitivity(cohort: Dict[str, Any], out_path: Path) -> Path:
    pairs = cohort["h1_coreness_sensitivity"]["per_subject_pairs"]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    fig.patch.set_facecolor("white")

    same = [(p["delta_main"], p["delta_coreness"]) for p in pairs if p["same_sign"]]
    disc = [(p["delta_main"], p["delta_coreness"]) for p in pairs if p["direction_discordant"]]
    zero = [(p["delta_main"], p["delta_coreness"]) for p in pairs if p["one_is_zero"]]

    ax.axhline(0.0, color="#cccccc", lw=0.8, ls=":")
    ax.axvline(0.0, color="#cccccc", lw=0.8, ls=":")
    ax.plot([-1, 1], [-1, 1], color="#bbbbbb", lw=0.8, ls="--")

    if same:
        x, y = zip(*same)
        ax.scatter(x, y, s=90, c=COL_NONSIG, edgecolors="black", linewidths=0.8,
                   alpha=0.9, label=f"same sign (n={len(same)})")
    if disc:
        x, y = zip(*disc)
        ax.scatter(x, y, s=110, c=COL_SIG, edgecolors="black", linewidths=0.8,
                   alpha=0.95, marker="X", label=f"direction-discordant (n={len(disc)})")
    if zero:
        x, y = zip(*zero)
        ax.scatter(x, y, s=110, c=COL_NEUTRAL, edgecolors="black", linewidths=0.8,
                   alpha=0.95, marker="P", label=f"one-is-zero (n={len(zero)})")

    cs = cohort["h1_coreness_sensitivity"]
    p_w = cs["wilcoxon_greater"]["p_value"]
    n = cs["n"]
    ax.text(0.04, 0.96,
            f"n={n}\nsame-sign={cs['n_same_sign_with_main']}\n"
            f"direction-discordant={cs['n_direction_discordant']}\n"
            f"one-is-zero={cs['n_one_is_zero']}\n"
            f"Wilcoxon p={p_w:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FS_TICK - 1, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92))

    ax.set_xlim(-0.65, 0.85)
    ax.set_ylim(-0.65, 0.85)
    ax.set_xlabel("delta endpoint vs middle (main definition, top-3 rank)", fontsize=FS_LABEL)
    ax.set_ylabel("delta endpoint vs middle (coreness, top-20%)", fontsize=FS_LABEL)
    ax.set_title("H1 endpoint definition is not robust under coreness top-20%",
                 fontsize=FS_LABEL, pad=4)
    ax.legend(loc="lower right", fontsize=FS_TICK - 1, frameon=False)
    style_panel(ax, label="")
    return savefig_pub(fig, out_path)


def make_jaccard_per_subject_bar(cohort: Dict[str, Any], out_path: Path) -> Path:
    geom = cohort["template_pair_geometry"]["per_subject"]
    rows = [r for r in geom if r.get("endpoint_defined")]
    rows.sort(key=lambda r: r.get("jaccard_endpoint") or 0.0, reverse=True)
    n = len(rows)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.32 * n + 4), 6))
    fig.patch.set_facecolor("white")

    x = np.arange(n)
    vals = np.array([r.get("jaccard_endpoint") or 0.0 for r in rows])
    colors = [COL_FWDREV if r.get("forward_reverse_reproduced") else COL_NONFWD for r in rows]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)

    ax.axhline(0.4, color="#cccccc", lw=0.8, ls=":")
    ax.axhline(np.median(vals), color="#888", lw=1.0, ls="-", alpha=0.6)
    ax.text(n - 0.5, np.median(vals) + 0.02, f"median {np.median(vals):.2f}",
            fontsize=FS_TICK - 2, color="#666", ha="right")

    labels_x = [f"{r['dataset'][0].upper()}:{_short_subject_id(r['subject_id'])}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=60, ha="right", fontsize=FS_TICK - 3)
    ax.set_ylabel("Jaccard(T0_endpoint, T1_endpoint)", fontsize=FS_LABEL)
    ax.set_title("per-subject T0 ∩ T1 endpoint Jaccard (full-data, all endpoint-defined)",
                 fontsize=FS_LABEL, pad=4)
    ax.set_ylim(0, 1.05)

    legend = [
        mpatches.Patch(color=COL_FWDREV, label="forward/reverse"),
        mpatches.Patch(color=COL_NONFWD, label="non-forward/reverse"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=FS_TICK - 2, frameon=False)
    style_panel(ax, label="")
    return savefig_pub(fig, out_path)


def make_fwdrev_small_multiples(cohort: Dict[str, Any], out_path: Path) -> Path:
    """One mini panel per fwd/rev subject showing T0 source-vs-T1 sink overlap."""
    fwd_rows = [r for r in cohort["template_pair_geometry"]["per_subject"]
                if r.get("forward_reverse_reproduced") and r.get("endpoint_defined")]
    fwd_rows.sort(key=lambda r: (r["dataset"], r["subject_id"]))
    n = len(fwd_rows)
    if n == 0:
        return out_path

    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    for idx, row in enumerate(fwd_rows):
        ax = axes[idx // ncols, idx % ncols]
        sid = row["subject_id"]
        ds = row["dataset"]
        sub = load_subject(sid, ds)
        if sub is None or len(sub.get("per_template", [])) < 2:
            ax.set_axis_off()
            continue
        pt = sub["per_template"]
        t0_src = set(pt[0].get("source") or [])
        t0_snk = set(pt[0].get("sink") or [])
        t1_src = set(pt[1].get("source") or [])
        t1_snk = set(pt[1].get("sink") or [])

        sets = [
            ("T0 src", t0_src),
            ("T1 snk", t1_snk),
            ("T0 snk", t0_snk),
            ("T1 src", t1_src),
        ]
        pos = [0, 1, 2.5, 3.5]
        colors = ["#C97B5F", "#5A7E9F", "#5A7E9F", "#C97B5F"]

        for label, (name, s), x_p, c in zip([s[0] for s in sets], sets, pos, colors):
            ax.scatter([x_p] * len(s), np.arange(len(s)), s=140,
                       c=c, edgecolors="black", linewidths=0.7, zorder=2)
            for j, ch in enumerate(sorted(s)):
                ax.text(x_p, j, ch, fontsize=8, ha="center", va="center",
                        color="white", fontweight="bold")
            ax.text(x_p, -1.0, name, ha="center", va="top",
                    fontsize=FS_TICK - 2, fontweight="bold")

        # connection: T0 src ↔ T1 snk overlap
        for ch in t0_src & t1_snk:
            j0 = sorted(t0_src).index(ch)
            j1 = sorted(t1_snk).index(ch)
            ax.plot([0, 1], [j0, j1], color=COL_SWAP, lw=2.0, alpha=0.7)
        for ch in t0_snk & t1_src:
            j0 = sorted(t0_snk).index(ch)
            j1 = sorted(t1_src).index(ch)
            ax.plot([2.5, 3.5], [j0, j1], color=COL_SWAP, lw=2.0, alpha=0.7)

        sw = row.get("swap_score", float("nan")) or 0.0
        ax.set_title(f"{ds[0].upper()}:{sid}\nswap score={sw:.2f}",
                     fontsize=FS_TICK)
        ax.set_xlim(-0.6, 4.1)
        ax.set_ylim(-1.7, max(8, max(len(t0_src), len(t1_snk), len(t0_snk), len(t1_src)) + 0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)

    # blank remaining axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_axis_off()

    fig.suptitle("Forward/reverse subjects: T0 src ↔ T1 snk role swap",
                 fontsize=FS_TITLE, y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig_pub(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="generate every figure")
    ap.add_argument("--main", action="store_true", help="6-panel main figure")
    ap.add_argument("--coreness", action="store_true", help="coreness sensitivity scatter")
    ap.add_argument("--per-subject-jaccard", action="store_true")
    ap.add_argument("--fwdrev-multiples", action="store_true")
    ap.add_argument("--example-subject", default="635")
    ap.add_argument("--example-dataset", default="epilepsiae")
    args = ap.parse_args()

    if not (args.all or args.main or args.coreness
            or args.per_subject_jaccard or args.fwdrev_multiples):
        args.all = True

    cohort = load_cohort()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.main:
        make_main_figure(
            cohort, FIG_DIR / "pr6_template_pair_geometry_main.png",
            example_subject=args.example_subject,
            example_dataset=args.example_dataset,
        )
    if args.all or args.coreness:
        make_coreness_sensitivity(
            cohort, FIG_DIR / "pr6_supp_coreness_sensitivity.png")
    if args.all or args.per_subject_jaccard:
        make_jaccard_per_subject_bar(
            cohort, FIG_DIR / "pr6_supp_endpoint_jaccard_per_subject.png")
    if args.all or args.fwdrev_multiples:
        make_fwdrev_small_multiples(
            cohort, FIG_DIR / "pr6_supp_fwdrev_small_multiples.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
