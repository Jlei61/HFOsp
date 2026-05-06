#!/usr/bin/env python3
"""Top-tier-journal-style supplementary figures for PR-6 rank displacement.

Plan: docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md

Produces 3 deliverables:
  1. cohort_displacement_heatmap.{png,pdf} — stable_k=2 cohort × channels,
     rows sorted by Kendall tau, columns sorted by rank_T_a_dense
     (T_a source -> sink), divergent RdBu palette, SOZ outlined.
  2. footrule_kendall_summary.{png,pdf} — 2-panel: footrule_normalized
     split by fwd/rev-reproduced flag; Kendall tau strip with reference lines.
  3. per_subject/<stem>_displacement.png — per-subject zoom-in heatstrip
     with channel labels, sorted by rank_T_a_dense (same anti-bias rule).

CRITICAL anti-bias rule (plan §0 禁区, Task 5 review fix):
    Columns of every heatmap row are sorted by rank_T_a_dense, NOT by Δr.
    Sorting by Δr would force any rank pair into a monotonic gradient,
    including random ones — that is circular sorting bias.

No statistical PASS gate. All annotations are descriptive.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE_ROOT))

from src.plot_style import (  # noqa: E402
    COL_NEUTRAL,
    COL_NONSIG,
    COL_SIG,
    DPI_PUB,
    FS_LABEL,
    FS_PANEL_LETTER,
    FS_TICK,
    FS_TITLE,
    style_panel,
)


def _canonical_data_root() -> Path:
    here = Path(__file__).resolve().parent
    common = subprocess.check_output(
        ["git", "-C", str(here), "rev-parse", "--git-common-dir"],
        text=True,
    ).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (here / common_path).resolve()
    return common_path.parent


DATA_ROOT = _canonical_data_root()
RES_DIR = DATA_ROOT / "results" / "interictal_propagation" / "rank_displacement"
PER_SUBJECT_DIR = RES_DIR / "per_subject"
FIG_DIR = RES_DIR / "figures"
PER_SUB_FIG_DIR = FIG_DIR / "per_subject"


CANDIDATE_RHO_THRESHOLD = -0.5  # PR-2.5 fwd/rev candidate gate on inter_cluster_corr_matrix

def _classify_pr25_status(record: dict) -> str:
    """Return one of 'reproduced' (TRUE), 'candidate_fail' (FALSE),
    'non_candidate' (None). Falls back to 'non_candidate' if PR-2.5
    fields are missing.

    The classification is based directly on PR-2.5 outputs - we do NOT
    re-derive the candidate threshold from inter_cluster_corr_matrix here,
    because PR-2.5 might use additional criteria beyond the bare ρ<-0.5 cut.
    """
    flag = record.get("fwd_rev_reproduced")
    if flag is True:
        return "reproduced"
    if flag is False:
        return "candidate_fail"
    return "non_candidate"


def _is_pr25_candidate(record: dict) -> bool:
    """Group A = PR-2.5 fwd/rev cohort (TRUE + FALSE = candidates that
    were tested). Group B = non-candidate (None)."""
    return _classify_pr25_status(record) in ("reproduced", "candidate_fail")


def load_cohort_records() -> List[dict]:
    """Load per-subject JSONs; only stable_k=2 with one valid pair."""
    records: List[dict] = []
    for path in sorted(PER_SUBJECT_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("stable_k") != 2:
            continue
        valid_pairs = [p for p in d.get("pairs", []) if p.get("exit_reason") == "ok"]
        if len(valid_pairs) != 1:
            continue
        d["primary_pair"] = valid_pairs[0]
        d["pr25_status"] = _classify_pr25_status(d)
        d["is_candidate"] = _is_pr25_candidate(d)
        records.append(d)
    return records


def sort_by_kendall_tau(records: List[dict]) -> List[dict]:
    return sorted(records, key=lambda r: r["primary_pair"].get("kendall_tau", 0.0))


def build_heatmap_matrix(
    records: List[dict],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """(subjects, max_n_valid) signed displacement matrix, NaN-padded.

    Columns within each row are arranged by rank_T_a_dense (T_a source first,
    sink last). NEVER sort by Δr.
    """
    sub_labels = [f"{r['dataset'][:3]}_{r['subject']}" for r in records]
    cached: List[Tuple[np.ndarray, np.ndarray]] = []
    max_n_valid = 0
    for r in records:
        pair = r["primary_pair"]
        delta = np.asarray(pair["signed_displacement_full"], dtype=float)
        joint = np.asarray(pair["joint_valid"], dtype=bool)
        soz_mask = np.asarray(
            pair.get("soz_mask", [False] * len(delta)), dtype=bool
        )
        rank_a_dense_full = np.asarray(pair["rank_a_dense_full"], dtype=float)
        valid_idx = np.where(joint)[0]
        if len(valid_idx) == 0:
            cached.append((np.array([]), np.array([], dtype=bool)))
            continue
        rank_a_dense_subset = rank_a_dense_full[valid_idx]
        order = np.argsort(rank_a_dense_subset)  # T_a source first → sink last
        delta_sorted = delta[valid_idx][order]
        soz_sorted = soz_mask[valid_idx][order]
        max_n_valid = max(max_n_valid, len(delta_sorted))
        cached.append((delta_sorted, soz_sorted))

    matrix = np.full((len(records), max_n_valid), np.nan)
    soz_overlay = np.zeros_like(matrix, dtype=bool)
    for i, (delta_sorted, soz_sorted) in enumerate(cached):
        n = len(delta_sorted)
        matrix[i, :n] = delta_sorted
        soz_overlay[i, :n] = soz_sorted
    return matrix, sub_labels, soz_overlay


def plot_cohort_heatmap(records: List[dict], out_stem: Path) -> None:
    sorted_records = sort_by_kendall_tau(records)
    matrix, _, soz_overlay = build_heatmap_matrix(sorted_records)
    sub_labels = [f"{r['dataset'][:3]}_{r['subject']}" for r in sorted_records]
    n_sub, n_ch = matrix.shape
    taus = np.array(
        [r["primary_pair"]["kendall_tau"] for r in sorted_records], dtype=float
    )
    f_norms = np.array(
        [r["primary_pair"]["footrule_normalized"] for r in sorted_records],
        dtype=float,
    )
    pr25_status = [r["pr25_status"] for r in sorted_records]
    # Color by F_norm > 2/3 grouping (NOT by PR-2.5 status — see Panel B/C
    # design: PR-2.5 ρ_inter < -0.5 is another PR's hard threshold).
    THRESH = 2 / 3

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(12, max(6.5, 0.42 * n_sub) + 1.2))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[8, 1.2, 0.35],
        wspace=0.10,
        top=0.86,
        bottom=0.10,
        left=0.10,
        right=0.94,
    )
    ax_h = fig.add_subplot(gs[0])
    ax_tau = fig.add_subplot(gs[1], sharey=ax_h)
    ax_cb = fig.add_subplot(gs[2])

    im = ax_h.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    # Gray hatch for NaN cells (channels beyond a subject's n_valid)
    nan_mask = ~np.isfinite(matrix)
    if nan_mask.any():
        ax_h.imshow(
            np.ma.masked_where(~nan_mask, np.ones_like(matrix)),
            aspect="auto",
            cmap="Greys",
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )

    # SOZ overlay: black border on cells
    for i in range(n_sub):
        for j in range(n_ch):
            if soz_overlay[i, j]:
                ax_h.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.7,
                    )
                )

    ax_h.set_yticks(range(n_sub))
    ax_h.set_yticklabels(sub_labels, fontsize=FS_TICK - 4)
    ax_h.set_xticks([0, n_ch - 1])
    ax_h.set_xticklabels(
        ["source\n(earliest in T_a)", "sink\n(latest in T_a)"],
        fontsize=FS_TICK - 4,
    )
    ax_h.set_xlabel(
        "Channel position along T_a (source → sink)",
        fontsize=FS_LABEL - 2,
        labelpad=8,
    )
    ax_h.set_title("a", fontsize=FS_PANEL_LETTER, loc="left", pad=10, fontweight="bold")
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    # Kendall τ side panel — color by F_norm > 2/3 grouping (consistent
    # with Panel B/C). PR-2.5 status (reproduced / cand-fail / non-cand)
    # is shown only as marker shape on Panel B/C, not on this bar chart
    # (would clutter the side panel).
    bar_colors = [COL_SIG if f > THRESH else COL_NONSIG for f in f_norms]
    bars = ax_tau.barh(
        range(n_sub), taus, color=bar_colors, edgecolor="black", linewidth=0.4
    )
    ax_tau.axvline(0, color="gray", linewidth=0.6)
    ax_tau.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.set_xlim(-1.05, 1.05)
    ax_tau.set_xticks([-1, 0, 1])
    ax_tau.tick_params(axis="x", labelsize=FS_TICK - 4)
    plt.setp(ax_tau.get_yticklabels(), visible=False)
    ax_tau.set_xlabel("Kendall τ", fontsize=FS_LABEL - 2, labelpad=8)
    ax_tau.spines["top"].set_visible(False)
    ax_tau.spines["right"].set_visible(False)

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Signed Δr (= rank_Tb − rank_Ta)", fontsize=FS_LABEL - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 4)

    # Legend — bars colored by F_norm > 2/3 grouping
    legend_handles = [
        mpatches.Patch(color=COL_SIG, label=f"F_norm > 2/3 (above asymptotic random)"),
        mpatches.Patch(color=COL_NONSIG, label=f"F_norm ≤ 2/3 (around random)"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="SOZ channel"),
    ]
    fig.suptitle(
        f"Per-channel signed rank displacement — stable_k=2 cohort (n={n_sub})",
        fontsize=FS_TITLE,
        y=0.97,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=FS_TICK - 3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _pr25_marker(status: str) -> Tuple[str, float]:
    """Marker shape + size for PR-2.5 status overlay."""
    return {
        "reproduced": ("o", 56),
        "candidate_fail": ("X", 90),
        "non_candidate": ("s", 56),
    }.get(status, ("o", 56))


def _draw_pr25_scatter(ax, xs, ys, statuses, base_col, edgecolor="black"):
    """Scatter with marker shape encoding PR-2.5 status."""
    for x, y, st in zip(xs, ys, statuses):
        m, s = _pr25_marker(st)
        ax.scatter(
            [x], [y],
            color=base_col,
            edgecolors=edgecolor,
            linewidths=0.6 if st != "candidate_fail" else 1.0,
            s=s, zorder=3, marker=m,
        )


def _label_subjects_in_box(ax, xs, ys, labels, x_range, y_range, fontsize):
    """Annotate subjects whose (x, y) falls in the given box."""
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    for x, y, lbl in zip(xs, ys, labels):
        if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
            ax.annotate(
                lbl,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=fontsize,
                color="black",
                alpha=0.85,
            )


def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    """2-panel scatter visualization (visualization contract, post-hoc).

    Panel B (where does PR-2.5's hard threshold miss continuous reversal?):
        x = ρ_inter (PR-2.5 inter_cluster_corr_matrix Spearman)
        y = F_norm (this supplementary's Diaconis-Graham normalized footrule)
        reference lines: x = -0.5 (PR-2.5 candidate gate, FYI only)
                         y = 2/3 (D-G asymptotic random expectation, FYI only)
        Subjects in the upper-right quadrant (ρ_inter ≥ -0.5 AND F_norm > 2/3)
        are precisely the ones missed by PR-2.5's hard cut yet flagged by the
        continuous metric.

    Panel C (does reversal degree predict SOZ enrichment?):
        x = F_norm
        y = soz_contribution_excess (= contribution_fraction - channel_fraction)
        reference lines: x = 2/3 (D-G asymptotic random, FYI only)
                         y = 0   (SOZ at chance, FYI only)
        Spearman ρ summary - a near-zero ρ means reversal degree does NOT
        predict SOZ enrichment in this cohort.

    Both panels use marker shape for PR-2.5 status (circle/X/square) as
    pure descriptive overlay; no panel uses categorical grouping or
    between-group MW-U. The reference lines are explicitly labeled
    'reference, FYI only' - they are NOT decision rules.
    """
    from scipy.stats import spearmanr

    # Build per-subject vectors
    subjects, labels, F_arr, tau_arr, rho_inter_arr, excess_arr = [], [], [], [], [], []
    pr25 = []
    for r in records:
        pair = r["primary_pair"]
        f_norm = pair.get("footrule_normalized")
        tau = pair.get("kendall_tau")
        excess = pair.get("soz_contribution_excess")
        iccm = r.get("inter_cluster_corr_matrix")
        rho_inter = (
            iccm[0][1] if iccm and len(iccm) >= 2 and len(iccm[0]) >= 2 else None
        )
        if (
            f_norm is None or tau is None or rho_inter is None
            or (isinstance(f_norm, float) and np.isnan(f_norm))
            or (isinstance(tau, float) and np.isnan(tau))
            or (isinstance(rho_inter, float) and np.isnan(rho_inter))
        ):
            continue
        subjects.append(r)
        labels.append(f"{r['dataset'][:3]}_{r['subject']}")
        F_arr.append(f_norm)
        tau_arr.append(tau)
        rho_inter_arr.append(rho_inter)
        excess_arr.append(excess if excess is not None else float("nan"))
        pr25.append(r["pr25_status"])

    F_arr = np.array(F_arr, dtype=float)
    rho_inter_arr = np.array(rho_inter_arr, dtype=float)
    excess_arr = np.array(excess_arr, dtype=float)
    n = len(F_arr)

    # Color subjects by PR-2.5 status for visual separation
    def _color_for(st: str) -> str:
        return COL_SIG if st in ("reproduced", "candidate_fail") else COL_NONSIG

    point_colors = [_color_for(st) for st in pr25]

    # Continuous summaries
    rho_FB, p_FB = spearmanr(rho_inter_arr, F_arr)
    excess_finite = np.isfinite(excess_arr)
    if excess_finite.sum() >= 4:
        rho_FC, p_FC = spearmanr(F_arr[excess_finite], excess_arr[excess_finite])
    else:
        rho_FC, p_FC = float("nan"), float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

    # ====== Panel B: ρ_inter vs F_norm ======
    ax = axes[0]
    # Reference lines (FYI only, NOT decision rules)
    ax.axvline(-0.5, color="gray", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(2 / 3, color="gray", linewidth=0.7, linestyle="--", zorder=1)
    # Quadrant shading: upper-right = "PR-2.5 missed but F_norm flags"
    ax.axvspan(-0.5, 1.0, ymin=(2/3 - 0) / 1.0,
               ymax=1.0, color=COL_SIG, alpha=0.06, zorder=0)
    # Scatter, colored by PR-2.5 status, marker by status
    for x, y, st, col in zip(rho_inter_arr, F_arr, pr25, point_colors):
        m, s = _pr25_marker(st)
        ax.scatter([x], [y], color=col, edgecolors="black",
                   linewidths=0.8 if st == "candidate_fail" else 0.5,
                   s=s, zorder=3, marker=m)
    # Label TR-quadrant subjects (PR-2.5 missed but F > 2/3)
    _label_subjects_in_box(
        ax, rho_inter_arr, F_arr, labels,
        x_range=(-0.5, 1.0), y_range=(2 / 3, 1.05),
        fontsize=FS_TICK - 5,
    )
    # Reference text
    ax.text(-0.51, 0.05, "ρ = −0.5\n(PR-2.5 gate, FYI)",
            fontsize=FS_TICK - 5, ha="right", va="bottom", color="gray")
    ax.text(0.95, 2 / 3 + 0.02, "F = 2/3 (D-G asymptotic random, FYI)",
            fontsize=FS_TICK - 5, ha="right", va="bottom", color="gray")
    # Quadrant annotation (corner labels)
    ax.text(0.95, 0.97, "PR-2.5 missed\n(ρ ≥ −0.5, F > 2/3)",
            fontsize=FS_TICK - 4, ha="right", va="top",
            color=COL_SIG, fontweight="bold")
    ax.text(-0.95, 0.97, "both flag high",
            fontsize=FS_TICK - 4, ha="left", va="top", color="black")
    ax.text(0.95, 0.05, "both around random",
            fontsize=FS_TICK - 4, ha="right", va="bottom", color="gray")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ρ_inter  (PR-2.5 inter_cluster_corr_matrix Spearman)",
                  fontsize=FS_LABEL - 2)
    ax.set_ylabel("F_norm  (Diaconis-Graham normalized footrule)",
                  fontsize=FS_LABEL - 2)
    info = (
        f"Spearman ρ(ρ_inter, F_norm) = {rho_FB:.3f}, p = {p_FB:.2g}, n = {n}\n"
        f"reference lines are descriptive (FYI), not decision rules"
    )
    ax.text(0.5, 1.04, info, transform=ax.transAxes,
            fontsize=FS_TICK - 4, ha="center", color="black")
    style_panel(ax, label="b")

    # ====== Panel C: F_norm vs soz_contribution_excess ======
    ax = axes[1]
    ax.axvline(2 / 3, color="gray", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="-", zorder=1)
    for x, y, st, col in zip(F_arr, excess_arr, pr25, point_colors):
        if not np.isfinite(y):
            continue
        m, s = _pr25_marker(st)
        ax.scatter([x], [y], color=col, edgecolors="black",
                   linewidths=0.8 if st == "candidate_fail" else 0.5,
                   s=s, zorder=3, marker=m)
    # SOZ at chance reference label (placed near top-left so it doesn't crowd points)
    ax.text(0.40, 0.005, "SOZ at chance (excess = 0, FYI)",
            fontsize=FS_TICK - 5, ha="left", va="bottom", color="gray")
    ax.text(2 / 3 + 0.005, ax.get_ylim()[1] - 0.01,
            "F = 2/3\n(D-G ref, FYI)",
            fontsize=FS_TICK - 5, ha="left", va="top", color="gray")
    ax.set_xlabel("F_norm  (Diaconis-Graham normalized footrule)",
                  fontsize=FS_LABEL - 2)
    ax.set_ylabel(
        "soz_contribution_excess\n(= SOZ contribution_fraction − channel_fraction)",
        fontsize=FS_LABEL - 3,
    )
    info = (
        f"Spearman ρ(F_norm, SOZ excess) = {rho_FC:.3f}, p = {p_FC:.2g}, "
        f"n = {int(excess_finite.sum())}\n"
        f"low |ρ| ⇒ reversal degree does NOT predict SOZ enrichment"
    )
    ax.text(0.5, 1.04, info, transform=ax.transAxes,
            fontsize=FS_TICK - 4, ha="center", color="black")
    style_panel(ax, label="c")

    # Combined legend at the bottom
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_SIG, markeredgecolor="black",
                   markersize=8, label="PR-2.5 reproduced (n=6)"),
        plt.Line2D([0], [0], marker="X", color="w",
                   markerfacecolor=COL_SIG, markeredgecolor="black",
                   markersize=11, label="PR-2.5 candidate-fail (n=1, huanghanwen)"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COL_NONSIG, markeredgecolor="black",
                   markersize=8, label="PR-2.5 non-candidate (n=16)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=FS_TICK - 3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Stdout summary for archive doc fill-in
    print(f"  Panel B Spearman ρ(ρ_inter, F_norm) = {rho_FB:.3f}, p = {p_FB:.3g}, n = {n}")
    print(f"  Panel C Spearman ρ(F_norm, SOZ excess) = {rho_FC:.3f}, p = {p_FC:.3g}, n = {int(excess_finite.sum())}")
    # Quadrant counts for Panel B
    tr_count = int(((rho_inter_arr >= -0.5) & (F_arr > 2/3)).sum())
    tl_count = int(((rho_inter_arr < -0.5) & (F_arr > 2/3)).sum())
    br_count = int(((rho_inter_arr >= -0.5) & (F_arr <= 2/3)).sum())
    bl_count = int(((rho_inter_arr < -0.5) & (F_arr <= 2/3)).sum())
    print(f"  Panel B quadrants: TL (both flag high)={tl_count}  "
          f"TR (PR-2.5 missed)={tr_count}  "
          f"BR (both low)={br_count}  BL (anomalous)={bl_count}")


def plot_per_subject_strip(record: dict, out_path: Path) -> None:
    pair = record["primary_pair"]
    delta = np.asarray(pair["signed_displacement_full"], dtype=float)
    joint = np.asarray(pair["joint_valid"], dtype=bool)
    soz_mask = np.asarray(
        pair.get("soz_mask", [False] * len(delta)), dtype=bool
    )
    rank_a_dense_full = np.asarray(pair["rank_a_dense_full"], dtype=float)
    channel_names = record["channel_names"]
    valid_idx = np.where(joint)[0]
    if len(valid_idx) == 0:
        return
    delta_v = delta[valid_idx]
    chs_v = [channel_names[i] for i in valid_idx]
    soz_v = soz_mask[valid_idx]
    rank_a_v = rank_a_dense_full[valid_idx]
    # Sort by rank_T_a_dense (T_a source -> sink), NOT by Δr — anti-bias rule.
    order = np.argsort(rank_a_v)
    delta_sorted = delta_v[order]
    chs_sorted = [chs_v[i] for i in order]
    soz_sorted = soz_v[order]

    n_ch = len(delta_sorted)
    vmax = float(np.max(np.abs(delta_sorted))) if n_ch > 0 else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * n_ch), 2.4))
    im = ax.imshow(
        delta_sorted[None, :],
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    for j, is_soz in enumerate(soz_sorted):
        if is_soz:
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, -0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.0,
                )
            )
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(
        chs_sorted, rotation=60, fontsize=FS_TICK - 4, ha="right"
    )
    ax.set_yticks([])
    sub_label = f"{record['dataset']} {record['subject']}"
    fwd = "✓" if record.get("fwd_rev_reproduced") else "✗"
    tau = pair.get("kendall_tau", float("nan"))
    f_norm = pair.get("footrule_normalized", float("nan"))
    ax.set_title(
        f"{sub_label}  |  k={record.get('stable_k')}  |  "
        f"fwd/rev={fwd}  |  τ={tau:.3f}  |  F_norm={f_norm:.3f}\n"
        f"channels arranged by rank_T_a (source → sink)",
        fontsize=FS_LABEL - 2,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Δr", fontsize=FS_LABEL - 4)
    cb.ax.tick_params(labelsize=FS_TICK - 4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_PUB, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--what",
        default="all",
        choices=["all", "cohort", "summary", "per_subject"],
    )
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_SUB_FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = load_cohort_records()
    print(f"Loaded {len(records)} stable_k=2 subjects")

    if args.what in ("all", "cohort"):
        plot_cohort_heatmap(records, FIG_DIR / "cohort_displacement_heatmap")
        print("Wrote cohort_displacement_heatmap.{png,pdf}")

    if args.what in ("all", "summary"):
        plot_footrule_summary(records, FIG_DIR / "footrule_kendall_summary")
        print("Wrote footrule_kendall_summary.{png,pdf}")

    if args.what in ("all", "per_subject"):
        for r in records:
            stem = f"{r['dataset']}_{r['subject']}"
            plot_per_subject_strip(
                r, PER_SUB_FIG_DIR / f"{stem}_displacement.png"
            )
        print(f"Wrote per-subject strips for {len(records)} subjects")


if __name__ == "__main__":
    main()
