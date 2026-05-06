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


def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    """2-panel descriptive: F_norm distribution + Kendall tau split by F_norm > 2/3.

    Grouping rule (avoids PR-2.5's hard threshold ρ_inter < -0.5):
        Group A: F_norm > 2/3 (above Diaconis-Graham asymptotic random)
        Group B: F_norm ≤ 2/3 (around random)
    The 2/3 cutoff is theoretically motivated (asymptotic random
    expectation for normalized Spearman footrule under uniform random
    permutation), NOT another PR's hard threshold.

    Statistical summaries:
      - Spearman correlation between F_norm and Kendall τ (continuous,
        principled). Confirms the two metrics measure the same
        underlying geometry.
      - Panel B has NO MW-U (grouping by F_norm and testing F_norm
        is tautological).
      - Panel C: MW-U on Kendall τ between the F_norm-defined groups
        (orthogonal-ish; descriptive only, NOT a PASS gate).

    PR-2.5 status (reproduced / candidate-fail / non-candidate) is shown
    via marker shape ONLY as descriptive overlay, NOT for grouping.
    """
    from scipy.stats import mannwhitneyu, spearmanr

    THRESH = 2 / 3
    hi_F, lo_F = [], []
    hi_T, lo_T = [], []
    hi_status, lo_status = [], []
    all_F, all_T = [], []
    for r in records:
        f_norm = r["primary_pair"].get("footrule_normalized")
        tau = r["primary_pair"].get("kendall_tau")
        if (
            f_norm is None or tau is None
            or (isinstance(f_norm, float) and np.isnan(f_norm))
            or (isinstance(tau, float) and np.isnan(tau))
        ):
            continue
        all_F.append(f_norm)
        all_T.append(tau)
        if f_norm > THRESH:
            hi_F.append(f_norm)
            hi_T.append(tau)
            hi_status.append(r["pr25_status"])
        else:
            lo_F.append(f_norm)
            lo_T.append(tau)
            lo_status.append(r["pr25_status"])

    n_hi, n_lo = len(hi_F), len(lo_F)

    # Statistical summaries (descriptive only)
    rho, rho_p = spearmanr(all_F, all_T) if len(all_F) >= 4 else (float("nan"), float("nan"))
    if n_hi >= 2 and n_lo >= 2:
        mw = mannwhitneyu(hi_T, lo_T, alternative="two-sided")
        T_U, T_p = float(mw.statistic), float(mw.pvalue)
    else:
        T_U, T_p = float("nan"), float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # ===== Panel B: F_norm distribution split by F_norm > 2/3 =====
    ax = axes[0]
    positions = [0, 1]
    parts = ax.violinplot(
        [hi_F, lo_F],
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for pc, col in zip(parts["bodies"], [COL_SIG, COL_NONSIG]):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    rng = np.random.default_rng(42)

    # Marker shape by PR-2.5 status (descriptive overlay only)
    def _scatter_with_pr25(ax, xs, ys, statuses, base_col):
        for x, y, st in zip(xs, ys, statuses):
            if st == "reproduced":
                ax.scatter([x], [y], color=base_col, edgecolors="black",
                           linewidths=0.4, s=36, zorder=3, marker="o")
            elif st == "candidate_fail":
                ax.scatter([x], [y], facecolor=base_col, edgecolors="black",
                           linewidths=0.9, s=58, zorder=4, marker="X")
            else:  # non_candidate
                ax.scatter([x], [y], color=base_col, edgecolors="black",
                           linewidths=0.4, s=36, zorder=3, marker="s")

    if hi_F:
        jit = rng.uniform(-0.12, 0.12, size=n_hi)
        _scatter_with_pr25(ax, jit + 0.0, hi_F, hi_status, COL_SIG)
    if lo_F:
        jit = rng.uniform(-0.12, 0.12, size=n_lo)
        _scatter_with_pr25(ax, jit + 1.0, lo_F, lo_status, COL_NONSIG)

    ax.axhline(THRESH, color="black", linewidth=0.7, linestyle="--", zorder=2)
    ax.text(
        1.6, THRESH, f"F_norm = 2/3\n(asymptotic random,\nDiaconis-Graham)",
        fontsize=FS_TICK - 4, va="center", color="black",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"F_norm > 2/3\n(n={n_hi})", f"F_norm ≤ 2/3\n(n={n_lo})"],
        fontsize=FS_TICK - 3,
    )
    ax.set_ylabel("Footrule (Diaconis-Graham normalized)", fontsize=FS_LABEL - 2)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(-0.6, 1.6)
    # No MW-U on F_norm (tautological); show grouping rule + Spearman summary.
    info = (
        "groups split by F_norm > 2/3 (theoretical asymptotic random reference)\n"
        f"Spearman ρ(F_norm, τ) = {rho:.3f}, p = {rho_p:.2g} (n={len(all_F)}, continuous)"
    )
    ax.text(0.5, 1.05, info, transform=ax.transAxes,
            fontsize=FS_TICK - 4, ha="center", color="black")
    style_panel(ax, label="b")

    # ===== Panel C: Kendall τ split by F_norm > 2/3, MW-U on τ =====
    ax = axes[1]
    parts = ax.violinplot(
        [hi_T, lo_T],
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for pc, col in zip(parts["bodies"], [COL_SIG, COL_NONSIG]):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    if hi_T:
        jit = rng.uniform(-0.12, 0.12, size=n_hi)
        _scatter_with_pr25(ax, jit + 0.0, hi_T, hi_status, COL_SIG)
    if lo_T:
        jit = rng.uniform(-0.12, 0.12, size=n_lo)
        _scatter_with_pr25(ax, jit + 1.0, lo_T, lo_status, COL_NONSIG)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", zorder=2)
    ax.axhline(-0.5, color="gray", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"F_norm > 2/3\n(n={n_hi})", f"F_norm ≤ 2/3\n(n={n_lo})"],
        fontsize=FS_TICK - 3,
    )
    ax.set_ylabel("Kendall τ between Tₐ and T_b ranks", fontsize=FS_LABEL - 2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-0.6, 1.6)
    p_str = (
        f"Mann-Whitney U on τ (between F_norm-defined groups)\n"
        f"U = {T_U:.1f}, p = {T_p:.3g}"
    ) if not np.isnan(T_p) else ""
    ax.text(0.5, 1.05, p_str, transform=ax.transAxes,
            fontsize=FS_TICK - 4, ha="center", color="black")

    # Legend showing PR-2.5 marker overlay
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_NEUTRAL, markeredgecolor="black",
                   markersize=8, label="PR-2.5 reproduced"),
        plt.Line2D([0], [0], marker="X", color="w",
                   markerfacecolor=COL_NEUTRAL, markeredgecolor="black",
                   markersize=10, label="PR-2.5 candidate-fail"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COL_NEUTRAL, markeredgecolor="black",
                   markersize=8, label="PR-2.5 non-candidate"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=FS_TICK - 5, frameon=False, title="marker = PR-2.5 status (overlay only)",
              title_fontsize=FS_TICK - 5)
    style_panel(ax, label="c")

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    # Print test results to stdout for archive doc fill-in
    print(
        f"  Spearman ρ(F_norm, τ) = {rho:.3f}, p = {rho_p:.4g}, n = {len(all_F)}"
    )
    print(
        f"  MW-U τ (grouped by F_norm > 2/3): U = {T_U:.2f}, p = {T_p:.4g}"
    )
    print(
        f"    medians: hi (n={n_hi}) τ = {np.median(hi_T):+.3f}, F = {np.median(hi_F):.3f}"
    )
    print(
        f"    medians: lo (n={n_lo}) τ = {np.median(lo_T):+.3f}, F = {np.median(lo_F):.3f}"
    )


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
