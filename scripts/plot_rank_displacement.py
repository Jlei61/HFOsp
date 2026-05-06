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
    pr25_status = [r["pr25_status"] for r in sorted_records]

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

    # Kendall τ side panel — color by PR-2.5 candidate status
    # rust = candidate cohort (TRUE + FALSE), gray = non-candidate (None)
    # Candidate-fail subject (FALSE) gets a hatched fill to distinguish from reproduced.
    bar_colors = [
        COL_SIG if s in ("reproduced", "candidate_fail") else COL_NONSIG
        for s in pr25_status
    ]
    bars = ax_tau.barh(
        range(n_sub), taus, color=bar_colors, edgecolor="black", linewidth=0.4
    )
    for bar, status in zip(bars, pr25_status):
        if status == "candidate_fail":
            bar.set_hatch("///")
            bar.set_edgecolor("black")
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

    # Legend — three categories matching the right-side bar colors
    legend_handles = [
        mpatches.Patch(color=COL_SIG, label="PR-2.5 fwd/rev cohort: reproduced"),
        mpatches.Patch(facecolor=COL_SIG, edgecolor="black", hatch="///",
                       label="PR-2.5 fwd/rev cohort: candidate-fail"),
        mpatches.Patch(color=COL_NONSIG, label="non-candidate (ρ_inter ≥ −0.5)"),
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
        ncol=4,
        fontsize=FS_TICK - 4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    """2-panel descriptive: F_norm violin + Kendall tau strip plot.

    Grouping (per archive doc §5.2): PR-2.5 fwd/rev cohort
    (candidates that passed the ρ_inter < -0.5 gate, TRUE + FALSE = n=7)
    vs non-candidates (None = n=16). Within the candidate cohort,
    candidate-fail (FALSE) is shown with a hatched marker so the
    reader can tell reproduced (TRUE) from candidate-fail.

    Mann-Whitney U test between groups is annotated above each panel.
    Asymptotic 2/3 reference is a Diaconis-Graham n->inf expectation
    (NOT a precise baseline, never used as a PASS gate).
    """
    from scipy.stats import mannwhitneyu

    cand_F, noncand_F = [], []
    cand_T, noncand_T = [], []
    cand_subjects, noncand_subjects = [], []  # for hatch marking
    cand_status = []  # 'reproduced' or 'candidate_fail'
    for r in records:
        f_norm = r["primary_pair"].get("footrule_normalized")
        tau = r["primary_pair"].get("kendall_tau")
        if (
            f_norm is None or tau is None
            or (isinstance(f_norm, float) and np.isnan(f_norm))
            or (isinstance(tau, float) and np.isnan(tau))
        ):
            continue
        if r["is_candidate"]:
            cand_F.append(f_norm)
            cand_T.append(tau)
            cand_subjects.append(f"{r['dataset']}_{r['subject']}")
            cand_status.append(r["pr25_status"])
        else:
            noncand_F.append(f_norm)
            noncand_T.append(tau)
            noncand_subjects.append(f"{r['dataset']}_{r['subject']}")

    n_cand = len(cand_F)
    n_noncand = len(noncand_F)

    def _mw(a: list, b: list) -> Tuple[float, float]:
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan")
        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)

    F_U, F_p = _mw(cand_F, noncand_F)
    T_U, T_p = _mw(cand_T, noncand_T)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))

    # Panel B: footrule normalized
    ax = axes[0]
    positions = [0, 1]
    parts = ax.violinplot(
        [cand_F, noncand_F],
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
    # Candidate cohort: hatch-mark candidate_fail subjects
    if cand_F:
        cand_jitter = rng.uniform(-0.12, 0.12, size=len(cand_F))
        for x, y, status in zip(cand_jitter, cand_F, cand_status):
            if status == "candidate_fail":
                ax.scatter(
                    [x], [y],
                    facecolor=COL_SIG, edgecolors="black",
                    linewidths=0.8, s=48, zorder=4,
                    marker="X",
                )
            else:
                ax.scatter(
                    [x], [y],
                    color=COL_SIG, edgecolors="black",
                    linewidths=0.4, s=32, zorder=3,
                )
    if noncand_F:
        noncand_jitter = rng.uniform(-0.12, 0.12, size=len(noncand_F))
        ax.scatter(
            noncand_jitter + 1.0,
            noncand_F,
            color=COL_NONSIG, edgecolors="black",
            linewidths=0.4, s=32, zorder=3,
        )
    ax.axhline(2 / 3, color="gray", linewidth=0.6, linestyle=":")
    ax.text(
        1.55, 2 / 3, "asymptotic random\nreference (≈ 2/3)",
        fontsize=FS_TICK - 4, va="center", color="gray",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"PR-2.5 cohort\n(n={n_cand})", f"non-candidate\n(n={n_noncand})"],
        fontsize=FS_TICK - 3,
    )
    ax.set_ylabel("Footrule (Diaconis-Graham normalized)", fontsize=FS_LABEL - 2)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(-0.6, 1.6)
    # MW-U annotation + grouping rule
    p_str = (
        f"groups split by PR-2.5 candidate gate (ρ_inter < −0.5)\n"
        f"Mann-Whitney U  p = {F_p:.3g}"
    ) if not np.isnan(F_p) else ""
    ax.text(0.5, 1.06, p_str, transform=ax.transAxes,
            fontsize=FS_TICK - 4, ha="center", color="black")
    style_panel(ax, label="b")

    # Panel C: Kendall tau strip
    ax = axes[1]
    if cand_T:
        for tau_v, status in zip(cand_T, cand_status):
            if status == "candidate_fail":
                ax.scatter(
                    [tau_v], [1.0],
                    facecolor=COL_SIG, edgecolors="black",
                    linewidths=0.8, s=58, zorder=4, marker="X",
                )
            else:
                ax.scatter(
                    [tau_v], [1.0],
                    color=COL_SIG, edgecolors="black",
                    linewidths=0.4, s=44, zorder=3,
                )
    if noncand_T:
        ax.scatter(
            noncand_T,
            np.zeros(len(noncand_T)) + 0.0,
            color=COL_NONSIG, edgecolors="black",
            linewidths=0.4, s=44, zorder=3,
        )
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="x", labelsize=FS_TICK - 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [f"non-candidate\n(n={n_noncand})", f"PR-2.5 cohort\n(n={n_cand})"],
        fontsize=FS_TICK - 4,
    )
    ax.set_xlabel("Kendall τ between Tₐ and T_b ranks", fontsize=FS_LABEL - 2)
    ax.set_ylim(-0.7, 1.7)
    p_str = f"Mann-Whitney U  p = {T_p:.3g}" if not np.isnan(T_p) else ""
    ax.text(0.5, 1.04, p_str, transform=ax.transAxes,
            fontsize=FS_TICK - 3, ha="center", color="black")
    # custom legend with X marker for candidate_fail
    legend_handles = [
        mpatches.Patch(color=COL_SIG, label="PR-2.5 cohort: reproduced"),
        plt.Line2D([0], [0], marker="X", color="w",
                   markerfacecolor=COL_SIG, markeredgecolor="black",
                   markersize=8, label="candidate-fail (huanghanwen)"),
        mpatches.Patch(color=COL_NONSIG, label="non-candidate (ρ_inter ≥ −0.5)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=FS_TICK - 5, frameon=False)
    style_panel(ax, label="c")

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    # Print test results to stdout for archive doc fill-in
    print(
        f"  MW-U F_norm:  U={F_U:.3f}  p={F_p:.4g}   "
        f"medians: cand={np.median(cand_F):.3f} (n={n_cand}) vs "
        f"noncand={np.median(noncand_F):.3f} (n={n_noncand})"
    )
    print(
        f"  MW-U Kendall: U={T_U:.3f}  p={T_p:.4g}   "
        f"medians: cand={np.median(cand_T):.3f} vs noncand={np.median(noncand_T):.3f}"
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
