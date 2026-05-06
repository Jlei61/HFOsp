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

    # Kendall τ side panel — bar length encodes τ, single neutral color
    # (paper-level: no binary classification overlay; the rank-ordering
    # of rows + bar length together convey the continuous spectrum).
    ax_tau.barh(
        range(n_sub), taus,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.4,
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

    # Legend — paper-level: only the displacement colorbar and SOZ outline
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="SOZ channel"),
    ]
    fig.suptitle(
        f"Per-channel signed rank displacement — two-template subjects (n={n_sub})",
        fontsize=FS_TITLE,
        y=0.97,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        fontsize=FS_TICK - 2,
        frameon=False,
        bbox_to_anchor=(0.96, 0.93),
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    """Paper-level 2-panel summary of cohort rank-displacement geometry.

    Panel B - Reversal spectrum:
        x = subjects ranked by F_norm (descending)
        y = F_norm (Diaconis-Graham normalized footrule)
        Shows that the cohort is a CONTINUOUS spectrum, not a binary.
        The 2/3 horizontal line is annotated as the asymptotic
        random-permutation reference (Diaconis-Graham 1977, n→∞ expectation),
        descriptive only - NOT a classifier threshold.

    Panel C - SOZ contribution vs reversal strength:
        x = F_norm
        y = soz_contribution_excess (baseline-corrected)
        Spearman ρ summary; near-zero correlation = honest negative finding.
        Marker shape = dataset (epilepsiae circle, yuquan square),
        purely for visual disambiguation, not for grouping.

    NO PR-2.5 internal-workflow language anywhere on this figure (no
    'reproduced', 'candidate-fail', 'non-candidate', 'PR-2.5 gate',
    'PR-2.5 missed'). Those classifications belong in the methods archive,
    not on a paper-level figure. The figure tells one story:
    template-pair rank reversal is a cohort-level continuous spectrum,
    and SOZ channel involvement does NOT explain that spectrum.
    """
    from scipy.stats import spearmanr

    # Build per-subject vectors
    subjects, labels, F_arr, tau_arr, excess_arr, datasets = (
        [], [], [], [], [], []
    )
    for r in records:
        pair = r["primary_pair"]
        f_norm = pair.get("footrule_normalized")
        tau = pair.get("kendall_tau")
        excess = pair.get("soz_contribution_excess")
        if (
            f_norm is None or tau is None
            or (isinstance(f_norm, float) and np.isnan(f_norm))
            or (isinstance(tau, float) and np.isnan(tau))
        ):
            continue
        subjects.append(r)
        labels.append(f"{r['dataset'][:3]}_{r['subject']}")
        F_arr.append(f_norm)
        tau_arr.append(tau)
        excess_arr.append(excess if excess is not None else float("nan"))
        datasets.append(r.get("dataset", "unknown"))

    F_arr = np.array(F_arr, dtype=float)
    tau_arr = np.array(tau_arr, dtype=float)
    excess_arr = np.array(excess_arr, dtype=float)
    n = len(F_arr)

    # Sort for spectrum panel (descending F_norm)
    order = np.argsort(-F_arr)
    F_sorted = F_arr[order]
    tau_sorted = tau_arr[order]
    labels_sorted = [labels[i] for i in order]
    datasets_sorted = [datasets[i] for i in order]

    # Spearman correlation for Panel C
    excess_finite = np.isfinite(excess_arr)
    if excess_finite.sum() >= 4:
        rho_FC, p_FC = spearmanr(F_arr[excess_finite], excess_arr[excess_finite])
    else:
        rho_FC, p_FC = float("nan"), float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0),
                              gridspec_kw={"width_ratios": [1.7, 1.0]})

    # ====== Panel B: Ranked F_norm spectrum (cohort continuous spectrum) ======
    ax = axes[0]
    x_positions = np.arange(n)
    ax.bar(
        x_positions, F_sorted,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.4, width=0.78,
        zorder=3,
    )
    # 2/3 reference line, annotated descriptively
    ax.axhline(2 / 3, color="gray", linewidth=0.7, linestyle="--", zorder=2)
    ax.text(
        n - 0.5, 2 / 3 + 0.012,
        "2/3 — asymptotic random-permutation reference (Diaconis-Graham 1977, n→∞)",
        fontsize=FS_TICK - 4, ha="right", va="bottom", color="gray",
    )
    # Subject labels along x
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_sorted, rotation=70, fontsize=FS_TICK - 4, ha="right")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("subjects, ranked by F_norm (descending)",
                  fontsize=FS_LABEL - 1, labelpad=8)
    ax.set_ylabel("F_norm  (Diaconis-Graham normalized footrule)",
                  fontsize=FS_LABEL - 1)
    style_panel(ax, label="b")

    # ====== Panel C: F_norm vs soz_contribution_excess ======
    ax = axes[1]
    ax.axvline(2 / 3, color="lightgray", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="-", zorder=1)
    for x, y, ds in zip(F_arr, excess_arr, datasets):
        if not np.isfinite(y):
            continue
        marker = "o" if ds == "epilepsiae" else "s"
        ax.scatter([x], [y], color=COL_NEUTRAL, edgecolors="black",
                   linewidths=0.5, s=64, zorder=3, marker=marker)
    # Set explicit y-limits to give room for reference text
    finite_excess = excess_arr[excess_finite]
    y_pad = 0.04
    y_lo = float(np.min(finite_excess)) - y_pad
    y_hi = float(np.max(finite_excess)) + y_pad
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0.35, 1.05)
    # Reference labels (small, gray, FYI), placed in low-density corners
    ax.text(2 / 3 + 0.005, y_lo + 0.005,
            "F = 2/3 (random ref)",
            fontsize=FS_TICK - 5, ha="left", va="bottom", color="gray")
    ax.text(0.36, 0.005, "SOZ at chance",
            fontsize=FS_TICK - 5, ha="left", va="bottom", color="gray")
    ax.set_xlabel("F_norm  (Diaconis-Graham normalized footrule)",
                  fontsize=FS_LABEL - 1, labelpad=8)
    ax.set_ylabel(
        "SOZ contribution_excess\n"
        "(SOZ contribution_fraction − channel_fraction)",
        fontsize=FS_LABEL - 2,
    )
    info = (
        f"Spearman ρ = {rho_FC:.3f}, p = {p_FC:.2g}, n = {int(excess_finite.sum())}"
    )
    ax.text(0.5, 1.025, info, transform=ax.transAxes,
            fontsize=FS_TICK - 2, ha="center", color="black")
    # Dataset legend (purely shape disambiguation)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_NEUTRAL, markeredgecolor="black",
                   markersize=8, label="Epilepsiae"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COL_NEUTRAL, markeredgecolor="black",
                   markersize=8, label="Yuquan"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=FS_TICK - 3, frameon=False, title="dataset",
              title_fontsize=FS_TICK - 3)
    style_panel(ax, label="c")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Stdout summary
    print(f"  Panel B (spectrum): n={n}, F_norm range = [{F_sorted.min():.3f}, {F_sorted.max():.3f}]")
    print(f"  Panel C Spearman ρ(F_norm, SOZ excess) = {rho_FC:.3f}, p = {p_FC:.3g}, n = {int(excess_finite.sum())}")


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
