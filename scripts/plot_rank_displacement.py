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


def sort_by_footrule_desc(records: List[dict]) -> List[dict]:
    """Most reversal at top: F_norm descending."""
    return sorted(
        records,
        key=lambda r: -r["primary_pair"].get("footrule_normalized", 0.0),
    )


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
    """Single paper-level supplementary figure.

    Main heatmap: per-channel Δr across two-template subjects.
    Three right-side summary tracks (sharing the same y-axis = subject):
        - F_norm (Diaconis-Graham normalized footrule), with 2/3 reference
        - Kendall τ between Tₐ and T_b ranks, with τ = 0 reference
        - SOZ contribution_excess, with excess = 0 reference (chance baseline)

    Each row of the entire panel is one subject - reader sees the channel-
    level pattern, overall reversal magnitude, and SOZ contribution all
    in the same coordinate system.

    No PR-2.5 internal classifications, no group colors. Subjects sorted
    by F_norm descending (most reversal at top).
    """
    sorted_records = sort_by_footrule_desc(records)
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
    soz_excess = np.array(
        [
            r["primary_pair"].get("soz_contribution_excess")
            if r["primary_pair"].get("soz_contribution_excess") is not None
            else np.nan
            for r in sorted_records
        ],
        dtype=float,
    )

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(15, max(7.0, 0.42 * n_sub) + 1.2))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[7.0, 1.0, 1.2, 1.2, 0.30],
        wspace=0.10,
        top=0.86,
        bottom=0.10,
        left=0.08,
        right=0.96,
    )
    ax_h = fig.add_subplot(gs[0])
    ax_F = fig.add_subplot(gs[1], sharey=ax_h)
    ax_tau = fig.add_subplot(gs[2], sharey=ax_h)
    ax_soz = fig.add_subplot(gs[3], sharey=ax_h)
    ax_cb = fig.add_subplot(gs[4])

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
    # No panel letter — this is a single composite supplementary figure
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    # === Right-side summary tracks (share y-axis with heatmap) ===
    # All three tracks: single neutral color, dashed reference line.
    # No grouping, no classification colors — paper-level continuous spectrum.

    # Track 1: F_norm horizontal mini-bars (range 0..1, ref at 2/3)
    ax_F.barh(
        range(n_sub), f_norms,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.4,
    )
    ax_F.axvline(2 / 3, color="gray", linewidth=0.7, linestyle="--")
    ax_F.set_xlim(0, 1.05)
    ax_F.set_xticks([0, 2 / 3, 1])
    ax_F.set_xticklabels(["0", "2/3", "1"], fontsize=FS_TICK - 4)
    plt.setp(ax_F.get_yticklabels(), visible=False)
    ax_F.set_xlabel("F_norm", fontsize=FS_LABEL - 2, labelpad=8)
    ax_F.spines["top"].set_visible(False)
    ax_F.spines["right"].set_visible(False)

    # Track 2: Kendall τ horizontal mini-bars (range -1..+1, ref at 0)
    ax_tau.barh(
        range(n_sub), taus,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.4,
    )
    ax_tau.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax_tau.set_xlim(-1.05, 1.05)
    ax_tau.set_xticks([-1, 0, 1])
    ax_tau.tick_params(axis="x", labelsize=FS_TICK - 4)
    plt.setp(ax_tau.get_yticklabels(), visible=False)
    ax_tau.set_xlabel("Kendall τ", fontsize=FS_LABEL - 2, labelpad=8)
    ax_tau.spines["top"].set_visible(False)
    ax_tau.spines["right"].set_visible(False)

    # Track 3: SOZ contribution_excess horizontal mini-bars (ref at 0)
    finite_excess = np.isfinite(soz_excess)
    excess_for_plot = np.where(finite_excess, soz_excess, 0.0)
    bar_colors_soz = [
        COL_NEUTRAL if finite_excess[i] else "lightgray"
        for i in range(n_sub)
    ]
    ax_soz.barh(
        range(n_sub), excess_for_plot,
        color=bar_colors_soz, edgecolor="black", linewidth=0.4,
    )
    ax_soz.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    excess_max = float(np.nanmax(np.abs(soz_excess[finite_excess]))) if finite_excess.any() else 0.4
    soz_xlim = max(0.4, excess_max + 0.05)
    ax_soz.set_xlim(-soz_xlim, soz_xlim)
    ax_soz.set_xticks([-soz_xlim + 0.05, 0, soz_xlim - 0.05])
    ax_soz.set_xticklabels([f"{-soz_xlim+0.05:.2f}", "0", f"{soz_xlim-0.05:.2f}"],
                           fontsize=FS_TICK - 4)
    plt.setp(ax_soz.get_yticklabels(), visible=False)
    ax_soz.set_xlabel(
        "SOZ excess\n(contribution − channel fraction)",
        fontsize=FS_LABEL - 3, labelpad=8,
    )
    ax_soz.spines["top"].set_visible(False)
    ax_soz.spines["right"].set_visible(False)

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
        choices=["all", "cohort", "per_subject"],
    )
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_SUB_FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = load_cohort_records()
    print(f"Loaded {len(records)} stable_k=2 subjects")

    if args.what in ("all", "cohort"):
        plot_cohort_heatmap(records, FIG_DIR / "cohort_displacement_heatmap")
        print("Wrote cohort_displacement_heatmap.{png,pdf}")

    if args.what in ("all", "per_subject"):
        for r in records:
            stem = f"{r['dataset']}_{r['subject']}"
            plot_per_subject_strip(
                r, PER_SUB_FIG_DIR / f"{stem}_displacement.png"
            )
        print(f"Wrote per-subject strips for {len(records)} subjects")


if __name__ == "__main__":
    main()
