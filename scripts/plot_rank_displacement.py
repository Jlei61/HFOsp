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
    fwd_rev_flags = [bool(r.get("fwd_rev_reproduced")) for r in sorted_records]

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(11, max(6, 0.36 * n_sub)))
    gs = fig.add_gridspec(1, 3, width_ratios=[8, 1.2, 0.4], wspace=0.05)
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
        ["T_a source\n(earliest in T_a)", "T_a sink\n(latest in T_a)"],
        fontsize=FS_TICK - 2,
    )
    ax_h.set_xlabel(
        "Channel position along T_a's source→sink axis",
        fontsize=FS_LABEL,
    )
    ax_h.set_title("a", fontsize=FS_PANEL_LETTER, loc="left", pad=10, fontweight="bold")
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    # Kendall τ side panel
    bar_colors = [COL_SIG if f else COL_NONSIG for f in fwd_rev_flags]
    ax_tau.barh(
        range(n_sub), taus, color=bar_colors, edgecolor="black", linewidth=0.4
    )
    ax_tau.axvline(0, color="gray", linewidth=0.6)
    ax_tau.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.set_xlim(-1.05, 1.05)
    ax_tau.set_xticks([-1, 0, 1])
    ax_tau.tick_params(axis="x", labelsize=FS_TICK - 4)
    plt.setp(ax_tau.get_yticklabels(), visible=False)
    ax_tau.set_xlabel("Kendall τ", fontsize=FS_LABEL - 2)
    ax_tau.spines["top"].set_visible(False)
    ax_tau.spines["right"].set_visible(False)

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Signed Δr (= rank_Tb − rank_Ta)", fontsize=FS_LABEL - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 4)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COL_SIG, label="forward/reverse reproduced (OR rule)"),
        mpatches.Patch(color=COL_NONSIG, label="not reproduced / not testable"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="SOZ channel"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=FS_TICK - 2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )

    fig.suptitle(
        f"Per-channel signed rank displacement, stable_k=2 cohort (n={n_sub})",
        fontsize=FS_TITLE,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    """2-panel descriptive: F_norm violin + Kendall tau strip plot."""
    fwd_yes, fwd_no = [], []
    tau_yes, tau_no = [], []
    for r in records:
        f_norm = r["primary_pair"].get("footrule_normalized")
        tau = r["primary_pair"].get("kendall_tau")
        if (
            f_norm is None or tau is None
            or (isinstance(f_norm, float) and np.isnan(f_norm))
            or (isinstance(tau, float) and np.isnan(tau))
        ):
            continue
        if r.get("fwd_rev_reproduced"):
            fwd_yes.append(f_norm)
            tau_yes.append(tau)
        else:
            fwd_no.append(f_norm)
            tau_no.append(tau)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Panel B: footrule normalized split
    ax = axes[0]
    positions = [0, 1]
    parts = ax.violinplot(
        [fwd_yes, fwd_no],
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
    for pos, vals, col in zip(
        positions, [fwd_yes, fwd_no], [COL_SIG, COL_NONSIG]
    ):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.array([pos] * len(vals)) + jitter,
            vals,
            color=col,
            edgecolors="black",
            linewidths=0.4,
            s=28,
            zorder=3,
        )
    # Diaconis-Graham F_norm has *asymptotic* random expectation 2/3 (n -> inf).
    # Finite-n random expectation is slightly below; this is a reference, not a gate.
    ax.axhline(2 / 3, color="gray", linewidth=0.6, linestyle=":")
    ax.text(
        1.45,
        2 / 3,
        "asymptotic random\nreference (≈ 2/3)",
        fontsize=FS_TICK - 4,
        va="center",
        color="gray",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"reproduced\n(n={len(fwd_yes)})",
            f"not reproduced\n(n={len(fwd_no)})",
        ],
        fontsize=FS_TICK - 2,
    )
    ax.set_ylabel(
        "Footrule (Diaconis-Graham normalized)", fontsize=FS_LABEL
    )
    ax.set_ylim(0, 1.05)
    style_panel(ax, label="b")

    # Panel C: Kendall tau strip
    ax = axes[1]
    if tau_yes:
        ax.scatter(
            tau_yes,
            np.zeros(len(tau_yes)) + 1.0,
            color=COL_SIG,
            edgecolors="black",
            linewidths=0.4,
            s=44,
            zorder=3,
            label=f"reproduced (n={len(tau_yes)})",
        )
    if tau_no:
        ax.scatter(
            tau_no,
            np.zeros(len(tau_no)) + 0.0,
            color=COL_NONSIG,
            edgecolors="black",
            linewidths=0.4,
            s=44,
            zorder=3,
            label=f"not reproduced (n={len(tau_no)})",
        )
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="x", labelsize=FS_TICK - 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["not\nreproduced", "reproduced"], fontsize=FS_TICK - 2
    )
    ax.set_xlabel(
        "Kendall τ between Tₐ and T_b ranks", fontsize=FS_LABEL
    )
    ax.set_ylim(-0.7, 1.7)
    style_panel(ax, label="c")
    ax.legend(loc="upper left", fontsize=FS_TICK - 4, frameon=False)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
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


if __name__ == "__main__":
    main()
