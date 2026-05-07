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
    f_norms = np.array(
        [r["primary_pair"]["footrule_normalized"] for r in sorted_records],
        dtype=float,
    )
    # Kendall τ track removed: ρ(F_norm, τ) ≈ -0.92 in this cohort, the
    # τ bars are visually a mirror of the F_norm bars and add no new info.
    # τ values still computed in run_rank_displacement and recorded in
    # per-subject JSON / archive doc.
    # SOZ contribution_excess is intentionally NOT plotted on the figure -
    # SOZ definition / channel coverage in the lagPat selected set is not
    # yet stable enough for a paper-level claim (see archive doc §5.1, §6).

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    # Layout:
    #   [ heatmap (top x-axis)  | F_norm track ]   <- main row
    #   [ horizontal colorbar   | SOZ legend   ]   <- bottom row
    # τ track removed (highly collinear with F_norm: ρ ≈ -0.92);
    # SOZ track removed (lagPat / SOZ coverage not yet stable for paper).
    fig = plt.figure(figsize=(13.5, max(8.5, 0.46 * n_sub) + 2.0))
    # Two-row layout: heatmap+F_norm / colorbar.
    # Legend is placed as a separate axes above the heatmap (between
    # suptitle and the heatmap's top xticks) so it sits at the natural
    # entry point of the reader's gaze.
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[7.0, 1.3],
        height_ratios=[1.0, 0.045],
        wspace=0.14,
        hspace=0.06,
        top=0.84,
        bottom=0.07,
        left=0.11,
        right=0.97,
    )
    ax_h = fig.add_subplot(gs[0, 0])
    ax_F = fig.add_subplot(gs[0, 1], sharey=ax_h)
    ax_cb = fig.add_subplot(gs[1, 0])  # horizontal colorbar under heatmap
    # Legend axes positioned in the gap between suptitle (y~0.96) and
    # the heatmap top xticks/xlabel (top edge of heatmap = 0.84). Centered
    # on the heatmap column [0.11, 0.78].
    ax_legend = fig.add_axes([0.20, 0.91, 0.55, 0.035])
    ax_legend.axis("off")

    # NaN cells (channels beyond a subject's n_valid) render as white via
    # cmap.set_bad("white"). The divergent palette around 0 is light
    # pink/blue (not pure white), so data and NaN regions remain
    # visually distinguishable.
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")
    im = ax_h.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # SOZ overlay deliberately removed: SOZ definition / lagPat coverage
    # not stable enough for paper-level annotation. soz_overlay is still
    # computed in build_heatmap_matrix and recorded in per-subject JSON.

    # === Variable-k swap endpoint markers (FW max-null dual-tier) ===
    # Triangle markers at the decision_k boundary cells:
    #   left  marker at column = decision_k - 1   ('>' pointing rightward)
    #   right marker at column = n_valid - decision_k ('<' pointing leftward)
    # Together '> ... <' visually signals the inward exchange = swap.
    # Strict (p_fw < 0.05)    -> filled black '>' / '<'
    # Candidate (p_fw < 0.20) -> open grey '>' / '<'
    # Asterisks deliberately NOT used: '*' / '**' are the statistical
    # convention for p<0.05 / p<0.01, and our candidate threshold (p<0.20)
    # would mislead readers if mapped to '*'.
    n_strict_drawn = 0
    n_cand_drawn = 0
    for row_i, r in enumerate(sorted_records):
        sw = r["primary_pair"].get("swap_sweep") or {}
        if sw.get("exit_reason") != "ok":
            continue
        cls = sw.get("swap_class", "none")
        if cls == "none":
            continue
        dk = int(sw["decision_k"])
        n_v = int(sw["n_valid"])
        if 2 * dk > n_v:
            continue
        if cls == "strict":
            face, edge, lw, marker_size = "black", "black", 0.8, 100
            n_strict_drawn += 1
        else:  # candidate
            face, edge, lw, marker_size = "none", "0.30", 1.5, 90
            n_cand_drawn += 1
        # Left boundary: ">" pointing rightward (toward center)
        ax_h.scatter(
            [dk - 1], [row_i], marker=">", s=marker_size,
            facecolors=face, edgecolors=edge, linewidths=lw, zorder=6,
        )
        # Right boundary: "<" pointing leftward (toward center)
        ax_h.scatter(
            [n_v - dk], [row_i], marker="<", s=marker_size,
            facecolors=face, edgecolors=edge, linewidths=lw, zorder=6,
        )

    ax_h.set_yticks(range(n_sub))
    ax_h.set_yticklabels(sub_labels, fontsize=FS_TICK + 2)
    # Move heatmap x-axis (ticks + label) to the TOP, since the bottom row
    # is now occupied by the horizontal colorbar.
    ax_h.xaxis.tick_top()
    ax_h.xaxis.set_label_position("top")
    ax_h.set_xticks([0, n_ch - 1])
    ax_h.set_xticklabels(
        ["source\n(earliest in T_a)", "sink\n(latest in T_a)"],
        fontsize=FS_TICK + 4,
    )
    ax_h.set_xlabel(
        "Channel position along T_a (source → sink)",
        fontsize=FS_LABEL + 3,
        labelpad=12,
    )
    # Hide top + right spines (cleaner paper-style; xticks at top still draw)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    ax_h.tick_params(axis="x", which="both", bottom=False, top=True)

    # === F_norm summary track (right of heatmap, shared y) ===
    # Bar length encodes continuous F_norm. To make 2/3 visible as the
    # asymptotic random null (Diaconis-Graham 1977), shade the [0, 2/3]
    # range as a soft "null zone" background — bars whose tip lies inside
    # the shaded band are at-or-below random; bars extending past it are
    # above random expectation. NOT a per-subject classification: the
    # bars themselves are still single-color.
    ax_F.axvspan(0, 2 / 3, color="lightgray", alpha=0.45, zorder=0)
    ax_F.barh(
        range(n_sub), f_norms,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.5,
        zorder=2,
    )
    # Prominent 2/3 reference line (thicker rust dashed for paper-level clarity)
    ax_F.axvline(2 / 3, color=COL_SIG, linewidth=3.2, linestyle="--",
                 zorder=3)
    ax_F.set_xlim(0, 1.05)
    ax_F.set_xticks([0, 2 / 3, 1])
    ax_F.set_xticklabels(["0", "2/3", "1"], fontsize=FS_TICK + 2)
    # Match heatmap: x-axis at top
    ax_F.xaxis.tick_top()
    ax_F.xaxis.set_label_position("top")
    ax_F.tick_params(axis="x", which="both", bottom=False, top=True)
    plt.setp(ax_F.get_yticklabels(), visible=False)
    ax_F.set_xlabel("F_norm", fontsize=FS_LABEL + 3, labelpad=12)
    ax_F.spines["top"].set_visible(False)
    ax_F.spines["right"].set_visible(False)
    # Random-null annotation removed: redundant with the dashed 2/3 reference,
    # the "2/3" xtick label, and the shaded null zone — three signals already
    # convey the same meaning. Removing the inline text frees the bottom
    # corner for the swap legend without overlap.

    # Horizontal colorbar directly under heatmap (close, not separated)
    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("Signed Δr  (= rank_T_b − rank_T_a)",
                 fontsize=FS_LABEL + 2, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK + 1)

    # Swap marker legend at top of figure (between suptitle and heatmap
    # xticks). Single horizontal row: "> < strict (n, p<0.05)  > < candidate (n, p<0.20)".
    # Markers are plotted via scatter at fixed x positions inside ax_legend
    # so the visual mirrors what appears on the heatmap.
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    # Strict swatch (filled black > <) + label
    ax_legend.scatter([0.06], [0.5], marker=">", s=110,
                      facecolors="black", edgecolors="black",
                      linewidths=0.8, transform=ax_legend.transAxes)
    ax_legend.scatter([0.10], [0.5], marker="<", s=110,
                      facecolors="black", edgecolors="black",
                      linewidths=0.8, transform=ax_legend.transAxes)
    ax_legend.text(0.13, 0.5,
                   f"strict (n={n_strict_drawn},  p_fw < 0.05)",
                   ha="left", va="center",
                   fontsize=FS_TICK + 2, fontweight="bold",
                   transform=ax_legend.transAxes)
    # Candidate swatch (open grey > <) + label
    ax_legend.scatter([0.55], [0.5], marker=">", s=100,
                      facecolors="none", edgecolors="0.30",
                      linewidths=1.5, transform=ax_legend.transAxes)
    ax_legend.scatter([0.59], [0.5], marker="<", s=100,
                      facecolors="none", edgecolors="0.30",
                      linewidths=1.5, transform=ax_legend.transAxes)
    ax_legend.text(0.62, 0.5,
                   f"candidate (n={n_cand_drawn},  p_fw < 0.20)",
                   ha="left", va="center",
                   fontsize=FS_TICK + 2, fontweight="bold", color="0.20",
                   transform=ax_legend.transAxes)

    fig.suptitle(
        f"Per-channel signed rank displacement — two-template subjects (n={n_sub})",
        fontsize=FS_TITLE + 4,
        y=0.96,
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


def plot_swap_cardinality_heatmap(records: List[dict], out_stem: Path) -> None:
    """Subject × k swap_score heatmap (user-locked design 2026-05-07 v2).

    Layout:
      - Main heatmap: rows = subjects in F_norm-descending order (matches
        cohort_displacement_heatmap row ordering exactly); columns = absolute
        k = 2 .. global_k_max; cell color = swap_score(k); cells with
        k > floor(n_valid/2) drawn white (NaN; cmap.set_bad).
      - Black-ring marker on decision_k cell when has_swap=True. decision_k
        = argmax_k swap_score(k) (smallest-k tie); under FW max-null this
        is the single k that defines T_obs = max swap_score.
      - Right narrow F_norm reference track (random null at 2/3).
      - Bottom horizontal colorbar for swap_score.

    Decision (FW-corrected): has_swap iff T_obs >= score_floor AND p_fw < alpha_fw,
    where T_obs = max_k swap_score and p_fw uses the max-null distribution
    over permuted rank_b. Per-k null_95th is descriptive only and NOT a gate.
    """
    swap_records = []
    for r in records:
        sw = r["primary_pair"].get("swap_sweep") or {}
        if sw.get("exit_reason") != "ok":
            continue
        swap_records.append((r, sw))

    # Row order = F_norm desc (same as Panel A)
    swap_records.sort(
        key=lambda rs: -rs[0]["primary_pair"].get("footrule_normalized", 0.0)
    )

    n_rows = len(swap_records)
    k_max_global = max(sw["k_max"] for _, sw in swap_records)
    n_cols = k_max_global - 1  # k = 2 .. k_max_global

    # Build matrix; NaN where k > floor(n_valid/2)
    matrix = np.full((n_rows, n_cols), np.nan)
    decision_marks: List[Tuple[int, int]] = []
    for i, (r, sw) in enumerate(swap_records):
        for k_str, v in sw["swap_score_by_k"].items():
            k = int(k_str)
            matrix[i, k - 2] = v
        if sw["has_swap"] and sw.get("decision_k") is not None:
            decision_marks.append((i, int(sw["decision_k"]) - 2))

    sub_labels = [
        f"{r['dataset'][:3]}_{r['subject']}" for r, _ in swap_records
    ]
    f_norm_vals = [
        r["primary_pair"].get("footrule_normalized", float("nan"))
        for r, _ in swap_records
    ]

    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[10.0, 1.4, 0.4],
        height_ratios=[14.0, 0.5],
        hspace=0.10, wspace=0.10,
        top=0.86, bottom=0.06, left=0.10, right=0.97,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_fnorm = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_cb = fig.add_subplot(gs[1, 0])

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    im = ax_main.imshow(
        matrix, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest", origin="upper",
    )

    # decision_k black-ring markers
    for (row, col) in decision_marks:
        ax_main.scatter(
            col, row, s=70,
            facecolors="none", edgecolors="black", linewidths=1.6,
            zorder=5,
        )

    # X axis on top; absolute k tick labels
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels(
        [str(k) for k in range(2, k_max_global + 1)], fontsize=FS_TICK - 1
    )
    ax_main.set_xlabel("endpoint cardinality k", fontsize=FS_LABEL)
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels(sub_labels, fontsize=FS_TICK - 1)
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)

    # F_norm right track (shared y) - bar chart
    ax_fnorm.barh(
        range(n_rows), f_norm_vals,
        color=COL_NEUTRAL, edgecolor="black", linewidth=0.4, height=0.82,
    )
    ax_fnorm.axvline(2 / 3, ls="--", color="grey", lw=1.0, zorder=2)
    ax_fnorm.set_xlim(0, 1.05)
    ax_fnorm.set_xticks([0.0, 0.5, 1.0])
    ax_fnorm.set_xticklabels(["0", "0.5", "1"], fontsize=FS_TICK - 2)
    ax_fnorm.xaxis.set_ticks_position("top")
    ax_fnorm.xaxis.set_label_position("top")
    ax_fnorm.set_xlabel("F_norm\n(2/3 = null)", fontsize=FS_LABEL - 3)
    plt.setp(ax_fnorm.get_yticklabels(), visible=False)
    for spine in ("top", "right", "left"):
        ax_fnorm.spines[spine].set_visible(False)

    # Bottom horizontal colorbar
    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("swap_score(k)  ·  black ring = decision_k (has_swap=True)",
                 fontsize=FS_LABEL - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 1)

    n_swap = sum(1 for _, sw in swap_records if sw["has_swap"])
    fig.suptitle(
        f"Variable-k swap_score per subject — stable_k=2 cohort "
        f"(n={n_rows}, has_swap = {n_swap})\n"
        f"FW max-null: T_obs = max_k swap_score(k); has_swap iff "
        f"T_obs ≥ 0.5 AND p_fw < 0.05  (1000 perm, seed=0)",
        fontsize=FS_TITLE - 2, y=0.97,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI_PUB,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--what",
        default="all",
        choices=["all", "cohort", "per_subject", "swap"],
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Subject stems (<dataset>_<subject>) to exclude from cohort heatmap. "
             "Output filename gets a _excl_<slug> suffix; per_subject strips unaffected.",
    )
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_SUB_FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = load_cohort_records()
    print(f"Loaded {len(records)} stable_k=2 subjects")

    if args.what in ("all", "cohort"):
        if args.exclude:
            excl = set(args.exclude)
            kept = [r for r in records if f"{r['dataset']}_{r['subject']}" not in excl]
            slug = "_".join(s.split("_", 1)[1] for s in sorted(excl))
            out_stem = FIG_DIR / f"cohort_displacement_heatmap_excl_{slug}"
            plot_cohort_heatmap(kept, out_stem)
            print(f"Wrote {out_stem.name}.{{png,pdf}} (n={len(kept)}, excluded {sorted(excl)})")
        else:
            plot_cohort_heatmap(records, FIG_DIR / "cohort_displacement_heatmap")
            print("Wrote cohort_displacement_heatmap.{png,pdf}")

    if args.what in ("all", "per_subject"):
        for r in records:
            stem = f"{r['dataset']}_{r['subject']}"
            plot_per_subject_strip(
                r, PER_SUB_FIG_DIR / f"{stem}_displacement.png"
            )
        print(f"Wrote per-subject strips for {len(records)} subjects")

    if args.what in ("all", "swap"):
        plot_swap_cardinality_heatmap(
            records, FIG_DIR / "swap_cardinality_heatmap"
        )
        n_swap = sum(
            1 for r in records
            if (r["primary_pair"].get("swap_sweep") or {}).get("has_swap")
        )
        print(
            f"Wrote swap_cardinality_heatmap.{{png,pdf}} "
            f"(n={len(records)}, has_swap = {n_swap})"
        )
        # Retire old scatter+curves figure: remove if still on disk
        old_paths = [
            FIG_DIR / f"swap_classification.{ext}" for ext in ("png", "pdf")
        ]
        for p in old_paths:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
