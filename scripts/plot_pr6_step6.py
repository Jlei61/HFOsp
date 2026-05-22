#!/usr/bin/env python3
"""PR-6 Step 6 figures — held-out time template stability.

Plan: docs/archive/topic1/pr6_template_anchoring/
      pr6_step6_held_out_template_plan_2026-05-10.md §8

Generates four figures from
``results/interictal_propagation/pr6_step6_held_out_template/cohort_summary.json``:

1. tier_distribution_bar.{png,pdf}
2. template_spearman_recall_box.{png,pdf}    (renamed from plan's _jaccard_box)
3. endpoint_position_recall_scatter.{png,pdf}
4. swap_class_transitions.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE_ROOT))

from src.plot_style import (  # noqa: E402
    COL_EPILEPSIAE,
    COL_NEUTRAL,
    COL_NONSIG,
    COL_SIG,
    COL_YUQUAN,
    DPI_PUB,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    savefig_pub,
    style_panel,
)


COHORT_PATH = (
    WORKTREE_ROOT
    / "results"
    / "interictal_propagation"
    / "pr6_step6_held_out_template"
    / "cohort_summary.json"
)
RD_PATH = (
    WORKTREE_ROOT
    / "results"
    / "interictal_propagation"
    / "rank_displacement"
    / "cohort_summary.json"
)
FIG_DIR = (
    WORKTREE_ROOT
    / "results"
    / "interictal_propagation"
    / "pr6_step6_held_out_template"
    / "figures"
)


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree."""
    global COHORT_PATH, RD_PATH, FIG_DIR
    COHORT_PATH = (
        WORKTREE_ROOT
        / "results"
        / "interictal_propagation_masked"
        / "pr6_step6_held_out_template"
        / "cohort_summary.json"
    )
    RD_PATH = (
        WORKTREE_ROOT
        / "results"
        / "interictal_propagation_masked"
        / "rank_displacement"
        / "cohort_summary.json"
    )
    FIG_DIR = (
        WORKTREE_ROOT
        / "results"
        / "interictal_propagation_masked"
        / "pr6_step6_held_out_template"
        / "figures"
    )

SWAP_COLORS = {
    "strict": COL_SIG,         # rust
    "candidate": COL_NEUTRAL,  # dust
    "none": COL_NONSIG,        # gray
    "unknown": "#CFCFCF",
}
TIER_COLORS = {
    "strong": "#5B7E62",       # Morandi forest (green-ish)
    "moderate": "#C9A86A",     # mustard
    "weak": "#B07A6E",         # terracotta
    "fail": "#A35E48",         # rust (deeper)
}


def _load_swap_classes() -> Dict[str, str]:
    """Return stem -> swap_class (strict/candidate/none) using §8 dual-tier label."""
    rd = json.loads(RD_PATH.read_text())
    out: Dict[str, str] = {}
    for r in rd:
        if r.get("stable_k") != 2:
            continue
        stem = f"{r['dataset']}_{r['subject']}"
        cls = "none"
        for p in r.get("pairs") or []:
            sw = p.get("swap_sweep") or {}
            cand = sw.get("swap_class")
            if cand == "strict":
                cls = "strict"
                break
            if cand == "candidate":
                cls = "candidate"
        out[stem] = cls
    return out


def _load_records() -> List[dict]:
    raw = json.loads(COHORT_PATH.read_text())
    return [
        r
        for r in raw.get("subjects", [])
        if r.get("tier") in {"strong", "moderate", "weak", "fail"}
    ]


def _attach_swap_class(records: List[dict]) -> None:
    swap_lookup = _load_swap_classes()
    for r in records:
        r["swap_class_full"] = swap_lookup.get(r["stem"], "unknown")


# ---------------------------------------------------------------------------
# Figure 1: tier distribution bar
# ---------------------------------------------------------------------------
def fig_tier_distribution(records: List[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    tiers = ["strong", "moderate", "weak", "fail"]
    counts = [sum(1 for r in records if r["tier"] == t) for t in tiers]

    bars = ax.bar(
        tiers,
        counts,
        color=[TIER_COLORS[t] for t in tiers],
        edgecolor="black",
        linewidth=1.2,
        alpha=0.92,
    )
    for rect, c in zip(bars, counts):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.3,
            str(c),
            ha="center",
            va="bottom",
            fontsize=FS_LABEL,
            fontweight="bold",
        )

    style_panel(ax)
    ax.set_title(
        f"PR-6 Step 6 — held-out tier distribution (n={len(records)})",
        fontsize=FS_TITLE,
    )
    ax.set_xlabel("Tier", fontsize=FS_LABEL)
    ax.set_ylabel("Subject count", fontsize=FS_LABEL)
    ax.set_ylim(0, max(counts) * 1.20 + 1)

    out = FIG_DIR / "tier_distribution_bar.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: template_spearman + endpoint_position_recall box, by swap_class
# ---------------------------------------------------------------------------
def fig_spearman_recall_box(records: List[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    classes = ["strict", "candidate", "none"]
    by_class = {c: [r for r in records if r["swap_class_full"] == c] for c in classes}

    for ax, field, label in zip(
        axes,
        ["template_spearman", "endpoint_position_recall"],
        ["Template Spearman ρ\n(first vs second projected)",
         "Endpoint position recall\n(direction-preserving, baseline ≈ 0.30)"],
    ):
        positions = list(range(1, len(classes) + 1))
        data_by_class: List[np.ndarray] = []
        for c in classes:
            vals = np.asarray(
                [r[field] for r in by_class[c] if r.get(field) is not None],
                dtype=float,
            )
            data_by_class.append(vals)

        bp = ax.boxplot(
            data_by_class,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=False,
            medianprops=dict(linewidth=2.2, color="black"),
        )
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(SWAP_COLORS[c])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")

        for pos, c in zip(positions, classes):
            vals = np.asarray(
                [r[field] for r in by_class[c] if r.get(field) is not None],
                dtype=float,
            )
            jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(
                pos + jitter, vals,
                color=SWAP_COLORS[c],
                edgecolors="white",
                linewidths=0.6,
                s=42,
                zorder=3,
                alpha=0.95,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{c}\n(n={len(by_class[c])})" for c in classes],
            fontsize=FS_TICK,
        )
        if field == "template_spearman":
            ax.axhline(0.7, ls="--", lw=1.0, color="black", alpha=0.6)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.axhline(0.6, ls="--", lw=1.0, color="black", alpha=0.6)
            ax.axhline(0.3, ls=":", lw=1.0, color="black", alpha=0.4)
            ax.set_ylim(-0.05, 1.10)
        style_panel(ax)
        ax.set_title(label, fontsize=FS_TITLE - 1)
        ax.set_xlabel("§8 swap_class", fontsize=FS_LABEL)
        ax.set_ylabel(field.replace("_", " "), fontsize=FS_LABEL)

    fig.suptitle(
        f"PR-6 Step 6 — held-out validation by §8 swap_class (n={len(records)})",
        fontsize=FS_TITLE,
    )
    fig.tight_layout()

    out = FIG_DIR / "template_spearman_recall_box.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: per-subject scatter of template_spearman vs endpoint_position_recall,
# colored by tier, with strong-tier threshold lines
# ---------------------------------------------------------------------------
def fig_spearman_recall_scatter(records: List[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    for r in records:
        x = r.get("template_spearman")
        y = r.get("endpoint_position_recall")
        if x is None or y is None:
            continue
        marker = "o" if r["dataset"] == "epilepsiae" else "s"
        edge_color = (
            COL_SIG if r["swap_class_full"] == "strict"
            else (COL_NEUTRAL if r["swap_class_full"] == "candidate" else "white")
        )
        ax.scatter(
            x, y,
            s=120,
            color=TIER_COLORS[r["tier"]],
            edgecolors=edge_color,
            linewidths=2.0 if edge_color != "white" else 1.0,
            marker=marker,
            alpha=0.92,
            zorder=3,
        )

    ax.axvline(0.7, ls="--", lw=1.0, color="black", alpha=0.5)
    ax.axhline(0.6, ls="--", lw=1.0, color="black", alpha=0.5)
    ax.fill_betweenx([0.6, 1.10], 0.7, 1.05, color="#5B7E62", alpha=0.05)
    ax.text(
        0.86, 0.105,
        "strong-tier zone",
        fontsize=FS_TICK,
        color="#5B7E62",
        ha="center",
        transform=ax.transAxes,
    )

    style_panel(ax)
    ax.set_title(
        f"PR-6 Step 6 — Spearman vs recall by tier (n={len(records)})",
        fontsize=FS_TITLE,
    )
    ax.set_xlabel("Template Spearman ρ", fontsize=FS_LABEL)
    ax.set_ylabel("Endpoint position recall", fontsize=FS_LABEL)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.10)

    legend_elems = []
    for tier, col in TIER_COLORS.items():
        if any(r["tier"] == tier for r in records):
            legend_elems.append(
                plt.scatter([], [], color=col, edgecolor="black",
                            linewidths=1.0, s=120, label=tier)
            )
    legend_elems.append(
        plt.scatter([], [], color="white", edgecolor=COL_SIG,
                    linewidths=2.0, s=120, label="§8 strict")
    )
    legend_elems.append(
        plt.scatter([], [], color="white", edgecolor=COL_NEUTRAL,
                    linewidths=2.0, s=120, label="§8 candidate")
    )
    legend_elems.append(
        plt.scatter([], [], color=COL_NEUTRAL, marker="o", s=120, label="epilepsiae")
    )
    legend_elems.append(
        plt.scatter([], [], color=COL_NEUTRAL, marker="s", s=120, label="yuquan")
    )
    ax.legend(
        handles=legend_elems,
        loc="lower right",
        fontsize=FS_TICK - 1,
        frameon=True,
    )

    out = FIG_DIR / "endpoint_position_recall_scatter.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: swap_class concordance matrix (first × second_projected)
# ---------------------------------------------------------------------------
def fig_swap_class_transitions(records: List[dict]) -> Path:
    classes = ["strict", "candidate", "none"]
    M = np.zeros((3, 3), dtype=int)
    for r in records:
        first = r.get("swap_class_first")
        second = r.get("swap_class_second_projected")
        if first not in classes or second not in classes:
            continue
        M[classes.index(first), classes.index(second)] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(M, cmap="YlOrBr", vmin=0, vmax=max(M.max(), 1))
    for i in range(3):
        for j in range(3):
            txt_color = "black" if M[i, j] < M.max() * 0.6 else "white"
            ax.text(
                j, i, str(int(M[i, j])),
                ha="center", va="center",
                fontsize=FS_LABEL + 2,
                fontweight="bold",
                color=txt_color,
            )

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=FS_TICK)
    ax.set_yticklabels(classes, fontsize=FS_TICK)
    ax.set_xlabel("swap_class (second-half projected)", fontsize=FS_LABEL)
    ax.set_ylabel("swap_class (first-half train)", fontsize=FS_LABEL)
    ax.set_title(
        f"PR-6 Step 6 — swap_class concordance (diagonal = concordant; n={int(M.sum())})",
        fontsize=FS_TITLE - 1,
    )

    diag = int(np.trace(M))
    total = int(M.sum())
    frac = diag / max(total, 1)
    ax.text(
        0.5, -0.20,
        f"concordant = {diag}/{total} ({frac:.1%})",
        transform=ax.transAxes,
        ha="center",
        fontsize=FS_LABEL,
        fontweight="bold",
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()

    out = FIG_DIR / "swap_class_transitions.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PR-6 Step 6 held-out template figures")
    parser.add_argument(
        "--masked-features",
        action="store_true",
        help="Consume masked PR-6 Step 6 + rank_displacement cohort_summary "
             "under results/interictal_propagation_masked/ and write figures next "
             "to them. Mirrors scripts/plot_rank_displacement.py --masked-features.",
    )
    args = parser.parse_args()
    if args.masked_features:
        _apply_masked_paths()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_records()
    _attach_swap_class(records)

    p1 = fig_tier_distribution(records)
    p2 = fig_spearman_recall_box(records)
    p3 = fig_spearman_recall_scatter(records)
    p4 = fig_swap_class_transitions(records)

    print(f"Wrote {p1.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p2.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p3.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p4.relative_to(WORKTREE_ROOT)}")


if __name__ == "__main__":
    main()
