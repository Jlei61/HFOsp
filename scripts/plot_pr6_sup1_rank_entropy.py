#!/usr/bin/env python3
"""PR-6-sup1 figures — first-rank entropy / symmetry-breaking diagnostic.

Plan v3: docs/archive/topic1/pr6_template_anchoring/
         pr6_supplementary_rank_entropy_plan_2026-05-10.md §9
Extension agreed 2026-05-10:
  + cohort H_p_norm overlay on normalized rank position x ∈ [0, 1]
  + swap subset (strict + candidate) per-subject panels

Inputs (read after cohort run):
  results/interictal_propagation/pr6_sup1_rank_entropy/per_subject/<stem>.json
  results/interictal_propagation/pr6_sup1_rank_entropy/cohort_summary.json
  results/interictal_propagation/rank_displacement/cohort_summary.json (for swap_class)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


SUP1_DIR = (
    WORKTREE_ROOT / "results" / "interictal_propagation" / "pr6_sup1_rank_entropy"
)
PER_SUBJECT_DIR = SUP1_DIR / "per_subject"
COHORT_PATH = SUP1_DIR / "cohort_summary.json"
RD_COHORT = (
    WORKTREE_ROOT
    / "results"
    / "interictal_propagation"
    / "rank_displacement"
    / "cohort_summary.json"
)
FIG_DIR = SUP1_DIR / "figures"

SWAP_COLORS = {
    "strict": COL_SIG,         # rust
    "candidate": COL_NEUTRAL,  # dust
    "none": COL_NONSIG,        # gray
    "unknown": "#CFCFCF",
}

X_GRID = np.linspace(0.0, 1.0, 51)  # 51-point common grid for interpolation


def _load_per_subject() -> List[dict]:
    out: List[dict] = []
    for jp in sorted(PER_SUBJECT_DIR.glob("*.json")):
        d = json.loads(jp.read_text())
        if d.get("exit_reason") and not d.get("clusters"):
            continue
        out.append(d)
    return out


def _interp_to_grid(H_p_norm: List[float], n_valid: int) -> np.ndarray:
    """Interpolate H_p_norm onto X_GRID using normalized rank position.

    Source x: (p - 1) / (n_valid - 1) for p = 1, ..., n_valid.
    """
    src_x = np.linspace(0.0, 1.0, n_valid)
    return np.interp(X_GRID, src_x, np.asarray(H_p_norm, dtype=float))


def _collect_curves(records: List[dict]) -> Dict[str, Any]:
    """Pull H_p_norm interpolated curves per cluster + Δ + meta per subject."""
    curves_by_cluster: Dict[str, List[Tuple[str, np.ndarray]]] = {"0": [], "1": []}
    delta_per_cluster: Dict[str, List[float]] = {"0": [], "1": []}
    delta_subject: List[float] = []
    swap_class_subject: List[str] = []
    n_valid_subject: List[int] = []
    drop_rate_subject: List[float] = []
    stems: List[str] = []
    is_max_subject: List[bool] = []
    pct_subject: List[float] = []

    for d in records:
        stem = d["stem"]
        swap = d.get("swap_class_full") or "unknown"
        n_valid = int(d.get("n_valid_channels") or 0)
        n_valid_subject.append(n_valid)
        swap_class_subject.append(swap)
        stems.append(stem)

        # cluster-level
        cluster_drops = []
        for cid, c in d["clusters"].items():
            H = c.get("H_p_norm")
            if H is None:
                continue
            curves_by_cluster[cid].append((stem, _interp_to_grid(H, n_valid)))
            if c.get("delta") is not None and np.isfinite(c["delta"]):
                delta_per_cluster[cid].append(float(c["delta"]))
            dr = c.get("drop_rate_k")
            if dr is not None and np.isfinite(dr):
                cluster_drops.append(float(dr))
        drop_rate_subject.append(
            float(np.mean(cluster_drops)) if cluster_drops else float("nan")
        )

        sl = d.get("subject_level") or {}
        delta_subject.append(
            float(sl.get("delta_obs_subject"))
            if sl.get("delta_obs_subject") is not None
            and np.isfinite(sl.get("delta_obs_subject"))
            else float("nan")
        )
        is_max_subject.append(bool(sl.get("is_subject_combo_max")))
        pct_subject.append(
            float(sl.get("subject_combo_percentile"))
            if sl.get("subject_combo_percentile") is not None
            and np.isfinite(sl.get("subject_combo_percentile"))
            else float("nan")
        )

    return {
        "curves_by_cluster": curves_by_cluster,
        "delta_per_cluster": delta_per_cluster,
        "delta_subject": np.asarray(delta_subject, dtype=float),
        "swap_class_subject": swap_class_subject,
        "n_valid_subject": np.asarray(n_valid_subject, dtype=int),
        "drop_rate_subject": np.asarray(drop_rate_subject, dtype=float),
        "stems": stems,
        "is_max_subject": is_max_subject,
        "pct_subject": np.asarray(pct_subject, dtype=float),
    }


# ---------------------------------------------------------------------------
# Figure 1 (主): H_p_norm cohort overlay on normalized rank position
# ---------------------------------------------------------------------------
def fig_cohort_overlay_normalized(data: Dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)

    for cid, ax in zip(["0", "1"], axes):
        curves = data["curves_by_cluster"][cid]
        if not curves:
            ax.set_title(f"cluster {cid} — no eligible subjects", fontsize=FS_TITLE - 1)
            continue

        # Match by stem to swap_class
        stem_to_swap = dict(zip(data["stems"], data["swap_class_subject"]))
        # Plot individual curves colored by swap_class
        for stem, y in curves:
            cls = stem_to_swap.get(stem, "unknown")
            color = SWAP_COLORS.get(cls, COL_NEUTRAL)
            ax.plot(X_GRID, y, color=color, alpha=0.30, linewidth=1.0, zorder=2)

        # Cohort median + IQR
        Y = np.stack([y for _, y in curves], axis=0)
        med = np.median(Y, axis=0)
        q25 = np.percentile(Y, 25, axis=0)
        q75 = np.percentile(Y, 75, axis=0)
        ax.fill_between(X_GRID, q25, q75, color="black", alpha=0.10, zorder=3)
        ax.plot(X_GRID, med, color="black", linewidth=2.5, zorder=4,
                label=f"cohort median (n={len(curves)})")

        # Reference: full-entropy ceiling at 1.0
        ax.axhline(1.0, ls="--", lw=0.8, color="gray", alpha=0.6)

        style_panel(ax)
        ax.set_title(f"Cluster {cid}", fontsize=FS_TITLE)
        ax.set_xlabel("normalized rank position  (0 = fastest, 1 = slowest)",
                      fontsize=FS_LABEL)
        if cid == "0":
            ax.set_ylabel("H_p_norm  (channel-identity entropy at rank position)",
                          fontsize=FS_LABEL)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.05)

        # Legend per axis
        legend_handles = [
            plt.Line2D([0], [0], color=SWAP_COLORS["strict"], lw=2,
                       label=f"§8 strict (n={data['swap_class_subject'].count('strict')})"),
            plt.Line2D([0], [0], color=SWAP_COLORS["candidate"], lw=2,
                       label=f"§8 candidate (n={data['swap_class_subject'].count('candidate')})"),
            plt.Line2D([0], [0], color=SWAP_COLORS["none"], lw=2,
                       label=f"§8 none (n={data['swap_class_subject'].count('none')})"),
            plt.Line2D([0], [0], color="black", lw=2.5,
                       label="cohort median"),
        ]
        ax.legend(handles=legend_handles, loc="lower center", fontsize=FS_TICK - 1,
                  frameon=True, ncol=2)

    fig.suptitle(
        "PR-6-sup1 — H_p_norm vs normalized rank position "
        "(roof-shape = endpoints determined, middle jittery)",
        fontsize=FS_TITLE,
    )
    fig.tight_layout()

    out = FIG_DIR / "H_p_norm_cohort_overlay.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Δ_subject by swap_class box plot
# ---------------------------------------------------------------------------
def fig_delta_by_swapclass_box(data: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    classes = ["strict", "candidate", "none"]
    positions = list(range(1, len(classes) + 1))

    by_class: Dict[str, List[float]] = {c: [] for c in classes}
    for delta, cls in zip(data["delta_subject"], data["swap_class_subject"]):
        if cls in by_class and np.isfinite(delta):
            by_class[cls].append(float(delta))

    box_data = [np.asarray(by_class[c]) for c in classes]
    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55, patch_artist=True,
        medianprops=dict(linewidth=2.2, color="black"),
    )
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(SWAP_COLORS[c])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    rng = np.random.RandomState(0)
    for pos, c in zip(positions, classes):
        vals = by_class[c]
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.array([pos] * len(vals)) + jitter, vals,
            color=SWAP_COLORS[c], edgecolors="white", linewidths=0.6,
            s=55, zorder=3, alpha=0.95,
        )

    ax.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.5,
               label="Δ = 0 (no endpoint vs middle gap)")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{c}\n(n={len(by_class[c])})" for c in classes], fontsize=FS_TICK,
    )
    style_panel(ax)
    ax.set_title(
        f"Δ_subject by §8 swap_class (n={int(np.sum(np.isfinite(data['delta_subject'])))})",
        fontsize=FS_TITLE,
    )
    ax.set_xlabel("§8 swap_class", fontsize=FS_LABEL)
    ax.set_ylabel(
        "Δ_subject  =  mean(H endpoints) − mean(H middle)\n"
        "(positive = confluence prediction; negative = endpoints determined)",
        fontsize=FS_LABEL - 1,
    )
    ax.legend(loc="upper right", fontsize=FS_TICK - 1, frameon=True)

    out = FIG_DIR / "delta_by_swapclass_box.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: endpoint_pair_percentile_panel (3-panel, plan v3 §9)
# ---------------------------------------------------------------------------
def fig_endpoint_percentile_panel(data: Dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # --- Panel A: percentile vs Δ_subject scatter ---
    ax = axes[0]
    for delta, pct, cls in zip(
        data["delta_subject"], data["pct_subject"], data["swap_class_subject"]
    ):
        if not np.isfinite(delta) or not np.isfinite(pct):
            continue
        color = SWAP_COLORS.get(cls, COL_NEUTRAL)
        ax.scatter(pct, delta, color=color, edgecolors="black",
                   linewidths=0.7, s=80, alpha=0.92)
    ax.axvline(1.0, ls="--", lw=1.0, color="black", alpha=0.5)
    ax.axhline(0.0, ls=":", lw=0.8, color="gray", alpha=0.6)
    style_panel(ax)
    ax.set_title("A. subject_combo_percentile × Δ_subject", fontsize=FS_TITLE - 1)
    ax.set_xlabel("subject_combo_percentile  (1.0 = endpoint pair is max)",
                  fontsize=FS_LABEL - 1)
    ax.set_ylabel("Δ_subject", fontsize=FS_LABEL)
    ax.set_xlim(-0.05, 1.05)

    # --- Panel B: n_valid distribution + min_attainable_p_N1 floor ---
    ax = axes[1]
    n_valids = data["n_valid_subject"]
    unique_nv = sorted(set(int(x) for x in n_valids))
    counts = [int(np.sum(n_valids == nv)) for nv in unique_nv]
    bars = ax.bar(unique_nv, counts, color=COL_NEUTRAL, edgecolor="black",
                  linewidth=1.2, alpha=0.85)
    ax2 = ax.twinx()
    floors = [1.0 / (nv * (nv - 1) // 2) for nv in unique_nv]  # cluster-level
    ax2.plot(unique_nv, floors, "o-", color=COL_SIG, lw=1.8, markersize=9,
             label="cluster-level min_attainable_p_N1")
    ax2.axhline(0.05, ls="--", lw=1.0, color="black", alpha=0.5,
                label="conventional 0.05 line")
    ax2.set_ylabel("cluster-level min_attainable_p_N1\n(1 / C(n_valid, 2))",
                   fontsize=FS_LABEL - 1, color=COL_SIG)
    ax2.tick_params(axis="y", labelcolor=COL_SIG)
    ax2.legend(loc="upper right", fontsize=FS_TICK - 2, frameon=True)
    style_panel(ax)
    ax.set_title("B. n_valid distribution + min_attainable_p_N1 floor",
                 fontsize=FS_TITLE - 1)
    ax.set_xlabel("n_valid (cluster channels)", fontsize=FS_LABEL)
    ax.set_ylabel("subject count", fontsize=FS_LABEL)

    # --- Panel C: is_subject_combo_max stacked bar by swap_class ---
    ax = axes[2]
    classes = ["strict", "candidate", "none"]
    counts_max: Dict[str, int] = {c: 0 for c in classes}
    counts_not: Dict[str, int] = {c: 0 for c in classes}
    for is_max, cls in zip(data["is_max_subject"], data["swap_class_subject"]):
        if cls not in counts_max:
            continue
        if is_max:
            counts_max[cls] += 1
        else:
            counts_not[cls] += 1
    positions = np.arange(len(classes))
    not_max_vals = [counts_not[c] for c in classes]
    max_vals = [counts_max[c] for c in classes]
    ax.bar(positions, not_max_vals, color=COL_NONSIG, edgecolor="black",
           linewidth=1.0, label="is_subject_combo_max = False")
    ax.bar(positions, max_vals, bottom=not_max_vals, color=COL_SIG,
           edgecolor="black", linewidth=1.0,
           label="is_subject_combo_max = True")
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, fontsize=FS_TICK)
    style_panel(ax)
    ax.set_title("C. is_subject_combo_max by §8 swap_class",
                 fontsize=FS_TITLE - 1)
    ax.set_xlabel("§8 swap_class", fontsize=FS_LABEL)
    ax.set_ylabel("subject count", fontsize=FS_LABEL)
    ax.legend(loc="upper right", fontsize=FS_TICK - 1, frameon=True)

    fig.suptitle(
        "PR-6-sup1 — endpoint pair percentile panel",
        fontsize=FS_TITLE,
    )
    fig.tight_layout()

    out = FIG_DIR / "endpoint_pair_percentile_panel.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: swap subset (strict + candidate) per-subject H_p_norm + delta
# ---------------------------------------------------------------------------
def fig_swap_subset_per_subject(records: List[dict]) -> Path:
    swap_records = [
        d for d in records
        if d.get("swap_class_full") in {"strict", "candidate"}
    ]
    if not swap_records:
        return FIG_DIR / "swap_subset_per_subject.png"  # nothing to plot

    swap_records.sort(
        key=lambda d: (
            0 if d.get("swap_class_full") == "strict" else 1,
            d["stem"],
        )
    )
    n_subj = len(swap_records)
    n_cols = 6
    n_rows = (n_subj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), sharey=True
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, d in enumerate(swap_records):
        ax = axes[idx // n_cols, idx % n_cols]
        n_valid = int(d["n_valid_channels"])
        cls = d["swap_class_full"]
        sl = d.get("subject_level") or {}
        d_subj = sl.get("delta_obs_subject")
        is_max = sl.get("is_subject_combo_max")
        pct = sl.get("subject_combo_percentile")

        for cid, c in d["clusters"].items():
            H = c.get("H_p_norm")
            if H is None:
                continue
            x = np.linspace(0.0, 1.0, len(H))
            ax.plot(x, H, marker="o", markersize=3.5, lw=1.4,
                    color=COL_YUQUAN if cid == "0" else COL_EPILEPSIAE,
                    label=f"cluster {cid}", alpha=0.9)
        ax.axhline(1.0, ls="--", lw=0.6, color="gray", alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.4, 1.05)
        ax.tick_params(labelsize=FS_TICK - 3)
        if d_subj is None or pct is None:
            subtitle = "subject-level excluded"
        else:
            subtitle = f"Δ_subj={d_subj:.3f}, pct={pct:.3f}, max={is_max}"
        ax.set_title(
            f"{d['stem']} ({cls}, n_valid={n_valid})\n{subtitle}",
            fontsize=FS_TICK - 1,
        )
        if idx == 0:
            ax.legend(fontsize=FS_TICK - 3, loc="lower center")

    # Hide unused panels
    for idx in range(n_subj, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle(
        f"PR-6-sup1 — swap subset H_p_norm per subject (n={n_subj}: "
        f"{sum(1 for d in swap_records if d['swap_class_full'] == 'strict')} strict + "
        f"{sum(1 for d in swap_records if d['swap_class_full'] == 'candidate')} candidate)",
        fontsize=FS_TITLE,
    )
    fig.tight_layout()

    out = FIG_DIR / "swap_subset_per_subject.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 (bridge): F_norm × subject_combo_percentile scatter
# ---------------------------------------------------------------------------
def fig_bridge_with_rank_displacement(data: Dict[str, Any]) -> Optional[Path]:
    if not RD_COHORT.exists():
        return None
    rd = json.loads(RD_COHORT.read_text())
    rd_lookup: Dict[str, Dict[str, Any]] = {}
    for r in rd:
        if r.get("stable_k") != 2:
            continue
        stem = f"{r['dataset']}_{r['subject']}"
        # F_norm is on the single pair for stable_k=2
        pairs = r.get("pairs") or []
        if not pairs:
            continue
        rd_lookup[stem] = {
            "F_norm": pairs[0].get("footrule_normalized"),
            "fwd_rev": bool(r.get("fwd_rev_reproduced")),
        }

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for stem, pct, cls in zip(
        data["stems"], data["pct_subject"], data["swap_class_subject"]
    ):
        if stem not in rd_lookup:
            continue
        F = rd_lookup[stem]["F_norm"]
        if F is None or not np.isfinite(F):
            continue
        if not np.isfinite(pct):
            continue
        color = SWAP_COLORS.get(cls, COL_NEUTRAL)
        marker = "*" if rd_lookup[stem]["fwd_rev"] else "o"
        size = 160 if rd_lookup[stem]["fwd_rev"] else 90
        ax.scatter(F, pct, color=color, edgecolors="black",
                   linewidths=0.8, s=size, marker=marker, alpha=0.92)

    ax.axhline(1.0, ls="--", lw=1.0, color="black", alpha=0.4,
               label="endpoint pair = max (sup1)")
    ax.axvline(2 / 3, ls="--", lw=1.0, color=COL_SIG, alpha=0.6,
               label="2/3 random reversal floor (rd)")
    style_panel(ax)
    ax.set_title(
        "PR-6-sup1 × rank_displacement bridge",
        fontsize=FS_TITLE,
    )
    ax.set_xlabel("rank_displacement F_norm  (0 = no swap, 1 = full reversal)",
                  fontsize=FS_LABEL - 1)
    ax.set_ylabel("sup1 subject_combo_percentile  (1 = endpoint pair is max)",
                  fontsize=FS_LABEL - 1)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.05)

    legend_extra = [
        plt.Line2D([0], [0], color=SWAP_COLORS["strict"], marker="o", lw=0,
                   markersize=10, label="§8 strict"),
        plt.Line2D([0], [0], color=SWAP_COLORS["candidate"], marker="o", lw=0,
                   markersize=10, label="§8 candidate"),
        plt.Line2D([0], [0], color=SWAP_COLORS["none"], marker="o", lw=0,
                   markersize=10, label="§8 none"),
        plt.Line2D([0], [0], color=COL_NEUTRAL, marker="*", lw=0,
                   markersize=14, label="rd fwd/rev reproduced"),
    ]
    ax.legend(handles=legend_extra, loc="lower right", fontsize=FS_TICK - 1,
              frameon=True)

    out = FIG_DIR / "bridge_rank_displacement.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_per_subject()
    data = _collect_curves(records)

    p1 = fig_cohort_overlay_normalized(data)
    p2 = fig_delta_by_swapclass_box(data)
    p3 = fig_endpoint_percentile_panel(data)
    p4 = fig_swap_subset_per_subject(records)
    p5 = fig_bridge_with_rank_displacement(data)

    print(f"Wrote {p1.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p2.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p3.relative_to(WORKTREE_ROOT)}")
    print(f"Wrote {p4.relative_to(WORKTREE_ROOT)}")
    if p5:
        print(f"Wrote {p5.relative_to(WORKTREE_ROOT)}")


if __name__ == "__main__":
    main()
