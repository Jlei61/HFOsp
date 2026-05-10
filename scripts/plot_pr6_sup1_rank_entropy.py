#!/usr/bin/env python3
"""PR-6-sup1 figures — first-rank entropy / symmetry-breaking diagnostic.

Plan v3: docs/archive/topic1/pr6_template_anchoring/
         pr6_supplementary_rank_entropy_plan_2026-05-10.md
Figure-design discipline 2026-05-10:
  * No internal codebase terminology in legend / labels (no §X, no
    cluster_id; describe what the variable represents scientifically).
  * Single shared legend per figure, not per-panel.
  * Tight axes — xlim(0,1) ylim(0,1) without decorative whitespace
    when ranges are naturally bounded.
  * Categorical colors must be perceptually distinguishable in print +
    grayscale.
  * Each panel title states the scientific question, not just the
    variable name.
  * fig.suptitle dropped unless it adds new context.

Reference (memory): feedback_figure_self_contained_paper_grade.md
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


# High-contrast categorical palette (perceptually distinct in print + grayscale).
# Maps the rank_displacement variable-k swap classifier label (strict /
# candidate / none) to a color, but the LEGEND text describes what the
# label scientifically represents (reversal strength), not the codebase
# section.
SWAP_COLORS = {
    "strict": "#C0392B",     # firebrick — strong reversal
    "candidate": "#2980B9",  # blue — suggestive reversal
    "none": "#7F8C8D",       # dark gray — no reversal
    "unknown": "#BDC3C7",
}
SWAP_LABEL_PLAIN = {
    "strict": "Strong reversal",
    "candidate": "Suggestive reversal",
    "none": "No reversal",
    "unknown": "Unclassified",
}

X_GRID = np.linspace(0.0, 1.0, 51)


def _load_per_subject() -> List[dict]:
    out: List[dict] = []
    for jp in sorted(PER_SUBJECT_DIR.glob("*.json")):
        d = json.loads(jp.read_text())
        if d.get("exit_reason") and not d.get("clusters"):
            continue
        out.append(d)
    return out


def _interp_to_grid(H_p_norm: List[float], n_valid: int) -> np.ndarray:
    src_x = np.linspace(0.0, 1.0, n_valid)
    return np.interp(X_GRID, src_x, np.asarray(H_p_norm, dtype=float))


def _collect_curves(records: List[dict]) -> Dict[str, Any]:
    """All cluster-level curves flattened into one list (KMeans cluster_id
    has no semantic ordering — see verification 2026-05-10: 16 vs 19 split
    on which cluster is larger across cohort)."""
    flat_curves: List[Tuple[str, str, np.ndarray]] = []  # (stem, swap, curve)
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

        cluster_drops = []
        for cid, c in d["clusters"].items():
            H = c.get("H_p_norm")
            if H is None:
                continue
            flat_curves.append((stem, swap, _interp_to_grid(H, n_valid)))
            dr = c.get("drop_rate_k")
            if dr is not None and np.isfinite(dr):
                cluster_drops.append(float(dr))
        drop_rate_subject.append(
            float(np.mean(cluster_drops)) if cluster_drops else float("nan")
        )

        sl = d.get("subject_level") or {}
        d_obs = sl.get("delta_obs_subject")
        delta_subject.append(
            float(d_obs) if d_obs is not None and np.isfinite(d_obs) else float("nan")
        )
        is_max_subject.append(bool(sl.get("is_subject_combo_max")))
        pct = sl.get("subject_combo_percentile")
        pct_subject.append(
            float(pct) if pct is not None and np.isfinite(pct) else float("nan")
        )

    return {
        "flat_curves": flat_curves,
        "delta_subject": np.asarray(delta_subject, dtype=float),
        "swap_class_subject": swap_class_subject,
        "n_valid_subject": np.asarray(n_valid_subject, dtype=int),
        "drop_rate_subject": np.asarray(drop_rate_subject, dtype=float),
        "stems": stems,
        "is_max_subject": is_max_subject,
        "pct_subject": np.asarray(pct_subject, dtype=float),
    }


def _swap_legend_handles(swap_class_subject: List[str]) -> list:
    """Build single shared legend with plain-English labels + cohort counts."""
    counts = {
        c: swap_class_subject.count(c) for c in ["strict", "candidate", "none"]
    }
    return [
        plt.Line2D(
            [0], [0], color=SWAP_COLORS[c], lw=2.5,
            label=f"{SWAP_LABEL_PLAIN[c]}  (n={counts[c]})",
        )
        for c in ["strict", "candidate", "none"]
    ]


# ---------------------------------------------------------------------------
# Figure 1 (主): H_p_norm cohort overlay — single panel, no template split
# ---------------------------------------------------------------------------
def fig_cohort_overlay_normalized(data: Dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    flat = data["flat_curves"]  # list of (stem, swap, curve)
    if not flat:
        return FIG_DIR / "H_p_norm_cohort_overlay.png"

    # Individual curves — per cluster (so 70 curves total for n=35 stable_k=2)
    for stem, swap, y in flat:
        ax.plot(X_GRID, y, color=SWAP_COLORS.get(swap, SWAP_COLORS["unknown"]),
                alpha=0.28, linewidth=1.0, zorder=2)

    Y = np.stack([y for _, _, y in flat], axis=0)
    med = np.median(Y, axis=0)
    q25 = np.percentile(Y, 25, axis=0)
    q75 = np.percentile(Y, 75, axis=0)
    ax.fill_between(X_GRID, q25, q75, color="black", alpha=0.13, zorder=3)
    ax.plot(X_GRID, med, color="black", linewidth=2.8, zorder=4,
            label=f"Cohort median ({Y.shape[0]} cluster curves)")

    style_panel(ax)
    ax.set_xlabel("Normalized rank position", fontsize=FS_LABEL)
    ax.set_ylabel("Channel-identity entropy", fontsize=FS_LABEL)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(0)

    handles = _swap_legend_handles(data["swap_class_subject"])
    handles.append(plt.Line2D([0], [0], color="black", lw=2.8,
                              label="Cohort median + IQR"))
    ax.legend(
        handles=handles, loc="lower center", fontsize=FS_TICK - 1,
        frameon=True, ncol=2,
    )

    fig.tight_layout()
    out = FIG_DIR / "H_p_norm_cohort_overlay.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Δ_subject by reversal class
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
               label="Δ = 0  no endpoint–middle gap")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{SWAP_LABEL_PLAIN[c]}\n(n={len(by_class[c])})" for c in classes],
        fontsize=FS_TICK,
    )
    style_panel(ax)
    ax.set_title(
        "Δ < 0 across all subjects  ⇒  endpoints more determined than middle",
        fontsize=FS_TITLE - 1,
    )
    ax.set_xlabel("Forward/reverse template-pair classification", fontsize=FS_LABEL)
    ax.set_ylabel("Δ  (endpoint − middle entropy)", fontsize=FS_LABEL)
    ax.legend(loc="upper right", fontsize=FS_TICK - 1, frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "delta_by_swapclass_box.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: endpoint_pair_percentile_panel — plain-English questions per panel
# ---------------------------------------------------------------------------
def fig_endpoint_percentile_panel(data: Dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.5))

    # --- Panel A: percentile vs Δ scatter ---
    ax = axes[0]
    for delta, pct, cls in zip(
        data["delta_subject"], data["pct_subject"], data["swap_class_subject"]
    ):
        if not np.isfinite(delta) or not np.isfinite(pct):
            continue
        ax.scatter(pct, delta, color=SWAP_COLORS.get(cls, SWAP_COLORS["unknown"]),
                   edgecolors="black", linewidths=0.7, s=85, alpha=0.92)
    ax.axvline(1.0, ls="--", lw=1.2, color="black", alpha=0.6,
               label="confluence prediction (percentile = 1, Δ > 0)")
    ax.axhline(0.0, ls=":", lw=0.9, color="gray", alpha=0.7)
    ax.legend(loc="upper right", fontsize=FS_TICK - 2, frameon=True)
    style_panel(ax)
    ax.set_title("A.  Endpoint-pair percentile vs Δ", fontsize=FS_TITLE - 1)
    ax.set_xlabel("Endpoint-pair percentile\n(1 = endpoints have largest Δ)",
                  fontsize=FS_LABEL - 1)
    ax.set_ylabel("Δ", fontsize=FS_LABEL)
    ax.set_xlim(0.0, 1.0)
    ax.margins(x=0)

    # --- Panel B: n_valid distribution + min p_N1 floor ---
    ax = axes[1]
    n_valids = data["n_valid_subject"]
    unique_nv = sorted(set(int(x) for x in n_valids))
    counts = [int(np.sum(n_valids == nv)) for nv in unique_nv]
    bars = ax.bar(unique_nv, counts, color="#BDC3C7", edgecolor="black",
                  linewidth=1.2, alpha=0.85, label="Subjects with this n_valid")
    ax2 = ax.twinx()
    floors = [1.0 / (nv * (nv - 1) // 2) for nv in unique_nv]
    ax2.plot(unique_nv, floors, "o-", color="#C0392B", lw=2.0, markersize=10,
             label="Smallest reachable p (1 / C(n,2))")
    ax2.axhline(0.05, ls="--", lw=1.0, color="black", alpha=0.55,
                label="p = 0.05 reference")
    ax2.set_ylabel("Smallest reachable p_N1", fontsize=FS_LABEL - 1, color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    style_panel(ax)
    ax.set_title("B.  Smallest reachable p depends on n_valid",
                 fontsize=FS_TITLE - 1)
    ax.set_xlabel("n_valid\n(channels per template)", fontsize=FS_LABEL)
    ax.set_ylabel("Subject count", fontsize=FS_LABEL)

    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=FS_TICK - 2, frameon=True)

    # --- Panel C: did endpoint pair achieve max? ---
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
    ax.bar(positions, not_max_vals, color="#7F8C8D", edgecolor="black",
           linewidth=1.0, label="Endpoint pair NOT max  (Δ < some other pair)")
    ax.bar(positions, max_vals, bottom=not_max_vals, color="#C0392B",
           edgecolor="black", linewidth=1.0,
           label="Endpoint pair = max  (confluence-style)")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [SWAP_LABEL_PLAIN[c] for c in classes],
        fontsize=FS_TICK - 1, rotation=15, ha="right",
    )
    style_panel(ax)
    ax.set_title("C.  Did the endpoint pair achieve max-Δ?",
                 fontsize=FS_TITLE - 1)
    ax.set_xlabel("Forward/reverse template-pair class", fontsize=FS_LABEL)
    ax.set_ylabel("Subject count", fontsize=FS_LABEL)
    ax.legend(loc="lower right", fontsize=FS_TICK - 2, frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "endpoint_pair_percentile_panel.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: swap subset (strict + candidate) per-subject panels
# ---------------------------------------------------------------------------
def fig_swap_subset_per_subject(records: List[dict]) -> Path:
    swap_records = [
        d for d in records
        if d.get("swap_class_full") in {"strict", "candidate"}
    ]
    if not swap_records:
        return FIG_DIR / "swap_subset_per_subject.png"

    # Group: strict first, then candidate; alphabetical within group
    swap_records.sort(
        key=lambda d: (
            0 if d.get("swap_class_full") == "strict" else 1,
            d["stem"],
        )
    )
    n_subj = len(swap_records)
    n_cols = 5
    n_rows = (n_subj + n_cols - 1) // n_cols

    # Larger panels + tighter font for subtitle (paper-grade clarity)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows),
        sharey=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, d in enumerate(swap_records):
        ax = axes[idx // n_cols, idx % n_cols]
        n_valid = int(d["n_valid_channels"])
        cls = d["swap_class_full"]
        cls_color = SWAP_COLORS[cls]
        sl = d.get("subject_level") or {}
        d_subj = sl.get("delta_obs_subject")
        is_max = sl.get("is_subject_combo_max")
        pct = sl.get("subject_combo_percentile")

        for cid, c in d["clusters"].items():
            H = c.get("H_p_norm")
            if H is None:
                continue
            x = np.linspace(0.0, 1.0, len(H))
            ax.plot(x, H, marker="o", markersize=3.2, lw=1.4,
                    color="#2980B9" if cid == "0" else "#E67E22",
                    label=f"T_{cid}" if idx == 0 else None,
                    alpha=0.92)
        ax.axhline(1.0, ls="--", lw=0.6, color="gray", alpha=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.tick_params(labelsize=FS_TICK - 4)

        # Color-coded class chip in upper-right
        ax.text(0.97, 0.06, SWAP_LABEL_PLAIN[cls],
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_TICK - 4, color="white",
                bbox=dict(facecolor=cls_color, edgecolor="none",
                          boxstyle="round,pad=0.25"))

        if d_subj is None or pct is None:
            sub = "subject-level excluded\n(low kept events)"
        else:
            sub = f"Δ = {d_subj:+.3f}     percentile = {pct:.3f}"
        ax.set_title(
            f"{d['stem']}  (n_valid = {n_valid})\n{sub}",
            fontsize=FS_TICK - 2,
        )

        if idx == 0:
            ax.legend(fontsize=FS_TICK - 3, loc="lower center", ncol=2)

    # Hide unused panels
    for idx in range(n_subj, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    # Shared axis labels
    fig.text(0.5, 0.01, "Normalized rank position", ha="center",
             fontsize=FS_LABEL)
    fig.text(0.005, 0.5, "Channel-identity entropy", va="center",
             rotation="vertical", fontsize=FS_LABEL)

    fig.tight_layout(rect=[0.018, 0.025, 1, 1], h_pad=2.4, w_pad=1.2)
    out = FIG_DIR / "swap_subset_per_subject.png"
    savefig_pub(fig, out)
    savefig_pub(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: bridge with rank_displacement
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
        pairs = r.get("pairs") or []
        if not pairs:
            continue
        rd_lookup[stem] = {
            "F_norm": pairs[0].get("footrule_normalized"),
            "fwd_rev": bool(r.get("fwd_rev_reproduced")),
        }

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    for stem, pct, cls in zip(
        data["stems"], data["pct_subject"], data["swap_class_subject"]
    ):
        if stem not in rd_lookup or not np.isfinite(pct):
            continue
        F = rd_lookup[stem]["F_norm"]
        if F is None or not np.isfinite(F):
            continue
        marker = "*" if rd_lookup[stem]["fwd_rev"] else "o"
        size = 200 if rd_lookup[stem]["fwd_rev"] else 90
        ax.scatter(F, pct,
                   color=SWAP_COLORS.get(cls, SWAP_COLORS["unknown"]),
                   edgecolors="black", linewidths=0.8,
                   s=size, marker=marker, alpha=0.92)

    ax.axhline(1.0, ls="--", lw=1.0, color="black", alpha=0.45,
               label="endpoint-pair Δ is max (sup1)")
    ax.axvline(2 / 3, ls="--", lw=1.0, color="#C0392B", alpha=0.65,
               label="random reversal floor 2/3 (rank displacement)")
    style_panel(ax)
    ax.set_xlabel("Forward/reverse displacement F_norm  "
                  "(0 = no swap, 1 = full reversal)",
                  fontsize=FS_LABEL - 1)
    ax.set_ylabel("Endpoint-pair percentile  (1 = endpoints have largest Δ)",
                  fontsize=FS_LABEL - 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.margins(0)

    legend_handles = _swap_legend_handles(data["swap_class_subject"])
    legend_handles.append(
        plt.Line2D([0], [0], color="#7F8C8D", marker="*", lw=0,
                   markersize=14, label="Forward/reverse reproduced (rd)"),
    )
    legend_handles.append(
        plt.Line2D([0], [0], ls="--", color="black", lw=1.0,
                   label="endpoint-pair Δ is max"),
    )
    legend_handles.append(
        plt.Line2D([0], [0], ls="--", color="#C0392B", lw=1.0,
                   label="random reversal floor 2/3"),
    )
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=FS_TICK - 1, frameon=True)

    fig.tight_layout()
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
