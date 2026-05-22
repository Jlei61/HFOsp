#!/usr/bin/env python3
"""PR-6 supplement: cluster rank distribution small multiples for swap cohort.

Renders two figures:

  * ``pr6_supp_swap_cluster_rank_multiples`` — all 8 forward/reverse
    candidates (subjects passing PR-2.5 fwd/rev reproducibility), with
    "strong-swap" subjects (``exceeds_null_95 == True``) shown first and
    "non-strong" subjects last.
  * ``pr6_supp_swap_cluster_rank_multiples_nonstrong`` — only the
    forward/reverse candidates that did NOT pass the swap-score
    permutation null.

For each subject, render the per-cluster mean rank distribution in the
same visual style as the bottom-right panel of
``epilepsiae_<sid>_propagation.png``:

  * x-axis = rank (0=first/source, n_ch-1=last/sink)
  * y-axis = channels in fixed source→sink order
  * one line + shaded mean±std band per cluster (C0 blue / C1 red)

Swap nodes — channels that flip endpoint role between T0 and T1
(``(T0.source ∩ T1.sink) ∪ (T0.sink ∩ T1.source)``) — are highlighted by

  * larger star marker (vs small circle for non-swap)
  * full color saturation on the cluster lines at those channels
  * bold y-tick labels

Output: ``results/interictal_propagation/template_anchoring/figures/
pr6_supp_swap_cluster_rank_multiples.{png,pdf}``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (  # noqa: E402
    savefig_pub,
    FS_LABEL, FS_TICK, FS_TITLE,
    COL_SWAP_LABEL,
    COL_CLUSTER_T0,
    COL_CLUSTER_T1,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("results/interictal_propagation")
PROP_PER_SUBJECT = RESULTS_ROOT / "per_subject"
TEMPLATE_DIR = RESULTS_ROOT / "template_anchoring"
ANC_PER_SUBJECT = TEMPLATE_DIR / "per_subject"
FIG_DIR = TEMPLATE_DIR / "figures"
COHORT_FP = TEMPLATE_DIR / "cohort_summary.json"
RD_PER_SUBJECT = RESULTS_ROOT / "rank_displacement" / "per_subject"
RD_COHORT_FP = RESULTS_ROOT / "rank_displacement" / "cohort_summary.json"

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree.

    Mirrors scripts/plot_rank_displacement.py:_apply_masked_paths. Swaps every
    `results/interictal_propagation/...` global to `results/interictal_propagation_masked/...`
    (including the two rank_displacement sub-paths that drive the
    rank-displacement strict panel).
    """
    global RESULTS_ROOT, PROP_PER_SUBJECT, TEMPLATE_DIR, ANC_PER_SUBJECT
    global FIG_DIR, COHORT_FP, RD_PER_SUBJECT, RD_COHORT_FP
    RESULTS_ROOT = Path("results/interictal_propagation_masked")
    PROP_PER_SUBJECT = RESULTS_ROOT / "per_subject"
    TEMPLATE_DIR = RESULTS_ROOT / "template_anchoring"
    ANC_PER_SUBJECT = TEMPLATE_DIR / "per_subject"
    FIG_DIR = TEMPLATE_DIR / "figures"
    COHORT_FP = TEMPLATE_DIR / "cohort_summary.json"
    RD_PER_SUBJECT = RESULTS_ROOT / "rank_displacement" / "per_subject"
    RD_COHORT_FP = RESULTS_ROOT / "rank_displacement" / "cohort_summary.json"

# Cluster colors (match per-subject propagation figure)
# Imported aliases for compactness in plotting calls.
COL_C0 = COL_CLUSTER_T0   # T0 / forward (blue)
COL_C1 = COL_CLUSTER_T1   # T1 / reverse (red)
COL_FADE = 0.30      # alpha for non-swap markers


# ---------------------------------------------------------------------------
# Helpers (reuse loaders from existing propagation pipeline)
# ---------------------------------------------------------------------------

def _resolve_subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return EPILEPSIAE_ROOT / subject / "all_recs"


def _load_lagpat(subject_dir: Path) -> Dict[str, Any]:
    from src.interictal_propagation import load_subject_propagation_events
    return load_subject_propagation_events(subject_dir)


def _valid_event_indices(bools: np.ndarray, min_participating: int = 3) -> np.ndarray:
    counts = np.sum(bools > 0, axis=0)
    return np.where(counts >= min_participating)[0].astype(int)


def _fixed_channel_order(ranks: np.ndarray, bools: np.ndarray) -> np.ndarray:
    """Channels sorted by joint (all-events) mean rank, ties broken by count.

    The joint ordering deliberately interleaves T0-source and T1-source
    channels — this is what makes the role-swap crossings between C0 and C1
    legible in the plot. Matches the ordering used in the per-subject
    propagation figure.
    """
    n_ch = ranks.shape[0]
    mean_rank = np.full(n_ch, np.nan, dtype=float)
    counts = np.sum(bools > 0, axis=1).astype(int) if bools.size else np.zeros(n_ch, dtype=int)
    for idx in range(n_ch):
        vals = np.asarray(ranks[idx, bools[idx] > 0], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean_rank[idx] = float(np.mean(vals))
    if np.any(np.isfinite(mean_rank)):
        fill = np.nanmax(mean_rank[np.isfinite(mean_rank)]) + 1.0
    else:
        fill = 1.0
    keys = np.where(np.isfinite(mean_rank), mean_rank, fill)
    return np.lexsort((np.arange(n_ch), -counts, keys))


# ---------------------------------------------------------------------------
# Per-cluster mean rank profile
# ---------------------------------------------------------------------------

def _cluster_mean_std(
    ranks: np.ndarray,
    bools: np.ndarray,
    valid_events: np.ndarray,
    labels: np.ndarray,
    channel_order: np.ndarray,
    cluster_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_ch = len(channel_order)
    means = np.full(n_ch, np.nan)
    stds = np.full(n_ch, np.nan)
    sel = labels == cluster_id
    eidx = valid_events[sel]
    if eidx.size == 0:
        return means, stds
    for plot_idx, ci_raw in enumerate(channel_order):
        vals = np.asarray(ranks[ci_raw, eidx], dtype=float)
        mask = np.asarray(bools[ci_raw, eidx], dtype=bool)
        vals = vals[mask & np.isfinite(vals)]
        if vals.size > 0:
            means[plot_idx] = float(np.mean(vals))
            stds[plot_idx] = float(np.std(vals))
    return means, stds


# ---------------------------------------------------------------------------
# Per-panel drawing
# ---------------------------------------------------------------------------

def _swap_nodes_pr6_h2(dataset: str, subject: str) -> Tuple[set, int, int]:
    """PR-6 H2 swap nodes from anchoring JSON's per_template source/sink sets."""
    anc_fp = ANC_PER_SUBJECT / f"{dataset}_{subject}.json"
    with open(anc_fp) as f:
        anc = json.load(f)
    pt = anc["per_template"]
    t0_src = set(pt[0].get("source") or [])
    t0_snk = set(pt[0].get("sink") or [])
    t1_src = set(pt[1].get("source") or [])
    t1_snk = set(pt[1].get("sink") or [])
    swap = (t0_src & t1_snk) | (t0_snk & t1_src)
    cid_t0 = int(pt[0].get("cluster_id", 0))
    cid_t1 = int(pt[1].get("cluster_id", 1))
    return swap, cid_t0, cid_t1


def _swap_nodes_rank_displacement(dataset: str, subject: str) -> Tuple[set, int, int, int]:
    """rank_displacement swap nodes via swap_sweep.decision_k.

    For decision_k = k, swap nodes are channels in
        (top-k of T0  ∩  bottom-k of T1)
      ∪ (bottom-k of T0  ∩  top-k of T1)
    using the dense rank vectors ``rank_a_dense_full`` / ``rank_b_dense_full``
    over the joint-valid channels. Returns (swap_set, cid_a, cid_b, decision_k).
    """
    fp = RD_PER_SUBJECT / f"{dataset}_{subject}.json"
    with open(fp) as f:
        rd = json.load(f)
    p = rd["pairs"][0]
    ch_names = list(p["channel_names"])
    joint_valid = np.asarray(p["joint_valid"], dtype=bool)
    rank_a = np.asarray(p["rank_a_dense_full"], dtype=float)
    rank_b = np.asarray(p["rank_b_dense_full"], dtype=float)
    sw = p["swap_sweep"]
    k = int(sw["decision_k"])

    # Restrict to joint-valid
    valid_idx = np.where(joint_valid)[0]
    valid_names = [ch_names[i] for i in valid_idx]
    valid_a = rank_a[valid_idx]
    valid_b = rank_b[valid_idx]

    # top-k = smallest k ranks (rank 0,1,...,k-1); bottom-k = largest k ranks
    src_a = {valid_names[i] for i in np.argsort(valid_a)[:k]}
    snk_a = {valid_names[i] for i in np.argsort(valid_a)[-k:]}
    src_b = {valid_names[i] for i in np.argsort(valid_b)[:k]}
    snk_b = {valid_names[i] for i in np.argsort(valid_b)[-k:]}

    swap = (src_a & snk_b) | (snk_a & src_b)
    cid_a = int(p["cluster_id_a"])
    cid_b = int(p["cluster_id_b"])
    return swap, cid_a, cid_b, k


def _draw_subject_panel(
    ax: plt.Axes,
    dataset: str,
    subject: str,
    swap_nodes: set,
    cid_t0: int,
    cid_t1: int,
    title_text: str,
) -> None:
    subject_dir = _resolve_subject_dir(dataset, subject)
    loaded = _load_lagpat(subject_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    channel_names: List[str] = list(loaded["channel_names"])

    prop_fp = PROP_PER_SUBJECT / f"{dataset}_{subject}.json"
    with open(prop_fp) as f:
        prop = json.load(f)

    adaptive = prop.get("adaptive_cluster", {})
    labels = np.asarray(adaptive.get("labels", []), dtype=int)

    valid_events = _valid_event_indices(bools, min_participating=3)
    if labels.size != valid_events.size:
        labels = np.asarray(prop.get("cluster", {}).get("labels", []), dtype=int)
    if labels.size != valid_events.size:
        ax.text(0.5, 0.5, "labels missing", transform=ax.transAxes,
                ha="center", va="center")
        return

    channel_order = _fixed_channel_order(ranks, bools)
    ordered_names = [channel_names[i] for i in channel_order]
    n_ch = len(channel_order)

    y_pos = np.arange(n_ch, dtype=float)

    cluster_data = [
        (cid_t0, COL_C0, "T0 (forward)"),
        (cid_t1, COL_C1, "T1 (reverse)"),
    ]

    for cid, color, _label in cluster_data:
        means, stds = _cluster_mean_std(ranks, bools, valid_events, labels,
                                        channel_order, cid)
        valid = np.isfinite(means)
        if not np.any(valid):
            continue
        # shaded band
        ax.fill_betweenx(
            y_pos[valid],
            (means - stds)[valid], (means + stds)[valid],
            color=color, alpha=0.12, linewidth=0,
        )
        # line
        ax.plot(means[valid], y_pos[valid], "-", color=color, lw=1.8,
                alpha=0.85, zorder=8)

        # markers: faded circle for non-swap, bright star for swap
        # clip_on=False so markers at x=0 or y=0 aren't cut by the tight axes
        for plot_idx, ch_name in enumerate(ordered_names):
            if not np.isfinite(means[plot_idx]):
                continue
            is_swap = ch_name in swap_nodes
            if is_swap:
                ax.scatter(
                    [means[plot_idx]], [y_pos[plot_idx]],
                    marker="*", s=180, color=color, alpha=1.0,
                    edgecolors="black", linewidths=0.7, zorder=12,
                    clip_on=False,
                )
            else:
                ax.scatter(
                    [means[plot_idx]], [y_pos[plot_idx]],
                    marker="o", s=28, color=color, alpha=COL_FADE,
                    edgecolors="none", zorder=7,
                    clip_on=False,
                )

    # y-axis labels — swap-node names in orange + bold; others muted gray
    ax.set_yticks(y_pos)
    fontsize = 11 if n_ch <= 10 else 9
    ax.set_yticklabels(ordered_names, fontsize=fontsize)
    for tick_label, ch_name in zip(ax.get_yticklabels(), ordered_names):
        if ch_name in swap_nodes:
            tick_label.set_fontweight("bold")
            tick_label.set_color(COL_SWAP_LABEL)
        else:
            tick_label.set_color("#888888")

    # Fully tight axes: first channel flush with top, last channel flush with
    # bottom, x starts at 0. clip_on=False on markers prevents edge clipping.
    ax.set_ylim(0.0, n_ch - 1)
    ax.invert_yaxis()
    ax.set_xlim(0.0, n_ch - 1)
    ax.set_xlabel("Rank", fontsize=FS_LABEL - 2)
    ax.tick_params(axis="x", labelsize=FS_TICK - 2)

    # pad>0 lifts the title above the axes top spine (further from panel content).
    ax.set_title(title_text, fontsize=FS_TICK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------

def _select_fwdrev_all(cohort: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list(cohort["h2_forward_reverse_swap"]["per_subject"])
    # strong first (passes null95), within each group sort by dataset+id
    rows.sort(
        key=lambda r: (not r.get("exceeds_null_95"), r["dataset"], r["subject_id"])
    )
    return rows


def _select_nonstrong_candidates(cohort: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = cohort["h2_forward_reverse_swap"]["per_subject"]
    weak = [r for r in rows if not r.get("exceeds_null_95")]
    weak.sort(key=lambda r: (r["dataset"], r["subject_id"]))
    return weak


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------

def _format_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    return f"{p:.3f}" if p >= 1e-3 else "<0.001"


LEGEND_W = 3.8  # inches reserved on the right for the vertical legend


def _render(
    panels: List[Dict[str, Any]],
    *,
    out_stem: Path,
    ncols: int = 3,
    panel_w: float = 5.2,
    panel_h: float = 5.8,
) -> Path:
    """Render small multiples with a dedicated right-side legend column.

    Each item in ``panels`` is a dict with:
      dataset, subject, swap_nodes (set[str]), cid_t0, cid_t1, title.
    No figure-level suptitle; per-panel titles only.
    """
    n = len(panels)
    if n == 0:
        raise SystemExit(f"empty cohort: cannot render {out_stem}")

    nrows = math.ceil(n / ncols)
    panel_block_w = panel_w * ncols
    total_w = panel_block_w + LEGEND_W
    total_h = panel_h * nrows

    fig = plt.figure(figsize=(total_w, total_h))
    fig.patch.set_facecolor("white")

    # Panel block ends at right_frac; legend block lives in (right_frac, 1.0].
    right_frac = panel_block_w / total_w

    # Leave generous top room so the lifted per-panel titles don't run into
    # the figure top edge.
    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.05, right=right_frac - 0.015,
        top=0.91, bottom=0.07,
        wspace=0.45, hspace=0.50,
    )

    axes: List[List[plt.Axes]] = []
    for r in range(nrows):
        row_axes: List[plt.Axes] = []
        for c in range(ncols):
            row_axes.append(fig.add_subplot(gs[r, c]))
        axes.append(row_axes)

    for idx, item in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        _draw_subject_panel(
            ax,
            dataset=item["dataset"],
            subject=item["subject"],
            swap_nodes=item["swap_nodes"],
            cid_t0=item["cid_t0"],
            cid_t1=item["cid_t1"],
            title_text=item["title"],
        )

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_axis_off()

    legend_handles = [
        Line2D([0], [0], color=COL_C0, lw=2.5, label="T0 (forward) mean rank"),
        Line2D([0], [0], color=COL_C1, lw=2.5, label="T1 (reverse) mean rank"),
        Line2D([0], [0], marker="*", color="0.3", markerfacecolor="0.3",
               markersize=14, lw=0,
               markeredgecolor="black", label="swap node\n(T0↔T1 role flip)"),
        Line2D([0], [0], marker="o", color="0.55", markerfacecolor="0.55",
               markersize=6, lw=0, alpha=COL_FADE,
               label="non-swap channel"),
        mpatches.Patch(color="#888888", alpha=0.25,
                       label="cluster mean ± 1 SD"),
    ]
    # Pin the legend to the far right of the reserved legend column.
    fig.legend(
        handles=legend_handles,
        loc="center right",
        ncol=1,
        fontsize=FS_TICK,
        frameon=False,
        bbox_to_anchor=(0.995, 0.5),
        handletextpad=0.8,
        labelspacing=1.6,
        borderaxespad=0.0,
    )

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _panel_pr6_h2(row: Dict[str, Any]) -> Dict[str, Any]:
    ds, sid = row["dataset"], row["subject_id"]
    swap_nodes, cid_t0, cid_t1 = _swap_nodes_pr6_h2(ds, sid)
    n_swap = len(swap_nodes)
    title = (
        f"{ds[0].upper()}:{sid}\n"
        f"p={_format_p(row.get('null_p', float('nan')))}   n_swap={n_swap}"
    )
    return dict(dataset=ds, subject=sid, swap_nodes=swap_nodes,
                cid_t0=cid_t0, cid_t1=cid_t1, title=title)


def _panel_rank_displacement(rd_record: Dict[str, Any]) -> Dict[str, Any]:
    ds, sid = rd_record["dataset"], rd_record["subject"]
    swap_nodes, cid_a, cid_b, _decision_k = _swap_nodes_rank_displacement(ds, sid)
    n_swap = len(swap_nodes)
    title = f"{ds[0].upper()}:{sid}    n_swap={n_swap}"
    return dict(dataset=ds, subject=sid, swap_nodes=swap_nodes,
                cid_t0=cid_a, cid_t1=cid_b, title=title)


def make_fwdrev_all(cohort: Dict[str, Any]) -> Path:
    subjects = _select_fwdrev_all(cohort)
    panels = [_panel_pr6_h2(r) for r in subjects]
    # 2 rows × 4 cols (narrower panels)
    return _render(
        panels,
        out_stem=FIG_DIR / "pr6_supp_swap_cluster_rank_multiples",
        ncols=4,
        panel_w=3.6,
        panel_h=4.8,
    )


def make_nonstrong(cohort: Dict[str, Any]) -> Path:
    subjects = _select_nonstrong_candidates(cohort)
    panels = [_panel_pr6_h2(r) for r in subjects]
    return _render(
        panels,
        out_stem=FIG_DIR / "pr6_supp_swap_cluster_rank_multiples_nonstrong",
        ncols=2,
        panel_w=5.4,
        panel_h=5.8,
    )


def _select_rank_displacement_strict() -> List[Dict[str, Any]]:
    fp = RD_COHORT_FP
    with open(fp) as f:
        recs = json.load(f)
    strict = [
        r for r in recs
        if r.get("pairs") and r["pairs"][0].get("swap_sweep", {}).get("swap_class") == "strict"
    ]
    strict.sort(
        key=lambda r: (r["dataset"], r["subject"])
    )
    return strict


def make_rank_displacement_strict() -> Path:
    recs = _select_rank_displacement_strict()
    panels = [_panel_rank_displacement(r) for r in recs]
    # 2 rows × 5 cols (10 panels, no empty cells), narrower panels.
    return _render(
        panels,
        out_stem=FIG_DIR / "pr6_supp_rank_displacement_swap_strict",
        ncols=5,
        panel_w=3.2,
        panel_h=4.8,
    )


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="PR-6 swap cluster rank multiples (fwd/rev all + non-strong + rank_displacement strict)"
    )
    parser.add_argument(
        "--masked-features",
        action="store_true",
        help="Consume masked PR-6 + rank_displacement outputs under "
             "results/interictal_propagation_masked/ and write figures next "
             "to them. Mirrors scripts/plot_rank_displacement.py --masked-features.",
    )
    args = parser.parse_args()
    if args.masked_features:
        _apply_masked_paths()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(COHORT_FP) as f:
        cohort = json.load(f)
    out_a = make_fwdrev_all(cohort)
    out_b = make_nonstrong(cohort)
    out_c = make_rank_displacement_strict()
    for p in (out_a, out_b, out_c):
        print(f"Saved: {p}")
        print(f"Saved: {p.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
