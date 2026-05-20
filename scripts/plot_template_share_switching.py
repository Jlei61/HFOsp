"""Paper-grade figures for template share + switching around seizures.

Fig A — Template share (dominant template fraction) baseline / pre / post
Fig B — Next-event transition lift baseline / pre / post

Cohort: stable_k=2 subjects with usable seizure windows (n=27 after the
8 yuquan-no-seizure + 1 epilepsiae label-mismatch drops).
Source data:
  results/interictal_propagation/pr5b_recruitment_shift_extended.json
  results/interictal_propagation/pr5b_recruitment_shift.json (1096 only)
  results/interictal_propagation/pr5_transition_windows.json

Outputs:
  results/interictal_propagation/template_share_switching/figures/
    fig_a_template_share.{png,pdf}
    fig_b_template_switching.{png,pdf}
    README.md
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plot_style import (  # noqa: E402
    COL_EPILEPSIAE, COL_YUQUAN, COL_SIG, COL_NONSIG,
    FS_TICK, FS_LABEL, FS_TITLE,
    style_panel, savefig_pub, dataset_color,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PR-4D strict-match subjects (rate burst enrich >=1.5 AND |rho(dom_frac, |Δt_sz|)| >= 0.15)
# Source: docs/archive/topic1/propagation/topic1_pr4_ppt_figures.md §fig5 part B
STRICT_MATCH_SUBJECTS = {
    "epilepsiae/1125",
    "epilepsiae/1096",
    "epilepsiae/916",
    "epilepsiae/635",
    "epilepsiae/442",
    "epilepsiae/1150",
    "epilepsiae/139",
    "yuquan/sunyuanxin",
    "yuquan/litengsheng",
}


def _load_share_records() -> List[Dict[str, Any]]:
    """Combine extended-cohort + 1096-original into a flat list of records.

    Each record:
      {dataset, subject_id, base, pre, post, dpre, dpost, dpostpre,
       dom_agreement, n_seizures, label='dataset/subject_id'}
    """
    ext_path = ROOT / "results/interictal_propagation/pr5b_recruitment_shift_extended.json"
    orig_path = ROOT / "results/interictal_propagation/pr5b_recruitment_shift.json"
    with open(ext_path) as f:
        ext = json.load(f)
    with open(orig_path) as f:
        orig = json.load(f)

    records: List[Dict[str, Any]] = []
    seen: set = set()
    for r in ext["per_subject"]["main"]:
        sh = r["weighted_per_state"]["dom_global_share"]
        dh = r["deltas"]["dom_global_share"]
        if not np.isfinite(sh["baseline"]) or not np.isfinite(sh["pre"]) or not np.isfinite(sh["post"]):
            continue
        label = f"{r['dataset']}/{r['subject_id']}"
        seen.add(label)
        records.append({
            "label": label,
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "base": sh["baseline"],
            "pre": sh["pre"],
            "post": sh["post"],
            "dpre": dh["pre_minus_baseline"],
            "dpost": dh["post_minus_baseline"],
            "dpostpre": dh["post_minus_pre"],
            "n_seizures": r["n_seizures_usable"],
        })
    # Patch in 1096 from original
    for r in orig["per_subject"]["main"]:
        label = f"{r['dataset']}/{r['subject_id']}"
        if label in seen:
            continue
        if r["subject_id"] != "1096":
            continue
        sh = r["weighted_per_state"]["dom_global_share"]
        dh = r["deltas"]["dom_global_share"]
        records.append({
            "label": label,
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "base": sh["baseline"],
            "pre": sh["pre"],
            "post": sh["post"],
            "dpre": dh["pre_minus_baseline"],
            "dpost": dh["post_minus_baseline"],
            "dpostpre": dh["post_minus_pre"],
            "n_seizures": r["n_seizures_usable"],
        })
    return records


def _load_transition_records() -> List[Dict[str, Any]]:
    path = ROOT / "results/interictal_propagation/pr5_transition_windows.json"
    with open(path) as f:
        d = json.load(f)
    out: List[Dict[str, Any]] = []
    for r in d["per_subject"]:
        s = r["states"]
        if not (np.isfinite(s["baseline"]["lift"]) and np.isfinite(s["pre"]["lift"]) and np.isfinite(s["post"]["lift"])):
            continue
        out.append({
            "label": f"{r['dataset']}/{r['subject_id']}",
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "base": s["baseline"]["lift"],
            "pre": s["pre"]["lift"],
            "post": s["post"]["lift"],
            "dpost": r["deltas"]["post_minus_baseline_lift"],
            "dpre": r["deltas"]["pre_minus_baseline_lift"],
            "dpostpre": r["deltas"]["post_minus_pre_lift"],
            "n_seizures": r["n_seizures"],
        })
    return out


def _wilcoxon_sign(values: np.ndarray) -> Tuple[float, float, int, int]:
    """Return (median, wilcoxon_p, n_pos, n_neg) ignoring exact zeros."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    nz = v[v != 0]
    if nz.size < 2:
        return float(np.median(v)) if v.size else float("nan"), float("nan"), int(np.sum(v > 0)), int(np.sum(v < 0))
    try:
        _, p = stats.wilcoxon(nz)
    except Exception:
        p = float("nan")
    return float(np.median(v)), float(p), int(np.sum(v > 0)), int(np.sum(v < 0))


def _plot_per_subject_triplet(
    ax: plt.Axes,
    records: List[Dict[str, Any]],
    *,
    y_label: str,
    y_ref: float | None = None,
    sort_by: str = "dpost",
    drop_labels: set | None = None,
) -> None:
    """Per-subject triplet line plot (base -> pre -> post)."""
    drop_labels = drop_labels or set()
    recs = [r for r in records if r["label"] not in drop_labels]
    recs = sorted(recs, key=lambda r: r[sort_by], reverse=True)
    x = np.arange(3)
    for r in recs:
        y = [r["base"], r["pre"], r["post"]]
        is_strict = r["label"] in STRICT_MATCH_SUBJECTS
        color = dataset_color(r["dataset"])
        ax.plot(
            x, y,
            color=color,
            alpha=0.95 if is_strict else 0.55,
            lw=1.8 if is_strict else 1.1,
            marker="o" if is_strict else "o",
            ms=6 if is_strict else 3.5,
            mec="black" if is_strict else color,
            mew=0.5 if is_strict else 0,
            zorder=3 if is_strict else 2,
        )
    if y_ref is not None:
        ax.axhline(y_ref, color="#888888", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "pre", "post"], fontsize=FS_TICK)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.25, 2.25)


def _plot_cohort_delta(
    ax: plt.Axes,
    records: List[Dict[str, Any]],
    *,
    y_label: str,
    zero_ref: float = 0.0,
    drop_labels: set | None = None,
) -> Dict[str, Tuple[float, float, int, int, int]]:
    """Box + strip for cohort deltas. Returns dict[key]=(med,p,npos,nneg,n)."""
    keys = [("dpre", "pre − baseline"), ("dpost", "post − baseline"), ("dpostpre", "post − pre")]
    drop_labels = drop_labels or set()
    data = []
    for k, _ in keys:
        vals = np.array(
            [r[k] for r in records
             if r["label"] not in drop_labels and np.isfinite(r[k])],
            dtype=float,
        )
        data.append(vals)

    positions = np.arange(len(keys))
    ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COL_SIG, lw=2.0),
        whiskerprops=dict(color="#666666", lw=1.0),
        capprops=dict(color="#666666", lw=1.0),
        boxprops=dict(facecolor="#EEEAE3", edgecolor="#666666"),
    )
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            np.full_like(vals, positions[i]) + jitter,
            vals,
            s=22, color=COL_NONSIG, edgecolor="black", lw=0.4, alpha=0.85, zorder=3,
        )

    ax.axhline(zero_ref, color="#666666", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([k[1] for k in keys], fontsize=FS_TICK)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_stats: Dict[str, Tuple[float, float, int, int, int]] = {}
    for (k, _), vals in zip(keys, data):
        med, p, npos, nneg = _wilcoxon_sign(vals)
        out_stats[k] = (med, p, npos, nneg, int(vals.size))
    return out_stats


def _annotate_cohort_stats(
    ax: plt.Axes,
    stats_dict: Dict[str, Tuple[float, float, int, int, int]],
    *,
    place: str = "top",
) -> None:
    """Print n / median / W-p / sign on top of each box (in-axes), single block."""
    keys = ["dpre", "dpost", "dpostpre"]
    y0, y1 = ax.get_ylim()
    # Reserve top space for annotations (instead of bottom which collides
    # with x-tick labels). Push y1 up.
    if place == "top":
        head_room = 0.35 * (y1 - y0)
        new_y1 = y1 + head_room
        ax.set_ylim(y0, new_y1)
        text_y = y1 + 0.02 * (y1 - y0)
        va = "bottom"
    else:
        new_y0 = y0 - 0.28 * (y1 - y0)
        ax.set_ylim(new_y0, y1)
        text_y = y0 - 0.04 * (y1 - y0)
        va = "top"

    for i, k in enumerate(keys):
        med, p, npos, nneg, n = stats_dict[k]
        star = ""
        if np.isfinite(p):
            if p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
        txt = (
            f"n={n}\n"
            f"med {med:+.3f}\n"
            f"W p={p:.3f}{star}\n"
            f"{npos}+/{nneg}−"
        )
        ax.text(i, text_y, txt, ha="center", va=va, fontsize=10,
                color="#222222",
                linespacing=1.25)


def _plot_legend_inset(ax: plt.Axes, *, loc: str = "upper left") -> None:
    """Shared legend: dataset colors + strict-match emphasis."""
    handles = [
        plt.Line2D([0], [0], color=COL_EPILEPSIAE, lw=2.0, marker="o", ms=6, label="Epilepsiae"),
        plt.Line2D([0], [0], color=COL_YUQUAN, lw=2.0, marker="o", ms=6, label="Yuquan"),
        plt.Line2D([0], [0], color="#444444", lw=2.0, marker="o", ms=6, mec="black", mew=0.5,
                   label="rate-burst seizure-enrich (n=9)"),
    ]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=10)


def fig_a_template_share(out_dir: Path) -> None:
    records = _load_share_records()
    logger.info("Fig A cohort: %d subjects", len(records))

    fig = plt.figure(figsize=(12, 5.8))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.95], wspace=0.32,
        left=0.07, right=0.98, top=0.88, bottom=0.18,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_per_subject_triplet(
        ax_a, records,
        y_label="dominant template share",
        y_ref=0.5,
    )
    ax_a.set_ylim(0.30, 1.0)
    ax_a.set_title("a   per-subject dominant share by window", fontsize=FS_TITLE, loc="left", pad=8)
    _plot_legend_inset(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    stats_dict = _plot_cohort_delta(
        ax_b, records,
        y_label="Δ dominant share",
        zero_ref=0.0,
    )
    ax_b.set_title("b   cohort paired differences", fontsize=FS_TITLE, loc="left", pad=8)
    # Tight y-range based on actual values, room added by _annotate_cohort_stats
    lo = min(min(r["dpre"] for r in records), min(r["dpost"] for r in records), min(r["dpostpre"] for r in records))
    hi = max(max(r["dpre"] for r in records), max(r["dpost"] for r in records), max(r["dpostpre"] for r in records))
    ax_b.set_ylim(lo - 0.02, hi + 0.04)
    _annotate_cohort_stats(ax_b, stats_dict)

    out_png = out_dir / "fig_a_template_share.png"
    out_pdf = out_dir / "fig_a_template_share.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info("Wrote %s + .pdf", out_png)


def fig_b_template_switching(out_dir: Path) -> None:
    records = _load_transition_records()
    logger.info("Fig B cohort: %d subjects", len(records))

    # huanghanwen: 1 seizure, n_pairs<10, lift>3 — exclude from figure
    # (kept in JSON for the record).
    drop = {"yuquan/huanghanwen"}

    fig = plt.figure(figsize=(12, 5.8))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.95], wspace=0.32,
        left=0.07, right=0.98, top=0.88, bottom=0.18,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_per_subject_triplet(
        ax_a, records,
        y_label="next-event transition lift",
        y_ref=1.0,
        drop_labels=drop,
    )
    ax_a.set_title("a   per-subject transition lift by window", fontsize=FS_TITLE, loc="left", pad=8)
    ax_a.set_ylim(0.55, 1.45)
    _plot_legend_inset(ax_a, loc="upper right")
    ax_a.text(
        0.02, 0.04,
        "1 subject excluded (n_pairs<10, lift outlier)",
        transform=ax_a.transAxes,
        fontsize=9, color="#888888", ha="left", va="bottom",
    )

    ax_b = fig.add_subplot(gs[0, 1])
    stats_dict = _plot_cohort_delta(
        ax_b, records,
        y_label="Δ transition lift",
        zero_ref=0.0,
        drop_labels=drop,
    )
    ax_b.set_title("b   cohort paired differences", fontsize=FS_TITLE, loc="left", pad=8)
    finite_records = [r for r in records if r["label"] not in drop]
    lo = min(min(r["dpre"] for r in finite_records), min(r["dpost"] for r in finite_records), min(r["dpostpre"] for r in finite_records))
    hi = max(max(r["dpre"] for r in finite_records), max(r["dpost"] for r in finite_records), max(r["dpostpre"] for r in finite_records))
    ax_b.set_ylim(lo - 0.02, hi + 0.04)
    _annotate_cohort_stats(ax_b, stats_dict)

    out_png = out_dir / "fig_b_template_switching.png"
    out_pdf = out_dir / "fig_b_template_switching.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info("Wrote %s + .pdf", out_png)


def main() -> None:
    out_dir = ROOT / "results/interictal_propagation/template_share_switching/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_a_template_share(out_dir)
    fig_b_template_switching(out_dir)


if __name__ == "__main__":
    main()
