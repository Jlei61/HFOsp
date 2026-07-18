#!/usr/bin/env python3
"""Paper-ready COHORT figures for Topic5 V3d scaffold A/B lateral switching.

Three cohort-level statistical figures, each answering one independent question
(CLAUDE.md §7 — one panel/figure, one question):

  1. cohort_two_state_vs_geometry.png
     Is peri-ictal energy avoiding the A<->B contrast-axis midpoint just an artifact
     of how anti-correlated the two interictal templates happen to be, or does it
     hold even for near-orthogonal / positively-correlated template pairs?
  2. cohort_ab_typing.png  (the key figure)
     Do a subject's seizures reproducibly type into an A-source-dominant vs a
     B-source-dominant state, or is the lateral composition indistinguishable
     seizure to seizure?
  3. cohort_h1_nearonset_forest.png
     Cohort-level near-onset lateral-polarization locking test, restricted to the
     handful of subjects with enough eligible seizures to test it at all.

This is a read-only downstream figure script: it does not touch
src/topic5_scaffold_ab_contrast.py or any committed producer/plotting script, and
does not recompute anything -- every number is read verbatim from
results/topic5_ictal_recruitment/scaffold_ab_switching/{cohort_analysis.json,
per_subject/<ds_sid>_scaffold_ab_summary.json}.

Output: results/paper-ready-figure/fig_topic5_scaffold_ab/figures/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plot_style import style_panel, savefig_pub  # noqa: E402

DATA_DIR = ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
COHORT_JSON = DATA_DIR / "cohort_analysis.json"
PER_SUBJECT_DIR = DATA_DIR / "per_subject"
OUT_DIR = ROOT / "results/paper-ready-figure/fig_topic5_scaffold_ab/figures"

# ---------------------------------------------------------------------------
# Locked palette (docs/figure_style_guide.md §0 + this figure's own spec):
#   A source side / template A = red; B source side / template B = blue;
#   bimodal/switch = purple; mid/low-data = grays; tier categories use
#   colors that do not clash with the A/B red-blue axis.
# ---------------------------------------------------------------------------
COL_A = "#B2182B"
COL_B = "#2166AC"
COL_BIMODAL = "#762A83"
COL_LOWDATA = "#BDBDBD"
COL_MID = "#E3E3E3"
COL_RECIP = "#4D4D4D"
COL_OBLIQUE = "#1B9E77"
COL_LOCKED = "#B2182B"
COL_NOTLOCKED = "#9B9B9B"

DELTA_SIDE = 0.2  # |C_AB| side-label threshold baked into the upstream data


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _pretty(ds_sid: str) -> str:
    """'epilepsiae_1146' -> 'E1146', 'yuquan_xuxinyi' -> 'Y-xuxinyi'."""
    if ds_sid.startswith("epilepsiae_"):
        return "E" + ds_sid[len("epilepsiae_"):]
    if ds_sid.startswith("yuquan_"):
        return "Y-" + ds_sid[len("yuquan_"):]
    return ds_sid


def _bare_label(ds_sid: str) -> str:
    """'epilepsiae_1146' -> '1146', 'yuquan_xuxinyi' -> 'xuxinyi'."""
    return ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid


def _load_cohort() -> dict:
    return json.loads(COHORT_JSON.read_text())


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return savefig_pub(fig, out_path, dpi=300)


# ============================================================================
# Figure 1 -- cohort_two_state_vs_geometry.png
# ============================================================================
NEAR_ORTHO_SUBJECTS = ["epilepsiae_1084", "epilepsiae_1150", "epilepsiae_583"]
RHO_POS_SUBJECTS = ["epilepsiae_922", "yuquan_xuxinyi", "yuquan_zhangkexuan"]
LABEL_SUBJECTS = set(NEAR_ORTHO_SUBJECTS) | set(RHO_POS_SUBJECTS)

TIER_STYLE = {
    "reciprocal": dict(color=COL_RECIP, marker="o"),
    "oblique": dict(color=COL_OBLIQUE, marker="^"),
}


def plot_fig1_two_state_vs_geometry() -> Path:
    cohort = _load_cohort()
    points = cohort["cohort"]["avoid_middle_scatter"]["points"]
    plotted = [p for p in points if p["frac_near_zero"] is not None]
    skipped = [p["subject"] for p in points if p["frac_near_zero"] is None]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    style_panel(ax)
    ax.tick_params(labelsize=10.5, width=1.2, length=5)

    for tier in ("reciprocal", "oblique"):
        sub = [p for p in plotted if p["template_pair_tier"] == tier]
        if not sub:
            continue
        sty = TIER_STYLE[tier]
        ax.scatter(
            [p["rho_AB"] for p in sub], [p["frac_near_zero"] for p in sub],
            color=sty["color"], marker=sty["marker"], s=95,
            edgecolor="white", linewidth=0.9, zorder=3,
            label=f"{tier} template pair (n={len(sub)})",
        )

    # Defensive: if a tier outside the locked reciprocal/oblique pair ever shows up,
    # surface it instead of silently dropping those subjects from the plot.
    unexpected = sorted({p["template_pair_tier"] for p in plotted} - set(TIER_STYLE))
    if unexpected:
        sub = [p for p in plotted if p["template_pair_tier"] in unexpected]
        ax.scatter([p["rho_AB"] for p in sub], [p["frac_near_zero"] for p in sub],
                   color="#999999", marker="s", s=95, edgecolor="white", linewidth=0.9,
                   zorder=3, label=f"other tier ({','.join(unexpected)}, n={len(sub)})")
        print(f"WARNING fig1: unexpected template_pair_tier values: {unexpected}")

    for p in plotted:
        if p["subject"] in LABEL_SUBJECTS:
            ax.annotate(
                _bare_label(p["subject"]), (p["rho_AB"], p["frac_near_zero"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.6,
                color="0.15", zorder=4,
            )

    xs = np.array([p["rho_AB"] for p in plotted])
    ys = np.array([p["frac_near_zero"] for p in plotted])
    xpad = 0.07 * (xs.max() - xs.min())
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(-0.018, ys.max() * 1.22)

    ax.axhline(0, color="0.35", lw=1.1, ls="--", zorder=1)

    ax.set_xlabel(r"Template-pair correlation $\rho_{AB}$", fontsize=11.5)
    ax.set_ylabel(
        rf"Fraction of windows near axis midpoint ($|C_{{AB}}|$ < {DELTA_SIDE})", fontsize=11.5,
    )
    ax.set_title(
        "Ictal energy avoids the A-B axis midpoint across template geometries",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)

    n_le10 = int(np.sum(ys <= 0.10))
    caption_lines = [
        f"Even near $\\rho_{{AB}}$≈0 or $\\rho_{{AB}}$>0 (geometry does not force lateralization), the "
        f"midpoint stays mostly empty (≤10% of windows for {n_le10}/{len(plotted)} subjects) — the",
        "two-state occupancy is largely real, not a geometric artifact. Reciprocal-tier subjects "
        "(anti-correlated template pairs) sit at 0.",
    ]
    if skipped:
        caption_lines.append(
            f"({len(skipped)} subjects [{', '.join(_bare_label(s) for s in skipped)}] have no usable "
            "peri-onset windows and are not plotted.)"
        )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.235)
    fig.text(0.5, 0.005, "\n".join(caption_lines), ha="center", va="bottom", fontsize=8.6,
              color="0.25", transform=fig.transFigure)

    out_path = OUT_DIR / "cohort_two_state_vs_geometry.png"
    return _save(fig, out_path)


# ============================================================================
# Figure 2 -- cohort_ab_typing.png  (key figure)
# ============================================================================
CATEGORY_ORDER = ["A_type", "bimodal", "mid", "low_data", "B_type"]
CATEGORY_COLOR = {
    "A_type": COL_A,
    "bimodal": COL_BIMODAL,
    "mid": COL_MID,
    "low_data": COL_LOWDATA,
    "B_type": COL_B,
}
CATEGORY_LEGEND_LABEL = {
    "A_type": "A-dominant seizure",
    "bimodal": "bimodal (switches side)",
    "mid": "no clear side",
    "low_data": "low data (<3 usable windows)",
    "B_type": "B-dominant seizure",
}


def _classify_seizure(sd: dict) -> str:
    """One mutually-exclusive category per seizure, priority order:
    data-quality gate first (low_data), then the dynamic bimodal/switch label,
    then the whole-seizure A/B side call, with 'mid' as the remaining catch-all.
    This partition sums exactly to n_seizures (verified against seizure_type_counts
    in cohort_analysis.json: A_type+B_type+mid == n_seizures already; bimodal/
    low_data are pulled out of that partition here so the stacked bar is clean).
    """
    npw = sd.get("n_present_windows")
    if npw is None or npw < 3:
        return "low_data"
    if sd.get("event_class") == "switch":
        return "bimodal"
    if sd.get("seizure_type") == "A_type":
        return "A_type"
    if sd.get("seizure_type") == "B_type":
        return "B_type"
    return "mid"


def _subject_counts(subject_rec: dict) -> dict:
    counts = {c: 0 for c in CATEGORY_ORDER}
    for sd in subject_rec["seizure_details"]:
        counts[_classify_seizure(sd)] += 1
    return counts


def plot_fig2_ab_typing() -> Path:
    cohort = _load_cohort()
    per_subject = cohort["per_subject"]
    typing = cohort["cohort"]["typing"]

    rows = []
    for rec in per_subject:
        rows.append({
            "subject": rec["subject"],
            "n_seizures": rec["n_seizures"],
            "counts": _subject_counts(rec),
            "null_p": rec["typing_purity_test"]["null_p"],
            "two_type": bool(rec["two_type_distinguishable"]),
        })
    rows.sort(key=lambda r: (-r["n_seizures"], r["subject"]))

    n = len(rows)
    row_h = 0.40
    fig_h = max(6.5, n * row_h + 2.6)
    fig, ax = plt.subplots(figsize=(10.6, fig_h))

    ys = np.arange(n)[::-1]
    for row, y in zip(rows, ys):
        left = 0
        for cat in CATEGORY_ORDER:
            w = row["counts"][cat]
            if w == 0:
                continue
            kwargs = {}
            if cat == "low_data":
                kwargs.update(hatch="///", edgecolor="white", linewidth=0.4)
            else:
                kwargs.update(edgecolor="white", linewidth=0.4)
            ax.barh(y, w, left=left, height=0.72, color=CATEGORY_COLOR[cat], zorder=3, **kwargs)
            left += w

        marker_null_p = row["null_p"]
        if marker_null_p is not None and marker_null_p == marker_null_p and marker_null_p < 0.05:
            ax.annotate("*", xy=(1.015, y), xycoords=("axes fraction", "data"),
                        ha="left", va="center", fontsize=17, color="black",
                        fontweight="bold", clip_on=False, zorder=5)
        if row["two_type"]:
            # unfilled circle -- needs a real marker (Text can't do an open ring cleanly)
            ax.plot([1.055], [y], transform=ax.get_yaxis_transform(), marker="o",
                     markersize=7, markerfacecolor="none", markeredgecolor="black",
                     markeredgewidth=1.2, clip_on=False, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{_pretty(r['subject'])}  (n={int(round(r['n_seizures']))})" for r in rows], fontsize=9.3)
    ax.set_ylim(-0.7, n - 0.3)

    max_n = max(r["n_seizures"] for r in rows)
    ax.set_xlim(0, max_n * 1.02)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9.5, width=1.1, length=5)
    ax.set_xlabel("Number of seizures", fontsize=11.5)

    fig.suptitle(
        "Seizures reproducibly occupy A-source vs B-source states (per subject)",
        fontsize=13, fontweight="bold", x=0.42, y=0.995,
    )
    fig.text(
        0.42, 0.965,
        f"significant A/B typing: {int(typing['n_typing_significant'])}/{int(typing['n_subjects_analyzed'])} "
        f"subjects   |   two seizure types distinguishable: "
        f"{int(typing['n_two_type_distinguishable'])}/{int(typing['n_subjects_analyzed'])} subjects",
        ha="center", fontsize=10, color="0.25",
    )

    legend_handles = [Patch(facecolor=CATEGORY_COLOR[c],
                             edgecolor="0.5" if c in ("mid", "low_data") else "none",
                             hatch="///" if c == "low_data" else None,
                             label=CATEGORY_LEGEND_LABEL[c])
                       for c in CATEGORY_ORDER]
    legend_handles += [
        Line2D([0], [0], marker="*", linestyle="none", markersize=13, color="black",
               label="A/B typing exceeds label-shuffle null (p<0.05)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor="none", markeredgecolor="black",
               label="two seizure types distinguishable (≥2 A-type & ≥2 B-type)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(left=0.19, right=0.90, top=0.93, bottom=0.14)
    out_path = OUT_DIR / "cohort_ab_typing.png"
    return _save(fig, out_path)


# ============================================================================
# Figure 3 -- cohort_h1_nearonset_forest.png
# ============================================================================


def plot_fig3_h1_forest() -> Path:
    cohort = _load_cohort()
    per_subject = cohort["per_subject"]
    h1_cohort = cohort["cohort"]["H1"]

    eligible_subjects = [rec["subject"] for rec in per_subject if rec.get("H1_eligible")]
    rows = []
    for subj in eligible_subjects:
        summary_fp = PER_SUBJECT_DIR / f"{subj}_scaffold_ab_summary.json"
        h1 = json.loads(summary_fp.read_text())["H1"]
        rows.append({
            "subject": subj,
            "L_obs": h1["L_obs"],
            "L_null_p95": h1["L_null_p95"],
            "locked": bool(h1["subject_locked"]),
            "p": h1["p"],
            "n_valid_seizures": h1.get("n_valid_seizures"),
        })
    rows.sort(key=lambda r: r["L_obs"], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, 1.15 * n + 2.9))
    style_panel(ax)
    ax.tick_params(labelsize=11, width=1.2, length=5)

    ys = np.arange(n)[::-1]
    for row, y in zip(rows, ys):
        color = COL_LOCKED if row["locked"] else COL_NOTLOCKED
        ax.plot([row["L_obs"], row["L_null_p95"]], [y, y], color="0.72", lw=1.4, zorder=1)
        ax.plot([row["L_null_p95"], row["L_null_p95"]], [y - 0.16, y + 0.16],
                 color="0.35", lw=2.2, solid_capstyle="butt", zorder=2)
        ax.scatter([row["L_obs"]], [y], s=190, color=color, edgecolor="black",
                   linewidth=1.0, zorder=3)

    ax.axvline(0, color="0.55", ls="--", lw=1.1, zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{_pretty(r['subject'])}  (n={int(round(r['n_valid_seizures']))} seizures)" for r in rows],
        fontsize=11,
    )
    ax.set_ylim(-0.75, n - 0.25)

    xs = [r["L_obs"] for r in rows] + [r["L_null_p95"] for r in rows]
    xpad = 0.18 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)

    ax.set_xlabel(
        "Near-onset lateral polarization increase\n"
        r"($|C_{AB}|$ near-onset $-$ $|C_{AB}|$ far pre-ictal)", fontsize=11,
    )
    k, m, p_cohort = int(h1_cohort["k"]), int(h1_cohort["m"]), h1_cohort["p"]
    ax.set_title(
        "Near-onset lateral polarization is not a cohort-level effect\n"
        f"(only {k}/{m} eligible subjects locked)",
        fontsize=12.5, fontweight="bold", pad=12,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=COL_LOCKED, markeredgecolor="black", label="locked (exceeds null)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=COL_NOTLOCKED, markeredgecolor="black", label="not locked"),
        Line2D([0], [0], color="0.35", lw=2.2, label="95th percentile of time-shuffled null"),
        Line2D([0], [0], color="0.55", ls="--", lw=1.1, label="no near-onset change"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, 0.115))

    n_cohort = len(per_subject)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.80, bottom=0.34)
    fig.text(
        0.5, 0.01,
        f"k/m = {k}/{m} eligible subjects locked, one-sided exact binomial p = {p_cohort:.2f}. "
        f"Only subjects meeting the near-onset testability criteria (≥3 usable seizures, testable\n"
        f"scaffold geometry) reach this test; the other {n_cohort - m}/{n_cohort} cohort subjects "
        "do not qualify.",
        ha="center", va="bottom", fontsize=8.4, color="0.25", transform=fig.transFigure,
    )

    out_path = OUT_DIR / "cohort_h1_nearonset_forest.png"
    return _save(fig, out_path)


def main() -> None:
    _setup_rc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_fig1_two_state_vs_geometry()
    p2 = plot_fig2_ab_typing()
    p3 = plot_fig3_h1_forest()
    for p in (p1, p2, p3):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
