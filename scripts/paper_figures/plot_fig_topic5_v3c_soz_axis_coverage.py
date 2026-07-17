#!/usr/bin/env python3
"""Paper-ready Topic 5 V3c figure: interictal HFO axis coverage of clinical SOZ,
and the gated ictal recruitment timing of the axis-surplus.

One composite figure (mirrors the energy-field paper figure genre):
  A  the two spatial labels on one real patient layout (axis vs clinical SOZ)
  B  cohort decision — do the three tests pass? (per-subject counts + cohort null)
  C  evidence ladder — what can we claim, and what we cannot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts._topic5_v3_io import classify_subject_contacts  # noqa: E402
from scripts._topic5_v3c_io import axis_soz_join, load_soz  # noqa: E402
from src.topic5_v3_mode_transition import load_v3_config  # noqa: E402

RES = ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage/broad"
LAYOUT = ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
OUT_DIR = ROOT / "results/paper-ready-figure/fig_topic5_v3c_soz_axis_coverage/figures"
EXAMPLE = "epilepsiae_1146"

COL_COVER = "#c0392b"    # A∩S : axis contact that is clinical SOZ
COL_SURP = "#1f8a8a"     # A∖S : axis surplus (beyond SOZ)
COL_BG = "#d0d0d0"       # background / non-axis pool
COL_GREEN = "#5f9e6e"
COL_AMBER = "#d3a13c"
COL_RED = "#c25b58"
COL_TRACK = "#e3e3e3"    # unfilled part of a count bar


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.8, "ytick.labelsize": 9.2,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _panel_label(ax, label):
    ax.text(-0.02, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold")


def _load_categories(ds_sid):
    cfg = load_v3_config()
    dataset, subj = ds_sid.split("_", 1)
    cls = classify_subject_contacts(ds_sid, "broad", cfg)
    j = axis_soz_join(cls, load_soz(dataset, subj))
    return set(j["covered"]), set(j["surplus"]), j


# ---------------------------------------------------------------- Panel A
def _draw_geometry(ax):
    ax.set_axis_off()
    _panel_label(ax, "A")
    ax.text(0.02, 0.99, "Two spatial labels on one patient", fontsize=10.6,
            fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.945,
            "On each electrode the axis covers the clinical onset zone (red) and\n"
            "extends beyond it into surplus contacts (teal). Timing is tested in B.",
            fontsize=8.5, color="0.30", ha="left", va="top", linespacing=1.3)

    rec = json.loads((LAYOUT / f"{EXAMPLE}_t_a.json").read_text())["channels"]
    covered, surplus, j = _load_categories(EXAMPLE)
    xy = {c["name"]: (float(c["x_norm"]), float(c["y_norm"])) for c in rec}
    shaft = {c["name"]: c["shaft"] for c in rec}
    along = {c["name"]: float(c.get("along_axis_mm", 0.0)) for c in rec}

    pts = np.array(list(xy.values()))
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin) * 1.14
    rect = (0.05, 0.34, 0.90, 0.50)   # x0,y0,w,h inside the axis-off panel

    # subtle background card so the layout reads as one anatomical panel
    ax.add_patch(Rectangle((rect[0] - 0.01, rect[1] - 0.02), rect[2] + 0.02, rect[3] + 0.04,
                           facecolor="#f4f6f8", edgecolor="#e2e6ea", linewidth=1.0, zorder=0))

    def T(nm):
        x, y = xy[nm]
        return (rect[0] + rect[2] / 2 + (x - cx) / span * rect[2],
                rect[1] + rect[3] / 2 + (y - cy) / span * rect[3])

    # faint shaft connectors + shaft name at the shallow end
    for sh in set(shaft.values()):
        names = sorted([n for n in xy if shaft[n] == sh], key=lambda n: along[n])
        pxy = np.array([T(n) for n in names])
        if len(pxy) > 1:
            ax.plot(pxy[:, 0], pxy[:, 1], color="0.70", lw=1.3, zorder=1)
        hx, hy = pxy[0]
        ax.text(hx - 0.018, hy + 0.02, sh, fontsize=8.0, color="0.35",
                ha="right", va="bottom", zorder=2, style="italic")
    # contacts by category
    for grp, col in [(covered, COL_COVER), (surplus, COL_SURP)]:
        p = np.array([T(n) for n in xy if n in grp])
        ax.scatter(p[:, 0], p[:, 1], s=150, facecolor=col, edgecolor="white",
                   linewidth=1.3, zorder=3)

    # inline legend (markers + text), placed under the layout
    ax.scatter([0.09], [0.205], s=150, facecolor=COL_COVER, edgecolor="white", linewidth=1.3)
    ax.text(0.145, 0.205, "axis ∩ SOZ  —  covered clinical onset zone", fontsize=8.9,
            ha="left", va="center", color="0.12")
    ax.scatter([0.09], [0.115], s=150, facecolor=COL_SURP, edgecolor="white", linewidth=1.3)
    ax.text(0.145, 0.115, "axis surplus  —  extends beyond the SOZ", fontsize=8.9,
            ha="left", va="center", color="0.12")
    ax.text(0.02, 0.02,
            f"E1146:  axis covers all {j['n_soz']}/{j['n_soz']} clinical SOZ contacts, "
            f"plus {j['n_surplus']} surplus.",
            fontsize=8.9, ha="left", va="center", color="0.12", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------- Panel B
def _count_bar(ax, y, x0, w, n_pass, n_total, color):
    ax.add_patch(Rectangle((x0, y - 0.028), w, 0.056, facecolor=COL_TRACK,
                           edgecolor="none"))
    if n_total > 0:
        ax.add_patch(Rectangle((x0, y - 0.028), w * n_pass / n_total, 0.056,
                               facecolor=color, edgecolor="none"))
    ax.text(x0 + w + 0.015, y, f"{n_pass}/{n_total} subjects", fontsize=8.4,
            ha="left", va="center", color="0.30")


def _draw_decisions(ax):
    ax.set_axis_off()
    _panel_label(ax, "B")
    ax.text(0.0, 0.99, "Cohort decision: do the tests pass?", fontsize=10.6,
            fontweight="bold", ha="left", va="top")
    ax.text(0.0, 0.925, "Each subject vs its own same-shaft null; cohort = median-null test",
            fontsize=8.4, color="0.35", ha="left", va="top")

    cov = pd.read_csv(RES / "coverage_subject.csv")
    cov = cov[cov["eligible"] == True]  # noqa: E712
    covj = json.loads((RES / "coverage_cohort.json").read_text())
    sur = pd.read_csv(RES / "surplus_spatial" / "surplus_subject.csv")
    surj = json.loads((RES / "surplus_spatial" / "surplus_spatial_cohort.json").read_text())
    lat = pd.read_csv(RES / "latency" / "latency_subject.csv")
    latj = json.loads((RES / "latency" / "latency_cohort.json").read_text())

    n_cov = int((cov["coverage_null_p"] < 0.05).sum()); N_cov = len(cov)
    n_sur = int((sur["dist_null_p"] < 0.05).sum()); N_sur = int(surj.get("n_spatial_eligible", len(sur)))
    lat_e = lat[lat["eligible"] == True]  # noqa: E712
    n_lat = int((lat_e["auc_null_p"] < 0.05).sum()); N_lat = len(lat_e)

    rows = [
        (0.71, "Axis covers clinical SOZ", n_cov, N_cov, COL_GREEN,
         f"cohort p = {covj['p_value']:.3f}", "beyond implant geometry", True),
        (0.47, "Surplus hugs SOZ in space", n_sur, N_sur, COL_AMBER,
         f"cohort p = {surj['p_value']:.2f}", "spatial structure not established", False),
        (0.23, "Surplus recruited later than SOZ", n_lat, N_lat, COL_GREEN,
         f"AUC {latj['obs_cohort_median_auc']:.2f}, +{latj['delta_t_med']:.1f}s, p = {latj['p_value']:.3f}",
         "downstream scaffold", True),
    ]
    for y, name, nps, N, col, stat, verdict, ok in rows:
        ax.text(0.0, y + 0.075, name, fontsize=9.4, fontweight="bold", ha="left", va="center", color="0.10")
        _count_bar(ax, y, 0.0, 0.34, nps, N, col)
        ax.text(0.52, y + 0.03, stat, fontsize=8.5, ha="left", va="center", color="0.20")
        mark = "✓" if ok else "—"
        mc = COL_GREEN if ok else COL_AMBER
        ax.text(0.52, y - 0.035, f"{mark} {verdict}", fontsize=8.8, ha="left", va="center",
                color=mc, fontweight="bold")

    ax.text(0.0, 0.035,
            "Cohort-level: axis covers SOZ beyond geometry and its surplus fires downstream;\n"
            "per-subject signal is sparse and spatial specificity is not established.",
            fontsize=8.7, color="0.10", ha="left", va="center", fontweight="bold", linespacing=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------- Panel C
def _draw_ladder(ax):
    ax.set_axis_off()
    _panel_label(ax, "C")
    ax.text(0.0, 0.99, "What can we claim?", fontsize=10.6, fontweight="bold", ha="left", va="top")

    covj = json.loads((RES / "coverage_cohort.json").read_text())
    surj = json.loads((RES / "surplus_spatial" / "surplus_spatial_cohort.json").read_text())
    latj = json.loads((RES / "latency" / "latency_cohort.json").read_text())

    cards = [
        ("PASS", COL_GREEN, "Axis covers clinical SOZ",
         f"beyond same-shaft implant geometry (cohort p = {covj['p_value']:.3f}).\n"
         "6/7 subjects fully covered; 1 (E635) misses 3 SOZ contacts."),
        ("LIMITED", COL_AMBER, "Structured peri-SOZ surplus",
         f"surplus is only weakly closer to SOZ than random (cohort p = {surj['p_value']:.2f});\n"
         "spatial specificity is NOT established."),
        ("PASS", COL_GREEN, "Surplus recruited downstream",
         f"fires ~{latj['delta_t_med']:.1f}s after the SOZ core (AUC {latj['obs_cohort_median_auc']:.2f}, "
         f"p = {latj['p_value']:.3f});\nassay-quality gated (1 subject excluded)."),
        ("NO", COL_RED, "Beyond HFO enrichment",
         "the geometry null does not control for SOZ being HFO-rich;\n"
         "an HFO-rate-matched null is required (not yet run)."),
    ]
    ys = [0.80, 0.575, 0.35, 0.125]
    for y, (status, color, title, body) in zip(ys, cards):
        ax.add_patch(Rectangle((0.0, y - 0.058), 0.155, 0.115, facecolor=color, edgecolor="none"))
        ax.text(0.0775, y, status, color="white", fontsize=9.0, fontweight="bold", ha="center", va="center")
        ax.text(0.19, y + 0.042, title, fontsize=9.2, fontweight="bold", ha="left", va="center", color="0.10")
        ax.text(0.19, y - 0.005, body, fontsize=7.9, ha="left", va="top", color="0.30", linespacing=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _write_readme():
    (OUT_DIR / "README.md").write_text(
        "# Topic 5 V3c — interictal axis coverage of clinical SOZ (paper figure)\n\n"
        "### topic5_v3c_soz_axis_coverage_main.png\n\n"
        "Paper-ready summary of V3c. Panel A shows the two spatial labels (interictal HFO\n"
        "propagation axis vs clinical SOZ) on the real E1146 electrode layout: the axis covers\n"
        "all clinical SOZ and extends into a surplus set along the same shafts. Panel B is the\n"
        "cohort adjudication of the three tests (coverage / spatial structure / recruitment\n"
        "timing) with per-subject counts and the cohort median-null decision. Panel C is the\n"
        "evidence ladder separating the licensed claims (coverage beyond geometry; surplus\n"
        "recruited downstream) from the unsupported ones (structured peri-SOZ surplus; beyond\n"
        "HFO enrichment).\n\n"
        "**关注点**：承重口径=覆盖超出植入几何（整队 p=0.006）+ 外扩触点发作时偏晚约 3 秒（下游）；\n"
        "空间“贴着 SOZ 成环”未确立（p=0.11）；同杆 null 只控几何、不控 HFO 富集（措辞封在 beyond geometry）。\n"
    )


def plot():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_rc()
    fig = plt.figure(figsize=(12.4, 6.9))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.05, 1.5],
                          height_ratios=[1.0, 1.0], left=0.045, right=0.985,
                          bottom=0.04, top=0.85, wspace=0.16, hspace=0.30)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_geometry(ax_a)
    _draw_decisions(ax_b)
    _draw_ladder(ax_c)
    fig.suptitle("Interictal axis covers clinical SOZ beyond geometry; its surplus is recruited downstream",
                 fontsize=13.0, fontweight="bold", x=0.5, ha="center", y=0.955)
    out = OUT_DIR / "topic5_v3c_soz_axis_coverage_main.png"
    fig.savefig(out, dpi=300)
    fig.savefig(OUT_DIR / "topic5_v3c_soz_axis_coverage_main.pdf")
    plt.close(fig)
    _write_readme()
    print(f"wrote {out}")


if __name__ == "__main__":
    plot()
