"""Didactic schematic: why reading interictal propagation *direction* needs real
electrode coordinates, and how a coordinate-blind shaft-collapsed read-out fails on a
real subject.

Two panels, one scientific question each (CLAUDE.md §7):
  A. Idealized mechanism -- collapsing each SEEG shaft to its mean timing discards the
     within-shaft order, so the only expressible direction is shaft-to-shaft, which can
     sit ~90 deg off the true propagation axis.
  B. Real failure -- E1146 (narrow substrate, template TB): the two shafts' mean timings
     nearly tie, so the shaft-collapsed axis points 105 deg away from the coordinate axis
     and predicts held-out order WORSE than a random direction.

In-figure text is English (no CJK font dependency); the Chinese walk-through lives in the
figures/README.md.  Real Panel-B geometry is reconstructed from the same loaders the axis
runner uses (build_plane_xy on the t_a reference plane + class_aggregate_contact_values),
so the drawn axes match results/.../axis_robustness/per_subject/narrow/epilepsiae_1146.json.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_event_resolved_alignment import (
    load_event_labels_ranks, map_clusters_to_templates,
    class_aggregate_contact_values, build_plane_xy)

RESULTS = Path("/home/honglab/leijiaxin/HFOsp/results")
GEOM_NARROW = (RESULTS / "spatial_modulation" / "propagation_geometry"
               / "observation_readout" / "real_subjects")
AXIS_JSON = Path("results/topic5_ictal_recruitment/field_reversal/axis_robustness"
                 "/per_subject/narrow/epilepsiae_1146.json")
OUT = Path("results/topic5_ictal_recruitment/field_reversal/axis_robustness/figures"
           "/why_coordinates_schematic.png")

COORD_C = "#2ca02c"   # coordinate axis (correct)
SHAFT_C = "#d62728"   # shaft-collapsed axis (wrong)
CMAP = "viridis"      # early (dark) -> late (bright)


def _shaft(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else name


def _load_real_case():
    """E1146 narrow TB: contacts (x,y) on the t_a reference plane, TB timing value, shaft
    label, plus the coordinate + shaft-collapse axis units from the axis JSON."""
    pa = json.load(open(GEOM_NARROW / "epilepsiae_1146_t_a.json"))
    pb = json.load(open(GEOM_NARROW / "epilepsiae_1146_t_b.json"))
    bundle = load_event_labels_ranks("epilepsiae", "1146", broad=False)
    order = bundle["channel_names"]

    def vec(name_to_val):
        return [name_to_val.get(o, np.nan) for o in order]

    ta_rank = vec({c["name"]: c["typical_rank"] for c in pa["channels"]})
    tb_rank = vec({c["name"]: c["typical_rank"] for c in pb["channels"]})
    c0 = np.asarray(bundle["cluster_template_ranks"][0], float)
    c1 = np.asarray(bundle["cluster_template_ranks"][1], float)
    cmap = map_clusters_to_templates(c0, c1, ta_rank, tb_rank)
    tb_label = {v: k for k, v in cmap["map"].items()}["t_b"]

    pxy = build_plane_xy(pa)                       # reference frame = t_a plane (P0)
    cav = class_aggregate_contact_values(bundle, tb_label)
    names = [n for n in cav if n in pxy]
    x = np.array([pxy[n][0] for n in names])
    y = np.array([pxy[n][1] for n in names])
    val = np.array([cav[n]["value"] for n in names])
    shafts = np.array([_shaft(n) for n in names])

    aj = json.load(open(AXIS_JSON))["per_class"]["TB"]
    return dict(x=x, y=y, val=val, shafts=shafts, names=names,
                raw_unit=np.array(aj["raw_contact_unit"], float),
                seq_unit=np.array(aj["sequence_unit"], float),
                angle=aj["angle_sequence_raw"],
                rho_raw=aj["held_out_raw_median"],
                rho_seq=aj["held_out_sequence_median"],
                rho_null=aj["held_out_null_median"])


def _axis_arrow(ax, cx, cy, unit, span, color, ls, label, label_end=+1):
    """Draw a propagation axis as a line through (cx,cy) along +/-unit, arrowhead on the
    increasing-value (late) end. `label_end` (+1 late / -1 early) picks which end the text
    sits at, to dodge clutter."""
    u = unit / (np.linalg.norm(unit) + 1e-12)
    p_late = np.array([cx, cy]) + u * span
    p_early = np.array([cx, cy]) - u * span
    ax.plot([p_early[0], p_late[0]], [p_early[1], p_late[1]],
            color=color, ls=ls, lw=2.6, zorder=4, solid_capstyle="round")
    ax.annotate("", xy=p_late, xytext=np.array([cx, cy]) + u * span * 0.55,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6), zorder=5)
    tip = p_late if label_end > 0 else p_early
    ux = u[0] * label_end
    ax.text(tip[0], tip[1], "  " + label if ux >= 0 else label + "  ",
            color=color, fontsize=9.5, fontweight="bold", va="center",
            ha="left" if ux >= 0 else "right", zorder=6)


def _panel_ideal(ax):
    """Idealized: 2 parallel shafts, propagation runs UP each shaft (within-shaft timing
    gradient). Coordinate read-out recovers UP; shaft-collapse can only say left<->right."""
    ny = 6
    yy = np.linspace(0.0, 1.0, ny)
    xs = [0.0, 0.9]
    X, Y, V = [], [], []
    for xc in xs:
        for yv in yy:
            X.append(xc + 0.02 * np.sin(yv * 6)); Y.append(yv); V.append(yv)
    X, Y, V = np.array(X), np.array(Y), np.array(V)
    ax.scatter(X, Y, c=V, cmap=CMAP, s=150, edgecolor="k", linewidth=0.6, zorder=3)

    # shaft means (big hollow markers): both sit at mid-height with identical mean timing
    for xc in xs:
        ax.scatter([xc], [0.5], s=430, marker="D", facecolor="none",
                   edgecolor=SHAFT_C, linewidth=2.4, zorder=4)
    ax.text(0.0, 0.78, "shaft means\n(equal timing)", color=SHAFT_C, fontsize=8.5,
            ha="center", va="bottom", fontweight="bold")

    cx, cy = X.mean(), Y.mean()
    _axis_arrow(ax, cx, cy, np.array([0.0, 1.0]), 0.62, COORD_C, "-", "coordinate")
    _axis_arrow(ax, 0.45, 0.5, np.array([1.0, 0.0]), 0.52, SHAFT_C, "--", "shaft-collapse")

    ax.set_title("A  Idealized: direction needs real coordinates",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.text(0.5, -0.30,
            "Propagation runs UP each shaft (within-shaft timing order).\n"
            "Collapsing each shaft to its mean erases that order, so the\n"
            "shaft-collapsed read-out can only express shaft↔shaft (≈90° off).",
            transform=ax.transAxes, fontsize=9, ha="center", va="top", color="#333333")
    ax.set_xlim(-0.45, 1.5); ax.set_ylim(-0.15, 1.25)


def _panel_real(ax, d):
    sc = ax.scatter(d["x"], d["y"], c=d["val"], cmap=CMAP, s=170,
                    edgecolor="k", linewidth=0.6, zorder=3)
    # shaft-mean markers + per-shaft mean timing label (shows the near-tie). Labels to the
    # LEFT of each diamond so they clear the shaft-collapse arrowhead.
    for s in sorted(set(d["shafts"])):
        m = d["shafts"] == s
        mx, my, mv = d["x"][m].mean(), d["y"][m].mean(), d["val"][m].mean()
        ax.scatter([mx], [my], s=430, marker="D", facecolor="none",
                   edgecolor=SHAFT_C, linewidth=2.4, zorder=4)
        ax.text(mx - 0.05, my, f"{s} mean {mv:.2f}", color=SHAFT_C, fontsize=8,
                ha="right", va="center", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    cx, cy = d["x"].mean(), d["y"].mean()
    span = 0.46 * max(d["x"].ptp(), d["y"].ptp())
    _axis_arrow(ax, cx, cy, d["raw_unit"], span, COORD_C, "-", "coordinate")
    _axis_arrow(ax, cx, cy, d["seq_unit"], span, SHAFT_C, "--", "shaft-collapse",
                label_end=-1)

    ax.set_title("B  Real failure — E1146 (narrow, TB)",
                 fontsize=11.5, fontweight="bold", loc="left")
    txt = (f"axes {d['angle']:.0f}° apart\n"
           f"held-out order prediction (Spearman ρ):\n"
           f"  coordinate  ρ = {d['rho_raw']:+.2f}\n"
           f"  shaft-collapse  ρ = {d['rho_seq']:+.2f}  (worse than random)\n"
           f"  random axis  ρ = {d['rho_null']:+.2f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8.8, va="top",
            ha="right", family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.95))
    ax.text(0.5, -0.30,
            "The two shafts' mean timings nearly tie (0.51 vs 0.50), so collapsing to\n"
            "shaft means throws away almost all signal; the residual points ~orthogonal\n"
            "to the true axis. The real order lives in the within-shaft (esp. 11-contact) gradient.",
            transform=ax.transAxes, fontsize=9, ha="center", va="top", color="#333333")
    return sc


def main():
    d = _load_real_case()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax in (axA, axB):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbbbbb")

    _panel_ideal(axA)
    sc = _panel_real(axB, d)

    # shared legend (one per figure) + one shared colorbar
    handles = [
        Line2D([0], [0], color=COORD_C, lw=2.6, label="coordinate axis (uses real x,y) — correct"),
        Line2D([0], [0], color=SHAFT_C, lw=2.6, ls="--",
               label="shaft-collapsed axis (drops within-shaft position) — misleads"),
        Line2D([0], [0], marker="D", color=SHAFT_C, ls="none", markerfacecolor="none",
               markeredgewidth=2.0, markersize=11, label="shaft mean (all the collapse keeps)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9.0,
               frameon=False, bbox_to_anchor=(0.5, 1.005))
    cbar = fig.colorbar(sc, ax=(axA, axB), fraction=0.03, pad=0.02, shrink=0.7)
    cbar.set_label("activation timing (early → late)", fontsize=9)
    cbar.set_ticks([d["val"].min(), d["val"].max()])
    cbar.set_ticklabels(["early", "late"])

    fig.suptitle("Reading propagation direction requires real coordinates, "
                 "not electrode-shaft order", fontsize=12.5, fontweight="bold", y=1.06)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.2, wspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
