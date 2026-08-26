#!/usr/bin/env python3
"""Draw the Epi-PRSSM H1--H3 hypothesis overview.

This is a schematic, not a result figure.  It translates the current v0.2
joint time--mark specification into three reader-facing panels:

H1  a state persists across IEDs and predicts future timing and repertoire
    after observer correction is switched off;
H2  that pre-event state changes the event continuation and may link the
    frozen interictal model to early-ictal recruitment;
H3  IED exposure over a finite timescale physically updates the state, above
    and beyond the information carried by event innovation.

The producer intentionally does not read evidence cards.  Every trajectory,
event raster and network is an illustrative glyph and is labelled as such in
the metadata and README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np

from src.topic5_epi_prssm.contracts import code_revision, package_hash
from src.topic5_epi_prssm.figure_style import COLOR, DOUBLE_COLUMN_MM, MM, apply_style


ASSET_ID = "epi_prssm_hypothesis_overview"
WIDTH_MM = DOUBLE_COLUMN_MM
HEIGHT_MM = 118.0

FS_PANEL = 10.5
FS_TITLE = 8.8
FS_SUBTITLE = 7.4
FS_LABEL = 7.0
FS_SMALL = 6.6

INK = "#2F2F2F"
MID = "#777777"
LIGHT = "#D5D5D5"
PALE = "#EEEEEE"
BLUE = COLOR["G1"]
TEAL = COLOR["G2"]
PURPLE = COLOR["G3"]
RUST = COLOR["exposure"]
ONSET = COLOR["onset"]


def _arrow(ax, start, end, *, color=INK, lw=0.9, style="-", mutation=8,
           connectionstyle="arc3", zorder=3):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, linestyle=style, color=color,
        connectionstyle=connectionstyle, shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _event_raster(ax, x, y, pattern, *, scale=1.0, alpha=1.0,
                  edgecolor="white", zorder=5):
    """Small contact-by-order event glyph; values are illustrative ranks."""
    cell_w, cell_h = 0.0105 * scale, 0.0115 * scale
    gap = 0.0022 * scale
    n = len(pattern)
    for i, value in enumerate(pattern):
        colour = plt.cm.viridis(float(value)) if np.isfinite(value) else (0.9, 0.9, 0.9, 1)
        ax.add_patch(Rectangle(
            (x - cell_w / 2, y + (i - (n - 1) / 2) * (cell_h + gap) - cell_h / 2),
            cell_w, cell_h, facecolor=colour, edgecolor=edgecolor,
            linewidth=0.25, alpha=alpha, zorder=zorder,
        ))


def _network(ax, center, route, *, radius=0.010, route_color=BLUE,
             faint=False, scale=1.0, zorder=4):
    """Compact fixed graph with one highlighted event route."""
    cx, cy = center
    coords = np.array([
        [-0.060, 0.030], [-0.020, 0.060], [0.030, 0.052], [0.065, 0.010],
        [0.038, -0.050], [-0.018, -0.060], [-0.065, -0.020], [0.000, 0.000],
    ])
    coords *= scale
    coords[:, 0] += cx
    coords[:, 1] += cy
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
             (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
    base_alpha = 0.34 if not faint else 0.18
    for i, j in edges:
        ax.plot(coords[[i, j], 0], coords[[i, j], 1], color="#AFAFAF",
                lw=0.55, alpha=base_alpha, zorder=zorder)
    for step, (i, j) in enumerate(zip(route[:-1], route[1:])):
        _arrow(ax, tuple(coords[i]), tuple(coords[j]), color=route_color,
               lw=1.15, mutation=6.5, zorder=zorder + 1)
    for i, (xx, yy) in enumerate(coords):
        in_route = i in route
        ax.add_patch(Circle(
            (xx, yy), radius * (1.08 if in_route else 0.85),
            facecolor="white", edgecolor=route_color if in_route else "#8B8B8B",
            linewidth=0.9 if in_route else 0.55,
            alpha=0.95 if not faint else 0.65, zorder=zorder + 2,
        ))
    return coords


def _base_axis(ax, letter: str, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.025, 1.025, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="top", ha="left", color=INK)
    ax.text(0.085, 1.015, title, transform=ax.transAxes, fontsize=FS_TITLE,
            fontweight="bold", va="top", ha="left", color=INK)


def draw_h1(ax):
    """Wide H1 panel: history and open-loop readout share one clock axis."""
    _base_axis(ax, "A", "H1 | Persistent state across irregularly timed IEDs")
    ax.text(0.025, 0.875, "observed long history", fontsize=FS_LABEL,
            color=INK, va="top")
    ax.text(0.405, 0.875, "observer off", fontsize=FS_LABEL,
            color=MID, ha="center", va="top")
    ax.text(0.535, 0.875, "future IEDs", fontsize=FS_LABEL,
            color=MID, ha="center", va="top")

    # Left two-thirds: irregular event sequence plus latent trajectory.
    x_start, x_end = 0.035, 0.635
    y_time = 0.665
    ax.plot([x_start, x_end], [y_time, y_time], color=INK, lw=0.75)
    _arrow(ax, (x_end - 0.015, y_time), (x_end + 0.008, y_time),
           color=INK, lw=0.75, mutation=6.5)
    event_x = np.array([0.07, 0.145, 0.245, 0.335, 0.455, 0.555, 0.615])
    patterns = [
        [0.05, 0.35, 0.65, 0.92], [0.82, 0.58, 0.31, 0.06],
        [0.12, 0.45, 0.72, 0.94], [0.76, 0.55, 0.25, 0.03],
        [0.10, 0.40, 0.74, 0.96], [0.70, 0.48, 0.22, 0.04],
        [0.08, 0.38, 0.68, 0.90],
    ]
    boundary = 0.405
    for xx, pat in zip(event_x, patterns):
        future = xx > boundary
        _event_raster(ax, xx, y_time + 0.115, pat, scale=0.82,
                      alpha=0.55 if future else 1.0,
                      edgecolor="#F7F7F7")
        ax.plot([xx, xx], [y_time - 0.018, y_time + 0.018],
                color=INK if not future else MID, lw=0.55)
    ax.plot([boundary, boundary], [0.155, 0.840], color="#858585", lw=0.8,
            ls=(0, (2.0, 2.0)))
    ax.text(x_end, y_time - 0.070, "clock time", fontsize=FS_SMALL,
            color=MID, ha="right", va="top")

    xs = np.linspace(x_start, x_end, 250)
    state = 0.335 + 0.090 * np.sin(2 * np.pi * (xs - 0.01) / 0.56) + 0.018 * np.sin(17 * xs)
    left = xs <= boundary
    ax.plot(xs[left], state[left], color=TEAL, lw=1.6)
    ax.plot(xs[~left], state[~left], color=TEAL, lw=1.35, ls=(0, (4, 2)))
    ax.text(x_start, 0.480, r"inferred state  $z(t)$", fontsize=FS_LABEL,
            color=TEAL, ha="left")
    ax.text(0.520, 0.205, "autonomous rollout", fontsize=FS_SMALL,
            color=TEAL, ha="center")
    for xx in event_x[event_x > boundary]:
        yy = np.interp(xx, xs, state)
        _arrow(ax, (xx, yy + 0.008), (xx, y_time + 0.030), color=TEAL,
               lw=0.65, mutation=5.0, zorder=2)

    # Right third: the joint prediction target, kept separate from the timeline.
    ax.plot([0.675, 0.675], [0.120, 0.865], color=LIGHT, lw=0.65)
    ax.text(0.710, 0.820, "critical test", fontsize=FS_SMALL, color=MID,
            fontweight="bold", va="top")
    ax.text(0.710, 0.690, "predict future timing", fontsize=FS_SUBTITLE,
            color=INK, va="center")
    ax.text(0.710, 0.555, "+ spatial repertoire", fontsize=FS_SUBTITLE,
            color=INK, va="center")
    _arrow(ax, (0.715, 0.405), (0.800, 0.405), color=TEAL, lw=1.1,
           mutation=7)
    ax.plot([0.825, 0.955], [0.330, 0.330], color=INK, lw=0.65)
    ax.plot([0.850, 0.850], [0.330, 0.395], color=INK, lw=0.8)
    ax.plot([0.925, 0.925], [0.330, 0.430], color=INK, lw=0.8)
    _event_raster(ax, 0.890, 0.535, [0.08, 0.35, 0.68, 0.94], scale=0.92)
    ax.text(0.890, 0.155, "no future marks fed back", fontsize=FS_SMALL,
            color=MID, ha="center", va="top")


def draw_h2(ax):
    _base_axis(ax, "B", "H2 | State-shaped repertoire")

    # H2a: same prefix, different continuation under different pre-event states.
    ax.text(0.055, 0.925, "H2a  single-event distribution", fontsize=FS_SUBTITLE,
            color=INK, fontweight="bold", va="top")
    ax.text(0.055, 0.865, "same observed prefix", fontsize=FS_LABEL,
            color=MID, va="top")
    prefix_xy = [(0.13, 0.735), (0.22, 0.735)]
    ax.plot([p[0] for p in prefix_xy], [p[1] for p in prefix_xy], color=INK,
            lw=1.0, zorder=2)
    for order, (xx, yy) in enumerate(prefix_xy):
        ax.add_patch(Circle((xx, yy), 0.018, facecolor="white", edgecolor=INK,
                            linewidth=0.9, zorder=3))
        ax.text(xx, yy, str(order + 1), fontsize=FS_SMALL, color=INK,
                ha="center", va="center", zorder=4)
    _arrow(ax, (0.245, 0.735), (0.34, 0.735), color=INK, lw=0.8,
           mutation=7)

    ax.add_patch(Circle((0.42, 0.785), 0.038, facecolor="white",
                        edgecolor=BLUE, linewidth=1.2))
    ax.add_patch(Circle((0.42, 0.665), 0.038, facecolor="white",
                        edgecolor=PURPLE, linewidth=1.2))
    ax.text(0.42, 0.785, r"$z_1$", fontsize=FS_LABEL, color=BLUE,
            ha="center", va="center")
    ax.text(0.42, 0.665, r"$z_2$", fontsize=FS_LABEL, color=PURPLE,
            ha="center", va="center")
    _arrow(ax, (0.46, 0.785), (0.59, 0.785), color=BLUE, lw=1.0, mutation=7)
    _arrow(ax, (0.46, 0.665), (0.59, 0.665), color=PURPLE, lw=1.0, mutation=7)

    _network(ax, (0.75, 0.785), [6, 0, 1, 2, 3], route_color=BLUE,
             radius=0.008, scale=0.78)
    _network(ax, (0.75, 0.665), [6, 0, 7, 4, 5], route_color=PURPLE,
             radius=0.008, scale=0.78)
    ax.text(0.91, 0.785, "A", fontsize=FS_LABEL, color=BLUE,
            fontweight="bold", ha="center", va="center")
    ax.text(0.91, 0.665, "B / STOP", fontsize=FS_LABEL, color=PURPLE,
            fontweight="bold", ha="center", va="center")
    ax.text(0.75, 0.575, "different suffix distributions", fontsize=FS_SMALL,
            color=MID, ha="center", va="top")

    # H2b: frozen interictal state to onset and early recruitment.
    ax.plot([0.055, 0.945], [0.525, 0.525], color=LIGHT, lw=0.65)
    ax.text(0.055, 0.475, "H2b  interictal-to-ictal link", fontsize=FS_SUBTITLE,
            color=INK, fontweight="bold", va="top")
    x0, x1 = 0.10, 0.91
    y0 = 0.300
    ax.plot([x0, x1], [y0, y0], color=INK, lw=0.75)
    _arrow(ax, (x1 - 0.02, y0), (x1 + 0.015, y0), color=INK, lw=0.75,
           mutation=6.5)
    last_ied = 0.31
    onset_x = 0.76
    _event_raster(ax, last_ied, y0 + 0.075, [0.08, 0.38, 0.67, 0.93], scale=0.85)
    ax.plot([last_ied, last_ied], [y0 - 0.012, y0 + 0.012], color=INK, lw=0.6)
    ax.axvline(onset_x, ymin=0.105, ymax=0.435, color=ONSET, lw=1.0)
    ax.axvspan(onset_x, onset_x + 0.12, ymin=0.105, ymax=0.435,
               color=ONSET, alpha=0.08, lw=0)
    ax.text(last_ied, 0.205, "last IED", fontsize=FS_SMALL,
            color=MID, ha="center", va="top")
    ax.text(onset_x, 0.420, "clinical onset", fontsize=FS_LABEL,
            color=ONSET, ha="center", va="bottom")
    ax.text(0.535, 0.390, "frozen-state rollout", fontsize=FS_SMALL,
            color=TEAL, ha="center", va="bottom")

    xs = np.linspace(last_ied, onset_x, 140)
    curve = 0.355 + 0.025 * np.sin(np.linspace(0, 1.6 * np.pi, len(xs))) + 0.06 * (xs - last_ied) / (onset_x - last_ied)
    ax.plot(xs, curve, color=TEAL, lw=1.45)
    # Early recruitment as ordered contact glyphs, not a measured heatmap.
    recruit_x = [0.79, 0.825, 0.86, 0.895]
    recruit_y = [0.345, 0.325, 0.365, 0.315]
    for rank, (xx, yy) in enumerate(zip(recruit_x, recruit_y)):
        ax.add_patch(Circle((xx, yy), 0.012, facecolor=plt.cm.viridis(rank / 3),
                            edgecolor="white", linewidth=0.4, zorder=5))
    ax.text(0.845, 0.205, "early recruitment", fontsize=FS_SMALL,
            color=INK, ha="center", va="top")
    ax.text(0.50, 0.090, "test against matched pseudo-onsets and nuisance timing",
            fontsize=FS_SMALL, color=MID, ha="center", va="center")


def draw_h3(ax):
    _base_axis(ax, "C", "H3 | IED-driven state update")
    ax.text(0.945, 0.925, "independent extension", transform=ax.transAxes,
            fontsize=FS_SMALL, color=RUST, ha="right", va="top")

    # Event train and finite-timescale exposure kernel.
    ax.text(0.055, 0.865, r"IED history  $\times$  timescale $\tau$",
            fontsize=FS_SUBTITLE, color=INK, fontweight="bold", va="top")
    base_y = 0.700
    ax.plot([0.07, 0.93], [base_y, base_y], color=INK, lw=0.75)
    impulse_x = np.array([0.11, 0.19, 0.24, 0.43, 0.48, 0.53, 0.72, 0.88])
    heights = np.array([0.055, 0.085, 0.045, 0.070, 0.100, 0.062, 0.075, 0.050])
    for xx, hh in zip(impulse_x, heights):
        ax.plot([xx, xx], [base_y, base_y + hh], color=RUST, lw=1.25)
        ax.add_patch(Circle((xx, base_y + hh), 0.006, facecolor=RUST,
                            edgecolor="none", zorder=4))
    ax.text(0.93, 0.660, "clock time", fontsize=FS_SMALL, color=MID,
            ha="right", va="top")

    xs = np.linspace(0.07, 0.93, 320)
    exposure = np.zeros_like(xs)
    tau = 0.105
    for xx, hh in zip(impulse_x, heights):
        exposure += (xs >= xx) * (hh / heights.max()) * np.exp(-(xs - xx) / tau)
    exposure = 0.505 + 0.095 * exposure / exposure.max()
    ax.fill_between(xs, 0.505, exposure, color=RUST, alpha=0.13, lw=0)
    ax.plot(xs, exposure, color=RUST, lw=1.45)
    ax.text(0.08, 0.485, r"integrated exposure  $x_{\tau}(t)$",
            fontsize=FS_LABEL, color=RUST, ha="left", va="top")

    # Identification: observation innovation versus physical forcing.
    ax.plot([0.055, 0.945], [0.425, 0.425], color=LIGHT, lw=0.65)
    ax.text(0.055, 0.385, "observer or physical forcing?",
            fontsize=FS_SUBTITLE, color=INK, fontweight="bold", va="top")

    ax.add_patch(Circle((0.16, 0.280), 0.036, facecolor="white",
                        edgecolor=BLUE, linewidth=1.1))
    ax.text(0.16, 0.280, r"$\eta_e$", fontsize=FS_LABEL, color=BLUE,
            ha="center", va="center")
    ax.text(0.16, 0.220, "unexpected part", fontsize=FS_SMALL, color=BLUE,
            ha="center", va="top")

    ax.add_patch(Circle((0.16, 0.125), 0.036, facecolor="white",
                        edgecolor=RUST, linewidth=1.1))
    ax.text(0.16, 0.125, r"$x_{\tau}$", fontsize=FS_LABEL, color=RUST,
            ha="center", va="center")
    ax.text(0.16, 0.065, "every IED", fontsize=FS_SMALL, color=RUST,
            ha="center", va="top")

    state_center = (0.51, 0.205)
    ax.add_patch(Circle(state_center, 0.064, facecolor="white",
                        edgecolor=TEAL, linewidth=1.35, zorder=4))
    ax.text(*state_center, "state\n$z(t)$", fontsize=FS_LABEL, color=TEAL,
            ha="center", va="center", linespacing=1.0, zorder=5)

    _arrow(ax, (0.20, 0.280), (0.44, 0.235), color=BLUE, lw=0.85,
           style=(0, (2.5, 2.0)), mutation=7)
    _arrow(ax, (0.20, 0.125), (0.44, 0.175), color=RUST, lw=1.35,
           mutation=8)
    ax.text(0.31, 0.295, "T1: observer", fontsize=FS_SMALL, color=BLUE,
            ha="center")
    ax.text(0.31, 0.090, "T2: physical", fontsize=FS_SMALL, color=RUST,
            ha="center")

    _arrow(ax, (0.58, 0.205), (0.72, 0.205), color=TEAL, lw=1.1,
           mutation=8)
    _event_raster(ax, 0.81, 0.275, [0.08, 0.38, 0.70, 0.94], scale=0.90)
    ax.plot([0.75, 0.90], [0.135, 0.135], color=INK, lw=0.65)
    ax.plot([0.78, 0.78], [0.135, 0.175], color=INK, lw=0.8)
    ax.plot([0.86, 0.86], [0.135, 0.195], color=INK, lw=0.8)
    ax.text(0.82, 0.335, "future timing + repertoire", fontsize=FS_SMALL,
            color=INK, ha="center", va="bottom")

def _save(fig, root: Path, run_id: str) -> dict[str, str]:
    out_root = root / "figures" / "revisions" / run_id / ASSET_ID
    figure_dir = out_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{ASSET_ID}.png"
    pdf = figure_dir / f"{ASSET_ID}.pdf"
    svg = figure_dir / f"{ASSET_ID}.svg"

    # Preserve the requested 180 mm physical width.  The shared Epi-PRSSM style
    # currently sets savefig.bbox='tight', which silently crops exact dimensions.
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0)
    fig.savefig(svg, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    metadata = {
        "asset_id": ASSET_ID,
        "paper_slot": "TBD",
        "status": "EXPLORATORY_SCHEMATIC",
        "run_id": run_id,
        "specification": "docs/superpowers/specs/2026-08-19-topic5-epi-prssm-v0_2.md",
        "figure_contract": "docs/superpowers/specs/2026-08-18-topic5-epi-prssm-figure-contract.md",
        "style_reference": "user-provided multi-panel RNN figure; visual grammar only",
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "schematic_not_data": True,
        "hypotheses": {
            "H1": "A persistent state spans IEDs and predicts future event timing and spatial repertoire after observer correction is disabled.",
            "H2": "The pre-event state changes the within-event continuation and may link the frozen interictal model to early-ictal recruitment.",
            "H3": "IED exposure integrated over a finite timescale physically updates the state beyond event innovation alone.",
        },
        "claim_boundaries": [
            "Every trajectory, raster and graph is illustrative and is not derived from patient data.",
            "H3 is an independent extension and is not a gate for H1 or H2.",
            "Network shaping means a change in functional event timing or repertoire, not anatomical rewiring.",
            "H2b is a state link, not proof that the state or IED exposure causes seizure onset.",
        ],
        "files": {"png": str(png), "pdf": str(pdf), "svg": str(svg)},
        "code_revision": code_revision(),
        "package_hash": package_hash(),
    }
    metadata_path = out_root / f"{ASSET_ID}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    readme = figure_dir / "README.md"
    readme.write_text(
        f"# {ASSET_ID}\n\n"
        f"### {ASSET_ID}.png\n\n"
        "这是一张假设示意图，不展示实验效应。A 表示关掉 observer correction 后，跨事件状态仍应同时预测下一次何时发生以及走哪条空间路线；B 把 H2a 的单事件分岔与 H2b 的间期到发作联系放在同一状态框架中；C 区分事件带来的观测新息与每次 IED 都会交付的物理暴露。\n\n"
        "**关注点**：H3 用 rust 色画成独立扩展，不是 H1/H2 的总闸门；图中的 network shaping 只表示功能性事件时刻/repertoire 改变，不表示解剖重连。\n"
    )
    return {
        "png": str(png), "pdf": str(pdf), "svg": str(svg),
        "metadata": str(metadata_path), "readme": str(readme),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results" / "epi_prssm" / "v0_2",
        help="Versioned result root; paper-ready Fig1--Fig4 are never touched.",
    )
    args = parser.parse_args()

    apply_style()
    rcParams.update({
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_LABEL,
        "ytick.labelsize": FS_LABEL,
        "legend.fontsize": FS_LABEL,
    })
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(2, 2, left=0.035, right=0.985, top=0.940,
                            bottom=0.055, wspace=0.14, hspace=0.24,
                            height_ratios=(0.70, 1.25))
    ax_h1 = fig.add_subplot(grid[0, :])
    ax_h2 = fig.add_subplot(grid[1, 0])
    ax_h3 = fig.add_subplot(grid[1, 1])
    draw_h1(ax_h1)
    draw_h2(ax_h2)
    draw_h3(ax_h3)

    # Light separators preserve the reference figure's open white layout.
    y = (ax_h1.get_position().y0 + ax_h2.get_position().y1) / 2
    fig.lines.append(plt.Line2D([0.035, 0.985], [y, y], transform=fig.transFigure,
                                color="#E2E2E2", lw=0.55, zorder=0))
    x = (ax_h2.get_position().x1 + ax_h3.get_position().x0) / 2
    fig.lines.append(plt.Line2D([x, x], [0.055, ax_h2.get_position().y1],
                                transform=fig.transFigure, color="#E2E2E2",
                                lw=0.55, zorder=0))
    fig.text(0.985, 0.988, "SCHEMATIC · NOT DATA", ha="right", va="top",
             fontsize=FS_SMALL, color="#8A8A8A")

    outputs = _save(fig, args.output_root, args.run_id)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
