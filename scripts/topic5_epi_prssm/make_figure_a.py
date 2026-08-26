#!/usr/bin/env python3
"""Figure A -- data object, three state objects and the unified model ladder.

asset_id: epi_prssm_architecture_ladder
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.topic5_epi_prssm.contracts import FROZEN, OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, FS_AXIS, FS_TICK, FS_TITLE, LW_MAIN, LW_REFERENCE,
    figure, panel_letter, save_asset,
)

ASSET = "epi_prssm_architecture_ladder"
FIG_ROOT = OUTPUT_ROOT / "figures"


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def panel_a(ax):
    _blank(ax)
    ax.set_title("Three state objects", fontsize=FS_TITLE, loc="left", pad=5)
    bands = [(0.68, 0.30, "#EFEFEF", "event-internal", r"fast  $s_{e,k}$"),
             (0.36, 0.30, "#E4F1F1", "physical", r"slow  $z_e=[H_e,r_e]$"),
             (0.04, 0.30, "#F2EFE6", "inference memory", r"observer  $c_e$")]
    for y, h, colour, kind, label in bands:
        ax.add_patch(Rectangle((0.0, y), 1.0, h, facecolor=colour, edgecolor="none", zorder=0))
        ax.text(0.015, y + h - 0.04, label, fontsize=FS_TICK, va="top", color="#222222")
        ax.text(0.015, y + 0.03, kind, fontsize=FS_TICK - 1.0, va="bottom",
                color="#666666", style="italic")
    events = [0.16, 0.34, 0.55, 0.79]
    for x in events:
        ax.plot([x, x], [0.02, 0.98], color="#C7C7C7", lw=0.5, zorder=1)
    # fast event state: squares inside one event
    for i, x in enumerate(events):
        for k in range(3):
            ax.add_patch(Rectangle((x - 0.018 + 0.016 * k, 0.86 - 0.045 * k), 0.012, 0.028,
                                   facecolor=COLOR["G2"], edgecolor="none", zorder=3))
    # slow generative state: continuous line with circles at event times
    t = np.linspace(0.02, 0.98, 300)
    curve = 0.50 + 0.06 * np.sin(2 * np.pi * t * 1.1) * np.exp(-0.4 * t)
    ax.plot(t, curve, color=COLOR["G3"], lw=LW_MAIN, zorder=3)
    for x in events:
        ax.add_patch(Circle((x, float(np.interp(x, t, curve))), 0.013,
                            facecolor=COLOR["G3"], edgecolor="white", lw=0.5, zorder=4))
    # observer state: stepped triangles, only updated at events
    level = 0.12
    for i, x in enumerate(events):
        ax.plot([x, events[i + 1] if i + 1 < len(events) else 0.98], [level, level],
                color=COLOR["observer"], lw=LW_MAIN, zorder=3)
        ax.plot([x], [level], marker="^", ms=4.5, color=COLOR["observer"], zorder=4)
        level += 0.035
    ax.text(0.5, -0.045, "time", fontsize=FS_TICK, ha="center", color="#555555")


def panel_b(ax):
    _blank(ax)
    ax.set_title("Scaffold and fixed repertoire", fontsize=FS_TITLE, loc="left", pad=5)
    rng = np.random.default_rng(3)
    positions = np.column_stack([np.linspace(0.08, 0.40, 7) + rng.normal(0, 0.010, 7),
                                 0.52 + rng.normal(0, 0.06, 7)])
    for i in range(7):
        for j in range(i + 1, 7):
            if abs(i - j) <= 2:
                ax.plot(*zip(positions[i], positions[j]), color="#C9C9C9", lw=0.6, zorder=1)
    for i, (x, y) in enumerate(positions):
        ax.add_patch(Circle((x, y), 0.021, facecolor=COLOR["scaffold"], edgecolor="white",
                            lw=0.6, zorder=3))
    ax.text(0.24, 0.86, r"graph $\mathcal{G}_p$", fontsize=FS_TICK,
            ha="center", color=COLOR["scaffold"])
    heights = np.array([0.9, 0.62, 0.75, 0.41, 0.55, 0.30, 0.68])
    x0 = np.linspace(0.60, 0.94, 7)
    ax.bar(x0, heights * 0.20, width=0.030, bottom=0.48, color=COLOR["scaffold"], zorder=3)
    residual = np.array([0.05, -0.03, 0.02, 0.06, -0.05, 0.03, -0.02])
    ax.bar(x0, residual * 1.3, width=0.030, bottom=0.44, color=COLOR["G2"], zorder=3)
    ax.text(0.77, 0.86, r"$\mu_p$  train-only", fontsize=FS_TICK,
            ha="center", color=COLOR["scaffold"])
    ax.text(0.77, 0.30, r"$D_\psi(z_e^-)$  residual", fontsize=FS_TICK,
            ha="center", color=COLOR["G2"])


def panel_c(ax):
    _blank(ax)
    ax.set_title("G0-G3 generator ladder", fontsize=FS_TITLE, loc="left", pad=5)
    columns = [("G0", COLOR["G0"], "leak\nonly", False),
               ("G1", COLOR["G1"], "linear\nmessages", True),
               ("G2", COLOR["G2"], "gated\nmessages", True),
               ("G3", COLOR["G3"], "+ resource\nmodulation", True)]
    for c, (name, colour, caption, arrows) in enumerate(columns):
        x0 = 0.045 + c * 0.245
        ax.add_patch(FancyBboxPatch((x0, 0.30), 0.20, 0.52, boxstyle="round,pad=0.012",
                                    facecolor="white", edgecolor=colour, lw=1.0, zorder=2))
        nodes = np.array([[x0 + 0.05, 0.68], [x0 + 0.15, 0.68],
                          [x0 + 0.05, 0.46], [x0 + 0.15, 0.46]])
        if arrows:
            for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
                ax.add_patch(FancyArrowPatch(nodes[a], nodes[b], arrowstyle="-|>",
                                             mutation_scale=6, color=colour, lw=0.8,
                                             shrinkA=6, shrinkB=6, zorder=3))
        else:
            for node in nodes:
                ax.add_patch(FancyArrowPatch(node + [0.0, 0.045], node + [0.0, 0.012],
                                             arrowstyle="-|>", mutation_scale=6,
                                             color=colour, lw=0.8, zorder=3))
        for node in nodes:
            ax.add_patch(Circle(node, 0.017, facecolor=colour, edgecolor="white", lw=0.5, zorder=4))
        ax.text(x0 + 0.10, 0.87, name, fontsize=FS_AXIS, ha="center", color=colour,
                fontweight="bold")
        ax.text(x0 + 0.10, 0.25, caption, fontsize=FS_TICK - 0.8, ha="center", va="top",
                color="#444444")
        if name == "G3":
            ax.add_patch(Circle((x0 + 0.10, 0.565), 0.030, facecolor="white",
                                edgecolor=COLOR["G3"], lw=1.0, zorder=5))
            ax.text(x0 + 0.10, 0.565, r"$r$", fontsize=FS_AXIS, ha="center", va="center",
                    color=COLOR["G3"], zorder=6)
    ax.text(0.045, 0.05, "no node-to-node arrow in G0", fontsize=FS_TICK - 0.6,
            color="#444444")


def panel_d(ax):
    _blank(ax)
    ax.set_title("Transition versus correction", fontsize=FS_TITLE, loc="left", pad=5)
    ax.add_patch(FancyBboxPatch((0.06, 0.55), 0.24, 0.26, boxstyle="round,pad=0.012",
                                facecolor="white", edgecolor=COLOR["G3"], lw=1.0))
    ax.text(0.18, 0.68, r"$z_{e-1}^{+}$", fontsize=FS_AXIS, ha="center", color=COLOR["G3"])
    ax.add_patch(FancyBboxPatch((0.42, 0.55), 0.24, 0.26, boxstyle="round,pad=0.012",
                                facecolor="white", edgecolor=COLOR["G3"], lw=1.0))
    ax.text(0.54, 0.68, r"$z_{e}^{-}$", fontsize=FS_AXIS, ha="center", color=COLOR["G3"])
    ax.add_patch(FancyBboxPatch((0.76, 0.55), 0.20, 0.26, boxstyle="round,pad=0.012",
                                facecolor="white", edgecolor=COLOR["G3"], lw=1.0))
    ax.text(0.86, 0.68, r"$\widehat{H}_{e}^{+}$", fontsize=FS_AXIS, ha="center", color=COLOR["G3"])
    ax.add_patch(FancyArrowPatch((0.31, 0.68), (0.41, 0.68), arrowstyle="-|>",
                                 mutation_scale=8, color=COLOR["G3"], lw=LW_MAIN))
    ax.text(0.36, 0.75, r"$\Delta t$", fontsize=FS_TICK, ha="center", color=COLOR["G3"])
    ax.add_patch(FancyArrowPatch((0.67, 0.68), (0.755, 0.68), arrowstyle="-|>",
                                 mutation_scale=8, color=COLOR["observer"], lw=0.9,
                                 linestyle=(0, (2.5, 1.6))))
    ax.add_patch(FancyBboxPatch((0.42, 0.16), 0.24, 0.20, boxstyle="round,pad=0.012",
                                facecolor="#F2EFE6", edgecolor=COLOR["observer"], lw=0.9))
    ax.text(0.54, 0.26, r"observer $c_e$", fontsize=FS_AXIS, ha="center", color="#5A5A5A")
    ax.add_patch(FancyArrowPatch((0.62, 0.36), (0.80, 0.545), arrowstyle="-|>",
                                 mutation_scale=8, color=COLOR["observer"], lw=0.9,
                                 linestyle=(0, (2.5, 1.6))))
    ax.plot([0.60, 0.68], [0.46, 0.46], color=COLOR["exposure"], lw=0.9)
    ax.plot([0.64, 0.64], [0.42, 0.50], color=COLOR["exposure"], lw=0.9)
    ax.text(0.70, 0.455, r"no write into $r$", fontsize=FS_TICK - 0.6, va="center",
            color=COLOR["exposure"])
    ax.text(0.06, 0.05, "solid: transition    dashed: correction", fontsize=FS_TICK - 0.6,
            color="#444444")


def panel_e(ax):
    _blank(ax)
    ax.set_title("State-conditioned readout", fontsize=FS_TITLE, loc="left", pad=5)
    ax.axvline(0.50, color="#8A8A8A", lw=0.8, ls=(0, (4, 2)))
    ax.text(0.505, 0.95, "event boundary", fontsize=FS_TICK - 0.6, va="top", color="#444444")
    for k in range(3):
        ax.add_patch(Rectangle((0.10 + 0.09 * k, 0.55), 0.06, 0.16,
                               facecolor=COLOR["G2"], edgecolor="none"))
    ax.text(0.235, 0.78, r"prefix $c_{e,1:k}$", fontsize=FS_AXIS, ha="center", color=COLOR["G2"])
    ax.add_patch(Circle((0.235, 0.34), 0.045, facecolor="white", edgecolor=COLOR["G3"], lw=1.0))
    ax.text(0.235, 0.34, r"$z_e^-$", fontsize=FS_AXIS, ha="center", va="center", color=COLOR["G3"])
    ax.add_patch(FancyArrowPatch((0.235, 0.39), (0.235, 0.53), arrowstyle="-|>",
                                 mutation_scale=7, color=COLOR["G3"], lw=0.9))
    ax.add_patch(FancyArrowPatch((0.34, 0.63), (0.47, 0.63), arrowstyle="-|>",
                                 mutation_scale=8, color=COLOR["scaffold"], lw=LW_MAIN))
    for k in range(2):
        ax.add_patch(Rectangle((0.57 + 0.09 * k, 0.55), 0.06, 0.16,
                               facecolor=COLOR["G1"], edgecolor="none"))
    ax.add_patch(Rectangle((0.75, 0.55), 0.06, 0.16, facecolor="white",
                           edgecolor=COLOR["scaffold"], lw=0.9))
    ax.text(0.78, 0.63, "STOP", fontsize=FS_TICK - 0.6, ha="center", va="center",
            color=COLOR["scaffold"])
    ax.text(0.68, 0.78, "suffix and STOP", fontsize=FS_AXIS, ha="center", color=COLOR["G1"])
    ax.text(0.03, 0.10, r"$\log p(\mathcal{E}_{e}) = \mu_p + D_\psi(\mathrm{prefix},\,"
                        r"\mathcal{G}_p,\, z_{e}^-)$", fontsize=FS_TICK, color="#222222")


def panel_f(ax):
    _blank(ax)
    ax.set_title("Four independent questions", fontsize=FS_TITLE, loc="left", pad=5)
    boxes = [("H1", "slow state\nexists?", 0.010, 0.60, COLOR["G2"]),
             ("H2a", "changes one\nevent?", 0.365, 0.60, COLOR["G1"]),
             ("H2b", "moves before\na seizure?", 0.720, 0.60, COLOR["onset"]),
             ("H3", "IED exposure\nupdates it?", 0.365, 0.14, COLOR["exposure"])]
    for name, body, x, y, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.255, 0.30, boxstyle="round,pad=0.010",
                                    facecolor="white", edgecolor=colour, lw=1.0))
        ax.text(x + 0.018, y + 0.245, name, fontsize=FS_TICK + 0.5, fontweight="bold",
                color=colour)
        ax.text(x + 0.018, y + 0.175, body, fontsize=FS_TICK - 0.6, va="top", color="#333333")
    ax.add_patch(FancyArrowPatch((0.278, 0.75), (0.352, 0.75), arrowstyle="-|>",
                                 mutation_scale=7, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((0.633, 0.75), (0.707, 0.75), arrowstyle="-|>",
                                 mutation_scale=7, color="#666666", lw=0.9))
    ax.add_patch(FancyArrowPatch((0.49, 0.58), (0.49, 0.46), arrowstyle="-|>",
                                 mutation_scale=7, color=COLOR["exposure"], lw=0.9,
                                 linestyle=(0, (2.5, 1.6))))
    ax.text(0.02, 0.05, "side branch, not a gate", fontsize=FS_TICK - 0.6,
            color=COLOR["exposure"])


def main() -> None:
    fig, axes = figure(DOUBLE_COLUMN_MM, 118.0, nrows=2, ncols=3)
    fig.subplots_adjust(left=0.030, right=0.988, top=0.915, bottom=0.035, wspace=0.13, hspace=0.28)
    for ax, letter, painter in zip(axes.ravel(), "ABCDEF",
                                   [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]):
        painter(ax)
        panel_letter(ax, letter, dx=-0.015, dy=1.17)
    files = save_asset(fig, ASSET, FIG_ROOT, metadata={
        "asset_id": ASSET, "provisional_role": "Figure A",
        "kind": "schematic; carries no cohort statistic",
        "state_objects": ["fast event state s_{e,k}", "slow generative state z_e=[H_e, r_e]",
                          "observer state c_e"],
        "generator_ladder": ["G0 leaky baseline", "G1 stable graph-CLDS",
                             "G2 bounded graph-GRU-ODE", "G3 + autonomous resource"],
        "resource_ladder": ["R0 none", "R1 autonomous", "R2 single-event depletion",
                            "R3 integrated exposure"],
        "colour_mapping": COLOR, "frozen_constants": {k: FROZEN[k] for k in
                                                      ("state_dim_H", "observer_dim",
                                                       "session_join_seconds",
                                                       "split_fractions")},
        "claim_boundary": [
            "G0 is drawn without node-to-node messages and is never called a graph RNN",
            "the resource is a bounded model variable, not a measured metabolic quantity",
            "TA/TB never enter the observer",
            "a positive H3 is not a precondition for the rest of the model",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "这张图是方法示意，不含任何队列统计。A 把三种状态画在同一条时间轴上："
                "事件内部的快状态（方块）、跨事件自主演化的慢生成状态（连续曲线）、"
                "只在观测到事件时才更新的推断记忆（阶梯三角）。B 说明患者固定 repertoire "
                "与围绕它的动态残差是两回事。C 是四级生成器阶梯，G0 故意不画节点间箭头。"
                "D 把物理演化（实线）与观测校正（虚线）分开，并标出主 observer 不逐事件写资源。"
                "E 是读出接口与事件边界。F 说明四个问题各自独立，H3 只是侧支。",
        "focus": "确认 G0 那一列没有节点间箭头，且 D 中虚线只进入图状态估计、没有指向资源。",
    }])
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
