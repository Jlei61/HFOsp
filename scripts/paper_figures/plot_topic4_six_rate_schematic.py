"""Draw the frozen six-population rate model, using its actual projected graph.

This is a structural schematic, not a new simulation or an accepted SNN reduction.
All 28 nonzero source-to-target population pathways are shown. Red/blue pathways
encode AMPA/GABA signs in the mean input; they do not encode the sign of the full
rate Jacobian, which also includes the recurrent input variance.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915"
OUT = ROOT / "results/topic4_sef_hfo/six_rate_model_structure_20260917"
FIGURES = OUT / "figures"
STEM = "six_rate_model_structure_colored"
E_COLOR = "#df252d"
I_COLOR = "#467fb7"
INK = "#363636"
RADIUS = 0.40

# Positions are explanatory coordinates, not physical cell locations.
CENTERS = {
    0: np.array([1.2, 2.65]), 3: np.array([1.2, 0.95]),
    2: np.array([4.6, 2.65]), 5: np.array([4.6, 0.95]),
    1: np.array([8.0, 2.65]), 4: np.array([8.0, 0.95]),
}
LABELS = {0: "r_E^A", 1: "r_E^B", 2: "r_E^S", 3: "r_I^A", 4: "r_I^B", 5: "r_I^S"}


def boundary(node: int, angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return CENTERS[node] + RADIUS * np.array([np.cos(a), np.sin(a)])


def connection(ax, points, *, color=INK, lw=2.05, both_ends=False, layer=2,
               inhibitory=False):
    """Receiving ends: E circles and I triangles, as requested by the user."""
    p = np.asarray(points, float)
    tangent = p[-1] - p[-2]
    tangent /= np.linalg.norm(tangent)
    vertices = p.copy()
    center = p[-1] - 0.080 * tangent
    vertices[-1] = center
    terminals = [(center, tangent)]
    if both_ends:
        initial = p[1] - p[0]
        initial /= np.linalg.norm(initial)
        vertices[0] = p[0] + 0.080 * initial
        terminals.append((vertices[0], -initial))
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    # Each new path's white under-stroke sits above earlier paths, creating a
    # small crossing gap. It does not erase node labels or terminal markers.
    ax.add_patch(PathPatch(path, fc="none", ec="white", lw=lw + 2.3,
                           capstyle="round", joinstyle="round", zorder=layer))
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw,
                           capstyle="round", joinstyle="round", zorder=layer + .1))
    for terminal, incoming in terminals:
        if inhibitory:
            normal = np.array([-incoming[1], incoming[0]])
            triangle = [terminal + .072 * incoming,
                        terminal - .048 * incoming + .069 * normal,
                        terminal - .048 * incoming - .069 * normal]
            ax.add_patch(Polygon(triangle, closed=True, fc=color, ec="white",
                                 lw=.45, zorder=80))
        else:
            ax.add_patch(Circle(terminal, 0.057, fc=color, ec="white", lw=.45, zorder=80))


def edge_points(source: int, target: int):
    s, t = CENTERS[source], CENTERS[target]
    if source == target:
        # Compact loops above E and below I.
        sign = 1 if source < 3 else -1
        return [boundary(source, sign * 45),
                s + [0.69, sign * 1.00],
                s + [-0.69, sign * 1.00],
                boundary(source, sign * 135)]
    if s[0] == t[0]:
        if source < 3:
            return [boundary(source, -67), s + [0.76, -0.56],
                    t + [0.76, 0.56], boundary(target, 67)]
        return [boundary(source, 113), s + [-0.76, 0.56],
                t + [-0.76, -0.56], boundary(target, -113)]
    direction = 1 if t[0] > s[0] else -1
    if s[1] == t[1]:
        # Same-type reciprocal pathways share a straight line with two terminals.
        return [boundary(source, 0 if direction > 0 else 180),
                s + [direction * 1.0, 0], t + [-direction * 1.0, 0],
                boundary(target, 180 if direction > 0 else 0)]
    if source < 3:
        source_angle, target_angle = (-25, 155) if direction > 0 else (225, 45)
    else:
        source_angle, target_angle = (25, 205) if direction > 0 else (135, -45)
    # Separated ports and mild curvature keep reciprocal E/I links distinguishable.
    displacement = t - s
    normal = np.array([-displacement[1], displacement[0]])
    normal /= np.linalg.norm(normal)
    c1 = s + 0.34 * displacement + 0.22 * normal
    c2 = s + 0.66 * displacement + 0.22 * normal
    return [boundary(source, source_angle), c1, c2, boundary(target, target_angle)]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    spec = json.loads((SOURCE / "model_spec.json").read_text())
    with np.load(SOURCE / "projected_graph.npz") as data:
        W = data["W"].sum(axis=0)
        Q = data["Q"].sum(axis=0)
        names = data["names"].tolist()
    edges = [(int(source), int(target)) for target, source in np.argwhere(W > 0)]
    assert len(edges) == 28
    assert not any((s in (0, 3) and t in (1, 4)) or
                   (s in (1, 4) and t in (0, 3)) for s, t in edges)
    assert np.array_equal(W > 0, Q > 0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11.4, 5.6), facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set(xlim=(0.15, 9.05), ylim=(-0.54, 4.16), aspect="equal")
    ax.axis("off")

    remote = [(s, t) for s, t in edges if CENTERS[s][0] != CENTERS[t][0]]
    handled = set()
    stroke_groups = []
    # Blue first, red second; local pathways on top. Gaps clarify crossings.
    ordering = sorted(edges, key=lambda st: (st not in remote, st[0] < 3, st))
    for source, target in ordering:
        if (source, target) in handled:
            continue
        pair = (source != target and CENTERS[source][1] == CENTERS[target][1]
                and (target, source) in edges)
        represented = [(source, target)]
        if pair:
            represented.append((target, source))
        handled.update(represented)
        stroke_groups.append(represented)
        connection(ax, edge_points(source, target), color=E_COLOR if source < 3 else I_COLOR,
                   both_ends=pair, layer=2 + len(stroke_groups) * 2,
                   inhibitory=source >= 3)
    assert handled == set(edges)

    for e, i, title in [(0, 3, "Core A"), (2, 5, "Surround"), (1, 4, "Core B")]:
        x = CENTERS[e][0]
        ax.text(x, 4.00, title, fontsize=17, ha="center", va="center", color=INK)
        for node in (e, i):
            center = CENTERS[node]
            color = E_COLOR if node < 3 else I_COLOR
            ax.add_patch(Circle(center, RADIUS, fc=color, ec="none", zorder=70))
            ax.text(*center, "$" + LABELS[node] + "$", color="white",
                    fontsize=22, ha="center", va="center", zorder=85)
        if e in (0, 1):
            ax.text(x, 3.57, r"$J_{EE,\mathrm{core}}$", color=INK,
                    fontsize=13, ha="center", va="center")

    legend_y = -0.33
    connection(ax, [[2.1, legend_y], [2.3, legend_y], [2.5, legend_y], [2.7, legend_y]],
               color=E_COLOR, lw=2.05)
    ax.text(2.84, legend_y, "E / AMPA", fontsize=12, va="center", color=INK)
    connection(ax, [[5.25, legend_y], [5.45, legend_y], [5.65, legend_y], [5.85, legend_y]],
               color=I_COLOR, lw=2.05, inhibitory=True)
    ax.text(5.99, legend_y, "I / GABA", fontsize=12, va="center", color=INK)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{STEM}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)

    with (OUT / "population_connections.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "transmitter",
                                               "W_sum_base", "Q_sum_base", "core_multiplier"])
        writer.writeheader()
        for source, target in edges:
            writer.writerow(dict(source=names[source], target=names[target],
                                 transmitter="AMPA" if source < 3 else "GABA",
                                 W_sum_base=float(W[target, source]),
                                 Q_sum_base=float(Q[target, source]),
                                 core_multiplier=source == target and source in (0, 1)))
    metadata = {
        "figure": STEM,
        "producer": str(Path(__file__).relative_to(ROOT)),
        "source_model": str(SOURCE.relative_to(ROOT)),
        "model_layer": spec["model_layer"],
        "display_order": ["Core A", "Surround", "Core B"],
        "populations": names,
        "drawn_population_pathways": len(edges),
        "drawn_strokes": len(stroke_groups),
        "stroke_groups_source_target_indices": stroke_groups,
        "self_pathways": sum(s == t for s, t in edges),
        "inter_region_pathways": len(remote),
        "direct_core_A_B_pathways": 0,
        "external_drive": spec["external_drive"],
        "shared_parameter": "Displayed J_EE,core is the dimensionless multiplier g in model.py: W_AE,AE and W_BE,BE scale by g; corresponding Q scales by g^2.",
        "omitted_from_drawing": ["External input arrows and repeated moment labels, for visual simplicity only", "Two synaptic filters per population", "368 delay bins", "Recurrent variance closure", "Empirical threshold distributions"],
        "adaptation_a_or_Z_M": "Absent in this source model",
        "edge_convention": "Source E: red AMPA with a circular receiving terminal; source I: blue GABA with a triangular receiving terminal, per user instruction. Triangles point toward the receiving population. Same-type reciprocal pathways share one line with the corresponding marker at each end, without implying equal weights. Crossing gaps indicate no junction; line width does not encode weights, delays or variance coefficients.",
        "synaptic_terminal_counts": {"E_circles": sum(s < 3 for s, t in edges), "I_triangles": sum(s >= 3 for s, t in edges)},
        "node_convention": "Solid red/blue circles contain the six population rates. Self-loops are within-population recurrence, not single-cell autapses.",
        "scope": "Historical six-population model structure; no claim of native-SNN correspondence or clinical mechanism.",
        "human_visual_acceptance": "PENDING",
    }
    (OUT / "figure_metadata_colored.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (FIGURES / "README.md").write_text(
        "### six_rate_model_structure.png / .pdf / .svg\n\n"
        "首版保留空心节点和全部外源输入矩标注，连接拓扑来自原六群体模型。"
        "三角箭头表示AMPA、圆点端表示GABA；该版保留用于与彩色修订比较。"
        "**关注点**：实际28条群体通路及外源输入均值、方差的对应。\n\n"
        "### six_rate_model_structure_colored.png / .pdf / .svg\n\n"
        "修订版采用实心红/蓝节点及白色群体率标签，所有连接按发送端类型着色：E为红色AMPA，I为蓝色GABA，"
        "兴奋性接收端用红色圆圈，抑制性接收端用指向接收群体的蓝色三角形。全部28条有向通路保留，以24条曲线表示：同类群体间的双向通路合为两端均带相应突触符号的单线，"
        "不表示两个方向权重相等；两核无直接连接，分别与外围耦合，J_EE,core仍为两核共同的无量纲E→E倍率。"
        "为简洁省略外源输入箭头和重复标注，实际输入、突触滤波、延迟及方差闭合均未修改；原模型不含a/Z/M。"
        "**关注点**：中间连线的颜色、接收端方向与交叉处断口；该图已由Agent自查，待用户目视确认，不代表原生SNN对应关系通过。\n"
    )
    print(json.dumps({"output": str(FIGURES), "pathways": len(edges),
                      "populations": len(names), "formats": ["png", "pdf", "svg"]}))


if __name__ == "__main__":
    main()
