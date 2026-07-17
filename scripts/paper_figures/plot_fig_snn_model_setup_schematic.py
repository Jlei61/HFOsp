"""Draw a standalone setup schematic for the Topic 4 SNN model.

This is a plotting-only conceptual panel. It does not consume simulation
artifacts and does not rerun the SNN. The panel is meant to serve the same
role as a model-cortex setup cartoon: E/I populations, the anisotropic E->E
scaffold, dual low-threshold E cores, virtual SEEG readout, and optional slow
fields are shown in one self-contained schematic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
FIG_NAME = "fig_snn_model_setup_schematic"
OUTDIR = ROOT / f"results/paper-ready-figure/{FIG_NAME}/figures"

RED = "#d7191c"
BLUE = "#1f4fd8"
ORANGE = "#d9851f"
CYAN = "#1f9e9e"
GREEN = "#4c8c2b"
GREY = "#505050"
LIGHT_GREY = "#f2f4f7"
E_SHADE = "#f4b266"


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _triangle(center: tuple[float, float], width: float, height: float) -> np.ndarray:
    x, y = center
    return np.array(
        [
            [x - width / 2.0, y - height / 2.0],
            [x + width / 2.0, y - height / 2.0],
            [x, y + height / 2.0],
        ]
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.4,
    alpha: float = 1.0,
    style: str = "-|>",
    rad: float = 0.0,
    ls: str = "-",
    zorder: int = 5,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)


def _text(ax, x: float, y: float, s: str, **kwargs) -> None:
    defaults = dict(fontsize=9, color="0.15", ha="center", va="center")
    defaults.update(kwargs)
    txt = ax.text(x, y, s, **defaults)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def _draw_population(ax) -> None:
    core_x = np.array([-1.85, 1.85])

    # Soft cortical sheet.
    ax.add_patch(
        Ellipse(
            (0.0, 0.18),
            width=8.75,
            height=2.22,
            facecolor=LIGHT_GREY,
            edgecolor="none",
            alpha=0.96,
            zorder=0,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.0, 0.18),
            width=8.05,
            height=1.55,
            facecolor="#dfefff",
            edgecolor="none",
            alpha=0.42,
            zorder=0,
        )
    )

    # Anisotropic E->E footprint corridor. The arrow is double-headed because
    # this is a recurrent scaffold orientation, not an imposed direction.
    ax.add_patch(
        Ellipse(
            (0.0, 0.36),
            width=5.80,
            height=0.72,
            facecolor=E_SHADE,
            edgecolor=ORANGE,
            linewidth=1.25,
            alpha=0.28,
            zorder=1,
        )
    )
    _arrow(ax, (-2.95, 0.36), (2.95, 0.36), color=ORANGE, lw=1.9, style="<|-|>", zorder=4)
    _text(ax, 0.0, 0.77, "anisotropic E->E scaffold (AR>1)", fontsize=9.2, color=ORANGE)

    # Low-threshold cores.
    for label, x0 in zip(("core A", "core B"), core_x):
        ax.add_patch(Circle((x0, 0.52), 0.50, facecolor="none", edgecolor=RED, lw=1.8, ls="--", zorder=6))
        ax.add_patch(Circle((x0, 0.52), 0.50, facecolor=RED, edgecolor="none", alpha=0.08, zorder=2))
        _text(ax, x0, 1.08, label, fontsize=8.7, color=RED, fontweight="bold")

    # E row: triangles, filled only inside the low-threshold cores.
    xs = np.linspace(-3.7, 3.7, 13)
    for x in xs:
        in_core = np.any(np.abs(x - core_x) < 0.42)
        tri = Polygon(
            _triangle((x, 0.50), 0.40, 0.46),
            closed=True,
            facecolor=RED if in_core else "white",
            edgecolor=RED,
            linewidth=1.9,
            alpha=1.0,
            zorder=5,
        )
        ax.add_patch(tri)

    # I row: dotted circles.
    for x in xs:
        ax.add_patch(
            Circle(
                (x, -0.33),
                0.20,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=1.8,
                linestyle=(0, (1.2, 1.2)),
                zorder=5,
            )
        )

    # Local E/I motifs and stochastic ignition.
    for x0 in core_x:
        _arrow(ax, (x0 - 0.35, -0.12), (x0 - 0.35, 0.25), color=BLUE, lw=1.2, zorder=6)
        _arrow(ax, (x0 + 0.35, -0.12), (x0 + 0.35, 0.25), color=BLUE, lw=1.2, zorder=6)
        _arrow(ax, (x0 - 0.30, 1.54), (x0 - 0.04, 0.86), color=GREY, lw=1.0, alpha=0.65, zorder=5)
        _arrow(ax, (x0 + 0.30, 1.54), (x0 + 0.04, 0.86), color=GREY, lw=1.0, alpha=0.65, zorder=5)

    _text(ax, -4.35, 0.50, "E cells", fontsize=10.5, color=RED, ha="right", fontweight="bold")
    _text(ax, -4.35, -0.33, "I cells", fontsize=10.5, color=BLUE, ha="right", fontweight="bold")
    _text(ax, -4.18, 0.05, "model\ncortex", fontsize=10, ha="right", color="0.20")
    _text(ax, 0.0, 1.56, "OU/Poisson background can nucleate either end", fontsize=8.6, color=GREY)


def _draw_top_profile(ax) -> None:
    x = np.linspace(-4.0, 4.0, 500)
    core_x = np.array([-1.85, 1.85])
    y = 1.82 + 0.50 * (_gauss(x, core_x[0], 0.48) + _gauss(x, core_x[1], 0.48))
    ax.plot(x, y, color=RED, lw=2.3, zorder=7)
    ax.plot([-4.0, 4.0], [1.82, 1.82], color=RED, lw=1.0, alpha=0.25, zorder=6)
    for x0 in core_x:
        ax.plot([x0, x0], [0.10, 2.43], color="0.20", lw=1.0, ls="--", zorder=3)
    _text(ax, 0.0, 2.47, "E-core excitability: 18 - Vth", fontsize=10.2, color=RED, fontweight="bold")
    _arrow(ax, (-2.33, 2.21), (-1.37, 2.21), color=RED, lw=1.1, style="<|-|>", zorder=8)
    _text(ax, -1.85, 2.33, "core radius", fontsize=8.2, color=RED)


def _draw_readout(ax) -> None:
    # Same virtual montage can read both directions. Two colored shafts mirror
    # the Topic 4 figure style: along-axis orange, cross-axis cyan.
    contact_x = np.linspace(-3.1, 3.1, 8)
    y0 = -1.05
    ax.plot(contact_x, np.full_like(contact_x, y0), color=ORANGE, lw=1.5, alpha=0.85, zorder=4)
    for i, x in enumerate(contact_x):
        ax.add_patch(Circle((x, y0), 0.105, facecolor="white", edgecolor=ORANGE, linewidth=1.5, zorder=8))
        if i in (0, len(contact_x) - 1):
            _text(ax, x, y0 + 0.27, f"A{i + 1}", fontsize=7.5, color=ORANGE)

    cross_y = np.linspace(-1.38, -0.70, 4)
    cross_x = np.full_like(cross_y, 0.0)
    ax.plot(cross_x, cross_y, color=CYAN, lw=1.5, alpha=0.85, zorder=4)
    for j, y in enumerate(cross_y):
        ax.add_patch(Circle((0.0, y), 0.095, facecolor="white", edgecolor=CYAN, linewidth=1.35, zorder=8))
        if j == len(cross_y) - 1:
            _text(ax, 0.34, y, "B shaft", fontsize=7.5, color=CYAN, ha="left")

    # Gaussian contact readout kernels.
    for x in (-2.20, 0.0, 2.20):
        ax.add_patch(Ellipse((x, -0.64), 1.0, 0.33, facecolor="0.60", edgecolor="none", alpha=0.14, zorder=2))
        _arrow(ax, (x, -0.90), (x, -0.45), color="0.45", lw=0.8, alpha=0.55, zorder=3)

    _text(ax, 2.75, -1.34, "virtual SEEG readout\npeak order -> forward / reverse", fontsize=8.7, color="0.15")


def _draw_slow_profiles(ax) -> None:
    x = np.linspace(-4.0, 4.0, 500)
    active = -1.85
    q_base = -1.94
    q = q_base + 0.34 - 0.34 * _gauss(x, active, 0.85)
    gk = -2.31 + 0.30 * _gauss(x, active, 0.38)
    ax.plot(x, q, color=BLUE, lw=2.0, ls="--", zorder=7)
    ax.plot(x, gk, color=GREEN, lw=2.0, ls="-.", zorder=7)
    _text(ax, -3.95, -2.16, "M3A optional\nslow fields", fontsize=8.2, color="0.25", ha="left")
    _text(ax, -0.95, -1.67, "q_I depletion: broad disinhibition", fontsize=8.0, color=BLUE, ha="left")
    _text(ax, -0.95, -2.22, "g_K recovery: local brake", fontsize=8.0, color=GREEN, ha="left")
    _text(
        ax,
        1.85,
        -2.52,
        "Fig4/5 baseline: slow=None; slow variables are a separate mechanism screen",
        fontsize=7.7,
        color="0.35",
        ha="center",
        style="italic",
    )


def _write_readme() -> None:
    text = """# SNN model setup schematic

### snn_model_setup_schematic.png / .pdf / .svg

这张图是 Topic 4 SNN 部分的设置示意图，不是新的仿真结果。它把模型皮层画成 E/I LIF population：红色三角是 E 细胞，蓝色圆圈是 I 细胞；两端红色虚线圈表示低阈值 E core；中间橙色带表示各向异性的 E→E 长轴连接 scaffold。底部触点表示同一虚拟 SEEG montage 对传播事件做 peak-order readout。

图下方的 `q_I` / `g_K` 曲线只是 M3A 慢变量扩展层的可选示意；当前 Fig4/5 baseline 仍是 `slow=None`。因此这张图只能用来解释模型设置和读出合同，不能单独写成真实病人机制已被证明。

**关注点**：先看低阈值 core 是否位于同一 E→E 长轴两端，再看同一 readout 是否被定义为方向读出层，最后确认慢变量没有被误画成 baseline 必备机制。
"""
    (OUTDIR / "README.md").write_text(text)


def _write_metadata(paths: dict[str, Path]) -> None:
    meta = {
        "figure": FIG_NAME,
        "type": "conceptual_setup_schematic",
        "simulation_rerun": False,
        "output_paths": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "model_contract": {
            "core_engine": "current-based E/I LIF network with delayed AMPA/GABA currents and OU/Poisson background drive",
            "spatial_scaffold": "anisotropic recurrent E->E footprint along a long axis; recurrent orientation, not an imposed propagation direction",
            "pathology": "dual low-threshold E cores at opposite ends of the axis",
            "readout": "virtual SEEG contact envelopes converted to peak/onset order for forward/reverse event labels",
            "baseline_slow_state": "slow=None for Fig4/Fig5 baseline",
            "optional_slow_state": "M3A q_I depletion and g_K recovery are shown only as an extension layer",
        },
        "claim_boundary": [
            "This schematic explains model setup only.",
            "It is not evidence that real patient propagation mechanisms are proven.",
            "The q_I/g_K layer must not be read as active in the Fig4/Fig5 baseline unless a specific slow-variable run is being discussed.",
        ],
    }
    (OUTDIR / "snn_model_setup_schematic_metadata.json").write_text(json.dumps(meta, indent=2))


def compose() -> dict[str, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 6.35), facecolor="white")
    ax.set_xlim(-4.75, 4.75)
    ax.set_ylim(-2.66, 2.65)
    ax.axis("off")

    _draw_population(ax)
    _draw_top_profile(ax)
    _draw_readout(ax)
    _draw_slow_profiles(ax)

    ax.text(-4.62, 2.48, "A", fontsize=19, fontweight="bold", ha="left", va="center", color="0.05")
    _text(ax, -0.05, 2.63, "SNN model setup", fontsize=13.5, fontweight="bold", color="0.05")

    png = OUTDIR / "snn_model_setup_schematic.png"
    pdf = OUTDIR / "snn_model_setup_schematic.pdf"
    svg = OUTDIR / "snn_model_setup_schematic.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paths = {"png": png, "pdf": pdf, "svg": svg}
    _write_readme()
    _write_metadata(paths)
    return paths


def main() -> int:
    os.chdir(ROOT)
    paths = compose()
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
