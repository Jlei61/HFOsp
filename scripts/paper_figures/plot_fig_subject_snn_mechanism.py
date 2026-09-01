"""Standalone high-resolution mechanism panel for the E1146 subject-SNN figure.

This is a plotting-only companion to ``plot_fig_subject_snn.py``. It consumes the
same subject-specific figdata and redraws only the left "mechanism" panel at a
PPT-friendly resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

from scripts.paper_figures.plot_fig_subject_snn import (
    HOMOGENEOUS_CORE,
    TA_COLOR,
    TB_COLOR,
    _display_radius,
    _display_xy,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"

FIG_NAME = "fig_subject_snn_epilepsiae_1146_mechanism"
DEFAULT_TAG = "epilepsiae_1146_twoend_equal_tsrc_s3"

FWD_SHADE = "#f4b266"
AXIS_COL = "#a65f00"
E_COL = "#d62728"
I_COL = "#1f77b4"
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]

MAX_E_BACKGROUND = 900
MAX_I_BACKGROUND = 260


def _axis(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    return u, np.array([-u[1], u[0]])


def _shaft(name: str) -> str:
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _shaft_color(name: str, shafts: list[str]) -> str:
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _load_figdata(tag: str):
    path = RUN / f"figdata_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True), path


def _infer_seed(tag: str) -> int | None:
    matches = re.findall(r"_s(\d+)(?:$|_)", tag)
    return int(matches[-1]) if matches else None


def _reconstruct_posI(fd, tag: str) -> tuple[np.ndarray | None, dict]:
    """Reconstruct I-neuron coordinates from the deterministic placement RNG.

    Current figdata stores only posE. ``place_neurons`` uses a single uniform draw
    with E first and I second, so the seed in the tag is enough to recover posI.
    """
    seed = _infer_seed(tag)
    if seed is None:
        return None, {"posI_source": "not_reconstructed", "reason": "seed_not_in_tag"}
    posE = np.asarray(fd["posE"], float)
    L = float(fd["L"])
    f_E = 0.8
    n_total = int(round(len(posE) / f_E))
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n_total, 2)).astype(float)
    max_abs_diff = float(np.max(np.abs(pos[: len(posE)] - posE)))
    meta = {
        "posI_source": "deterministic_reconstruction_from_place_neurons",
        "seed": seed,
        "n_E": int(len(posE)),
        "n_I": int(n_total - len(posE)),
        "max_abs_diff_reconstructed_E_vs_figdata": max_abs_diff,
    }
    if max_abs_diff > 1e-5:
        meta["posI_source"] = "not_reconstructed"
        meta["reason"] = "reconstructed_E_did_not_match_figdata"
        return None, meta
    return pos[len(posE) :], meta


def _sample_indices(idx: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if len(idx) <= max_n:
        return idx
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def _same_fraction_core_n(n_core: int, n_bg: int, n_bg_plot: int) -> int:
    if n_core <= 0 or n_bg <= 0:
        return 0
    return max(1, int(round(n_core * n_bg_plot / n_bg)))


def _draw_contacts(
    ax,
    contacts: np.ndarray,
    names: list[str],
    shafts: list[str],
    core_a: list[str],
    core_b: list[str],
    *,
    homogeneous_cores: bool = False,
    semantic_core_colors: bool = False,
) -> None:
    core_a_set = set(core_a)
    core_b_set = set(core_b)
    for sh in shafts:
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        c = contacts[idx]
        ax.plot(c[:, 0], c[:, 1], color="black", lw=1.45, alpha=0.82, zorder=7)
        if semantic_core_colors:
            edgecolors = [
                TA_COLOR if names[i] in core_a_set
                else TB_COLOR if names[i] in core_b_set
                else "black"
                for i in idx
            ]
            linewidths = [
                2.15 if names[i] in core_a_set or names[i] in core_b_set else 1.65
                for i in idx
            ]
            ax.scatter(
                c[:, 0], c[:, 1], s=72, marker="o", fc="white",
                ec=edgecolors, lw=linewidths, zorder=8,
            )
        else:
            ax.scatter(
                c[:, 0], c[:, 1], s=72, marker="o", fc="white",
                ec="black", lw=1.65, zorder=8,
            )
    if semantic_core_colors:
        return
    core_groups = (
        ((core_a, HOMOGENEOUS_CORE), (core_b, HOMOGENEOUS_CORE))
        if homogeneous_cores
        else ((core_a, "crimson"), (core_b, "#2166ac"))
    )
    for members, color in core_groups:
        for name in members:
            if name not in names:
                continue
            xy = contacts[names.index(name)]
            ax.scatter([xy[0]], [xy[1]], s=112, marker="o", fc="none", ec=color,
                       lw=2.1, zorder=10)


def _add_legend(ax, *, homogeneous_cores: bool = False) -> None:
    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=E_COL, markeredgecolor="none",
               alpha=0.70, markersize=6.6, label="E neuron"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=I_COL, markeredgecolor="none",
               alpha=0.70, markersize=6.4, label="I neuron"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgewidth=1.5,
               linewidth=1.2, markersize=6.4, label="electrode"),
        Line2D([0], [0], color=HOMOGENEOUS_CORE, linestyle="--", linewidth=2.0,
               label="low-$V_\\theta$ core"),
        Patch(facecolor=FWD_SHADE, edgecolor=AXIS_COL, alpha=0.35, label="anisotropic E→E axis"),
    ]
    if not homogeneous_cores:
        handles[3:4] = [
            Line2D([0], [0], color="crimson", linestyle="--", linewidth=2.0,
                   label="low-$V_\\theta$ core A"),
            Line2D([0], [0], color="#2166ac", linestyle="--", linewidth=2.0,
                   label="low-$V_\\theta$ core B"),
        ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fontsize=9.2 if homogeneous_cores else 7.0,
        handlelength=1.25,
        borderpad=0.32,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.86)
    leg.get_frame().set_edgecolor("0.75")
    leg.get_frame().set_linewidth(0.8)


def _plot_mechanism(
    fd,
    ax,
    *,
    clean: bool,
    posI: np.ndarray | None = None,
    plot_seed: int = 0,
    display: dict | None = None,
    homogeneous_cores: bool = False,
    semantic_core_colors: bool = False,
    show_basic_labels: bool = False,
    show_title: bool = True,
) -> dict:
    pos_native = np.asarray(fd["posE"], float)
    vth = np.asarray(fd["vth"], float)
    foci_native = np.asarray(fd["foci"], float)
    contacts_native = np.asarray(fd["contacts"], float)
    names = [str(x) for x in fd["names"]]
    reg = fd["reg"].item()
    core_a = list(reg["source_names"])
    core_b = list(reg["sink_names"])
    L = float(fd["L"])
    core_r_native = float(fd["core_r"])
    shafts = sorted(set(_shaft(n) for n in names))
    source_native = foci_native[0]
    sink_native = foci_native[1]
    pos = _display_xy(pos_native, display)
    foci = _display_xy(foci_native, display)
    contacts = _display_xy(contacts_native, display)
    posI_display = None if posI is None else _display_xy(posI, display)
    core_r = _display_radius(core_r_native, display)
    source = foci[0]
    sink = foci[1]
    inter_core = float(np.linalg.norm(sink - source))
    theta_display = float(np.degrees(np.arctan2(sink[1] - source[1], sink[0] - source[0])))
    ar = 2.0
    # This is a reader-facing corridor overlay, not the literal single-cell
    # connection kernel.  Centre it on BOTH equal low-threshold cores so the
    # drawing does not imply a privileged source or a one-way scaffold.
    ellipse_center = 0.5 * (source + sink)
    major_radius = 0.5 * inter_core + 1.7
    minor_radius = max(1.45 * core_r, 1.7)

    ax.set_facecolor("white")
    rng = np.random.default_rng(plot_seed)
    source_mask_E = np.linalg.norm(pos_native - source_native, axis=1) <= core_r_native
    sink_mask_E = np.linalg.norm(pos_native - sink_native, axis=1) <= core_r_native
    core_mask_E = source_mask_E | sink_mask_E
    e_bg_all = np.flatnonzero(~core_mask_E)
    e_core_all = np.flatnonzero(core_mask_E)
    e_bg_idx = _sample_indices(e_bg_all, MAX_E_BACKGROUND, rng)
    e_core_idx = _sample_indices(
        e_core_all,
        _same_fraction_core_n(len(e_core_all), len(e_bg_all), len(e_bg_idx)),
        rng,
    )

    if posI is not None:
        source_mask_I = np.linalg.norm(posI - source_native, axis=1) <= core_r_native
        sink_mask_I = np.linalg.norm(posI - sink_native, axis=1) <= core_r_native
        core_mask_I = source_mask_I | sink_mask_I
        i_bg_all = np.flatnonzero(~core_mask_I)
        i_core_all = np.flatnonzero(core_mask_I)
        i_bg_idx = _sample_indices(i_bg_all, MAX_I_BACKGROUND, rng)
        i_core_idx = _sample_indices(
            i_core_all,
            _same_fraction_core_n(len(i_core_all), len(i_bg_all), len(i_bg_idx)),
            rng,
        )
        ax.scatter(
            posI_display[i_bg_idx, 0],
            posI_display[i_bg_idx, 1],
            s=18,
            c=I_COL,
            marker="o",
            alpha=0.46,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            posI_display[i_core_idx, 0],
            posI_display[i_core_idx, 1],
            s=18,
            c=I_COL,
            marker="o",
            alpha=0.46,
            linewidths=0,
            rasterized=True,
            zorder=6,
        )
    else:
        source_mask_I = np.array([], dtype=bool)
        i_bg_idx = np.array([], dtype=int)
        i_core_idx = np.array([], dtype=int)

    ax.scatter(
        pos[e_bg_idx, 0],
        pos[e_bg_idx, 1],
        s=24,
        c=E_COL,
        marker="^",
        alpha=0.44,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        pos[e_core_idx, 0],
        pos[e_core_idx, 1],
        s=24,
        c=E_COL,
        marker="^",
        alpha=0.44,
        linewidths=0,
        rasterized=True,
        zorder=5,
    )
    ax.add_patch(
        Ellipse(
            ellipse_center,
            width=2.0 * major_radius,
            height=2.0 * minor_radius,
            angle=theta_display,
            fc=FWD_SHADE,
            ec=AXIS_COL,
            lw=2.0,
            alpha=0.24,
            zorder=4,
        )
    )

    core_cols = (
        (TA_COLOR, TB_COLOR)
        if semantic_core_colors
        else (HOMOGENEOUS_CORE, HOMOGENEOUS_CORE)
        if homogeneous_cores
        else (TA_COLOR, TB_COLOR)
    )
    core_labels = (
        ("Core 1", "Core 2")
        if show_basic_labels
        else ("", "")
        if homogeneous_cores
        else ("core A", "core B")
    )
    for focus, color, label in zip((source, sink), core_cols, core_labels):
        ax.add_patch(plt.Circle(focus, core_r, fill=False, ec=color, lw=2.2, ls="--", zorder=9))
        if label:
            ax.text(
                focus[0],
                focus[1] + 1.18 * core_r,
                label,
                color=color,
                fontsize=7.4,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
            )

    # Arrow-free line: the E->E kernel is spatially anisotropic but does not
    # prescribe a single propagation direction.  Core A or B may nucleate.
    ax.plot(
        [source[0], sink[0]],
        [source[1], sink[1]],
        color=AXIS_COL,
        lw=2.1,
        solid_capstyle="round",
        zorder=8,
    )

    _draw_contacts(
        ax,
        contacts,
        names,
        shafts,
        core_a,
        core_b,
        homogeneous_cores=homogeneous_cores,
        semantic_core_colors=semantic_core_colors,
    )
    if show_basic_labels:
        neuron_handles = [
            Line2D(
                [0], [0], marker="^", color="none", markerfacecolor=E_COL,
                markeredgecolor=E_COL, markersize=6.5, label="E neuron",
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=I_COL,
                markeredgecolor=I_COL, markersize=6.1, label="I neuron",
            ),
        ]
        leg = ax.legend(
            handles=neuron_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
            frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.78",
            fontsize=7.8, handlelength=0.9, handletextpad=0.45,
            borderpad=0.42, labelspacing=0.28, borderaxespad=0.0,
        )
        leg.get_frame().set_linewidth(0.75)
        leg.set_zorder(20)
        axis_unit = (sink - source) / inter_core
        normal = np.asarray([-axis_unit[1], axis_unit[0]])
        label_xy = ellipse_center + (minor_radius + 0.45) * normal
        ax.text(
            label_xy[0], label_xy[1], "anisotropic E→E",
            color=AXIS_COL, fontsize=7.8, fontweight="bold",
            ha="center", va="bottom", zorder=14,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5),
        )
    else:
        _add_legend(ax, homogeneous_cores=homogeneous_cores)

    if display is None:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
    else:
        ax.set_xlim(*display["xlim"])
        ax.set_ylim(*display["ylim"])
    ax.set_aspect("equal")
    if clean:
        ax.set_axis_off()
        ax.patch.set_visible(True)
    else:
        if show_title:
            ax.set_title("mechanism", fontsize=18, fontweight="bold", pad=10)
        ax.set_xlabel("TA shared axis (mm)" if display else "x (mm)", fontsize=12.5)
        ax.set_ylabel("transverse (mm)" if display else "y (mm)", fontsize=12.5)
        ax.tick_params(axis="both", labelsize=10.5, length=4.0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
            sp.set_color("0.25")

    return {
        "core_a": core_a,
        "core_b": core_b,
        "core_r": core_r,
        "core_r_native_sheet": core_r_native,
        "core_r_display_mm": core_r,
        "theta_deg": theta_display,
        "theta_deg_native_sheet": float(fd["theta_deg"]),
        "theta_deg_display": theta_display,
        "display_axis": None if display is None else {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in display.items()
        },
        "ee_ar": ar,
        "ee_ellipse_center": ellipse_center.tolist(),
        "ee_ellipse_major_radius": major_radius,
        "ee_ellipse_minor_radius": minor_radius,
        "L": L,
        "contact_names": names,
        "plotted_neurons": {
            "E_background": int(len(e_bg_idx)),
            "I_background": int(len(i_bg_idx)),
            "E_core": int(len(e_core_idx)),
            "I_core": int(len(i_core_idx)),
            "E_total": int(len(pos)),
            "I_total": int(0 if posI is None else len(posI)),
            "max_E_background": MAX_E_BACKGROUND,
            "max_I_background": MAX_I_BACKGROUND,
            "core_sampling": "same_fraction_as_background",
            "markers": {"E": "triangle", "I": "circle"},
        },
    }


def _save_one(
    fd,
    outdir: Path,
    stem: str,
    *,
    clean: bool,
    dpi: int,
    posI: np.ndarray | None,
    plot_seed: int,
) -> dict:
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="white")
    meta = _plot_mechanism(fd, ax, clean=clean, posI=posI, plot_seed=plot_seed)
    pad = 0.0 if clean else 0.08
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    svg = outdir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=pad, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    meta.update({"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT))})
    return meta


def _write_readme(outdir: Path, fig_name: str) -> None:
    (outdir / "README.md").write_text(
        f"""# {fig_name}

### {fig_name}_hires.png / .pdf / .svg

这是 E1146 subject-specific SNN 图的独立高清 mechanism panel，重画自同一个 figdata，不是从原四联图裁切。红色三角是降采样后的兴奋性神经元，蓝色圆点是按同一 seed 确定性重建并降采样后的抑制性神经元；两端虚线圆分别是等强低阈值 core A/B。暖色透明带表示连接更易沿轴延伸的 E→E 各向异性 scaffold；中线不加箭头，因为传播方向取决于哪一端先自发成核。病人真实触点统一用黑色绘制。

**关注点**：看两个低阈值 core 是否分别落在两类真实模板的最早触点区，以及同一 E→E 长轴是否连接两端而不预设单向传播。

### {fig_name}_ppt_clean.png / .pdf / .svg

这是给 PPT 使用的无坐标轴版本，保留降采样 E/I 神经元、双低阈值 core、触点排布和 AR=2 的 E→E 长轴带，去掉坐标刻度和标题，便于直接放进幻灯片并自行加说明。

**关注点**：这张图只表达模型底物和电极排布如何对齐；不能单独作为真实病人机制被证明的证据。
""",
        encoding="utf-8",
    )


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--fig-name", default=FIG_NAME)
    ap.add_argument("--dpi", type=int, default=500)
    args = ap.parse_args()

    fd, source_path = _load_figdata(args.tag)
    outdir = ROOT / "results" / "paper-ready-figure" / args.fig_name / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    posI, posI_meta = _reconstruct_posI(fd, args.tag)
    plot_seed = int((posI_meta.get("seed") or 0) + 101)
    hires = _save_one(
        fd,
        outdir,
        f"{args.fig_name}_hires",
        clean=False,
        dpi=args.dpi,
        posI=posI,
        plot_seed=plot_seed,
    )
    clean = _save_one(
        fd,
        outdir,
        f"{args.fig_name}_ppt_clean",
        clean=True,
        dpi=args.dpi,
        posI=posI,
        plot_seed=plot_seed,
    )
    _write_readme(outdir, args.fig_name)

    meta = {
        "figure": args.fig_name,
        "source_figdata": str(source_path.relative_to(ROOT)),
        "source_tag": args.tag,
        "plot_seed": plot_seed,
        "posI": posI_meta,
        "outputs": {"hires": hires, "ppt_clean": clean},
        "notes": [
            "Plotting-only; no SNN rerun.",
            "Same E1146 template_source placement as Fig4A.",
            "Both equal low-threshold template-source cores are drawn.",
            "E/I neurons are downsampled for presentation; exact plotted counts are in outputs.*.plotted_neurons.",
            "Both equal low-threshold cores are drawn; neither end is privileged in the schematic.",
            "E->E scaffold is drawn as an arrow-free AR=2 schematic corridor, not a literal single-cell kernel.",
            "Standalone mechanism panel for PPT use; not a new scientific result.",
        ],
    }
    meta_path = outdir / f"{args.fig_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
