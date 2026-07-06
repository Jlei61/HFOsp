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
) -> None:
    for sh in shafts:
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        c = contacts[idx]
        ax.plot(c[:, 0], c[:, 1], color="black", lw=1.45, alpha=0.82, zorder=7)
        ax.scatter(c[:, 0], c[:, 1], s=72, marker="o", fc="white", ec="black", lw=1.65, zorder=8)


def _add_legend(ax) -> None:
    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=E_COL, markeredgecolor="none",
               alpha=0.70, markersize=6.6, label="E neuron"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=I_COL, markeredgecolor="none",
               alpha=0.70, markersize=6.4, label="I neuron"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgewidth=1.5,
               linewidth=1.2, markersize=6.4, label="electrode"),
        Line2D([0], [0], color="crimson", linestyle="--", linewidth=2.0, label="source core"),
        Patch(facecolor=FWD_SHADE, edgecolor=AXIS_COL, alpha=0.35, label="E->E AR=2"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fontsize=7.0,
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
) -> dict:
    pos = np.asarray(fd["posE"], float)
    vth = np.asarray(fd["vth"], float)
    foci = np.asarray(fd["foci"], float)
    contacts = np.asarray(fd["contacts"], float)
    names = [str(x) for x in fd["names"]]
    reg = fd["reg"].item()
    core_a = list(reg["source_names"])
    core_b = list(reg["sink_names"])
    L = float(fd["L"])
    core_r = float(fd["core_r"])
    u, p = _axis(float(fd["theta_deg"]))
    shafts = sorted(set(_shaft(n) for n in names))
    source = foci[0]
    sink = foci[1]
    inter_core = float(np.linalg.norm(sink - source))
    ar = 2.0
    major_radius = 0.48 * inter_core
    minor_radius = major_radius / ar
    ellipse_center = source + u * major_radius

    ax.set_facecolor("white")
    rng = np.random.default_rng(plot_seed)
    source_mask_E = np.linalg.norm(pos - source, axis=1) <= core_r
    e_bg_all = np.flatnonzero(~source_mask_E)
    e_core_all = np.flatnonzero(source_mask_E)
    e_bg_idx = _sample_indices(e_bg_all, MAX_E_BACKGROUND, rng)
    e_core_idx = _sample_indices(
        e_core_all,
        _same_fraction_core_n(len(e_core_all), len(e_bg_all), len(e_bg_idx)),
        rng,
    )

    if posI is not None:
        source_mask_I = np.linalg.norm(posI - source, axis=1) <= core_r
        i_bg_all = np.flatnonzero(~source_mask_I)
        i_core_all = np.flatnonzero(source_mask_I)
        i_bg_idx = _sample_indices(i_bg_all, MAX_I_BACKGROUND, rng)
        i_core_idx = _sample_indices(
            i_core_all,
            _same_fraction_core_n(len(i_core_all), len(i_bg_all), len(i_bg_idx)),
            rng,
        )
        ax.scatter(
            posI[i_bg_idx, 0],
            posI[i_bg_idx, 1],
            s=18,
            c=I_COL,
            marker="o",
            alpha=0.46,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            posI[i_core_idx, 0],
            posI[i_core_idx, 1],
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
            angle=float(fd["theta_deg"]),
            fc=FWD_SHADE,
            ec=AXIS_COL,
            lw=2.0,
            alpha=0.24,
            zorder=4,
        )
    )

    ax.add_patch(plt.Circle(source, core_r, fill=False, ec="crimson", lw=2.2, ls="--", zorder=9))

    ax.annotate(
        "",
        xy=source + u * (1.45 * major_radius),
        xytext=source,
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COL, lw=2.5, mutation_scale=17),
        zorder=12,
    )

    _draw_contacts(ax, contacts, names, shafts)
    _add_legend(ax)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    if clean:
        ax.set_axis_off()
        ax.patch.set_visible(True)
    else:
        ax.set_title("mechanism", fontsize=18, fontweight="bold", pad=10)
        ax.set_xlabel("x (mm)", fontsize=13)
        ax.set_ylabel("y (mm)", fontsize=13)
        ax.tick_params(axis="both", labelsize=11, length=4.0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
            sp.set_color("0.25")

    return {
        "source_core": core_a,
        "hidden_sink_core_not_drawn": core_b,
        "core_r": core_r,
        "theta_deg": float(fd["theta_deg"]),
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

这是 E1146 subject-specific SNN 图的独立高清 mechanism panel，重画自同一个 figdata，不是从原四联图裁切。红色三角是进一步降采样后的兴奋性神经元，蓝色圆点是按同一 seed 确定性重建并降采样后的抑制性神经元；core 内外都使用同一套 E/I 颜色、marker、点大小和透明度。虚线圆是单侧 source core，暖色透明椭圆是 AR=2 的 E→E 长轴连接 lobe。病人真实触点统一用黑色绘制。

**关注点**：看 source core 是否落在真实 template-source 触点附近，以及 E→E 连接 lobe 是否沿病人传播轴外延。

### {fig_name}_ppt_clean.png / .pdf / .svg

这是给 PPT 使用的无坐标轴版本，保留降采样 E/I 神经元、单侧 source core、触点排布和 AR=2 的 E→E 长轴椭圆，去掉坐标刻度和标题，便于直接放进幻灯片并自行加说明。

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
            "Only the source-side core is drawn; the opposite template core is retained in metadata but hidden.",
            "E/I neurons are downsampled for presentation; exact plotted counts are in outputs.*.plotted_neurons.",
            "E->E scaffold is drawn as an AR=2 schematic ellipse, not as a rectangular corridor.",
            "Standalone mechanism panel for PPT use; not a new scientific result.",
        ],
    }
    meta_path = outdir / f"{args.fig_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
