#!/usr/bin/env python3
"""Render the dual-core spatial-Z bifurcation diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map  # noqa: E402
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def panel_label(axis, label):
    axis.text(-0.14, 1.08, label, transform=axis.transAxes,
              fontsize=13, fontweight="bold", va="top", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z")
    parser.add_argument(
        "--result", default=base
        + "/bifurcation/dualcore_spatial_z_bifurcation.json")
    parser.add_argument(
        "--out-dir", default=base + "/bifurcation/figures")
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    arrays_path = result_path.with_suffix(".npz")
    payload = json.loads(result_path.read_text())
    arrays = np.load(arrays_path, allow_pickle=False)
    model = load_patient_coarse_model(payload["substrate"]["model"]["path"])
    z_map = load_z_map(Path(payload["substrate"]["z_map"]["path"]))
    folds = payload["folds"]
    empirical = payload["empirical_ou_on_projection"]
    s_empirical = float(empirical["s_from_core_median"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(11.4, 7.1), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.02, 1.2],
                               height_ratios=[1.0, 1.02])

    # A: Z(x) at the empirical OU-on onset projection.
    axis_a = figure.add_subplot(grid[0, 0])
    profile = z_map.depletion_profile(
        core_a_weight=1.0, core_b_weight=1.0, surround_weight=0.70)
    z_field = (1.0 - s_empirical * profile).reshape(
        model.n_grid, model.n_grid)
    image = axis_a.imshow(
        z_field, origin="lower", extent=[0, model.sheet_l_mm] * 2,
        vmin=0.65, vmax=1.0, cmap="viridis", interpolation="nearest")
    colors = ("#f28e2b", "#31a6b8")
    for index, (center, color) in enumerate(zip(z_map.centers_mm, colors)):
        axis_a.scatter(center[0], center[1], s=90, facecolors="none",
                       edgecolors=color, linewidths=2.0)
        axis_a.text(center[0] + 0.45, center[1] + 0.45,
                    f"core {'AB'[index]}", color=color, weight="bold")
    axis_a.set(xlabel="sheet x (mm)", ylabel="sheet y (mm)",
               title="Spatial Z at the OU-on transition")
    colorbar = figure.colorbar(image, ax=axis_a, fraction=0.047, pad=0.03)
    colorbar.set_label("inhibitory efficacy  Z(x)")
    axis_a.text(
        0.02, 0.02,
        rf"$s_{{SNN}}={s_empirical:.3f}$  |  "
        rf"$Z_{{core}}={empirical['z_core_median']:.3f}$, "
        rf"$Z_{{sur}}={empirical['z_surround_median']:.3f}$",
        transform=axis_a.transAxes, fontsize=7.5, color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2.5,
              "edgecolor": "none"})
    panel_label(axis_a, "A")

    # B: low and recruited roots on the same spatial path.
    axis_b = figure.add_subplot(grid[0, 1])
    recovery = float(folds["recruited_recovery"]["s"])
    entry = float(folds["runaway_entry"]["s"])
    axis_b.axvspan(recovery, entry, color="#eee8d5", alpha=0.72,
                   label="low/high root coexistence")
    axis_b.plot(arrays["low_branch__s"], arrays["low_branch__mean_e_hz"],
                color="#4e79a7", lw=2.0, label="low root")
    axis_b.plot(arrays["outer_high__s"], arrays["outer_high__mean_e_hz"],
                color="#d55e00", lw=1.8, label="recruited tonic root")
    axis_b.plot(arrays["recruited_arc__s"],
                arrays["recruited_arc__mean_e_hz"], color="#8c2d04",
                lw=1.0, alpha=0.72, label="spatial fold chain")
    axis_b.axvline(recovery, color="#8c2d04", ls="--", lw=1.2)
    axis_b.axvline(entry, color="#4e79a7", ls="--", lw=1.2)
    axis_b.axvline(s_empirical, color="#cc3366", ls=":", lw=1.8,
                   label="median OU-on onset")
    axis_b.scatter(
        [recovery, entry],
        [folds["recruited_recovery"]["mean_e_rate_hz"],
         folds["runaway_entry"]["mean_e_rate_hz"]],
        c=["#8c2d04", "#4e79a7"], s=34, zorder=4)
    axis_b.annotate("high branch folds", (recovery, 100),
                    xytext=(0.12, 35), textcoords="data",
                    arrowprops={"arrowstyle": "->", "lw": 0.8})
    axis_b.annotate("low root disappears", (entry, 0.0625),
                    xytext=(0.225, 0.25), textcoords="data",
                    arrowprops={"arrowstyle": "->", "lw": 0.8})
    axis_b.set_yscale("log")
    axis_b.set(xlim=(0, 0.405), ylim=(0.025, 520),
               xlabel="spatial disinhibition  s  (larger = weaker inhibition)",
               ylabel="mean E rate (Hz, log scale)",
               title="Runaway boundary and hysteresis")
    axis_b.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
    panel_label(axis_b, "B")

    # C: spatial shape of the two zero modes.
    subgrid = grid[1, 0].subgridspec(1, 2, wspace=0.08)
    mode_axes = [figure.add_subplot(subgrid[0, 0]),
                 figure.add_subplot(subgrid[0, 1])]
    mode_keys = (("entry_fold__critical_mode_e", "runaway-entry zero mode"),
                 ("recovery_fold__critical_mode_e", "recovery zero mode"))
    maximum = max(float(np.max(np.abs(arrays[key]))) for key, _ in mode_keys)
    mode_image = None
    for index, (axis, (key, title)) in enumerate(zip(mode_axes, mode_keys)):
        mode_image = axis.imshow(
            arrays[key], origin="lower", extent=[0, model.sheet_l_mm] * 2,
            cmap="coolwarm", vmin=-maximum, vmax=maximum,
            interpolation="nearest")
        for center, color in zip(z_map.centers_mm, colors):
            axis.scatter(center[0], center[1], s=55, facecolors="none",
                         edgecolors=color, linewidths=1.4)
        axis.set(title=title, xlabel="x (mm)")
        if index == 0:
            axis.set_ylabel("sheet y (mm)")
        else:
            axis.set_yticklabels([])
    mode_bar = figure.colorbar(mode_image, ax=mode_axes, fraction=0.045, pad=0.03)
    mode_bar.set_label("normalized E component")
    panel_label(mode_axes[0], "C")

    # D: independent core-A/core-B Z map at fixed empirical surround Z.
    axis_d = figure.add_subplot(grid[1, 1])
    z_a = arrays["phase__z_a"]
    z_b = arrays["phase__z_b"]
    state = arrays["phase__state_code"]
    cmap = ListedColormap(["#ffffff", "#4e79a7", "#d55e00", "#7b6fa6"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    half_x = float(np.diff(z_a).mean() / 2)
    half_y = float(np.diff(z_b).mean() / 2)
    axis_d.imshow(
        state, origin="lower", interpolation="nearest", cmap=cmap, norm=norm,
        extent=[z_a[0] - half_x, z_a[-1] + half_x,
                z_b[0] - half_y, z_b[-1] + half_y], aspect="equal")
    axis_d.scatter(empirical["z_core_median"], empirical["z_core_median"],
                   marker="D", s=45, color="#cc3366", edgecolor="white",
                   linewidth=0.8, label="OU-on median projection")
    axis_d.set(
        xlabel=r"core-A efficacy  $Z_A$", ylabel=r"core-B efficacy  $Z_B$",
        title=r"Independent cores at fixed $Z_{surround}=0.80$")
    legend = [
        Patch(facecolor="#d55e00", label="recruited root only"),
        Patch(facecolor="#7b6fa6", label="low + recruited roots"),
        plt.Line2D([], [], marker="D", ls="none", color="#cc3366",
                   markeredgecolor="white", label="OU-on median"),
    ]
    axis_d.legend(handles=legend, frameon=False, fontsize=7.5,
                  loc="upper left")
    axis_d.text(
        0.02, 0.02, "lower Z = stronger disinhibition",
        transform=axis_d.transAxes, fontsize=7.3,
        bbox={"facecolor": "white", "alpha": 0.82,
              "edgecolor": "none", "pad": 2})
    panel_label(axis_d, "D")

    figure.suptitle(
        "Frozen data-driven two-core substrate: spatial-Z fold structure",
        fontsize=13.5, fontweight="bold")
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "dualcore_spatial_z_bifurcation_diagnostic"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "status": "DUAL_CORE_SPATIAL_Z_BIFURCATION_DIAGNOSTIC_RENDERED",
        "figure_role": "mechanistic diagnostic; not yet a paper Fig.5 panel",
        "source": {"json": str(result_path), "sha256": sha256(result_path),
                   "npz": str(arrays_path), "npz_sha256": sha256(arrays_path)},
        "outputs": {
            "png": {"path": str(png), "sha256": sha256(png)},
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
        },
        "panel_semantics": {
            "A": "model Z projection on the frozen data-driven dual-core geometry",
            "B": "low/recruited root coexistence and spatial fold chain",
            "C": "critical spatial eigenmodes at entry and recovery folds",
            "D": "finite multi-start root map for independent core A/B Z",
        },
        "claim_boundary": payload["claim_boundary"],
    }
    atomic_json(metadata, stem.with_suffix(".metadata.json"))
    readme = output_dir / "README.md"
    readme.write_text(
        "### dualcore_spatial_z_bifurcation_diagnostic.png\n\n"
        "这是一张确定性机制诊断图，不是已经锁版的 Fig.5。A 把模型 Z 投影到冻结的 data-driven 双核几何；B 显示低根、持续招募根和折叠链；C 显示入口与恢复折点的零模空间形状；D 在固定 surround Z 下分别改变两个 core 的 Z。\n\n"
        "它支持的是 1 mm 粗粒化快子系统存在空间 saddle-node 链，并说明 OU-on SNN 的中位转变发生在根共存区；不能据此声称真实患者存在同样的抑制场或热力学相变。\n\n"
        "**关注点**：看 B 中两个边界与粉色 OU-on 位置的关系，以及 C/D 是否显示由单个 core 的局部模式先触发。\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
