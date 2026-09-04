#!/usr/bin/env python3
"""Render paper-facing Figure 5C/D candidates from the verified spatial-Z result.

Panel C is the smallest scale supported by the deterministic reduction: the
per-neuron mean E rate inside data-driven core A along the symmetric spatial-Z
continuation path.  It is deliberately not labelled as a literal single-cell
bifurcation because each 2-mm coarse unit represents a local E/I population.

Panel D links the local diagram to space.  Its left map shows the physical
runaway-entry zero mode; its right map shows the finite multi-start root
catalog when core-A and core-B inhibitory efficacy are varied independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map  # noqa: E402
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


LOW = "#355C8A"
HIGH = "#C7472F"
FOLD_CHAIN = "#8F2D1E"
OU = "#D62745"
COEXIST = "#D8D2E8"
RECRUITED_ONLY = "#E88963"
CORE_A = "#F28E2B"
CORE_B = "#2FA7B8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _panel_label(axis, label: str, *, x: float = -0.16, y: float = 1.10) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, ha="left", va="top",
        fontsize=17, fontweight="bold", clip_on=False,
    )


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.6, width=0.8, length=3.0)


def _plot_panel_c(axis, arrays, payload) -> dict:
    folds = payload["folds"]
    empirical = payload["empirical_ou_on_projection"]
    recovery = float(folds["recruited_recovery"]["s"])
    entry = float(folds["runaway_entry"]["s"])
    onset_median = float(empirical["s_from_core_median"])
    z_q10, z_q90 = map(float, empirical["z_core_q10_q90"])
    onset_q10, onset_q90 = 1.0 - z_q90, 1.0 - z_q10

    axis.axvspan(
        recovery, entry, color=COEXIST, alpha=0.42, lw=0,
        label="low/high coexistence",
    )
    axis.axvspan(
        onset_q10, onset_q90, color=OU, alpha=0.10, lw=0,
        label="OU-on onset q10–q90",
    )
    axis.plot(
        arrays["low_branch__s"], arrays["low_branch__core_a_hz"],
        color=LOW, lw=2.2, solid_capstyle="round", label="low root",
    )
    axis.plot(
        arrays["outer_high__s"], arrays["outer_high__core_a_hz"],
        color=HIGH, lw=2.2, solid_capstyle="round", label="tonic root",
    )
    axis.plot(
        arrays["recruited_arc__s"], arrays["recruited_arc__core_a_hz"],
        color=FOLD_CHAIN, lw=1.05, alpha=0.72,
        label="spatial fold chain",
    )
    axis.plot(
        arrays["entry_fold__s"], arrays["entry_fold__core_a_hz"],
        color=LOW, lw=1.0, ls="--", alpha=0.82,
    )
    axis.axvline(onset_median, color=OU, lw=1.3, ls=":")

    recovery_rate = float(
        folds["recruited_recovery"]["regional_e_rate_hz"]["core_a"])
    entry_rate = float(
        folds["runaway_entry"]["regional_e_rate_hz"]["core_a"])
    axis.scatter(
        [recovery, entry], [recovery_rate, entry_rate], s=34,
        c=[FOLD_CHAIN, LOW], edgecolor="white", lw=0.7, zorder=6,
    )
    axis.annotate(
        "low root ends", xy=(entry, entry_rate), xytext=(0.245, 2.0),
        fontsize=7.3, color=LOW,
        arrowprops={"arrowstyle": "->", "color": LOW, "lw": 0.8},
    )
    axis.annotate(
        "OU-on median", xy=(onset_median, 370.0), xytext=(0.205, 405.0),
        fontsize=7.3, color=OU,
        arrowprops={"arrowstyle": "-", "color": OU, "lw": 0.8},
    )
    axis.text(
        0.196, 0.17, "coexistence", transform=axis.get_xaxis_transform(),
        fontsize=7.1, color="#5C536B", ha="center",
    )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 0.405)
    axis.set_ylim(0.15, 520.0)
    axis.set_xlabel(r"Core disinhibition  $D_A=1-Z_A$", fontsize=9.0)
    axis.set_ylabel(r"Core-A E rate (Hz neuron$^{-1}$)", fontsize=9.0)
    axis.set_title("Core-A local fixed-point structure", fontsize=10.2,
                   fontweight="bold", pad=8)
    # The main log-scale view hides the very narrow arclength turn.  The inset
    # shows the actual fold geometry and reports the independent zero-mode
    # bracket, rather than representing a solver jump as a bifurcation.
    inset = axis.inset_axes([0.565, 0.17, 0.385, 0.235])
    fold_s = np.asarray(arrays["entry_fold__s"], float)
    fold_rate = np.asarray(arrays["entry_fold__core_a_hz"], float)
    inset.plot((fold_s - entry) * 1e6, fold_rate, color=LOW, lw=1.2)
    inset.scatter([0.0], [entry_rate], s=14, color=LOW,
                  edgecolor="white", lw=0.45, zorder=4)
    inset.axvline(0.0, color="0.45", lw=0.6, ls=":")
    eigen_bracket = folds["runaway_entry"][
        "fixed_point_eigenvalue_real_bracket"]
    inset.text(
        0.04, 0.08,
        rf"Re $\lambda$: {eigen_bracket[0] * 1e5:.1f} $\to$ {eigen_bracket[1] * 1e5:.1f} $\times10^{{-5}}$",
        transform=inset.transAxes, fontsize=5.6, color="0.30",
    )
    inset.set_title("saddle-node zoom", fontsize=6.3, pad=2)
    inset.set_xlabel(r"$D_A-D_{fold}$  ($\times10^{-6}$)", fontsize=5.4,
                     labelpad=1)
    inset.set_ylabel("rate (Hz)", fontsize=5.4, labelpad=1)
    inset.tick_params(labelsize=5.0, width=0.55, length=1.8, pad=1)
    inset.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Line2D([], [], color=LOW, lw=2.2, label="low root"),
            Line2D([], [], color=HIGH, lw=2.2, label="tonic root"),
            Line2D([], [], color=FOLD_CHAIN, lw=1.1,
                   label="spatial fold chain"),
            Patch(facecolor=COEXIST, alpha=0.65,
                  label="low/high coexistence"),
        ],
        loc="upper left", frameon=False, fontsize=6.9, ncol=2,
        handlelength=1.8, columnspacing=0.9,
    )
    _style_axis(axis)
    return {
        "semantic": (
            "core-A per-neuron mean E rate along the symmetric spatial-Z "
            "fixed-point continuation; not a literal single-neuron bifurcation"
        ),
        "x_definition": "D_A=1-Z_A=s on Z_A=Z_B=1-s, Z_surround=1-0.70s",
        "runaway_entry_s": entry,
        "recruited_recovery_s": recovery,
        "ou_on_median_s": onset_median,
        "ou_on_q10_q90_s": [onset_q10, onset_q90],
        "y_scale": "log",
    }


def _plot_mode_map(axis, arrays, payload, model, z_map):
    mode = np.asarray(arrays["entry_fold__critical_mode_e"], float)
    energy = np.square(mode)
    energy /= max(float(np.max(energy)), 1e-12)
    image = axis.imshow(
        energy, origin="lower", extent=[0.0, model.sheet_l_mm] * 2,
        cmap="Reds", norm=PowerNorm(gamma=0.52, vmin=0.0, vmax=1.0),
        interpolation="nearest",
        aspect="equal",
    )
    core_fields = (
        np.asarray(z_map.core_a_fraction_e).reshape(model.n_grid, model.n_grid),
        np.asarray(z_map.core_b_fraction_e).reshape(model.n_grid, model.n_grid),
    )
    cell_width = float(model.sheet_l_mm / model.n_grid)
    centers = (np.arange(model.n_grid) + 0.5) * cell_width
    for index, (field, color) in enumerate(zip(core_fields, (CORE_A, CORE_B))):
        axis.contour(
            centers, centers, field, levels=[0.10], colors=[color],
            linewidths=1.35,
        )
        center = np.asarray(z_map.centers_mm[index], float)
        axis.text(
            center[0], center[1] + 1.7, f"core {'AB'[index]}", color=color,
            fontsize=7.0, fontweight="bold", ha="center", va="bottom",
        )
    fold_mode = payload["folds"]["runaway_entry_critical_mode"]
    axis.scatter(
        *fold_mode["peak_xy_mm"], marker="+", s=52, color="white",
        lw=1.4, zorder=6,
    )
    axis.set_xlim(0.0, model.sheet_l_mm)
    axis.set_ylim(0.0, model.sheet_l_mm)
    axis.set_xlabel("sheet x (mm)", fontsize=8.4)
    axis.set_ylabel("sheet y (mm)", fontsize=8.4)
    axis.set_title("Runaway-entry zero mode", fontsize=9.4,
                   fontweight="bold", pad=7)
    axis.text(
        0.03, 0.97,
        f"{100.0 * fold_mode['regional_e_mode']['core_a']['fraction_of_e_mode_energy']:.1f}% in core A",
        transform=axis.transAxes, ha="left", va="top", fontsize=7.0,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.48,
              "edgecolor": "none", "pad": 2.2},
    )
    axis.tick_params(labelsize=7.1, width=0.8, length=2.7)
    axis.spines[["top", "right"]].set_visible(False)
    return image


def _plot_phase_map(axis, arrays, payload):
    z_a = np.asarray(arrays["phase__z_a"], float)
    z_b = np.asarray(arrays["phase__z_b"], float)
    state = np.asarray(arrays["phase__state_code"], int)
    if not set(np.unique(state)).issubset({2, 3}):
        raise RuntimeError("paper Panel D expects recruited-only/coexistence states")
    cmap = ListedColormap([RECRUITED_ONLY, COEXIST])
    norm = BoundaryNorm([1.5, 2.5, 3.5], cmap.N)
    half_a = float(np.diff(z_a).mean() / 2.0)
    half_b = float(np.diff(z_b).mean() / 2.0)
    axis.imshow(
        state, origin="lower", interpolation="nearest", cmap=cmap, norm=norm,
        extent=[z_a[0] - half_a, z_a[-1] + half_a,
                z_b[0] - half_b, z_b[-1] + half_b],
        aspect="equal",
    )
    empirical = payload["empirical_ou_on_projection"]
    z_median = float(empirical["z_core_median"])
    axis.plot([z_a[0], z_a[-1]], [z_a[0], z_a[-1]], color="white",
              ls="--", lw=1.0, alpha=0.9)
    axis.scatter(
        z_median, z_median, marker="D", s=45, color=OU,
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    axis.text(
        z_median + 0.012, z_median - 0.014, "OU-on median",
        color=OU, fontsize=6.9, ha="left", va="top",
    )
    axis.set_xlim(z_a[0] - half_a, z_a[-1] + half_a)
    axis.set_ylim(z_b[0] - half_b, z_b[-1] + half_b)
    axis.set_xlabel(r"core-A efficacy  $Z_A$", fontsize=8.4)
    axis.set_ylabel(r"core-B efficacy  $Z_B$", fontsize=8.4)
    axis.set_title(r"Independent cores ($Z_{surround}=0.80$)", fontsize=9.4,
                   fontweight="bold", pad=7)
    axis.legend(
        handles=[
            Patch(facecolor=RECRUITED_ONLY, label="tonic root only"),
            Patch(facecolor=COEXIST, label="low + tonic roots"),
        ],
        loc="upper right", frameon=False, fontsize=6.8,
        borderaxespad=0.3, handlelength=1.3,
    )
    axis.text(
        0.02, 0.02, "lower Z = weaker inhibition",
        transform=axis.transAxes, fontsize=6.7, ha="left", va="bottom",
    )
    _style_axis(axis)


def _save_all(figure, stem: Path) -> dict:
    outputs = {}
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        figure.savefig(
            path, bbox_inches="tight", pad_inches=0.025,
            facecolor="white", **options,
        )
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z")
    parser.add_argument(
        "--result", default=base
        + "/bifurcation/dualcore_spatial_z_bifurcation.json")
    parser.add_argument(
        "--out-dir",
        default="results/paper-ready-figure/fig5_dual_core_spatial_z/figures",
    )
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    arrays_path = result_path.with_suffix(".npz")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "DUAL_CORE_SPATIAL_Z_FOLD_CHAIN_ESTABLISHED":
        raise RuntimeError("Figure 5C/D requires the verified spatial-Z fold result")
    arrays = np.load(arrays_path, allow_pickle=False)
    model_path = Path(payload["substrate"]["model"]["path"])
    z_map_path = Path(payload["substrate"]["z_map"]["path"])
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    z_map.validate(model)
    if not np.isclose(model.sheet_l_mm / model.n_grid, 2.0):
        raise RuntimeError("Figure contract expects the audited 2-mm reduction")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    output_dir = (ROOT / args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standalone C for direct insertion into the main figure.
    fig_c, axis_c = plt.subplots(figsize=(4.35, 3.35))
    panel_c_meta = _plot_panel_c(axis_c, arrays, payload)
    _panel_label(axis_c, "C", x=-0.18, y=1.13)
    fig_c.tight_layout(pad=0.5)
    outputs_c = _save_all(
        fig_c, output_dir / "fig5-panel-c-core-a-bifurcation")
    plt.close(fig_c)

    # Standalone D retains both physical space and independent-core state space.
    fig_d = plt.figure(figsize=(7.9, 3.35), facecolor="white")
    grid_d = fig_d.add_gridspec(
        1, 2, width_ratios=[1.0, 1.12], left=0.08, right=0.985,
        bottom=0.16, top=0.88, wspace=0.28,
    )
    axis_d1 = fig_d.add_subplot(grid_d[0, 0])
    axis_d2 = fig_d.add_subplot(grid_d[0, 1])
    mode_image = _plot_mode_map(axis_d1, arrays, payload, model, z_map)
    _plot_phase_map(axis_d2, arrays, payload)
    _panel_label(axis_d1, "D", x=-0.24, y=1.15)
    colorbar = fig_d.colorbar(mode_image, ax=axis_d1, fraction=0.046, pad=0.035)
    colorbar.set_label("normalized zero-mode energy", fontsize=7.3)
    colorbar.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    outputs_d = _save_all(
        fig_d, output_dir / "fig5-panel-d-spatial-z-phase")
    plt.close(fig_d)

    # One proof sheet at the intended bottom-row proportions.
    combined = plt.figure(figsize=(12.2, 3.45), facecolor="white")
    grid = combined.add_gridspec(
        1, 3, width_ratios=[1.28, 0.82, 1.0], left=0.055, right=0.985,
        bottom=0.17, top=0.88, wspace=0.30,
    )
    axis_c = combined.add_subplot(grid[0, 0])
    axis_d1 = combined.add_subplot(grid[0, 1])
    axis_d2 = combined.add_subplot(grid[0, 2])
    _plot_panel_c(axis_c, arrays, payload)
    mode_image = _plot_mode_map(axis_d1, arrays, payload, model, z_map)
    _plot_phase_map(axis_d2, arrays, payload)
    _panel_label(axis_c, "C", x=-0.16, y=1.15)
    _panel_label(axis_d1, "D", x=-0.23, y=1.15)
    colorbar = combined.colorbar(
        mode_image, ax=axis_d1, fraction=0.047, pad=0.036)
    colorbar.set_label("normalized zero-mode energy", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.7, width=0.7, length=2.4)
    outputs_cd = _save_all(
        combined, output_dir / "fig5-panels-cd-dual-core-spatial-z")
    plt.close(combined)

    phase = payload["phase_map"]
    metadata = {
        "status": "FIG5_CD_DUAL_CORE_SPATIAL_Z_CANDIDATE_RENDERED",
        "figure_role": (
            "paper-facing Figure 5C/D candidate; it must be paired only with "
            "A/B rendered from the same dualcore_s39 + Joint=1.25 substrate"
        ),
        "source": {
            "result_json": {"path": str(result_path),
                            "sha256": _sha256(result_path)},
            "arrays_npz": {"path": str(arrays_path),
                           "sha256": _sha256(arrays_path)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": _sha256(z_map_path)},
        },
        "substrate": payload["substrate"],
        "panel_C": panel_c_meta,
        "panel_D": {
            "left": (
                "physical 20-mm sheet map of normalized squared E component "
                "of the runaway-entry fixed-point zero mode"
            ),
            "right": (
                "finite multi-start root catalog over independently varied "
                "Z_A and Z_B at fixed Z_surround=0.80"
            ),
            "phase_scan": phase,
            "critical_mode": payload["folds"]["runaway_entry_critical_mode"],
        },
        "outputs": {
            "panel_C": outputs_c,
            "panel_D": outputs_d,
            "combined_CD": outputs_cd,
        },
        "claim_boundary": payload["claim_boundary"],
    }
    metadata_path = output_dir / "fig5-dual-core-spatial-z-cd-metadata.json"
    _atomic_json(metadata, metadata_path)
    (output_dir / "README.md").write_text(
        "### fig5-panel-c-core-a-bifurcation.png / .pdf / .svg\n\n"
        "Fig.5C 候选。横轴是对称空间路径上的 core-A 失抑制 `D_A=1-Z_A`，纵轴是 core A 内每个 E 神经元的平均发放率。蓝线、红线和深红细线分别是低根、tonic 根和空间 fold chain；粉色线/带是 100 个 OU-on SNN 在 operational onset 的中位数及 q10–q90。\n\n"
        "这不是字面意义上的单个 LIF 神经元分岔：当前确定性模型最细是 2 mm E/I population unit。图中使用 core-A per-neuron mean，是现有证据允许的局部尺度。\n\n"
        "**关注点**：OU-on 中位转变位于低/高根共存区；继续耗竭到 `s=0.337591`，低根才在 saddle-node 消失。\n\n"
        "### fig5-panel-d-spatial-z-phase.png / .pdf / .svg\n\n"
        "Fig.5D 候选。左图把 runaway-entry 的 fixed-point 零模放回 20 mm 双核 sheet；能量的 93.7% 位于 core A。右图固定 `Z_surround=0.80`，分别扫描 `Z_A` 与 `Z_B`，显示仅 tonic 根和低根+tonic 根共存的有限 multi-start catalog。\n\n"
        "**关注点**：入口不是全片同时失稳；任一 core 的 Z 足够低都可移除低根，core A 更早达到边界。右图是 11×11 有限 root catalog，不把未找到的根写成数学不存在。\n\n"
        "### fig5-panels-cd-dual-core-spatial-z.png / .pdf / .svg\n\n"
        "C/D 同行 proof sheet，尺寸比例按 Fig.5 下排版准备。只允许与同一 `dualcore_s39 + Joint=1.25` 底物重画的 A/B 合并；不能直接与旧 `joint_04_control seed1801` A/B 拼成同一实验。\n\n"
        "**关注点**：C 回答局部快系统有什么分支，D 回答临界模式在哪里以及两个 core 是否必须同步。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "out_dir": str(output_dir),
        "metadata": str(metadata_path),
    }, indent=2))


if __name__ == "__main__":
    main()
