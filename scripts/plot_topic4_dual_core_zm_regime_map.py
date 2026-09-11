#!/usr/bin/env python3
"""Plot the native spatial-Z x dynamic-M regime and basin audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DEFAULT_JSON = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/zm_regime_map/"
    "native_spatial_z_dynamic_m_tauGABA9.json")
DEFAULT_EXTENSION_JSON = DEFAULT_JSON.with_name(
    "native_spatial_z_dynamic_m_extension_m4_8_tauGABA9.json")
FOLD_S_ETA0 = 0.33759138010148526


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle,
                 values[-1] + (values[-1] - middle[-1])]


def _label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.08, label, transform=axis.transAxes,
              fontsize=15, fontweight="bold", ha="left", va="top")


def _basin_code(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low_recruited = low >= 2
    high_recruited = high >= 2
    code = np.full(low.shape, 3, np.int8)
    code[(low == 0) & (high == 0)] = 0
    code[(low == 0) & high_recruited] = 1
    code[low_recruited & high_recruited] = 2
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION_JSON)
    args = parser.parse_args()
    result_path = args.result.resolve()
    meta = json.loads(result_path.read_text(encoding="utf-8"))
    data_path = result_path.with_suffix(".npz")
    data = np.load(data_path)
    s = np.asarray(data["core_disinhibition_s"], float)
    gain = np.asarray(data["adaptation_gain_scale"], float)
    low = np.asarray(data["low_initial__state_code"], int)
    high = np.asarray(data["high_initial__state_code"], int)
    low_mean_all = np.asarray(data["low_initial__mean_rate_hz"], float)
    high_mean_all = np.asarray(data["high_initial__mean_rate_hz"], float)
    source_paths = [result_path, data_path]
    extension_path = args.extension.resolve()
    if extension_path.exists():
        extension_data_path = extension_path.with_suffix(".npz")
        extension = np.load(extension_data_path)
        np.testing.assert_allclose(
            np.asarray(extension["core_disinhibition_s"], float), s)
        gain = np.r_[gain, np.asarray(extension["adaptation_gain_scale"], float)]
        low = np.concatenate(
            [low, np.asarray(extension["low_initial__state_code"], int)], axis=1)
        high = np.concatenate(
            [high, np.asarray(extension["high_initial__state_code"], int)], axis=1)
        low_mean_all = np.concatenate(
            [low_mean_all,
             np.asarray(extension["low_initial__mean_rate_hz"], float)], axis=1)
        high_mean_all = np.concatenate(
            [high_mean_all,
             np.asarray(extension["high_initial__mean_rate_hz"], float)], axis=1)
        order = np.argsort(gain)
        gain = gain[order]
        low, high = low[:, order], high[:, order]
        low_mean_all, high_mean_all = (
            low_mean_all[:, order], high_mean_all[:, order])
        source_paths.extend([extension_path, extension_data_path])
    basin = _basin_code(low, high)
    time_s = np.asarray(data["time_ms"], float) / 1000.0

    fig = plt.figure(figsize=(11.7, 7.2))
    grid = fig.add_gridspec(2, 2, left=0.075, right=0.96, bottom=0.09,
                            top=0.90, wspace=0.30, hspace=0.40)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        "Native delayed dual-core regime map: frozen spatial Z, dynamic M",
        fontsize=13, fontweight="bold", y=0.965)

    state_colors = ["#E8EBEF", "#77A7C7", "#D96B5C", "#76519B"]
    state_cmap = ListedColormap(state_colors)
    state_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    panels = (
        (ax_a, low, "low / nearest-pre-fold initial condition"),
        (ax_b, high, "recruited-basin initial condition"),
    )
    image = None
    for axis, values, title in panels:
        image = axis.pcolormesh(
            _edges(s), _edges(gain), values.T, cmap=state_cmap,
            norm=state_norm, shading="flat")
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xlabel(r"core disinhibition $s$  ($Z_{A,B}=1-s$)")
        axis.set_ylabel(r"adaptation gain / rev21 $\eta_m$")
        if np.max(gain) > 3.0:
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_yticks(gain)
            axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            axis.set_ylim(0.0, _edges(gain)[-1])
    bar = fig.colorbar(image, ax=[ax_a, ax_b], ticks=[0, 1, 2, 3],
                       fraction=0.035, pad=0.025)
    bar.ax.set_yticklabels(["low", "localized", "tonic", "oscillatory"],
                           fontsize=7)
    _label(ax_a, "A")
    _label(ax_b, "B")

    basin_colors = ["#DDE7F2", "#6B4C93", "#C84D4D", "#D7B65D"]
    basin_cmap = ListedColormap(basin_colors)
    basin_image = ax_c.pcolormesh(
        _edges(s), _edges(gain), basin.T, cmap=basin_cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), shading="flat")
    basin_bar = fig.colorbar(
        basin_image, ax=ax_c, ticks=[0, 1, 2, 3], fraction=0.05)
    basin_bar.ax.set_yticklabels(
        ["both low", "low/high coexist", "both recruited", "mixed/localized"],
        fontsize=7)
    ax_c.set_title("outcome coexistence (two standardized initial states)",
                   fontsize=10, fontweight="bold")
    ax_c.set_xlabel(r"core disinhibition $s$")
    ax_c.set_ylabel(r"adaptation gain / rev21 $\eta_m$")
    if np.max(gain) > 3.0:
        ax_c.set_yscale("symlog", linthresh=1.0)
        ax_c.set_yticks(gain)
        ax_c.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_c.set_ylim(0.0, _edges(gain)[-1])
    _label(ax_c, "C")

    gain_index = int(np.argmin(np.abs(gain - 1.0)))
    low_mean = low_mean_all[:, gain_index]
    high_mean = high_mean_all[:, gain_index]
    ax_d.plot(s, low_mean, "o-", color="#2F5D95", ms=3.5, lw=1.0,
              label="low initial")
    ax_d.plot(s, high_mean, "o-", color="#8A3E77", ms=3.5, lw=1.0,
              label="recruited initial")
    ax_d.axhline(120.0, color="#C7254E", lw=0.8, ls=":",
                 label="global-rate gate")
    ax_d.axvline(FOLD_S_ETA0, color="0.25", lw=0.9, ls="--",
                 label=r"$\eta_m=0$ low-fold reference")
    ax_d.set_yscale("symlog", linthresh=1.0)
    ax_d.set_ylim(bottom=0.0)
    ax_d.set_xlabel(r"core disinhibition $s$")
    ax_d.set_ylabel("tail population E rate (Hz)")
    ax_d.set_title(r"rev21 M gain section ($\tau_M=500$ ms)",
                   fontsize=10, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=7, loc="best")
    _label(ax_d, "D")

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    out = result_path.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "dualcore-native-spatial-z-dynamic-m-regime-audit"
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(fig)

    metadata = {
        "status": "NATIVE_SPATIAL_Z_DYNAMIC_M_REGIME_AUDIT",
        "sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "counts": {
            branch: {
                label: int(np.sum(values == code))
                for label, code in (
                    ("low", 0), ("intermediate", 1),
                    ("tonic_recruited", 2),
                    ("oscillatory_recruited", 3))
            }
            for branch, values in (
                ("low_initial", low), ("high_initial", high))
        },
        "basin_code_counts": {
            label: int(np.sum(basin == code))
            for label, code in (
                ("both_low", 0), ("low_high_coexistence", 1),
                ("both_recruited", 2), ("mixed_or_localized", 3))
        },
        "eta_m_zero_fold_reference_s": FOLD_S_ETA0,
        "outputs": outputs,
        "panel_semantics": {
            "A": "native-dt outcome from the low fixed point, or nearest pre-fold state where that root is absent",
            "B": "native-dt outcome from the standardized recruited initial basin",
            "C": "coexistence or common-outcome classification from A and B",
            "D": "rev21 adaptation-gain section with the eta_m=0 fold shown only as a reference",
        },
        "claim_boundary": meta["claim_boundary"],
    }
    _atomic(stem.with_suffix(".metadata.json"),
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    readme = out / "README.md"
    _atomic(
        readme,
        "### dualcore-native-spatial-z-dynamic-m-regime-audit.png\n\n"
        "这张图在原生 0.1-ms 步长下，把双 core 的冻结空间 Z 场与动态 M 适应反馈组成二维 regime/basin 图。A 从低 fixed point 出发；当该根已在 fold 后消失时，改用紧邻 fold 的标准化低态作为初值。B 从 recruited 初态出发，C 明确标出两种初态是否收敛到不同状态，D 给出 rev21 M 强度截面；虚线只是 eta_m=0 时已验证的 low-fold 参考，不被外推成整条二维分岔边界。\n\n"
        "**关注点**：这是有限的 deterministic coarse-system regime map，不是热力学相图，也不是含 OU 的 full-SNN 转变概率图。\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
