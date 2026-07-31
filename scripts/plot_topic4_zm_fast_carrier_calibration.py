#!/usr/bin/env python3
"""Diagnostic figure for the Phase-D baseline-calibration NO-GO."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_fast_carrier_calibration as CAL  # noqa: E402


BASE = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair"
RUN = BASE / "calibration/dynamic_preentry"
FIG = BASE / "figures"


def _load(stem):
    receipt = json.loads((RUN / f"{stem}.json").read_text())
    with np.load(RUN / f"{stem}.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return receipt, arrays


def main():
    verdict = json.loads((BASE / "calibration/calibration_dominance_verdict.json").read_text())
    if verdict["verdict"] != "NO_GO_baseline_calibration_failed_zero_spike_dominance":
        raise RuntimeError("refusing to draw a NO-GO figure from a different verdict")
    reference, ref = _load("reference__noise_replay")
    arms = []
    for scale_i in (0.8, 1.0, 1.2):
        stem = f"sE1.2_sI{scale_i:g}_sM1__noise_replay"
        receipt, arrays = _load(stem)
        arms.append((scale_i, receipt, arrays))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])

    t = np.arange(ref["r_core"].size) * 0.025
    ax0.plot(t, ref["r_core"], color="#333333", lw=1.0, label="current Z/M reference")
    for lo, hi in CAL.event_windows(ref["r_core"]):
        ax0.axvspan(lo * 0.025, hi * 0.025, color="#f2b134", alpha=0.16, lw=0)
    ax0.set_ylabel("core E rate (Hz)")
    ax0.set_title("a  dynamic pre-entry reference", loc="left", fontweight="bold")
    ax0.legend(frameon=False, loc="upper left")
    ax0.text(
        0.99,
        0.95,
        "15 returning events\nmedian peak 68.6 Hz",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        color="#333333",
    )

    ax1.plot(t, ref["r_core"], color="#b8b8b8", lw=0.9, label="reference")
    colors = {0.8: "#f4a261", 1.0: "#e76f51", 1.2: "#b2182b"}
    for scale_i, receipt, arrays in arms:
        ta = np.arange(arrays["r_core"].size) * 0.025
        ax1.plot(ta, arrays["r_core"], lw=1.25, color=colors[scale_i], label=fr"$s_I={scale_i:g}$")
    ax1.set_xlabel("time from canonical initial state (s)")
    ax1.set_ylabel("core E rate (Hz)")
    ax1.set_title("b  maximum-excitation conductance panel", loc="left", fontweight="bold")
    ax1.legend(frameon=False, ncol=4, loc="upper left")
    ax1.text(0.99, 0.90, "all three: exactly 0 E spikes", transform=ax1.transAxes,
             ha="right", va="top", color="#b2182b", fontweight="bold")
    ax1.set_xlim(0, 8.5)

    labels = ["reference"] + [fr"$s_I={value:g}$" for value, _, _ in arms]
    totals = [int(round(float(np.sum(ref["r_all"]) * 32000 * 0.025)))] + [0, 0, 0]
    bars = ax2.bar(np.arange(4), np.asarray(totals) + 1, color=["#555555"] + [colors[x] for x, _, _ in arms])
    ax2.set_yscale("log")
    ax2.set_ylabel("total E spikes + 1 (log)")
    ax2.set_xticks(np.arange(4), labels, rotation=20, ha="right")
    ax2.set_title("c  baseline reachability", loc="left", fontweight="bold")
    for bar, total in zip(bars, totals, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{total:,}", ha="center", va="bottom", fontsize=8)

    x = np.arange(4)
    vinf = [reference["diagnostics"]["median_vinf_mv"]] + [
        receipt["diagnostics"]["median_vinf_mv"] for _, receipt, _ in arms
    ]
    tau = [reference["diagnostics"]["median_tau_eff_ms"]] + [
        receipt["diagnostics"]["median_tau_eff_ms"] for _, receipt, _ in arms
    ]
    ax3.plot(x, vinf, "o-", color="#3a6ea5", label=r"median $V_\infty$")
    ax3.axhline(11.0, color="#3a6ea5", ls="--", lw=0.8, alpha=0.6, label=r"$E_I=11$ mV")
    ax3.set_ylabel(r"$V_\infty$ (mV)", color="#3a6ea5")
    ax3.tick_params(axis="y", labelcolor="#3a6ea5")
    ax3.set_xticks(x, labels, rotation=20, ha="right")
    twin = ax3.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(x, tau, "s-", color="#7b3294", label=r"median $\tau_{eff}$")
    twin.set_ylabel(r"$\tau_{eff}$ (ms)", color="#7b3294")
    twin.tick_params(axis="y", labelcolor="#7b3294")
    ax3.set_title("d  membrane operating point", loc="left", fontweight="bold")
    handles = ax3.get_lines() + twin.get_lines()
    ax3.legend(handles, [item.get_label() for item in handles], frameon=False, fontsize=8,
               loc="upper right")

    fig.suptitle(
        "Phase D baseline calibration: conductance replacement prevents the native IED generator",
        fontsize=12,
        fontweight="bold",
    )
    FIG.mkdir(parents=True, exist_ok=True)
    output = FIG / "phaseD_baseline_calibration_no_go.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    readme = FIG / "README.md"
    readme.write_text(
        "### phaseD_baseline_calibration_no_go.png\n\n"
        "这是一张 Phase D 诊断图，不是 paper-ready ictal lifecycle 图。a 显示原始动态 Z/M "
        "在 onset 前 8.5 s 内保留 15 个返回式间期事件；b 显示最大兴奋尺度下三个抑制尺度均为严格零放电。"
        "c 把总 E spike 数的差异压缩到同一视图，d 显示 conductance 臂把膜工作点压低并缩短有效膜时间常数。\n\n"
        "**关注点**：本轮在 baseline preservation 门即 NO-GO，尚未测试 fast carrier、空间载体、终止或恢复。\n"
    )
    print(output)
    print(verdict["verdict_sha256"])


if __name__ == "__main__":
    main()
