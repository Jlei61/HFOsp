#!/usr/bin/env python3
"""Render the completed Figure 5 A-C panels before Panel D is available."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig5_data_driven_zm_main import (
    _load_npz,
    _panel_label,
    _plot_runaway_energy,
    _plot_event_order,
    _plot_readout,
    _plot_trajectory,
    _require_sustained_runaway,
)


def _save(fig, path):
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=(
        "config/topic4_data_driven_zm_ictal_transition_v1.json"))
    parser.add_argument("--replay", default=(
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition/fig5_replay/"
        "joint_04_control_seed_1801_frames.npz"))
    parser.add_argument("--out-dir", default=(
        "results/paper-ready-figure/fig5/figures/panels"))
    parser.add_argument("--sheet-size-mm", type=float, default=20.0)
    parser.add_argument("--display-onset-offset-ms", type=float, default=300.0)
    args = parser.parse_args()

    replay_path = ROOT / args.replay
    replay = _load_npz(replay_path)
    metadata = json.loads(replay_path.with_suffix(".json").read_text())
    verification = metadata.get(
        "verification_against_reference_run",
        metadata.get("verification_against_archived_run"),
    )
    if not verification or not verification["all_match"]:
        raise RuntimeError("the Figure 5 replay does not match the archived run")
    morphology = _require_sustained_runaway(
        metadata, allow_exploratory_workpoint=True)
    config = json.loads((ROOT / args.config).read_text())
    morphology_onset_ms = float(metadata.get(
        "morphology_onset_ms", metadata["model_ictal_onset_ms"]))
    display_onset_ms = morphology_onset_ms + float(args.display_onset_offset_ms)
    display_window_start_ms = max(0.0, morphology_onset_ms - 1200.0)
    extent = (0.0, float(args.sheet_size_mm))
    positions = replay["positions_E"]
    contacts = replay["contact_xy_mm"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 3.5))
    _, rate_ax = _plot_readout(
        ax, replay, display_onset_ms, morphology,
        window_start_ms=display_window_start_ms)
    _panel_label(rate_ax, "A", x=-0.055, y=1.35)
    _save(fig, out_dir / "fig5-panel-a-readout.png")

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    eta_m = metadata.get("workpoint_parameters", config["zm"])["eta_m"]
    _plot_trajectory(ax, replay, display_onset_ms, eta_m)
    _panel_label(ax, "B", x=-0.23, y=1.12)
    _save(fig, out_dir / "fig5-panel-b-zm-trajectory.png")

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    rank_map = _plot_event_order(ax, replay, positions, contacts, extent)
    _panel_label(ax, "C", x=-0.24, y=1.13)
    colorbar = fig.colorbar(rank_map, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("contact order (0 = first)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    _save(fig, out_dir / "fig5-panel-c-event-order.png")

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    energy_map, _ = _plot_runaway_energy(
        ax, replay, contacts, extent, display_onset_ms,
        float(metadata["activity_window_ms"]), start_offset_ms=0.0)
    colorbar = fig.colorbar(energy_map, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"activity energy ($\times 10^3$ Hz$^2$)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    _save(fig, out_dir / "fig5-panel-c-runaway-energy.png")

    (out_dir / "README.md").write_text(
        "### fig5-panel-a-readout.png\n"
        "同一条 Joint 轨迹的 15 触点连续有符号 30-80 Hz model-current readout。"
        "蓝色窗口是按冻结规则选择并经患者模板审计确认为 Model TB 的发作前事件，"
        "红色虚线是全局持续高活动开始的展示时刻；顶部两条线是全局神经元与空间招募比例。\n\n"
        "**关注点**：runaway 由群体发放率持续升高定义；30-80 Hz 接触振幅本身只小幅改变，"
        "每触点尺度冻结在进入前。\n\n"
        "### fig5-panel-b-zm-trajectory.png\n"
        "同一条轨迹在 h 加权的失抑制 D=1-z 与适应 A=eta_m*m 平面上的投影。三角、空心圆和红点分别标记样本事件、进入前 500 ms 和进入时刻。\n\n"
        "**关注点**：这是轨迹投影，不是解析相图或分离曲线。\n\n"
        "### fig5-panel-c-event-order.png\n"
        "冻结规则选择的一次真实间期事件。每一个点都是一个 SNN E 神经元，颜色表示该神经元"
        "在这次事件里的真实首次放电时间；触点颜色表示同一次事件的 contact order，"
        "使用和 Fig4 完全一致的原始 0-20 mm sheet x/y 坐标。\n\n"
        "**关注点**：没有 onset 分箱或平均场；亮暗先后来自逐神经元 spike。\n\n"
        "### fig5-panel-c-runaway-energy.png\n"
        "红线后 0 至 +100 ms 的全场活动能量图，定义为局部 E 神经元 100 ms 平均"
        "发放率的平方，单位为 10^3 Hz^2，使用与患者发作场图一致的蓝色色标；"
        "红线对应原形态检测后 300 ms、连续读出中约 1500 ms 的全局高活动开始段。\n\n"
        "**关注点**：看 runaway 时刻全场能量招募，不把普通发放率误作能量。\n",
        encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "panels": 4}))


if __name__ == "__main__":
    main()
