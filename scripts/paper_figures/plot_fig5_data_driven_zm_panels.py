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
    REFERENCE_FIGDATA,
    _accepted_display_xy,
    _load_accepted_display,
    _load_npz,
    _panel_label,
    _plot_energy,
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
    parser.add_argument("--extent-mm", type=float, default=12.0)
    parser.add_argument("--reference-figdata", default=REFERENCE_FIGDATA)
    args = parser.parse_args()

    replay_path = ROOT / args.replay
    replay = _load_npz(replay_path)
    metadata = json.loads(replay_path.with_suffix(".json").read_text())
    if not metadata["verification_against_archived_run"]["all_match"]:
        raise RuntimeError("the Figure 5 replay does not match the archived run")
    morphology = _require_sustained_runaway(metadata)
    config = json.loads((ROOT / args.config).read_text())
    onset_ms = float(metadata["model_ictal_onset_ms"])
    extent = (-float(args.extent_mm), float(args.extent_mm))
    display = _load_accepted_display(replay, ROOT / args.reference_figdata)
    positions = _accepted_display_xy(replay["positions_E"], display)
    contacts = _accepted_display_xy(replay["contact_xy_mm"], display)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 3.5))
    _, rate_ax = _plot_readout(ax, replay, onset_ms, morphology)
    _panel_label(rate_ax, "A", x=-0.055, y=1.35)
    _save(fig, out_dir / "fig5-panel-a-readout.png")

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    _plot_trajectory(ax, replay, onset_ms, config["zm"]["eta_m"])
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
    energy_map = _plot_energy(
        ax, replay, replay["positions_E"], replay["contact_xy_mm"],
        positions, contacts, extent)
    colorbar = fig.colorbar(energy_map, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"activity energy ($\times 10^3$ Hz$^2$)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    _save(fig, out_dir / "fig5-panel-c-early-energy.png")

    (out_dir / "README.md").write_text(
        "### fig5-panel-a-readout.png\n"
        "同一条 Joint 轨迹的 15 触点连续有符号 30-80 Hz model-current readout。"
        "蓝色窗口是按冻结规则选择并经患者模板审计确认为 Model TB 的发作前事件，"
        "红色虚线是 runaway 进入时刻；顶部灰线是群体 E 发放率。\n\n"
        "**关注点**：runaway 由群体发放率持续升高定义；30-80 Hz 接触振幅本身只小幅改变，"
        "每触点尺度冻结在进入前。\n\n"
        "### fig5-panel-b-zm-trajectory.png\n"
        "同一条轨迹在 h 加权的失抑制 D=1-z 与适应 A=eta_m*m 平面上的投影。三角、空心圆和红点分别标记样本事件、进入前 500 ms 和进入时刻。\n\n"
        "**关注点**：这是轨迹投影，不是解析相图或分离曲线。\n\n"
        "### fig5-panel-c-event-order.png\n"
        "冻结规则选择的 Model TB 发作前事件。神经元和触点颜色都表示相对首次放电顺序，"
        "并刚体注册到 Fig4 已验收的 E1146 平面。\n\n"
        "**关注点**：触点方向与 Fig4 完全同合同，数字标签 0 的 MTB 语义来自患者模板审计。\n\n"
        "### fig5-panel-c-early-energy.png\n"
        "进入后 100 ms 的逐神经元 spike-rate-squared 场，显示单位为 10^3 Hz^2；"
        "触点颜色由同一个场用固定 0.75 mm 核采样。\n\n"
        "**关注点**：背景与触点使用同一活动量，不混用 current energy。\n",
        encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "panels": 4}))


if __name__ == "__main__":
    main()
