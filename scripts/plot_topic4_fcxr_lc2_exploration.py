#!/usr/bin/env python3
"""Diagnostic figures for FCXR-LC2 closed-loop exploration."""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core", "closed_loop_exploration")
FIG = os.path.join(OUT, "figures")
RUST = "#A35E48"
BLUE = "#2166AC"
ORANGE = "#D97706"
GREY = "#6B7280"


def _json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_r1_characterization():
    d = _json("r1_resegmentation_summary.json")
    z = np.load(os.path.join(OUT, "r1_sensor_support_map.npz"))
    gap = d["segmentation"]["gap"]
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)

    # The trajectory segmentation is displayed explicitly; labels come from the rate+LFP test.
    a = ax[0]
    a.broken_barh([(1500, 1246), (3912, 67)], (0.58, 0.25), facecolors=RUST, label="active bout")
    a.broken_barh([(2846, 966)], (0.15, 0.25), facecolors="#D1D5DB", label="rest-like gap")
    a.set_xlim(1400, 4550); a.set_ylim(0, 1)
    a.set_yticks([]); a.set_xlabel("HEO2 time (ms)")
    a.set_title("a  active bouts and intervening gap", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    a.text(0.02, 0.93, f"gap rate={gap['gap_rate50_median_hz']:.3f} Hz\n"
                        f"interictal q95={gap['rate_ref_interevent_q95_hz']:.1f} Hz",
           transform=a.transAxes, va="top", fontsize=8)

    a = ax[1]
    lfp = np.asarray(gap["lfp_delta_db_per_contact"], float)
    colors = np.where(np.abs(lfp) <= 3.0, BLUE, RUST)
    a.bar(np.arange(1, lfp.size + 1), lfp, color=colors, width=0.75)
    a.axhspan(-3, 3, color="#DBEAFE", alpha=0.45, zorder=0)
    a.axhline(0, color="black", lw=0.7)
    a.set_xlabel("virtual contact"); a.set_ylabel("gap - interictal (dB)")
    a.set_title("b  virtual-SEEG returns to low band", loc="left", fontsize=10)
    a.text(0.98, 0.95, f"{gap['contacts_within_3db']}/{lfp.size} within ±3 dB",
           transform=a.transAxes, ha="right", va="top", fontsize=8)

    a = ax[2]
    pos = np.asarray(z["pos_E"], float)
    support = np.asarray(z["recruited_support"], bool)
    a.scatter(pos[~support, 0], pos[~support, 1], s=3, c="#D1D5DB", alpha=0.35, linewidths=0)
    a.scatter(pos[support, 0], pos[support, 1], s=5, c=ORANGE, alpha=0.75, linewidths=0)
    a.set_aspect("equal"); a.set_xlim(0, 20); a.set_ylim(0, 20)
    a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    a.set_title("c  recurrent-drive support", loc="left", fontsize=10)
    a.text(0.02, 0.98, f"{d['support']['n']}/{d['support']['sample_n']} sampled E cells\n"
                        f"{d['support']['occupied_spatial_blocks']}/16 blocks",
           transform=a.transAxes, va="top", fontsize=8)
    _save(fig, "r1_sensor_characterization")


def plot_r1_pareto():
    with open(os.path.join(OUT, "r1_sensor_pareto.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r["false_latch_fraction"]) for r in rows])
    y = np.array([float(r["heo2_active_support_duty"]) for r in rows])
    c = np.array([float(r["heo1_support_duty"]) for r in rows])
    g = np.array([float(r["gap_persistence"]) for r in rows])
    tau = np.array([float(r["tau_ms"]) for r in rows])
    pareto = np.array([r["pareto"] == "True" for r in rows])
    selected = _json("r1_resegmentation_summary.json")["selected_candidates"]

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.45), constrained_layout=True)
    sizes = 20 + 110 * g / max(np.max(g), 1e-12)
    sc = ax[0].scatter(x, y, c=c, s=sizes, cmap="viridis", vmin=0, vmax=1,
                       alpha=0.65, edgecolor=np.where(pareto, "black", "none"), linewidth=0.7)
    for q in selected:
        ax[0].scatter(q["false_latch_fraction"], q["heo2_active_support_duty"], marker="*",
                      s=120, color=RUST, edgecolor="white", linewidth=0.6, zorder=4)
    ax[0].set_xlabel("baseline false-latch fraction")
    ax[0].set_ylabel("HEO2 active-bout support duty")
    ax[0].set_title("a  Pareto sensor trade-off", loc="left", fontsize=10)
    cb = fig.colorbar(sc, ax=ax[0], pad=0.02)
    cb.set_label("HEO1 support duty")

    for pf in sorted(set(x)):
        m = x == pf
        order = np.argsort(tau[m])
        ax[1].plot(tau[m][order], y[m][order], marker="o", ms=3, lw=1,
                   label=f"false latch={pf:.3f}")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\tau_H$ (ms)")
    ax[1].set_ylabel("HEO2 active-bout support duty")
    ax[1].set_title("b  memory-duration trade-off", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, "r1_sensor_pareto")


def plot_screen():
    path = os.path.join(OUT, "h_loop_screen.json")
    if not os.path.isfile(path):
        return False
    d = _json("h_loop_screen.json")
    rows = d["rows"]
    ids = sorted({r["candidate_id"] for r in rows})
    fig, ax = plt.subplots(2, 3, figsize=(10.4, 6.0), constrained_layout=True, sharex=True, sharey=True)
    label_color = dict(screen_survivor=RUST, unresolved_1s=ORANGE, decay_low=BLUE,
                       saturated_tonic="#7C3AED", numerical_failure="black")
    for a, cid in zip(ax.flat, ids):
        rr = [r for r in rows if r["candidate_id"] == cid]
        for r in rr:
            a.scatter(r["rho_fraction"], r["k_ratio"], s=25 + min(r["tail_rate_hz"], 250) * 0.35,
                      color=label_color[r["label"]], alpha=0.8, edgecolor="white", linewidth=0.4)
        a.set_title(f"{cid}: tau={rr[0]['tau_ms']:.0f} ms", fontsize=9)
        a.set_xticks([.1, .2, .35, .5, .7]); a.set_yticks([.05, .1, .2])
    for a in ax[-1]: a.set_xlabel(r"$\rho_H/g_{sat}$")
    for a in ax[:, 0]: a.set_ylabel(r"$k_H/\theta_H$")
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in label_color.items()]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.02), fontsize=8)
    _save(fig, "h_loop_screen")
    return True


def plot_forks():
    path = os.path.join(OUT, "frozen_fork_map.json")
    if not os.path.isfile(path):
        return False
    d = _json("frozen_fork_map.json")
    rows = d["rows"]
    cands = sorted({r["candidate_run_id"] for r in rows})
    arms = ["A_low", "A_high", "B", "C", "D1", "D2"]
    rate = np.full((len(cands), len(arms)), np.nan)
    h = np.full_like(rate, np.nan)
    for r in rows:
        i, j = cands.index(r["candidate_run_id"]), arms.index(r["arm"])
        rate[i, j] = r["state_tail_1s"]["rate_mean_hz"]
        h[i, j] = r["state_tail_1s"]["h_mean"] / r["theta"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    for a, mat, title, lab in ((ax[0], rate, "a  tail activity", "rate (Hz)"),
                               (ax[1], h, "b  normalized H state", r"$H/\theta_H$")):
        im = a.imshow(mat, aspect="auto", cmap="magma")
        a.set_xticks(range(len(arms)), arms, rotation=35, ha="right")
        a.set_yticks(range(len(cands)), cands)
        a.set_title(title, loc="left", fontsize=10)
        fig.colorbar(im, ax=a, pad=0.02, label=lab)
    _save(fig, "frozen_fork_map")
    return True


def write_readme(screen=False, forks=False):
    text = """### r1_sensor_characterization.png

这张图把 HEO2 旧参考轨迹拆成两个活动段和中间的 rest-like gap，并同时给出虚拟 SEEG 与局部 recurrent-drive support。它说明旧的完整窗口不能被当成一段连续高态，但活动段仍提供了闭环 H 的候选传感范围。

**关注点**：中间 gap 的低率与 15/15 触点回到 ±3 dB，而 recruited support 只占采样 E 细胞的一部分。

### r1_sensor_pareto.png

这张图展示 baseline false latch、HEO1 持续支持、HEO2 活动段支持和长间隙残留之间的折中。红星是按预注册角色选择的六个候选；这些点只进入闭环筛查，不代表已经存在高态 basin。

**关注点**：不存在一个同时桥接完整长 gap 且对所有间期事件零误锁的单一时间常数。
"""
    if screen:
        text += """
### h_loop_screen.png

六个 H 传感候选各自扫描平滑宽度与反馈强度。点的颜色是 1 s 开发标签，大小近似尾段活动率；`screen_survivor` 只表示值得延长，不能解释成双稳态或发作 carrier。

**关注点**：看 survivor 是否形成局部参数窗口，以及高反馈是否转成 saturated tonic 或数值失败。
"""
    if forks:
        text += """
### frozen_fork_map.png

这张图并列 healthy low/high 初值、susceptible low/high 初值与两个冻结 X 负荷。只有 A-low/A-high/B 回低、C 保持有限高态、D1 或 D2 回低，才构成 frozen H/X geometry candidate。

**关注点**：C 与 B 是否真正分离，以及 X 是否消灭高态而不是把所有条件一起压低。
"""
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "README.md"), "w") as f:
        f.write(text.strip() + "\n")


def main():
    plot_r1_characterization()
    plot_r1_pareto()
    has_screen = plot_screen()
    has_forks = plot_forks()
    write_readme(has_screen, has_forks)


if __name__ == "__main__":
    main()
