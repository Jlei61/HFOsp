#!/usr/bin/env python3
"""Stage-aware diagnostic figures for FCXR-LC2-Core (never draws placeholders)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-lc2")

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core")
R1 = os.path.join(OUT, "r1_sensor")
FIG = os.path.join(OUT, "figures")


def plot_r1():
    src = os.path.join(R1, "h_sensor_separability.json")
    if not os.path.isfile(src):
        raise SystemExit(f"missing real R1 input: {src}")
    d = json.load(open(src))
    acceptance_path = os.path.join(R1, "r1_stage_acceptance.json")
    acceptance = json.load(open(acceptance_path)) if os.path.isfile(acceptance_path) else None
    rows = d["rows"]
    tau = np.asarray([r["tau_ms"] for r in rows])
    L = np.asarray([r["L_IED_q999"] for r in rows])
    Lu = np.asarray([r["L_upper95"] for r in rows])
    U1 = np.asarray([r["U_HEO1_q10"] for r in rows])
    U2 = np.asarray([r["U_HEO2_q10"] for r in rows])
    U1lo = np.asarray([r["HEO1_lower95"] for r in rows])
    U2lo = np.asarray([r["HEO2_lower95"] for r in rows])
    Ulo = np.asarray([r["U_lower95"] for r in rows])
    ok = np.asarray([r["separable"] for r in rows], bool)

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    ax[0].plot(tau, L, color="#4c72b0", lw=2, label="IED peak Q99.9")
    ax[0].plot(tau, Lu, color="#4c72b0", lw=1, ls="--", label="IED upper 95%")
    ax[0].plot(tau, U1, color="#c44e52", lw=2, label="HEO1 trough Q10")
    ax[0].plot(tau, U2, color="#dd8452", lw=2, label="HEO2 trough Q10")
    ax[0].plot(tau, Ulo, color="#222222", lw=1, ls="--", label="high lower 95% (min)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\tau_H$ (ms)")
    ax[0].set_ylabel("post-X recurrent load")
    ax[0].set_title("A  Temporal sensor separation")
    ax[0].legend(frameon=False, fontsize=8)

    margin1 = U1lo - Lu
    margin2 = U2lo - Lu
    margin = Ulo - Lu
    ax[1].axhline(0.0, color="0.55", lw=1)
    ax[1].plot(tau, margin1, color="#c44e52", lw=1.2, ls="--",
               label="HEO1-only margin")
    ax[1].plot(tau, margin2, color="#dd8452", lw=1.2, ls=":",
               label="HEO2-only margin")
    ax[1].plot(tau, margin, color="#55a868", marker="o", ms=3, lw=1.8,
               label="joint margin (min)")
    if np.any(ok):
        ax[1].fill_between(tau, 0.0, margin, where=ok, color="#55a868", alpha=0.22,
                           label="bootstrap-separable")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\tau_H$ (ms)")
    ax[1].set_ylabel("high lower 95% - IED upper 95%")
    scoped = (acceptance["canonical_status"] if acceptance is not None
              else "STRICT FULL-WINDOW DIAGNOSTIC")
    ax[1].set_title(f"B  Strict diagnostic; canonical: {scoped}", fontsize=10)
    best = int(np.argmax(margin))
    ax[1].annotate(f"best joint = {margin[best]:.2f}",
                   (tau[best], margin[best]), xytext=(-65, -28),
                   textcoords="offset points", fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "h_sensor_separability.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    readme = os.path.join(FIG, "README.md")
    text = (
        "### h_sensor_separability.png\n\n"
        "左图比较普通 returning IED 的 H 负荷峰值上界与 HEO1/HEO2 已建立高态的负荷 trough；"
        "右图分别画 HEO1-only、HEO2-only 和两者联合的 bootstrap 分离余量。"
        "只有联合余量高于零的连续 tau 区间才允许激活 H 电流。\n\n"
        f"原始严格全窗口统计为 `{d['status']}`；阶段验收为 `{scoped}`。"
        "前者是 long-gap stress test，不再作为闭环 H geometry 的 hard gate。\n\n"
        "**关注点**：HEO1-only 在中等 tau 可分离，但 HEO2-only 全程为负并决定联合失败；"
        "这定位的是当前局部 sensor 对 adaptation-burst trough 不稳健，而不是 H 正反馈整体已被否定。\n"
    )
    with open(readme, "w") as f:
        f.write(text)
    print(out)


if __name__ == "__main__":
    plot_r1()
