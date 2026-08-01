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
    rows = d["rows"]
    tau = np.asarray([r["tau_ms"] for r in rows])
    L = np.asarray([r["L_IED_q999"] for r in rows])
    Lu = np.asarray([r["L_upper95"] for r in rows])
    U1 = np.asarray([r["U_HEO1_q10"] for r in rows])
    U2 = np.asarray([r["U_HEO2_q10"] for r in rows])
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

    margin = Ulo - Lu
    ax[1].axhline(0.0, color="0.55", lw=1)
    ax[1].plot(tau, margin, color="#55a868", marker="o", ms=3, lw=1.6)
    if np.any(ok):
        ax[1].fill_between(tau, 0.0, margin, where=ok, color="#55a868", alpha=0.22,
                           label="bootstrap-separable")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\tau_H$ (ms)")
    ax[1].set_ylabel("high lower 95% - IED upper 95%")
    ax[1].set_title(f"B  Gate margin: {d['status']}")
    if np.any(ok):
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
        "右图直接画 bootstrap 后的分离余量。只有右图高于零的连续 tau 区间才允许激活 H 电流。\n\n"
        f"本次判决为 `{d['status']}`；图只回答时间传感器是否可分离，不代表已经得到双稳态或发作生命周期。\n\n"
        "**关注点**：看 bootstrap 余量是否稳定高于零，而不是只看点估计曲线是否交叉。\n"
    )
    with open(readme, "w") as f:
        f.write(text)
    print(out)


if __name__ == "__main__":
    plot_r1()
