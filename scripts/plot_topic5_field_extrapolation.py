"""诊断图：间期 broad 顺序场 + 隐身电极 predicted vs 发作 z-ER 招募序。

左：归一化平面上间期顺序场热图 (低=早=源)，叠核心触点(圈)+隐身电极(方块)。
右：散点 —— 场预测(F) / 自身 broad rank(C) 对发作 z-ER 序；标 F/C/radial。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.propagation_contact_plane_readout import make_plane_grid, smooth_field
from src.topic5_field_extrapolation import (
    load_broad_axis_record, channel_names_from_pool, broad_minus_narrow,
    DEF_BROAD_POOL, DEF_NARROW_POOL,
)

OUT = Path("results/topic5_ictal_recruitment/field_extrapolation/figures")


def plot_subject(ds_sid):
    res = json.load(open(Path("results/topic5_ictal_recruitment/field_extrapolation/per_subject") / f"{ds_sid}.json"))
    det = res["detail"]
    rec = load_broad_axis_record(ds_sid)
    by = {c["name"]: c for c in rec["channels"]}
    hidden = set(broad_minus_narrow(channel_names_from_pool(ds_sid, DEF_BROAD_POOL),
                                    channel_names_from_pool(ds_sid, DEF_NARROW_POOL)))

    X, Y = make_plane_grid()
    fld = smooth_field(rec, X, Y, scalar="rank")
    T = np.where(fld["mask"], fld["T"], np.nan)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))

    im = ax0.pcolormesh(X, Y, T, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax0, label="interictal recruitment-order field (low=early=source)")
    for c in rec["channels"]:
        if not (np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])):
            continue
        if c["name"] in hidden:
            ax0.scatter(c["x_norm"], c["y_norm"], marker="s", s=70, facecolors="none",
                        edgecolors="crimson", linewidths=1.6, zorder=5)
        else:
            ax0.scatter(c["x_norm"], c["y_norm"], marker="o", s=28, facecolors="none",
                        edgecolors="white", linewidths=1.0, zorder=4)
    ax0.set_title(f"{ds_sid}  interictal broad order field\ncircle=core(narrow)  square=hidden(broad-minus-narrow)")
    ax0.set_xlabel("x_norm (along axis)")
    ax0.set_ylabel("y_norm (transverse)")

    pred = np.array(det["predicted"]); own = np.array(det["own_rank"]); ict = np.array(det["ictal"])
    ax1.scatter(pred, ict, c="crimson", s=55, label=f"field-pred F (LOO)  rho={res['F']:.2f}", zorder=5)
    ax1.scatter(own, ict, c="gray", s=40, marker="x", label=f"own broad rank C  rho={res['C']:.2f}", zorder=4)
    ax1.set_xlabel("predicted interictal order (low=early)")
    ax1.set_ylabel("ictal z-ER recruit order r_sz (low=early)")
    rl = res.get("ictal_reliability", {})
    ax1.set_title(f"hidden n={res['n_hidden_eval']} | radial rho={res['radial_baseline']:.2f} "
                  f"| F_p={res['F_p_value']:.2f}\nictal: {rl.get('health')} s_sz={rl.get('s_sz'):.2f} "
                  f"| verdict: {res['verdict']}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ds_sid}_field_extrapolation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()
    for sid in args.subjects:
        plot_subject(sid)


if __name__ == "__main__":
    main()
