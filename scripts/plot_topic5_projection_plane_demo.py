"""方法学示意图（didactic）：如何从真实 3D 电极布局定出 2D 投影平面。

平面不是随手切：u = 传播轴（把每触点间期传播顺序 along_axis_mm 对 3D 坐标最小二乘回归得到的方向）；
v = 去掉 u 分量后触点云的第一主成分（横向）；w = u×v 垂直方向被丢弃。用真实被试的 u/v（display frame）。
配色遵循 docs/figure_style_guide.md（viridis 顺序）。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/field_dynamics/figures/methods"
GEO_DIR = {"narrow": _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects",
           "broad": _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"}


def load(subject, substrate):
    from scripts.plot_contact_plane_static import _subject_display_frame, _attach_real_coords, _coord_array
    ta = json.load(open(GEO_DIR[substrate] / f"{subject}_t_a.json"))
    tb = json.load(open(GEO_DIR[substrate] / f"{subject}_t_b.json"))
    _attach_real_coords([ta, tb])
    C, ok = _coord_array(ta)
    r = np.array([c.get("typical_rank", np.nan) for c in ta["channels"]], float)
    keep = ok & np.isfinite(r)
    C, r = C[keep], r[keep]
    fr = _subject_display_frame([ta, tb])
    u, v = np.asarray(fr["u"], float), np.asarray(fr["v"], float)
    v01 = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.zeros_like(r)
    return C - C.mean(0), u, v, v01, float(fr["sigma_mm"])


def field2d(x, y, val, sig, xlim, ylim, n=200):
    gx = np.linspace(*xlim, n); gy = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(gx, gy); S = np.zeros_like(XX); W = np.zeros_like(XX); s2 = 2 * sig ** 2
    for xi, yi, vi in zip(x, y, val):
        w = np.exp(-((XX - xi) ** 2 + (YY - yi) ** 2) / s2); S += w; W += w * vi
    T = np.where(S > 1e-9, W / S, np.nan)
    return np.where(S >= 0.03 * S.max(), T, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--substrate", choices=list(GEO_DIR), default="narrow")
    args = ap.parse_args()
    C, u, v, val, sig = load(args.subject, args.substrate)
    w = np.cross(u, v)
    au, av, aw = C @ u, C @ v, C @ w
    eu, ev, ew = np.ptp(au), np.ptp(av), np.ptp(aw)

    fig = plt.figure(figsize=(15.0, 6.0), constrained_layout=True)
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axB = fig.add_subplot(1, 2, 2)

    # --- ① 3D 云 + 定平面 ---
    axA.scatter(C[:, 0], C[:, 1], C[:, 2], c=val, cmap="viridis", vmin=0, vmax=1,
                s=55, edgecolors="white", linewidths=0.6, depthshade=False)
    # (u,v) 平面（半透明平行四边形）
    pu, pv = eu / 2 * 1.05, ev / 2 * 1.05
    corners = np.array([pu * u + pv * v, pu * u - pv * v, -pu * u - pv * v, -pu * u + pv * v])
    axA.add_collection3d(Poly3DCollection([corners], alpha=0.13, facecolor="0.5", edgecolor="0.5"))
    O = np.zeros(3); wlen = 0.5 * (eu + ev) / 2
    utip, vtip, wtip = u * eu * 0.5, v * ev * 0.5, w * wlen
    axA.quiver(*O, *utip, color="crimson", lw=2.5, arrow_length_ratio=0.12)
    axA.quiver(*O, *vtip, color="royalblue", lw=2.5, arrow_length_ratio=0.14)
    axA.quiver(*O, *wtip, color="0.35", lw=2.0, linestyle="--", arrow_length_ratio=0.18)
    axA.text(*(utip * 1.12), "u = 传播轴", color="crimson", fontsize=11, fontweight="bold")
    axA.text(*(vtip * 1.15 + w * wlen * 0.25), "v = 横向主成分", color="royalblue", fontsize=11, fontweight="bold")
    axA.text(*(wtip * 1.18), f"w 丢弃（本例仅 {ew:.1f}mm）", color="0.3", fontsize=10)
    # 收紧 3D 范围到所有绘制元素的包围盒（去掉空盒子）
    allp = np.vstack([C, corners, O, utip, vtip, wtip])
    for setter, i in [(axA.set_xlim3d, 0), (axA.set_ylim3d, 1), (axA.set_zlim3d, 2)]:
        lo, hi = allp[:, i].min(), allp[:, i].max(); pad = 0.08 * (hi - lo + 1e-9); setter(lo - pad, hi + pad)
    axA.set_box_aspect((np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), np.ptp(allp[:, 2])))
    axA.set_xticks([]); axA.set_yticks([]); axA.set_zticks([])
    axA.set_title("① 真实 3D 电极 → 定平面 (u, v)，丢弃 w", fontsize=13)
    axA.view_init(elev=20, azim=-62)

    # --- ② 投影到 (u,v) ---
    x, y = au, av
    m = 2 * sig; xlim = (x.min() - m, x.max() + m); ylim = (y.min() - m, y.max() + m)
    axB.imshow(field2d(x, y, val, sig, xlim, ylim), origin="lower", extent=[*xlim, *ylim],
               aspect="equal", cmap="viridis", vmin=0, vmax=1, zorder=0)
    sc = axB.scatter(x, y, c=val, cmap="viridis", vmin=0, vmax=1, s=70,
                     edgecolors="white", linewidths=0.8, zorder=3)
    axB.set_xlim(*xlim); axB.set_ylim(*ylim); axB.set_aspect("equal", adjustable="box")
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_xlabel("u →", fontsize=11); axB.set_ylabel("v →", fontsize=11)
    axB.set_title("② 投影到 (u, v) 平面 = 场的画布", fontsize=13)

    fig.text(0.505, 0.5, "投影 →", ha="center", va="center", fontsize=15, color="crimson", fontweight="bold")
    cb = fig.colorbar(sc, ax=axB, fraction=0.045, pad=0.02)
    cb.set_label("间期传播顺序（0=早 → 1=晚）")
    fig.suptitle(f"如何从真实 3D 电极布局定 2D 投影平面 — {args.subject}"
                 f"（u=传播轴 / v=横向主成分 / w={ew:.1f}mm 丢弃）", fontsize=13.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"field_projection_plane_{args.subject}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out, f"| extent u/v/w = {eu:.1f}/{ev:.1f}/{ew:.1f} mm")


if __name__ == "__main__":
    main()
