"""方法学示意图（didactic）：从离散电极 forward 投影到连续场 F(x,y)。

投影方向：forward（电极 → 场，Nadaraya-Watson 归一化加权平均，同 R_smooth_rank 口径），
不是把场"回采样"到电极。少字三段式。配色/自包含遵循 docs/figure_style_guide.md（viridis）。

`--subject epilepsiae_1146 --substrate narrow`：用该被试**真实电极布局**（3D→2D display frame）
+ 间期传播顺序 typical_rank 作为电极值；不给 --subject 则用合成布局。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def synthetic_layout():
    rng = np.random.default_rng(4); P = []
    for x0, y0 in [(6, -16), (14, -5), (24, 6), (32, 17)]:
        for k in range(5):
            P.append((x0 + 9 * k + rng.normal(0, 0.5), y0 + rng.normal(0, 1.0)))
    P = np.array(P); x, y = P[:, 0], P[:, 1]
    v = (x - x.min()) / (x.max() - x.min())
    return x, y, v, 6.0, (-2, 62), (-26, 26), "合成布局"


def real_layout(subject, substrate):
    from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                                   _attach_real_coords)
    ta = json.load(open(GEO_DIR[substrate] / f"{subject}_t_a.json"))
    tb = json.load(open(GEO_DIR[substrate] / f"{subject}_t_b.json"))
    _attach_real_coords([ta, tb])
    frame = _subject_display_frame([ta, tb])
    x, y = _display_points(ta, frame)
    r = np.array([c.get("typical_rank", np.nan) for c in ta["channels"]], float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x, y, r = x[ok], y[ok], r[ok]
    v = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.zeros_like(r)
    return x, y, v, float(frame["sigma_mm"]), frame["xlim"], frame["ylim"], f"{subject} 真实布局"


def _central_rep(x, y, xlim, ylim):
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    return int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))   # 最居中的电极（核不出框）


def main():
    from scripts.plot_contact_plane_static import _smooth_rank_field_mm   # 真实 field 画法
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--substrate", choices=list(GEO_DIR), default="narrow")
    args = ap.parse_args()
    if args.subject:
        x, y, v, sig, xlim, ylim, tag = real_layout(args.subject, args.substrate)
        stem = f"electrode_to_field_projection_{args.subject}"
    else:
        x, y, v, sig, xlim, ylim, tag = synthetic_layout()
        stem = "electrode_to_field_projection"
    x, y, v = np.asarray(x), np.asarray(y), np.asarray(v)
    # 用真实 field 的 xlim/ylim（不加边距），图幅按数据长宽比 → 三格 equal-aspect 铺满、边框/标题不歪
    W, H = xlim[1] - xlim[0], ylim[1] - ylim[0]
    pw = 4.6; fig = plt.figure(figsize=(3 * pw + 1.3, pw * (H / W) + 1.2), layout="constrained")
    ax = fig.subplots(1, 3)

    sc = ax[0].scatter(x, y, c=v, cmap="viridis", vmin=0, vmax=1, s=110,
                       edgecolors="white", linewidths=1.0, zorder=3)
    ax[0].set_title("① 离散电极（采样点）", fontsize=13)

    ax[1].scatter(x, y, c=v, cmap="viridis", vmin=0, vmax=1, s=80,
                  edgecolors="white", linewidths=0.8, zorder=3)
    idx = _central_rep(x, y, xlim, ylim)
    for r in (sig, 2 * sig):
        ax[1].add_patch(plt.Circle((x[idx], y[idx]), r, fill=False,
                                    ec="crimson", lw=1.4, ls="--", alpha=0.9, zorder=2))
    ax[1].annotate("高斯核 σ / 2σ", (x[idx] + sig * 0.7, y[idx] + sig * 0.7),
                   (x[idx] + 2 * sig + 1.5, y[idx] + 2 * sig + 1.5), color="crimson", fontsize=10,
                   ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))
    ax[1].set_title("② 各电极按高斯核加权", fontsize=13)

    # ③ 完全按真实 field 画（同一 _smooth_rank_field_mm：真实 σ、grid、mask）
    _, _, T, _, _ = _smooth_rank_field_mm(x, y, v, np.ones_like(v), xlim, ylim, sig)
    ax[2].imshow(T, origin="lower", extent=[*xlim, *ylim], aspect="equal",
                 cmap="viridis", vmin=0, vmax=1, zorder=0)
    ax[2].scatter(x, y, c="none", edgecolors="0.35", s=16, linewidths=0.5, zorder=3)
    ax[2].set_title("③ 连续场 F(x, y)", fontsize=13)

    for a in ax:
        a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_aspect("equal", adjustable="box")
        a.set_xticks([]); a.set_yticks([])
    for x0 in (0.325, 0.635):
        fig.text(x0, 0.5, "投影 →", ha="center", va="center", fontsize=15,
                 color="crimson", fontweight="bold")

    cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("间期传播顺序（0=早 → 1=晚）" if args.subject else "值（viridis）")
    fig.suptitle(f"从电极投影到场（forward：离散电极 → 连续场 F）— {tag}", fontsize=14)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out, f"| n_contacts={len(x)} sigma={sig:.1f} range={W:.0f}x{H:.0f}mm")


if __name__ == "__main__":
    main()
