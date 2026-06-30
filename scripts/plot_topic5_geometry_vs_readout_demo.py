"""方法学反例示意图（didactic，非结果图）：相同的电极 rank 分布 / 线性方向拟合，
可对应两种不同的 2D 传播几何 —— 只看 readout 会丢掉「窄通道 vs 宽铺开」这一几何信息。

构造方式（Anscombe 式）：让传播顺序 rank 只依赖沿轴位置 x —— 两种几何共享同一组
(x, rank)，因此 rank 直方图、rank-vs-沿轴位置、线性方向拟合三者严格相同；唯一差别是
横向(y)铺展：
  几何①窄通道：横向范围恒定且小 —— 沿单一走廊传播
  几何②扇形铺开：横向范围随进程变宽 —— 早期聚焦、晚期宽波面

回应合作者「投影到场上没有新信息 / 2D 切片引入假设」的方法学讨论：本图证明几何相对
1D readout 确实是真信息（窄/宽分辨不开 ⟺ readout 相同但传播模式不同）；但它不替 2D
切片背书——若②的横向铺展落在被丢掉的第三维，当前 2D 实现连这个差别都会抹掉，故几何
检验应在 3D 坐标里做。详见 README。

配色/布局遵循 docs/figure_style_guide.md：传播顺序=viridis（深紫=最早 → 黄=最晚），
共享 colorbar，equal aspect。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（仓库 canonical 口径，见 scripts/plot_pr25_split_half_schematic.py）
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/field_dynamics/figures/methods"

XS_COLS = np.array([10, 20, 30, 40, 50, 60], float)   # 沿轴位置 (mm)
N_PER = 5
RANKS_COL = np.linspace(0.0, 1.0, len(XS_COLS))        # 每列传播顺序 (0=early -> 1=late)
XLIM, YLIM, SIG = (0, 70), (-32, 32), 6.0


def build(half_widths):
    X, Y, R = [], [], []
    for xc, rc, hw in zip(XS_COLS, RANKS_COL, half_widths):
        for y in np.linspace(-hw, hw, N_PER):
            X.append(xc); Y.append(y); R.append(rc)
    return np.array(X), np.array(Y), np.array(R)


def dir_fit(x, y, r):
    """线性方向拟合 rank ~ a*x + b*y（与 src.topic5_ictal_field_dynamics.field_gradient 同口径）。"""
    coef, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), r, rcond=None)
    a, b = coef[0], coef[1]
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


def smooth_field(x, y, r, n=160):
    gx = np.linspace(*XLIM, n); gy = np.linspace(*YLIM, n)
    XX, YY = np.meshgrid(gx, gy)
    S = np.zeros_like(XX); W = np.zeros_like(XX)
    s2 = 2 * SIG ** 2
    for xi, yi, ri in zip(x, y, r):
        w = np.exp(-((XX - xi) ** 2 + (YY - yi) ** 2) / s2)
        S += w; W += w * ri
    T = np.where(S > 1e-9, W / S, np.nan)
    return np.where(S >= 0.03 * S.max(), T, np.nan)


def panel_geo(a, x, y, r, fit_ang, title):
    a.imshow(smooth_field(x, y, r), origin="lower", extent=[*XLIM, *YLIM],
             aspect="equal", cmap="viridis", vmin=0, vmax=1, alpha=0.45, zorder=0)
    sc = a.scatter(x, y, c=r, cmap="viridis", vmin=0, vmax=1, s=85,
                   edgecolors="white", linewidths=0.7, zorder=3)
    a.axhline(0, color="0.35", lw=1.0, ls="--", zorder=1)
    cx, cy, L = x.mean(), y.mean(), 22.0
    dx, dy = L * np.cos(np.radians(fit_ang)), L * np.sin(np.radians(fit_ang))
    a.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
               arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.4), zorder=4)
    a.text(cx, 27.5, f"方向拟合 {fit_ang:.0f}°（早→晚）", color="crimson",
           ha="center", va="center", fontsize=10, fontweight="bold")
    a.set_xlim(*XLIM); a.set_ylim(*YLIM); a.set_aspect("equal", adjustable="box")
    a.set_xlabel("沿轴位置 (mm)"); a.set_ylabel("横向位置 (mm)")
    a.set_title(title, fontsize=12)
    return sc


def main():
    xa, ya, ra = build(np.full(len(XS_COLS), 5.0))          # 几何① 窄通道
    xb, yb, rb = build(np.linspace(2.0, 28.0, len(XS_COLS)))  # 几何② 扇形铺开
    ang_a, mag_a = dir_fit(xa, ya, ra)
    ang_b, mag_b = dir_fit(xb, yb, rb)
    print(f"几何① 方向拟合 angle={ang_a:.2f}°, |grad|={mag_a:.5f}")
    print(f"几何② 方向拟合 angle={ang_b:.2f}°, |grad|={mag_b:.5f}")
    print(f"rank 直方图相同? {np.allclose(np.sort(ra), np.sort(rb))} | "
          f"方向拟合相同? {np.isclose(ang_a, ang_b) and np.isclose(mag_a, mag_b)}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    sc = panel_geo(ax[0], xa, ya, ra, ang_a, "几何① 窄通道：沿单一走廊传播")
    panel_geo(ax[1], xb, yb, rb, ang_b, "几何② 扇形铺开：早期聚焦、晚期宽波面")

    ax[2].scatter(xa, ra, c=ra, cmap="viridis", vmin=0, vmax=1, s=160,
                  edgecolors="0.2", linewidths=0.8, zorder=2, label="几何①")
    ax[2].scatter(xb, rb, facecolors="none", edgecolors="crimson", s=90,
                  linewidths=1.6, marker="X", zorder=3, label="几何②")
    ax[2].set_xlim(*XLIM); ax[2].set_ylim(-0.08, 1.12)
    ax[2].set_xlabel("沿轴位置 (mm)"); ax[2].set_ylabel("传播顺序 (0=早 → 1=晚)")
    ax[2].set_title("折叠成电极 readout：①②完全重合", fontsize=12)
    ax[2].legend(loc="upper left", fontsize=10, frameon=False)
    ax[2].text(0.5, 0.04,
               f"rank 直方图：①≡②\n方向拟合：{ang_a:.0f}° = {ang_b:.0f}°，|梯度| 相同",
               transform=ax[2].transAxes, ha="center", va="bottom", fontsize=10,
               bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))

    cb = fig.colorbar(sc, ax=ax[:2], fraction=0.025, pad=0.01)
    cb.set_label("传播顺序 (0=early → 1=late)")
    fig.suptitle("反例：相同的电极 rank 分布 / 线性方向拟合，对应两种不同的 2D 传播几何\n"
                 "只保留 rank 沿轴分布 + 方向拟合，会丢掉「窄通道 vs 宽铺开」这一几何信息",
                 fontsize=13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "geometry_vs_rank_readout_counterexample.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
