"""方法学示意图（didactic，非结果图）：场（空间汇集）相对逐电极读出的优势。

回应合作者「投影到场上没有新信息」。诚实定位（与我们的科学结论一致）：
  - 场的优势 = 用分辨率换信噪比。相邻电极共享同一个底层场值，把同一组电极（含侧向电极）
    空间汇集，可把每通道独立的噪声平均掉，于是一个「弱而粗」的传播结构被去噪恢复 → 显著。
  - 方向（梯度角度）更细，但平滑救不了它：仿真里平滑前后角度都甩 ~35-40°（见 print），
    即"方向更细节但不显著"。所以我们 lead with 场，而非方向。
  - 代价：场会模糊掉细结构。这是"方向其实更细"的来源，图注里如实标注，不做强 argue。

构造：固定一组 2D 电极（4 杆 × 6 触点、杆间错开 = 有侧向覆盖，两种读法共用同一组电极）；
真实场 = 沿轴的弱平滑梯度；一次"发作" = 真实 + 各通道独立噪声。逐电极 = 直接看带噪值；
场 = 对同一组电极做高斯空间汇集（Nadaraya-Watson）。与真实结构的相关 r 量化恢复优劣。

配色/布局遵循 docs/figure_style_guide.md：传播顺序 = viridis（深紫=最早 → 黄=最晚），
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
NOISE, KSIG = 0.6, 9.0
XLIM, YLIM = (-4, 50), (-26, 26)


def electrodes(rng):
    """固定 2D 电极布局：4 根杆、杆间错开 → 含侧向电极。"""
    P = []
    for x0, y0 in [(2, -18), (10, -6), (20, 6), (28, 18)]:
        for k in range(6):
            P.append((x0 + 8 * k + rng.normal(0, 0.5), y0 + rng.normal(0, 1.0)))
    return np.array(P)


def smooth_at(P, v, sig):
    x, y = P[:, 0], P[:, 1]
    D2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    W = np.exp(-D2 / (2 * sig ** 2)); W /= W.sum(1, keepdims=True)
    return W @ v


def grid_field(P, v, sig, n=170):
    gx = np.linspace(*XLIM, n); gy = np.linspace(*YLIM, n)
    XX, YY = np.meshgrid(gx, gy)
    x, y = P[:, 0], P[:, 1]
    S = np.zeros_like(XX); WT = np.zeros_like(XX)
    s2 = 2 * sig ** 2
    for xi, yi, vi in zip(x, y, v):
        w = np.exp(-((XX - xi) ** 2 + (YY - yi) ** 2) / s2)
        S += w; WT += w * vi
    T = np.where(S > 1e-9, WT / S, np.nan)
    return np.where(S >= 0.03 * S.max(), T, np.nan)


def panel(ax, P, color_vals, bg_field, title, sub):
    if bg_field is not None:
        ax.imshow(bg_field, origin="lower", extent=[*XLIM, *YLIM], aspect="equal",
                  cmap="viridis", vmin=0, vmax=1, alpha=0.5, zorder=0)
    sc = ax.scatter(P[:, 0], P[:, 1], c=color_vals, cmap="viridis", vmin=0, vmax=1,
                    s=95, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("沿轴位置 (mm)"); ax.set_ylabel("横向位置 (mm)")
    ax.set_title(title, fontsize=12)
    if sub:
        ax.text(0.5, 0.025, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5, bbox=dict(boxstyle="round", fc="0.96", ec="0.6"))
    return sc


def angle(P, v):
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(v))])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return np.degrees(np.arctan2(c[1], c[0]))


def main():
    rng = np.random.default_rng(3)
    P = electrodes(rng)
    x = P[:, 0]
    f = x - x.mean(); f = (f - f.min()) / (f.max() - f.min())   # 真实弱场 0..1

    v = np.clip(f + rng.normal(0, NOISE, len(f)), 0, 1)          # 一次发作（展示用）
    vf = smooth_at(P, v, KSIG)
    vf = (vf - vf.min()) / (vf.max() - vf.min() + 1e-9)
    r_raw = np.corrcoef(v, f)[0, 1]
    r_fld = np.corrcoef(vf, f)[0, 1]

    # 蒙特卡洛：多次发作里 场 vs 逐电极 的结构恢复 + 方向稳定性
    M = 400; cr = []; cf = []; ar = []; af = []
    for _ in range(M):
        vv = f + rng.normal(0, NOISE, len(f)); vvf = smooth_at(P, vv, KSIG)
        cr.append(np.corrcoef(vv, f)[0, 1]); cf.append(np.corrcoef(vvf, f)[0, 1])
        ar.append(angle(P, vv)); af.append(angle(P, vvf))
    frac = np.mean(np.array(cf) > np.array(cr))
    wstd = lambda a: np.std((np.array(a) + 90) % 180 - 90)
    print(f"展示realization: r_raw={r_raw:.2f}  r_field={r_fld:.2f}")
    print(f"MC(M={M}): corr-to-truth raw {np.mean(cr):.2f}  field {np.mean(cf):.2f}  | "
          f"field>raw 在 {frac:.0%} 的发作里")
    print(f"MC 方向角散度: raw {wstd(ar):.0f}°  field {wstd(af):.0f}°  (平滑救不了方向)")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
    panel(ax[0], P, f, grid_field(P, f, KSIG),
          "① 假设的真实传播结构（仿真：弱、空间平滑）", None)
    panel(ax[1], P, v, None,
          "② 逐电极读出（真实 + 各通道独立噪声）", f"与真实结构相关 r = {r_raw:.2f}")
    sc = panel(ax[2], P, vf, grid_field(P, v, KSIG),
               "③ 同一组电极汇集成场（空间去噪）", f"与真实结构相关 r = {r_fld:.2f}")

    cb = fig.colorbar(sc, ax=ax, fraction=0.020, pad=0.01)
    cb.set_label("传播顺序 (0=early → 1=late)")
    fig.suptitle(
        "场的优势：同一组电极（含侧向电极），逐个读被噪声淹没；空间汇集成场把「弱而粗」的传播结构去噪恢复\n"
        f"（结构恢复 r：逐电极 {np.mean(cr):.2f} → 场 {np.mean(cf):.2f}，场更优在 {frac:.0%} 的发作里。"
        f"方向更细但两种读法里都甩 ~{wstd(ar):.0f}° → 不显著；故 lead with 场，不强 argue 方向）",
        fontsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "field_advantage_spatial_pooling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
