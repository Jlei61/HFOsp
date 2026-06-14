"""Stage 3 局部→全局层级图（探索性，分源；review-corrected 2026-06-14 C1）。
四面板各答一个独立问题：duration / n_part / source_core_ignite_frac / sign。
连续量面板按 source=neg/pos 分面，报 median + bootstrap 95% CI + permutation 效应量。
禁止: 不分源的 "local≈global" 单值；不写 "波前截断"（只 contained / relay-failure）。"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese font: matplotlib indexes the Noto Sans CJK .ttc under the JP family.
# Noto Sans CJK has no italic variant; disable italic globally so CJK text renders.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
RNG = np.random.default_rng(0)

required = {"cell", "seed", "tag", "event_id", "t_on", "source", "sign",
            "n_part", "propagation_class", "source_core_ignite_frac",
            "other_core_ignite_frac", "duration"}


def load():
    d = pd.read_csv(os.path.join(ROOT, "event_types.csv"))
    assert required <= set(d.columns), f"missing cols: {required - set(d.columns)}"
    assert set(d["propagation_class"]) >= {"local", "readable_global"}
    return d


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    meds = [np.median(RNG.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(meds, [2.5, 97.5]))


def perm_effect(a, b, n=5000):
    """median(global) - median(local); two-sided permutation p on the label."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return dict(delta=np.nan, p=np.nan, n_local=len(a), n_global=len(b))
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        d = np.median(pool[na:]) - np.median(pool[:na])
        cnt += abs(d) >= abs(obs)
    return dict(delta=round(float(obs), 4), p=round((cnt + 1) / (n + 1), 4),
                n_local=na, n_global=len(b))


def effect_table(d):
    out = {}
    for metric in ("duration", "n_part", "source_core_ignite_frac"):
        out[metric] = {}
        for src in ("neg", "pos"):
            loc = d[(d.propagation_class == "local") & (d.source == src)][metric].dropna()
            glo = d[(d.propagation_class == "readable_global") & (d.source == src)][metric].dropna()
            out[metric][src] = dict(
                local_median=round(float(np.median(loc)), 4) if len(loc) else None,
                local_ci=[round(float(c), 4) for c in boot_ci(loc)],
                global_median=round(float(np.median(glo)), 4) if len(glo) else None,
                global_ci=[round(float(c), 4) for c in boot_ci(glo)],
                **perm_effect(loc, glo))
    # D: sign vs source among readable_global
    rg = d[d.propagation_class == "readable_global"]
    out["sign_by_source"] = {
        src: dict(forward=int((rg[rg.source == src].sign == 1.0).sum()),
                  reverse=int((rg[rg.source == src].sign == -1.0).sum()))
        for src in ("neg", "pos")}
    return out


# 颜色锁 figure_style_guide：local vs global 用顺序色（viridis 两端）；
# 方向 forward/reverse 用 diverging 红蓝。
_VIRIDIS = plt.get_cmap("viridis")
COL_LOCAL = _VIRIDIS(0.15)    # 顺序色靠前端 = local（小局部）
COL_GLOBAL = _VIRIDIS(0.85)   # 顺序色靠后端 = global（可读全局）
COL_FORWARD = "#b2182b"       # diverging 红 = forward (+1)
COL_REVERSE = "#2166ac"       # diverging 蓝 = reverse (-1)


def _draw_continuous_panel(ax, d, metric, ylabel, eff, title):
    """A/B/C 分源连续量面板：neg/pos 两分面，各画 local vs global box+散点，
    median 标 95% CI 误差棒，面板内文字注 Δ/p。"""
    sources = ["neg", "pos"]
    classes = [("local", COL_LOCAL), ("readable_global", COL_GLOBAL)]
    width = 0.32
    # x 位置：每个 source 一组，组内 local/global 两个位置
    positions = {("neg", "local"): 0.0, ("neg", "readable_global"): width + 0.04,
                 ("pos", "local"): 1.1, ("pos", "readable_global"): 1.1 + width + 0.04}
    for src in sources:
        for cls, col in classes:
            x0 = positions[(src, cls)]
            vals = d[(d.propagation_class == cls) & (d.source == src)][metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            # box（无离群点标记）
            bp = ax.boxplot(vals, positions=[x0], widths=width, patch_artist=True,
                            showfliers=False, manage_ticks=False, zorder=2)
            for patch in bp["boxes"]:
                patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
            for el in ("whiskers", "caps", "medians"):
                for ln in bp[el]:
                    ln.set_color(col)
            # 抖动散点
            jit = (RNG.random(len(vals)) - 0.5) * width * 0.7
            ax.scatter(x0 + jit, vals, s=8, color=col, alpha=0.45, zorder=3,
                       edgecolors="none")
            # median + 95% CI 误差棒
            med = float(np.median(vals))
            lo, hi = boot_ci(vals)
            ax.errorbar([x0], [med], yerr=[[med - lo], [hi - med]], fmt="D",
                        color=col, markersize=6, capsize=4, lw=1.6, zorder=4,
                        markeredgecolor="k", markeredgewidth=0.6)
    # x 轴：source 组标签
    ax.set_xticks([(positions[("neg", "local")] + positions[("neg", "readable_global")]) / 2,
                   (positions[("pos", "local")] + positions[("pos", "readable_global")]) / 2])
    ax.set_xticklabels(["负端起 (neg)", "正端起 (pos)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    # 面板内 Δ/p 注释（每 source 一行）
    lines = []
    for src in sources:
        e = eff[metric][src]
        dlt = e["delta"]; p = e["p"]
        lines.append(f"{src}: Δ(global−local)={dlt:+.3f}, p={p:.3f}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))
    ax.grid(axis="y", ls=":", alpha=0.4)


def _draw_sign_panel(ax, eff):
    """D：堆叠条 forward/reverse × source，标计数。"""
    sources = ["neg", "pos"]
    xpos = [0, 1]
    fwd = [eff["sign_by_source"][s]["forward"] for s in sources]
    rev = [eff["sign_by_source"][s]["reverse"] for s in sources]
    bw = 0.55
    b1 = ax.bar(xpos, fwd, bw, color=COL_FORWARD, alpha=0.85, label="正向 (forward)")
    b2 = ax.bar(xpos, rev, bw, bottom=fwd, color=COL_REVERSE, alpha=0.85,
                label="反向 (reverse)")
    # 计数标注
    for i, s in enumerate(sources):
        if fwd[i] > 0:
            ax.text(xpos[i], fwd[i] / 2, str(fwd[i]), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        if rev[i] > 0:
            ax.text(xpos[i], fwd[i] + rev[i] / 2, str(rev[i]), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        ax.text(xpos[i], fwd[i] + rev[i] + 0.6, f"{fwd[i]}/{fwd[i] + rev[i]} 正向",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["负端起 (neg)", "正端起 (pos)"])
    ax.set_ylabel("可读全局事件数")
    ax.set_title("方向读出：起点端→方向？", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(f + r for f, r in zip(fwd, rev)) * 1.25)
    ax.grid(axis="y", ls=":", alpha=0.4)


def make_figure(d, eff, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    _draw_continuous_panel(axes[0, 0], d, "duration", "持续时间 (ms)",
                           eff, "A 持续更久？")
    _draw_continuous_panel(axes[0, 1], d, "n_part", "点亮触点数 (/12)",
                           eff, "B 扩散更广？")
    _draw_continuous_panel(axes[1, 0], d, "source_core_ignite_frac", "源核点火强度",
                           eff, "C 点火能量（分源）")
    _draw_sign_panel(axes[1, 1], eff)

    # 单一共享图例（顺序色 local/global + diverging 方向）
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COL_LOCAL, alpha=0.5, edgecolor=COL_LOCAL, label="小局部 (local)"),
        Patch(facecolor=COL_GLOBAL, alpha=0.5, edgecolor=COL_GLOBAL, label="可读全局 (readable_global)"),
        Patch(facecolor=COL_FORWARD, alpha=0.85, label="正向 (forward, sign=+1)"),
        Patch(facecolor=COL_REVERSE, alpha=0.85, label="反向 (reverse, sign=-1)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=9.5)

    fig.suptitle("两端等强病灶单网络自发事件：小局部 vs 少数可读全局（分源）",
                 fontsize=13, fontweight="bold", y=1.0)
    # C4 边界 caption
    fig.text(0.5, 0.005,
             "区分 local↔global 的是传播（duration + spread），不是点火能量；"
             "contained propagation / relay-failure，未证明波前截断。",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    d = load()
    eff = effect_table(d)

    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out_json = os.path.join(ROOT, "stage3_hierarchy_effect_sizes.json")
    with open(out_json, "w") as f:
        json.dump(eff, f, indent=2, ensure_ascii=False)

    out_png = os.path.join(ROOT, "figures", "stage3_local_global_hierarchy.png")
    make_figure(d, eff, out_png)

    # stdout 打印四组效应量
    for metric in ("duration", "n_part", "source_core_ignite_frac"):
        for src in ("neg", "pos"):
            e = eff[metric][src]
            print(f"[{metric:>24s} | {src}] local_median={e['local_median']} "
                  f"global_median={e['global_median']} delta={e['delta']} p={e['p']} "
                  f"(n_local={e['n_local']}, n_global={e['n_global']})")
    for src in ("neg", "pos"):
        s = eff["sign_by_source"][src]
        print(f"[sign_by_source | {src}] forward={s['forward']} reverse={s['reverse']}")
    print(f"\nwrote: {out_json}")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
