#!/usr/bin/env python3
"""Figures for early-to-late template axes and shared-plane field analysis."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT = ROOT / "results/topic5_ictal_recruitment/template_axis_field"
COMMON = ROOT / "results/topic5_ictal_recruitment/template_axis_field_common_support"
HFA = ROOT / "results/topic5_ictal_recruitment/template_axis_field_hfa"
FIG = RESULT / "figures"

COLORS = {"different": "#8a8f98", "reversed": "#2b7a78", "same": "#d59a2e"}


def _bool(v):
    return str(v).lower() == "true"


def _save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_axis_cohort():
    rows = list(csv.DictReader((RESULT / "axis_cohort.csv").open()))
    ok = [r for r in rows if r.get("status") == "ok"]
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), gridspec_kw={"width_ratios": [1.7, 1, 1]})

    ax = axes[0]
    for i, r in enumerate(ok):
        c = float(r["cos_uA_uB"])
        qc = _bool(r["axis_pair_qc_pass"])
        y = (0 if r["dataset"] == "epilepsiae" else 1) + rng.uniform(-0.16, 0.16)
        if qc:
            ax.scatter(c, y, s=55, marker="o", c=COLORS[r["relation"]],
                       edgecolor="white", linewidth=0.7, alpha=0.95, zorder=3)
        else:
            ax.scatter(c, y, s=34, marker="x", color=COLORS[r["relation"]],
                       linewidth=0.9, alpha=0.95, zorder=3)
    ax.axvspan(-1, -0.5, color=COLORS["reversed"], alpha=0.08)
    ax.axvspan(0.5, 1, color=COLORS["same"], alpha=0.10)
    ax.axvline(-0.5, color="0.55", ls="--", lw=0.9)
    ax.axvline(0.5, color="0.55", ls="--", lw=0.9)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set(xlim=(-1.03, 1.03), yticks=[0, 1],
           yticklabels=["Epilepsiae", "Yuquan"], xlabel=r"Signed axis cosine  $u_A^T u_B$")
    ax.set_title("A  Spatial relation of A/B propagation axes", loc="left", fontsize=11, fontweight="bold")
    ax.text(-0.75, 1.35, "reversed line", ha="center", fontsize=8, color=COLORS["reversed"])
    ax.text(0, 1.35, "different", ha="center", fontsize=8, color="0.4")
    ax.text(0.75, 1.35, "same line", ha="center", fontsize=8, color=COLORS["same"])
    ax.text(-0.98, -0.38, "○ strict-stability subset   × direction still estimated", fontsize=7.5, color="0.35")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    labels = ["different", "reversed", "same"]
    all_counts = [sum(r["relation"] == x for r in ok) for x in labels]
    geometry_rows = [r for r in ok if _bool(r.get("geometry_2d_supported"))]
    strict_rows = [r for r in ok if _bool(r.get("strict_stability_pass"))]
    geometry_counts = [sum(r["relation"] == x for r in geometry_rows) for x in labels]
    strict_counts = [sum(r["relation"] == x for r in strict_rows) for x in labels]
    # Three evidence tiers: every estimable pair has two directions; geometry/stability
    # are annotations, not axis-existence gates.
    bottom = np.zeros(3)
    for lab, vals in zip(labels, zip(all_counts, geometry_counts, strict_counts)):
        ax.bar([0, 1, 2], vals, bottom=bottom, color=COLORS[lab], width=0.62, label=lab)
        bottom += np.asarray(vals)
    ax.set(xticks=[0, 1, 2], xticklabels=[f"Defined\nn={len(ok)}",
                                          f"2D geometry\nn={len(geometry_rows)}",
                                          f"Strict stable\nn={len(strict_rows)}"],
           ylabel="Patients")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("B  Relation counts", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    stages = ["Rank-displacement", "stable k=2", "A/B axes defined", "2D geometry"]
    values = [len(rows), sum(r.get("stable_k") == "2" for r in rows), len(ok), len(geometry_rows)]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, values, color=["#c7cbd1", "#aeb5be", "#73808f", "#2b7a78"], height=0.58)
    for yy, v in zip(y, values):
        ax.text(v + 0.7, yy, str(v), va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=stages, xlim=(0, max(values) * 1.18), xlabel="Patients")
    ax.set_title("C  Structural attrition", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Early-to-late template axes reveal a collinear A/B subset", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "axis_collinearity_cohort")


def _metric(subject, name):
    d = json.loads((RESULT / "per_subject" / f"{subject}.json").read_text())
    return d["field"]["statistics"]["nulls"]["within_shaft"]["metrics"][name]


def plot_field_summary():
    bb = json.loads((RESULT / "cohort_summary.json").read_text())
    common = json.loads((COMMON / "cohort_summary.json").read_text())
    hfa = json.loads((HFA / "cohort_summary.json").read_text())
    primary_subset = "shared_2d_geometry_60deg"
    subjects = bb["metrics"][primary_subset]["within_shaft"]["shared_maxab"]["subjects"]
    own = np.asarray([_metric(s, "own_maxab")["obs_subject"] for s in subjects])
    shared = np.asarray([_metric(s, "shared_maxab")["obs_subject"] for s in subjects])
    own_margin = np.asarray([_metric(s, "own_maxab")["margin_vs_null_median"] for s in subjects])
    shared_margin = np.asarray([_metric(s, "shared_maxab")["margin_vs_null_median"] for s in subjects])

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), gridspec_kw={"width_ratios": [1, 1.6, 1.15]})
    ax = axes[0]
    for i, s in enumerate(subjects):
        ax.plot([0, 1], [own[i], shared[i]], color="0.72", lw=1)
        ax.scatter([0, 1], [own[i], shared[i]], c=["#6f879c", "#2b7a78"], s=34, zorder=3)
    ax.plot([0, 1], [np.median(own), np.median(shared)], color="black", lw=2.2, marker="o", ms=5)
    ax.set(xticks=[0, 1], xticklabels=["Own planes", "Shared plane"], ylim=(0, 1.03),
           ylabel="Early-ictal field similarity |r|")
    ax.set_title("A  Same patients, same fields", loc="left", fontsize=11, fontweight="bold")
    paired = bb["paired_shared_vs_own"][primary_subset]["within_shaft"]
    ax.text(0.5, 0.04, f"paired median Δ = {paired['median_shared_minus_own']:+.3f}\n"
                       f"Wilcoxon p = {paired['wilcoxon_p_two_sided']:.3f}", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = ["own_a_abs", "own_b_abs", "own_maxab", "shared_a_abs", "shared_b_abs", "shared_maxab"]
    labels = ["Own A", "Own B", "Own maxAB", "Shared A", "Shared B", "Shared maxAB"]
    vals = []
    for name in names:
        v = bb["metrics"][primary_subset]["within_shaft"][name]
        vals.append((v["obs_median"], v["null_median"], v["null_p95"], v["p_upper"]))
    x = np.arange(len(vals))
    ax.scatter(x, [v[0] for v in vals], color=["#6f879c"] * 3 + ["#2b7a78"] * 3,
               s=58, zorder=3, label="observed median")
    ax.scatter(x, [v[1] for v in vals], color="0.65", marker="_", s=150, linewidths=2,
               label="null median")
    for xx, (_, med, p95, p) in zip(x, vals):
        ax.vlines(xx, med, p95, color="0.65", lw=1.5)
        ax.hlines(p95, xx - 0.12, xx + 0.12, color="0.5", lw=1)
        ax.text(xx, 1.015, f"p={p:.3f}" if p >= 0.001 else "p<.001", rotation=55,
                ha="left", va="top", fontsize=7)
    ax.axvline(2.5, color="0.85", lw=1)
    ax.set(xticks=x, xticklabels=labels, ylim=(0.35, 1.05), ylabel="Cohort median |r|")
    ax.tick_params(axis="x", rotation=28)
    ax.set_title("B  2D collinear subset vs within-shaft null (n=7)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    sensitivity = [
        ("2D 60°\nBB", bb, "shared_2d_geometry_60deg"),
        ("strict stable\nBB", bb, "shared_strict_stability_60deg"),
        ("45°\nBB", bb, "shared_45deg"),
        ("bootstrap-stable\nBB", bb, "shared_robust_pair_bootstrap"),
        ("2D 60°\ncommon support", common, "shared_2d_geometry_60deg"),
        ("2D 60°\nHFA", hfa, "shared_2d_geometry_60deg"),
    ]
    for i, (lab, d, subset) in enumerate(sensitivity):
        v = d["metrics"][subset]["within_shaft"]["shared_maxab"]
        n = d["denominators"][subset]
        ax.scatter(i, v["p_upper"], s=48, color="#2b7a78" if v["p_upper"] < .05 else "#9ca3aa")
        ax.text(i, min(v["p_upper"] * 1.45, 0.85), f"n={n}", ha="center", fontsize=7)
    ax.axhline(0.05, color="#b44", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set(xticks=np.arange(len(sensitivity)), xticklabels=[x[0] for x in sensitivity],
           ylabel="Cohort null p (upper-tail)", ylim=(5e-4, 1.2))
    ax.tick_params(axis="x", rotation=28, labelsize=7.5)
    ax.set_title("C  Sensitivity and sample-size boundary", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Shared-plane broadband evidence depends on the axis-stability tier", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "own_vs_shared_ictal_field")


def _smooth_grid(points, values, support, n=100):
    p = np.asarray(points, float)
    v = np.asarray(values, float)
    s = np.asarray(support, float)
    pad = 0.12
    xs = np.linspace(p[:, 0].min() - pad, p[:, 0].max() + pad, n)
    ys = np.linspace(p[:, 1].min() - pad, p[:, 1].max() + pad, n)
    gx, gy = np.meshgrid(xs, ys)
    nn = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    np.fill_diagonal(nn, np.inf)
    sigma = float(np.median(nn.min(1)))
    d2 = (gx[..., None] - p[:, 0]) ** 2 + (gy[..., None] - p[:, 1]) ** 2
    w = s[None, None, :] * np.exp(-d2 / (2 * sigma ** 2))
    field = (w * v[None, None, :]).sum(2) / np.maximum(w.sum(2), 1e-12)
    return gx, gy, field


def plot_shared_examples():
    subjects = ["epilepsiae_1084", "epilepsiae_548"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    for row, sid in enumerate(subjects):
        d = json.loads((RESULT / "per_subject" / f"{sid}.json").read_text())
        f = d["field"]
        pts = np.asarray(f["planes"]["shared"]["points"], float)
        ea, eb, sz = map(np.asarray, (f["earliness_a"], f["earliness_b"], f["seizure_mean"]))
        sa, sb = map(np.asarray, (f["support_a"], f["support_b"]))
        items = [(ea, sa, "Template A earliness"), (eb, sb, "Template B earliness"),
                 (sz, (sa + sb) / 2, "Mean early-ictal energy")]
        vmax_tpl = max(np.nanmax(np.abs(ea)), np.nanmax(np.abs(eb)))
        for col, (vals, sup, title) in enumerate(items):
            ax = axes[row, col]
            gx, gy, fld = _smooth_grid(pts, vals, sup)
            if col < 2:
                im = ax.pcolormesh(gx, gy, fld, shading="auto", cmap="coolwarm",
                                   vmin=-vmax_tpl, vmax=vmax_tpl)
            else:
                im = ax.pcolormesh(gx, gy, fld, shading="auto", cmap="viridis")
            ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap="coolwarm" if col < 2 else "viridis",
                       edgecolor="black", linewidth=0.55, s=34,
                       vmin=-vmax_tpl if col < 2 else None, vmax=vmax_tpl if col < 2 else None)
            ax.axhline(0, color="white", alpha=.35, lw=.6)
            ax.arrow(pts[:, 0].min(), pts[:, 1].min(), 0.18, 0, color="black",
                     width=.002, head_width=.025, length_includes_head=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("shared propagation axis")
            if col == 0:
                rel = d["axis_pair"]["relation"]
                ax.set_ylabel(f"{sid.replace('epilepsiae_', 'E')}\n{rel['relation']}, cos={rel['cosine']:.2f}\ntransverse")
            else:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    fig.suptitle("A and B fields represented on one interictally defined 2D plane", fontsize=13)
    fig.tight_layout()
    _save(fig, "shared_plane_examples")


def write_readme():
    text = """# 图说明

### axis_collinearity_cohort.png

全 rank-displacement cohort 的 A/B 单模板传播轴关系。轴由 `u_T=-gradient(-z(rank_T))/||gradient||` 定义，正向统一为早→晚。左图把线是否共线（`|cos|`）和传播方向同向/反向（signed cosine）分开；28 名可建轴患者均显示方向，圆点只标 strict-stability 高置信层，叉号也已有可估计方向。中、右图分别给出全部可建轴、二维几何和严格稳定性层的关系计数，以及从 40 名输入到 28 名可建双轴、26 名具备二维几何的结构分母。

**关注点**：28 名全部都有 A/B 两个方向并进入结构分布；26 名具备二维几何。13 名 strict-stability 是高置信敏感性层，不是“是否有方向”的门槛。

### own_vs_shared_ictal_field.png

对有 ictal cache、二维几何且共线的 7 名患者，比较每个模板各自平面与统一共享平面的早期发作场读出。统计先按发作计算、再按患者折叠；null 在每根电极杆内洗牌，并在每次洗牌后重新选择 maxAB。右图同时给出 strict-stability、角度、pair-bootstrap、support 和 HFA 敏感性。

**关注点**：二维几何主分母 n=7 的 broadband shared field 未超过 within-shaft null；阳性只出现在 strict-stability n=6 等子集。患者内观测相关也没有高于 own planes，不能写成共享平面已有稳定增益。

### shared_plane_examples.png

一名反向和一名同向患者的共享平面示例。A、B 和早期发作能量都使用同一条仅由间期 A/B 梯度决定的角平分轴；颜色场为显示用途，正式统计在触点位置计算。

**关注点**：共享投影的价值首先是让共线 A/B 模式处在同一坐标系中，而不是保证相关系数数值上升。
"""
    (FIG / "README.md").write_text(text)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_axis_cohort()
    plot_field_summary()
    plot_shared_examples()
    write_readme()
    print(f"wrote figures and {FIG / 'README.md'}")


if __name__ == "__main__":
    main()
