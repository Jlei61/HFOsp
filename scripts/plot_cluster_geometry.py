#!/usr/bin/env python3
"""Topic 1 cluster geometry visualization — figures.

Per-subject 3-panel + cohort 2x2 + showcase 3 figures, all under the
template-matching metric (see archive plan §3.3 / §5).

Design doc: docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (  # noqa: E402
    COL_EPILEPSIAE,
    COL_NEUTRAL,
    COL_NIGHT,
    COL_NONSIG,
    COL_SIG,
    COL_YUQUAN,
    DPI_PUB,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    add_significance_bracket,
    dataset_color,
    new_figure,
    savefig_pub,
    style_panel,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("plot_cluster_geometry")


GEOMETRY_DIR = Path("results/interictal_propagation/cluster_geometry")
PER_SUBJECT_DIR = GEOMETRY_DIR / "per_subject"
FIGURES_DIR = GEOMETRY_DIR / "figures"
PER_SUBJECT_FIG_DIR = FIGURES_DIR / "per_subject"
SHOWCASE_FIG_DIR = FIGURES_DIR / "showcase"


# Cluster colors: distinct, Morandi-flavored
CLUSTER_PALETTE = [
    "#6F8FA8",  # Morandi blue
    "#A35E48",  # Morandi rust
    "#9DAA90",  # Morandi sage
    "#C9A86A",  # Morandi mustard
    "#7E6E84",  # Morandi plum
    "#B07A6E",  # Morandi terracotta
]
TEMPLATE_MARKER_SIZE = 360  # large stars
EVENT_MARKER_SIZE = 18
BOUNDARY_MARKER_SIZE = 36


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_subject_geometry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_cohort_summary() -> Optional[Dict[str, Any]]:
    p = GEOMETRY_DIR / "cohort_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _list_per_subject_files() -> List[Path]:
    if not PER_SUBJECT_DIR.exists():
        return []
    return sorted(PER_SUBJECT_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Per-subject 3-panel figure
# ---------------------------------------------------------------------------


def plot_per_subject_geometry(
    data: Dict[str, Any],
    output_path: Path,
    is_showcase: bool = False,
) -> Path:
    """Produce the 3-panel per-subject figure described in §5.1.

    Panel a: MDS scatter + boundary events + templates as stars
    Panel b: per-event silhouette ranked bar (cluster-blocked, sil-sorted)
    Panel c: cluster template profile (channel-rank lines + IQR band)
    """
    if data.get("status") != "ok":
        logger.warning("Skip %s: status=%s", data.get("subject"), data.get("status"))
        return output_path

    dataset = data.get("dataset", "")
    subject = data.get("subject", "")
    chosen_k = int(data["chosen_k"])
    events = data["events"]
    templates = data["templates"]
    channel_names = data.get("channel_names", [])
    n_ch = len(channel_names)

    figsize = (18, 5.5) if not is_showcase else (22, 7)
    fig, axes = new_figure(nrows=1, ncols=3, figsize=figsize)

    # -----------------------------------------------------------------------
    # Panel a: MDS scatter + boundary + templates
    # -----------------------------------------------------------------------
    ax = axes[0]
    style_panel(ax, label="a", label_x=-0.16, label_y=1.06)
    ax.set_facecolor("#FAFAFA")

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        agree_pts = [
            (ev["mds_x"], ev["mds_y"])
            for ev in events
            if ev["kmeans_label"] == k
            and ev.get("agreement", True)
            and ev["mds_x"] is not None
            and np.isfinite(ev["mds_x"])
        ]
        bound_pts = [
            (ev["mds_x"], ev["mds_y"])
            for ev in events
            if ev["kmeans_label"] == k
            and not ev.get("agreement", True)
            and ev["mds_x"] is not None
            and np.isfinite(ev["mds_x"])
        ]
        if agree_pts:
            xs, ys = zip(*agree_pts)
            ax.scatter(xs, ys, s=EVENT_MARKER_SIZE, c=col, alpha=0.55,
                       edgecolors="none", label=f"cluster {k}", zorder=2)
        if bound_pts:
            xs, ys = zip(*bound_pts)
            ax.scatter(xs, ys, s=BOUNDARY_MARKER_SIZE, facecolors="none",
                       edgecolors=col, linewidths=1.3, alpha=0.9,
                       zorder=3)

    # Template stars
    for k, tmpl in enumerate(templates):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        ax.scatter([tmpl["mds_x"]], [tmpl["mds_y"]],
                   marker="*", s=TEMPLATE_MARKER_SIZE,
                   c=col, edgecolors="black", linewidths=1.5,
                   zorder=4, label=f"T{k}")

    ax.set_xlabel("MDS-1 (template-matching metric)", fontsize=FS_LABEL)
    ax.set_ylabel("MDS-2", fontsize=FS_LABEL)

    n_total = data["n_events_total"]
    n_used = data["n_events_used_for_mds"]
    sil_med = data["silhouette_median"]
    agreement = data["agreement_overall"]
    stress = data["stress"]
    line1 = f"n={n_total}  k={chosen_k}  sil={sil_med:.3f}  agreement={agreement:.3f}"
    line2 = f"MDS on {n_used} subsample" if data.get("subsampled") else f"MDS on full {n_used} events"
    if data.get("stress_warning") or data.get("imputation_warning"):
        line2 += f"   ⚠ stress={stress:.2f}"
    ax.set_title(f"{line1}\n{line2}", fontsize=FS_TITLE - 4, loc="left")

    # -----------------------------------------------------------------------
    # Panel b: per-event silhouette ranked bar
    # -----------------------------------------------------------------------
    ax = axes[1]
    style_panel(ax, label="b", label_x=-0.12, label_y=1.06)

    # Group by cluster, sort sils within cluster (desc); plot as filled curves
    cluster_block_centers: List[Tuple[int, float]] = []
    cur = 0
    n_total_events = sum(
        1 for ev in events if ev["silhouette"] is not None and np.isfinite(ev["silhouette"])
    )
    gap = max(int(n_total_events * 0.01), 50)
    n_neg_total = 0

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        sil_in_k = np.array(
            sorted(
                [
                    ev["silhouette"]
                    for ev in events
                    if ev["kmeans_label"] == k
                    and ev["silhouette"] is not None
                    and np.isfinite(ev["silhouette"])
                ],
                reverse=True,
            ),
            dtype=float,
        )
        n_in_k = sil_in_k.size
        if n_in_k == 0:
            continue
        block_start = cur
        x = np.arange(n_in_k) + cur
        # Positive portion fill
        pos_mask = sil_in_k >= 0
        neg_mask = ~pos_mask
        n_neg_total += int(neg_mask.sum())
        # Plot positive band in cluster color
        if pos_mask.any():
            ax.fill_between(
                x, np.zeros(n_in_k), np.where(pos_mask, sil_in_k, 0.0),
                color=col, alpha=0.55, linewidth=0,
            )
        # Plot negative band in rust (boundary events)
        if neg_mask.any():
            ax.fill_between(
                x, np.where(neg_mask, sil_in_k, 0.0), np.zeros(n_in_k),
                color=COL_SIG, alpha=0.75, linewidth=0,
            )
        # Outline curve
        ax.plot(x, sil_in_k, color=col, lw=1.0, alpha=0.9)
        cluster_block_centers.append((k, (block_start + cur + n_in_k) / 2))
        cur += n_in_k + gap

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("event (ordered by cluster, then silhouette desc)", fontsize=FS_LABEL)
    ax.set_ylabel("per-event silhouette", fontsize=FS_LABEL)
    if cluster_block_centers:
        ax.set_xticks([c[1] for c in cluster_block_centers])
        ax.set_xticklabels(
            [f"cluster {c[0]}" for c in cluster_block_centers], fontsize=FS_TICK
        )
    ax.set_ylim(-1.05, 1.05)
    neg_frac = n_neg_total / max(n_total_events, 1)
    ax.set_title(
        f"per-event silhouette\nnegative {n_neg_total:,}/{n_total_events:,} ({neg_frac:.1%})",
        fontsize=FS_TITLE - 4, loc="left",
    )

    # -----------------------------------------------------------------------
    # Panel c: cluster template profile
    # -----------------------------------------------------------------------
    ax = axes[2]
    style_panel(ax, label="c", label_x=-0.12, label_y=1.06)

    templates_real = np.asarray(data["templates_real"], dtype=float)  # (k, n_ch)

    # Choose ordering channel: dominant cluster (largest n_events)
    cluster_n_events = [
        sum(1 for ev in events if ev["kmeans_label"] == k) for k in range(chosen_k)
    ]
    dominant_k = int(np.argmax(cluster_n_events))
    dom_template = templates_real[dominant_k]
    finite_mask = np.isfinite(dom_template)
    if not finite_mask.any():
        # Fallback: pick first cluster with finite values
        for k in range(chosen_k):
            if np.any(np.isfinite(templates_real[k])):
                dominant_k = k
                dom_template = templates_real[k]
                finite_mask = np.isfinite(dom_template)
                break

    # Build channel order: finite channels of dominant template sorted by
    # template rank ascending; non-finite channels appended at end
    finite_idx = np.where(finite_mask)[0]
    sorted_finite = finite_idx[np.argsort(dom_template[finite_idx])]
    nonfinite_idx = np.where(~finite_mask)[0]
    channel_order = np.concatenate([sorted_finite, nonfinite_idx])

    # Per-cluster IQR computed on raw rank vectors of events in that cluster
    n_show = len(sorted_finite)  # only show finite channels of dominant template
    x_pos = np.arange(n_show)

    iqr_low = {
        int(t["cluster_id"]): np.asarray(t.get("template_iqr_low", []), dtype=float)
        for t in templates
    }
    iqr_high = {
        int(t["cluster_id"]): np.asarray(t.get("template_iqr_high", []), dtype=float)
        for t in templates
    }

    for k in range(chosen_k):
        col = CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)]
        tline = templates_real[k, channel_order[:n_show]]
        finite_line = np.isfinite(tline)
        # IQR band when present
        lo = iqr_low.get(k)
        hi = iqr_high.get(k)
        if lo is not None and hi is not None and lo.size == n_ch and hi.size == n_ch:
            lo_ord = lo[channel_order[:n_show]]
            hi_ord = hi[channel_order[:n_show]]
            band_finite = np.isfinite(lo_ord) & np.isfinite(hi_ord)
            if band_finite.any():
                ax.fill_between(
                    x_pos[band_finite], lo_ord[band_finite], hi_ord[band_finite],
                    color=col, alpha=0.18, linewidth=0, zorder=1,
                )
        ax.plot(
            x_pos[finite_line],
            tline[finite_line],
            color=col, lw=2.2,
            marker="o", markersize=5,
            label=f"cluster {k}",
            zorder=3,
        )

    ax.set_xlabel("channel (ordered by dominant cluster rank)", fontsize=FS_LABEL)
    ax.set_ylabel("template mean rank", fontsize=FS_LABEL)
    ax.set_title("cluster template profile", fontsize=FS_TITLE - 4, loc="left")
    ax.legend(loc="best", fontsize=FS_TICK - 2, frameon=False)

    # Channel name ticks (sparse if many)
    if n_show <= 12:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [channel_names[ci] for ci in channel_order[:n_show]],
            rotation=45, ha="right", fontsize=FS_TICK - 2,
        )
    else:
        step = max(1, n_show // 10)
        sel = np.arange(0, n_show, step)
        ax.set_xticks(sel)
        ax.set_xticklabels(
            [channel_names[channel_order[i]] for i in sel],
            rotation=45, ha="right", fontsize=FS_TICK - 2,
        )

    fig.suptitle(
        f"{dataset.capitalize()} · {subject} — cluster geometry (template-matching metric)",
        fontsize=FS_TITLE,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return savefig_pub(fig, output_path)


# ---------------------------------------------------------------------------
# Cohort 2x2 summary figure
# ---------------------------------------------------------------------------


def plot_cohort_summary(
    cohort: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Produce the 2x2 cohort summary described in §5.2."""
    per_subject = cohort.get("per_subject", {})
    if not per_subject:
        logger.warning("Cohort summary has no included subjects; skip figure")
        return output_path

    # Build per-subject record list
    records: List[Dict[str, Any]] = []
    for key, v in per_subject.items():
        if "/" in key:
            ds, sub = key.split("/", 1)
        else:
            ds, sub = "unknown", key
        records.append(
            {
                "key": key,
                "dataset": ds,
                "subject": sub,
                "silhouette": float(v["silhouette_median"]) if v["silhouette_median"] is not None else float("nan"),
                "agreement": float(v["agreement_overall"]) if v["agreement_overall"] is not None else float("nan"),
                "stable_k": int(v["stable_k"]),
                "n_events": int(v["n_events"]),
                "stress": float(v["stress"]) if v["stress"] is not None else float("nan"),
            }
        )

    fig, axes = new_figure(nrows=2, ncols=2, figsize=(16, 12))

    # ------- Panel a: per-subject silhouette ranked bar ---------------
    ax = axes[0, 0]
    style_panel(ax, label="a", label_x=-0.10, label_y=1.04)
    rec_a = sorted(records, key=lambda r: -r["silhouette"] if np.isfinite(r["silhouette"]) else 0.0)
    x = np.arange(len(rec_a))
    cols = [dataset_color(r["dataset"]) for r in rec_a]
    ax.bar(x, [r["silhouette"] for r in rec_a], color=cols, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset'][:1].upper()}:{r['subject'][:8]}" for r in rec_a],
                       rotation=70, ha="right", fontsize=FS_TICK - 4)
    ax.set_ylabel("silhouette median (template-matching)", fontsize=FS_LABEL)
    ax.set_title("per-subject cluster validity", fontsize=FS_TITLE - 1)
    ax.set_ylim(-0.1, 1.0)

    # ------- Panel b: per-subject agreement ranked bar ---------------
    ax = axes[0, 1]
    style_panel(ax, label="b", label_x=-0.10, label_y=1.04)
    rec_b = sorted(records, key=lambda r: -r["agreement"] if np.isfinite(r["agreement"]) else 0.0)
    x = np.arange(len(rec_b))
    cols = [
        COL_SIG if r["agreement"] < 0.85 else dataset_color(r["dataset"])
        for r in rec_b
    ]
    ax.bar(x, [r["agreement"] for r in rec_b], color=cols, edgecolor="white", linewidth=0.5)
    ax.axhline(0.85, color=COL_SIG, lw=1.0, ls="--", alpha=0.6,
               label="< 0.85 highlighted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset'][:1].upper()}:{r['subject'][:8]}" for r in rec_b],
                       rotation=70, ha="right", fontsize=FS_TICK - 4)
    ax.set_ylabel("KMeans-vs-template-matching agreement", fontsize=FS_LABEL)
    ax.set_title("audit: label agreement under unified metric", fontsize=FS_TITLE - 1)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=FS_TICK - 2, frameon=False)

    # ------- Panel c: joint silhouette vs agreement scatter -----------
    ax = axes[1, 0]
    style_panel(ax, label="c", label_x=-0.10, label_y=1.04)
    sil_arr = np.array([r["silhouette"] for r in records])
    agr_arr = np.array([r["agreement"] for r in records])
    valid = np.isfinite(sil_arr) & np.isfinite(agr_arr)
    if int(valid.sum()) >= 3:
        rho, pval = spearmanr(sil_arr[valid], agr_arr[valid])
    else:
        rho, pval = float("nan"), float("nan")

    marker_for_k = {2: "o", 4: "^", 6: "s"}
    for r in records:
        marker = marker_for_k.get(r["stable_k"], "D")
        col = dataset_color(r["dataset"])
        ax.scatter(r["silhouette"], r["agreement"],
                   s=110, marker=marker, c=col,
                   edgecolors="black", linewidths=0.8, alpha=0.85)

    ax.set_xlabel("silhouette median", fontsize=FS_LABEL)
    ax.set_ylabel("KMeans-matching agreement", fontsize=FS_LABEL)
    ax.set_title(
        f"joint relationship  Spearman ρ = {rho:.3f} (p = {pval:.3g})",
        fontsize=FS_TITLE - 1,
    )

    # Legend for shape
    handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="",
                   markersize=8, label="k=2"),
        plt.Line2D([0], [0], marker="^", color="black", linestyle="",
                   markersize=8, label="k=4"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="",
                   markersize=8, label="k=6"),
        plt.Line2D([0], [0], marker="o", color=COL_YUQUAN, linestyle="",
                   markersize=8, label="Yuquan"),
        plt.Line2D([0], [0], marker="o", color=COL_EPILEPSIAE, linestyle="",
                   markersize=8, label="Epilepsiae"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=FS_TICK - 2, frameon=False, ncol=2)

    # ------- Panel d: boundary fraction by n_participating bin --------
    ax = axes[1, 1]
    style_panel(ax, label="d", label_x=-0.10, label_y=1.04)
    bin_keys = ["3-4", "5-8", "9+"]
    boundary_data = cohort.get("boundary_fraction_by_nparticipating", {})
    bin_vals: List[List[float]] = [
        [v for v in boundary_data.get(b, []) if v is not None and np.isfinite(v)]
        for b in bin_keys
    ]
    positions = np.arange(len(bin_keys))
    rng = np.random.default_rng(42)
    for i, vals in enumerate(bin_vals):
        if not vals:
            continue
        vals_arr = np.array(vals)
        # Box / scatter
        bp = ax.boxplot(
            vals_arr, positions=[positions[i]], widths=0.5,
            patch_artist=True, showfliers=False, showcaps=False,
            medianprops=dict(linewidth=2.2, color="black"),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COL_NEUTRAL)
            patch.set_alpha(0.4)
            patch.set_edgecolor("black")
        jit = positions[i] + rng.normal(0, 0.06, vals_arr.size)
        ax.scatter(jit, vals_arr, s=55, c=COL_NIGHT, edgecolors="white",
                   linewidths=0.8, alpha=0.85, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"n_part {b}" for b in bin_keys], fontsize=FS_TICK)
    ax.set_ylabel("boundary-event fraction (per subject)", fontsize=FS_LABEL)
    ax.set_title("metric drift by n_participating", fontsize=FS_TITLE - 1)
    ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))

    fig.suptitle(
        f"Cluster geometry cohort summary  "
        f"(included={cohort.get('n_subjects_included', '?')}, "
        f"excluded={cohort.get('n_subjects_excluded', '?')})",
        fontsize=FS_TITLE + 1, y=0.998,
    )
    fig.tight_layout()
    return savefig_pub(fig, output_path)


# ---------------------------------------------------------------------------
# Showcase selector
# ---------------------------------------------------------------------------


def _select_low_agreement_showcase(
    cohort: Dict[str, Any],
    min_n_events: int = 200,
    target_subjects_to_avoid: Optional[List[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Pick a non-trivial low-agreement subject for the showcase trio."""
    target_subjects_to_avoid = set(target_subjects_to_avoid or [])
    candidates: List[Tuple[float, str, str]] = []
    for key, v in cohort.get("per_subject", {}).items():
        if "/" not in key:
            continue
        ds, sub = key.split("/", 1)
        if sub in target_subjects_to_avoid:
            continue
        if v.get("n_events", 0) < min_n_events:
            continue
        agr = v.get("agreement_overall", 1.0)
        if agr is None or not np.isfinite(agr):
            continue
        if agr >= 0.85:
            continue
        sil = v.get("silhouette_median", 0.0) or 0.0
        candidates.append((sil, ds, sub))
    if not candidates:
        return None
    candidates.sort()  # ascending silhouette
    return (candidates[0][1], candidates[0][2])


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------


def _write_figures_readme() -> None:
    readme = FIGURES_DIR / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    text = """# 间期事件 cluster geometry 可视化（template-matching metric）

> 设计文档：`docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md`
> 距离定义：shared-channel mean squared deviation（与 `assign_events_to_templates` 一致；不是 KMeans 训练时用的 NaN→0 距离）

## per_subject/

每个 subject 一张 1×3 图。

### `<dataset>_<subject>_geometry.png`

- **Panel a**：MDS 散点图。事件按 PR-2 KMeans label 上色；模板用大星黑边突出；KMeans label 与 template-matching reassign 不一致的事件叠加空心圆。Title 写 `n=… k=… sil=… agreement=…`，stress 超阈或 imputation 比例过高时挂 ⚠️。
- **Panel b**：每事件 silhouette（template-matching metric 下）排序条。按 cluster 分块、cluster 内按 silhouette 降序。silhouette < 0 的事件用 Morandi rust 高亮——这些事件被 KMeans 归到本簇，但在 template-matching metric 下离对家更近。
- **Panel c**：cluster template profile。横轴 = 通道（按 dominant cluster 的 template rank 升序），纵轴 = template mean rank；每 cluster 一条折线。直接展示每个传播模式的"哪些通道早 / 哪些通道晚"，是 panel a 的 MDS 视图无法替代的结构层信息。

**关注点**：
- Panel a 的两类事件云是否明显分开；模板大星是否落在各自事件云的"中心"
- Panel b 的 silhouette 分布是否大部分为正；负 silhouette 的比例
- Panel c 的两条 template 折线是否反向（forward/reverse）或重合（纯 identity bias）

## showcase/

3 张精修单图，相同 panel 结构：

- `958_geometry_showcase.png` —— k=2 forward/reverse 的教科书 case（老论文 E3 复现）
- `huangwanling_geometry_showcase.png` —— k=4 多模态最干净的 subject（raw_tau ≈ centered_tau）
- `<low_agreement>_geometry_showcase.png` —— cohort 跑完后 agreement 最低的 subject 之一，用来诚实展示 metric drift 的真实长相

**关注点**：每张图的 panel a 和 panel c 应能一眼看出该 subject 的传播模式是 forward/reverse 互逆还是完全独立的多模态。

## cohort_geometry_summary.png

一张 2×2 大图。

- **Panel a**：per-subject silhouette median 排序条；YQ 蓝、EPI 赭，stable_k 用形状区分（圆 / 三角 / 方）
- **Panel b**：per-subject KMeans-matching agreement 排序条；agreement < 0.85 用 Morandi rust 高亮，标 0.85 dashed line
- **Panel c**：silhouette median vs agreement joint scatter；Spearman ρ + p 值在 title；marker shape = stable_k，color = dataset
- **Panel d**：boundary-event fraction（agreement = False 的比例）按 n_participating bin（3-4 / 5-8 / 9+）做 box + scatter，看 metric drift 是否集中在低参与事件

**关注点**：
- Panel a 的 silhouette 中位数是否大体 > 0.3（cluster validity 整体成立）
- Panel b 的 agreement 是否大部分 > 0.85（PR-2 决策稳定）
- Panel c 的 ρ 反映两个问题（cluster validity 弱 / metric drift 大）是否耦合
- Panel d 的箱体如果在 3-4 显著高、9+ 显著低，则 metric drift 的源头 = 低 n_participating 事件
"""
    readme.write_text(text)
    logger.info("Wrote %s", readme)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-subject", action="store_true", default=False,
                        help="Plot per-subject 3-panel figures.")
    parser.add_argument("--cohort", action="store_true", default=False,
                        help="Plot cohort 2x2 summary.")
    parser.add_argument("--showcase", action="store_true", default=False,
                        help="Plot showcase trio (958, huangwanling, low-agreement).")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Equivalent to --per-subject --cohort --showcase.")
    parser.add_argument("--subjects", action="append", default=None,
                        help="Restrict per-subject plots to specific subjects.")
    args = parser.parse_args()

    if args.all:
        args.per_subject = True
        args.cohort = True
        args.showcase = True

    if not (args.per_subject or args.cohort or args.showcase):
        parser.error("must specify at least one of --per-subject / --cohort / --showcase / --all")

    PER_SUBJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SHOWCASE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    files = _list_per_subject_files()
    if not files:
        logger.warning("No per-subject geometry JSONs found in %s — run scripts/run_cluster_geometry.py first", PER_SUBJECT_DIR)
        return

    if args.per_subject:
        for path in files:
            data = _load_subject_geometry(path)
            if data is None or data.get("status") != "ok":
                continue
            ds = data.get("dataset", "x")
            sub = data.get("subject", path.stem)
            if args.subjects and sub not in args.subjects:
                continue
            out = PER_SUBJECT_FIG_DIR / f"{ds}_{sub}_geometry.png"
            plot_per_subject_geometry(data, out, is_showcase=False)

    if args.showcase:
        cohort = _load_cohort_summary()
        # Fixed showcase candidates by priority. Yuquan huangwanling is the
        # original multimodal (k=4) showcase but is currently excluded by
        # data-freshness; fall back to Epilepsiae 818 (also k=4 in archive).
        fixed_showcase: List[Tuple[str, str]] = [("epilepsiae", "958")]
        for cand in [("yuquan", "huangwanling"), ("epilepsiae", "818"),
                     ("epilepsiae", "1077"), ("epilepsiae", "zhangjinhan")]:
            json_path = PER_SUBJECT_DIR / f"{cand[0]}_{cand[1]}.json"
            if json_path.exists():
                data = _load_subject_geometry(json_path)
                if data and data.get("status") == "ok":
                    fixed_showcase.append(cand)
                    break

        if cohort:
            avoid = {sub for _, sub in fixed_showcase}
            low_pick = _select_low_agreement_showcase(
                cohort, target_subjects_to_avoid=list(avoid)
            )
            if low_pick:
                fixed_showcase.append(low_pick)

        for ds, sub in fixed_showcase:
            json_path = PER_SUBJECT_DIR / f"{ds}_{sub}.json"
            data = _load_subject_geometry(json_path)
            if data is None or data.get("status") != "ok":
                logger.warning("Showcase skip %s/%s: no geometry data", ds, sub)
                continue
            out = SHOWCASE_FIG_DIR / f"{sub}_geometry_showcase.png"
            plot_per_subject_geometry(data, out, is_showcase=True)

    if args.cohort:
        cohort = _load_cohort_summary()
        if cohort is None:
            logger.warning("No cohort_summary.json found at %s", GEOMETRY_DIR)
        else:
            out = FIGURES_DIR / "cohort_geometry_summary.png"
            plot_cohort_summary(cohort, out)

    _write_figures_readme()


if __name__ == "__main__":
    main()
