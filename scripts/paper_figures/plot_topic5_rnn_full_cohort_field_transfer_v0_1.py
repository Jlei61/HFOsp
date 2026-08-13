#!/usr/bin/env python3
"""Paper-ready E1146 and cohort panels for full-cohort RNN field transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_contact_plane_static import (  # noqa: E402
    _limits_with_padding,
    _smooth_rank_field_mm,
)
from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    _canonical_transverse_sign,
)


RED = "#B2182B"
BLUE = "#2166AC"
GREY = "#9AA0A6"
DARK = "#25313A"
TIMING_CMAP = "viridis"
ENERGY_CMAP = "Blues"
DISPLAY_SIGMA_MM = 6.0


def _minmax(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if not np.any(ok):
        return out
    lo, hi = float(np.min(x[ok])), float(np.max(x[ok]))
    out[ok] = (x[ok] - lo) / (hi - lo) if hi > lo else 0.5
    return out


def _pstar(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)
    ax.tick_params(labelsize=10.5, width=0.9, length=3)


def _paired_panel(
    ax: plt.Axes,
    left: np.ndarray,
    right: np.ndarray,
    labels: tuple[str, str],
    colors: tuple[str, str],
    ylabel: str,
    title: str,
    p: float,
) -> None:
    left, right = np.asarray(left, float), np.asarray(right, float)
    rng = np.random.default_rng(20260811 + len(left))
    jitter = rng.normal(0, 0.025, len(left))
    for j, (a, b) in enumerate(zip(left, right)):
        ax.plot([0 + jitter[j], 1 + jitter[j]], [a, b], color="#CBD0D4", lw=0.8, alpha=0.72, zorder=1)
    ax.scatter(np.full(len(left), 0.0) + jitter, left, s=28, color=colors[0],
               edgecolor="white", linewidth=0.55, alpha=0.9, zorder=3)
    ax.scatter(np.full(len(right), 1.0) + jitter, right, s=28, color=colors[1],
               edgecolor="white", linewidth=0.55, alpha=0.9, zorder=3)
    for x, values, color in ((0, left, colors[0]), (1, right, colors[1])):
        med = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        ax.plot([x - 0.16, x + 0.16], [med, med], color=DARK, lw=2.1, zorder=4)
        ax.plot([x, x], [q1, q3], color=DARK, lw=1.6, zorder=4)
        ax.scatter([x], [med], s=34, color=color, edgecolor=DARK, linewidth=0.9, zorder=5)
    lo = float(np.nanmin(np.r_[left, right]))
    hi = float(np.nanmax(np.r_[left, right]))
    span = max(hi - lo, 0.08)
    y = hi + 0.12 * span
    ax.plot([0, 0, 1, 1], [y - 0.018 * span, y, y, y - 0.018 * span], color=DARK, lw=1.0)
    ax.text(0.5, y + 0.02 * span, _pstar(float(p)), ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=DARK)
    ax.set_ylim(lo - 0.10 * span, y + 0.15 * span)
    ax.set_xticks([0, 1], labels, fontsize=11.0)
    ax.set_ylabel(ylabel, fontsize=12.0)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=9)
    _style_axis(ax)


def plot_cohort(analysis: Path, out_dir: Path) -> dict:
    inter = pd.read_csv(analysis / "interictal_patient_statistics.csv").sort_values("subject")
    ictal = pd.read_csv(analysis / "ictal_patient_statistics.csv")
    ictal = ictal[ictal.group_id == "all_phenotype_matched"].sort_values("subject")
    cohort = pd.read_csv(analysis / "ictal_cohort_statistics.csv")
    summary = json.loads((analysis / "MODEL_FIELD_MANIFEST.json").read_text())
    inter_stat = summary["interictal"]
    ictal_stat = cohort[cohort.group_id == "all_phenotype_matched"].iloc[0]
    if len(inter) != 34 or len(ictal) != 17 or int(ictal.n_seizures.sum()) != 167:
        raise RuntimeError("figure_denominator_mismatch")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), layout="constrained")
    _paired_panel(
        axes[0], inter.native_model, inter.static_only,
        ("RNN rollout", "Static"), (RED, GREY),
        "Propagation correlation", "Interictal  ·  n=34",
        float(inter_stat["wilcoxon_one_sided_native_gt_static_p"]),
    )
    _paired_panel(
        axes[1], ictal.data, ictal.channel_null_median,
        ("RNN field", "Shuffled channels"), (RED, GREY),
        "Field concordance |r|", "Early ictal  ·  n=17",
        float(ictal_stat.wilcoxon_one_sided_data_gt_null_p),
    )
    axes[0].text(-0.22, 1.06, "a", transform=axes[0].transAxes, fontsize=15,
                 fontweight="bold", va="top")
    axes[1].text(-0.22, 1.06, "b", transform=axes[1].transAxes, fontsize=15,
                 fontweight="bold", va="top")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "topic5_rnn_full_cohort_interictal_ictal"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "interictal_n": 34,
        "ictal_n": 17,
        "ictal_seizures": 167,
        "interictal_p": float(inter_stat["wilcoxon_one_sided_native_gt_static_p"]),
        "ictal_p": float(ictal_stat.wilcoxon_one_sided_data_gt_null_p),
    }


def _field_geometry(record: dict) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    shared = record["interictal_field"]["planes"]["shared"]
    points = np.asarray(shared["points"], float) * float(shared["scale_mm"])
    points[:, 1] *= _canonical_transverse_sign(shared["w"])
    xlim = _limits_with_padding(points[:, 0], include_zero=True, min_span=35.0)
    ylim = _limits_with_padding(points[:, 1], include_zero=True, min_span=35.0)
    return points, xlim, ylim


def _draw_map(
    ax: plt.Axes,
    points: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cmap: str,
    title: str,
    title_color: str,
    *,
    show_x: bool,
    show_y: bool,
) -> None:
    X, Y, field, _, _ = _smooth_rank_field_mm(
        points[:, 0], points[:, 1], np.asarray(values, float), np.asarray(support, float),
        xlim, ylim, DISPLAY_SIGMA_MM,
    )
    ax.imshow(field, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
              aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ok = np.isfinite(values)
    ax.scatter(points[ok, 0], points[ok, 1], c=np.asarray(values)[ok], cmap=cmap,
               vmin=0, vmax=1, s=31, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=title_color, pad=6)
    ax.tick_params(labelsize=9.5, length=2.5)
    if show_x:
        ax.set_xlabel("Propagation axis (mm)", fontsize=11.0)
    else:
        ax.tick_params(labelbottom=False)
    if show_y:
        ax.set_ylabel("Transverse (mm)", fontsize=11.0)
    else:
        ax.tick_params(labelleft=False)


def plot_e1146(canonical_root: Path, analysis: Path, out_dir: Path) -> dict:
    subject = "epilepsiae_1146"
    record = json.loads((
        canonical_root / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{subject}.json"
    ).read_text())
    with np.load(analysis / "model_fields" / f"{subject}.npz", allow_pickle=True) as model:
        model_names = np.asarray(model["contact_order"], str)
        model_rank_a, model_rank_b = np.asarray(model["rank_a"], float), np.asarray(model["rank_b"], float)
        model_support_a = np.asarray(model["support_a"], float)
        model_support_b = np.asarray(model["support_b"], float)
    with np.load(analysis / "e1146_early_ictal_activation.npz", allow_pickle=True) as ictal:
        ictal_names = np.asarray(ictal["contact_order"], str)
        activation = np.asarray(ictal["activation"], float)
        n_seizures = int(ictal["n_seizures"])
    field = record["interictal_field"]
    names = np.asarray(field["contact_order"], str)
    if not np.array_equal(names, model_names) or not np.array_equal(names, ictal_names):
        raise RuntimeError("e1146_contact_order_mismatch")
    points, xlim, ylim = _field_geometry(record)
    emp_a, emp_b = _minmax(field["rank_a"]), _minmax(field["rank_b"])
    rnn_a, rnn_b = _minmax(model_rank_a), _minmax(model_rank_b)
    energy = _minmax(activation)
    emp_support_a = np.asarray(field["support_a"], float)
    emp_support_b = np.asarray(field["support_b"], float)
    energy_support = 0.5 * (emp_support_a + emp_support_b)

    fig = plt.figure(figsize=(8.0, 5.15), layout="constrained", facecolor="white")
    grid = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 0.045, 1.1, 0.055], wspace=0.06, hspace=0.05
    )
    ax_oa = fig.add_subplot(grid[0, 0])
    ax_ra = fig.add_subplot(grid[0, 1], sharex=ax_oa, sharey=ax_oa)
    ax_ob = fig.add_subplot(grid[1, 0], sharex=ax_oa, sharey=ax_oa)
    ax_rb = fig.add_subplot(grid[1, 1], sharex=ax_oa, sharey=ax_oa)
    ax_e = fig.add_subplot(grid[:, 3], sharex=ax_oa, sharey=ax_oa)
    _draw_map(ax_oa, points, emp_a, emp_support_a, xlim, ylim, TIMING_CMAP,
              "TA · observed", RED, show_x=False, show_y=True)
    _draw_map(ax_ra, points, rnn_a, model_support_a, xlim, ylim, TIMING_CMAP,
              "TA · RNN", RED, show_x=False, show_y=False)
    _draw_map(ax_ob, points, emp_b, emp_support_b, xlim, ylim, TIMING_CMAP,
              "TB · observed", BLUE, show_x=True, show_y=True)
    _draw_map(ax_rb, points, rnn_b, model_support_b, xlim, ylim, TIMING_CMAP,
              "TB · RNN", BLUE, show_x=True, show_y=False)
    _draw_map(ax_e, points, energy, energy_support, xlim, ylim, ENERGY_CMAP,
              "Early-ictal energy", DARK, show_x=True, show_y=False)
    ax_oa.text(-0.28, 1.12, "a", transform=ax_oa.transAxes, fontsize=15,
               fontweight="bold", va="top")
    ax_e.text(-0.18, 1.06, "b", transform=ax_e.transAxes, fontsize=15,
              fontweight="bold", va="top")
    timing_cax = fig.add_subplot(grid[:, 2])
    timing_cbar = fig.colorbar(
        ScalarMappable(Normalize(0, 1), cmap=TIMING_CMAP), cax=timing_cax
    )
    timing_cbar.set_ticks([0, 1])
    timing_cbar.set_ticklabels(["Early", "Late"])
    timing_cbar.set_label("Relative timing", fontsize=10.5)
    timing_cbar.ax.tick_params(labelsize=9)
    cax = fig.add_subplot(grid[:, 4])
    cbar = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=ENERGY_CMAP), cax=cax)
    lo, hi = float(np.nanmin(activation)), float(np.nanmax(activation))
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"{lo:.1f}", f"{hi:.1f}"])
    cbar.set_label("Energy (robust z)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("E1146", x=0.015, ha="left", fontsize=14, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "topic5_rnn_e1146_field_transfer"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source = pd.DataFrame({
        "contact": names, "x_mm": points[:, 0], "y_mm": points[:, 1],
        "empirical_ta_rank": np.asarray(field["rank_a"], float),
        "rnn_ta_rank": model_rank_a,
        "empirical_tb_rank": np.asarray(field["rank_b"], float),
        "rnn_tb_rank": model_rank_b,
        "early_ictal_activation": activation,
    })
    source.to_csv(analysis / "e1146_figure_source.csv", index=False)
    return {
        "subject": subject,
        "n_contacts": int(len(names)),
        "n_seizures": n_seizures,
        "rnn_vs_empirical_ta_spearman": float(spearmanr(model_rank_a, field["rank_a"]).statistic),
        "rnn_vs_empirical_tb_spearman": float(spearmanr(model_rank_b, field["rank_b"]).statistic),
    }


def _write_readme(out_dir: Path, cohort: dict, e1146: dict) -> None:
    text = f"""# Topic 5 RNN full-cohort field-transfer figures

### topic5_rnn_e1146_field_transfer.png / .pdf / .svg

E1146 的真实 SEEG tissue-plane layout。左两列分别比较经验 TA/TB rank field 与只用间期事件训练、在 heldout events 自由生成后得到的 RNN TA/TB field；右列是同患者 {e1146['n_seizures']} 次 Figure 3 phenotype-matched 发作在 onset 后 0–10 s 的早期能量中位场。TA/TB 标题使用论文固定红/蓝语义色，场内颜色仍统一为 viridis 的 early-to-late 顺序；发作能量使用 Blues。

**关注点**：模型场和发作场共享冻结的患者 SEEG 平面，但发作数据不参与 RNN 训练、mode 匹配或场构造。RNN 对经验场的 Spearman 为 TA={e1146['rnn_vs_empirical_ta_spearman']:.3f}、TB={e1146['rnn_vs_empirical_tb_spearman']:.3f}。

### topic5_rnn_full_cohort_interictal_ictal.png / .pdf / .svg

左图以患者为统计单位，比较 34 位 K=2 患者的收敛 RNN native rollout 与 static-only generator 的 heldout transition correlation；右图比较 17 位患者、全部 167 次合格发作先在患者内折叠后的 RNN-field concordance 与 synchronized all-contact channel-shuffle null。图内只用星号编码预定义单侧 paired Wilcoxon，精确 P 值和效应量保存在统计 JSON/CSV。

**关注点**：间期 denominator 固定为 34；发作期 denominator 固定为 Figure 3D 的 17 人/167 次发作。右侧 control 固定放在右边，不使用旧 history-RNN outer cache。
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/topic5_rnn_full_cohort_field_transfer_v0_1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures")
    args = parser.parse_args()
    cohort = plot_cohort(args.analysis, args.out_dir)
    e1146 = plot_e1146(args.canonical_root, args.analysis, args.out_dir)
    _write_readme(args.out_dir, cohort, e1146)
    (args.analysis / "FIGURE_SUMMARY.json").write_text(
        json.dumps({"cohort": cohort, "e1146": e1146}, ensure_ascii=False, indent=2) + "\n"
    )
    assets = {}
    for path in sorted(args.out_dir.glob("topic5_rnn_*.*")):
        assets[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    complete = {
        "status": "complete",
        "interictal_subjects": 34,
        "ictal_subjects": 17,
        "ictal_seizures": 167,
        "model_fields_frozen_before_target": 17,
        "unresolved_oom": 0,
        "target_used_for_training_or_selection": False,
        "assets": assets,
    }
    (args.analysis / "PIPELINE_COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
