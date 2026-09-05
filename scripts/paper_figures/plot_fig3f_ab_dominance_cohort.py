#!/usr/bin/env python3
"""Render the selected Figure 3F heatmap and its inferential companion."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import savefig_pub, style_panel  # noqa: E402


DATA_DIR = ROOT / "results/paper-ready-figure/fig3f_ab_dominance_cohort"
COHORT_JSON = DATA_DIR / "fig3f_ab_dominance_cohort.json"
FIG_DIR = DATA_DIR / "figures"
PAIRED_STEM = "fig3f_ab_dominance_paired"
HEATMAP_STEM = "fig3f_ab_dominance_heatmap"
SUPPLEMENTARY_TABLE = ROOT / "docs/paper-draft/cohort_contract_and_supplementary_tables.md"
EPILEPSIAE_ORDER_SOURCE = (
    ROOT / "ReplayIED/inter_events/epilepsiae_interictal/"
    "plotting_figAdd_personalKuramoto_withDelay.py"
)
YUQUAN_ORDER_SOURCE = (
    ROOT / "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/"
    "plotting_fig1_hfoHist.py"
)

COL_SIG = "#A35E48"
COL_NONSIG = "#A9A9A9"
COL_MEDIAN = "#202020"
START_SEC, STOP_SEC, STEP_SEC = -120.0, 20.0, 2.0
HEATMAP_SORT_WINDOW = (-30.0, 20.0)
# F is reduced more strongly than the left-column panels in the final board.
# Keep patient IDs dense, but compensate the source type for that reduction.
HEATMAP_AXIS_LABEL_FONTSIZE = 17.5
HEATMAP_X_TICK_FONTSIZE = 14.5
HEATMAP_DENSE_Y_TICK_FONTSIZE = 12.5
HEATMAP_COLORBAR_LABEL_FONTSIZE = 15.5
HEATMAP_COLORBAR_TICK_FONTSIZE = 13.5
HEATMAP_FIGSIZE = (6.10, 3.20)


def _literal_list_assignment(path: Path, variable: str) -> list[str]:
    tree = ast.parse(path.read_text())
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable
               for target in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, list):
                matches.append([str(item) for item in value])
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one literal {variable}, found {len(matches)}")
    return matches[0]


@lru_cache(maxsize=None)
def _supplementary_ids(prefix: str) -> tuple[str, ...]:
    ids = re.findall(
        rf"^\|\s*({re.escape(prefix)}\d+)\s*\|",
        SUPPLEMENTARY_TABLE.read_text(),
        flags=re.MULTILINE,
    )
    expected = [f"{prefix}{index}" for index in range(1, 21)]
    if ids != expected:
        raise RuntimeError(
            f"supplementary {prefix}-label contract mismatch: {ids} != {expected}"
        )
    return tuple(ids)


@lru_cache(maxsize=None)
def _manuscript_orders() -> tuple[tuple[str, ...], tuple[str, ...]]:
    epilepsiae = tuple(_literal_list_assignment(EPILEPSIAE_ORDER_SOURCE, "sub_list"))
    yuquan = tuple(
        _literal_list_assignment(YUQUAN_ORDER_SOURCE, "sub1_list")
        + _literal_list_assignment(YUQUAN_ORDER_SOURCE, "sub2_list")
    )
    if len(epilepsiae) != 20 or len(epilepsiae) != len(set(epilepsiae)):
        raise RuntimeError("legacy Epilepsiae manuscript order is not 20 unique subjects")
    # Only the first 18 legacy Yuquan entries have public Y1-Y18 positions.
    if len(yuquan) < 18 or len(yuquan) != len(set(yuquan)):
        raise RuntimeError("legacy Yuquan manuscript order is invalid")
    return epilepsiae, yuquan


def _pretty(subject: str) -> str:
    """Return the Table S1/S2 manuscript ID; never expose raw IDs on canvas."""
    dataset, sid = subject.split("_", 1)
    epilepsiae_order, yuquan_order = _manuscript_orders()
    if dataset == "epilepsiae":
        if sid not in epilepsiae_order:
            raise ValueError(f"{subject}: absent from Table S2 source order")
        return _supplementary_ids("E")[epilepsiae_order.index(sid)]
    if dataset == "yuquan":
        legacy_public = yuquan_order[:18]
        if sid not in legacy_public:
            raise ValueError(
                f"{subject}: Y19/Y20 require the private crosswalk; refusing to guess"
            )
        return _supplementary_ids("Y")[legacy_public.index(sid)]
    raise ValueError(f"unsupported dataset in {subject}")


def _fmt_p(value) -> str:
    if value is None:
        return "P not estimable"
    value = float(value)
    if value < 0.001:
        return "P<0.001"
    return f"P={value:.3f}"


def _primary_rows(payload: dict) -> list[dict]:
    rows = [
        record for record in payload["subjects"]
        if record.get("status") == "ok"
        and record.get("primary", {}).get("eligible")
    ]
    rows.sort(key=lambda record: float(record["primary"]["delta"]), reverse=True)
    return rows


def _spread_labels(values: list[float], *, lo: float = 0.035, hi: float = 0.965,
                   gap: float = 0.045) -> list[float]:
    """Greedy non-overlap placement for a small set of right-side labels."""
    if not values:
        return []
    order = np.argsort(values)
    placed = np.asarray(values, float)[order]
    placed[0] = max(placed[0], lo)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    if placed[-1] > hi:
        placed -= placed[-1] - hi
        for index in range(len(placed) - 2, -1, -1):
            placed[index] = min(placed[index], placed[index + 1] - gap)
    placed = np.clip(placed, lo, hi)
    out = np.empty(len(values), float)
    out[order] = placed
    return out.tolist()


def make_paired_figure(payload: dict) -> plt.Figure:
    """Compact patient-level paired slope plot; no legend box is required."""
    rows = _primary_rows(payload)
    if len(rows) != payload.get("n_primary_eligible"):
        raise RuntimeError("payload subject count is inconsistent")
    if not rows:
        raise RuntimeError("no primary-eligible subjects")

    fig, ax = plt.subplots(figsize=(4.25, 3.20), facecolor="white")
    style_panel(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, width=0, pad=7)

    for record in rows:
        primary = record["primary"]
        values = [float(primary["polar_far"]), float(primary["polar_near"])]
        locked = primary.get("subject_locked") is True
        color = COL_SIG if locked else COL_NONSIG
        ax.plot([0, 1], values, color=color, lw=1.25 if locked else 0.8,
                alpha=0.95 if locked else 0.52, zorder=1)
        ax.scatter([0], [values[0]], s=22, facecolor="white", edgecolor=color,
                   linewidth=0.9, zorder=2)
        ax.scatter([1], [values[1]], s=24, facecolor=color, edgecolor="white",
                   linewidth=0.45, zorder=2)

    far = np.asarray([record["primary"]["polar_far"] for record in rows], float)
    near = np.asarray([record["primary"]["polar_near"] for record in rows], float)
    medians = [float(np.median(far)), float(np.median(near))]
    ax.plot([0, 1], medians, color=COL_MEDIAN, lw=2.35, zorder=4)
    ax.scatter([0, 1], medians, marker="D", s=53, color=COL_MEDIAN,
               edgecolor="white", linewidth=0.65, zorder=5)

    locked_rows = [record for record in rows if record["primary"]["subject_locked"]]
    label_values = [float(record["primary"]["polar_near"]) for record in locked_rows]
    label_y = _spread_labels(label_values)
    for record, y_text in zip(locked_rows, label_y):
        y = float(record["primary"]["polar_near"])
        ax.plot([1.015, 1.075], [y, y_text], color=COL_SIG, lw=0.65,
                clip_on=False, zorder=2)
        ax.text(1.09, y_text, _pretty(record["subject"]), color=COL_SIG,
                fontsize=7.6, ha="left", va="center", clip_on=False)

    ax.annotate(
        "cohort median",
        xy=(1, medians[1]),
        xytext=(0.96, min(1.01, medians[1] + 0.11)),
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=COL_MEDIAN,
        arrowprops={"arrowstyle": "-", "color": COL_MEDIAN, "lw": 0.75},
    )
    cohort = payload["primary_cohort_hierarchical_time_null"]
    ax.text(
        0.02,
        0.98,
        f"n={cohort['n']}   median Δ={cohort['median_delta']:+.2f}\n"
        f"hierarchical time null: {_fmt_p(cohort['p_one_sided'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="0.20",
    )

    ax.set_xlim(-0.16, 1.26)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Far pre-ictal\n−120 to −60 s", "Near onset\n−30 to +10 s"],
                       fontsize=9.2)
    ax.set_ylabel(r"A/B dominance, $|\mathrm{mean}\ C_{AB}|$", fontsize=10.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=9.0, length=3.5, width=1.0)
    ax.grid(axis="y", color="0.92", lw=0.55, zorder=0)
    fig.subplots_adjust(left=0.19, right=0.84, bottom=0.23, top=0.96)
    return fig


def _subject_timecourse(
    record: dict, *, data_dir: Path = DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / "per_subject" / (
        f"{record['subject']}_fig3f_ab_dominance_timecourse.npz"
    )
    with np.load(path) as data:
        centers = np.asarray(data["window_center_sec"], float)
        c_ab = np.asarray(data["C_AB"], float)
    return centers, np.nanmedian(c_ab, axis=0)


def _heatmap_pattern_metrics(trajectory: np.ndarray, centers: np.ndarray) -> dict:
    """Classify peri-onset color as red-type, blue-type, or indeterminate.

    The display-only type uses the signed mean C_AB over the full plotted
    peri-onset tail [-30,+20) s, including time after clinical onset.  Coherence
    is retained as a within-type sorting diagnostic: ``abs(mean C_AB) /
    mean(abs(C_AB))`` approaches one for a stable sign and zero when red/blue
    epochs cancel.  This classification changes row order only; it does not
    alter eligibility, statistics, or plotted values.
    """
    trajectory = np.asarray(trajectory, float)
    centers = np.asarray(centers, float)
    near = (
        (centers >= HEATMAP_SORT_WINDOW[0])
        & (centers < HEATMAP_SORT_WINDOW[1])
        & np.isfinite(trajectory)
    )
    values = trajectory[near]
    if not len(values):
        return {
            "group": "mixed_or_weak",
            "group_rank": 2,
            "signed_mean": np.nan,
            "mean_abs": np.nan,
            "coherence": 0.0,
        }
    signed_mean = float(np.mean(values))
    mean_abs = float(np.mean(np.abs(values)))
    coherence = abs(signed_mean) / mean_abs if mean_abs > 1e-12 else 0.0
    if signed_mean > 0:
        group, group_rank = "red_type", 0
    elif signed_mean < 0:
        group, group_rank = "blue_type", 1
    else:
        group, group_rank = "mixed_or_weak", 2
    return {
        "group": group,
        "group_rank": group_rank,
        "signed_mean": signed_mean,
        "mean_abs": mean_abs,
        "coherence": float(coherence),
    }


def _heatmap_items(payload: dict, *, data_dir: Path = DATA_DIR) -> list[dict]:
    items = []
    for record in _primary_rows(payload):
        centers, trajectory = _subject_timecourse(record, data_dir=data_dir)
        metrics = _heatmap_pattern_metrics(trajectory, centers)
        items.append({
            "record": record,
            "centers": centers,
            "trajectory": trajectory,
            "metrics": metrics,
        })

    def sort_key(item: dict):
        metrics = item["metrics"]
        # Within red/blue blocks, put the strongest peri-onset direction first.
        secondary = -abs(metrics["signed_mean"])
        tertiary = -metrics["coherence"]
        return (
            metrics["group_rank"],
            secondary,
            tertiary,
            item["record"]["subject"],
        )

    return sorted(items, key=sort_key)


def make_heatmap_figure(payload: dict, *, data_dir: Path = DATA_DIR) -> plt.Figure:
    """Selected display retaining the full signed peri-onset trajectory."""
    items = _heatmap_items(payload, data_dir=data_dir)
    centers = items[0]["centers"]
    for item in items[1:]:
        if not np.allclose(centers, item["centers"]):
            raise RuntimeError("subject time grids differ")
    matrix = np.vstack([item["trajectory"] for item in items])
    vmax = max(0.5, float(np.nanpercentile(np.abs(matrix), 98)))

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE, facecolor="white")
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[START_SEC, STOP_SEC, len(items) - 0.5, -0.5],
    )
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_xlim(START_SEC, STOP_SEC)
    ax.set_xticks([-120, -60, 0, 20])
    ax.set_xlabel("Time (s)", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE, labelpad=5)
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels(
        [_pretty(item["record"]["subject"]) for item in items],
        fontsize=HEATMAP_DENSE_Y_TICK_FONTSIZE,
    )
    ax.tick_params(
        axis="x", labelsize=HEATMAP_X_TICK_FONTSIZE, length=3.5, width=1.0, pad=3
    )
    ax.tick_params(axis="y", length=0, width=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.046)
    colorbar.set_label(
        r"Signed A/B contrast, $C_{AB}$",
        fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        labelpad=6,
    )
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE, length=3)
    ticks = np.linspace(-vmax, vmax, 5)
    tick_labels = [f"{value:.2f}" for value in ticks]
    tick_labels[0] = f"B  {ticks[0]:.2f}"
    tick_labels[-1] = f"A  +{ticks[-1]:.2f}"
    colorbar.set_ticks(ticks, labels=tick_labels)
    group_ranks = [item["metrics"]["group_rank"] for item in items]
    for index in range(1, len(group_ranks)):
        if group_ranks[index] != group_ranks[index - 1]:
            ax.hlines(index - 0.5, START_SEC, STOP_SEC, color="white",
                      lw=1.5, zorder=3, clip_on=True)
    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.18, top=0.94)
    return fig


def _write_sidecars(
    payload: dict,
    paths: dict[str, Path],
    *,
    fig_dir: Path = FIG_DIR,
    data_dir: Path = DATA_DIR,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    cohort = payload["primary_cohort_hierarchical_time_null"]
    summary = {
        "panel": "Figure 3F",
        "source": str(COHORT_JSON.relative_to(ROOT)),
        "canonical_n": payload["n_canonical_subjects"],
        "primary_n": cohort["n"],
        "primary_statistic": payload["primary_statistic"],
        "primary_cohort_hierarchical_time_null": cohort,
        "primary_wilcoxon_greater": payload["primary_wilcoxon_greater"],
        "primary_subject_locked_count": payload["primary_subject_locked_count"],
        "within_shaft_sensitivity_locked_count": payload[
            "within_shaft_sensitivity_locked_count"
        ],
        "within_shaft_sensitivity_cohort_hierarchical_time_null": payload[
            "within_shaft_sensitivity_cohort_hierarchical_time_null"
        ],
        "files": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "heatmap_sort": {
            "window_sec": list(HEATMAP_SORT_WINDOW),
            "coherence_definition": "abs(mean C_AB) / mean(abs(C_AB))",
            "type_definition": "sign(mean C_AB) over [-30,+20) s",
            "group_order": ["red_type", "blue_type", "mixed_or_weak"],
            "rows": [
                {
                    "subject": item["record"]["subject"],
                    **item["metrics"],
                }
                for item in _heatmap_items(payload, data_dir=data_dir)
            ],
        },
        "manuscript_y_labels": {
            "supplementary_table": str(SUPPLEMENTARY_TABLE.relative_to(ROOT)),
            "epilepsiae_order_source": str(EPILEPSIAE_ORDER_SOURCE.relative_to(ROOT)),
            "yuquan_order_source": str(YUQUAN_ORDER_SOURCE.relative_to(ROOT)),
            "labels": [
                {
                    "subject": item["record"]["subject"],
                    "manuscript_label": _pretty(item["record"]["subject"]),
                }
                for item in _heatmap_items(payload, data_dir=data_dir)
            ],
        },
        "interpretation": (
            "The signed heatmap is the selected Figure 3F trajectory display. The paired "
            "panel is a compact inferential companion; neither requires the shared-plane subset."
        ),
    }
    (fig_dir / "fig3f_ab_dominance_render_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    p_text = _fmt_p(cohort["p_one_sided"]).replace("P", "p")
    readme = (
        f"### {PAIRED_STEM}.png\n\n"
        "配对统计图纳入冻结 own-field 2-D 合同中的全部患者；每条细线是一名患者，连接 far pre-ictal 与 near-onset 的患者内发作中位数。"
        "黑色菱形为 cohort median，超过患者内时间环移 null 的患者用 rust 色并直接标注编号，因此图内不再放底部 legend。"
        f"完整 cohort 的层级时间 null 为 {p_text}。\n\n"
        "**关注点**：看总体线条是否向上，以及黑色 cohort median 的变化；这检验的是 A/B 相对优势增强，不自动等同于方向翻转。\n\n"
        f"### {HEATMAP_STEM}.png\n\n"
        "Figure 3F 主图保留全部患者从 −120 s 到 +20 s 的 signed C_AB 轨迹；患者按包含发作后时间的 −30 至 +20 s signed mean 分型排序：红色型在上、蓝色型在下。"
        "纵轴使用 Supplementary Tables S1/S2 的匿名投稿编号 Y1–Y20/E1–E20，由 legacy cohort order 代码解析，不显示数据库编号或真实姓名。"
        "红、蓝分别表示 A、B 相对占优，白色表示没有稳定的 template dominance；虚线为 clinical onset。\n\n"
        "**关注点**：看临近 onset 是否出现跨患者一致的颜色加深，以及这种变化是同向选择、反向选择，还是患者间异质。\n"
    )
    (fig_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=COHORT_JSON)
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory containing per_subject timecourses; defaults to input JSON parent",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.out_dir.resolve()
    data_dir = input_path.parent if args.data_dir is None else args.data_dir.resolve()
    payload = json.loads(input_path.read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stem, maker in (
        (PAIRED_STEM, make_paired_figure),
        (HEATMAP_STEM, lambda item: make_heatmap_figure(item, data_dir=data_dir)),
    ):
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            savefig_pub(maker(payload), path, dpi=300)
            paths[f"{stem}_{suffix}"] = path
    _write_sidecars(payload, paths, fig_dir=output_dir, data_dir=data_dir)


if __name__ == "__main__":
    main()
