#!/usr/bin/env python3
"""Redraw Figure 2F from all-event Timing+Space template fields.

The patient denominator is data-derived after the frozen 2D/shared-plane gates.
The observed statistic and both spatial nulls are read from the matching
all-event field analysis; no display-grid pixel enters the statistic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig2_shared_field_reversal_row import (  # noqa: E402
    COHORT_COLOR,
    NULL_COLOR,
    POINT_COLOR,
    _stable_seed,
    build_cohort_shift_null,
)
from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (  # noqa: E402
    _pretty as _manuscript_id,
)
from scripts.paper_figures.paper_figure_source_registry import (  # noqa: E402
    registered_path,
)


DEFAULT_INPUT = registered_path("fig2", "f", "analysis_root")
DEFAULT_OUTPUT = registered_path("fig2", "f", "staging_root")


def _significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = [
        {
            "subject_id": str(row["subject_id"]),
            "display_id": _manuscript_id(str(row["subject_id"])),
            "r": float(row["observed_shared_field_r"]),
            "n_contacts": int(row["n_contacts"]),
        }
        for row in rows
    ]
    if len(parsed) < 2:
        raise ValueError("Figure 2F requires at least two eligible patients")
    if len({row["subject_id"] for row in parsed}) != len(parsed):
        raise ValueError("duplicate patient in Figure 2F cohort")
    return parsed


def _load_channel_nulls(path: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        arrays = {
            row["subject_id"]: np.asarray(
                archive[f"{row['subject_id']}__channel"], dtype=float,
            )
            for row in rows
        }
    if any(len(values) < 20 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("invalid full-contact null draws")
    return arrays


def _build_raw_cohort_median_null(
    rows: list[dict],
    channel_nulls: dict[str, np.ndarray],
    *,
    n_draws: int,
    base_seed: int,
) -> tuple[dict, np.ndarray]:
    """Build the raw-r cohort-median null used for aligned display diamonds."""
    ordered = sorted(rows, key=lambda row: str(row["subject_id"]))
    observed = float(np.median([float(row["r"]) for row in ordered]))
    arrays = [
        np.asarray(channel_nulls[str(row["subject_id"])], dtype=float)
        for row in ordered
    ]
    rng = np.random.default_rng(_stable_seed("cohort:channel", base_seed))
    batches = []
    for start in range(0, n_draws, 5_000):
        n_batch = min(5_000, n_draws - start)
        samples = np.column_stack(
            [values[rng.integers(0, len(values), size=n_batch)] for values in arrays]
        )
        batches.append(np.median(samples, axis=1))
    null_r = np.concatenate(batches)
    p_value = float((1 + np.sum(null_r <= observed)) / (n_draws + 1))
    return {
        "n_subjects": len(ordered),
        "n_draws": int(n_draws),
        "observed_median_r": observed,
        "null_median_r": float(np.median(null_r)),
        "null_q025_r": float(np.percentile(null_r, 2.5)),
        "null_q975_r": float(np.percentile(null_r, 97.5)),
        "p_negative": p_value,
        "test": (
            "hierarchical full-contact-shuffle randomization of the raw "
            "cohort median r; lower tail"
        ),
    }, null_r


def _draw_distribution(
    ax: plt.Axes,
    rows: list[dict],
    *,
    sign_test_p: float,
) -> dict:
    ordered = sorted(rows, key=lambda row: float(row["r"]))
    values = np.asarray([row["r"] for row in ordered], float)
    ypos = np.arange(len(ordered), dtype=float)
    ax.axvspan(-1.0, 0.0, color="#EFF4F3", zorder=0)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.75, zorder=1)
    for value, y, row in zip(values, ypos, ordered):
        ax.plot([0.0, value], [y, y], color="#CFD8DC", linewidth=0.65, zorder=1)
        ax.scatter(
            [value], [y], s=24, facecolors=POINT_COLOR, edgecolors="white",
            linewidths=0.7, zorder=4,
        )
        if value <= -0.82:
            align, offset = "left", 0.035
        elif value >= 0.82:
            align, offset = "right", -0.035
        else:
            align = "right" if value < 0 else "left"
            offset = -0.035 if value < 0 else 0.035
        ax.text(
            value + offset, y, row["display_id"], ha=align, va="center",
            fontsize=7.8, color="#526874",
        )
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    summary_y = -1.35
    ax.plot(
        [q25, q75], [summary_y, summary_y], color=COHORT_COLOR,
        linewidth=4.2, alpha=0.35, solid_capstyle="round", zorder=2,
    )
    ax.scatter(
        [median], [summary_y], marker="D", s=27, facecolor=COHORT_COLOR,
        edgecolor="white", linewidth=0.65, zorder=5,
    )
    ax.text(
        0.02, 0.98,
        (
            f"negative: {_significance_label(sign_test_p)}\n"
            f"{int(np.sum(values < 0))}/{len(values)} patients"
        ),
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        color="#222222",
    )
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.9, len(ordered) + 2.25)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    return {
        "n": len(values),
        "n_negative": int(np.sum(values < 0)),
        "median": float(median),
        "iqr": [float(q25), float(q75)],
        "range": [float(np.min(values)), float(np.max(values))],
    }


def _draw_null(
    ax: plt.Axes,
    null_delta: np.ndarray,
    observed: float,
    *,
    p_negative: float,
) -> None:
    counts, edges = np.histogram(null_delta, bins=45, range=(-0.8, 0.8), density=True)
    counts = np.convolve(counts, np.asarray([1, 2, 3, 2, 1]) / 9.0, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = counts / max(float(np.max(counts)), 1e-12)
    ax.fill_between(centers, 0.0, counts, color=NULL_COLOR, alpha=0.75, linewidth=0)
    ax.plot(centers, counts, color="#8A8A8A", linewidth=0.75)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.7)
    ax.axvline(observed, color=COHORT_COLOR, linewidth=1.6)
    ax.scatter(
        [observed], [0.0], marker="D", s=28, facecolor=COHORT_COLOR,
        edgecolor="white", linewidth=0.6, clip_on=False, zorder=5,
    )
    null_median = float(np.median(null_delta))
    bracket_y = 1.06
    bracket_h = 0.045
    ax.plot(
        [observed, observed, null_median, null_median],
        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
        color="#222222", linewidth=0.85, clip_on=False, zorder=6,
    )
    ax.text(
        0.5 * (observed + null_median),
        bracket_y + bracket_h + 0.025,
        _significance_label(p_negative),
        ha="center", va="bottom", fontsize=7.6, color="#222222",
        clip_on=False,
    )
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(0.0, 1.34)
    ax.set_yticks([])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=9.0, length=2.6, pad=2.0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.65)


def build(input_root: Path, output_root: Path, *, seed: int, draws: int) -> dict:
    rows = _load_rows(input_root / "shared_field_similarity_subjects.csv")
    nulls = _load_channel_nulls(input_root / "shared_field_similarity_null_draws.npz", rows)
    statistics = json.loads(
        (input_root / "shared_field_similarity_statistics.json").read_text(encoding="utf-8")
    )
    summary, _null_delta = build_cohort_shift_null(
        rows, nulls, n_draws=draws, base_seed=seed,
    )
    raw_null_summary, raw_null_r = _build_raw_cohort_median_null(
        rows, nulls, n_draws=draws, base_seed=seed,
    )
    expected = statistics["cohort_summary"]["channel"]
    if not np.isclose(
        summary["p_negative"], expected["cohort_shift_p_negative"],
        atol=1e-12, rtol=0.0,
    ):
        raise ValueError("Figure 2F null does not match all-event statistics")
    if not np.isclose(
        raw_null_summary["p_negative"], expected["cohort_shift_p_negative"],
        atol=1e-12, rtol=0.0,
    ):
        raise ValueError("raw-r display null significance does not match the frozen test")

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(3.15, 3.45), facecolor="white")
        grid = fig.add_gridspec(
            2, 1, height_ratios=(1.75, 0.75), left=0.13, right=0.985,
            top=0.94, bottom=0.17, hspace=0.025,
        )
        distribution_ax = fig.add_subplot(grid[0, 0])
        null_ax = fig.add_subplot(grid[1, 0])
        sign_test_p = float(expected["sign_test_p_more_negative_than_half"])
        distribution = _draw_distribution(
            distribution_ax,
            rows,
            sign_test_p=sign_test_p,
        )
        _draw_null(
            null_ax,
            raw_null_r,
            float(distribution["median"]),
            p_negative=float(raw_null_summary["p_negative"]),
        )
        fig.text(
            0.55, 0.035, "Cohort median r vs spatial null",
            ha="center", va="bottom", fontsize=10.0, color="#222222",
        )
        png = figures / "fig2-panelf.png"
        pdf = figures / "fig2-panelf.pdf"
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    field_summary = json.loads((input_root / "cohort_summary.json").read_text())
    payload = {
        "schema_version": "figure2f_all_event_timing_plus_space_v3",
        "template_contract": "timing_plus_space_all_events_missing_view_v1",
        "input_root": str(input_root),
        "distribution": distribution,
        "full_contact_shuffle": summary,
        "display_raw_cohort_median_null": raw_null_summary,
        "reported_statistics": statistics["cohort_summary"],
        "shared_field_change": field_summary["shared_field_change"],
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "display_boundary": (
            "All eligible shared-plane patients are shown without subgroup or example highlighting. "
            "Observed r and spatial nulls use exact contact-evaluated fields, not display pixels."
        ),
        "significance_display": {
            "negative_prevalence": {
                "test": "one-sided exact sign test against 50% negative",
                "p": sign_test_p,
                "symbol": _significance_label(sign_test_p),
            },
            "spatial_null": {
                "test": raw_null_summary["test"],
                "p": float(raw_null_summary["p_negative"]),
                "symbol": _significance_label(float(raw_null_summary["p_negative"])),
            },
            "diamond_alignment": (
                "upper and lower diamonds both equal the observed raw cohort median r"
            ),
            "symbols": "ns: P>=0.05; *: P<0.05; **: P<0.01; ***: P<0.001",
        },
    }
    (figures / "fig2-panelf-metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (figures / "README.md").write_text(
        f"""# Figure 2F — all-event Timing+Space refresh

### fig2-panelf.png / .pdf

全部 {distribution['n']} 名同时具有二维几何和 shared TA/TB 场的患者各贡献一个 signed Pearson r；
{distribution['n_negative']}/{distribution['n']} 为负，中位 r={distribution['median']:+.3f}。
相对 50% 负相关比例的一侧 exact sign test 为 P={sign_test_p:.5g}（{_significance_label(sign_test_p)}）。
上下两个菱形均为同一个 raw cohort median r={raw_null_summary['observed_median_r']:+.3f}；横条为
患者 observed r 的 IQR。下图为全触点打乱后重建 TB 场的 raw cohort-median null，null median r=
{raw_null_summary['null_median_r']:+.3f}，lower-tail permutation P={raw_null_summary['p_negative']:.5g}
（{_significance_label(float(raw_null_summary['p_negative']))}）。冻结的 subject-centered cohort-shift
检验仍完整保留在 metadata，且当前 empirical P 与 raw-r display null 一致。

**关注点**：所有间期事件均进入模板聚类；方向不可估事件只缺失空间视图，不被删除。图中不按
axis cosine、same/reversed 或稳定性分组，也不突出尚未重新冻结的 Figure 2E 示例。
""",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--draws", type=int, default=100_000)
    args = parser.parse_args()
    result = build(args.input_root.resolve(), args.output_root.resolve(), seed=args.seed, draws=args.draws)
    print(json.dumps(result["distribution"], ensure_ascii=False, indent=2))
    print(json.dumps(result["full_contact_shuffle"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
