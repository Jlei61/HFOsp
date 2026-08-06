#!/usr/bin/env python3
"""Paper-ready Figure 2 last row: shared-axis TA/TB field reversal.

The left block reuses the canonical frozen-interictal field renderer to show
four transparently selected, strongly negative TA/TB examples.  The right
block contains all shared-axis patients with supported 2D geometry and the
cohort-median-shift distribution obtained from the already-computed full-
contact TB-payload shuffle.  No axis, plane, bandwidth or field is refit here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    DEFAULT_DISPLAY_SIGMA_MM,
    DEFAULT_YUQUAN_CROSSWALK,
    TA_COLOR,
    TB_COLOR,
    _display_name,
    _load_yuquan_crosswalk,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
from scripts.plot_topic5_template_field_ta_tb_contact_scatter import _zscore  # noqa: E402
from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (  # noqa: E402
    _pretty as _manuscript_id,
)


INPUT_ROOT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUT = ROOT / "results/paper-ready-figure/fig2_shared_field_reversal"
DEFAULT_NULL_DRAWS = INPUT_ROOT / "shared_field_similarity_null_draws.npz"
DEFAULT_STATISTICS = INPUT_ROOT / "shared_field_similarity_statistics.json"
N_EXAMPLES = 4
EXAMPLE_SUBJECT_IDS = (
    "epilepsiae_384",
    "epilepsiae_548",
    "epilepsiae_583",
    "yuquan_zhaochenxi",
)
COMMON_DISPLAY_X_SPAN_MM = 50.0
COMMON_DISPLAY_Y_SPAN_MM = 60.0
COHORT_COLOR = "#4F7F78"
POINT_COLOR = "#6F95AC"
EXAMPLE_COLOR = "#254F65"
NULL_COLOR = "#B8B8B8"


def _portable_path(path: Path) -> str:
    """Repository-relative in production, absolute for isolated test outputs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _stable_seed(text: str, base_seed: int) -> int:
    """Match the frozen shared-field-null producer's deterministic seed."""
    digest = hashlib.sha256(f"{base_seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def load_shared_field_rows(
    input_root: Path,
    *,
    yuquan_labels: Mapping[str, str],
) -> list[dict]:
    """Load the complete shared-axis + supported-2D cohort."""
    rows: list[dict] = []
    for path in sorted((input_root / "per_subject").glob("*.json")):
        record = json.loads(path.read_text())
        pair = record.get("axis_pair") or {}
        field = record.get("interictal_field") or {}
        models = field.get("field_models") or {}
        if (
            record.get("status") != "ok"
            or not pair.get("geometry_2d_supported")
            or "shared_a" not in models
            or "shared_b" not in models
        ):
            continue
        ta = _zscore(np.asarray(models["shared_a"]["template_field"], dtype=float))
        tb = _zscore(np.asarray(models["shared_b"]["template_field"], dtype=float))
        if ta.shape != tb.shape or ta.shape != (len(field["contact_order"]),):
            raise ValueError(f"field/contact mismatch for {record['subject_id']}")
        r_value = float(np.corrcoef(ta, tb)[0, 1])
        subject_id = str(record["subject_id"])
        display_id = _manuscript_id(subject_id)
        if subject_id.startswith("yuquan_"):
            private_crosswalk_id = _display_name(subject_id, yuquan_labels)
            if display_id != private_crosswalk_id:
                raise ValueError(
                    f"Yuquan manuscript/private crosswalk mismatch for {subject_id}: "
                    f"{display_id} vs {private_crosswalk_id}"
                )
        rows.append(
            {
                "subject_id": subject_id,
                "display_id": display_id,
                "record": record,
                "n_contacts": int(len(ta)),
                "r": r_value,
            }
        )
    if len(rows) != 12:
        raise ValueError(
            f"expected 12 shared-axis patients with supported 2D geometry, found {len(rows)}"
        )
    return sorted(rows, key=lambda row: float(row["r"]))


def load_channel_nulls(path: Path, rows: Sequence[dict]) -> dict[str, np.ndarray]:
    """Load the frozen full-contact shuffle draws for exactly this cohort."""
    with np.load(path) as archive:
        nulls = {
            str(row["subject_id"]): np.asarray(
                archive[f"{row['subject_id']}__channel"], dtype=float
            )
            for row in rows
        }
    if any(len(values) < 20 or not np.isfinite(values).all() for values in nulls.values()):
        raise ValueError("invalid or under-resolved channel-shuffle draws")
    return nulls


def select_examples(rows: Sequence[dict], n_examples: int = N_EXAMPLES) -> list[dict]:
    """Locked legible negative-field examples; cohort inference remains all-subject."""
    if n_examples != len(EXAMPLE_SUBJECT_IDS):
        raise ValueError("the paper-ready example count is locked to four")
    by_id = {str(row["subject_id"]): row for row in rows}
    missing = [subject_id for subject_id in EXAMPLE_SUBJECT_IDS if subject_id not in by_id]
    if missing:
        raise ValueError(f"locked shared-field examples are unavailable: {missing}")
    examples = [by_id[subject_id] for subject_id in EXAMPLE_SUBJECT_IDS]
    if any(float(row["r"]) >= 0 for row in examples):
        raise ValueError("locked example set unexpectedly includes non-negative r")
    return examples


def _restore_compact_axis_ticks(ax: plt.Axes) -> None:
    """Restore sparse physical-mm ticks after the canonical compact painter."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="both", labelsize=5.6, length=2.0, width=0.6, pad=1.0)


def apply_common_display_window(
    payload_a: dict,
    payload_b: dict,
    *,
    x_span_mm: float = COMMON_DISPLAY_X_SPAN_MM,
    y_span_mm: float = COMMON_DISPLAY_Y_SPAN_MM,
) -> None:
    """Crop shared-plane examples to one physical display scale without refitting."""
    all_x = np.concatenate([np.asarray(payload_a["xs"], float), np.asarray(payload_b["xs"], float)])
    all_y = np.concatenate([np.asarray(payload_a["ys"], float), np.asarray(payload_b["ys"], float)])
    required_x = float(np.ptp(all_x))
    required_y = float(np.ptp(all_y))
    if required_x > x_span_mm or required_y > y_span_mm:
        raise ValueError(
            "common display window would crop contacts: "
            f"required={required_x:.2f}x{required_y:.2f} mm, "
            f"available={x_span_mm:.2f}x{y_span_mm:.2f} mm"
        )
    center_x = 0.5 * (float(np.min(all_x)) + float(np.max(all_x)))
    center_y = 0.5 * (float(np.min(all_y)) + float(np.max(all_y)))
    xlim = (center_x - 0.5 * x_span_mm, center_x + 0.5 * x_span_mm)
    ylim = (center_y - 0.5 * y_span_mm, center_y + 0.5 * y_span_mm)
    for payload in (payload_a, payload_b):
        payload["frame"] = {**payload["frame"], "xlim": xlim, "ylim": ylim}


def build_cohort_shift_null(
    rows: Sequence[dict],
    channel_nulls: Mapping[str, np.ndarray],
    *,
    n_draws: int,
    base_seed: int,
) -> tuple[dict, np.ndarray]:
    """Reproduce the frozen hierarchical cohort-median-shift randomization."""
    ordered = sorted(rows, key=lambda row: str(row["subject_id"]))
    observed = np.asarray([row["r"] for row in ordered], dtype=float)
    arrays = [np.asarray(channel_nulls[str(row["subject_id"])], dtype=float) for row in ordered]
    centers = np.asarray([np.median(values) for values in arrays], dtype=float)
    observed_delta = float(np.median(observed - centers))
    rng = np.random.default_rng(_stable_seed("cohort:channel", base_seed))
    # Keep the producer's 5,000-draw batching so the persisted Monte-Carlo P
    # value is bit-reproducible rather than merely asymptotically equivalent.
    batches = []
    for start in range(0, n_draws, 5_000):
        n_batch = min(5_000, n_draws - start)
        samples = np.column_stack(
            [values[rng.integers(0, len(values), size=n_batch)] for values in arrays]
        )
        batches.append(np.median(samples - centers[None, :], axis=1))
    null_delta = np.concatenate(batches)
    p_value = float((1 + np.sum(null_delta <= observed_delta)) / (n_draws + 1))
    summary = {
        "n_subjects": len(ordered),
        "n_draws": int(n_draws),
        "observed_median_delta": observed_delta,
        "null_median": float(np.median(null_delta)),
        "null_q025": float(np.percentile(null_delta, 2.5)),
        "null_q975": float(np.percentile(null_delta, 97.5)),
        "p_negative": p_value,
        "test": (
            "hierarchical full-contact-shuffle randomization of the centered "
            "subject-level cohort median; lower tail"
        ),
    }
    return summary, null_delta


def draw_patient_distribution(
    ax: plt.Axes,
    rows: Sequence[dict],
    examples: Sequence[dict],
) -> dict:
    """All-patient observed signed correlations, ordered without subgrouping."""
    ordered = sorted(rows, key=lambda row: float(row["r"]))
    values = np.asarray([row["r"] for row in ordered], dtype=float)
    example_ids = {str(row["subject_id"]) for row in examples}
    ypos = np.arange(len(ordered), dtype=float)

    ax.axvspan(-1.0, 0.0, color="#EFF4F3", zorder=0)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.75, zorder=1)
    for value, y, row in zip(values, ypos, ordered):
        is_example = str(row["subject_id"]) in example_ids
        ax.plot([0.0, value], [y, y], color="#CFD8DC", linewidth=0.65, zorder=1)
        ax.scatter(
            [value],
            [y],
            s=30 if is_example else 23,
            facecolors=EXAMPLE_COLOR if is_example else POINT_COLOR,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        align = "right" if value < 0 else "left"
        offset = -0.035 if value < 0 else 0.035
        ax.text(
            value + offset,
            y,
            str(row["display_id"]),
            ha=align,
            va="center",
            fontsize=5.8 if is_example else 5.5,
            color=EXAMPLE_COLOR if is_example else "#526874",
            fontweight="bold" if is_example else "normal",
        )

    q25, median, q75 = np.percentile(values, [25, 50, 75])
    summary_y = -1.35
    ax.plot(
        [q25, q75],
        [summary_y, summary_y],
        color=COHORT_COLOR,
        linewidth=4.2,
        alpha=0.35,
        solid_capstyle="round",
        zorder=2,
    )
    ax.scatter(
        [median], [summary_y], marker="D", s=27,
        facecolor=COHORT_COLOR, edgecolor="white", linewidth=0.65, zorder=5,
    )
    ax.text(
        0.02,
        0.98,
        f"{int(np.sum(values < 0))}/{len(values)} negative\nmedian r={median:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#222222",
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.9, len(ordered) - 0.35)
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


def draw_cohort_null(
    ax: plt.Axes,
    null_delta: np.ndarray,
    summary: Mapping[str, object],
) -> None:
    """Full-contact-shuffle cohort-median-shift density and observed shift."""
    counts, edges = np.histogram(null_delta, bins=45, range=(-0.8, 0.8), density=True)
    counts = np.convolve(counts, np.asarray([1, 2, 3, 2, 1]) / 9.0, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = counts / max(float(np.max(counts)), 1e-12)
    observed = float(summary["observed_median_delta"])
    ax.fill_between(centers, 0.0, counts, color=NULL_COLOR, alpha=0.75, linewidth=0)
    ax.plot(centers, counts, color="#8A8A8A", linewidth=0.75)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.7)
    ax.axvline(observed, color=COHORT_COLOR, linewidth=1.6)
    ax.scatter(
        [observed], [0.0], marker="D", s=28,
        facecolor=COHORT_COLOR, edgecolor="white", linewidth=0.6,
        clip_on=False, zorder=5,
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=6.2, length=2.2, pad=1.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.65)


def build_figure(
    rows: Sequence[dict],
    *,
    channel_nulls: Mapping[str, np.ndarray],
    out_dir: Path,
    seed: int = 20260721,
    n_cohort_draws: int = 100_000,
    expected_statistics: Mapping[str, object] | None = None,
) -> dict:
    examples = select_examples(rows)
    null_summary, null_delta = build_cohort_shift_null(
        rows,
        channel_nulls,
        n_draws=n_cohort_draws,
        base_seed=seed,
    )
    if expected_statistics is not None:
        frozen = expected_statistics["cohort_summary"]["channel"]
        frozen_p = float(frozen["cohort_shift_p_negative"])
        if not np.isclose(null_summary["p_negative"], frozen_p, atol=1e-12, rtol=0):
            raise ValueError(
                "reconstructed channel-shuffle cohort P does not match frozen statistics: "
                f"{null_summary['p_negative']} vs {frozen_p}"
            )
        paired_wilcoxon_p = float(
            frozen["paired_wilcoxon_observed_less_than_null_median_p"]
        )
        within_shaft_p = float(
            expected_statistics["cohort_summary"]["within_shaft"]["cohort_shift_p_negative"]
        )
    else:
        paired_wilcoxon_p = float("nan")
        within_shaft_p = float("nan")

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.15, 3.05), facecolor="white")
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=(4.65, 2.05),
            left=0.086,
            right=0.992,
            top=0.925,
            bottom=0.155,
            wspace=0.12,
        )
        fields_grid = outer[0, 0].subgridspec(
            2, 5, width_ratios=(1.0, 1.0, 1.0, 1.0, 0.065), wspace=0.27, hspace=0.22,
        )
        field_axes: list[plt.Axes] = []
        for column, row in enumerate(examples):
            dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
                row["record"], display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM
            )
            if mode != "shared":
                raise ValueError(f"example {row['subject_id']} is not rendered on a shared plane")
            apply_common_display_window(dat_a, dat_b)
            ax_a = fig.add_subplot(fields_grid[0, column])
            ax_b = fig.add_subplot(fields_grid[1, column], sharex=ax_a, sharey=ax_a)
            draw_interictal_rank_field_panel(
                ax_a,
                dat_a,
                "TA",
                compact=True,
                panel_title=str(row["display_id"]),
                contact_outline_lw=0.78,
                contact_size=28,
                show_template_tag=False,
            )
            draw_interictal_rank_field_panel(
                ax_b,
                dat_b,
                "TB",
                compact=True,
                contact_outline_lw=0.78,
                contact_size=28,
                show_template_tag=False,
            )
            ax_a.set_title(
                str(row["display_id"]), fontsize=8.6, fontweight="bold",
                color="#222222", pad=2.5,
            )
            ax_b.set_title("")
            _restore_compact_axis_ticks(ax_a)
            _restore_compact_axis_ticks(ax_b)
            if column == 0:
                ax_a.set_ylabel("y (mm)", fontsize=6.5, color="#222222", labelpad=0.5)
                ax_b.set_ylabel("y (mm)", fontsize=6.5, color="#222222", labelpad=0.5)
                ax_a.text(
                    -0.53, 0.5, "TA field", transform=ax_a.transAxes,
                    ha="center", va="center", rotation=90, fontsize=7.0,
                    fontweight="bold", color=TA_COLOR, clip_on=False,
                )
                ax_b.text(
                    -0.53, 0.5, "TB field", transform=ax_b.transAxes,
                    ha="center", va="center", rotation=90, fontsize=7.0,
                    fontweight="bold", color=TB_COLOR, clip_on=False,
                )
            field_axes.extend([ax_a, ax_b])

        cbar_ax = fig.add_subplot(fields_grid[:, 4])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
            cax=cbar_ax,
            orientation="vertical",
        )
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["0 (early)", "1 (late)"])
        colorbar.ax.tick_params(labelsize=6.0, length=2.0, pad=1.5)

        right_grid = outer[0, 1].subgridspec(2, 1, height_ratios=(1.55, 0.75), hspace=0.025)
        distribution_ax = fig.add_subplot(right_grid[0, 0])
        null_ax = fig.add_subplot(right_grid[1, 0])
        distribution_summary = draw_patient_distribution(distribution_ax, rows, examples)
        draw_cohort_null(null_ax, null_delta, null_summary)

        fig.canvas.draw()
        field_left = min(ax.get_position().x0 for ax in field_axes)
        field_right = max(ax.get_position().x1 for ax in field_axes)
        field_bottom = min(ax.get_position().y0 for ax in field_axes)
        field_top = max(ax.get_position().y1 for ax in field_axes)
        cbar_position = cbar_ax.get_position()
        cbar_ax.set_position(
            [cbar_position.x0, field_bottom, cbar_position.width, field_top - field_bottom]
        )
        adjusted_cbar_position = cbar_ax.get_position()
        fig.text(
            0.5 * (adjusted_cbar_position.x0 + adjusted_cbar_position.x1),
            field_top + 0.026,
            "Normalized\nranks",
            ha="center",
            va="bottom",
            fontsize=6.3,
            linespacing=0.95,
            color="#222222",
        )
        common_xlabel_y = 0.030
        fig.text(
            0.5 * (field_left + field_right),
            common_xlabel_y,
            "Shared TA axis (mm)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#222222",
        )
        null_position = null_ax.get_position()
        fig.text(
            0.5 * (null_position.x0 + null_position.x1),
            common_xlabel_y,
            "TA–TB reversal vs spatial null (Δr)",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#222222",
        )

        figures = out_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        png = figures / "fig2_shared_field_reversal_last_row.png"
        pdf = figures / "fig2_shared_field_reversal_last_row.pdf"
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    null_npz = out_dir / "fig2_shared_field_reversal_cohort_null.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(null_npz, channel_cohort_median_shift=null_delta)
    metadata = {
        "figure_role": "Figure 2 last-row candidate: shared-axis TA/TB field reversal",
        "cohort": "shared_field_available and geometry_2d_supported",
        "estimand": (
            "signed Pearson r across exact contact-evaluated shared_a/shared_b template fields"
        ),
        "grouping": "none",
        "example_selection": (
            "four locked negative-correlation cases selected for legible 2D geometry and visible "
            "field reversal; E958 excluded for dense/tall geometry and E1146 excluded because it "
            "is already used elsewhere in Figure 2"
        ),
        "example_display_window_mm": {
            "x_span": COMMON_DISPLAY_X_SPAN_MM,
            "y_span": COMMON_DISPLAY_Y_SPAN_MM,
            "display_only": True,
        },
        "examples": [
            {
                "subject_id": row["subject_id"],
                "display_id": row["display_id"],
                "r": float(row["r"]),
                "n_contacts": int(row["n_contacts"]),
            }
            for row in examples
        ],
        "distribution": distribution_summary,
        "full_contact_shuffle": null_summary,
        "statistical_sensitivities": {
            "paired_wilcoxon_observed_less_than_subject_null_median_p": paired_wilcoxon_p,
            "within_shaft_cohort_shift_p_negative": within_shaft_p,
        },
        "subjects": [
            {
                "subject_id": row["subject_id"],
                "display_id": row["display_id"],
                "r": float(row["r"]),
                "n_contacts": int(row["n_contacts"]),
            }
            for row in rows
        ],
        "display_boundary": (
            "Left maps reuse the canonical 6-mm Viridis display renderer on each frozen shared "
            "plane; observed r and the null use the exact frozen contact-evaluated fields, not "
            "display-grid pixels. The four maps use a common 50x60-mm display window centered on "
            "their contact extents; this display crop does not alter axes, contacts, ranks or scoring. "
            "Examples are locked morphology illustrations selected among "
            "negative cases for legible geometry; the all-subject distribution carries the cohort view. The "
            "P value is the hierarchical full-contact-shuffle cohort-median-shift test; paired "
            "Wilcoxon and within-shaft results remain sensitivity statistics."
        ),
        "outputs": {
            "png": _portable_path(png),
            "pdf": _portable_path(pdf),
            "cohort_null_npz": _portable_path(null_npz),
        },
    }
    metadata_path = out_dir / "figures/fig2_shared_field_reversal_last_row_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def _write_readme(out_dir: Path, metadata: Mapping[str, object]) -> None:
    distribution = metadata["distribution"]
    null = metadata["full_contact_shuffle"]
    sensitivity = metadata["statistical_sensitivities"]
    example_text = "、".join(
        f"{row['display_id']} (r={row['r']:+.2f})" for row in metadata["examples"]
    )
    text = f"""# Fig2 shared-field reversal 最后一行候选

### fig2_shared_field_reversal_last_row.png / .pdf

左侧复用冻结间期场的统一画图函数，在相同 shared plane 上成对展示 TA/TB Viridis rank field；
横轴是冻结 shared TA propagation axis，显示带宽固定为 6 mm；四人统一裁到以各自触点范围为中心的
50 × 60 mm display-only 窗口，轴、触点、rank、kernel 与统计均不改变。4个显示案例锁定为
{example_text}；它们均为负相关且二维几何易读。E958 因触点过密、图形瘦长而排除，E1146 因已在
Figure 2 前文出现而不重复。这些图用于直观说明场的反向形态，不是独立抽样验证。

右上纳入全部 {distribution['n']} 名 shared-axis 且二维几何有效的患者，不按 axis cosine、
same/reversed 标签或 strict-stability 分组。当前 {distribution['n_negative']}/{distribution['n']}
为负，中位 r={distribution['median']:+.3f}。右下是 TB earliness 与 support 在全部触点间联合打乱、
重建 TB field 后的层级 cohort-median-shift null；观测 Δmedian={null['observed_median_delta']:+.3f}，
lower-tail permutation P={null['p_negative']:.5f}。

**关注点**：图内不直接写精确 P；caption/metadata 中的 P 只对应全触点打乱的 cohort 中位移位检验。
逐患者 observed-vs-null-median 的配对 Wilcoxon 为
P={sensitivity['paired_wilcoxon_observed_less_than_subject_null_median_p']:.5f}，
within-shaft cohort sensitivity 为 P={sensitivity['within_shaft_cohort_shift_p_negative']:.5f}；
因此正文安全口径是“cohort median 比全触点随机化更负”，不能泛化成所有 null 或多数单患者均显著。
"""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--null-draws", type=Path, default=DEFAULT_NULL_DRAWS)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--yuquan-crosswalk", type=Path, default=DEFAULT_YUQUAN_CROSSWALK)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-cohort-draws", type=int, default=100_000)
    args = parser.parse_args()
    yuquan_labels = _load_yuquan_crosswalk(args.yuquan_crosswalk)
    rows = load_shared_field_rows(args.input_root, yuquan_labels=yuquan_labels)
    channel_nulls = load_channel_nulls(args.null_draws, rows)
    statistics = json.loads(args.statistics.read_text())
    metadata = build_figure(
        rows,
        channel_nulls=channel_nulls,
        out_dir=args.output_dir,
        seed=args.seed,
        n_cohort_draws=args.n_cohort_draws,
        expected_statistics=statistics,
    )
    _write_readme(args.output_dir, metadata)
    print(f"[done] wrote {metadata['outputs']['png']}")
    print(
        f"[done] shared-2D n={metadata['distribution']['n']}, "
        f"negative={metadata['distribution']['n_negative']}, "
        f"median={metadata['distribution']['median']:+.3f}, "
        f"channel-shuffle P={metadata['full_contact_shuffle']['p_negative']:.5f}"
    )


if __name__ == "__main__":
    main()
