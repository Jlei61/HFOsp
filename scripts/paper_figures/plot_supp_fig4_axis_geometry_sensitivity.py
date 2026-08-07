#!/usr/bin/env python3
"""Build Supplementary Figure 4 without duplicating the main 2-D field panel.

Panel A is a compact rendering of the accepted directed template-axis half
rose.  Panel B adds the non-overlapping held-out spatial read-back: whether a
path axis fitted from one event half predicts contact rank in the other half.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_figures import plot_fig2_template_axis_direction as angle_fig  # noqa: E402
from src.plot_style import COL_YQ  # noqa: E402


GEOMETRY_JSON = (
    ROOT / "results/spatial_modulation/propagation_geometry/cohort_summary.json"
)
OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig4_axis_geometry"
FIG_DIR = OUT_ROOT / "figures"
COL_EPI_PURPLE = "#7A3E87"


def _heldout_rows() -> list[dict]:
    payload = json.loads(GEOMETRY_JSON.read_text())
    rows = []
    for record in payload["subjects"]:
        path = record.get("path_axis", {})
        rho = path.get("heldout_spearman_rho")
        if rho is None or not np.isfinite(float(rho)):
            continue
        rows.append(
            {
                "dataset": record["dataset"],
                "subject": str(record["subject"]),
                "rho": float(rho),
                "tier": path.get("eligibility_tier"),
                "weak_axis": bool(path.get("weak_axis")),
            }
        )
    if len(rows) != 26:
        raise RuntimeError(f"expected 26 held-out axis subjects, found {len(rows)}")
    return rows


def _draw_compact_half_rose(ax: plt.Axes, rows: list[dict]) -> dict:
    angles = np.asarray([row["directed_angle_deg"] for row in rows], dtype=float)
    edges_deg = np.linspace(0.0, 180.0, angle_fig.N_BINS + 1)
    edges = np.deg2rad(edges_deg)
    centers = edges[:-1] + np.diff(edges) / 2
    counts = np.histogram(angles, bins=edges_deg)[0]
    proportions = counts / len(rows)
    radial_max = max(0.40, np.ceil(proportions.max() / 0.05) * 0.05 * 1.22)
    ax.bar(
        centers,
        proportions,
        width=np.diff(edges) * 0.90,
        facecolor=to_rgba(angle_fig.COHORT_COLOR, 0.28),
        edgecolor=angle_fig.COHORT_COLOR,
        linewidth=0.85,
        zorder=2,
    )
    for center, value, count in zip(centers, proportions, counts):
        ax.text(
            center,
            max(0.048, value * 0.64),
            f"n={int(count)}",
            ha="center",
            va="center",
            fontsize=6.0,
            color=angle_fig.TEXT_COLOR,
        )
    for boundary in (0.0, np.pi):
        ax.plot(
            [boundary, boundary],
            [0, radial_max],
            color=angle_fig.BOUNDARY_COLOR,
            lw=0.9,
            zorder=5,
        )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, radial_max)
    ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
    ax.set_xticklabels(
        ["0°\nsame", "", "", "90°", "", "", "180°\nopposite"],
        fontsize=6.8,
    )
    radial_ticks = [value for value in (0.10, 0.20, 0.30) if value < radial_max]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([])
    for value in radial_ticks:
        ax.text(
            np.deg2rad(7),
            value,
            f"{int(100 * value)}%",
            ha="left",
            va="bottom",
            fontsize=5.6,
            color="0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
            zorder=6,
        )
    ax.grid(color=angle_fig.GRID_COLOR, lw=0.5)
    ax.spines["polar"].set_color("0.55")
    ax.spines["polar"].set_linewidth(0.6)
    q25, median, q75 = np.percentile(angles, [25, 50, 75])
    ax.plot(
        [np.deg2rad(median), np.deg2rad(median)],
        [0.0, radial_max * 0.96],
        color="black",
        lw=1.2,
        zorder=5,
    )
    iqr_angles = np.linspace(np.deg2rad(q25), np.deg2rad(q75), 80)
    ax.plot(
        iqr_angles,
        np.full_like(iqr_angles, radial_max * 0.94),
        color="black",
        lw=2.2,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.2, label="Median direction"),
            Line2D([0], [0], color="black", lw=2.2, label="IQR"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.00, 1.03),
        ncol=2,
        frameon=False,
        fontsize=5.7,
        handlelength=1.7,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )
    return {
        "n": int(len(rows)),
        "bin_edges_deg": edges_deg.tolist(),
        "bin_counts": counts.astype(int).tolist(),
        "median_deg": float(median),
        "iqr_deg": [float(q25), float(q75)],
    }


def _draw_heldout(ax: plt.Axes, rows: list[dict]) -> dict:
    rng = np.random.default_rng(44)
    summaries = {}
    for xpos, dataset, color, label in (
        (0, "yuquan", COL_YQ, "Yuquan"),
        (1, "epilepsiae", COL_EPI_PURPLE, "Epilepsiae"),
    ):
        subset = [row for row in rows if row["dataset"] == dataset]
        values = np.asarray([row["rho"] for row in subset], dtype=float)
        if values.size >= 2:
            violin = ax.violinplot(
                [values],
                positions=[xpos],
                widths=0.30,
                showextrema=False,
                showmedians=False,
            )["bodies"][0]
            violin.set_facecolor(color)
            violin.set_edgecolor(color)
            violin.set_alpha(0.20)
        jitter = rng.uniform(-0.045, 0.045, values.size)
        ax.scatter(
            xpos + jitter,
            values,
            s=19,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.86,
            zorder=3,
        )
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        ax.vlines(xpos, q25, q75, color="black", lw=1.2, zorder=4)
        ax.hlines(median, xpos - 0.12, xpos + 0.12, color="black", lw=2)
        summaries[dataset] = {
            "n": int(values.size),
            "median_spearman_rho": median,
            "iqr_spearman_rho": [float(q25), float(q75)],
        }
    ax.axhline(0.0, color="0.55", lw=0.7)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(-0.28, 1.02)
    ax.set_xticks([0, 1], ["Yuquan", "Epilepsiae"])
    ax.set_ylabel("Held-out $\\rho$", fontsize=8.5, labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.tick_params(labelsize=8)
    for tick, color in zip(ax.get_xticklabels(), (COL_YQ, COL_EPI_PURPLE)):
        tick.set_color(color)
        tick.set_fontweight("bold")
    all_values = np.asarray([row["rho"] for row in rows], dtype=float)
    cohort_q25, cohort_q75 = np.percentile(all_values, [25, 75])
    summaries["cohort"] = {
        "n": int(all_values.size),
        "median_spearman_rho": float(np.median(all_values)),
        "iqr_spearman_rho": [float(cohort_q25), float(cohort_q75)],
    }
    return summaries


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    angle_rows = angle_fig._load_rows()
    heldout = _heldout_rows()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(5.75, 2.48), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.00, 0.78], wspace=0.46,
        left=0.105, right=0.975, bottom=0.20, top=0.85,
    )
    ax_rose = fig.add_subplot(grid[0, 0], projection="polar")
    ax_rho = fig.add_subplot(grid[0, 1])
    rose_summary = _draw_compact_half_rose(ax_rose, angle_rows)
    rho_summary = _draw_heldout(ax_rho, heldout)
    fig.canvas.draw()
    rose_pos = ax_rose.get_position()
    rho_pos = ax_rho.get_position()
    panel_y = max(rose_pos.y1, rho_pos.y1) + 0.075
    fig.text(
        rose_pos.x0 - 0.055,
        panel_y,
        "A",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        rho_pos.x0 - 0.105,
        panel_y,
        "B",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    stem = FIG_DIR / "supp_fig4_axis_direction_and_heldout_readback"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cohort = rho_summary["cohort"]
    caption_title = (
        "Direction and held-out spatial read-back of interictal recruitment axes."
    )
    caption_body = (
        f"**A,** Half-rose distribution of the directed angular separation "
        f"between the spatial axes associated with paired interictal temporal "
        f"templates TA and TB across {rose_summary['n']} patients, where "
        f"0\u00b0 denotes the same direction and 180\u00b0 the opposite direction; "
        f"sector labels give patient counts in 30\u00b0 bins, the radial black "
        f"line marks the median and the outer black arc the interquartile range "
        f"(IQR). **B,** Patient-level Spearman correlation between held-out "
        f"contact recruitment rank and the path-axis coordinate fitted using "
        f"the complementary event half in Yuquan "
        f"(n = {rho_summary['yuquan']['n']} patients) and Epilepsiae "
        f"(n = {rho_summary['epilepsiae']['n']} patients). Points denote "
        f"patients (blue, Yuquan; purple, Epilepsiae), violins show the "
        f"distributions, horizontal black lines mark "
        f"medians and vertical black lines mark IQRs. Across both cohorts, the "
        f"median held-out \u03c1 was {cohort['median_spearman_rho']:.3f} "
        f"(IQR, {cohort['iqr_spearman_rho'][0]:.3f}\u2013"
        f"{cohort['iqr_spearman_rho'][1]:.3f}; n = {cohort['n']} patients)."
    )
    metadata = {
        "figure": "Supplementary Figure 4",
        "caption": (
            f"Supplementary Fig. 4 | {caption_title} "
            f"{caption_body.replace('**', '')}"
        ),
        "main_figure_field_panel_repeated": False,
        "panel_a": {
            "source": (
                "results/interictal_propagation_masked/template_gradient_fields/"
                "axis_cohort.csv"
            ),
            "producer_contract": (
                "scripts/paper_figures/plot_fig2_template_axis_direction.py"
            ),
            **rose_summary,
        },
        "panel_b": {
            "source": str(GEOMETRY_JSON.relative_to(ROOT)),
            "definition": (
                "Spearman correlation between held-out contact rank and the "
                "path-axis coordinate fitted from the complementary event half"
            ),
            "summary": rho_summary,
        },
        "outputs": {
            "png": str(stem.with_suffix(".png").relative_to(ROOT)),
            "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig4_axis_geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig4_axis_direction_and_heldout_readback.png\n\n"
        f"**Supplementary Fig. 4 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        f"**关注点**：B 共 n={cohort['n']}，中位 ρ="
        f"{cohort['median_spearman_rho']:.3f}，IQR "
        f"{cohort['iqr_spearman_rho'][0]:.3f}–"
        f"{cohort['iqr_spearman_rho'][1]:.3f}。"
        "已进入主图的 2D field panel 不在补图重复；完整患者数值进入补充表。\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
