#!/usr/bin/env python3
"""Figure 2 cohort half-rose for the directed template-axis pair angle.

The geometry and typography follow the accepted Figure R-B rose panels, but
the encoding is deliberately cohort-neutral: one muted color represents the
subject-count distribution, without borrowing the template-specific TA/TB
red/blue semantics.  The directed 3-D angle is intrinsically defined on
0--180 degrees and is never mirrored into an artificial full circle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields/axis_cohort.csv"
OUT = ROOT / "results/paper-ready-figure/fig2_template_axis_direction/figures"

# Cohort-level encoding: intentionally distinct from the locked TA/TB red/blue palette.
COHORT_COLOR = "#4F7F78"
BOUNDARY_COLOR = "#333333"
GRID_COLOR = "#D4D4D4"
TEXT_COLOR = "#222222"
N_BINS = 6


def _directed_angle_deg(cosine: float) -> float:
    """Return 0=same early-to-late direction and 180=opposed direction."""
    return float(np.degrees(np.arccos(np.clip(float(cosine), -1.0, 1.0))))


def _load_rows() -> list[dict]:
    rows = []
    with INPUT.open() as handle:
        for raw in csv.DictReader(handle):
            if raw.get("status") != "ok" or raw.get("axis_pair_estimable") != "True":
                continue
            cosine = float(raw["cos_ta_tb"])
            rows.append(
                {
                    "subject_id": raw["subject_id"],
                    "dataset": raw["dataset"],
                    "cos_ta_tb": cosine,
                    "directed_angle_deg": _directed_angle_deg(cosine),
                    "geometry_2d_supported": raw.get("geometry_2d_supported") == "True",
                    "strict_stability_pass": raw.get("strict_stability_pass") == "True",
                }
            )
    if len(rows) != 28:
        raise ValueError(f"expected 28 axis-estimable subjects, found {len(rows)}")
    if sum(row["geometry_2d_supported"] for row in rows) != 26:
        raise ValueError("expected 26 subjects with supported 2-D geometry")
    return rows


def _legend_handles() -> list[object]:
    return [
        mpatches.Patch(
            facecolor=to_rgba(COHORT_COLOR, 0.28), edgecolor=COHORT_COLOR,
            linewidth=0.9, label="Patients per 30° bin",
        ),
    ]


def _plot_legend(png: Path, pdf: Path) -> None:
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(1.72, 0.32), facecolor="white")
        fig.legend(
            handles=_legend_handles(),
            loc="center",
            ncol=1,
            frameon=False,
            fontsize=6.0,
            handlelength=1.45,
            handletextpad=0.35,
        )
        fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _plot(rows: list[dict], png: Path, pdf: Path) -> dict:
    angles_deg = np.asarray([row["directed_angle_deg"] for row in rows], dtype=float)
    geometry_2d = np.asarray([row["geometry_2d_supported"] for row in rows], dtype=bool)
    edges_deg = np.linspace(0.0, 180.0, N_BINS + 1)
    edges_rad = np.deg2rad(edges_deg)
    centers = edges_rad[:-1] + 0.5 * np.diff(edges_rad)
    bin_counts = np.histogram(angles_deg, bins=edges_deg)[0]
    proportions = bin_counts.astype(float) / len(rows)
    histogram_top = float(np.ceil(max(proportions.max(), 0.05) / 0.05) * 0.05)
    radial_max = float(max(0.40, histogram_top * 1.38))

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(3.50, 2.18), facecolor="white")
        ax = fig.add_subplot(111, projection="polar")

        ax.bar(
            centers,
            proportions,
            width=np.diff(edges_rad) * 0.92,
            facecolor=to_rgba(COHORT_COLOR, 0.28),
            edgecolor=COHORT_COLOR,
            linewidth=0.9,
            align="center",
            zorder=2,
        )

        for center, proportion, count in zip(centers, proportions, bin_counts):
            ax.text(
                center,
                max(0.055, proportion * 0.66),
                f"n={int(count)}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=TEXT_COLOR,
                zorder=8,
            )

        for boundary in (0.0, np.pi):
            ax.plot(
                [boundary, boundary],
                [0.0, radial_max],
                color=BOUNDARY_COLOR,
                linewidth=1.0,
                solid_capstyle="butt",
                zorder=7,
            )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0.0)
        ax.set_thetamax(180.0)
        ax.set_ylim(0.0, radial_max)
        ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
        ax.set_xticklabels(["", "", "", "", "", "", ""])
        ax.set_yticks(np.arange(0.10, histogram_top + 1e-9, 0.10))
        ax.set_yticklabels([])
        ax.grid(color=GRID_COLOR, linewidth=0.55, alpha=0.95)
        ax.spines["polar"].set_color("#777777")
        ax.spines["polar"].set_linewidth(0.65)

        fig.text(0.50, 0.745, "90°\northogonal", ha="center", va="bottom",
                 fontsize=7.0, color=TEXT_COLOR)
        fig.text(0.055, 0.205, "180°", ha="left", va="top",
                 fontsize=7.0, color=TEXT_COLOR)
        fig.text(0.945, 0.205, "0°", ha="right", va="top",
                 fontsize=7.0, color=TEXT_COLOR)
        ax.set_position([0.12, 0.015, 0.76, 0.88])
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    q25, median, q75 = np.percentile(angles_deg, [25, 50, 75])
    return {
        "n_axis_estimable": len(rows),
        "n_geometry_2d_supported": int(np.sum(geometry_2d)),
        "n_single_shaft": int(np.sum(~geometry_2d)),
        "directed_angle_definition": "degrees(arccos(cos(u_TA, u_TB))); 0=same, 180=opposed",
        "display_geometry": "directed half-rose on the intrinsic 0-to-180-degree support; no artificial mirroring",
        "bin_edges_deg": edges_deg.tolist(),
        "bin_counts": bin_counts.astype(int).tolist(),
        "bin_proportions": proportions.tolist(),
        "median_deg": float(median),
        "iqr_deg": [float(q25), float(q75)],
        "range_deg": [float(np.min(angles_deg)), float(np.max(angles_deg))],
        "visual_encoding": {
            "muted_teal_bars": "cohort distribution of the directed template-axis pair angle",
            "individual_subject_points": "not shown because the rose histogram already encodes the cohort distribution",
            "zero_and_180_boundaries": "neutral black; no TA/TB color semantics",
            "bar_labels": "raw subject count per 30-degree bin",
            "classification_or_shading": "none",
        },
        "style_source": "Figure R-B rose geometry and typography adapted to a neutral cohort-level encoding",
        "subjects": rows,
    }


def main() -> None:
    rows = _load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "template_axis_directed_angle_cohort.png"
    pdf = OUT / "template_axis_directed_angle_cohort.pdf"
    legend_png = OUT / "template_axis_directed_angle_cohort_legend.png"
    legend_pdf = OUT / "template_axis_directed_angle_cohort_legend.pdf"
    metadata = _plot(rows, png, pdf)
    _plot_legend(legend_png, legend_pdf)
    (OUT / "template_axis_directed_angle_cohort_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"[done] wrote {png}")
    print(f"[done] wrote {pdf}")
    print(f"[done] wrote {legend_png}")
    print(f"[done] wrote {legend_pdf}")


if __name__ == "__main__":
    main()
