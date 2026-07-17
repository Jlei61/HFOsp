#!/usr/bin/env python3
"""Nature-style cohort panel for the directed TA/TB propagation-axis angle."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields/axis_cohort.csv"
OUT = ROOT / "results/paper-ready-figure/fig2_template_axis_direction/figures"
TEAL = "#287D78"
GREY = "#777777"


def _directed_angle_deg(cosine: float) -> float:
    """0=same early-to-late direction; 180=opposed early-to-late direction."""
    return float(np.degrees(np.arccos(np.clip(float(cosine), -1.0, 1.0))))


def _load_rows() -> list[dict]:
    rows = []
    for raw in csv.DictReader(INPUT.open()):
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
        raise ValueError("expected 26 subjects with supported 2D geometry")
    return rows


def _swarm_y(angles: np.ndarray, *, minimum_dx: float = 7.0, step: float = 0.13) -> np.ndarray:
    """Deterministic 1D beeswarm without introducing a categorical y-axis."""
    levels = [0]
    for level in range(1, 8):
        levels.extend([level, -level])
    assigned: list[tuple[float, int]] = []
    output = np.zeros(len(angles), dtype=float)
    for index in np.argsort(angles):
        angle = float(angles[index])
        for level in levels:
            if all(not (used_level == level and abs(angle - used_angle) < minimum_dx)
                   for used_angle, used_level in assigned):
                output[index] = level * step
                assigned.append((angle, level))
                break
        else:
            raise RuntimeError("swarm packing exhausted available levels")
    return output


def _plot(rows: list[dict], png: Path, pdf: Path) -> dict:
    angles = np.asarray([row["directed_angle_deg"] for row in rows], dtype=float)
    y = _swarm_y(angles)
    geometry_2d = np.asarray([row["geometry_2d_supported"] for row in rows], dtype=bool)
    q25, median, q75 = np.percentile(angles, [25, 50, 75])

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.0,
        "xtick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.50, 2.18))
        ax.axvline(90.0, color="0.68", linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
        ax.scatter(
            angles[geometry_2d],
            y[geometry_2d],
            s=31,
            facecolors=TEAL,
            edgecolors="white",
            linewidths=0.65,
            alpha=0.96,
            zorder=3,
        )
        ax.scatter(
            angles[~geometry_2d],
            y[~geometry_2d],
            s=34,
            facecolors="white",
            edgecolors=GREY,
            linewidths=1.0,
            zorder=4,
        )

        summary_y = -0.48
        ax.plot([q25, q75], [summary_y, summary_y], color="0.22", linewidth=3.2,
                solid_capstyle="round", zorder=2)
        ax.plot([median, median], [summary_y - 0.07, summary_y + 0.07], color="white",
                linewidth=3.0, zorder=3)
        ax.plot([median, median], [summary_y - 0.07, summary_y + 0.07], color="black",
                linewidth=1.15, zorder=4)
        ax.text(median, summary_y - 0.115, f"median {median:.0f}°", ha="center", va="top",
                fontsize=6.5, color="0.22")

        ax.set_xlim(-5.0, 185.0)
        ax.set_ylim(-0.72, 0.58)
        ax.set_xticks([0.0, 90.0, 180.0])
        ax.set_xticklabels(["0°\nsame", "90°\northogonal", "180°\nopposed"])
        ax.set_yticks([])
        ax.set_xlabel(r"Directed TA–TB axis angle, $\theta_{AB}$")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        handles = [
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                          markerfacecolor=TEAL, markeredgecolor="white",
                          label="2D geometry (n=26)"),
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                          markerfacecolor="white", markeredgecolor=GREY,
                          label="single-shaft (n=2)"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.3,
                  handletextpad=0.35, borderaxespad=0.15, labelspacing=0.25)
        fig.subplots_adjust(left=0.07, right=0.94, top=0.96, bottom=0.29)
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    return {
        "n_axis_estimable": len(rows),
        "n_geometry_2d_supported": int(np.sum(geometry_2d)),
        "n_single_shaft": int(np.sum(~geometry_2d)),
        "directed_angle_definition": "degrees(arccos(cos(u_TA, u_TB))); 0=same, 180=opposed",
        "median_deg": float(median),
        "iqr_deg": [float(q25), float(q75)],
        "range_deg": [float(np.min(angles)), float(np.max(angles))],
        "visual_encoding": {
            "filled_teal": "geometry_2d_supported",
            "open_grey": "single-shaft; direction retained but 2D geometry unsupported",
            "horizontal_black_bar": "cohort IQR",
            "vertical_black_tick": "cohort median",
            "vertical_dashed_line": "90-degree orthogonal reference only; not a classifier",
        },
        "subjects": rows,
    }


def main() -> None:
    rows = _load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "template_axis_directed_angle_cohort.png"
    pdf = OUT / "template_axis_directed_angle_cohort.pdf"
    metadata = _plot(rows, png, pdf)
    (OUT / "template_axis_directed_angle_cohort_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"[done] wrote {png}")
    print(f"[done] wrote {pdf}")


if __name__ == "__main__":
    main()
