#!/usr/bin/env python3
"""Per-subject contact distributions of frozen TA versus TB own fields.

Each panel uses one 2D-geometry-supported subject.  Every point is one contact;
contacts on the same shaft share a colour.  TA/TB field vectors are z-scored
within subject for display only, so the slope and Pearson correlation are
unchanged while axes remain comparable across patients.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_ab_fields import (
    DEFAULT_YUQUAN_CROSSWALK,
    _display_name,
    _load_yuquan_crosswalk,
)


DEFAULT_INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figures"
RELATION_ORDER = ("reversed", "same", "different")
RELATION_LABELS = {
    "reversed": "Reversed collinear",
    "same": "Same-direction collinear",
    "different": "Different-axis",
}
RELATION_COLORS = {"reversed": "#2F7F72", "same": "#A97828", "different": "#667B8A"}
SHAFT_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#EE8866",
    "#44AA99",
    "#999933",
    "#882255",
    "#117733",
)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("cannot standardize a constant field")
    return (values - float(np.mean(values))) / sd


def _load_rows(input_root: Path, yuquan_labels: dict[str, str]) -> list[dict]:
    rows = []
    for path in sorted((input_root / "per_subject").glob("*.json")):
        record = json.loads(path.read_text())
        pair = record.get("axis_pair") or {}
        if record.get("status") != "ok" or not pair.get("geometry_2d_supported"):
            continue
        field = record["interictal_field"]
        models = field["field_models"]
        ta = _zscore(np.asarray(models["own_a"]["template_field"], dtype=float))
        tb = _zscore(np.asarray(models["own_b"]["template_field"], dtype=float))
        if ta.shape != tb.shape or ta.shape != (len(field["contact_order"]),):
            raise ValueError(f"field/contact shape mismatch for {record['subject_id']}")
        relation = pair["relation"]["relation"]
        rows.append(
            {
                "subject_id": record["subject_id"],
                "display_id": _display_name(record["subject_id"], yuquan_labels),
                "relation": relation,
                "ta": ta,
                "tb": tb,
                "shafts": [str(value) for value in field["shafts"]],
                "n_contacts": int(len(ta)),
                "r": float(np.corrcoef(ta, tb)[0, 1]),
            }
        )
    if len(rows) != 26:
        raise ValueError(f"expected 26 geometry_2d_supported subjects, found {len(rows)}")
    return rows


def draw_contact_field_scatter(
    ax: plt.Axes,
    row: dict,
    limit: float,
    *,
    line_color: str | None = None,
    title_color: str | None = None,
    title: str | None = None,
    annotation: str | None = None,
    point_size: float = 29,
    title_size: float = 9.3,
    annotation_size: float = 7.2,
) -> None:
    """Draw one exact contact-field TA-vs-TB scatter.

    This public painter is shared by the diagnostic atlas and the paper-ready
    Figure 2 cohort row.  ``row['ta']``/``row['tb']`` are the exact contact-
    evaluated fields used to calculate ``row['r']``; no field is rebuilt here.
    """
    ta = row["ta"]
    tb = row["tb"]
    relation_color = str(
        line_color or RELATION_COLORS.get(str(row.get("relation")), "#3E6F8E")
    )
    heading_color = str(title_color or relation_color)
    ax.plot([-limit, limit], [-limit, limit], color="0.82", linewidth=0.8, linestyle=":", zorder=0)
    ax.plot([-limit, limit], [limit, -limit], color="0.82", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(0.0, color="0.9", linewidth=0.65, zorder=0)
    ax.axvline(0.0, color="0.9", linewidth=0.65, zorder=0)

    shaft_order = list(dict.fromkeys(row["shafts"]))
    shaft_color = {
        shaft: SHAFT_COLORS[index % len(SHAFT_COLORS)] for index, shaft in enumerate(shaft_order)
    }
    for shaft in shaft_order:
        keep = np.asarray([value == shaft for value in row["shafts"]], dtype=bool)
        ax.scatter(
            ta[keep],
            tb[keep],
            s=point_size,
            facecolors=shaft_color[shaft],
            edgecolors="white",
            linewidths=0.75,
            alpha=0.92,
            zorder=3,
        )

    if len(ta) >= 3 and np.ptp(ta) > 0:
        slope, intercept = np.polyfit(ta, tb, 1)
        xline = np.asarray([-limit, limit], dtype=float)
        ax.plot(
            xline,
            intercept + slope * xline,
            color=relation_color,
            linewidth=1.55,
            alpha=0.9,
            zorder=2,
        )

    ax.text(
        0.04,
        0.95,
        annotation or f"r={row['r']:+.2f} · n={row['n_contacts']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_size,
        color="0.18",
    )
    ax.set_title(
        str(title or row["display_id"]),
        fontsize=title_size,
        fontweight="bold",
        color=heading_color,
        pad=2.0,
    )
    for spine in ax.spines.values():
        spine.set_color(relation_color)
        spine.set_linewidth(1.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)


def _draw_subject(ax: plt.Axes, row: dict, limit: float) -> None:
    """Backward-compatible atlas wrapper."""
    draw_contact_field_scatter(ax, row, limit)


def plot_atlas(rows: list[dict], out_png: Path, out_pdf: Path) -> None:
    grouped = {
        relation: sorted(
            [row for row in rows if row["relation"] == relation], key=lambda row: row["r"]
        )
        for relation in RELATION_ORDER
    }
    expected = {"reversed": 7, "same": 5, "different": 14}
    counts = {relation: len(grouped[relation]) for relation in RELATION_ORDER}
    if counts != expected:
        raise ValueError(f"unexpected relation counts: {counts}")

    maximum = max(float(np.max(np.abs(np.concatenate([row["ta"], row["tb"]])))) for row in rows)
    limit = max(2.4, math.ceil(maximum * 2.0) / 2.0)
    fig = plt.figure(figsize=(13.4, 8.5))
    grid = fig.add_gridspec(
        4,
        7,
        left=0.075,
        right=0.995,
        top=0.94,
        bottom=0.085,
        hspace=0.48,
        wspace=0.32,
    )
    layout = {
        "reversed": [(0, column) for column in range(7)],
        "same": [(1, column) for column in range(5)],
        "different": [(2 + index // 7, index % 7) for index in range(14)],
    }
    axes = {}
    for relation in RELATION_ORDER:
        for row, (grid_row, grid_column) in zip(grouped[relation], layout[relation]):
            ax = fig.add_subplot(grid[grid_row, grid_column])
            _draw_subject(ax, row, limit)
            axes[(grid_row, grid_column)] = ax
    for grid_column in (5, 6):
        ax = fig.add_subplot(grid[1, grid_column])
        ax.axis("off")

    row_y = {"reversed": 0.985, "same": 0.748, "different": 0.512}
    row_x = {"reversed": 0.535, "same": 0.39, "different": 0.535}
    for relation in RELATION_ORDER:
        fig.text(
            row_x[relation],
            row_y[relation],
            f"{RELATION_LABELS[relation]}  ·  n={len(grouped[relation])}",
            ha="center",
            va="top",
            fontsize=11.2,
            fontweight="bold",
            color=RELATION_COLORS[relation],
        )
    fig.supxlabel("TA own field (z)", fontsize=11.2, y=0.025)
    fig.supylabel("TB own field (z)", fontsize=11.2, x=0.022)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yuquan-crosswalk", type=Path, default=DEFAULT_YUQUAN_CROSSWALK)
    args = parser.parse_args()
    yuquan_labels = _load_yuquan_crosswalk(args.yuquan_crosswalk)
    rows = _load_rows(args.input_root, yuquan_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / "template_field_ta_tb_contact_scatter_atlas.png"
    out_pdf = args.output_dir / "template_field_ta_tb_contact_scatter_atlas.pdf"
    plot_atlas(rows, out_png, out_pdf)
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {out_pdf}")


if __name__ == "__main__":
    main()
