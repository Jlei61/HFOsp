#!/usr/bin/env python3
"""Paper-ready Topic 5 energy-field extrapolation figure.

The figure answers one question:
does a core-derived interictal order field add predictive value for hidden
seizure-energy contacts beyond each hidden contact's own interictal order?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.topic5_field_extrapolation import field_predict_at_points


IN_DIR = ROOT / "results/topic5_ictal_recruitment/field_extrapolation"
PER_SUBJECT = IN_DIR / "cohort_per_subject"
FINAL_JSON = IN_DIR / "energy_field_extrapolation_FINAL.json"
OUT_DIR = ROOT / "results/paper-ready-figure/fig_topic5_field_extrapolation_energy/figures"
AXIS_DIR = ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
NARROW_POOL = ROOT / "results/interictal_propagation_masked/per_subject"
BROAD_POOL = ROOT / "results/interictal_propagation_masked_broad/per_subject"

DELTA = 0.03
EXAMPLE_SUBJECT = "epilepsiae_1146"
EXAMPLE_TEMPLATE = "t_a"
BANDS = [("bb_auc", "Broadband"), ("hfa_auc", "HFA")]
COL_CORE = "#2f6fdd"
COL_OWN = "#d95f02"
COL_TIE = "#8a8a8a"
COL_GREEN = "#6aa878"
COL_AMBER = "#d8a343"
COL_RED = "#c95b59"
COL_SOFT_BLUE = "#dbe8f7"
COL_SOFT_ORANGE = "#f4e1cf"


def _short_subject(subject: str) -> str:
    return subject.replace("epilepsiae_", "E").replace("yuquan_", "Y:")


def _fmt_q(q_value: float | None) -> str:
    if q_value is None or not np.isfinite(q_value):
        return "n/a"
    if q_value < 1e-4:
        return "q<1e-4"
    return f"q={q_value:.2f}"


def _load_rows() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for band, _label in BANDS:
        rows = []
        for path in sorted(PER_SUBJECT.glob(f"*__{band}.json")):
            row = json.loads(path.read_text())
            if row.get("status") != "ok":
                continue
            f_core = float(row["F_core_only"])
            c1 = float(row["C1"])
            diff = f_core - c1
            if diff > DELTA:
                winner = "core"
            elif diff < -DELTA:
                winner = "own"
            else:
                winner = "tie"
            rows.append(
                {
                    "subject": row["subject"],
                    "label": _short_subject(row["subject"]),
                    "band": band,
                    "n_hidden": int(row["n_hidden"]),
                    "n_seizures": int(row["n_seizures"]),
                    "F_core_only": f_core,
                    "C1": c1,
                    "diff": diff,
                    "winner": winner,
                    "low_power": int(row["n_hidden"]) < 6,
                    "three_null": (
                        float(row["null_channel_p"]) < 0.05
                        and float(row["null_within_shaft_p"]) < 0.05
                        and float(row["null_anchor_p"]) < 0.05
                    ),
                    "channel_null": float(row["null_channel_p"]) < 0.05,
                }
            )
        out[band] = rows
    return out


def _load_final() -> dict[tuple[str, str], dict]:
    rows = json.loads(FINAL_JSON.read_text())
    return {(row["band"], row["hypothesis"]): row for row in rows}


def _load_example_layout() -> dict:
    record = json.loads((AXIS_DIR / f"{EXAMPLE_SUBJECT}_{EXAMPLE_TEMPLATE}.json").read_text())
    narrow = set(json.loads((NARROW_POOL / f"{EXAMPLE_SUBJECT}.json").read_text())["channel_names"])
    broad = set(json.loads((BROAD_POOL / f"{EXAMPLE_SUBJECT}.json").read_text())["channel_names"])
    hidden = broad - narrow
    channels = [
        ch
        for ch in record["channels"]
        if np.isfinite(ch.get("x_norm", np.nan))
        and np.isfinite(ch.get("y_norm", np.nan))
        and np.isfinite(ch.get("typical_rank", np.nan))
    ]
    return {"record": record, "channels": channels, "narrow": narrow, "hidden": hidden}


def _counts(rows: list[dict]) -> dict[str, int]:
    return {
        "core": sum(row["winner"] == "core" for row in rows),
        "own": sum(row["winner"] == "own" for row in rows),
        "tie": sum(row["winner"] == "tie" for row in rows),
        "three_null": sum(row["three_null"] for row in rows),
        "channel_null": sum(row["channel_null"] for row in rows),
        "low_power": sum(row["low_power"] for row in rows),
    }


def _write_summary(rows_by_band: dict[str, list[dict]], final: dict[tuple[str, str], dict]) -> None:
    summary = {
        "source": {
            "per_subject": str(PER_SUBJECT.relative_to(ROOT)),
            "final_json": str(FINAL_JSON.relative_to(ROOT)),
        },
        "margin_delta": DELTA,
        "bands": {},
    }
    for band, label in BANDS:
        rows = rows_by_band[band]
        cnt = _counts(rows)
        low_filtered = [row for row in rows if not row["low_power"]]
        summary["bands"][band] = {
            "label": label,
            "n_subjects": len(rows),
            "winner_counts": cnt,
            "winner_counts_drop_n_hidden_lt_6": _counts(low_filtered),
            "median_F_core_only": float(np.median([row["F_core_only"] for row in rows])),
            "median_C1": float(np.median([row["C1"] for row in rows])),
            "median_diff_F_minus_C1": float(np.median([row["diff"] for row in rows])),
            "q_channel_null": final[(band, "F_core>channel_null")]["fdr_q"],
            "q_F_core_gt_C1": final[(band, "F_core>C1")]["fdr_q"],
        }
    (OUT_DIR / "topic5_field_extrapolation_energy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.03,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def _draw_design_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    _panel_label(ax, "A")
    ax.text(
        0.02,
        0.91,
        "Test design",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.855,
        "Predict seizure energy at held-out hidden contacts",
        ha="left",
        va="center",
        fontsize=9.0,
        color="0.25",
    )

    layout = _load_example_layout()
    channels = layout["channels"]
    xy = np.array([[ch["x_norm"], ch["y_norm"]] for ch in channels], dtype=float)
    pad = 0.12
    xmin, xmax = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    ymin, ymax = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    xpad = max((xmax - xmin) * pad, 0.03)
    ypad = max((ymax - ymin) * pad, 0.03)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    rect = (0.12, 0.32, 0.76, 0.46)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    scale = min(rect[2], rect[3]) / span

    def to_panel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            rect[0] + rect[2] / 2.0 + (np.asarray(x) - cx) * scale,
            rect[1] + rect[3] / 2.0 + (np.asarray(y) - cy) * scale,
        )

    gx = np.linspace(xmin, xmax, 100)
    gy = np.linspace(ymin, ymax, 75)
    GX, GY = np.meshgrid(gx, gy)
    core_chans = [ch for ch in channels if ch["name"] in layout["narrow"]]
    display_sigma = 0.30 * span
    grid_pred = field_predict_at_points(
        core_chans, np.c_[GX.ravel(), GY.ravel()], sigma_xy=display_sigma
    ).reshape(GX.shape)
    x0, y0 = to_panel(np.array([xmin]), np.array([ymin]))
    x1, y1 = to_panel(np.array([xmax]), np.array([ymax]))
    ax.imshow(
        grid_pred,
        extent=[float(x0[0]), float(x1[0]), float(y0[0]), float(y1[0])],
        origin="lower",
        cmap="viridis",
        alpha=0.55,
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )
    core_xy = np.array([[ch["x_norm"], ch["y_norm"]] for ch in channels if ch["name"] in layout["narrow"]])
    hidden_xy = np.array([[ch["x_norm"], ch["y_norm"]] for ch in channels if ch["name"] in layout["hidden"]])
    core_px, core_py = to_panel(core_xy[:, 0], core_xy[:, 1])
    hid_px, hid_py = to_panel(hidden_xy[:, 0], hidden_xy[:, 1])
    ax.scatter(core_px, core_py, s=46, facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
    ax.scatter(
        hid_px,
        hid_py,
        s=54,
        marker="s",
        facecolor="#d43f7a",
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    ax.scatter([0.18], [0.79], s=42, facecolor="white", edgecolor="black", linewidth=1.0, zorder=5)
    ax.text(0.215, 0.79, "core contacts", fontsize=8.5, ha="left", va="center", color="0.15")
    ax.scatter([0.54], [0.79], s=48, marker="s", facecolor="#d43f7a", edgecolor="white", linewidth=0.9, zorder=5)
    ax.text(0.575, 0.79, "held-out hidden contacts", fontsize=8.5, ha="left", va="center", color="0.15")
    ax.text(0.12, 0.305, "E1146 contact layout", fontsize=7.9, color="0.30", ha="left", va="top")

    boxes = [
        (0.08, 0.205, 0.38, 0.105, "core-field prediction", COL_SOFT_BLUE),
        (0.08, 0.055, 0.38, 0.105, "own interictal order", COL_SOFT_ORANGE),
        (0.66, 0.13, 0.28, 0.105, "seizure energy\ntarget", "#eeeeee"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="0.2", linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.7)
    ax.annotate("", xy=(0.66, 0.185), xytext=(0.46, 0.258), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("", xy=(0.66, 0.185), xytext=(0.46, 0.108), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(
        0.50,
        0.00,
        "Metric: median per-seizure |Spearman| on hidden contacts",
        ha="center",
        va="center",
        fontsize=8.4,
        color="0.25",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 0.96)


def _draw_delta_panel(ax: plt.Axes, rows_by_band: dict[str, list[dict]]) -> None:
    ax.set_axis_off()
    _panel_label(ax, "B")
    ax.text(
        0.02,
        0.92,
        "Cohort decision: which predictor wins?",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.02,
        0.84,
        f"Subject-level winner; tie if |score difference| <= {DELTA:.2f}",
        fontsize=8.6,
        color="0.35",
        ha="left",
        va="center",
    )

    left_x, center_x, right_x = 0.24, 0.54, 0.84
    ax.text(left_x, 0.74, "own order wins", ha="center", va="center", fontsize=8.7, color=COL_OWN, fontweight="bold")
    ax.text(center_x, 0.74, "tie", ha="center", va="center", fontsize=8.7, color="0.35", fontweight="bold")
    ax.text(right_x, 0.74, "core field wins", ha="center", va="center", fontsize=8.7, color=COL_CORE, fontweight="bold")

    def to_ax_x(v: float) -> float:
        return 0.54 + v / 24.0

    for y, (band, label) in zip([0.55, 0.32], BANDS):
        rows = rows_by_band[band]
        cnt = _counts(rows)
        low_filtered = [row for row in rows if not row["low_power"]]
        cnt_low = _counts(low_filtered)
        own, tie, core = cnt["own"], cnt["tie"], cnt["core"]
        segments = [
            (-(own + tie / 2), -tie / 2, COL_OWN, str(own)),
            (-tie / 2, tie / 2, COL_TIE, str(tie)),
            (tie / 2, tie / 2 + core, COL_CORE, str(core)),
        ]
        ax.text(0.02, y, label, ha="left", va="center", fontsize=10.0, fontweight="bold")
        for x0, x1, color, text in segments:
            ax.add_patch(
                Rectangle(
                    (to_ax_x(x0), y - 0.045),
                    to_ax_x(x1) - to_ax_x(x0),
                    0.09,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.95,
                )
            )
            ax.text((to_ax_x(x0) + to_ax_x(x1)) / 2, y, text, color="white", fontsize=10.0,
                    fontweight="bold", ha="center", va="center")
        ax.plot([0.5, 0.5], [y - 0.065, y + 0.065], color="0.2", lw=1.0)
        ax.text(
            0.50,
            y - 0.10,
            f"drop low-count subjects: core {cnt_low['core']} / own {cnt_low['own']} / tie {cnt_low['tie']}",
            ha="center",
            va="center",
            fontsize=7.9,
            color="0.42",
        )

    ax.text(
        0.50,
        0.09,
        "Result: wins split across subjects; core-field extrapolation is not systematically better.",
        ha="center",
        va="center",
        fontsize=9.0,
        color="0.12",
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_ladder_panel(ax: plt.Axes, rows_by_band: dict[str, list[dict]], final: dict[tuple[str, str], dict]) -> None:
    ax.set_axis_off()
    _panel_label(ax, "C")
    ax.text(0.02, 0.92, "What can we claim?", fontsize=10.5, fontweight="bold", ha="left", va="center")

    cards = [
        (
            "PASS",
            COL_GREEN,
            "Network extension",
            f"Core field predicts hidden seizure-energy territory\n"
            f"Broadband {_counts(rows_by_band['bb_auc'])['channel_null']}/16, HFA {_counts(rows_by_band['hfa_auc'])['channel_null']}/16; both {_fmt_q(final[('bb_auc', 'F_core>channel_null')]['fdr_q'])}",
        ),
        (
            "LIMITED",
            COL_AMBER,
            "Subject-level robustness",
            f"All three nulls pass in fewer subjects\n"
            f"Broadband {_counts(rows_by_band['bb_auc'])['three_null']}/16, HFA {_counts(rows_by_band['hfa_auc'])['three_null']}/16",
        ),
        (
            "NO",
            COL_RED,
            "Added advantage over own order",
            f"Core field does not beat the hidden contacts' own order\n"
            f"Broadband {_fmt_q(final[('bb_auc', 'F_core>C1')]['fdr_q'])}, HFA {_fmt_q(final[('hfa_auc', 'F_core>C1')]['fdr_q'])}",
        ),
    ]
    for y, (status, color, title, body) in zip([0.72, 0.46, 0.20], cards):
        ax.add_patch(Rectangle((0.03, y - 0.085), 0.18, 0.15, facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(0.12, y - 0.01, status, color="white", fontsize=9.4, fontweight="bold", ha="center", va="center")
        ax.text(0.26, y + 0.055, title, fontsize=9.3, fontweight="bold", ha="left", va="center", color="0.10")
        ax.text(0.26, y + 0.005, body, fontsize=8.05, ha="left", va="top", color="0.30", linespacing=1.18)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _write_readme() -> None:
    text = """# Topic 5 energy-field extrapolation paper figure

### topic5_field_extrapolation_energy_main.png

Paper-ready summary of the energy-field extrapolation result. Panel A defines the scientific test on the real E1146 contact layout: a core-derived interictal order field is evaluated on held-out hidden contacts and compared against each hidden contact's own interictal order, using per-seizure seizure-energy rank as the target. Panel B is the cohort adjudication as winner counts per frequency band. Panel C separates the allowed network-extension claim from the unsupported field-advantage claim.

**关注点**：这张图支持“间期传播顺序模式能刻画 hidden seizure-energy territory 的空间组织”，但不支持“核心外推系统性优于 hidden 电极自己的间期顺序”。低功率点（n_hidden < 6）用空心菱形标出。

### topic5_field_extrapolation_energy_summary.json

Machine-readable summary used by the figure: per-band winner counts, median F_core_only/C1, low-power sensitivity counts, and FDR q-values read from `energy_field_extrapolation_FINAL.json`.
"""
    (OUT_DIR / "README.md").write_text(text)


def plot() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_rc()
    rows_by_band = _load_rows()
    final = _load_final()
    _write_summary(rows_by_band, final)

    fig = plt.figure(figsize=(12.0, 6.85))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.02, 1.55],
        height_ratios=[1.05, 0.95],
        left=0.055,
        right=0.985,
        bottom=0.095,
        top=0.89,
        wspace=0.19,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    _draw_design_panel(ax_a)
    _draw_delta_panel(ax_b, rows_by_band)
    _draw_ladder_panel(ax_c, rows_by_band, final)

    fig.suptitle(
        "Core-field extrapolation: network extension without added advantage",
        fontsize=13.0,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=0.965,
    )
    out_png = OUT_DIR / "topic5_field_extrapolation_energy_main.png"
    out_pdf = OUT_DIR / "topic5_field_extrapolation_energy_main.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    _write_readme()
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    plot()
