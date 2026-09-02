#!/usr/bin/env python3
"""Render the corrected core-evidence interfaces for Group-Event State.

The v0.3.1 measurements are preserved in the payload as archival diagnostics,
but they cannot populate H1/H2a because the residual H+S estimand was not run.
Empty panels are explicit and are never filled with demonstration data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_group_event_state_core_figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM,
    MM,
    apply_style,
)
from src.topic5_group_event_state.v03.core_evidence import (  # noqa: E402
    HORIZON_MINUTES,
    build_payload,
    load_payload,
    validate_payload,
)


STATE = "#A35E48"       # Morandi rust: candidate state / cohort median
HISTORY = "#6B7280"     # interpretable history / neutral comparator
SHIFT = "#2F5D8A"       # wrong-time state
SUPPORT = "#DDE9E5"     # favourable region
NEUTRAL = "#C7C7C7"
TEXT = "#2B2B2B"
SUBJECT_COLORS = ("#2F5D8A", "#8A5A2F", "#7E6E84", "#5B8A72", "#A35E48")
FIGURE_STEMS = {
    "h1": "group_event_state_h1_future_blocks",
    "h2a": "group_event_state_h2a_repertoire",
    "transfer_feedback": "group_event_state_h2b_h3_transfer_feedback",
}


def _style() -> None:
    apply_style()
    # The older shared helper uses 6.8 pt ticks; this package honours the
    # project-level >=7 pt readability floor requested for final figures.
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
        }
    )


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def _subject_palette(payload: dict[str, Any]) -> dict[str, str]:
    aliases = [row["alias"] for row in payload["training"]]
    return {
        alias: SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
        for i, alias in enumerate(aliases)
    }


def _rows_by_subject(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["alias"]].append(row)
    return grouped


def _tick_labels(
    rows: list[dict[str, Any]], field: str, horizons: tuple[int, ...]
) -> list[str]:
    labels = []
    for horizon in horizons:
        n = sum(
            row["horizon_minutes"] == horizon
            and row.get(field) is not None
            and row.get("eligible", True)
            for row in rows
        )
        name = "2 h" if horizon == 120 else f"{horizon} min"
        labels.append(f"{name}\nn={n}")
    return labels


def _gain_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    *,
    shared_ylim: tuple[float, float] | None = None,
    show_legend: bool = False,
    horizons: tuple[int, ...] = HORIZON_MINUTES,
) -> None:
    x = np.arange(len(horizons), dtype=float)
    values: list[float] = []
    grouped = _rows_by_subject(rows)
    for group_index, (alias, subject_rows) in enumerate(grouped.items()):
        row_lookup = {row["horizon_minutes"]: row for row in subject_rows}
        lookup = {h: row.get(field) for h, row in row_lookup.items()}
        y = np.asarray([lookup.get(h) for h in horizons], dtype=object)
        finite = np.asarray([v is not None and np.isfinite(float(v)) for v in y])
        yf = np.asarray([float(v) if ok else np.nan for v, ok in zip(y, finite)])
        point_offset = (group_index - (len(grouped) - 1) / 2) * 0.075
        for xi, horizon, yi in zip(x, horizons, yf):
            if not np.isfinite(yi) or not row_lookup[horizon].get("eligible", True):
                continue
            values.append(float(yi))
            ax.scatter(
                [xi + point_offset], [yi], s=19,
                facecolor=colors[alias],
                edgecolor=colors[alias], linewidth=0.75, zorder=4,
            )
    medians: list[float] = []
    for horizon in horizons:
        vals = [
            float(row[field])
            for row in rows
            if row["horizon_minutes"] == horizon
            and row.get(field) is not None
            and row.get("eligible", True)
        ]
        medians.append(float(np.median(vals)) if len(vals) >= 2 else np.nan)
    # Do not connect medians across horizons: eligibility changes with horizon.
    for xi, median in zip(x, medians):
        if np.isfinite(median):
            ax.plot([xi - 0.15, xi + 0.15], [median, median], color=STATE, lw=2.1, zorder=5)
    if shared_ylim is None:
        max_abs = max([abs(v) for v in values] + [0.02])
        ylim = (-1.16 * max_abs, 1.16 * max_abs)
    else:
        ylim = shared_ylim
    ax.set_ylim(*ylim)
    ax.axhspan(0, ylim[1], color=SUPPORT, alpha=0.55, zorder=-3)
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.text(
        0.98,
        0.96,
        "favourable  ↑",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#4D766B",
        fontsize=7.0,
    )
    if not values:
        ax.text(
            0.5,
            0.49,
            "not yet run",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=8.0,
            fontweight="bold",
        )
    if len(horizons) == 1:
        ax.set_xlim(-0.42, 0.42)
    ax.set_xticks(x, _tick_labels(rows, field, horizons))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if show_legend and values:
        handles = [
            Line2D([0], [0], color=colors[a], marker="o", lw=0.8, ms=3.2, label=a)
            for a in colors
        ] + [Line2D([0], [0], color=STATE, lw=2, label="median (n≥2)")]
        ax.legend(handles=handles, loc="lower left", ncol=2, handlelength=1.4, columnspacing=0.8)
    _finish_axis(ax)


def _draw_h1_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    event_x = [0.9, 2.2, 3.5, 4.8]
    event_colors = ["#5D7C93", "#8A5A2F", "#5D7C93", "#7E6E84"]
    for x0, color in zip(event_x, event_colors):
        ys = [3.25, 3.7, 4.15]
        ax.scatter([x0] * 3, ys, s=[18, 28, 13], color=color, edgecolor="white", lw=0.45)
        ax.plot([x0, x0], [3.2, 4.2], color=color, lw=0.55, alpha=0.65)
    ax.text(0.7, 4.48, "Past group events", fontsize=6.8, fontweight="bold")
    state_x = np.linspace(0.7, 5.4, 100)
    state_y = 1.9 + 0.45 * np.sin(state_x * 0.95) + 0.13 * state_x
    ax.plot(state_x, state_y, color=STATE, lw=2.0)
    ax.scatter(event_x, np.interp(event_x, state_x, state_y), color=STATE, s=18, zorder=4)
    ax.text(0.7, 0.68, "Candidate state  S(t)", color=STATE, fontsize=6.6, fontweight="bold")
    ax.axvline(5.65, color="#4D4D4D", lw=0.85, ls=(0, (3, 2)))
    ax.add_patch(plt.Rectangle((5.8, 1.0), 3.6, 3.8, facecolor="#EEF1F2", edgecolor="none"))
    future_x = [6.4, 7.05, 8.1, 8.7]
    for x0, n in zip(future_x, [2, 3, 1, 3]):
        ax.scatter([x0] * n, np.linspace(3.7, 4.5, n), s=18, color="#A8B1B6")
    ax.annotate(
        "open-loop",
        xy=(8.9, 2.35),
        xytext=(6.0, 2.35),
        arrowprops=dict(arrowstyle="-|>", color=STATE, lw=1.2),
        color=STATE,
        fontsize=6.5,
        va="center",
    )
    ax.text(6.0, 1.12, "Future block\n5 / 30 / 120 min", fontsize=6.7, fontweight="bold")
    ax.text(
        0.0, 5.82, "State must predict\na future block",
        fontsize=6.8, fontweight="bold", va="top", linespacing=0.90,
    )


def render_h1(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM * MM, 70 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.60)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _draw_h1_schematic(axes[0])
    rows = payload["h1_future_block"]["rows"]
    colors = _subject_palette(payload)
    _gain_panel(
        axes[1], rows, "residual_gain_over_history", "Residual beyond history",
        "H+S gain\nover H", colors, show_legend=True,
    )
    _gain_panel(
        axes[2], rows, "correct_time_gain_over_shifted", "Time-specific state",
        "Correct-state gain\nover shifted state", colors,
    )
    _gain_panel(
        axes[3], rows, "dynamic_gain_over_mean", "Dynamic, not static",
        "Dynamic-state gain\nover mean state", colors,
    )
    for letter, ax in zip("ABCD", axes):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.18, top=0.91)
    if payload.get("status", "").startswith("v0_3_2_pipeline_accepted"):
        fig.text(0.995, 0.025, "Development only; assay power uncalibrated", ha="right", fontsize=7.0, color="#707070")
    return _save(fig, out_dir, FIGURE_STEMS["h1"])


def _draw_h2a_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    xy = np.asarray([[1.0, 3.5], [2.3, 4.25], [3.6, 3.2], [5.0, 4.65], [6.4, 3.05], [8.0, 4.2]])
    ax.plot(xy[:3, 0], xy[:3, 1], color="#5F6B70", lw=2.0)
    ax.scatter(xy[:3, 0], xy[:3, 1], s=32, facecolor="#5F6B70", edgecolor="white", lw=0.5, zorder=3)
    ax.text(0.45, 5.15, "Matched early prefix", fontsize=7.1, fontweight="bold")
    ax.annotate("", xy=(5.0, 5.25), xytext=(3.75, 3.35), arrowprops=dict(arrowstyle="-|>", color=STATE, lw=1.1))
    ax.annotate("", xy=(5.0, 1.65), xytext=(3.75, 3.25), arrowprops=dict(arrowstyle="-|>", color=SHIFT, lw=1.1))
    ax.text(7.45, 6.30, "state A", color=STATE, fontsize=7.2, fontweight="bold")
    ax.text(4.7, 0.72, "state B", color=SHIFT, fontsize=7.2, fontweight="bold")
    upper = np.asarray([[5.0, 5.25], [6.35, 5.8], [7.7, 5.1], [8.9, 5.65]])
    lower = np.asarray([[5.0, 1.65], [6.35, 2.15]])
    ax.plot(upper[:, 0], upper[:, 1], color=STATE, lw=1.7)
    ax.scatter(upper[:, 0], upper[:, 1], s=28, color=STATE, edgecolor="white", lw=0.45)
    ax.plot(lower[:, 0], lower[:, 1], color=SHIFT, lw=1.7)
    ax.scatter(lower[:, 0], lower[:, 1], s=28, color=SHIFT, edgecolor="white", lw=0.45)
    ax.plot([6.8, 8.9], [1.8, 1.8], color="#5F6B70", lw=0.75, ls=(0, (2, 2)))
    ax.text(7.85, 1.28, "STOP", ha="center", color="#5F6B70", fontsize=7.4)
    ax.text(
        0.0,
        6.72,
        "State-dependent\nevent path",
        fontsize=7.3,
        fontweight="bold",
        linespacing=0.95,
        va="top",
    )


def render_h2a(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM * MM, 70 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1], wspace=0.43)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _draw_h2a_schematic(axes[0])
    rows = payload["h2a_repertoire"]["rows"]
    colors = _subject_palette(payload)
    all_values = [
        abs(float(row[field]))
        for row in rows
        for field in ("gain_over_history", "gain_over_shifted")
        if row.get(field) is not None
    ]
    limit = max(max(all_values, default=0.01) * 1.12, 0.01)
    shared_ylim = (-limit, limit)
    specs = (
        ("continue", "Continue vs STOP", "Correct-state\nNLL gain"),
        ("positive_size", "Size | continue", ""),
        ("subset", "Contact subset | size", ""),
    )
    for ax, (endpoint, title, ylabel) in zip(axes[1:], specs):
        subset = [row for row in rows if row["endpoint"] == endpoint]
        ax.axhspan(0, shared_ylim[1], color=SUPPORT, alpha=0.55, zorder=-3)
        ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)), zorder=1)
        x_positions = np.asarray([0.0, 1.0])
        fields = ("gain_over_history", "gain_over_shifted")
        for patient_i, row in enumerate(subset):
            jitter = (patient_i - (len(subset) - 1) / 2) * 0.07
            for xi, field in zip(x_positions, fields):
                value = row.get(field)
                if value is None or not np.isfinite(float(value)):
                    continue
                ax.scatter(
                    xi + jitter,
                    float(value),
                    s=19,
                    color=colors[row["alias"]],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=4,
                )
        for xi, field in zip(x_positions, fields):
            vals = [float(row[field]) for row in subset if row.get(field) is not None]
            if len(vals) >= 2:
                median = float(np.median(vals))
                ax.plot([xi - 0.15, xi + 0.15], [median, median], color=STATE, lw=2.1, zorder=5)
        ax.set_xlim(-0.38, 1.38)
        ax.set_ylim(*shared_ylim)
        ax.set_xticks(x_positions, ["vs H", "vs shifted\n(mean of 5)"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(
            0.98, 0.96, "favourable  ↑", transform=ax.transAxes,
            ha="right", va="top", color="#4D766B", fontsize=7.0,
        )
        if endpoint == "continue":
            handles = [
                Line2D([0], [0], color=colors[a], marker="o", lw=0, ms=3.5, label=a)
                for a in colors
            ]
            ax.legend(handles=handles, loc="lower left", ncol=2, handlelength=0.8, columnspacing=0.7)
        _finish_axis(ax)
    for letter, ax in zip("ABCD", axes):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.042, right=0.99, bottom=0.18, top=0.91)
    if payload.get("status", "").startswith("v0_3_2_pipeline_accepted"):
        fig.text(0.99, 0.025, "Count-trained state; grammar transfer is objective-mismatched", ha="right", fontsize=7.0, color="#707070")
    return _save(fig, out_dir, FIGURE_STEMS["h2a"])


def _empty_or_curves(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    x_values: list[float],
    x_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    positive_support: bool = True,
) -> None:
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)))
    positions = np.arange(len(x_values), dtype=float)
    position_of = {float(value): position for value, position in zip(x_values, positions)}
    ax.set_xlim(-0.12, len(x_values) - 0.88)
    ax.set_xticks(positions, x_labels)
    if rows:
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("alias", row.get("subject", row.get("event_type", "group")))].append(row)
        all_y = []
        for i, (name, group) in enumerate(groups.items()):
            group = sorted(group, key=lambda r: r[x_field])
            x = [position_of[float(r[x_field])] for r in group]
            y = [float(r[y_field]) for r in group]
            all_y.extend(y)
            ax.plot(x, y, marker="o", ms=3.2, lw=0.9, color=SUBJECT_COLORS[i % len(SUBJECT_COLORS)], label=name)
        max_abs = max([abs(v) for v in all_y] + [0.02])
        ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    else:
        ax.set_ylim(-0.12, 0.12)
        ax.text(
            0.5, 0.49, "not yet run", transform=ax.transAxes,
            ha="center", va="center", color="#777777", fontsize=8.2, fontweight="bold",
        )
    if positive_support:
        ax.axhspan(0, ax.get_ylim()[1], color=SUPPORT, alpha=0.55, zorder=-3)
        ax.text(0.98, 0.96, "supports transfer  ↑", transform=ax.transAxes, ha="right", va="top", color="#4D766B", fontsize=7.0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish_axis(ax)


def _feedback_model_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    families = ("count/rate", "mark-specific")
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)))
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], families)
    if rows:
        for i, family in enumerate(families):
            vals = [float(r["future_block_log_score_gain_over_no_feedback"]) for r in rows if r["feedback_family"] == family]
            ax.scatter(np.full(len(vals), i), vals, s=22, color=SUBJECT_COLORS[i], alpha=0.8)
            if vals:
                ax.plot([i - 0.16, i + 0.16], [np.median(vals)] * 2, color=STATE, lw=2)
        all_y = [float(r["future_block_log_score_gain_over_no_feedback"]) for r in rows]
        max_abs = max([abs(v) for v in all_y] + [0.02])
        ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    else:
        ax.set_ylim(-0.12, 0.12)
        ax.text(0.5, 0.49, "not yet run", transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=8.2, fontweight="bold")
    ax.axhspan(0, ax.get_ylim()[1], color=SUPPORT, alpha=0.55, zorder=-3)
    ax.text(0.98, 0.96, "supports feedback  ↑", transform=ax.transAxes, ha="right", va="top", color="#4D766B", fontsize=7.0)
    ax.set_title("IED feedback beyond common state", loc="left", fontweight="bold")
    ax.set_ylabel("Future-block\nlog-score gain")
    _finish_axis(ax)


def render_transfer_feedback(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_MM * MM, 104 * MM))
    risk = payload["h2b_transfer"]["risk_rows"]
    field = payload["h2b_transfer"]["field_rows"]
    _empty_or_curves(
        axes[0, 0], risk, x_field="lead_minutes", y_field="brier_skill_state_over_history",
        x_values=[5, 15, 30, 60, 120, 360], x_labels=["5", "15", "30", "60", "120", "360"],
        title="Seizure-risk transfer", xlabel="Lead to seizure (min)", ylabel="Brier skill\nover history",
    )
    _empty_or_curves(
        axes[0, 1], field, x_field="lead_minutes", y_field="early_ictal_field_gain",
        x_values=[5, 30, 120, 360], x_labels=["5", "30", "120", "360"],
        title="Early-ictal field transfer", xlabel="Lead to seizure (min)", ylabel="Spatial prediction\ngain",
    )
    _feedback_model_panel(axes[1, 0], payload["h3_feedback"]["model_rows"])
    _empty_or_curves(
        axes[1, 1], payload["h3_feedback"]["impulse_rows"],
        x_field="lag_minutes", y_field="functional_state_change",
        x_values=[0, 30, 120, 360], x_labels=["0", "30", "120", "360"],
        title="Event-type impulse response", xlabel="Time after IED (min)", ylabel="Functional-state\nchange",
        positive_support=False,
    )
    for letter, ax in zip("ABCD", axes.flat):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.48)
    return _save(fig, out_dir, FIGURE_STEMS["transfer_feedback"])


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, Any]:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def _write_readme(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / "figures" / "README.md"
    v032 = payload.get("status", "").startswith("v0_3_2")
    if v032:
        h1_focus = (
            "只绘制事前资格合格的患者；不同 horizon 不连线，n=1 不画 cohort median，n=0 留空。"
            "当前 positive-recovery power 尚未定标，因此人体数值只能说明这版 count-trained representation 的观测表现。"
        )
        h2_focus = (
            "状态来自 30 分钟 count 任务并已冻结；grammar 只训练低容量 residual adapter。"
            "主图分别显示相对 H 与相对五个 shift 均值的增量；test-best-control 只保留在机器结果中作为敏感性，不再承担主结论。"
        )
    else:
        h1_focus = (
            "v0.3.1 没有运行这三个 residual 对比，所以当前 panel 留空；"
            "旧 S-vs-H 数字只保存在 payload 的 archival diagnostics 中，不能填入主图。"
        )
        h2_focus = (
            "v0.3.1 的 state path 未可靠训练且缺少 H+S 配对，因此当前留空；"
            "最终需要 contact subset 与 same-prefix continuation 的患者级增量稳定高于零。"
        )
    path.write_text(
        "# Group-Event State core evidence\n\n"
        "### group_event_state_h1_future_blocks.png\n\n"
        "这张图回答 H1：在显式多尺度历史 H 已经进入每个模型之后，动态状态 S 是否还对未来 5、30、120 分钟的事件块提供增量。B 比较 H+S 与 H，C 比较正确时刻与 block-shifted S，D 比较动态 S 与 TRAIN 均值 S；纵轴均为正值支持 residual state。\n\n"
        f"**关注点**：{h1_focus}\n\n"
        "### group_event_state_h2a_repertoire.png\n\n"
        "这张图回答 H2a：给定相同或相近的事件开头，H+S_correct 是否胜过 H，并处于 block-shifted state 的有利方向，进而改变事件继续/停止、继续时的招募规模以及具体触点集合。三个统计 panel 共用 y 轴。\n\n"
        f"**关注点**：{h2_focus}\n\n"
        "### group_event_state_h2b_h3_transfer_feedback.png\n\n"
        "这张图预先固定跨任务与反馈机制的最终接口。A/B 分别放冻结间期状态对发作风险和发作早期空间场的增量；C 比较 no-feedback、count/rate feedback 与 mark-specific feedback；D 显示不同 IED 类型的有符号状态冲击。当前这些实验尚未运行，因此只显示坐标、对照方向和 not yet run，不填模拟数据。\n\n"
        "**关注点**：H2b 必须以 held-out seizure 为分母；H3 必须先控制共同 pre-event state，且冲击允许正负方向。\n"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/data/hfosp_group_event_state_v0_3/pilot/"
            "summary_v0_3_1_closeout.json"
        ),
    )
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/group_event_state/core_evidence",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.payload is not None:
        payload = load_payload(args.payload)
        source_summary = None
    else:
        summary = json.loads(args.summary.read_text())
        payload = build_payload(summary)
        source_summary = str(args.summary)
    validate_payload(payload)
    payload_path = args.output_root / "core_evidence_payload.json"
    _atomic_json(payload_path, payload)
    files = {
        "h1": render_h1(payload, args.output_root),
        "h2a": render_h2a(payload, args.output_root),
        "h2b_h3": render_transfer_feedback(payload, args.output_root),
    }
    readme = _write_readme(args.output_root, payload)
    metadata = {
        "asset_id": "group_event_state_core_evidence",
        "paper_slot": "TBD",
        "status": payload.get("status", "CANDIDATE_FRAMEWORK"),
        "payload_format": payload["format"],
        "payload": str(payload_path),
        "source_summary": source_summary,
        "source_commit": payload["source"]["source_commit"],
        "figures": files,
        "readme": str(readme),
        "font_floor_pt": 7.0,
        "statistical_grammar": "patient-first; positive gain supports hypothesis; missing is not zero",
        "no_synthetic_observations_in_unrun_panels": True,
        "claim_boundary": payload["claim_boundary"],
    }
    metadata_path = args.output_root / "core_evidence_metadata.json"
    _atomic_json(metadata_path, metadata)
    print(json.dumps({"metadata": str(metadata_path), "figures": files}, indent=2))


if __name__ == "__main__":
    main()
