#!/usr/bin/env python3
"""Render v0.3.8 decision figures from the frozen machine summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_group_event_state_v038_figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.topic5_epi_prssm.figure_style import apply_style


GREEN = "#245C3F"
GREEN_LIGHT = "#DCE9E1"
BLUE = "#355C7D"
RUST = "#A35E48"
AMBER = "#D4A72C"
GREY = "#8A8F93"
BLACK = "#252525"
FAMILY_COLOR = {"event": BLUE, "grid": AMBER, "dual": GREEN}
ENDPOINTS = ("count", "burden", "community", "coupling", "mixture", "embedding", "mark")
ENDPOINT_LABEL = ("count", "burden", "community", "coupling", "mixture", "embedding", "rich mark")
INPUT_METADATA: dict[str, Any] = {}


def _style() -> None:
    apply_style()
    plt.rcParams.update({
        "font.size": 7.3, "axes.labelsize": 7.5, "axes.titlesize": 8.2,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5, "savefig.transparent": False,
    })


def _panel(ax: plt.Axes, title: str, letter: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")


def _save(fig: plt.Figure, directory: Path, name: str, metadata: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    (directory / f"{name}.metadata.json").write_text(
        json.dumps({**metadata, **INPUT_METADATA}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def _h1_rows(summary: dict[str, Any], cohort_name: str | None = None) -> list[dict[str, Any]]:
    cohorts = summary['h1'].values() if cohort_name is None else [summary['h1'][cohort_name]]
    return [row for cohort in cohorts for family in cohort.values()
            for row in family["rows"]]


def _horizon_curve(ax: plt.Axes, rows: list[dict[str, Any]], field: str,
                   title: str, letter: str) -> None:
    horizons = sorted({row["horizon_seconds"] for row in rows})
    x = np.arange(len(horizons))
    all_values = []
    for family in ("event", "grid", "dual"):
        medians = []
        for horizon in horizons:
            values = [row["evidence"][field]["median"] for row in rows
                      if row["family"] == family and row["horizon_seconds"] == horizon]
            values = [float(value) for value in values if value is not None]
            medians.append(np.median(values) if values else np.nan)
            all_values.extend(values)
        ax.plot(x, medians, marker="o", ms=3.8, lw=1.5,
                color=FAMILY_COLOR[family], label=family)
        for row in rows:
            if row["family"] != family:
                continue
            value = row["evidence"][field]["median"]
            if value is None:
                continue
            xi = horizons.index(row["horizon_seconds"])
            ax.scatter(xi, value, s=8, color=FAMILY_COLOR[family], alpha=0.22)
            all_values.append(float(value))
    bound = max([abs(value) for value in all_values] + [0.01]) * 1.12
    ax.axhspan(0, bound, color=GREEN_LIGHT, alpha=0.45, zorder=-5)
    ax.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax.set_ylim(-bound, bound)
    ax.set_xticks(x, [f"{value / 3600:g} h" for value in horizons])
    ax.set_xlabel("future horizon")
    ax.set_ylabel("patient-level score gain")
    _panel(ax, title, letter)


def render_h1(summary: dict[str, Any], directory: Path) -> None:
    _style()
    rows = _h1_rows(summary, 'long_0.5_2_6_8h')
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.5))
    _horizon_curve(axes[0, 0], rows, "gain_over_strong_baseline",
                   "Beyond strong multiscale history", "A")
    # A constant or matched-random arm nested above the strong baseline can
    # score worse than the baseline itself; the raw contrast then credits the
    # state with the control's harm (0.55 -> 0.12 for the short-scale
    # candidate).  Panel B reads the floored contrast when the summary carries
    # it and says so when it has to fall back.
    floored = any(
        "dynamic_over_constant_floored" in (row.get("evidence") or {}) for row in rows
    )
    _horizon_curve(axes[0, 1], rows,
                   "dynamic_over_constant_floored" if floored else "dynamic_over_constant",
                   "Beyond a FIT-period level (null floored at baseline)" if floored
                   else "Beyond a FIT-period level (NULL NOT FLOORED)", "B")
    _horizon_curve(axes[1, 0], rows, "correct_time_over_shifted",
                   "Correct time vs shifted state", "C")
    credit = summary["trained_long_credit"]["rows"]
    matrix = np.asarray([
        [row["endpoint_credit"][endpoint]["fraction_beyond_6h"]["median"]
         for endpoint in ENDPOINTS] for row in credit
    ], dtype=float) if credit else np.zeros((0, len(ENDPOINTS)))
    if matrix.size:
        cmap = plt.get_cmap('YlGn').copy(); cmap.set_bad('#D9D9D9')
        image = axes[1, 1].imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0,
                                  vmax=max(0.05, float(np.nanmax(matrix))))
        fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.02,
                     label="gradient fraction >6 h")
    axes[1, 1].set_xticks(range(len(ENDPOINTS)), ENDPOINT_LABEL, rotation=35, ha="right")
    axes[1, 1].set_yticks(range(len(credit)), [row["subject"].replace("epilepsiae_", "E") for row in credit])
    _panel(axes[1, 1], "Event-only / 8 h head: credit >6 h", "D")
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    _save(fig, directory, "group_event_state_v038_dynamic_state_and_credit", {
        "question": "Does a trained human observer contain time-specific information and use multi-hour history?",
        "panels": {
            "A": "increment beyond strong marked history",
            "B": "increment beyond a FIT-period constant state",
            "C": "correct-time state versus phase-preserving shift",
            "D": "gradient fraction from held-out human endpoint losses to events older than six hours; gray when no audited anchors",
        },
        "positive_direction": "above zero or larger green intensity",
        "statistical_unit": "patient; seeds are optimisation repeats",
        'h1_cohort': 'long_0.5_2_6_8h only; medium cohort plotted separately',
        'credit_source': 'legacy event-only checkpoints, 8 h prediction head; not dual H1 evidence',
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.6))
    for ax, field, title, letter in zip(axes,
            ('gain_over_strong_baseline', 'dynamic_over_constant_floored', 'correct_time_over_shifted'),
            ('Beyond history', 'Beyond FIT constant', 'Correct vs shifted time'), ('A', 'B', 'C')):
        _horizon_curve(ax, _h1_rows(summary, 'medium_0.5_2h'), field, title, letter)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, directory, 'group_event_state_v038_medium_dynamic_state', {
        'h1_cohort': 'medium_0.5_2h only', 'long_cohort_pooled': False,
        'interpretation': 'patient medians are descriptive; gates require joint within-seed controls',
    })


def _symmetric_limit(matrix: np.ndarray) -> float:
    finite = np.abs(matrix[np.isfinite(matrix)])
    return max(0.01, float(np.quantile(finite, 0.95))) if finite.size else 0.01


def render_endpoints(summary: dict[str, Any], directory: Path) -> None:
    _style()
    # The long-state claim is carried by the six-hour horizon.  Plotting two
    # hours here hid the decisive E1077/E1125 cells even though the companion
    # horizon curves were correct.
    h1 = [row for row in _h1_rows(summary)
          if row["horizon_seconds"] == 21600
          and row["subject"] in summary["cohorts"]["long"]]
    h1_labels, h1_matrix, h1_gate = [], [], []
    for row in h1:
        h1_labels.append(f"{row['subject'].replace('epilepsiae_', 'E')} {row['family']}")
        endpoint_row = []
        gate_row = []
        for endpoint in ENDPOINTS:
            margins = [
                row["endpoint_evidence"][endpoint][name]["median"]
                for name in (
                    "gain_over_strong_baseline",
                    "dynamic_over_constant",
                    "correct_time_over_shifted",
                )
            ]
            endpoint_row.append(
                np.nan if any(value is None for value in margins)
                else min(float(value) for value in margins)
            )
            gate_row.append(bool(row["endpoint_evidence"][endpoint]["directional_dynamic_endpoint"]))
        h1_matrix.append(endpoint_row)
        h1_gate.append(gate_row)
    h1_matrix = np.asarray(h1_matrix, dtype=float) if h1_matrix else np.zeros((0, len(ENDPOINTS)))
    h1_gate = np.asarray(h1_gate, dtype=bool) if h1_gate else np.zeros((0, len(ENDPOINTS)), dtype=bool)
    h2_rows = [row for cohort in summary["h2a"].values() for family in cohort.values()
               for row in family["rows"]]
    h2_fields = (
        (("state_gain_over_B_mark_contact", "constant_unexplained_contact", "correct_time_paired_contact"), "contacts"),
        (("state_gain_over_B_mark_stop", "constant_unexplained_stop", "correct_time_paired_stop"), "STOP"),
        (("state_gain_over_B_mark_grammar", "constant_unexplained_grammar", "correct_time_paired_grammar"), "grammar"),
        (("same_prefix_gain_over_B_mark", "same_prefix_constant_unexplained_grammar", "correct_time_same_prefix_grammar"), "suffix"),
        (("conditional_rich_mark_gain_over_B_mark", "constant_unexplained_rich_mark", "correct_time_rich_mark"), "rich mark"),
    )
    h2_candidate_keys = (
        "contact_subset", "continue_stop", "grammar",
        "same_prefix_continuation", "conditional_rich_mark",
    )
    h2_labels, h2_matrix, h2_gate = [], [], []
    for row in h2_rows:
        h2_labels.append(f"{row['subject'].replace('epilepsiae_', 'E')} {row['family']}")
        endpoint_row = []
        for fields, _label in h2_fields:
            margins = [row["contrasts"].get(field, {}).get("median") for field in fields]
            endpoint_row.append(
                np.nan if any(value is None for value in margins)
                else min(float(value) for value in margins)
            )
        h2_matrix.append(endpoint_row)
        h2_gate.append([bool(row["endpoint_candidates"][key]) for key in h2_candidate_keys])
    h2_matrix = np.asarray(h2_matrix, dtype=float) if h2_matrix else np.zeros((0, len(h2_fields)))
    h2_gate = np.asarray(h2_gate, dtype=bool) if h2_gate else np.zeros((0, len(h2_fields)), dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 5.1), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax, matrix, gate, labels, columns, title, letter in (
        (axes[0], h1_matrix, h1_gate, h1_labels, ENDPOINT_LABEL,
         "H1: weakest of three dynamic-state margins at 6 h", "A"),
        (axes[1], h2_matrix, h2_gate, h2_labels, [label for _fields, label in h2_fields],
         "H2a: weakest registered frozen-decoder margin", "B"),
    ):
        limit = _symmetric_limit(matrix)
        cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#BFBFBF')
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        passed_i, passed_j = np.where(gate)
        ax.scatter(passed_j, passed_i, s=13, marker="o", facecolors="none",
                   edgecolors=BLACK, linewidths=0.7)
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="held-out gain")
        _panel(ax, title, letter)
    fig.tight_layout(w_pad=1.2)
    _save(fig, directory, "group_event_state_v038_multi_pathology_endpoints", {
        "question": "Does the same interictal state inform more than event rate?",
        "panel_A": "at six hours, minimum of strong-baseline, constant and time-shift margins for each H1 endpoint",
        "panel_B": "minimum registered B_mark, constant and wrong-time margin for each frozen-decoder endpoint",
        "gate_symbol": "a hollow circle marks an endpoint that passes every registered directional control",
        "missing": "gray values lack a complete estimable contrast; neutral near-white cells have measured margins near zero",
    })


def render_dual_credit(summary: dict[str, Any], directory: Path) -> None:
    _style()
    rows = summary.get('dual_checkpoint_bound_credit', {}).get('rows', [])
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.65))
    cmap = plt.get_cmap('YlGn').copy(); cmap.set_bad('#E5E5E5')
    for ax, horizon, branch, letter in zip(axes.flat, (6., 6., 8., 8.),
                                          ('event', 'background', 'event', 'background'), 'ABCD'):
        selected = [row for row in rows if row['horizon_hours'] == horizon and row['branch'] == branch]
        matrix = np.asarray([[row['endpoint_fraction_beyond_6h'][endpoint]['median']
                              for endpoint in ENDPOINTS] for row in selected], dtype=float)
        if matrix.size:
            shown = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap, vmin=0, vmax=1)
            fig.colorbar(shown, ax=ax, fraction=0.045, pad=0.02, label='gradient fraction >6 h')
            for i, row in enumerate(selected):
                for j, endpoint in enumerate(ENDPOINTS):
                    passing = sum(all(seed['checks'].values()) and
                                  seed['endpoint_fraction_beyond_6h'][endpoint] is not None and
                                  seed['endpoint_fraction_beyond_6h'][endpoint] >= .01
                                  for seed in row['joint_evidence']['per_seed'])
                    if passing >= 3:
                        ax.scatter(j, i, s=23, facecolors='none', edgecolors=BLACK, linewidths=.7)
        ax.set_xticks(range(len(ENDPOINTS)), ENDPOINT_LABEL, rotation=35, ha='right')
        ax.set_yticks(range(len(selected)), [row['subject'].replace('epilepsiae_', 'E') for row in selected])
        _panel(ax, f'Dual / {horizon:g} h head: {branch}', letter)
    fig.tight_layout(w_pad=1.3, h_pad=1.6)
    _save(fig, directory, 'group_event_state_v038_same_checkpoint_dual_credit', {
        'source': 'dual_checkpoint_bound_credit; exact original H1 checkpoint hashes and 6/8h heads',
        'quantity': 'fraction of total input-gradient mass assigned to actual observations older than6h',
        'circle': 'same >=3 seeds have changed selected observer maps, trained stage, replay parity, >=3 anchors and endpoint fraction>=1%',
        'grey': 'no estimable gradient fraction; not a biological negative',
        'boundary': 'overlapping diagnostic anchors; no causal physiological intervention or new time-constant discovery',
    })


def render_h2b(summary: dict[str, Any], directory: Path) -> None:
    _style()
    rows = summary["h2b"]["rows"]
    labels = [row["subject"].replace("epilepsiae_", "E") for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.75), gridspec_kw={"width_ratios": [0.9, 1.25, 1.15]})
    counts = np.asarray([[row.get('clinical_seizures_by_phase', row['seizures_by_phase']).get(split, 0) for split in ("FIT", "INNER", "SELECTION")]
                         for row in rows], dtype=float) if rows else np.zeros((0, 3))
    image = axes[0].imshow(counts, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(3), ["FIT", "INNER", "held-out"], rotation=30, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axes[0].text(j, i, str(int(counts[i, j])), ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.02, label="distinct seizures")
    _panel(axes[0], "Clinical seizure inventory", "A")
    risk_fields = (
        ("event_only_state_gain_over_mark_history", "event"),
        ("grid_state_gain_over_mark_history", "grid"),
        ("background_state_gain_over_current_background_censored_logscore", "background"),
        ("state_gain_over_background_censored_logscore", "dual"),
    )
    time_fields = (
        ("event_correct_time_gain_over_shift", "event time"),
        ("event_dynamic_gain_over_fit_period_mean", "event level"),
        ("dual_correct_time_gain_over_shift", "dual time"),
        ("dual_dynamic_gain_over_fit_period_mean", "dual level"),
        ("background_correct_time_gain_over_shift", "bg time"),
        ("background_dynamic_gain_over_fit_period_mean", "bg level"),
    )
    for ax, fields, title, letter in (
        (axes[1], risk_fields, "Frozen state: seizure-risk gain", "B"),
        (axes[2], time_fields, "Seizure relation: temporal controls", "C"),
    ):
        all_values = []
        for index, (field, label) in enumerate(fields):
            values = []
            for row in rows:
                value = row["risk_contrasts"].get(field, {}).get("median")
                if value is None:
                    continue
                values.append(float(value)); all_values.append(float(value))
                marker = "o" if row["estimability"] == "REPEATED_HELD_OUT_SEIZURES" else "^"
                color = GREEN if marker == "o" else AMBER
                ax.scatter(index, value, s=20, marker=marker, color=color, alpha=0.8)
                candidate_family = label if label in row.get("risk_candidates", {}) else label.split()[0]
                if row["subject"] == "epilepsiae_922" \
                        and row.get("risk_candidates", {}).get(candidate_family, False):
                    ax.annotate("E922", (index, value), xytext=(3, 3),
                                textcoords="offset points", fontsize=5.8, color=GREEN)
            if values:
                ax.scatter(index, np.median(values), marker="_", s=150, linewidths=2, color=BLACK)
        bound = max([abs(value) for value in all_values] + [0.01]) * 1.18
        ax.axhspan(0, bound, color=GREEN_LIGHT, alpha=0.45, zorder=-5)
        ax.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)))
        ax.set_ylim(-bound, bound)
        ax.set_xticks(range(len(fields)), [label for _field, label in fields], rotation=32, ha="right")
        ax.set_ylabel("censored log-score gain")
        _panel(ax, title, letter)
    fig.tight_layout(w_pad=1.2)
    _save(fig, directory, "group_event_state_v038_frozen_seizure_transfer", {
        "question": "Does a fully frozen interictal state transfer to seizure risk or early ictal expression?",
        "panel_A": "clinical onset inventory by phase, not independent samples; actual scored-onset counts and episode grouping are separate in summary/report",
        "panel_B": "risk gain over the predeclared baseline for each frozen observer family",
        "panel_C": "correct-time and dynamic-over-constant controls",
        "symbols": "circles have at least three held-out seizures; triangles are descriptive only",
    })


def render_branch_attribution(summary: dict[str, Any], directory: Path) -> None:
    row = next((row for row in _h1_rows(summary, 'long_0.5_2_6_8h')
                if row['subject'] == 'epilepsiae_1125' and row['family'] == 'dual'
                and row['horizon_seconds'] == 21600), None)
    if row is None or not row.get('branch_control_evidence'):
        return
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), constrained_layout=True)
    cells = list(row['branch_control_evidence'].values())
    for ax, field, title, letter in zip(axes,
            ('dynamic_over_branch_constant', 'correct_time_over_branch_shift'),
            ('E1125: branch dynamics vs FIT constant', 'E1125: correct time vs branch shift'), ('A', 'B')):
        for i, (branch, color) in enumerate((('event', BLUE), ('background', GREEN))):
            values = np.array([cell[branch][field] for cell in cells], dtype=float)
            ax.scatter(i + np.linspace(-.11, .11, len(values)), values, color=color, s=23, zorder=3)
            ax.plot([i-.18, i+.18], [np.median(values)]*2, color=BLACK, lw=1.4)
        ax.axhline(0, color=GREY, lw=.8, ls='--')
        ax.set_xticks([0, 1], ['event branch', 'background branch'])
        ax.set_xlim(-.5, 1.5); ax.set_ylabel('control loss − correct-state loss')
        _panel(ax, title, letter)
    _save(fig, directory, 'group_event_state_v038_dual_branch_attribution', {
        'subject': row['subject'], 'family': 'dual', 'future_horizon_seconds': 21600,
        'n_optimization_seeds': len(cells), 'branch_controls': row['branch_control_evidence'],
        'interpretation': 'frozen same-checkpoint readout; only one branch changed; negative constant margin rejects event-specific temporal gain; post-review diagnostic'})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument('--allow-repair-diagnostic', action='store_true')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") not in ('COMPLETE', 'REVIEW_REPAIR_COMPLETE') and not (
            args.allow_repair_diagnostic and summary.get('status') == 'REVIEW_REPAIR_PARTIAL'):
        raise RuntimeError("paper decision figures require the complete frozen summary")
    INPUT_METADATA.update(summary_path=str(args.summary.resolve()),
                          summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                          producer_path=str(Path(__file__).resolve()),
                          producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          evidence_status=summary['status'])
    render_h1(summary, args.out_dir)
    render_endpoints(summary, args.out_dir)
    render_h2b(summary, args.out_dir)
    render_dual_credit(summary, args.out_dir)
    render_branch_attribution(summary, args.out_dir)
    (args.out_dir / "README.md").write_text(
        """# v0.3.8 核心结果图说明

### group_event_state_v038_dynamic_state_and_credit.png

这张图把“比强历史基线好”“不是整段常数”“必须对准时刻”和“人体训练后梯度确实回到六小时前”分成四块展示。前三项只是动态预测的必要对照，还须同seed训练与来源检查；dual的整模型收益也不能直接归给事件分支。D 只表示 observer 使用旧信息，不表示 IED 造成生理改变。前三幅只含原长尺度六位患者，D 明确为旧 event-only 模型的 8 h head，不能与 dual 的 6 h 预测增益相连。

**关注点**：不要用单独一块阳性替代完整动态状态判据。

### group_event_state_v038_medium_dynamic_state.png

单独展示原中尺度五位患者在 0.5/2 h 的三个对照增益，与长尺度患者不合并计算均值或中位数。点与线为描述统计，完整资格由机器表的同 seed 对照交集决定。

**关注点**：不能把不同队列的连线差异解释为纯时间尺度效应。

### group_event_state_v038_multi_pathology_endpoints.png

A 展示长队列六小时 future block 的七类端点，颜色取强基线、常数和错时三种 margin 中最弱的一项；B 展示冻结 contact decoder 上的触点、STOP、grammar、同前缀分叉和 rich mark。它直接检查状态是否只编码 event rate，还是能同时预测多类病理表达。

空心圆表示至少三个相同 seed 各自通过该端点的完整对照；底色是各对照中位增益的最小值，本身不是交集检验。灰色表示缺少完整可估对照，接近白色才是测得的接近零增益。后缀列须有严格后缀卡和同子集常数对照，旧全事件卡不放行。

**关注点**：空心圆只表示方向性对照通过；多端点状态结论还必须检查同seed、承重训练和上游权重身份。

### group_event_state_v038_same_checkpoint_dual_credit.png

分别展示dual的6/8小时future head，经event/background两条路径追溯到六小时前真实观察的梯度比例。所有卡绑定同一H1 checkpoint；空心圆还要求至少三个相同seed有选中输入矩阵更新、训练预算合格、状态重放及足够审计anchor。灰色为不可估或无可定义比例，anchor可能重叠，不能当作独立病例。

**关注点**：它补上同一模型的长程梯度链；不证明新生理时间常数或事件因果反馈。

### group_event_state_v038_dual_branch_attribution.png

E1125六小时预测中，分别只把event/background分支替换为FIT常数或段内错时，其余输入及读出权重保持冻结。五个点是五次优化重复，黑线是中位数；正值表示正确动态状态的损失更低。事件分支五个seed均未胜过其自身FIT常数，尽管都胜过错时；因此整个dual模型的动态增益不能归为事件分支的动态价值。

**关注点**：事件矩阵更新、长历史梯度与有益的事件动态不是同一件事。

### group_event_state_v038_frozen_seizure_transfer.png

A 给临床清单中的不同 onset 数，实际进入评分的发作数和聚集敏感性另见报告；例如E958的FIT清单3次只有2次进入正预测行，不能因此放行拟合。B 展示冻结状态的风险增量，C 检查正确时刻与常数对照。圆点表示至少三次被评分的held-out onset，三角只作描述；完整资格还要求训练、容量/灵敏度及上游来源检查。E922三次onset在6小时聚集规则下为一个跨分区簇，旧完整候选结论已撤回。

**关注点**：不可估不是生物学阴性，五分钟锚点不能冒充独立发作。
""",
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
