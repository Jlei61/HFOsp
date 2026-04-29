#!/usr/bin/env python3
"""PR-7 Template Antagonistic Temporal Pairing plotting.

Step 6 deliverable per archive §6.5; main figure 1 used by Step 3 to
display the H1 PASS/NULL readout in the cohort excess(Δt) curve.

Currently produces:
  - fig1_cohort_excess_curve.png — main figure 1 (H1 verdict)
  - per_subject/*.png — one excess(Δt) curve per H1 subject

Other figures (per-subject 4-null grid / direction asymmetry / exemplar)
are deferred to Step 6 once H2 cohort runs are also available for the
negative-control overlay.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import (  # noqa: E402
    COL_EPILEPSIAE,
    COL_NEUTRAL,
    COL_NONSIG,
    COL_SIG,
    COL_YUQUAN,
    DPI_PUB,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    new_figure,
    savefig_pub,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COHORT_SUMMARY = (
    ROOT
    / "results"
    / "interictal_propagation"
    / "template_pairing"
    / "cohort_summary.json"
)
PER_SUBJECT_DIR = (
    ROOT / "results" / "interictal_propagation" / "template_pairing" / "per_subject"
)
FIG_DIR = (
    ROOT
    / "results"
    / "interictal_propagation"
    / "template_pairing"
    / "figures"
)


# ---------------------------------------------------------------------------
# Tick formatting
# ---------------------------------------------------------------------------
DELTA_T_LABELS = [
    (1.0, "1s"),
    (5.0, "5s"),
    (10.0, "10s"),
    (30.0, "30s"),
    (60.0, "1min"),
    (300.0, "5min"),
    (1800.0, "30min"),
    (3600.0, "1h"),
]


def _format_xticks(ax: plt.Axes, dt_grid: Sequence[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks([dt for dt, _ in DELTA_T_LABELS if dt in set(dt_grid)])
    ax.set_xticklabels(
        [lab for dt, lab in DELTA_T_LABELS if dt in set(dt_grid)],
        fontsize=FS_TICK,
    )


# ---------------------------------------------------------------------------
# Cohort excess(Δt) curve helper
# ---------------------------------------------------------------------------
def _cohort_envelope(
    cohort_block: Dict[str, Any],
    null_id: str,
    dt_grid: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, q25, q75, n) arrays over Δt for one cohort × one null."""
    by_null = cohort_block.get("by_null", {}).get(null_id, {})
    medians, q25, q75, ns = [], [], [], []
    for dt in dt_grid:
        bucket = by_null.get(f"{dt}", {})
        ex = list(bucket.get("excess_per_subject", {}).values())
        if not ex:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
            ns.append(0)
            continue
        ex_arr = np.asarray(ex, dtype=float)
        medians.append(np.median(ex_arr))
        q25.append(np.percentile(ex_arr, 25))
        q75.append(np.percentile(ex_arr, 75))
        ns.append(ex_arr.size)
    return (
        np.asarray(medians, dtype=float),
        np.asarray(q25, dtype=float),
        np.asarray(q75, dtype=float),
        np.asarray(ns, dtype=int),
    )


def _verdict_text(triple_gate: Optional[Dict[str, Any]]) -> str:
    if not triple_gate:
        return "verdict: insufficient data"
    n2 = triple_gate.get("N2", {})
    n3 = triple_gate.get("N3", {})
    if not n2 or n2.get("skipped"):
        return "verdict: missing N2"
    lines = [
        "H1 verdict (n=" + str(n2.get("n_subjects", "?")) + ")",
        f"N2 main null:  PASS={n2.get('pass')}",
        f"  wilcoxon(10s) p = {n2.get('wilcoxon_10s', float('nan')):.4f}",
        f"  sign(10s) p     = {n2.get('sign_10s', float('nan')):.4f}",
        f"  median(10s)     = {n2.get('median_10s', float('nan')):+.4f}",
        f"  median(30s)     = {n2.get('median_30s', float('nan')):+.4f}",
    ]
    if n3 and not n3.get("skipped"):
        lines.append(
            f"N3 robustness: PASS={n3.get('pass')}, "
            f"wilc(10s) p={n3.get('wilcoxon_10s', float('nan')):.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main figure 1: cohort excess(Δt) curve
# ---------------------------------------------------------------------------
def plot_main_fig1(summary: Dict[str, Any]) -> Path:
    """Cohort excess(Δt) curve — H1 verdict readout.

    Red solid + envelope = H1 primary cohort N2 main null.
    Red dashed = H1 primary cohort N3 robustness null.
    Light gray dashed = H2 negative cohort N2 (drawn only if available).
    Yellow shading = packing-proximity diagnostic (1s/5s).
    Blue shading = expected ≈ 0 long-range (≥ 30min).
    Vertical lines at 10s (primary) and 30s (sensitivity).
    """
    dt_grid = tuple(summary["delta_t_grid_seconds"])
    fig, ax = new_figure(nrows=1, ncols=1, figsize=(8.0, 5.5))

    # Background shading: packing-proximity (1s/5s) and long-range (≥30min)
    ax.axvspan(0.5, 7.5, color="#FFF3D6", alpha=0.6, zorder=0)
    ax.axvspan(1500.0, 4500.0, color="#E0EAF2", alpha=0.6, zorder=0)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(10.0, color=COL_SIG, linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax.axvline(30.0, color=COL_SIG, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)

    h1_block = summary["cohorts"].get("h1_primary", {})
    h2_block = summary["cohorts"].get("h2_negative", {})

    # H1 N2 main null
    med_n2, q25_n2, q75_n2, n_n2 = _cohort_envelope(h1_block, "N2", dt_grid)
    ax.fill_between(
        dt_grid, q25_n2, q75_n2,
        color=COL_SIG, alpha=0.18, zorder=2,
        label=f"H1 fwd/rev N2 IQR (n={int(np.nanmax(n_n2))})",
    )
    ax.plot(
        dt_grid, med_n2, "-",
        color=COL_SIG, linewidth=2.2, marker="o", markersize=6, zorder=4,
        label="H1 fwd/rev (N2 main null) median",
    )

    # H1 N3 robustness
    med_n3, _, _, n_n3 = _cohort_envelope(h1_block, "N3", dt_grid)
    if np.any(np.isfinite(med_n3)):
        ax.plot(
            dt_grid, med_n3, "--",
            color=COL_SIG, linewidth=1.6, marker="s", markersize=5, alpha=0.7, zorder=3,
            label=f"H1 fwd/rev (N3 robustness) median",
        )

    # H2 negative cohort if available
    med_h2, _, _, n_h2 = _cohort_envelope(h2_block, "N2", dt_grid)
    if np.any(np.isfinite(med_h2)) and int(np.nanmax(n_h2)) > 0:
        ax.plot(
            dt_grid, med_h2, ":",
            color=COL_NONSIG, linewidth=1.8, marker="^", markersize=5, zorder=2,
            label=f"H2 non-fwdrev (N2) median (n={int(np.nanmax(n_h2))})",
        )

    # Annotations
    _format_xticks(ax, dt_grid)
    ax.set_xlabel("Δt (window after anchor event)", fontsize=FS_LABEL)
    ax.set_ylabel("excess = opposite_lift − same_lift", fontsize=FS_LABEL)
    ax.set_title(
        "PR-7 cohort excess(Δt) — short-window opposite-template signal",
        fontsize=FS_TITLE,
    )
    ax.text(
        2.0, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.02,
        "packing-proximity\ndiagnostic (1s/5s)",
        fontsize=10, color="#9E7A2D", ha="center", va="top",
    )
    ax.text(
        2600.0, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.02,
        "expected ≈ 0\n(slow-drive control)",
        fontsize=10, color="#476784", ha="center", va="top",
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)

    # Verdict text box
    verdict = _verdict_text(h1_block.get("triple_gate_pass"))
    ax.text(
        0.985, 0.98, verdict,
        transform=ax.transAxes, fontsize=10,
        ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAF8F4", edgecolor="#888", alpha=0.92),
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig1_cohort_excess_curve.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Per-subject excess(Δt) curve (cleaner alternative to the 4-null grid for now)
# ---------------------------------------------------------------------------
def plot_per_subject_curves() -> List[Path]:
    """One small figure per H1 subject: empirical excess across Δt under all
    4 nulls. Useful for diagnosing where the null comes from per subject."""
    out_paths: List[Path] = []
    per_dir = FIG_DIR / "per_subject"
    per_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    for f in files:
        with f.open() as fh:
            rec = json.load(fh)
        dt_grid = tuple(rec["delta_t_grid"])
        lift = rec["pairing_with_nulls"]["lift"]
        nulls_present = sorted(lift.keys())
        fig, ax = new_figure(nrows=1, ncols=1, figsize=(6.5, 4.0))
        for idx, null_id in enumerate(nulls_present):
            excess_dt = [
                lift[null_id][f"{dt}"]["excess"] for dt in dt_grid
            ]
            color = {"N0": COL_NEUTRAL, "N1": COL_YUQUAN, "N2": COL_SIG, "N3": COL_EPILEPSIAE}.get(
                null_id, COL_NONSIG
            )
            linewidth = 2.0 if null_id == "N2" else 1.2
            alpha = 1.0 if null_id == "N2" else 0.7
            ax.plot(
                dt_grid, excess_dt, "-o",
                color=color, linewidth=linewidth, markersize=5, alpha=alpha,
                label=null_id,
            )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(10.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axvline(30.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.3)
        _format_xticks(ax, dt_grid)
        ax.set_xlabel("Δt", fontsize=FS_LABEL)
        ax.set_ylabel("excess", fontsize=FS_LABEL)
        sid = f"{rec['dataset']}/{rec['subject_id']}"
        ax.set_title(
            f"{sid}  (N={rec['n_events_used']}, T_a={rec['n_T_a']}, T_b={rec['n_T_b']})",
            fontsize=FS_TITLE,
        )
        ax.legend(loc="best", fontsize=9)
        out = per_dir / f"{rec['dataset']}_{rec['subject_id']}_excess.png"
        savefig_pub(fig, out, dpi=DPI_PUB)
        plt.close(fig)
        out_paths.append(out)
    print(f"Wrote {len(out_paths)} per-subject figures to {per_dir}")
    return out_paths


# ---------------------------------------------------------------------------
# Step 3.5 burst-level diagnostic main figure (fig5)
# ---------------------------------------------------------------------------
def plot_main_fig5_burst_diagnostic(summary: Dict[str, Any]) -> Optional[Path]:
    """fig5: per-subject N2 (filled) + N1 (open) markers for run_length_lift,
    gap_to_iei_lift, lag1_same_excess. Cohort median bar + 1.0 baseline line.
    """
    burst_block = summary.get("burst_diagnostic_per_cohort", {}).get("h1_primary")
    if not burst_block or "by_null" not in burst_block:
        print("No burst diagnostic data in cohort_summary.json; skipping fig5.")
        return None

    n2 = burst_block["by_null"].get("N2", {})
    n1 = burst_block["by_null"].get("N1", {})
    if not n2:
        print("No N2 burst data; skipping fig5.")
        return None

    rll_n2 = n2.get("run_length_lift", {})
    rll_n1 = n1.get("run_length_lift", {})
    git_n2 = n2.get("gap_to_iei_lift", {})
    git_n1 = n1.get("gap_to_iei_lift", {})
    lag_n2 = n2.get("lag1_same_excess", {})
    lag_n1 = n1.get("lag1_same_excess", {})

    subjects = sorted(rll_n2.keys())
    if not subjects:
        return None
    short_labels = [s.replace("epilepsiae_", "E:").replace("yuquan_", "Y:") for s in subjects]
    x = np.arange(len(subjects))

    fig, axes = new_figure(nrows=1, ncols=3, figsize=(15.0, 5.0))
    panels = [
        (axes[0], "run_length_lift", rll_n2, rll_n1, 1.0, "lift", "(a) mean run length / null"),
        (axes[1], "gap_to_iei_lift", git_n2, git_n1, 1.0, "lift", "(b) gap-to-IEI ratio / null"),
        (axes[2], "lag1_same_excess", lag_n2, lag_n1, 0.0, "excess", "(c) lag-1 same-label excess"),
    ]
    for ax, key, n2_dict, n1_dict, baseline, ylabel, title in panels:
        n2_vals = [n2_dict.get(s, np.nan) for s in subjects]
        n1_vals = [n1_dict.get(s, np.nan) for s in subjects]
        ax.scatter(
            x, n2_vals, marker="o", s=110,
            facecolor=COL_SIG, edgecolor="black", zorder=4,
            label="N2 (main null)",
        )
        ax.scatter(
            x, n1_vals, marker="o", s=70,
            facecolor="white", edgecolor=COL_SIG, linewidths=1.5, zorder=3,
            label="N1 (sanity)",
        )
        # cohort median (N2)
        n2_arr = np.asarray([v for v in n2_vals if np.isfinite(v)])
        if n2_arr.size:
            ax.axhline(
                float(np.median(n2_arr)), color=COL_SIG, linestyle="--",
                linewidth=1.0, alpha=0.6,
            )
        ax.axhline(baseline, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=10)

    fig.suptitle(
        "PR-7 Step 3.5 — burst-level diagnostic (post-hoc exploratory)",
        fontsize=FS_TITLE,
    )
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig5_burst_diagnostic.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# README.md for figures dir (AGENTS.md spec)
# ---------------------------------------------------------------------------
def write_figures_readme(summary: Dict[str, Any]) -> None:
    h1 = summary["cohorts"].get("h1_primary", {})
    triple = h1.get("triple_gate_pass", {}).get("N2", {}) or {}
    n_h1 = h1.get("n_members", 0)
    pass_str = "PASS" if triple.get("pass") else "NULL"
    wilc = triple.get("wilcoxon_10s", float("nan"))
    sign = triple.get("sign_10s", float("nan"))
    med30 = triple.get("median_30s", float("nan"))
    burst_block = summary.get("burst_diagnostic_per_cohort", {}).get("h1_primary", {})
    burst_n2 = burst_block.get("by_null", {}).get("N2", {})
    rll_med = burst_n2.get("median_run_length_lift", float("nan"))
    rll_pos = burst_n2.get("n_subjects_run_length_lift_gt_1")
    git_med = burst_n2.get("median_gap_to_iei_lift", float("nan"))
    git_pos = burst_n2.get("n_subjects_gap_to_iei_lift_gt_1")
    lag1_med = burst_n2.get("median_lag1_same_excess", float("nan"))
    lag1_pos = burst_n2.get("n_subjects_lag1_excess_positive")

    md = f"""# PR-7 Template Antagonistic Temporal Pairing 图集

## 主图

### fig1_cohort_excess_curve.png

cohort 级 `excess(Δt) = opposite_lift − same_lift` 曲线。红实线为 H1
forward/reverse subset (n={n_h1}) 在 N2 主 null 下的中位数；红虚线为同
cohort 在 N3 robustness null 下的中位数；灰点线为 H2 non-fwdrev cohort
负对照（如已跑则显示）。1s/5s 区间为 packing-proximity 诊断，不进 PASS
判据；30min/1h 长尺度区间预期 ≈ 0。10s/30s 双门用红色虚直线标注。

**关注点**：H1 triple gate 当前判读 **{pass_str}**（Wilcoxon p={wilc:.3f}、
sign p={sign:.3f}、median(30s)={med30:+.3f}）。短窗 10s/30s 上没有
检测到 fwd/rev cohort 的 opposite-template excess；长窗几乎归零，慢漂移
共驱不是 confound。仅否定 short-window reciprocal coupling，**不**否定
PR-6 已建立的 fwd/rev 几何相关性，**不**否定其它形式的因果耦合。

### fig5_burst_diagnostic.png

PR-7 Step 3.5 post-hoc exploratory diagnostic。三个 panel 分别画
H1 cohort (n={n_h1}) 上每 subject 的 (a) `run_length_lift`、(b)
`gap_to_iei_lift`、(c) `lag1_same_excess`，每个数字在 N2 主 null（实心
红圆）和 N1 sanity（空心红圆）下分别给出。点线 baseline（lift=1 或
excess=0），虚线为 cohort 中位数（N2）。

**关注点**：观察是否存在 same-template persistence（form 2 vs form 5）：
- 当前 cohort 中位 `run_length_lift` (N2) = {rll_med:.3f}，
  {rll_pos}/{n_h1} subject > 1.0
- 中位 `gap_to_iei_lift` (N2) = {git_med:.3f}，
  {git_pos}/{n_h1} > 1.0
- 中位 `lag1_same_excess` (N2) = {lag1_med:+.4f}，
  {lag1_pos}/{n_h1} > 0
本 figure **不**进 H1 PASS 判据；仅作 H1 NULL 的机制解释。

## per_subject/

每个 H1 subject 一张 excess(Δt) 单图，叠加 N0/N1/N2/N3 四个 null 下
的曲线（N2 主 null 加粗）。

**关注点**：每个 subject 的 N2 与 N3 应方向一致；如某 subject 在 N2
显著正而 N3 反向，说明 burst 时间结构主导信号，需 follow-up 检查。
当前 6 个 H1 subject 中 5 个 N2/N3 同号；`epilepsiae/548` 全段强烈反向
（同模板 burst-clustered，非 packing artifact，5min 仍 ≈ −0.09）。
"""
    out = FIG_DIR / "README.md"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not COHORT_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing {COHORT_SUMMARY}; run run_pr7_template_pairing.py --cohort-stats first."
        )
    with COHORT_SUMMARY.open() as fh:
        summary = json.load(fh)

    plot_main_fig1(summary)
    plot_per_subject_curves()
    plot_main_fig5_burst_diagnostic(summary)
    write_figures_readme(summary)


if __name__ == "__main__":
    main()
