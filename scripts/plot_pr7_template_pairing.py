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
# Step 6 main figure 2 — Per-Subject Null Comparison Grid
# ---------------------------------------------------------------------------
def plot_main_fig2_per_subject_null_grid() -> Optional[Path]:
    """6-panel grid (2x3) for H1 cohort: per-subject excess(Δt) under all 4 nulls."""
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    if not files:
        print("No per-subject JSONs; skipping fig2.")
        return None

    fig, axes = new_figure(nrows=2, ncols=3, figsize=(15.0, 8.5))
    axes_flat = axes.flatten()
    null_color_map = {
        "N0": COL_NEUTRAL,
        "N1": COL_YUQUAN,
        "N2": COL_SIG,
        "N3": COL_EPILEPSIAE,
    }

    for idx, f in enumerate(files[:6]):
        ax = axes_flat[idx]
        with f.open() as fh:
            rec = json.load(fh)
        dt_grid = tuple(rec["delta_t_grid"])
        lift = rec["pairing_with_nulls"]["lift"]
        for null_id in ["N0", "N1", "N2", "N3"]:
            if null_id not in lift:
                continue
            ex = [lift[null_id][f"{dt}"]["excess"] for dt in dt_grid]
            ax.plot(
                dt_grid, ex, "-o",
                color=null_color_map[null_id],
                linewidth=2.0 if null_id == "N2" else 1.0,
                markersize=5 if null_id == "N2" else 3,
                alpha=1.0 if null_id == "N2" else 0.6,
                label=null_id + (" (main)" if null_id == "N2" else ""),
            )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axvline(10.0, color=COL_SIG, linestyle="--", linewidth=0.8, alpha=0.4)
        ax.axvline(30.0, color=COL_SIG, linestyle="--", linewidth=0.8, alpha=0.25)
        _format_xticks(ax, dt_grid)
        ax.set_xlabel("Δt", fontsize=FS_LABEL - 2)
        ax.set_ylabel("excess", fontsize=FS_LABEL - 2)
        sid = f"{rec['dataset']}/{rec['subject_id']}"
        ax.set_title(
            f"{sid}  (N={rec['n_events_used']})", fontsize=FS_TITLE - 2,
        )
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "PR-7 fig2 — per-subject excess(Δt) under N0/N1/N2/N3 nulls (H1 cohort)",
        fontsize=FS_TITLE,
    )
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig2_per_subject_null_grid.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Step 6 main figure 3 — Direction Asymmetry & Transition Odds
# ---------------------------------------------------------------------------
def plot_main_fig3_direction_and_transition() -> Optional[Path]:
    """Two-panel scatter: (a) direction symmetry (a→b lift vs b→a lift at Δt=10s),
    (b) transition_odds vs baseline_odds. Each point = one H1 subject.
    """
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    if not files:
        return None

    a_to_b: Dict[str, float] = {}
    b_to_a: Dict[str, float] = {}
    trans_odds: Dict[str, float] = {}
    base_odds: Dict[str, float] = {}
    for f in files:
        with f.open() as fh:
            rec = json.load(fh)
        sid = f"{rec['dataset']}/{rec['subject_id']}"
        n2_10 = rec["pairing_with_nulls"]["lift"].get("N2", {}).get("10.0", {})
        a_to_b[sid] = n2_10.get("a_to_b_lift", float("nan"))
        b_to_a[sid] = n2_10.get("b_to_a_lift", float("nan"))
        trans_odds[sid] = rec["transition_odds"].get("transition_odds", float("nan"))
        base_odds[sid] = rec["transition_odds"].get("baseline_odds", float("nan"))

    fig, axes = new_figure(nrows=1, ncols=2, figsize=(12.0, 5.5))

    # (a) Direction asymmetry — diagonal-symmetric scatter at Δt=10s under N2
    ax = axes[0]
    sids = sorted(a_to_b.keys())
    short = [s.replace("epilepsiae/", "E:").replace("yuquan/", "Y:") for s in sids]
    xs = [a_to_b[s] for s in sids]
    ys = [b_to_a[s] for s in sids]
    ax.scatter(
        xs, ys, s=130, marker="o",
        facecolor=COL_SIG, edgecolor="black", zorder=3,
    )
    for x, y, lab in zip(xs, ys, short):
        ax.annotate(lab, (x, y), fontsize=9,
                    textcoords="offset points", xytext=(6, 6))
    if xs and ys:
        lim_lo = min(min(xs), min(ys), 0.5)
        lim_hi = max(max(xs), max(ys), 1.5)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="black", linestyle=":", linewidth=1.0, alpha=0.5,
                label="symmetric (y=x)")
        ax.set_xlim(lim_lo - 0.05, lim_hi + 0.05)
        ax.set_ylim(lim_lo - 0.05, lim_hi + 0.05)
    ax.axhline(1.0, color=COL_NONSIG, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(1.0, color=COL_NONSIG, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("opposite_lift (a→b) at Δt=10s, N2 null", fontsize=FS_LABEL)
    ax.set_ylabel("opposite_lift (b→a) at Δt=10s, N2 null", fontsize=FS_LABEL)
    ax.set_title("(a) H1b direction symmetry", fontsize=FS_TITLE)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=10)

    # (b) Transition odds vs baseline odds
    ax = axes[1]
    xs = [base_odds[s] for s in sids]
    ys = [trans_odds[s] for s in sids]
    ax.scatter(
        xs, ys, s=130, marker="o",
        facecolor=COL_SIG, edgecolor="black", zorder=3,
    )
    for x, y, lab in zip(xs, ys, short):
        ax.annotate(lab, (x, y), fontsize=9,
                    textcoords="offset points", xytext=(6, 6))
    if xs and ys:
        lim_lo = min(min(xs), min(ys), 0.0)
        lim_hi = max(max(xs), max(ys), 1.5)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="black", linestyle=":", linewidth=1.0, alpha=0.5,
                label="independent (y=x)")
        ax.set_xlim(lim_lo - 0.05, lim_hi + 0.05)
        ax.set_ylim(lim_lo - 0.05, lim_hi + 0.05)
    ax.set_xlabel("i.i.d. baseline_odds (next-event)", fontsize=FS_LABEL)
    ax.set_ylabel("empirical transition_odds (next-event)", fontsize=FS_LABEL)
    ax.set_title("(b) next-event transition vs baseline", fontsize=FS_TITLE)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=10)

    fig.suptitle(
        "PR-7 fig3 — direction symmetry & next-event transition (H1 cohort)",
        fontsize=FS_TITLE,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig3_direction_and_transition.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Step 6 main figure 4 — Two Exemplar Subjects with N2 null distribution band
# ---------------------------------------------------------------------------
def plot_main_fig4_exemplars() -> Optional[Path]:
    """Two exemplar subjects: strongest negative + strongest positive at Δt=10s.
    Each panel: empirical excess(Δt) curve overlaid with N2 null distribution
    [25, 75] envelope + median.
    """
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    if not files:
        return None

    # Find strongest negative and strongest positive at 10s
    ex10: Dict[str, Tuple[float, str]] = {}
    for f in files:
        with f.open() as fh:
            rec = json.load(fh)
        sid = f"{rec['dataset']}/{rec['subject_id']}"
        ex10[sid] = (
            rec["pairing_with_nulls"]["lift"]["N2"]["10.0"]["excess"],
            str(f),
        )
    strongest_neg = min(ex10.items(), key=lambda kv: kv[1][0])
    strongest_pos = max(ex10.items(), key=lambda kv: kv[1][0])

    chosen = [strongest_neg, strongest_pos]
    titles = [
        f"(a) strongest negative: {strongest_neg[0]} (excess(10s)={strongest_neg[1][0]:+.3f})",
        f"(b) strongest positive: {strongest_pos[0]} (excess(10s)={strongest_pos[1][0]:+.3f})",
    ]

    fig, axes = new_figure(nrows=1, ncols=2, figsize=(13.0, 5.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ((sid, (val, fpath)), title) in enumerate(zip(chosen, titles)):
        ax = axes[idx]
        with open(fpath) as fh:
            rec = json.load(fh)
        dt_grid = tuple(rec["delta_t_grid"])
        # Build empirical excess and N2 null distribution per Δt
        emp_ex = []
        null_lo = []
        null_hi = []
        null_med = []
        n2_lift = rec["pairing_with_nulls"]["lift"]["N2"]
        n2_null = rec["pairing_with_nulls"]["null"]["N2"]
        for dt in dt_grid:
            emp_ex.append(n2_lift[f"{dt}"]["excess"])
            opp_dist = np.asarray(n2_null[f"{dt}"]["p_opposite_dist"], dtype=float)
            same_dist = np.asarray(n2_null[f"{dt}"]["p_same_dist"], dtype=float)
            null_p_opp = float(np.mean(opp_dist))
            null_p_same = float(np.mean(same_dist))
            opp_lift_dist = (
                opp_dist / max(null_p_opp, 1e-12)
                if null_p_opp > 0 else opp_dist
            )
            same_lift_dist = (
                same_dist / max(null_p_same, 1e-12)
                if null_p_same > 0 else same_dist
            )
            ex_dist = opp_lift_dist - same_lift_dist
            null_lo.append(np.percentile(ex_dist, 25))
            null_hi.append(np.percentile(ex_dist, 75))
            null_med.append(np.median(ex_dist))

        ax.fill_between(
            dt_grid, null_lo, null_hi,
            color=COL_NEUTRAL, alpha=0.3, label="N2 null IQR",
        )
        ax.plot(
            dt_grid, null_med, ":",
            color=COL_NEUTRAL, linewidth=1.4, alpha=0.8,
            label="N2 null median",
        )
        ax.plot(
            dt_grid, emp_ex, "-o",
            color=COL_SIG, linewidth=2.2, markersize=6, zorder=4,
            label="empirical",
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(10.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axvline(30.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.3)
        _format_xticks(ax, dt_grid)
        ax.set_xlabel("Δt", fontsize=FS_LABEL)
        ax.set_ylabel("excess", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE - 2)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":")

    fig.suptitle(
        "PR-7 fig4 — exemplars (strongest neg + strongest pos) with N2 null IQR",
        fontsize=FS_TITLE,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig4_exemplars.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


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
# Step 6 appendix 1 — N2 window sweep (10/30/60 min) cohort excess curves
# ---------------------------------------------------------------------------
SWEEP_DIR = (
    ROOT
    / "results"
    / "interictal_propagation"
    / "template_pairing"
    / "per_subject_n2_sweep"
)


def plot_appendix1_window_sweep() -> Optional[Path]:
    """Three cohort excess(Δt) curves overlaid for N2 window ∈ {10, 30, 60} min."""
    if not SWEEP_DIR.exists():
        print("No sweep dir; skipping appendix 1.")
        return None
    sweep_files = sorted(SWEEP_DIR.glob("*.json"))
    if not sweep_files:
        print("No sweep JSONs; skipping appendix 1.")
        return None

    by_window: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for f in sweep_files:
        with f.open() as fh:
            rec = json.load(fh)
        w = int(round(rec["n2_window_minutes"]))
        sid = f"{rec['dataset']}/{rec['subject_id']}"
        by_window.setdefault(w, {})[sid] = rec

    if not by_window:
        return None

    sorted_windows = sorted(by_window.keys())
    fig, ax = new_figure(nrows=1, ncols=1, figsize=(8.0, 5.5))
    color_for_window = {10: "#A35E48", 30: "#6F8FA8", 60: "#9DAA90"}
    for w in sorted_windows:
        # cohort median excess(Δt)
        records = list(by_window[w].values())
        if not records:
            continue
        dt_grid = records[0]["delta_t_grid"]
        cohort_excess: List[List[float]] = []
        for rec in records:
            n2 = rec["pairing_with_nulls"]["lift"]["N2"]
            cohort_excess.append([n2[f"{dt}"]["excess"] for dt in dt_grid])
        arr = np.asarray(cohort_excess)
        med = np.median(arr, axis=0)
        ax.plot(
            dt_grid, med, "-o",
            color=color_for_window.get(w, "gray"),
            linewidth=2.0, markersize=6,
            label=f"window = {w} min",
        )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(10.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(30.0, color=COL_SIG, linestyle="--", linewidth=1.0, alpha=0.3)
    _format_xticks(ax, dt_grid)
    ax.set_xlabel("Δt", fontsize=FS_LABEL)
    ax.set_ylabel("cohort median excess", fontsize=FS_LABEL)
    ax.set_title(
        "Appendix 1 — N2 window sweep robustness (H1 cohort, n=6)",
        fontsize=FS_TITLE,
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle=":")

    out = FIG_DIR / "appendix1_window_sweep.png"
    savefig_pub(fig, out, dpi=DPI_PUB)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Step 6 appendix 3 — Cohort audit transparency table
# ---------------------------------------------------------------------------
AUDIT_CSV_PATH = (
    ROOT
    / "results"
    / "interictal_propagation"
    / "template_pairing"
    / "pr7_cohort_audit.csv"
)


def plot_appendix3_audit_table() -> Optional[Path]:
    """Render the audit CSV as a colored table figure for transparency."""
    import csv as _csv

    if not AUDIT_CSV_PATH.exists():
        print("No audit CSV; skipping appendix 3.")
        return None
    with AUDIT_CSV_PATH.open() as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return None

    cols = [
        "subject_id",
        "dataset",
        "n_events_total",
        "min_cluster_n",
        "n_blocks",
        "total_coverage_hours",
        "forward_reverse_reproduced",
        "h1_primary_pass",
        "h2_negative_pass",
        "exit_reason",
    ]

    cell_text = []
    cell_colors = []
    for r in rows:
        cell = [r.get(c, "") for c in cols]
        cov = r.get("total_coverage_hours", "")
        if cov:
            try:
                cell[5] = f"{float(cov):.1f}"
            except ValueError:
                pass
        cell_text.append(cell)
        if r.get("h1_primary_pass") == "True":
            row_color = "#D4E5DC"
        elif r.get("h2_negative_pass") == "True":
            row_color = "#E5EDF1"
        else:
            row_color = "#F0F0F0"
        cell_colors.append([row_color] * len(cols))

    fig_height = max(6.0, 0.27 * (len(rows) + 2))
    fig, ax = new_figure(nrows=1, ncols=1, figsize=(15.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for j, _ in enumerate(cols):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#5A6A78")

    fig.suptitle(
        "Appendix 3 — PR-7 cohort audit (green=H1 primary, blue=H2 negative, gray=excluded)",
        fontsize=FS_TITLE,
    )
    out = FIG_DIR / "appendix3_cohort_audit.png"
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

    burst_block = summary.get("burst_diagnostic_per_cohort", {}).get("h1_primary", {})
    burst_n2 = burst_block.get("by_null", {}).get("N2", {}) if burst_block else {}
    rll_med = burst_n2.get("median_run_length_lift", float("nan")) if burst_n2 else float("nan")
    rll_pos = burst_n2.get("n_subjects_run_length_lift_gt_1") if burst_n2 else None
    git_med = burst_n2.get("median_gap_to_iei_lift", float("nan")) if burst_n2 else float("nan")
    git_pos = burst_n2.get("n_subjects_gap_to_iei_lift_gt_1") if burst_n2 else None
    lag1_med = burst_n2.get("median_lag1_same_excess", float("nan")) if burst_n2 else float("nan")
    lag1_pos = burst_n2.get("n_subjects_lag1_excess_positive") if burst_n2 else None

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

### fig2_per_subject_null_grid.png

6-panel 网格：每 subject 一个子图，叠加 N0/N1/N2/N3 四个 null 下的
excess(Δt) 曲线，N2 主 null 加粗。10s/30s 双门用红色虚直线标注。

**关注点**：N2 与 N1 应方向一致（robustness 守住）；当前 cohort 6/6
都成立。N3 在多数 subject 上接近 N2，仅个别 subject（1073）N3 比 N2
更负——但量级仍小，不影响 H1 NULL 判读。

### fig3_direction_and_transition.png

(a) H1b direction symmetry：散点 (a→b lift, b→a lift) at Δt=10s, N2 null。
对角 y=x 表示对称。
(b) next-event transition odds vs i.i.d. baseline_odds：散点。对角 y=x
表示独立 mark draw。

**关注点**：(a) 多数 subject 接近对角，没有方向偏好；548 显著偏离
y=x（a→b 比 b→a 更稀缺）。(b) 5 个 subject 接近 y=x（独立抽样），
548 是唯一显著偏离的——transition_odds 远低于 baseline，说明 next-event
更倾向同 cluster（burst-style 但量级有限）。

### fig4_exemplars.png

cohort 中两个极端 subject：(a) 最强负向 epilepsiae/548（excess(10s)=−0.20）；
(b) 最强正向 epilepsiae/635（excess(10s)=+0.03）。每子图叠加 N2 null
分布的 [25, 75] envelope（灰色阴影）+ null median（点线）+ empirical
曲线（红色实心）。

**关注点**：548 在短窗位于 N2 null IQR 之外（**single-subject outlier /
exploratory signal**——IQR 不是显著性边界，仅作可视化分布参考）；635
短窗轻微高于 null IQR，但量级远小于 548 的负向。两者共同 cohort 中位仍
接近 0。报告为 case-series exemplars，**不**升级为 cohort claim。

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

## Appendix

### appendix1_window_sweep.png

PR-7 Step 5 robustness：N2 主 null 在窗口 {{10, 30, 60}} min 三个尺度上重跑，
画 H1 cohort excess(Δt) 中位数。

**关注点**：三个 window 上 cohort triple-gate verdict 一致 NULL（Wilcoxon
p ∈ [0.78, 0.89]，所有 median(30s) < 0），方向稳健；**但**绝对量级随
window 单调放大（cohort median(30s) 在 w=10/30/60 min 分别 = −0.002 /
−0.015 / −0.029），主要由单一 subject 548 驱动（−0.10 / −0.20 / −0.30）。
**不应**说"三条曲线高度重合"——cohort 在 10s/30s 上 spread ~0.025；要写
的口径是"cohort verdict robust，但 magnitude window-sensitive，548 outlier
对 window 选择高敏感（与该 subject 同模板 burst 在 10–60 min 时间尺度上
的结构一致）"。

### appendix3_cohort_audit.png

完整 cohort audit 表格：30 个候选 subject，列出 n_events / min_cluster_n /
n_blocks / coverage / forward_reverse_reproduced / 5 条入选条件 / 退出原因。

**关注点**：绿色行 = H1 primary cohort（6 subject），蓝色行 = H2 negative
cohort（17 subject），灰色 = excluded（7 subject）。可独立验证 inclusion
逻辑。

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
    plot_main_fig2_per_subject_null_grid()
    plot_main_fig3_direction_and_transition()
    plot_main_fig4_exemplars()
    plot_main_fig5_burst_diagnostic(summary)
    plot_per_subject_curves()
    plot_appendix1_window_sweep()
    plot_appendix3_audit_table()
    write_figures_readme(summary)


if __name__ == "__main__":
    main()
