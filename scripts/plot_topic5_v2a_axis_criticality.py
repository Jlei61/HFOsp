#!/usr/bin/env python
"""Topic 5 V2a axial criticality null-gate figure.

This plot focuses on the only V2a leg with real time-surrogate tests:
the preictal dynamics leg. It visualizes why raw near-critical-looking
lambda values are not evidence of an axial critical state.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_ictal_recruitment/v2_criticality"
OUTDIR = BASE / "figures"
OUT = OUTDIR / "v2a_axis_criticality_null_summary.png"

COLORS = {"narrow": "#c0603a", "broad": "#3b6fb0"}
ALPHA = 0.05
P_EPS = 1e-6


def _read_rows(cohort: str) -> list[dict]:
    path = BASE / cohort / "phase2_dynamics_subject.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["cohort"] = cohort
    return rows


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _subject_label(row: dict) -> str:
    return str(row["subject"])


def _endpoint_pass(row: dict, prefix: str) -> bool:
    if prefix == "M":
        return _f(row, "M_phase_empirical_p") < ALPHA and _f(row, "M_block_empirical_p") < ALPHA
    if prefix == "lambda":
        return (
            _f(row, "lambda_trend_phase_empirical_p") < ALPHA
            and _f(row, "lambda_trend_block_empirical_p") < ALPHA
        )
    raise ValueError(prefix)


def _jitter(n: int, seed: int = 20260706) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.08, 0.08, n)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def _plot_m_loading(ax, rows: list[dict]) -> None:
    ax.axhline(0.0, color="0.55", lw=1.0, ls="--", zorder=1)
    offsets = {"narrow": -0.18, "broad": 0.18}
    labels = {"narrow": "narrow", "broad": "broad"}
    for cohort in ("narrow", "broad"):
        vals = [r for r in rows if r["cohort"] == cohort]
        x = np.full(len(vals), offsets[cohort]) + _jitter(len(vals), 1 if cohort == "narrow" else 2)
        y = np.array([_f(r, "M_loading_spearman") for r in vals])
        passed = np.array([_endpoint_pass(r, "M") for r in vals])
        ax.scatter(x[~passed], y[~passed], s=48, color=COLORS[cohort], alpha=0.82,
                   edgecolor="white", linewidth=0.6, label=f"{labels[cohort]} (0/{len(vals)} pass)")
        if passed.any():
            ax.scatter(x[passed], y[passed], s=70, color=COLORS[cohort], marker="*", edgecolor="black")
        med = float(np.nanmedian(y))
        ax.plot([offsets[cohort] - 0.12, offsets[cohort] + 0.12], [med, med],
                color=COLORS[cohort], lw=3)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.36, 0.28)
    ax.set_xticks([-0.18, 0.18])
    ax.set_xticklabels(["narrow", "broad"])
    ax.set_ylabel("M_loading alignment to HFO axis\n(Spearman rho)")
    ax.set_title("A. dominant mode\nno phase+block surrogate pass", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92)


def _plot_lambda(ax, rows: list[dict]) -> None:
    ax.axhline(0.0, color="0.55", lw=1.0, ls="--", zorder=1)
    for cohort in ("narrow", "broad"):
        vals = [r for r in rows if r["cohort"] == cohort]
        x = np.array([_f(r, "lambda_max_late") for r in vals])
        y = np.array([_f(r, "lambda_trend_spearman") for r in vals])
        passed = np.array([_endpoint_pass(r, "lambda") for r in vals])
        ax.scatter(x[~passed], y[~passed], s=52, color=COLORS[cohort], alpha=0.84,
                   edgecolor="white", linewidth=0.6, label=f"{cohort} (0/{len(vals)} pass)")
        if passed.any():
            ax.scatter(x[passed], y[passed], s=72, color=COLORS[cohort], marker="*", edgecolor="black")
    ax.set_xlim(0.89, 0.96)
    ax.set_ylim(-0.26, 0.26)
    ax.set_xlabel("raw late-window spectral radius lambda_max")
    ax.set_ylabel("lambda trend toward onset\n(Spearman rho)")
    ax.set_title("B. raw lambda is high\ntrend is not surrogate-supported", loc="left", fontweight="bold", fontsize=10.5)
    ax.text(0.892, 0.225, "raw lambda ~0.90-0.95 is a smooth-envelope artifact;\nread lambda_surplus / surrogates, not raw lambda",
            fontsize=8.5, color="0.28", va="top", style="italic")
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.92)


def _plot_p_heatmap(ax, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (0 if r["cohort"] == "narrow" else 1, int(r["subject"])))
    p_cols = [
        ("M phase", "M_phase_empirical_p"),
        ("M block", "M_block_empirical_p"),
        ("lambda phase", "lambda_trend_phase_empirical_p"),
        ("lambda block", "lambda_trend_block_empirical_p"),
    ]
    p = np.array([[max(_f(r, col), P_EPS) for _, col in p_cols] for r in rows_sorted], dtype=float)
    z = -np.log10(p)
    im = ax.imshow(z, aspect="auto", cmap="Blues", vmin=0, vmax=2.3)
    ax.set_xticks(range(len(p_cols)))
    ax.set_xticklabels([name for name, _ in p_cols], rotation=0)
    ax.set_yticks(range(len(rows_sorted)))
    ax.set_yticklabels([f"{r['cohort'][0]}:{_subject_label(r)}" for r in rows_sorted], fontsize=8)
    ax.set_title("C. surrogate p values: complete gate requires both cells in an endpoint",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.axhline(6.5, color="white", lw=2.0)

    for i, row in enumerate(rows_sorted):
        for j, (_, col) in enumerate(p_cols):
            pv = _f(row, col)
            sig = np.isfinite(pv) and pv < ALPHA
            text_color = "white" if z[i, j] > 1.35 else "0.25"
            ax.text(j, i, _fmt_p(pv), ha="center", va="center", fontsize=7.2,
                    color=text_color, fontweight="bold" if sig else "normal")
            if sig:
                ax.add_patch(plt.Rectangle((j - 0.49, i - 0.49), 0.98, 0.98,
                                           fill=False, edgecolor="#b22222", lw=1.2))
    ax.text(3.52, -0.75, "red box = nominal p<0.05", fontsize=8, color="#7a1b1b")
    cb = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    cb.set_label("-log10(p)")


def _summary_json(rows: list[dict]) -> dict:
    out = {}
    for cohort in ("narrow", "broad"):
        vals = [r for r in rows if r["cohort"] == cohort]
        out[cohort] = {
            "n": len(vals),
            "m_loading_phase_block_pass": int(sum(_endpoint_pass(r, "M") for r in vals)),
            "lambda_trend_phase_block_pass": int(sum(_endpoint_pass(r, "lambda") for r in vals)),
            "median_lambda_max_late": float(np.nanmedian([_f(r, "lambda_max_late") for r in vals])),
            "median_m_loading_spearman": float(np.nanmedian([_f(r, "M_loading_spearman") for r in vals])),
            "median_lambda_trend_spearman": float(np.nanmedian([_f(r, "lambda_trend_spearman") for r in vals])),
        }
    return out


def main() -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = _read_rows("narrow") + _read_rows("broad")

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28], hspace=0.40, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_m_loading(ax_a, rows)
    _plot_lambda(ax_b, rows)
    _plot_p_heatmap(ax_c, rows)

    summary = _summary_json(rows)
    fig.suptitle("Topic 5 V2a axial criticality sanity check: no surrogate-supported preictal axial state",
                 x=0.02, y=0.99, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(
        0.02,
        0.015,
        (
            f"Dynamics leg only (the V2a leg with real phase+block surrogate nulls): "
            f"M_loading pass narrow {summary['narrow']['m_loading_phase_block_pass']}/{summary['narrow']['n']}, "
            f"broad {summary['broad']['m_loading_phase_block_pass']}/{summary['broad']['n']}; "
            f"lambda trend pass narrow {summary['narrow']['lambda_trend_phase_block_pass']}/{summary['narrow']['n']}, "
            f"broad {summary['broad']['lambda_trend_phase_block_pass']}/{summary['broad']['n']}."
        ),
        fontsize=9,
        color="0.28",
    )
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"[fig] -> {OUT}")
    return OUT


if __name__ == "__main__":
    main()
