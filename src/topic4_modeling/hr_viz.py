"""Phase portrait + nullcline + regime-map visualization.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hr_core import HRParams


# ── Nullclines (closed-form) ─────────────────────────────────────────────

def compute_x_nullcline(x_grid: np.ndarray, params: HRParams,
                         z: float, I: float) -> np.ndarray:
    """y where dx/dt = 0: y = a x³ - b x² + z - I (with eta=0)."""
    p = params
    return p.a * x_grid**3 - p.b * x_grid**2 + z - I


def compute_y_nullcline(x_grid: np.ndarray, params: HRParams) -> np.ndarray:
    """y where dy/dt = 0: y = c - d x²."""
    p = params
    return p.c - p.d * x_grid**2


# ── Phase portrait ───────────────────────────────────────────────────────

def plot_phase_portrait(trajectory: np.ndarray, params: HRParams,
                         I: float, figsize=(8.0, 6.0)) -> plt.Figure:
    """Plot x-y phase plane with trajectory + nullclines at mean(z)."""
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z_mean = float(trajectory[:, 2].mean())
    x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
    y_xnull = compute_x_nullcline(x_grid, params, z_mean, I)
    y_ynull = compute_y_nullcline(x_grid, params)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, alpha=0.4, color="C0", linewidth=0.5, label="trajectory")
    ax.plot(x_grid, y_xnull, "r-", label=f"dx/dt=0  (z≈{z_mean:.2f})")
    ax.plot(x_grid, y_ynull, "g-", label="dy/dt=0")
    ax.scatter([x[0]], [y[0]], marker="o", color="black", zorder=5, label="start")
    ax.scatter([x[-1]], [y[-1]], marker="s", color="black", zorder=5, label="end")
    ax.set_xlabel("x (fast voltage-like)")
    ax.set_ylabel("y (spiking variable)")
    ax.set_title(f"HR phase portrait  (I={I:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    margin_y = 0.1 * (y.max() - y.min() + 1e-6)
    ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
    return fig


# ── Regime map heatmap ───────────────────────────────────────────────────

REGIME_COLOR = {
    "silent": "#dddddd",
    "excitable": "#4daf4a",
    "repetitive-burst": "#ff7f00",
    "unstable": "#e41a1c",
}
REGIME_ORDER = ["silent", "excitable", "repetitive-burst", "unstable"]


def plot_regime_map(sweep_df: pd.DataFrame, figsize=None) -> plt.Figure:
    """Subplot grid: rows=sigma_ou, cols=r_used, x-axis=I, colored by modal regime."""
    sigmas = sorted(sweep_df["sigma_ou"].unique())
    rs = sorted(sweep_df["r_used"].unique())
    Is = sorted(sweep_df["I"].unique())
    n_rows, n_cols = len(sigmas), len(rs)
    if figsize is None:
        figsize = (3.0 * n_cols, 1.2 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)
    for i, sigma in enumerate(sigmas):
        for j, r in enumerate(rs):
            ax = axes[i, j]
            cell_data = []
            for I in Is:
                sub = sweep_df[(sweep_df["sigma_ou"] == sigma)
                                & (sweep_df["r_used"] == r)
                                & (sweep_df["I"] == I)]
                cell_data.append(sub["regime"].mode().iloc[0] if not sub.empty else "silent")
            colors = [REGIME_COLOR[c] for c in cell_data]
            ax.bar(range(len(Is)), [1] * len(Is), color=colors, width=1.0,
                    edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(Is)))
            ax.set_xticklabels([f"{I:.1f}" for I in Is], rotation=45, fontsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"r={r:.4f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"σ_OU={sigma:.2f}", fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("I", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[r]) for r in REGIME_ORDER]
    fig.legend(handles, REGIME_ORDER, loc="upper center",
                ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.suptitle("HR single-node regime map", y=0.94, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
