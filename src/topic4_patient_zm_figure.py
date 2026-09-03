"""Shared paper-facing drawing helpers for the patient Z/M fold bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FOLD = "#C94C4C"
HIGH = "#8C5B9E"
RETURNED = "#C98A22"
LOW = "#304D73"
CORE = "#243F68"
MEAN = "#6F8EAE"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_projection(json_path):
    json_path = Path(json_path).resolve()
    payload = json.loads(json_path.read_text())
    npz_path = Path(payload["arrays"]["path"]).resolve()
    if sha256(npz_path) != payload["arrays"]["sha256"]:
        raise RuntimeError("dynamic-projection NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return payload, arrays


def draw_critical_manifold_trajectory(
        ax, projection, arrays, *, seed=1842, add_rate_colorbar=True,
        show_legend=True):
    """Draw the reduced Z/M critical manifold and one spatial SNN trajectory.

    The reduced branch is projected with ``A=eta_M tau_M r_E``.  The SNN has a
    spatial q field, so q_core and q_mean are intentionally shown as two paths
    rather than collapsed into one alleged switching coordinate.
    """
    manifold = projection["manifold"]
    eta_m = float(manifold["eta_m"])
    tau_m = float(manifold["tau_m_ms"])
    q_fold = float(manifold["q_fold"])
    rate_fold = float(manifold["mean_rate_e_hz_at_fold"])
    prefix = f"seed{int(seed)}"
    time = np.asarray(arrays[f"{prefix}_time_ms"], float)
    q_core = np.asarray(arrays[f"{prefix}_q_core"], float)
    q_mean = np.asarray(arrays[f"{prefix}_q_mean"], float)
    m = np.asarray(arrays[f"{prefix}_M"], float)
    rate = np.asarray(arrays[f"{prefix}_rate_E_20ms_hz"], float)
    run = next(row for row in projection["runs"] if int(row["seed"]) == int(seed))
    onset = float(run["scientific_onset_ms"])
    keep = time <= min(float(time[-1]), onset + 800.0)
    time, q_core, q_mean, m, rate = (
        value[keep] for value in (time, q_core, q_mean, m, rate))
    d_core, d_mean = 1.0 - q_core, 1.0 - q_mean
    adaptation = eta_m * m

    def branch(name, rate_name, color, style, width, label):
        q = np.asarray(arrays[f"manifold_{name}"], float)
        r = np.asarray(arrays[f"manifold_{rate_name}"], float)
        ax.plot(1.0 - q, eta_m * tau_m * r / 1000.0,
                color=color, ls=style, lw=width, label=label, zorder=1)

    branch("low_q", "low_rate_e_hz", LOW, "-", 1.2,
           "near-silent branch")
    branch("returned_q", "returned_rate_e_hz", RETURNED, "--", 1.35,
           "returned branch")
    branch("high_q", "high_rate_e_hz", HIGH, "-", 1.65,
           "high-rate skeleton\n(delay-unstable)")
    fold_d = 1.0 - q_fold
    fold_a = eta_m * tau_m * rate_fold / 1000.0
    ax.scatter(fold_d, fold_a, marker="*", s=82, color=FOLD,
               ec="white", lw=0.55, zorder=6, label="saddle-node (1 mm)")
    ax.axvline(fold_d, color=FOLD, ls=":", lw=0.85, alpha=0.8, zorder=0)

    ax.plot(d_core, adaptation, color=CORE, lw=1.35,
            label=r"SNN $q_{core}$", zorder=4)
    ax.plot(d_mean, adaptation, color=MEAN, lw=1.15, ls="--",
            label=r"SNN $q_{mean}$", zorder=3)
    stride = max(1, int(round(len(time) / 90)))
    norm = Normalize(0.0, 420.0)
    scatter = ax.scatter(
        d_mean[::stride], adaptation[::stride], c=rate[::stride],
        s=7.0, cmap="viridis", norm=norm, lw=0, zorder=5)
    onset_index = int(np.argmin(np.abs(time - onset)))
    ax.plot([d_core[onset_index], d_mean[onset_index]],
            [adaptation[onset_index], adaptation[onset_index]],
            color=FOLD, lw=0.8, zorder=6)
    ax.scatter([d_core[onset_index], d_mean[onset_index]],
               [adaptation[onset_index]] * 2, marker="D", s=20,
               color=FOLD, ec="white", lw=0.4, zorder=7)
    ax.text(fold_d + 0.004, fold_a + 0.003, "fold", color=FOLD,
            fontsize=6.7, ha="left", va="bottom")
    ax.text(max(d_core[onset_index], d_mean[onset_index]) + 0.003,
            adaptation[onset_index], "onset", color=FOLD,
            fontsize=6.5, va="center")

    ax.set_xlabel(r"Disinhibition $D=1-q$", fontsize=8.4)
    ax.set_ylabel(r"Adaptation $A=\eta_M M$", fontsize=8.4)
    ax.set_xlim(-0.006, 0.238)
    ax.set_ylim(-0.004, 0.108)
    ax.tick_params(labelsize=7.1, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=5.8, loc="upper left",
                  handlelength=2.0, borderaxespad=0.15, labelspacing=0.30)
    if add_rate_colorbar:
        color_axis = inset_axes(
            ax, width="4%", height="38%", loc="center left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes, borderpad=0)
        colorbar = ax.figure.colorbar(scatter, cax=color_axis)
        # Colorbar defaults rasterize dense solids.  With tight PDF bounding
        # boxes that can detach the gradient image from its vector outline.
        # Keep the small colorbar fully vector so PNG/PDF/SVG agree.
        colorbar.solids.set_rasterized(False)
        colorbar.set_ticks([0, 200, 400])
        colorbar.ax.tick_params(labelsize=5.7, length=1.5)
        colorbar.set_label(r"SNN $r_E$ (Hz)", fontsize=6.0, labelpad=1.5)
    return {
        "seed": int(seed),
        "trajectory_stop_ms": float(time[-1]),
        "scientific_onset_ms": onset,
        "q_coordinates": ["q_core", "q_mean"],
        "rate_encoding": "20-ms-smoothed SNN E rate on q_mean path",
        "fold": {"q": q_fold, "D": fold_d,
                 "mean_rate_e_hz": rate_fold, "A": fold_a},
        "high_branch_stability": "delay-unstable in the audited closure",
    }
