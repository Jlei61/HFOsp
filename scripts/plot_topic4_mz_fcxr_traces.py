"""Per-cell time-trace inspection (reviewer step 4): plot rate + population/core/axis/off-axis participation
over time so the dynamics TYPE can be judged by eye — interictal event train (brief spikes on a quiet floor),
long transient (elevated then decays), fixed high (flat plateau), or oscillatory high (sustained bursting).

Usage: python scripts/plot_topic4_mz_fcxr_traces.py <run_dir> [cell_label ...]
Default cells: the near-transition landmarks (D0.085 / D0.1, native-low + strong-kick high).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                       "fast_slow_dynamics", "figures")
DEFAULT_CELLS = ["D0.085_low_T1", "D0.085_high2_T1", "D0.1_low_T1", "D0.1_high2_T1"]


def _load(run_dir, label):
    npz = np.load(os.path.join(run_dir, "per_cell", f"{label}_trace.npz"))
    row = json.load(open(os.path.join(run_dir, "per_cell", f"{label}.json")))
    return npz, row


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(
        os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                     "fast_slow_dynamics", "runs", "*grid*")))[-1]
    cells = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_CELLS
    cells = [c for c in cells if os.path.exists(os.path.join(run_dir, "per_cell", f"{c}_trace.npz"))]
    if not cells:
        raise SystemExit(f"no trace npz for {DEFAULT_CELLS} in {run_dir}/per_cell")

    fig, axes = plt.subplots(len(cells), 2, figsize=(13, 2.6 * len(cells)), squeeze=False)
    for i, label in enumerate(cells):
        npz, row = _load(run_dir, label)
        af_bin = float(npz["af_bin_ms"][0]); rate_dt = float(npz["rate_dt_ms"][0])
        t_rate = np.arange(npz["rate_E"].size) * rate_dt
        t_af = np.arange(npz["af"].size) * af_bin
        a0 = float(row["analysis_start_ms"])
        axR, axP = axes[i]
        axR.plot(t_rate, npz["rate_E"], color="#333", lw=0.7)
        axR.axvline(a0, ls=":", color="0.6", lw=1)
        axR.set_ylabel("rate_E (Hz)"); axR.set_xlim(0, t_rate[-1] if t_rate.size else 1)
        axR.set_title(f"{label}  ->  {row.get('provisional_label','?')}   "
                      f"(env_occ={row.get('env_occ',float('nan')):.2f}, end={row.get('env_end_occ',float('nan')):.2f}, "
                      f"mod={row.get('env_modulation',float('nan')):.2f})", fontsize=9, loc="left")
        for key, c, lab in (("af", "#000", "population"), ("af_core", "#c44e52", "core"),
                            ("af_axis", "#dd8452", "axis-band"), ("af_off", "#4c72b0", "off-axis")):
            axP.plot(t_af[:npz[key].size], npz[key], color=c, lw=0.8, label=lab, alpha=0.85)
        axP.axhline(float(row["baseline_af_q95"]), ls="--", color="0.5", lw=0.8)
        axP.axvline(a0, ls=":", color="0.6", lw=1)
        axP.set_ylabel("active fraction"); axP.set_xlim(0, t_af[-1] if t_af.size else 1)
        if i == 0:
            axP.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.suptitle(f"FCXR Stage D per-cell traces (dynamics-type inspection)  —  {os.path.basename(run_dir)}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "landmark_traces.png"), dpi=150)
    print(f"wrote {OUT_DIR}/landmark_traces.png  ({len(cells)} cells from {os.path.basename(run_dir)})")


if __name__ == "__main__":
    main()
