#!/usr/bin/env python
"""Spatial-frame plotter for the SNN-native M4 exit line.

Reads an `arms_*.npz` / `exit_atlas_*.npz` and renders, for one arm/cell, a row of 2D
E-activity fields (the saved coarse `movie`, MOVIE_GRID x MOVIE_GRID @ 25ms bins) at chosen
timepoints -- e.g. baseline IED / pre-onset / ictal / termination / recovery. Overlays the
two low-threshold core centroids (source red, sink blue) and the E->E axis. This is the
source-space spatial readout (coarse); use for Figure-5 lifecycle frames (PASS) or the
failure-mode diagnostic (NO-GO). Per-neuron onset gradient needs E_spk_bool (a targeted re-run).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MOVIE_BIN_MS = 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--arm", required=True, help="cell label, e.g. D_tau3000_eta150")
    ap.add_argument("--times", default=None, help="comma ms; default 6 evenly spaced")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    key = f"{a.arm}__movie"
    if key not in d.files:
        raise SystemExit(f"{key} not in {a.npz}; arms present: "
                         f"{sorted({k.rsplit('__',1)[0] for k in d.files if k.endswith('__movie')})}")
    movie = np.asarray(d[key], float)                 # (n_frames, G, G)
    L = float(d["L"]) if "L" in d.files else 20.0
    src = np.asarray(d["src_xy"]) if "src_xy" in d.files else None
    snk = np.asarray(d["snk_xy"]) if "snk_xy" in d.files else None
    nfr = movie.shape[0]
    if a.times:
        times = [float(x) for x in a.times.split(",")]
    else:
        times = list(np.linspace(0.05, 0.95, 6) * nfr * MOVIE_BIN_MS)
    frames = [min(nfr - 1, max(0, int(t / MOVIE_BIN_MS))) for t in times]

    vmax = float(np.percentile(movie, 99.5)) or 1.0
    fig, axes = plt.subplots(1, len(frames), figsize=(2.5 * len(frames), 2.9), squeeze=False)
    for ci, (fr, t) in enumerate(zip(frames, times)):
        ax = axes[0][ci]
        ax.imshow(movie[fr], origin="lower", extent=(0, L, 0, L), cmap="magma",
                  vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        if src is not None:
            ax.plot(src[0], src[1], "o", mfc="none", mec="#ff3b3b", mew=1.6, ms=10)
        if snk is not None:
            ax.plot(snk[0], snk[1], "o", mfc="none", mec="#3b9bff", mew=1.6, ms=10)
        if src is not None and snk is not None:
            ax.plot([src[0], snk[0]], [src[1], snk[1]], "-", color="w", lw=0.6, alpha=0.5)
        ax.set_title(f"t={t:.0f} ms", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{a.arm} — source-space activity frames (magma; ○ red=source ○ blue=sink)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = a.out or os.path.join(os.path.dirname(a.npz), "figures", f"spatial_{a.arm}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
