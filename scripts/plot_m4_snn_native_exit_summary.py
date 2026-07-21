#!/usr/bin/env python
"""Composite NO-GO diagnostic for the SNN-native M4 exit line (Figure-5 diagnostic).

Pulls representative trajectories from several run npz files and lays them out in columns:
  row 1 = smoothed E rate (Hz)
  row 2 = mean q_I (inhibitory reserve, green) + S_G (divisive containment pool, red), overlaid
The row-2 overlay is the point: across every failure mode, q_I (refills only when QUIET) and S_G
(engages only when ACTIVE) are ANTI-PHASE -- there is no continuous trajectory with both high, so
the bounded state cannot be exited to interictal. Columns: bounded persist / open-loop hold rebound /
symmetric-current fragment-wander / asymmetric-current runaway-train.

Usage: pairs given as npz:arm:label triples via --panels.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DT = 0.1


def _smooth(x, win=250):
    x = np.asarray(x, float)
    if x.size < 3:
        return x
    k = min(win, max(1, x.size // 5))
    return np.convolve(x, np.ones(k) / k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True, help="each: npz_path::arm::title")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suptitle", default="SNN-native M4 exit — BOUNDED-NEGATIVE")
    a = ap.parse_args()
    panels = []
    for tok in a.panels:
        npz, arm, title = tok.split("::")
        d = np.load(npz, allow_pickle=True)
        rate = np.asarray(d[f"{arm}__rate"], float)
        qi = np.asarray(d[f"{arm}__trace_qI_mean"], float) if f"{arm}__trace_qI_mean" in d.files else None
        sg = np.asarray(d[f"{arm}__trace_SG"], float) if f"{arm}__trace_SG" in d.files else None
        panels.append((title, rate, qi, sg))

    nC = len(panels)
    fig, axes = plt.subplots(2, nC, figsize=(3.5 * nC, 5.0), squeeze=False, sharex="col")
    for ci, (title, rate, qi, sg) in enumerate(panels):
        t = np.arange(rate.size) * DT / 1000.0
        ax = axes[0][ci]
        ax.plot(t, _smooth(rate), lw=0.9, color="#1f77b4")
        ax.set_title(title, fontsize=9)
        if ci == 0:
            ax.set_ylabel("E rate (Hz)")
        ax = axes[1][ci]
        if qi is not None and qi.size:
            ax.plot(np.arange(qi.size) * DT / 1000.0, qi, lw=1.0, color="#2ca02c", label="q_I (reserve)")
        if sg is not None and sg.size:
            ax.plot(np.arange(sg.size) * DT / 1000.0, sg, lw=1.0, color="#d62728", label="S_G (containment)")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("t (s)")
        if ci == 0:
            ax.set_ylabel("q_I / S_G")
        if ci == nC - 1:
            ax.legend(fontsize=7, loc="center right")
    fig.suptitle(a.suptitle + "  —  q_I (green) & S_G (red) respond to activity with opposite signs (negatively correlated)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    fig.savefig(a.out.replace(".png", ".pdf"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
