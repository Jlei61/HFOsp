#!/usr/bin/env python
"""Diagnostic plotter for the Z/M-native containment-to-exit lifecycle (run_zm_snn_native_exit.py).

One column per arm, 3 rows (honest diagnostic -- the termination_class is in the title so the
success/failure TYPE is visible):
  row1 = E population rate: core (red) vs surround (blue) vs all (grey)   [the event itself]
  row2 = z (inhibitory efficacy): mean / core / surround / min           [disinhibition slow driver]
  row3 = m adaptation (purple) + S_G containment pool (red --) + H memory (brown)
Reads {arm}_seed{seed}.npz + lifecycle_seed{seed}.json. Saves PNG+PDF under figures/.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DT = 0.1
BIN_MS = 25.0  # rate/af bin (matches the runner)


def _t_rate(n):
    return np.arange(n) * BIN_MS


def _t_step(n):
    return np.arange(n) * DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results dir with {arm}_seed{seed}.npz + lifecycle json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jp = os.path.join(a.dir, f"lifecycle_seed{a.seed}.json")
    meta = json.load(open(jp)) if os.path.exists(jp) else {"rows": []}
    rows = {r["label"]: r for r in meta.get("rows", [])}
    arms = [r["label"] for r in meta.get("rows", [])] or \
        sorted(f[:-len(f"_seed{a.seed}.npz")] for f in os.listdir(a.dir) if f.endswith(f"_seed{a.seed}.npz"))
    if not arms:
        raise SystemExit(f"no arms found in {a.dir}")

    nC = len(arms)
    fig, axes = plt.subplots(3, nC, figsize=(4.2 * nC, 8.0), squeeze=False, sharex="col")
    for ci, arm in enumerate(arms):
        d = np.load(os.path.join(a.dir, f"{arm}_seed{a.seed}.npz"))
        r = rows.get(arm, {})

        ax = axes[0][ci]
        cr, sr, ar = d["core_rate"], d["surr_rate"], d["all_rate"]
        ax.plot(_t_rate(len(ar)), ar, lw=0.7, color="0.6", label="all E")
        ax.plot(_t_rate(len(cr)), cr, lw=1.0, color="#d62728", label="core")
        ax.plot(_t_rate(len(sr)), sr, lw=1.0, color="#1f77b4", label="surround")
        cls = r.get("termination_class", ""); rw = r.get("runaway_ms")
        ax.set_title(f"{arm}\ncls={cls}" + (f"  runaway@{rw:.0f}ms" if rw else ""), fontsize=9)
        if ci == 0:
            ax.set_ylabel("E rate (Hz)")
        if ci == nC - 1:
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[1][ci]
        for key, col, lab, ls in (("z_mean", "k", "z mean", "-"), ("z_core", "#ff7f0e", "z core", "-"),
                                  ("z_surround", "#2ca02c", "z surround", "-"), ("z_min", "0.5", "z min", "--")):
            v = d[key]
            if v.size:
                ax.plot(_t_step(len(v)), v, lw=1.0, color=col, ls=ls, label=lab)
        ax.set_ylim(-0.02, 1.05)
        if ci == 0:
            ax.set_ylabel("z (inhib. efficacy)")
        if ci == nC - 1:
            ax.legend(fontsize=6, loc="lower left")

        ax = axes[2][ci]
        mm = d["m_mean"]
        if mm.size:
            ax.plot(_t_step(len(mm)), mm, lw=1.0, color="#9467bd", label="m mean")
        ax.set_xlabel("t (ms)")
        if ci == 0:
            ax.set_ylabel("m / S_G / H")
        ax2 = ax.twinx()
        for key, col, lab in (("SG", "#d62728", "S_G"), ("H", "#8c564b", "H")):
            v = d[key]
            if v.size and float(np.max(v)) > 0:
                ax2.plot(_t_step(len(v)), v, lw=1.1, color=col, ls="--", label=lab)
        ax2.set_ylim(bottom=0)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")

    sub = meta.get("subject", ""); Ith = meta.get("I_th_EI")
    fig.suptitle(f"Z/M-native lifecycle — {sub}  (I_th_EI=q75={Ith:.3f}, lockpoint {meta.get('lockpoint','')})"
                 if Ith is not None else "Z/M-native lifecycle", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = a.out or os.path.join(a.dir, "figures", f"zm_lifecycle_seed{a.seed}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
