#!/usr/bin/env python
"""Diagnostic plotter for the SNN-native M4 exit line (Stage-1a exit atlas + Stage-2 arms).

Reads an `exit_atlas_*.npz` / `arms_*.npz` (+ its `.json`) and draws, one column per
cell/arm, a 3-row time-series panel:
  row1 = smoothed E population rate (Hz)   [+ inhibitory-pulse window shaded, exit_atlas]
  row2 = mean q_I (inhibitory resource; refill during a quiet hold is the exit lever)
  row3 = S_G (divisive containment pool) + p_mean/p_max (persistence recovery field)
The title carries the verdict / termination_class so the failure/success TYPE is visible
(honest diagnostic, not cherry-picked). Saves PNG + PDF next to a figures/ dir.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DT = 0.1  # ms/step


def _smooth(x, win=200):
    x = np.asarray(x, float)
    if x.size < 3:
        return x
    k = min(win, max(1, x.size // 5))
    return np.convolve(x, np.ones(k) / k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    a.json = a.json or a.npz.replace(".npz", ".json")
    d = np.load(a.npz, allow_pickle=True)
    meta_rows = {}
    if os.path.exists(a.json):
        j = json.load(open(a.json))
        meta = j.get("meta", {})
        meta_rows = {r["label"]: r for r in j.get("rows", []) if "label" in r}
    else:
        meta = {}
    # discover cell labels from the "<label>__rate" keys; order: baseline/arms first, then holds by duration
    def _sortkey(lab):
        if lab.startswith("hold"):
            try:
                return (1, float(lab[4:]))
            except ValueError:
                return (2, 0.0, lab)
        return (0, 0.0, lab)
    labels = sorted({k.rsplit("__", 1)[0] for k in d.files if k.endswith("__rate")}, key=_sortkey)
    if not labels:
        raise SystemExit(f"no '<label>__rate' arrays in {a.npz}")
    t0 = meta.get("t0"); holds = meta.get("holds")

    nC = len(labels)
    fig, axes = plt.subplots(3, nC, figsize=(3.6 * nC, 7.2), squeeze=False, sharex="col")
    for ci, lab in enumerate(labels):
        rate = np.asarray(d[f"{lab}__rate"], float)
        t = np.arange(rate.size) * DT
        qI = np.asarray(d[f"{lab}__trace_qI_mean"], float) if f"{lab}__trace_qI_mean" in d.files else None
        SG = np.asarray(d[f"{lab}__trace_SG"], float) if f"{lab}__trace_SG" in d.files else None
        pm = np.asarray(d[f"{lab}__trace_p_mean"], float) if f"{lab}__trace_p_mean" in d.files else None
        px = np.asarray(d[f"{lab}__trace_p_max"], float) if f"{lab}__trace_p_max" in d.files else None
        mr = meta_rows.get(lab, {})

        ax = axes[0][ci]
        ax.plot(t, _smooth(rate), lw=0.9, color="#1f77b4")
        ax.set_title(f"{lab}\n{mr.get('verdict','')} / {mr.get('termination_class','')}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("E rate (Hz)")
        # pulse window (exit_atlas): t0 -> t0+hold for this cell
        if t0 is not None and lab.startswith("hold"):
            try:
                h = float(lab.replace("hold", ""))
                ax.axvspan(t0, t0 + h, color="0.85", zorder=0)
            except ValueError:
                pass

        ax = axes[1][ci]
        if qI is not None and qI.size:
            ax.plot(np.arange(qI.size) * DT, qI, lw=0.9, color="#2ca02c")
        ax.set_ylim(0, 1.05)
        if ci == 0:
            ax.set_ylabel("mean q_I")

        ax = axes[2][ci]
        if SG is not None and SG.size:
            ax.plot(np.arange(SG.size) * DT, SG, lw=0.9, color="#d62728", label="S_G")
        if pm is not None and pm.size:
            ax.plot(np.arange(pm.size) * DT, pm, lw=0.9, color="#9467bd", label="p_mean")
        if px is not None and px.size:
            ax.plot(np.arange(px.size) * DT, px, lw=0.7, color="#9467bd", ls="--", label="p_max")
        ax.set_xlabel("t (ms)")
        if ci == 0:
            ax.set_ylabel("S_G / p")
        if ci == nC - 1 and ((SG is not None and SG.size) or (pm is not None and pm.size)):
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(a.title or os.path.basename(a.npz), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out or os.path.join(os.path.dirname(a.npz), "figures",
                                os.path.basename(a.npz).replace(".npz", ".png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
