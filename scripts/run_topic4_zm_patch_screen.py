#!/usr/bin/env python
"""Path-B cheap-first screen driver (task §7). Runs the reduced K-patch inhibitory-containment model
across inhibitory STRUCTURES (single global scalar S_G / patchwise local pools / patchwise+spatial-smooth
/ local+weak-global) and reports, for each, whether it desynchronizes the microdomains and sustains the
population carrier proxy (occupancy vs the OFF state). Cheap (seconds); NOT the SNN.

Saves a JSON summary + a 4-panel figure. A POSITIVE structure here only justifies MIGRATING to the full
anisotropic SNN (inhibition-side only, no E->E change), which must then pass the pre-registered A+B gate.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_patch_screen import PatchParams, simulate, population_signal, screen_metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "zm_patch_screen")
BASE = dict(K=16, I0=1.0, w_rec=2.0, sigma_I=0.4, w_c=0.05)   # calibrated relaxation-oscillation regime
SEEDS = range(4)

STRUCTURES = [
    ("global_homog", dict(mode="global", sigma_I=0.0, w_c=0.0), "single global scalar S_G (homogeneous) = SNN analogue"),
    ("global_het", dict(mode="global"), "single global scalar S_G (heterogeneous)"),
    ("patchwise", dict(mode="patchwise"), "patchwise independent local pools"),
    ("patchwise_smooth", dict(mode="patchwise", pool_sigma=1.0), "patchwise + spatial smoothing (sigma=1)"),
    ("local_weak_global", dict(mode="local_global", eps_global=0.2), "local + weak global (eps=0.2)"),
]


def _avg(kw):
    ms = [screen_metrics(simulate(PatchParams(**{**BASE, **kw, "seed": s}), T_ms=6000.0)) for s in SEEDS]
    keys = ("occupancy", "synchrony", "mean_activity", "pop_depth", "patch_osc")
    out = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    out["carrier_pass"] = int(sum(m["carrier_proxy"] for m in ms))
    out["n_seeds"] = len(SEEDS)
    return out


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    summary = {"base": BASE, "structures": {}, "K_scan_patchwise": {}}
    for name, kw, desc in STRUCTURES:
        summary["structures"][name] = dict(desc=desc, params=kw, **_avg(kw))
    for K in (2, 4, 8, 16, 32, 64):
        summary["K_scan_patchwise"][K] = _avg(dict(mode="patchwise", K=K))
    with open(os.path.join(OUT, "patch_screen_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # representative single-seed traces for the figure
    rg = simulate(PatchParams(**{**BASE, "mode": "global", "sigma_I": 0.0, "w_c": 0.0, "seed": 1}), T_ms=6000.0)
    rp = simulate(PatchParams(**{**BASE, "mode": "patchwise", "seed": 1}), T_ms=6000.0)
    tg, Pg = rg["t"], population_signal(rg["a"])
    tp, Pp = rp["t"], population_signal(rp["a"])

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.25, height_ratios=[1, 1.1, 1])
    # (A) population signal: global burst-train (collapses to OFF) vs patchwise (sustained)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tg, Pg, lw=0.8, color="#d62728", label="global scalar S_G — population P(t)")
    ax.plot(tp, Pp, lw=0.8, color="#1f77b4", label="patchwise local pools — population P(t)")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("population P = mean(a)")
    ax.set_title("(A) global S_G collapses P to the OFF state between synchronized bursts (train); "
                 "patchwise keeps P elevated (sustained)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    # (B,C) patch activity heatmaps: synchronized bands vs desynchronized speckle
    for col, r, lab in ((0, rg, "global: synchronized"), (1, rp, "patchwise: desynchronized")):
        axh = fig.add_subplot(gs[1, col])
        A = r["a"].T
        axh.imshow(A, aspect="auto", origin="lower", cmap="magma",
                   extent=[r["t"][0], r["t"][-1], 0, A.shape[0]], vmax=np.percentile(A, 99) + 1e-6)
        axh.set_xlabel("t (ms)"); axh.set_ylabel("patch index"); axh.set_title(f"(B/C) {lab}", fontsize=9)
    # (D) structure comparison bars
    axb = fig.add_subplot(gs[2, :])
    names = [s[0] for s in STRUCTURES]
    occ = [summary["structures"][n]["occupancy"] for n in names]
    sync = [summary["structures"][n]["synchrony"] for n in names]
    osc = [summary["structures"][n]["patch_osc"] for n in names]
    x = np.arange(len(names)); w = 0.27
    axb.bar(x - w, occ, w, label="pop occupancy (vs OFF)", color="#1f77b4")
    axb.bar(x, sync, w, label="synchrony (across patches)", color="#d62728")
    axb.bar(x + w, np.array(osc) * 10, w, label="patch oscillation ×10", color="#2ca02c")
    axb.axhline(0.8, color="#1f77b4", ls=":", lw=0.8)
    for i, n in enumerate(names):
        cp = summary["structures"][n]["carrier_pass"]
        axb.text(i, 1.05, f"{cp}/4", ha="center", fontsize=8,
                 color="green" if cp == 4 else "0.4", fontweight="bold")
    axb.set_xticks(x); axb.set_xticklabels(names, fontsize=8, rotation=8)
    axb.set_ylabel("metric"); axb.legend(fontsize=7, loc="center right")
    axb.set_title("(D) carrier-proxy by inhibitory structure (occ≥0.8 + oscillating + desync ⇒ pass); "
                  "number above = seeds passing", fontsize=9)

    fig.suptitle("Path-B cheap-first screen (reduced K-patch rate model, NOT the SNN): does spatially-resolved "
                 "inhibition turn the global-S_G burst train into a sustained population carrier?", fontsize=11)
    out = os.path.join(OUT, "figures", "patch_screen.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    for n in names:
        s = summary["structures"][n]
        print(f"  {n:20s} occ={s['occupancy']:.3f} sync={s['synchrony']:+.2f} osc={s['patch_osc']:.3f} pass={s['carrier_pass']}/4")


if __name__ == "__main__":
    main()
