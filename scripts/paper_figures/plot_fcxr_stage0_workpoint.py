"""FCXR Stage 0B workpoint figure: does full conductance preserve the interictal workpoint across the
allowed c_E bracket? Three panels, each one question:
  A  rate_E trace per c_E arm vs the current-model reference (silent? hot? interictal?)
  B  event-profile bars (n_returning / dur / participation / peak rate) vs the reference workpoint bands
  C  numerical safety over time (conductance-cap clip fraction + tau_eff_min) per c_E

Reads a workpoint run dir (default: latest_workpoint.json). Rebuildable from summary.json + per_cell traces.
Output: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/figures/stage0_workpoint.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
FIGDIR = os.path.join(OUT_ROOT, "figures")
C_COLORS = {0.85: "#4C72B0", 1.0: "#DD8452", 1.15: "#C44E52"}


def _load(run_dir):
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    traces = {}
    for f in glob.glob(os.path.join(run_dir, "per_cell", "*_trace.npz")):
        lab = os.path.basename(f).split("_seed")[0]
        traces[lab] = dict(np.load(f, allow_pickle=True))
    return summ, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = json.load(open(os.path.join(OUT_ROOT, "latest_workpoint.json")))["path"]
    summ, traces = _load(run_dir)
    rows = summ["rows"]
    ref = summ["reference_workpoint"]
    base = summ["baseline"]["baseline"]

    os.makedirs(FIGDIR, exist_ok=True)
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.28)

    # ---- Panel A: rate traces ----
    axA = fig.add_subplot(gs[0, 0])
    for r in rows:
        cE = r["cfg"]["c_E"]; lab = r["label"]
        col = C_COLORS.get(cE, "0.4")
        tr = traces.get(lab)
        if tr is None:
            continue
        dt = float(tr["trace_dt_ms"][0]); rate = np.asarray(tr["rate_E"], float)
        t = np.arange(rate.size) * dt / 1000.0
        axA.plot(t, rate, color=col, lw=0.8, alpha=0.9,
                 label=f"c_E={cE:g} ({r.get('phenotype','?')}, n_ret={r['event_profile']['n_returning']})")
    axA.axhspan(ref["act_lo"], ref["act_hi"], color="0.75", alpha=0.35, zorder=0,
                label=f"ref interictal peak band [{ref['act_lo']:.0f},{ref['act_hi']:.0f}]Hz")
    axA.set_xlabel("time (s)"); axA.set_ylabel("E rate (Hz)")
    axA.set_title("A · full-conductance dynamics per c_E", fontsize=11, loc="left")
    axA.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # ---- Panel B: event-profile bars vs reference bands ----
    axB = fig.add_subplot(gs[0, 1])
    metrics = [("n_returning", ref.get("n_events"), "returning\nevents"),
               ("duration_median_ms", ref["dur_med"], "dur\nmed (ms)"),
               ("peak_rate_median_hz", 0.5 * (ref["act_lo"] + ref["act_hi"]), "peak\nrate (Hz)")]
    x = np.arange(len(metrics)); w = 0.24
    for j, r in enumerate(rows):
        cE = r["cfg"]["c_E"]; col = C_COLORS.get(cE, "0.4")
        vals = [r["event_profile"].get(m[0], np.nan) for m in metrics]
        axB.bar(x + (j - (len(rows) - 1) / 2) * w, vals, w, color=col, label=f"c_E={cE:g}")
    for i, (_, refv, _) in enumerate(metrics):
        if refv is not None and np.isfinite(refv):
            axB.plot([i - 0.4, i + 0.4], [refv, refv], color="k", lw=1.6, ls="--", zorder=5)
    axB.set_xticks(x); axB.set_xticklabels([m[2] for m in metrics], fontsize=8)
    axB.set_title("B · event profile vs reference (dashed)", fontsize=11, loc="left")
    axB.legend(fontsize=7, loc="upper right")

    # ---- Panel C: numerical safety ----
    axC = fig.add_subplot(gs[0, 2])
    axC2 = axC.twinx()
    for r in rows:
        cE = r["cfg"]["c_E"]; lab = r["label"]; col = C_COLORS.get(cE, "0.4")
        tr = traces.get(lab)
        if tr is None:
            continue
        dt = float(tr["trace_dt_ms"][0])
        clip = np.asarray(tr["clip_frac"], float)
        taur = np.asarray(tr["tau_eff_ratio_min"], float) * 20.0   # ms
        t = np.arange(clip.size) * dt / 1000.0
        axC.plot(t, clip * 100, color=col, lw=0.9, label=f"c_E={cE:g}")
        axC2.plot(t[:taur.size], taur, color=col, lw=0.7, ls=":", alpha=0.6)
    axC.axhline(0.0, color="0.6", lw=0.6)
    axC2.axhline(2 * 0.1, color="crimson", lw=0.9, ls="--", alpha=0.7)
    axC.set_xlabel("time (s)"); axC.set_ylabel("cap-clip fraction (%)")
    axC2.set_ylabel("tau_eff_min (ms, dotted); 2·dt=red")
    axC.set_title("C · numerical safety (clip solid / tau_eff dotted)", fontsize=11, loc="left")
    axC.legend(fontsize=7, loc="upper left")

    verdict = summ.get("verdict", "?"); pick = summ.get("picked_c_E")
    fig.suptitle(f"MZ-FCXR Stage 0B workpoint (L=20 E1146 seed{summ.get('seed')}, T={summ.get('T'):.0f}ms)"
                 f"  —  verdict: {verdict}" + (f", picked c_E={pick}" if pick is not None else ""),
                 fontsize=12, y=1.02)
    out = os.path.join(FIGDIR, "stage0_workpoint.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
