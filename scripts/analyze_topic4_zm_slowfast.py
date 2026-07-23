#!/usr/bin/env python
"""Section-8 slow-fast analysis of a Z/M(+S_G) arm (task §8). Reads {arm}_seed{seed}.npz from the
carrier-gate runner. Treats z/m/S_G as slow parameters and characterizes the fast burst subsystem:

  - burst detection on the fine core rate -> cycle-to-cycle IBI + amplitude drift -> slowfast_verdict
    (candidate_inner_cycle / transient_burst_train / not_oscillatory; NEVER 'limit_cycle').
  - the S_G relaxation coupling: cross-correlation lag between core rate and S_G (does each burst drive
    an S_G surge that then collapses, re-permitting the next burst?).
  - the slow z_core drift under the fast cycle.

Writes a JSON summary + a 3-panel figure. This is the natural-trajectory ceiling: a limit-cycle CLAIM
would need a frozen-slow repeated trajectory, which this does not provide.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_slowfast import detect_bursts, cycle_stats, slowfast_verdict  # noqa: E402

DT_STEP_MS = 0.1


def _resample_to_bins(trace_step, n_bins, bin_ms):
    """Downsample a per-step (0.1 ms) trace to `n_bins` at `bin_ms` by bin-mean."""
    x = np.asarray(trace_step, float)
    if x.size == 0:
        return np.zeros(n_bins)
    bs = max(1, int(round(bin_ms / DT_STEP_MS)))
    out = np.array([x[i * bs:(i + 1) * bs].mean() if i * bs < x.size else x[-1] for i in range(n_bins)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", default="sg")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = np.load(os.path.join(a.dir, f"{a.arm}_seed{a.seed}.npz"), allow_pickle=True)
    core = np.asarray(d["core_rate"], float)
    bin_ms = float(d["rate_bin_ms"])
    n = core.size
    SG = _resample_to_bins(d["SG"], n, bin_ms) if np.asarray(d["SG"]).size else np.zeros(n)
    zc = _resample_to_bins(d["z_core"], n, bin_ms) if np.asarray(d["z_core"]).size else np.zeros(n)
    t = np.arange(n) * bin_ms

    pk, amp = detect_bursts(core, bin_ms)
    cs = cycle_stats(pk, amp, bin_ms)
    verdict = slowfast_verdict(cs)

    # ---- S_G relaxation coupling: xcorr(core, SG) lag (SG builds AFTER activity => SG lags core) ----
    lag_ms = float("nan"); xcorr_peak = float("nan")
    if pk.size >= 3 and SG.std() > 0 and core.std() > 0:
        c = (core - core.mean()) / (core.std() + 1e-12)
        s = (SG - SG.mean()) / (SG.std() + 1e-12)
        xc = np.correlate(s, c, mode="full")            # positive lag => SG follows core
        lags = (np.arange(xc.size) - (n - 1)) * bin_ms
        w = np.abs(lags) <= 500.0
        k = int(np.argmax(xc[w]))
        lag_ms = float(lags[w][k]); xcorr_peak = float(xc[w][k] / n)

    summary = dict(arm=a.arm, seed=a.seed, slowfast_verdict=verdict,
                   n_bursts=int(cs["n_bursts"]),
                   ibi_median_ms=float(np.median(cs["ibi_ms"])) if cs["ibi_ms"].size else None,
                   ibi_cv_tail=cs["ibi_cv_tail"], amp_cv_tail=cs["amp_cv_tail"],
                   ibi_drift_frac=cs["ibi_drift_frac"], amp_drift_frac=cs["amp_drift_frac"],
                   ibi_slope_ms_per_cycle=cs["ibi_slope"], amp_slope_per_cycle=cs["amp_slope"],
                   sg_core_xcorr_lag_ms=lag_ms, sg_core_xcorr_peak=xcorr_peak,
                   z_core_start=float(zc[:max(1, n // 50)].mean()), z_core_end=float(zc[-max(1, n // 50):].mean()),
                   note="natural-trajectory analysis; a limit-cycle claim needs a frozen-slow repeated "
                        "trajectory which this does not provide")
    with open(os.path.join(a.dir, f"slowfast_{a.arm}_seed{a.seed}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if o != o else o)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    # (1) zoomed core rate + bursts + S_G coupling
    if pk.size:
        c0 = int(pk[len(pk) // 3]); w0 = max(0, c0 - int(1500 / bin_ms)); w1 = min(n, c0 + int(1500 / bin_ms))
    else:
        w0, w1 = 0, min(n, int(3000 / bin_ms))
    ax[0].plot(t[w0:w1], core[w0:w1], lw=0.8, color="#d62728", label="core rate")
    inb = pk[(pk >= w0) & (pk < w1)]
    ax[0].plot(t[inb], core[inb], "kv", ms=4, label="burst")
    a2 = ax[0].twinx(); a2.plot(t[w0:w1], SG[w0:w1], lw=1.0, color="#1f77b4", label="S_G")
    a2.set_ylabel("S_G", color="#1f77b4")
    ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("core rate (Hz)")
    ax[0].set_title(f"(1) burst ↔ S_G relaxation coupling\nxcorr lag(S_G−core)={lag_ms:.0f}ms", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")
    # (2) cycle-to-cycle drift
    if cs["ibi_ms"].size:
        ax[1].plot(np.arange(cs["ibi_ms"].size), cs["ibi_ms"], "o-", ms=3, color="#9467bd", label="IBI (ms)")
        a3 = ax[1].twinx(); a3.plot(np.arange(amp.size), amp, "s-", ms=3, color="#ff7f0e", label="amp (Hz)")
        a3.set_ylabel("burst amp (Hz)", color="#ff7f0e")
    ax[1].set_xlabel("cycle index"); ax[1].set_ylabel("IBI (ms)", color="#9467bd")
    ax[1].set_title(f"(2) cycle-to-cycle drift\nIBI cv_tail={cs['ibi_cv_tail']:.2f} "
                    f"drift={cs['ibi_drift_frac']:.2f}", fontsize=9)
    # (3) slow z_core drift + burst rate
    ax[2].plot(t, zc, lw=1.0, color="#ff7f0e", label="z core (slow)")
    ax[2].set_xlabel("t (ms)"); ax[2].set_ylabel("z core", color="#ff7f0e"); ax[2].set_ylim(0, 1.02)
    a4 = ax[2].twinx()
    if pk.size >= 2:
        a4.plot(t[pk[1:]], 1000.0 / np.maximum(np.diff(pk) * bin_ms, 1e-9), "k.", ms=4)
        a4.set_ylabel("inst. burst rate (Hz)")
    ax[2].set_title("(3) slow z_core under the fast cycle", fontsize=9)

    fig.suptitle(f"Z/M {a.arm} seed{a.seed} slow-fast: {verdict}  "
                 f"(n_bursts={cs['n_bursts']}, IBI≈{summary['ibi_median_ms']}ms)  "
                 f"— natural trajectory, NOT a proven limit cycle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(a.dir, "figures", f"slowfast_{a.arm}_seed{a.seed}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130); fig.savefig(out.replace(".png", ".pdf"))
    print(f"verdict={verdict} n_bursts={cs['n_bursts']} ibi_med={summary['ibi_median_ms']} "
          f"ibi_drift={cs['ibi_drift_frac']:.2f} amp_drift={cs['amp_drift_frac']:.2f} lag={lag_ms:.0f}ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
