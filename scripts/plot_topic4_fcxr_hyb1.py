"""FCXR-HYB1 diagnostics.  Only stages that actually produced data get a figure.

The screen / lifecycle / candidate figures named in the plan are NOT drawn: the sprint stopped at
the baseline-preservation gate, so those stages have no input and a placeholder would misrepresent
the state.  Panel discipline (CLAUDE.md 7): one independent question per panel.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.topic4_fcxr_hyb1 as H         # noqa: E402

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "hyb1_lifecycle")
FIG = os.path.join(OUT, "figures")
OK, BAD, MID, WARM = "#2a7f62", "#b5292f", "#4a6fa5", "#c98a20"


def fig_gate():
    zc = json.load(open(os.path.join(OUT, "z_axis_calibration.json")))
    rows = {s: json.load(open(os.path.join(OUT, f"baseline_preservation_seed{s}.json")))
            for s in (1, 3)}
    cur = np.load(os.path.join(OUT, "z_survival_curve.npz"))
    fig, ax = plt.subplots(2, 2, figsize=(13.4, 9.4))
    a = ax[0, 0]

    # 1 -- is the Z hazard axis identifiable?
    h = np.asarray(cur["a_p"]) / (H.TAU_Z_DOWN_MS / 1000.0)
    a.plot(cur["theta"], h, color=MID, lw=1.8, label=r"$a_p(\theta)\,/\,\tau_{Z,down}$  (probe)")
    for k, v in zc["levels"].items():
        a.plot(v["I_th_EI"], v["h_Z_realised"], "o", color=BAD, ms=7)
        a.annotate(f"{k} {v['I_th_EI']:.1f}", (v["I_th_EI"], v["h_Z_realised"]),
                   textcoords="offset points", xytext=(8, 10 + 11 * list(zc["levels"]).index(k)),
                   fontsize=7, color=BAD,
                   arrowprops=dict(arrowstyle="-", color=BAD, lw=0.6))
    for k, p in zc["anchor_prediction"].items():
        a.axhline(p["h_Z_obs"], color=WARM, ls="--", lw=1.2)
        a.annotate(f"{k} observed ({p['rel_err']:+.0%})", (cur["theta"].min(), p["h_Z_obs"]),
                   ha="left", va="bottom", fontsize=7, color=WARM)
    a.set_xscale("log"); a.set_ylim(0, 0.06)
    a.set_xlabel(r"GABA-sensor threshold  $I_{th,EI}$")
    a.set_ylabel(r"pre-onset hazard  $h_Z$  (s$^{-1}$)")
    a.legend(fontsize=7.5, loc="upper right")
    a.set_title("1  Z hazard axis — LOCKED\n"
                "one 3 s probe re-predicts both same-seed anchors to +14% / +9%;\n"
                "H_LO sits ON q75, so only H_MID and H_HI are genuinely interior", fontsize=8.6)

    # 2 -- does the excess-K layer disturb the interictal potassium field?
    a = ax[0, 1]
    lbl, val, col = [], [], []
    for s in (1, 3):
        for on in (False, True):
            r = next(x for x in rows[s]["rows"] if x["dk_on"] == on)
            lbl.append(f"seed{s}\n{'feedback on' if on else 'sensor only'}")
            val.append(r["dk"]["frac_over"])
            col.append(BAD if on else MID)
    a.bar(range(4), val, color=col, alpha=0.9)
    a.axhline(1.0 - H.Q_BG, color=OK, ls="-.", lw=1.6,
              label=f"gate  {1.0 - H.Q_BG:g}   (= q99$_{{t,v}}$($\\delta$K) $\\leq$ 0.05 mM)")
    for i, v in enumerate(val):
        a.text(i, v * 1.03, f"{v:.3f}", ha="center", fontsize=8)
    a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels(lbl, fontsize=7.5)
    a.set_ylabel(r"$P_{t,v}(\delta K > 0.05\ \mathrm{mM})$  over occupied voxels")
    a.legend(fontsize=7.5, loc="lower right")
    a.set_title("2  interictal potassium — GATE FAILED, 30-70x over\n"
                "the excess field is already large with the feedback OFF, so this is not\n"
                "caused by the loop closing", fontsize=8.6)

    # 3 -- does it disturb the interictal event train?
    a = ax[1, 0]
    w, xs = 0.34, np.arange(2)
    for j, (key, nm, sc) in enumerate((("event_rate_hz", "event rate (Hz)", 1.0),
                                       ("iei_cv", "IEI CV", 1.0))):
        for k, on in enumerate((False, True)):
            v = [next(x for x in rows[s]["rows"] if x["dk_on"] == on)[key] * sc for s in (1, 3)]
            a.bar(j + (k - 0.5) * w, np.mean(v), w, color=(BAD if on else MID), alpha=0.9)
            a.plot([j + (k - 0.5) * w] * 2, v, "k.", ms=7)
    a.axhline(H.IEI_CV_MIN, color=OK, ls="-.", lw=1.4)
    a.text(1.42, H.IEI_CV_MIN, " CV floor 0.5", color=OK, fontsize=7.5, va="bottom", ha="right")
    a.set_xticks(xs); a.set_xticklabels(["returning event rate (Hz)", "IEI CV"], fontsize=8.5)
    a.set_ylabel("value  (bars = mean of the two seeds, dots = seeds)")
    a.set_title("3  interictal event train — rate falls ~40%, irregularity nearly doubles\n"
                "blue = sensor only, red = feedback on.  The rate stays inside the accepted band\n"
                "but the train is no longer the same statistical neighbourhood", fontsize=8.6)

    # 4 -- what does the disturbance look like in time?
    a = ax[1, 1]
    for s, ls in ((1, "-"), (3, "--")):
        for on, c in ((False, MID), (True, BAD)):
            r = next(x for x in rows[s]["rows"] if x["dk_on"] == on)
            y = np.asarray(r["dk"]["max_series"], float)
            t = np.arange(y.size) * (0.5 * 2) / 1000.0        # dt_ion 0.5 ms, trace_stride 2
            a.plot(t, y, ls, color=c, lw=1.2, alpha=0.85,
                   label=f"seed{s} {'on' if on else 'off'}")
    a.axhline(0.05, color=OK, ls="-.", lw=1.4)
    a.text(0.05, 0.055, "0.05 mM amplitude clause", color=OK, fontsize=7.5)
    a.set_xlabel("time (s)"); a.set_ylabel(r"$\max_v \delta K$  (mM)")
    a.legend(fontsize=7.2, ncol=2, loc="upper right")
    a.set_title("4  the field RATCHETS: clearance between events is incomplete\n"
                r"interictal events arrive every ~0.4-0.6 s while $\tau_K$ = 0.65 s, so each event"
                "\nadds a step the next inter-event gap cannot undo -- it never returns to zero",
                fontsize=8.6)

    fig.suptitle("FCXR-HYB1 — Z axis LOCKED, baseline preservation FAILED on both seeds. "
                 "The 12-cell screen and the lifecycle gates were never reached.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "hyb1_axis_and_baseline_gate.png"), dpi=175)
    plt.close(fig)
    print("[plot] hyb1_axis_and_baseline_gate.png")


if __name__ == "__main__":
    fig_gate()
