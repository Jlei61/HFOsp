"""M4 dynamic-q_I representative figure (Topic 4, summary style per figure_style_guide §116).

Reads the 4-arm run (results/topic4_m4_dynamic/dynamic_qi_traces.npz) and tells the mechanism story in
4 independent panels (§7 discipline):
  A substrate   : the two small axial foci (source/sink cores) + E->E axis on the neuron sheet — WHERE.
  B q_I(t)      : the inhibitory resource draining 1.0 -> floor across the first event — the DEPLETION.
  C activity(t) : per-neuron population rate (Hz) + the runaway-detection level + event span — RUNAWAY,
                  and the pool's braking (no_pool vs pool peak).
  D S_G(t)      : the shared divisive pool building up — the M4 BRAKE's response.

Colours: no_pool = warm (crimson), pool = cool (blue); k_q 0.35 = solid, 0.18 = dashed (style_guide §0).
UNITS: res rate is a per-step spike COUNT; converted to per-neuron mean Hz = count/NE/dt*1e3. The runaway
detector (_first_sustained, pilot convention) fires on the smoothed COUNT >= 120 == 37.5 Hz/neuron; that
level is drawn as a reference line, NOT relabelled Hz silently.

Plotting-only. Output: results/paper-ready-figure/fig_m4_dynamic_qi/figures/.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/topic4_m4_dynamic/dynamic_qi_traces.npz"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"

NE, DT = 32000, 0.1
CORE_R = 1.5                                          # PP.CORE_R (source/sink core radius)
RUNAWAY_COUNT = 120.0                                # _first_sustained threshold (per-step count)
NO_POOL_C, POOL_C = "#c1272d", "#1f6fb2"
LS = {0.35: "-", 0.18: "--"}


def _hz(count):
    return np.asarray(count, float) / NE / DT * 1e3


def _arm(d, label):
    m = json.loads(str(d[f"{label}__meta"]))
    return dict(meta=m,
                qI=d[f"{label}__trace_qI_mean"], SG=d[f"{label}__trace_SG"],
                rate=d[f"{label}__rate"], events=d[f"{label}__events"])


def main():
    d = np.load(NPZ, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    L = float(d["L"]); src = d["src_xy"]; snk = d["snk_xy"]; posE = d["posE"]
    labels = ["kq0.35_no_pool", "kq0.35_pool_aG6", "kq0.18_no_pool", "kq0.18_pool_aG6"]
    arms = {l: _arm(d, l) for l in labels}
    t = np.arange(len(arms[labels[0]]["qI"])) * DT                     # ms
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.0))
    fig.suptitle("M4 dynamic $q_I$: two axial foci deplete inhibition to a one-shot runaway; "
                 "the shared divisive pool $S_G$ brakes it (E1146, L=20)", fontsize=12.5, y=0.98)

    # ---- A substrate: two foci + axis ----
    a0 = ax[0, 0]
    a0.scatter(posE[:, 0], posE[:, 1], s=0.4, c="#d9d9d9", rasterized=True, linewidths=0)
    for c, name, col in [(src, "source core", "#e8743b"), (snk, "sink core", "#7b5cb8")]:
        a0.add_patch(Circle((c[0], c[1]), CORE_R, fill=False, ec=col, lw=2.2, zorder=5))
        a0.scatter([c[0]], [c[1]], s=42, c=col, marker="*", zorder=6, edgecolors="k", linewidths=0.4)
    a0.plot([src[0], snk[0]], [src[1], snk[1]], color="#a65f00", lw=1.6, ls=":", zorder=4)
    a0.text(0.5 * (src[0] + snk[0]), src[1] + 0.9, "E$\\to$E axis", color="#a65f00", ha="center",
            va="bottom", fontsize=9)
    a0.set_xlim(0, L); a0.set_ylim(0, L); a0.set_aspect("equal")
    a0.set_title("A  substrate: two axial foci", fontsize=11, loc="left")
    a0.set_xlabel("mm"); a0.set_ylabel("mm")
    a0.legend(handles=[Line2D([], [], marker="*", color="#e8743b", ls="none", label="source core"),
                       Line2D([], [], marker="*", color="#7b5cb8", ls="none", label="sink core")],
              fontsize=8, loc="upper right", framealpha=0.9)

    # ---- B q_I depletion ----
    b = ax[0, 1]
    for l in labels:
        m = arms[l]["meta"]; col = POOL_C if m["use_SG"] else NO_POOL_C
        b.plot(t, arms[l]["qI"], color=col, ls=LS[m["k_q"]], lw=1.6)
    b.axhline(meta["qI"]["q_min"], color="0.4", lw=0.8, ls=":")
    b.text(t[-1], meta["qI"]["q_min"] + 0.02, "$q_{min}$", ha="right", va="bottom", fontsize=8, color="0.4")
    b.set_ylim(0, 1.05); b.set_xlim(0, t[-1])
    b.set_title("B  inhibitory resource $q_I(t)$ (sheet mean)", fontsize=11, loc="left")
    b.set_xlabel("time (ms)"); b.set_ylabel("$q_I$  (1 = intact, 0.05 = drained)")

    # ---- C activity + runaway ----
    c = ax[1, 0]
    for l in labels:
        m = arms[l]["meta"]; col = POOL_C if m["use_SG"] else NO_POOL_C
        c.plot(t, _hz(arms[l]["rate"]), color=col, ls=LS[m["k_q"]], lw=1.1, alpha=0.9)
    c.axhline(_hz(RUNAWAY_COUNT), color="0.35", lw=0.9, ls="--")
    c.text(t[-1], _hz(RUNAWAY_COUNT) + 3, "runaway level (37.5 Hz/neuron, 100 ms)", ha="right", va="bottom",
           fontsize=7.5, color="0.35")
    # the single detected event spans the whole sustained runaway; mark its onset (cleaner than shading)
    ev = arms["kq0.35_no_pool"]["events"]
    if ev.size:
        c.axvline(ev[0, 0], color="0.55", lw=0.8, ls=":")
        c.text(ev[0, 0] + 40, 150, "first event $\\to$\nsustained runaway", fontsize=7.5, color="0.4", va="top")
    c.set_xlim(0, t[-1])
    c.set_title("C  population activity + runaway onset", fontsize=11, loc="left")
    c.set_xlabel("time (ms)"); c.set_ylabel("per-neuron rate (Hz)")

    # ---- D S_G pool response ----
    e = ax[1, 1]
    for l in labels:
        m = arms[l]["meta"]
        if not m["use_SG"]:
            continue
        e.plot(t, arms[l]["SG"], color=POOL_C, ls=LS[m["k_q"]], lw=1.6)
    e.axhline(0.0, color="0.6", lw=0.8)
    e.text(0.02, 0.94, "no_pool arms: $S_G\\equiv0$ (pool off)", transform=e.transAxes, fontsize=8.5,
           color=NO_POOL_C, va="top")
    e.set_xlim(0, t[-1])
    e.set_title("D  shared divisive pool $S_G(t)$", fontsize=11, loc="left")
    e.set_xlabel("time (ms)"); e.set_ylabel("$S_G$  (divisive brake)")

    # shared legend (arm identity)
    handles = [Line2D([], [], color=NO_POOL_C, lw=2, label="no pool ($\\alpha_G$=0)"),
               Line2D([], [], color=POOL_C, lw=2, label="pool ($\\alpha_G$=6)"),
               Line2D([], [], color="0.3", lw=2, ls="-", label="$k_q$=0.35 (fast deplete)"),
               Line2D([], [], color="0.3", lw=2, ls="--", label="$k_q$=0.18 (slow deplete)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_png = OUT / "fig_m4_dynamic_qi.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig_m4_dynamic_qi.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    # quick numeric recap for the caller
    for l in labels:
        m = arms[l]["meta"]
        print(f"  {l:18} verdict={m['verdict']:16} runaway_ms={m['runaway_ms']} "
              f"peak={_hz(arms[l]['rate']).max():.0f}Hz S_G_max={m['S_G_max']}")


if __name__ == "__main__":
    main()
