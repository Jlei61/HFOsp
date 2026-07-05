"""Aggregate the A2 axis-break sweep: did engaging the global inhibitory tank flip events from
axis-aligned to off-axis/global? Reads summary_*.json from the sweep dir, prints a table, and
draws two panels:
  A  biggest-event grad_align (x) vs isotropy (y), per cell — the axial cluster is top-left
     (align~1, isotropy<1); the target seizure zone is the shaded lower-right (axis readout dead).
  B  q_global_min (did the global tank actually drain?) vs biggest-event grad_align — tests the
     mechanism: does global disinhibition buy axis-break, or does the axial wiring win anyway?

Usage: python scripts/plot_a2_axisbreak_summary.py <sweep_dir>
"""
import json, sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/axisbreak"
cells = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/summary_*.json"))]
cells = [c for c in cells if c.get("biggest_event")]

# re-judge every cell from events_brief = single source of truth for the off-axis criterion
OFF_ISO, OFF_NFIRED = 0.50, 18000
for c in cells:
    eb = c.get("events_brief", [])
    off = [e for e in eb if e["isotropy"] >= OFF_ISO and e["n_fired"] >= OFF_NFIRED]
    c["_n_off"] = len(off)
    tonic = c["global_rate_hz"] > 60 or c["tonic_fraction"] > 0.5
    if not eb or max((e["n_fired"] for e in eb), default=0) < 1500:
        c["_regime"] = "quiet_or_tiny"
    elif off and tonic:
        c["_regime"] = "off_axis_TONIC"
    elif len(off) >= 2:
        c["_regime"] = "off_axis_SELF_LIMITING"
    elif len(off) == 1:
        c["_regime"] = "off_axis_oneshot"
    elif tonic:
        c["_regime"] = "tonic_axial"
    else:
        c["_regime"] = "axial_only"
print(f"{len(cells)} cells with events (off-axis = isotropy>={OFF_ISO} AND n_fired>={OFF_NFIRED})\n")
hdr = "tag                 mode      k   drv  gk   qGmin rate  tonicF regime               n_off | biggest: iso  nfired r95"
print(hdr)
for c in sorted(cells, key=lambda z: (z["mode"], z["drive"], z["k_use"], z["gk_max"])):
    b = c["biggest_event"]
    print("%-19s %-9s %.1f %.1f %.2f %.2f %5.1f %.3f  %-20s %d   |   %.2f %6d %4.0f"
          % (c["tag"], c["mode"], c["k_use"], c["drive"], c["gk_max"], c["q_global_min"],
             c["global_rate_hz"], c["tonic_fraction"], c["_regime"], c["_n_off"],
             b["isotropy"], b["n_fired"], b["r95_src"]))

target = [c for c in cells if c["_regime"] == "off_axis_SELF_LIMITING"]
broke = [c for c in cells if c["_n_off"] >= 1]
print(f"\nbroke the axis (>=1 off-axis global event): {len(broke)} -> {[c['tag'] for c in broke] or 'NONE'}")
print(f"TARGET off_axis_SELF_LIMITING: {len(target)} -> {[c['tag'] for c in target] or 'NONE'}")

fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.6))
mk = {"two_tank": "o", "core_only": "s", "per_core": "^"}
col = {"axial_only": "#1456c4", "off_axis_SELF_LIMITING": "#c0392b", "off_axis_oneshot": "#e67e22",
       "off_axis_TONIC": "#8e44ad", "tonic_axial": "#999", "quiet_or_tiny": "#bbb"}
# Panel A: the actual off-axis criterion plane (global n_fired x off-axis isotropy)
ax[0].axvspan(OFF_NFIRED, 33000, color="#fdeee6", zorder=0); ax[0].axhspan(OFF_ISO, 1.2, color="#fdeee6", zorder=0)
ax[0].axvline(OFF_NFIRED, color="#aaa", ls=":", lw=0.8); ax[0].axhline(OFF_ISO, color="#aaa", ls=":", lw=0.8)
ax[0].text(25500, 1.05, "off-axis GLOBAL\n(round + whole-field)", color="#c0392b", fontsize=8, ha="center")
for c in cells:
    b = c["biggest_event"]
    ax[0].scatter(b["n_fired"], b["isotropy"], marker=mk.get(c["mode"], "o"),
                  c=col.get(c["_regime"], "#333"), s=85, edgecolor="k", lw=0.5)
    ax[0].annotate(c["tag"], (b["n_fired"], b["isotropy"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax[0].set_xlabel("biggest event: # cells recruited (global; field=32000)")
ax[0].set_ylabel("biggest event: isotropy (round=off-axis; axial<0.4)")
ax[0].set_xlim(0, 33000); ax[0].set_ylim(0, 1.15)
ax[0].set_title("Did the biggest event break the corridor?\n(square=core_only control, circle=two_tank global tank)", fontsize=9.5)

# Panel B: does draining the global tank buy isotropy (off-axis shape)?
for c in cells:
    ax[1].scatter(c["q_global_min"], c["biggest_event"]["isotropy"], marker=mk.get(c["mode"], "o"),
                  c=col.get(c["_regime"], "#333"), s=85, edgecolor="k", lw=0.5)
ax[1].axhline(OFF_ISO, color="#aaa", ls=":", lw=0.8)
ax[1].set_xlabel("q_global_min  (1.0 = global tank never drained; low = global disinhibition engaged)")
ax[1].set_ylabel("biggest event isotropy (round = off-axis)")
ax[1].set_title("Does draining the global tank break the axis?\n(want: left = drained -> up = round)", fontsize=9.5)
import matplotlib.lines as ml
leg = [ml.Line2D([], [], marker="o", color="w", mfc=col[k], mec="k", label=k, ms=9) for k in col]
ax[1].legend(handles=leg, fontsize=7, loc="upper right")
fig.suptitle("A2 axis-break sweep — can the global inhibitory tank make a discrete off-axis self-limiting global event?",
             fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(f"{D}/figures", exist_ok=True)
fig.savefig(f"{D}/figures/axisbreak_summary.png", dpi=140)
print("\nwrote", f"{D}/figures/axisbreak_summary.png")
