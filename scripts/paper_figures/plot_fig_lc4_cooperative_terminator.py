#!/usr/bin/env python
"""Four questions about a load-activated outward current, one panel each.

This is a visual diagnostic of slow-variable dynamics, not the four-column mechanism/readout
standard: nothing here is a propagation event or an electrode readout, so that layout would have
nothing to put in its columns.  It follows the summary-figure precedent instead -- independent
questions side by side, statistics in the record rather than on the canvas.

* **does it stop** -- the discharge, read at the 100 ms scale the sustained-state criterion is
  stated at.  Two arms collapse inside 1.6 s; one never does.
* **does the wear clear** -- stopping is not recovering.  Every earlier arm that stopped landed on
  a smoulder that held the wear up, so the crossing of the level below which a frozen field no
  longer departs on its own is a separate question from the one above.
* **why one arm never won** -- the brake's own trajectory.  The two that terminated reached their
  peak within 1.2-1.7 s; the third was still climbing at 22.7 s while the discharge waxed and
  waned around it.  Burst-scale ripple is under 2% in all of them, so this is not the brake being
  released between bursts.
* **what it costs interictally** -- from the separate arms that start at rest, because the
  discharge arms cannot see it: they begin after entry has already happened.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability")
FORK = os.path.join(RES, "lc4_from_discharge")
REST = os.path.join(RES, "lc4_cooperative_2x2")
OUT = os.path.join(ROOT, "results/paper-ready-figure/fig_lc4_cooperative_terminator/figures")

FORK_OFFSET_MS = 12000.0     # the snapshot clock runs on the trajectory the state was taken from
DEPARTURE = 0.047            # below this a frozen wear field no longer departs on its own
CEIL = 0.03678               # top of the interictal recruited-fraction spread
REF_IED_HZ = 2.40            # interictal events per second with no mechanism
POST_SUPPRESS_S = (2.0, 5.0) # common window after the first successful suppression

ARMS = [("hill_slow", "cooperative, slow release", "#b2182b"),
        ("hill_fast", "cooperative, fast release", "#e8743b"),
        ("linear_fast", "linear, fast release", "#2166ac")]
GREY = "#9e9e9e"


def _smooth(x, w=100):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode="same")


def _wear(arm):
    z = np.load(os.path.join(FORK, f"arm_{arm}_traces.npz"))
    t, d = z["snapshot_t_ms"], z["snapshot_D_all"]
    # The loaded state resumes at 12 s.  An old template snapshot at t=0 is present in these
    # historical traces and is not part of the fork.  Key against the actual fork time rather
    # than merely dropping zero so future template snapshots cannot leak into the curve.
    keep = t >= FORK_OFFSET_MS
    return (t[keep] - FORK_OFFSET_MS) / 1000.0, d[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(17.0, 3.9), facecolor="white")

    # (a) does it stop -------------------------------------------------------------------------
    # The criterion itself, plotted: the share of each second the tissue spends above the
    # interictal spread, read at 100 ms.  Drawing the raw recruited fraction on a log axis instead
    # buries the answer under the silences between bursts.
    def _occ(af):
        hi = (_smooth(af) > CEIL).astype(float)
        n = len(hi) // 1000
        return np.arange(n) + 0.5, hi[:n * 1000].reshape(n, 1000).mean(axis=1)

    ref = np.load(os.path.join(RES, "global_burst_adaptation/arm_sensor_only_traces.npz"))["af"]
    ax[0].plot(*_occ(ref[12000:42000]), color=GREY, lw=1.6, label="no brake", zorder=1)
    for arm, lab, c in ARMS:
        ax[0].plot(*_occ(np.load(os.path.join(FORK, f"arm_{arm}_traces.npz"))["af"]),
                   color=c, lw=1.6, label=lab, zorder=2)
    ax[0].set_ylim(-0.03, 1.06); ax[0].set_xlim(0, 30)
    ax[0].set_xlabel("time from the start of braking (s)", fontsize=9)
    ax[0].set_ylabel("share of each second discharging", fontsize=9)
    ax[0].set_title("does the discharge stop", fontsize=10, fontweight="bold")

    # (b) does the wear clear ------------------------------------------------------------------
    missing_from = None
    for arm, lab, c in ARMS:
        t, d = _wear(arm)
        ax[1].plot(t, d, color=c, lw=1.5)
        end = json.load(open(os.path.join(FORK, f"arm_{arm}.json")))["wear_end"]
        missing_from = float(t[-1]) if missing_from is None else min(missing_from, float(t[-1]))
        # Do not connect the last recorded point to the final state: that line would look like a
        # measured trajectory.  The isolated marker is the only fact available after 18 s.
        ax[1].plot(30.0, end, "o", color=c, ms=5)
    if missing_from is not None and missing_from < 30.0:
        ax[1].axvspan(missing_from, 30.0, color="0.94", alpha=0.75, lw=0, zorder=0)
        ax[1].text((missing_from + 30.0) / 2.0, 5.4e-1,
                   "trajectory not recorded\n30 s markers = final state only",
                   fontsize=7.0, color="0.35", ha="center", va="top")
    ax[1].axhline(DEPARTURE, color="k", ls="--", lw=1.0)
    ax[1].text(0.4, DEPARTURE * 1.35,
               "lowest tested frozen-field departure level", fontsize=7.2)
    ax[1].set_yscale("log"); ax[1].set_ylim(8e-4, 0.7); ax[1].set_xlim(0, 30)
    ax[1].set_xlabel("time from the start of braking (s)", fontsize=9)
    ax[1].set_ylabel("mean E-cell wear  D", fontsize=9)
    ax[1].set_title("does the wear clear", fontsize=10, fontweight="bold")

    # (c) how the brake evolves ----------------------------------------------------------------
    hold_min = {}
    for arm, lab, c in ARMS:
        z = np.load(os.path.join(FORK, f"arm_{arm}_traces.npz"))
        cur = z["adap_current"]; dt = float(z["a_trace_dt_ms"][0])
        dt_s = dt / 1000.0
        t = np.arange(len(cur)) * dt_s
        ax[2].plot(t, cur, color=c, lw=1.3)
        i = int(np.argmax(cur))
        ax[2].plot(t[i], cur[i], "v", color=c, ms=6)
        # Placed in the empty upper band with leaders: the two winning arms peak 0.5 s apart, so
        # labels anchored to the markers overlap each other and the axis label.
        home = {"linear_fast": (7.0, 35.0), "hill_slow": (7.0, 30.5),
                "hill_fast": (19.5, 35.0)}[arm]
        ax[2].annotate(f"peaks at {t[i]:.1f} s", (t[i], cur[i]), xytext=home, fontsize=8,
                       color=c, ha="left", va="center",
                       arrowprops=dict(arrowstyle="-", color=c, lw=0.7, shrinkA=2, shrinkB=4))
        i0 = max(0, int(round(POST_SUPPRESS_S[0] / dt_s)))
        i1 = min(len(cur), int(round(POST_SUPPRESS_S[1] / dt_s)))
        hold_min[arm] = float(np.min(cur[i0:i1])) if i1 > i0 else float("nan")
    ax[2].axvspan(*POST_SUPPRESS_S, color="0.80", alpha=0.18, lw=0, zorder=0)
    ax[2].text(np.mean(POST_SUPPRESS_S), 37.1, "after first suppression",
               fontsize=7.0, color="0.35", ha="center", va="top")
    ax[2].text(8.0, 27.0,
               (f"2–5 s minimum:  Hill slow {hold_min['hill_slow']:.1f}  |  "
                f"Hill fast {hold_min['hill_fast']:.1f}"),
               fontsize=7.0, color="0.25", ha="left", va="center",
               bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.5))
    ax[2].set_xlim(0, 30); ax[2].set_ylim(0, 38)
    ax[2].set_xlabel("time from the start of braking (s)", fontsize=9)
    ax[2].set_ylabel("outward current delivered", fontsize=9)
    ax[2].set_title("how the brake evolves", fontsize=10, fontweight="bold")

    # (d) what it costs interictally -----------------------------------------------------------
    rest = {r["arm"]: r for r in json.load(open(os.path.join(REST, "cooperative_2x2.json")))["rows"]}
    order = [("hill_fast", "cooperative\nfast release", "#e8743b"),
             ("hill_slow", "cooperative\nslow release", "#b2182b"),
             ("linear_fast", "linear\nfast release", "#2166ac"),
             ("linear_slow", "linear\nslow load", "#7fb0d4")]
    rates = [rest[a]["n_returning_before_onset"] / 45.0 for a, _, _ in order]
    vals = [r / REF_IED_HZ * 100 for r in rates]
    ax[3].bar(range(4), rates, color=[c for _, _, c in order], width=0.66)
    for i, (rate, pct) in enumerate(zip(rates, vals)):
        ax[3].text(i, rate + 0.06, f"{rate:.2f}/s\n({pct:.0f}%)", ha="center", fontsize=8.0)
    ax[3].axhline(REF_IED_HZ, color="k", ls=":", lw=0.9)
    ax[3].text(3.45, REF_IED_HZ + 0.04, "no-brake pre-onset reference (5 s)",
               ha="right", fontsize=7.2)
    ax[3].set_xticks(range(4), [l for _, l, _ in order], fontsize=8)
    ax[3].set_ylim(0, 2.8); ax[3].set_ylabel("returning events per second", fontsize=9)
    ax[3].set_title("interictal cost (descriptive)", fontsize=10, fontweight="bold")
    ax[3].text(0.5, 0.97, "45 s arms vs 5 s reference",
               transform=ax[3].transAxes, fontsize=7.0, color="0.35",
               ha="center", va="top")

    for letter, a in zip("ABCD", ax):
        a.text(-0.13, 1.04, letter, transform=a.transAxes, fontsize=11,
               fontweight="bold", va="bottom", ha="left")
        a.tick_params(labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"lc4_cooperative_terminator.{ext}"),
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    meta = dict(
        figure="lc4_cooperative_terminator", kind="visual diagnostic, not a main claim figure",
        panels=["A: sustained-discharge occupancy", "B: mean E-cell wear and the tested departure level",
                "C: outward-current trajectory after the first suppression",
                "D: descriptive interictal event rate"],
        sources=dict(discharge_arms=FORK, rest_arms=REST,
                     no_brake_reference=os.path.join(RES, "global_burst_adaptation")),
        fork_note=("the discharge arms start 12 s into a real trajectory; the wear panel's solid "
                   "line ends at 18 s because the historical snapshot schedule was keyed from zero "
                   "while the loaded state resumed its own step counter.  The unshaded trajectory "
                   "is measured; the 30 s marker is an isolated final-state readout, with no line "
                   "drawn through the unrecorded interval"),
        interictal_reference_note=("panel D compares 45 s mechanism arms with a 5 s no-brake "
                                   "pre-onset reference.  The direction is descriptive; exact "
                                   "percentages are not a matched-window baseline gate"),
        departure_note=("D=0.047 is the lowest tested frozen-field departure level for the locked "
                        "spatial family, not a universal stability boundary for every D field"),
        claim_boundary=("one seed per arm.  The discharge arms do not test entry, and the rest "
                        "arms never entered, so no arm here shows a closed loop."))
    json.dump(meta, open(os.path.join(OUT, "lc4_cooperative_terminator_metadata.json"), "w"),
              indent=2)
    print(f"wrote {OUT}/lc4_cooperative_terminator.png")
    print("  interictal kept:", {a: f"{v:.0f}%" for (a, _, _), v in zip(order, vals)})


if __name__ == "__main__":
    main()
