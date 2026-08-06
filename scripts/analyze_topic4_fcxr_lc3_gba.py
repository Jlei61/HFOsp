#!/usr/bin/env python
"""Read the brake arms, and say which way each one failed if it failed.

The outcome alone separates only two of the possibilities.  An arm that returns to the
reference event distribution has closed the loop; an arm that goes silent has over-suppressed.
An arm that still smoulders could be any of three different things, and they call for three
different next steps:

* **never charged** -- the sensor did not reach the gate often enough, so the gate or the
  window is wrong and the strength is irrelevant;
* **let go too early** -- it charged, then fell away while the wear was still above the level
  that departs, which is the failure the existing relay already has;
* **held but too weak** -- it stayed engaged and the tissue smouldered through it, so the
  strength is what is wrong.

So the brake's own time course is read alongside the outcome.  The sensor is also
reconstructed offline from the active-fraction series and compared against what the engine
recorded: they are the same quantity computed two ways, and a disagreement would mean the
gate was set in different units from the thing it gates.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import run_topic4_fcxr_lc3 as E01  # noqa: E402
from mz_slow_vars import GBA_SENSE_BIN_MS  # noqa: E402
from src.topic4_fcxr_lc3_stage import reference_band  # noqa: E402

BASE = os.path.join(E01.OUT, "global_burst_adaptation")
FIGS = os.path.join(E01.OUT, "figures")
# The wear level below which the quiet state stops departing within the watch: the lowest
# frozen field that still departed sat at 0.0473 and took 7.0 s, and healthy wear never left.
WEAR_SAFE = 0.0473
ARM_COLOR = {"sensor_only": "#111111", "act_g006": "#1b4f9c",
             "act_g015": "#e07b39", "act_g039": "#c0392b"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def _offline_sensor(af, bin_ms, tau_sense):
    """The engine's burst sensor, recomputed from the series it reads."""
    out = np.empty_like(af, dtype=float)
    s, decay, gain = 0.0, np.exp(-bin_ms / tau_sense), GBA_SENSE_BIN_MS / tau_sense
    for i, f in enumerate(af):
        s = s * decay + f * gain
        out[i] = s
    return out


def _how_it_failed(rec, z):
    """Which of the three smouldering failures this is, from the brake's own trace."""
    a = np.asarray(z["gba_a"], float)
    if a.size == 0 or float(a.max()) <= 0.0:
        return "never charged", "the sensor never reached the gate"
    peak = float(a.max())
    # "Engaged" means at least half the peak; the question is whether it was still engaged
    # when the wear was still in the range that departs.
    dt = float(z["gba_trace_dt_ms"][0])
    engaged = a >= 0.5 * peak
    last_engaged_ms = float(np.nonzero(engaged)[0][-1] * dt)
    end_ms = float(a.size * dt)
    still_engaged = last_engaged_ms >= end_ms - 2.0 * dt
    wear_end = float(rec["wear_end"])
    if wear_end <= WEAR_SAFE:
        return "cleared the wear", f"wear ended at {wear_end:.4f}, inside the range that holds"
    if still_engaged:
        return "held but too weak", (f"still engaged at the end and wear stayed at "
                                     f"{wear_end:.4f}, above the {WEAR_SAFE:.4f} that departs")
    return "let go too early", (f"released at {last_engaged_ms/1000:.1f} s of "
                                f"{end_ms/1000:.0f} s with wear still at {wear_end:.4f}")


def _terminated(rec):
    """A bout whose end IS the end of the record has not terminated.

    The lifecycle detector reports `offset` as the window after the last ictal one, so a
    discharge that simply runs out of recording gets an offset equal to the run length. Reading
    that as a termination turns every non-terminating arm into a terminated one whose tail
    happens to be zero seconds long -- which is exactly how the control arm, known never to
    terminate, was first reported as "terminated and silenced".
    """
    if rec["offset_ms"] is None:
        return False
    return float(rec["offset_ms"]) < float(rec["run_ms"]) - 1e-9


def _verdict(rec, z):
    if rec["onset_ms"] is None:
        return "blocked entry", "the tissue never entered, so nothing downstream was tested"
    if not _terminated(rec):
        reasons = "; ".join(rec.get("lifecycle", {}).get("reasons", [])) or "no offset in the window"
        return "never stopped", f"entered and did not terminate inside the window ({reasons})"
    chk = rec.get("return_check") or {}
    if chk.get("returned"):
        return "came back", "returning events inside the frozen reference distribution"
    if rec["n_returning_after_offset"] == 0:
        return "silenced", "terminated, and nothing came back at all"
    how, why = _how_it_failed(rec, z)
    return f"still smouldering ({how})", why


def main():
    rows = []
    band = reference_band(json.load(open(E01.ARTIFACTS["lc1_baseline"])))
    for path in sorted(glob.glob(os.path.join(BASE, "arm_*.json"))):
        rec = json.load(open(path))
        if rec.get("status") != "COMPLETE":
            continue
        npz = path.replace(".json", "_traces.npz")
        if not os.path.isfile(npz):
            print(f"  {rec['arm']}: no trace on disk; outcome only")
            continue
        z = np.load(npz)
        verdict, why = _verdict(rec, z)
        off = _offline_sensor(np.asarray(z["af"], float), float(z["af_bin_ms"][0]),
                              5.0)
        rows.append(dict(rec=rec, z=z, verdict=verdict, why=why,
                         offline_sensor_max=float(off.max()),
                         engine_sensor_max=float(np.max(z["gba_burst"]))))

    if not rows:
        raise SystemExit("no completed arms with traces yet")

    print(f"{'arm':<14}{'eta':>6}{'onset':>7}{'offset':>8}{'back':>6}{'rate':>8}"
          f"{'wear':>8}{'brake max':>11}  verdict")
    for r in rows:
        rec = r["rec"]
        print(f"{rec['arm']:<14}{rec['eta_gba']:>6.0f}"
              f"{(str(int(rec['onset_ms']/1000))+'s' if rec['onset_ms'] else '--'):>7}"
              f"{(str(int(rec['offset_ms']/1000))+'s' if rec['offset_ms'] else '--'):>8}"
              f"{rec['n_returning_after_offset']:>6}{rec['tail_event_rate_hz']:>8.3f}"
              f"{rec['wear_end']:>8.4f}"
              f"{(rec['gba_a_max'] if rec['gba_a_max'] is not None else float('nan')):>11.4f}"
              f"  {r['verdict']}")
        print(f"{'':<14}{r['why']}")
    print(f"\n  reference band: {band['event_rate_lo']:.3f}-{band['event_rate_hi']:.2f} events/s")
    print("  sensor cross-check (engine against the same quantity recomputed from its input):")
    for r in rows:
        d = abs(r["engine_sensor_max"] - r["offline_sensor_max"])
        rel = d / max(r["offline_sensor_max"], 1e-12)
        print(f"    {r['rec']['arm']:<14} engine {r['engine_sensor_max']:.4f} vs "
              f"offline {r['offline_sensor_max']:.4f}  ({rel*100:.1f}% apart)")

    os.makedirs(FIGS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for r in rows:
        rec, z = r["rec"], r["z"]
        c = ARM_COLOR.get(rec["arm"], "#7f8c8d")
        lab = (f"{rec['arm']} (off)" if rec["eta_gba"] == 0
               else f"{rec['arm']} ({rec['eta_gba']/18:.2f} of leak)")
        t = np.arange(z["gba_a"].size) * float(z["gba_trace_dt_ms"][0]) / 1000.0
        ax1.plot(t, z["gba_a"], lw=1.3, color=c, label=lab)
        ts = np.asarray(z["snapshot_t_ms"], float) / 1000.0
        ax2.plot(ts, z["snapshot_D_all"], lw=1.3, color=c, label=lab)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("brake")
    ax1.set_title("does the brake charge, and does it hold", loc="left")
    ax1.legend(frameon=False, fontsize=7.5)
    ax2.axhline(WEAR_SAFE, color="#8fa3b8", lw=0.9, ls=(0, (4, 3)))
    ax2.text(ax2.get_xlim()[1], WEAR_SAFE, " below here the quiet\n state stops departing",
             fontsize=7.2, color="#5d6d7e", va="center")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("wear, whole array")
    ax2.set_title("and does the wear get below where it can hold", loc="left")
    fig.tight_layout()
    out = os.path.join(FIGS, "lc3_D_brake_and_wear.png")
    fig.savefig(out)
    plt.close(fig)

    payload = dict(schema="fcxr-lc3-gba-adjudication-1.0",
                   wear_safe_level=WEAR_SAFE,
                   reference_band=dict(lo=band["event_rate_lo"], hi=band["event_rate_hi"]),
                   rows=[dict(arm=r["rec"]["arm"], eta_gba=r["rec"]["eta_gba"],
                              verdict=r["verdict"], why=r["why"],
                              stage=r["rec"]["stage"],
                              onset_ms=r["rec"]["onset_ms"], offset_ms=r["rec"]["offset_ms"],
                              tail_event_rate_hz=r["rec"]["tail_event_rate_hz"],
                              wear_end=r["rec"]["wear_end"],
                              gba_a_max=r["rec"]["gba_a_max"],
                              engine_sensor_max=r["engine_sensor_max"],
                              offline_sensor_max=r["offline_sensor_max"]) for r in rows])
    with open(os.path.join(BASE, "adjudication.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"\n  figure: {out}")
    print(f"  written: {os.path.join(BASE, 'adjudication.json')}")


if __name__ == "__main__":
    main()
