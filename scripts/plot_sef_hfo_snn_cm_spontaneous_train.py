"""Multi-event electrode TRAIN view (user 2026-06-11): show the spontaneous event train over a LONG
window with the ∥ (along EE axis) and ⊥ (across) electrodes INTEGRATED into ONE panel, electrodes on
the y-axis, and electrodes that never participate DROPPED. Within each event the ∥ contacts peak in
SEQUENCE (slanted peak locus = direction); the ⊥ contacts peak TOGETHER (vertical = no direction);
the pattern REPEATS event after event = the spontaneous train. A dashed divider separates the two
shaft groups. Reads rep_<tag>.npz (full LFP trace) + readout_<tag>.json (event windows). NO re-sim.
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = os.path.join(OUT, "figures")
OFF = 1.35                                          # vertical offset between stacked traces


def _clean_events(events, sign):
    """Clean train events: self-terminated, readable, enough participating contacts, expected sign."""
    return [e for e in events
            if e.get("returned") and e.get("sign") == sign
            and e.get("axis_err") is not None and e.get("axis_err") < 25
            and e.get("n_part", 0) >= 7]


def _active_ordered(idxs, names, contacts, axis_unit, events):
    """Of the contacts in idxs, keep those that participate (non-None rank) in >=1 event, ordered
    along the shaft axis. Electrodes with no activity are dropped (user 2026-06-11)."""
    active_names = set()
    for e in events:
        ranks = e.get("ranks") or {}
        active_names.update(nm for nm, v in ranks.items() if v is not None)
    keep = [i for i in idxs if names[i] in active_names]
    if not keep:
        return []
    s = (contacts[keep] - contacts[keep].mean(0)) @ axis_unit
    return [keep[j] for j in np.argsort(s)]


def figure(tag, window_ms, all_events=False):
    z = np.load(os.path.join(OUT, "per_event", f"rep_{tag}.npz"), allow_pickle=True)
    d = json.load(open(os.path.join(OUT, f"readout_{tag}.json")))
    lfp = np.asarray(z["lfp"]).T                                # (n_contact, n_time)
    t = np.asarray(z["times"]); nc = int(z["nc"]); theta = float(z["theta"])
    names = [str(s) for s in z["names"]]; contacts = np.asarray(z["contacts"])
    sign = float(z["sign"]) if "sign" in z.files else 0.0
    events = d["events"] if all_events else _clean_events(d["events"], sign)
    win_lo, win_hi = 0.0, min(window_ms, float(t[-1]))
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])

    # active contacts per shaft, ordered along the shaft; ⊥ at the bottom, ∥ on top (one panel)
    perp_o = _active_ordered(list(range(nc, 2 * nc)), names, contacts, u_perp, events)
    par_o = _active_ordered(list(range(nc)), names, contacts, u_par, events)
    combined = perp_o + par_o                                   # bottom -> top
    nb = len(perp_o)                                            # boundary: ∥ group starts at index nb
    k = len(combined)
    if k == 0:
        print(f"  (skip {tag}: no active electrodes)"); return
    groups = [(0, nb, u_perp, "⊥ across axis\n(peaks align = no direction)", "winter"),
              (nb, k, u_par, "∥ along axis\n(peaks slant = direction)", "autumn")]

    sel = (t >= win_lo) & (t <= win_hi); ts = t[sel]
    sub = lfp[combined][:, sel]
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale

    fig, ax = plt.subplots(figsize=(15, max(4.5, 0.62 * k)))
    for g0, g1, _, _, cmap in groups:
        for i in range(g0, g1):
            col = plt.get_cmap(cmap)(0.15 + 0.6 * (i - g0) / max(g1 - g0 - 1, 1))
            ax.plot(ts, zt[i] + i * OFF, color=col, lw=0.9, alpha=0.95)
            ax.text(win_hi + 0.008 * (win_hi - win_lo), i * OFF, f" {names[combined[i]]}",
                    fontsize=8, va="center", color=col)
    if 0 < nb < k:
        ax.axhline((nb - 0.5) * OFF, color="0.45", ls="--", lw=1.1, zorder=1)
    for g0, g1, _, label, _ in groups:
        ax.text(win_lo - 0.008 * (win_hi - win_lo), (g0 + g1 - 1) / 2.0 * OFF, label,
                fontsize=9, va="center", ha="right", rotation=90, color="0.25")

    nev = 0
    for e in events:
        if e["t_off"] < win_lo or e["t_on"] > win_hi:
            continue
        nev += 1
        ax.axvspan(e["t_on"], e["t_off"], color="0.86", alpha=0.5, lw=0, zorder=0)
        ranks = e.get("ranks") or {}
        for g0, g1, *_ in groups:                              # locus per group (don't cross divider)
            pts = []
            for i in range(g0, g1):
                if ranks.get(names[combined[i]]) is None:
                    continue
                m = (ts >= e["t_on"]) & (ts <= e["t_off"])
                if m.sum() < 2:
                    continue
                pi = np.flatnonzero(m)[int(np.argmax(zt[i][m]))]
                pts.append((ts[pi], zt[i][pi] + i * OFF))
                ax.plot([ts[pi]], [zt[i][pi] + i * OFF], "o", ms=3.2, mfc="k", mec="white",
                        mew=0.4, zorder=5)
            if len(pts) >= 2:
                px, py = zip(*sorted(pts))
                ax.plot(px, py, "-", color="k", lw=1.0, alpha=0.75, zorder=4)

    ax.set_xlim(win_lo - 0.04 * (win_hi - win_lo), win_hi + 0.10 * (win_hi - win_lo))
    ax.set_yticks([]); ax.set_xlabel("time (ms)")
    dirword = "forward" if sign > 0 else ("reverse" if sign < 0 else "—")
    ax.set_title(
        f"SPONTANEOUS event train — {tag} ({dirword}; current-LFP)  |  {nev} events in view, "
        f"{len(events)} {'detected' if all_events else 'clean'}/{d['n_events']} total  |  active "
        f"electrodes only ({nb}⊥ + {k - nb}∥)\n∥ peaks slant = direction; ⊥ peaks align = no "
        f"direction; pattern repeats = the spontaneous train", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG, f"train_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}  ({nb}⊥ + {k - nb}∥ active electrodes, {nev} events)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["stage2_main_neg", "stage2_main_pos"])
    ap.add_argument("--window-ms", type=float, default=900.0,
                    help="window shows multiple events while keeping each event's sweep visible")
    ap.add_argument("--all-events", action="store_true",
                    help="show all detected events instead of the default clean expected-direction events")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for tag in a.tags:
        if os.path.exists(os.path.join(OUT, "per_event", f"rep_{tag}.npz")):
            figure(tag, a.window_ms, all_events=a.all_events)
        else:
            print(f"  (skip {tag}: no rep npz)")


if __name__ == "__main__":
    main()
