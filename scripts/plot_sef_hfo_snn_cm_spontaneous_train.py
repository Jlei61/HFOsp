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


def _active_combined(names, contacts, u_par, u_perp, events):
    """All contacts (BOTH electrodes — treated as one set of observations, user 2026-06-11) that
    participate (non-None rank) in >=1 event, ordered by position along the propagation axis (u_par,
    tiebreak u_perp). Electrodes with no activity are dropped. Returns (indices, is_perp flags)."""
    active_names = set()
    for e in events:
        active_names.update(nm for nm, v in (e.get("ranks") or {}).items() if v is not None)
    keep = [i for i in range(len(names)) if names[i] in active_names]
    if not keep:
        return [], []
    pp = np.array([contacts[i] @ u_par for i in keep])
    qq = np.array([contacts[i] @ u_perp for i in keep])
    order = np.lexsort((qq, pp))                                # primary u_par, secondary u_perp
    combined = [keep[j] for j in order]
    is_perp = [names[i].startswith("B") for i in combined]      # ⊥ shaft = the B-named contacts
    return combined, is_perp


def figure(tag, window_ms, all_events=False):
    z = np.load(os.path.join(OUT, "per_event", f"rep_{tag}.npz"), allow_pickle=True)
    d = json.load(open(os.path.join(OUT, f"readout_{tag}.json")))
    lfp = np.asarray(z["lfp"]).T                                # (n_contact, n_time)
    t = np.asarray(z["times"]); theta = float(z["theta"])
    names = [str(s) for s in z["names"]]; contacts = np.asarray(z["contacts"])
    sign = float(z["sign"]) if "sign" in z.files else 0.0
    events = d["events"] if all_events else _clean_events(d["events"], sign)
    win_lo, win_hi = 0.0, min(window_ms, float(t[-1]))
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])

    combined, is_perp = _active_combined(names, contacts, u_par, u_perp, events)
    k = len(combined)
    if k == 0:
        print(f"  (skip {tag}: no active electrodes)"); return
    n_perp = sum(is_perp)

    sel = (t >= win_lo) & (t <= win_hi); ts = t[sel]
    sub = lfp[combined][:, sel]
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale

    fig, ax = plt.subplots(figsize=(15, max(4.5, 0.62 * k)))
    for i in range(k):
        # colour by shaft (∥ along-axis = orange, ⊥ across = teal) so identity stays visible even
        # though both electrodes are stacked together as one observation set
        col = ("#1f9e9e" if is_perp[i] else "#e8743b")
        ax.plot(ts, zt[i] + i * OFF, color=col, lw=0.9, alpha=0.95)
        ax.text(win_hi + 0.008 * (win_hi - win_lo), i * OFF, f" {names[combined[i]]}",
                fontsize=8, va="center", color=col)

    nev = 0
    for e in events:
        if e["t_off"] < win_lo or e["t_on"] > win_hi:
            continue
        nev += 1
        ax.axvspan(e["t_on"], e["t_off"], color="0.86", alpha=0.5, lw=0, zorder=0)
        ranks = e.get("ranks") or {}
        pts = []                                               # ONE locus across BOTH electrodes
        for i in range(k):
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
            px, py = zip(*sorted(pts))                          # sorted by peak time
            ax.plot(px, py, "-", color="k", lw=1.0, alpha=0.75, zorder=4)

    ax.set_xlim(win_lo - 0.03 * (win_hi - win_lo), win_hi + 0.10 * (win_hi - win_lo))
    ax.set_yticks([]); ax.set_xlabel("time (ms)")
    ax.set_ylabel("electrodes ordered along propagation axis →", fontsize=9)
    dirword = "forward" if sign > 0 else ("reverse" if sign < 0 else "—")
    ax.set_title(
        f"SPONTANEOUS event train — {tag} ({dirword}; current-LFP)  |  {nev} events in view, "
        f"{len(events)} {'detected' if all_events else 'clean'}/{d['n_events']} total  |  active "
        f"electrodes only ({k - n_perp} ∥ orange + {n_perp} ⊥ teal, both treated as observations)\n"
        f"one peak locus per event across BOTH electrodes (sorted by time); slants = recruitment "
        f"order along the axis; repeats = the spontaneous train", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG, f"train_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}  ({k - n_perp}∥ + {n_perp}⊥ active electrodes, {nev} events)")


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
