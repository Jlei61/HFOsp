"""Multi-event electrode TRAIN view (user 2026-06-11): instead of zooming the read-out into ONE
event, show the TWO electrodes (∥ along EE axis, ⊥ across) over a LONG window with the whole train
of spontaneous events. Within each event the ∥ contacts peak in SEQUENCE (diagonal = direction);
the ⊥ contacts peak TOGETHER (vertical = no direction); and the pattern REPEATS event after event
= the spontaneous train. Reads rep_<tag>.npz (full LFP trace) + readout_<tag>.json (all event
windows). NO re-sim.
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = os.path.join(OUT, "figures")


def _panel(ax, lfp, names, t, win_lo, win_hi, events, axis_unit, contacts, title):
    """Stacked per-contact LFP traces over [win_lo, win_hi], contacts ordered along the shaft,
    event windows shaded, per-contact peak within each event marked (+ connected per event)."""
    sel = (t >= win_lo) & (t <= win_hi)
    ts = t[sel]
    s = (np.asarray(contacts, float) - np.asarray(contacts, float).mean(0)) @ axis_unit
    order = np.argsort(s)
    lfp = lfp[order][:, sel]; names = [names[i] for i in order]
    k = lfp.shape[0]
    base = np.median(lfp, axis=1, keepdims=True)
    scale = np.maximum(lfp.max(axis=1, keepdims=True) - base, 1e-9)
    z = (lfp - base) / scale
    off = 1.3
    for i in range(k):
        col = plt.get_cmap("viridis")(0.1 + 0.8 * i / max(k - 1, 1))
        ax.plot(ts, z[i] + i * off, color=col, lw=0.8, alpha=0.9)
        ax.text(win_hi + 0.01 * (win_hi - win_lo), i * off, f" {names[i]}", fontsize=7,
                va="center", color=col)
    nev = 0
    for e in events:
        if e["t_off"] < win_lo or e["t_on"] > win_hi:
            continue
        nev += 1
        ax.axvspan(e["t_on"], e["t_off"], color="0.86", alpha=0.5, lw=0, zorder=0)
        pts = []
        for i in range(k):
            # Only mark contacts that actually participated in THIS event. Otherwise
            # noisy non-participants can create a fake peak locus, especially on the
            # perpendicular shaft.
            ranks = e.get("ranks") or {}
            if ranks.get(names[i]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(z[i][m]))]
            pts.append((ts[pi], z[i][pi] + i * off))
            ax.plot([ts[pi]], [z[i][pi] + i * off], "o", ms=3.0, mfc="k", mec="white", mew=0.4, zorder=5)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts))
            ax.plot(px, py, "-", color="k", lw=0.9, alpha=0.7, zorder=4)   # peak locus per event
    ax.set_xlim(win_lo, win_hi + 0.10 * (win_hi - win_lo))
    ax.set_yticks([]); ax.set_title(f"{title}  ({nev} events in view)", fontsize=10)


def _clean_events(events, sign):
    """Clean train events: self-terminated, readable, enough participating contacts, expected sign."""
    return [e for e in events
            if e.get("returned")
            and e.get("sign") == sign
            and e.get("axis_err") is not None and e.get("axis_err") < 25
            and e.get("n_part", 0) >= 7]


def figure(tag, window_ms, all_events=False):
    z = np.load(os.path.join(OUT, "per_event", f"rep_{tag}.npz"), allow_pickle=True)
    d = json.load(open(os.path.join(OUT, f"readout_{tag}.json")))
    lfp = np.asarray(z["lfp"]).T                  # (n_contact, n_time)
    t = np.asarray(z["times"]); nc = int(z["nc"]); theta = float(z["theta"])
    names = [str(s) for s in z["names"]]; contacts = np.asarray(z["contacts"])
    sign = float(z["sign"]) if "sign" in z.files else 0.0
    events = d["events"] if all_events else _clean_events(d["events"], sign)
    win_lo, win_hi = 0.0, min(window_ms, float(t[-1]))
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2), sharex=True)
    _panel(axes[0], lfp[:nc], names[:nc], t, win_lo, win_hi, events, u_par, contacts[:nc],
           "∥ electrode (along EE axis) — peaks SWEEP within each event = direction")
    _panel(axes[1], lfp[nc:2 * nc], names[nc:2 * nc], t, win_lo, win_hi, events, u_perp, contacts[nc:2 * nc],
           "⊥ electrode (across EE axis) — peaks ALIGNED = no direction")
    axes[1].set_xlabel("time (ms)")
    dirword = "forward" if sign > 0 else ("reverse" if sign < 0 else "—")
    fig.suptitle(
        f"SPONTANEOUS event train — {tag} ({dirword}; current-LFP)\n"
        f"grey blocks = self-ignited events; top ∥ peaks slant, bottom ⊥ peaks align; "
        f"showing {len(events)} {'detected' if all_events else 'clean'} / {d['n_events']} events",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(FIG, f"train_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["stage2_main_neg", "stage2_main_pos"])
    ap.add_argument("--window-ms", type=float, default=900.0,
                    help="window shows multiple events while keeping each event's sweep visible "
                         "(~4-5 events; raise for a longer train overview, lower to zoom)")
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
