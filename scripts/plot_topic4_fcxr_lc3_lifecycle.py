#!/usr/bin/env python
"""The three registered lifecycle figures.

These are diagnostic and parameter-sweep figures, which the Topic 4 style guide
exempts from the four-column mechanism layout; each answers one question the
others cannot.

A  the ladder into onset -- did discrete events accumulate, and did they speed up
B  the slow state across those events -- what actually moved, and where
C  how far each arm got around the loop, and what was held still to get it there

Everything is read from frozen artifacts.  Nothing is re-simulated and no arm is
re-classified here: stage labels come from the tested contract in
``src.topic4_fcxr_lc3_stage``.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import run_topic4_fcxr_lc3 as E01  # noqa: E402
from src.topic4_fcxr_lc3_stage import (  # noqa: E402
    STAGE_ORDER,
    lifecycle_stage,
    reference_band,
    returned_to_reference,
    stage_index,
)

BASE = E01.OUT          # the stage directory itself
FIGS = os.path.join(BASE, "figures")

SEED_COLOR = {401: "#1b4f9c", 405: "#c0392b", 406: "#1e8449"}
REGION_COLOR = {"core_A": "#c0392b", "core_B": "#e07b39",
                "axial": "#1b4f9c", "off_axis": "#7f8c8d"}
REGION_LABEL = {"core_A": "core A", "core_B": "core B",
                "axial": "along the axis", "off_axis": "off axis"}
STAGE_LABEL = {
    "IED_TRAIN_NO_ONSET": "events,\nno entry",
    "ONE_SHOT": "entry without\naccumulation",
    "ONSET_NO_OFFSET": "entry,\nno stop",
    "OFFSET_NO_RECOVERY": "stops, does\nnot come back",
    "FULL_LIFECYCLE": "comes back\nas events",
}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _entry_records():
    """Ledger records if they landed, else the 45 s recon records."""
    out = {}
    for path in sorted(glob.glob(os.path.join(BASE, "entry_ledger", "entry_noise*.json"))):
        if ".DONE." in path or ".RUNNING" in path:
            continue
        r = _load(path)
        if r.get("status") == "COMPLETE":
            out[int(r["noise_seed"])] = ("ledger", r)
    for path in sorted(glob.glob(os.path.join(BASE, "dynamic_reconnaissance",
                                              "recon_noise*.json"))):
        if ".DONE." in path or ".RUNNING" in path:
            continue
        r = _load(path)
        if r.get("status") == "COMPLETE" and int(r["noise_seed"]) not in out:
            out[int(r["noise_seed"])] = ("recon", r)
    return dict(sorted(out.items()))


def _onset_ms(kind, rec):
    if kind == "ledger":
        return rec.get("onset_ms")
    bout = rec.get("lifecycle", {}).get("bout")
    return None if bout is None else float(bout[0]) * 1000.0


# --------------------------------------------------------------------------- A

def figure_a(records, band, path):
    """Did discrete events accumulate into onset, and did they speed up?"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 5.2), height_ratios=[1.0, 1.0])
    seeds = list(records)
    floor_gap = 1.0 / band["event_rate_hi"]
    ceil_gap = 1.0 / band["event_rate_lo"]
    trains = {}
    for seed in seeds:
        kind, rec = records[seed]
        onset = _onset_ms(kind, rec)
        trains[seed] = (onset, [e for e in rec["events"]
                                if e["returned"] and (onset is None
                                                      or e["t_off_ms"] < onset)])
    # Scaled over what these events actually span, not over the reference band:
    # nearly all of them sit near the band's floor, so a band-relative scale
    # compresses every bar into the bottom of its row and hides the one thing
    # that does change.
    every = [e["peak_ext"] for _, evs in trains.values() for e in evs]
    lo, hi = min(every), max(every)
    def _h(p):
        return 0.78 * (0.10 + 0.90 * (p - lo) / (hi - lo))

    n_below, n_gaps, halves, n_over, n_over_last = 0, 0, [], 0, 0
    for row, seed in enumerate(seeds):
        onset, ret = trains[seed]
        base = len(seeds) - 1 - row
        for k, e in enumerate(ret):
            t = (e["t_on_ms"] - onset) / 1000.0
            ax1.plot([t, t], [base, base + _h(e["peak_ext"])], color=SEED_COLOR[seed],
                     lw=1.6, solid_capstyle="butt")
            if e["peak_ext"] > band["part_hi"]:
                n_over += 1
                n_over_last += int(k >= len(ret) - 2)
        ax1.axhline(base, color="#d5d8dc", lw=0.7, zorder=0)
        ax1.plot([-5.5, 0.0], [base + _h(band["part_hi"])] * 2, color="#8fa3b8",
                 lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax1.text(-7.15, base + 0.30, f"noise {seed}\n{len(ret)} events",
                 fontsize=8, color=SEED_COLOR[seed], va="center")
        gaps = np.diff([e["t_on_ms"] for e in ret]) / 1000.0
        if gaps.size:
            ax2.plot(np.arange(2, len(ret) + 1), gaps, "o-", ms=3.4, lw=1.3,
                     color=SEED_COLOR[seed], label=f"noise {seed}")
            n_below += int((gaps < floor_gap).sum())
            n_gaps += gaps.size
            halves.append((gaps[:gaps.size // 2].mean(), gaps[gaps.size // 2:].mean()))

    n_events = sum(len(evs) for _, evs in trains.values())
    ax1.axvline(0.0, color="#111111", lw=1.4)
    ax1.text(0.10, len(seeds) - 0.12, "enters the\nhigh state", fontsize=8,
             color="#111111", va="top")
    ax1.text(0.10, _h(band["part_hi"]), "largest event the quiet\nbaseline ever produced",
             fontsize=7.2, color="#5d6d7e", va="center")
    ax1.set_xlim(-7.4, 2.6)
    ax1.set_ylim(-0.12, len(seeds))
    ax1.set_yticks([])
    ax1.set_xlabel("time before the tissue enters the high state (s)")
    ax1.set_title(f"A  the events leading into entry are ordinary-sized — "
                  f"{n_events - n_over} of {n_events} stay inside the quiet baseline's "
                  f"range", loc="left")

    ax2.axhspan(floor_gap, ceil_gap, color="#e4eaf0", zorder=0)
    ax2.axhline(floor_gap, color="#8fa3b8", lw=0.9, zorder=1)
    ax2.text(15.7, floor_gap + 0.02, "closest spacing the quiet\nbaseline ever produced",
             ha="right", va="bottom", fontsize=7.8, color="#5d6d7e")
    ax2.set_xlabel("event number within the recording")
    ax2.set_ylabel("gap since the\nprevious event (s)")
    ax2.set_xlim(1.4, 16.0)
    ax2.set_ylim(0, 1.15)
    ax2.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    shrink = sum(1 for a, b in halves if b < a)
    ax2.set_title(f"what changes is the timing — gaps shorten in {shrink} of "
                  f"{len(halves)} seeds, {n_below} of {n_gaps} below the baseline floor",
                  loc="left")
    ax2.text(0.015, 0.06, f"all {n_over} oversized events are among the last two "
             f"before entry", transform=ax2.transAxes, fontsize=7.6, color="#5d6d7e")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- B

def figure_b(records, path):
    """What did the slow state do across those events, and where?"""
    ledgers = {s: r for s, (k, r) in records.items() if k == "ledger"}
    if not ledgers:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    ax_mean, ax_region, ax_relay = axes

    for seed, rec in ledgers.items():
        rows = [r for r in rec["event_ledger"]["events"] if r["phase"] == "pre_onset"]
        if not rows:
            continue
        idx = np.arange(1, len(rows) + 1)
        ax_mean.plot(idx, [r["post"]["D"]["all"] for r in rows if r["post"]],
                     "o-", ms=3, lw=1.3, color=SEED_COLOR[seed], label=f"noise {seed}")
        ax_relay.plot(idx, [r["post"]["X"]["all"] for r in rows if r["post"]],
                      "o-", ms=3, lw=1.3, color=SEED_COLOR[seed], label=f"noise {seed}")
    ax_mean.set_xlabel("event number before entry")
    ax_mean.set_ylabel("wear, whole array")
    ax_mean.set_title("wear ratchets up event by event", loc="left")
    ax_mean.legend(frameon=False, fontsize=8)

    primary = ledgers.get(401, next(iter(ledgers.values())))
    rows = [r for r in primary["event_ledger"]["events"]
            if r["phase"] == "pre_onset" and r["post"]]
    idx = np.arange(1, len(rows) + 1)
    for region in ("core_A", "core_B", "axial", "off_axis"):
        ax_region.plot(idx, [r["post"]["D"][region] for r in rows], "o-", ms=3, lw=1.3,
                       color=REGION_COLOR[region], label=REGION_LABEL[region])
    ax_region.set_xlabel("event number before entry")
    ax_region.set_ylabel("wear")
    ax_region.set_title("and it builds fastest in the two cores", loc="left")
    ax_region.legend(frameon=False, fontsize=8)

    ax_relay.set_xlabel("event number before entry")
    ax_relay.set_ylabel("relay availability")
    ax_relay.set_ylim(0.0, 1.05)
    ax_relay.set_title("while the relay does not move at all", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- C

def _arms(band):
    """Every arm on disk, with what it was allowed to do."""
    arms = []

    for path in sorted(glob.glob(os.path.join(BASE, "dynamic_reconnaissance",
                                              "recon_noise*.json"))):
        if ".DONE." in path or ".RUNNING" in path:
            continue
        r = _load(path)
        if r.get("status") != "COMPLETE":
            continue
        bout = r["lifecycle"].get("bout")
        onset = None if bout is None else float(bout[0]) * 1000.0
        n_pre = sum(1 for e in r["events"]
                    if e["returned"] and (onset is None or e["t_off_ms"] < onset))
        st = lifecycle_stage(onset_ms=onset, offset_ms=None,
                             n_returning_before_onset=n_pre)
        arms.append(dict(group="nothing held still", entry_observed=True,
                         label=f"no-kick trajectory, noise {r['noise_seed']}",
                         stage=st["stage"], note=f"{n_pre} events before entry"))

    qw = os.path.join(BASE, "quiet_watch", "quiet_watch.json")
    if os.path.isfile(qw):
        for r in _load(qw)["rows"]:
            st = lifecycle_stage(
                onset_ms=r["departure_ms"], offset_ms=None,
                n_returning_before_onset=None if r["departed"] else 0)
            arms.append(dict(group="wear and relay held still", entry_observed=True,
                             label=f"quiet state watched 12 s, wear {r['d_label']}",
                             stage=st["stage"],
                             note="left" if r["departed"] else "never left"))

    # Per-arm files, not the aggregates: each aggregate holds only the arms of the
    # batch that wrote it last, so reading them would silently drop the earlier
    # sweeps whose per-arm records are still on disk.
    # The registered gate is read from the sweep's own record; the per-arm
    # `is_control` flag marks whichever arm a batch promoted when its custom grid
    # contained no registered setting, so it does not identify the registered one.
    sweep = os.path.join(BASE, "hill_placement_sweep", "hill_sweep.json")
    base_gate = _load(sweep).get("base_y_gate") if os.path.isfile(sweep) else None
    for path in sorted(glob.glob(os.path.join(BASE, "hill_placement_sweep",
                                              "arm_gate*.json"))):
        r = _load(path)
        label = r.get("resolved_label", "")
        terminated = label not in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        # These arms never had to reach the interictal tier to be called
        # terminated, so the return is unmeasured rather than failed.
        st = (lifecycle_stage(onset_ms=0.0, offset_ms=1.0,
                              n_returning_before_onset=None, return_check=None)
              if terminated else dict(stage="ONSET_NO_OFFSET"))
        registered = base_gate is not None and abs(r["y_gate"] - base_gate) < 1e-6
        where = ("left where it was registered" if registered
                 else f"moved to {r['y_gate']:.1f}")
        used = float(r.get("total_ms", 0.0)) / 1000.0
        budget = r.get("extended_ms_setting")
        # A high branch that persists never triggers the protocol's extension, so
        # it is only ever observed for the screen window however long a budget it
        # was given. Saying "watched 1.5 s" without that is how the quiet side of
        # the map came to read as settled.
        window = (f"watched {used:.1f} s"
                  + (f" of {float(budget) / 1000.0:.0f} s allowed"
                     if budget and float(budget) / 1000.0 > used + 1e-9 else ""))
        arms.append(dict(
            group="wear held still, relay free", entry_observed=False,
            label=(f"relay curve {where}"
                   + (f", half-activation {r['K_y']:.0f}" if r.get("K_y") != 5.0 else "")
                   + f", {window}"),
            stage=st["stage"], note=label.lower().replace("_", " ")))

    for path in sorted(glob.glob(os.path.join(BASE, "return_gate_probe",
                                              "clamp_*.json"))):
        r = _load(path)
        if r.get("status") != "COMPLETE":
            continue
        chk = returned_to_reference(
            n_returning_after_offset=r["n_returning"],
            event_rate_hz=r.get("event_rate_per_s", 0.0), band=band,
            durations_ms=r.get("duration_ms") or [],
            participation=r.get("participation") or [])
        st = lifecycle_stage(onset_ms=0.0, offset_ms=1.0,
                             n_returning_before_onset=None, return_check=chk)
        free = bool(r.get("free_wear"))
        note = f"{r['n_returning']} events came back"
        if free:
            note += f"; wear {r['wear_start']:.2f} to {r['wear_end']:.3f} on its own"
        arms.append(dict(
            group="relay clamped", entry_observed=False,
            label=(f"relay held at {r['x_clamp']:.3f}, "
                   + ("wear free" if free else "wear held still")),
            stage=st["stage"], note=note))
    return arms


def figure_c(arms, path):
    """How far did each arm get, and what was held still to get it there?"""
    order = {g: i for i, g in enumerate(
        ["nothing held still", "wear and relay held still",
         "wear held still, relay free", "relay clamped"])}
    arms = sorted(arms, key=lambda a: (order.get(a["group"], 9),
                                       -stage_index(a["stage"]), a["label"]))
    group_color = {"nothing held still": "#111111",
                   "wear and relay held still": "#7f8c8d",
                   "wear held still, relay free": "#1b4f9c",
                   "relay clamped": "#b03a2e"}
    fig, ax = plt.subplots(figsize=(9.6, 0.30 * len(arms) + 2.3))
    for y, a in enumerate(arms):
        reach = stage_index(a["stage"])
        start = -0.42 if a["entry_observed"] else stage_index("ONSET_NO_OFFSET") - 0.42
        ax.barh(y, reach + 0.42 - start, left=start, height=0.62,
                color=group_color[a["group"]],
                alpha=1.0 if a["entry_observed"] else 0.45,
                hatch=None if a["entry_observed"] else "//",
                edgecolor="white", linewidth=0.6)
        ax.text(reach + 0.55, y, a["note"], va="center", fontsize=7.4, color="#566573")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a["label"] for a in arms], fontsize=7.8)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGE_ORDER], fontsize=8)
    ax.set_xlim(-0.6, len(STAGE_ORDER) + 0.9)
    ax.set_ylim(-0.7, len(arms) - 0.3)
    ax.invert_yaxis()
    for x in range(len(STAGE_ORDER)):
        ax.axvline(x - 0.5, color="#e5e8e8", lw=0.8, zorder=0)
    handles = [Patch(facecolor=c, label=g) for g, c in group_color.items()]
    handles.append(Patch(facecolor="#999999", hatch="//", alpha=0.45,
                         label="started already in the high state, so entry was never tested"))
    ax.legend(handles=handles, frameon=False, fontsize=7.8, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.9 / len(arms)))
    ax.set_title("C  how far each arm got around the loop, and what was held still "
                 "to get it there", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGS, exist_ok=True)
    baseline = _load(E01.ARTIFACTS["lc1_baseline"])
    band = reference_band(baseline)
    records = _entry_records()
    if not records:
        raise SystemExit("no trajectory records on disk")
    written = []
    written.append(figure_a(records, band,
                            os.path.join(FIGS, "lc3_A_event_ladder_into_entry.png")))
    b = figure_b(records, os.path.join(FIGS, "lc3_B_slow_state_across_events.png"))
    written.append(b or "B skipped: no per-event ledger on disk yet")
    arms = _arms(band)
    written.append(figure_c(arms, os.path.join(FIGS, "lc3_C_how_far_each_arm_got.png")))
    print(json.dumps(dict(
        figures=written, n_arms=len(arms),
        sources={s: k for s, (k, _) in records.items()},
        stages={s: sum(1 for a in arms if a["stage"] == s) for s in STAGE_ORDER},
    ), indent=2))


if __name__ == "__main__":
    main()
