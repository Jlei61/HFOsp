#!/usr/bin/env python
"""Fig5 supplement: what the two states ARE, before asking how they respond.

This is a COMPANION to the accepted base Figure 5, not a replacement. The base
figure keeps its locked three-column syntax (slow field | 2D activity | one
continuous 15-contact readout) and is not redrawn here.

Four panels, one independent question each (CLAUDE.md section 7):

  A  how do the two slow variables move together on the way in?
     -> the (disinhibition, adaptation) trajectory, not two time series
  B  where does the net slow drive sit at the low-activity state?
  C  where does it sit 500 ms before the transition?

B and C are TIME AVERAGES over a 1 s window, not single frames. The quantity
`(1-z)*I_I - eta*m` contains the FAST inhibitory current I_I, so a single frame
maps where activity happened to be during that millisecond, not where the slow
state has moved: the whole-sheet mean changes by a median 7.2 % between frames
5 ms apart, and sampling a quiet millisecond at the low-activity state returns a
focal-core mean of -0.005 where the 1 s average is 4.00.
  D  does the focal core lead the approach, or fall behind it?
     -> per-CELL rate inside the node core against the rest of the sheet, which
        a spatial map at two instants cannot show

No difference map: D = C - B would be the same construct as B and C drawn a
third time. Panel D asks a question the maps cannot answer, and the per-cell
normalisation is what stops "the core is only 2 % of the sheet" from being read
as "the core is quiet".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPLAY = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/fig5_replay"
VERDICT = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/baseline_window_verdict.json"
OUT = ROOT / "results/paper-ready-figure/fig5_supplement_zm_states/figures"


def _state_labels():
    """`baseline` is used only if the three-clause rule says it was earned."""
    if VERDICT.exists():
        verdict = json.loads(VERDICT.read_text())
        if verdict.get("label_to_use") == "baseline":
            window = verdict.get("window_ms") or [500.0, 1000.0]
            return ("baseline", float(window[1]), verdict)
        return ("early transition", 1000.0, verdict)
    return ("low activity (label pending)", 1000.0, None)


def _nearest(times, target):
    return int(np.argmin(np.abs(np.asarray(times, float) - float(target))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1801)
    ap.add_argument("--candidate-id", default="joint_04_control")
    ap.add_argument("--pre-ictal-offset-ms", type=float, default=500.0)
    ap.add_argument("--average-window-ms", type=float, default=1000.0,
                    help="width of the time average behind panels B and C")
    args = ap.parse_args()

    stem = f"{args.candidate_id}_seed_{args.seed}_frames"
    meta = json.loads((REPLAY / f"{stem}.json").read_text())
    onset_ms = float(meta["model_ictal_onset_ms"])
    low_label, low_ms, verdict = _state_labels()
    pre_ms = onset_ms - args.pre_ictal_offset_ms

    with np.load(REPLAY / f"{stem}.npz", allow_pickle=False) as z:
        frame_t = np.asarray(z["frame_time_ms"], float)
        field = np.asarray(z["net_slow_field"], float)
        pos = np.asarray(z["positions_E"], float)
        h = np.asarray(z["h"], float)
        t_zm = np.asarray(z["zm_h_weighted_time_ms"], float)
        z_tr = np.asarray(z["zm_h_weighted_z"], float)
        m_tr = np.asarray(z["zm_h_weighted_m"], float)
        src = np.asarray(z["axis_source_xy"], float)
        snk = np.asarray(z["axis_sink_xy"], float)
        activity = np.asarray(z["activity_spike_counts"], float)
        occupancy = np.asarray(z["activity_cell_occupancy"], float)

    i_low, i_pre = _nearest(frame_t, low_ms), _nearest(frame_t, pre_ms)
    frame_dt = float(np.median(np.diff(frame_t)))
    half = max(1, int(round(0.5 * args.average_window_ms / frame_dt)))

    def _average(index):
        lo, hi = max(0, index - half), min(len(frame_t), index + half + 1)
        return field[lo:hi].mean(axis=0), (float(frame_t[lo]), float(frame_t[hi - 1]))

    f_low, span_low = _average(i_low)
    f_pre, span_pre = _average(i_pre)

    fig = plt.figure(figsize=(15.0, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.0, 1.05], wspace=0.36,
                          left=0.055, right=0.985, bottom=0.17, top=0.86)

    # ---- A: the joint slow trajectory ----
    ax = fig.add_subplot(gs[0, 0])
    keep = t_zm <= onset_ms
    x, y = 1.0 - z_tr[keep], m_tr[keep]        # disinhibition, adaptation
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", linewidth=1.8)
    lc.set_array(t_zm[keep][:-1] / 1000.0)
    ax.add_collection(lc)
    ax.autoscale_view()
    for t_mark, colour, name in ((low_ms, "#1b7837", low_label),
                                 (pre_ms, "#d6604d", "pre-ictal")):
        j = _nearest(t_zm, t_mark)
        ax.plot(1.0 - z_tr[j], m_tr[j], "o", ms=8, mfc=colour, mec="k",
                mew=1.0, zorder=5)
        ax.annotate(name, (1.0 - z_tr[j], m_tr[j]), textcoords="offset points",
                    xytext=(-6, 10) if name == "pre-ictal" else (9, -3),
                    ha="right" if name == "pre-ictal" else "left",
                    fontsize=8, color=colour, weight="bold")
    cb = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("time (s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("disinhibition  1 − z   (node-weighted)", fontsize=9)
    ax.set_ylabel("adaptation  m   (node-weighted)", fontsize=9)
    ax.set_title("slow-variable trajectory", fontsize=10)
    ax.tick_params(labelsize=8)

    # ---- B, C: where the net slow drive sits, on a SHARED scale ----
    # Separate scales. The two states differ ~20x, so a shared scale renders the
    # low-activity panel as one flat colour and hides its structure entirely; the
    # printed range on each colorbar carries the comparison instead.
    order = np.argsort(h)                      # node-field cells drawn on top
    core_cells = h >= 0.5
    for col, (frame, span, name) in enumerate(
            ((f_low, span_low, low_label), (f_pre, span_pre, "pre-ictal")), start=1):
        ax = fig.add_subplot(gs[0, col])
        vmax = float(np.percentile(frame, 99.5))
        sc = ax.scatter(pos[order, 0], pos[order, 1], c=frame[order], s=1.6,
                        cmap="plasma", vmin=0.0, vmax=vmax, linewidths=0, rasterized=True)
        ax.plot([src[0], snk[0]], [src[1], snk[1]], color="w", lw=1.1, alpha=0.75)
        ax.scatter(*src, marker="o", s=26, facecolor="none", edgecolor="w", lw=1.2)
        ax.scatter(*snk, marker="s", s=26, facecolor="none", edgecolor="w", lw=1.2)
        ax.scatter(pos[core_cells, 0], pos[core_cells, 1], s=0.8, c="#00e5ff",
                   linewidths=0, alpha=0.55, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_title(f"{name}   {span[0]:.0f}–{span[1]:.0f} ms mean", fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=9)
        if col == 1:
            ax.set_ylabel("y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("net slow current, 1 s mean" if col == 2 else "", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # ---- D: does the focal core lead the approach or fall behind it? ----
    ax = fig.add_subplot(gs[0, 3])
    n_grid = activity.shape[1]
    sheet_l = 20.0
    gx = np.clip((pos[:, 0] / sheet_l * n_grid).astype(int), 0, n_grid - 1)
    gy = np.clip((pos[:, 1] / sheet_l * n_grid).astype(int), 0, n_grid - 1)
    h_grid = np.zeros((n_grid, n_grid))
    np.maximum.at(h_grid, (gx, gy), h)
    core_grid = h_grid >= 0.5
    n_core = float(occupancy[core_grid].sum())
    n_rest = float(occupancy[~core_grid].sum())
    # 250 ms bins: the core is 727 cells, so at 100 ms its per-cell rate swings
    # between zero and one event and the trend is unreadable.
    step = max(1, int(round(250.0 / frame_dt)))
    centres, core_rate, rest_rate = [], [], []
    for start in range(0, len(frame_t) - step + 1, step):
        block = activity[start:start + step].sum(axis=0)
        centres.append(float(frame_t[start:start + step].mean()))
        core_rate.append(block[core_grid].sum() / n_core)
        rest_rate.append(block[~core_grid].sum() / n_rest)
    centres = np.asarray(centres)
    ax.plot(centres / 1000.0, core_rate, "-", color="#00a0b0", lw=1.8,
            label=f"node core  (n = {n_core:.0f})")
    ax.plot(centres / 1000.0, rest_rate, "-", color="#7f7f7f", lw=1.8,
            label=f"rest of sheet  (n = {n_rest:.0f})")
    ax.axvline(onset_ms / 1000.0, color="k", ls="--", lw=1.2)
    for t_mark, colour in ((low_ms, "#1b7837"), (pre_ms, "#d6604d")):
        ax.axvline(t_mark / 1000.0, color=colour, ls=":", lw=1.2)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("spikes per cell per 250 ms bin", fontsize=9)
    ax.set_title("core vs rest of sheet", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "fig5-supplement-zm-states"
    fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)

    core = h >= 0.5
    payload = {
        "figure": "fig5 supplement -- state characterization",
        "role": "visual diagnostic companion; the base Figure 5 is unchanged",
        "seed": args.seed, "candidate_id": args.candidate_id,
        "onset_ms": onset_ms, "onset_label_is_operational": True,
        "states": {"low_activity": {"label": low_label,
                                    "frame_time_ms": float(frame_t[i_low]),
                                    "average_window_ms": list(span_low)},
                   "pre_ictal": {"frame_time_ms": float(frame_t[i_pre]),
                                 "offset_ms": args.pre_ictal_offset_ms,
                                 "average_window_ms": list(span_pre)}},
        "net_slow_current": {
            state: {"core_mean": float(f[core].mean()),
                    "surround_mean": float(f[~core].mean()),
                    "p99_5": float(np.percentile(f, 99.5))}
            for state, f in (("low_activity", f_low), ("pre_ictal", f_pre))},
        "spatial_pattern_correlation_between_states": float(
            np.corrcoef(f_low, f_pre)[0, 1]),
        "core_vs_rest_per_cell_ratio": {
            f"{int(lo)}-{int(hi)}ms": float(
                activity[(frame_t >= lo) & (frame_t < hi)].sum(axis=0)[core_grid].sum()
                / n_core
                / max(activity[(frame_t >= lo) & (frame_t < hi)].sum(axis=0)[~core_grid].sum()
                      / n_rest, 1e-12))
            for lo, hi in ((0, 1000), (1000, 2000), (2000, 3000), (3000, onset_ms - 500),
                           (onset_ms, frame_t[-1]))},
        "baseline_label_source": ("baseline_window_verdict.json" if verdict
                                  else "not yet computed"),
        "claim_boundary": [
            "One network (seed 1801). Panels B-D are that network's fields, not a "
            "cohort statement.",
            "Panels B-D are 1 s TIME AVERAGES. The plotted quantity contains the fast "
            "inhibitory current, so a single frame reports where activity was during "
            "that millisecond rather than where the slow state has moved. An earlier "
            "single-frame reading of 'the focal core is at zero' was an artefact of "
            "sampling a quiet millisecond and has been withdrawn.",
            "The two terms are stored only as their sum, so this figure cannot "
            "attribute the difference to disinhibition rather than adaptation.",
            "The onset time is this round's operational definition (20 ms EMA of the "
            "population E rate >= 120 Hz sustained >= 100 ms), not a clinical seizure.",
            "The state on the left carries the name the three-clause rule assigned; if "
            "that rule did not find a qualifying window it is NOT called baseline.",
        ],
    }
    (OUT / "fig5-supplement-zm-states-metadata.json").write_text(
        json.dumps(payload, indent=2))
    print(json.dumps({"low_state": low_label,
                      "low_ms": float(frame_t[i_low]),
                      "pre_ms": float(frame_t[i_pre]),
                      "core_vs_surround": payload["net_slow_current"]}, indent=2))


if __name__ == "__main__":
    main()
