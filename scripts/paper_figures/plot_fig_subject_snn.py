"""Paper-ready subject-specific SNN panel (Topic 4), 1-row x 4-col, s3_brakeoff style.

Layout (matching scripts/paper_figures/plot_fig5_core_model_s3_brakeoff.py):
  mechanism | tempA source | tempB source | electrode readout

- mechanism: heterogeneity map (per-E-neuron 18-V_th, plasma), the two template-source cores
  (dashed circles), the patient electrodes (real contact layout), and the E->E long-axis corridor
  as a light band overlay. The core-member contacts are highlighted to show the cores OVERLAP the
  electrode interictal-event-onset (early) region.
- tempA source / tempB source: a representative propagation event when each template's source core
  is the ignition site (viridis relative onset), real contact layout.
- electrode readout: source-core run (tempA, forward shading) + sink-core run (tempB, reverse
  shading) concatenated on the patient montage; peak-locus per clean readable event.

Plotting-only: consumes figdata_<tag>.npz + readout_<tag>.json from run_sef_hfo_subject_snn.py
(template_source placement). No simulation rerun.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"

FWD_SHADE, REV_SHADE = "#f4b266", "#78a6d8"
TA_COLOR, TB_COLOR = "#B2182B", "#2166AC"
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
AXIS_COL = "#a65f00"
CORE_COLS = ("crimson", "#2166ac")
HOMOGENEOUS_CORE = "#B2182B"
OFF, SHADE_PAD_MS = 1.35, 22.0
DOT, ALPHA = 8.0, 0.90
TRACE_BAND_HZ = (30.0, 80.0)
TRACE_SCALE_PERCENTILE = 95.0


def _axis(theta_deg):
    th = np.deg2rad(theta_deg); u = np.array([np.cos(th), np.sin(th)])
    return u, np.array([-u[1], u[0]])


def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name)); return m.group(0) if m else str(name)


def _shaft_color(name, shafts):
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _load(tag):
    return (json.load(open(RUN / f"readout_{tag}.json")),
            np.load(RUN / f"figdata_{tag}.npz", allow_pickle=True))


def _registered_axis_display(fd) -> dict:
    """Express the accepted SNN sheet in its native registered TA-axis frame.

    The subject simulation is already built after one isotropic patient-plane
    registration.  Its source-to-sink axis is fixed before event readout and is
    stored in ``reg``.  The paper panel therefore needs only an origin shift and
    the (numerically negligible for E1146) projection onto that stored axis;
    it must not be fitted back onto a second patient-coordinate rendering.
    """
    reg = fd["reg"].item()
    if reg.get("coordinate_frame") == "gradient_shared":
        half = 0.5 * float(fd["L"])
        return {
            "matrix": np.eye(2),
            "offset": np.array([-half, -half]),
            "scale": 1.0,
            "xlim": (-half, half),
            "ylim": (-half, half),
            "record": "figdata.reg.gradient_contract",
            "axis_definition": "frozen template_propagation_axis_v2 shared plane",
            "axis_direction_convention": "positive early-to-late",
            "transverse_sign": int(reg["gradient_contract"]["transverse_sign"]),
            "fit_rmse_mm": 0.0,
            "fit_max_error_mm": 0.0,
        }
    axis = np.asarray(reg["axis_unit"], float)
    axis /= np.linalg.norm(axis)
    transverse = np.asarray([-axis[1], axis[0]], float)
    matrix = np.column_stack((axis, transverse))
    center = np.asarray(reg["center"], float)
    offset = -center @ matrix
    half = 0.5 * float(fd["L"])
    return {
        "matrix": matrix,
        "offset": offset,
        "scale": 1.0,
        "xlim": (-half, half),
        "ylim": (-half, half),
        "record": "figdata.reg",
        "axis_definition": "registered template-source TA axis",
        "axis_direction_convention": "source centroid to sink centroid",
        "transverse_sign": 1,
        "fit_rmse_mm": 0.0,
        "fit_max_error_mm": 0.0,
    }


def _display_xy(points, display):
    points = np.asarray(points, float)
    if display is None:
        return points
    return points @ np.asarray(display["matrix"], float) + np.asarray(display["offset"], float)


def _display_radius(radius, display):
    return float(radius) if display is None else float(radius) * float(display["scale"])


def _style(ax, L, *, display=None, formal=False, show_ylabel=True):
    if display is None:
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        xlabel, ylabel = "x (mm)", "y (mm)"
    else:
        ax.set_xlim(*display["xlim"]); ax.set_ylim(*display["ylim"])
        xlabel = (
            "TA–TB shared axis (mm)"
            if "shared plane" in str(display.get("axis_definition", ""))
            else "TA axis (mm)"
        )
        ylabel = "y (mm)"
    ax.set_aspect("equal")
    label_fs = 11.5 if formal else 7.6
    tick_fs = 10.0 if formal else 7.0
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=label_fs)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=tick_fs, length=3.2 if formal else 2.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("0.25")


def _ee_ellipse(foci, core_r, theta_deg):
    """Visualize the anisotropic E->E long-axis footprint as an oriented ellipse."""
    foci = np.asarray(foci, float)
    major = float(np.linalg.norm(foci[1] - foci[0]) / 2.0 + 2.0)
    minor = max(1.5 * float(core_r), 1.6)
    return Ellipse(
        xy=np.mean(foci, axis=0),
        width=2.0 * major,
        height=2.0 * minor,
        angle=float(theta_deg),
        facecolor=FWD_SHADE,
        edgecolor=AXIS_COL,
        lw=1.3,
        alpha=0.28,
        zorder=3,
    )


def _draw_contacts(ax, contacts, names, shafts, core_a, core_b, *, homogeneous_cores=False):
    contacts = np.asarray(contacts, float)
    for sh in shafts:
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        c = contacts[idx]; col = _shaft_color(sh, shafts)
        ax.plot(c[:, 0], c[:, 1], color=col, lw=1.0, alpha=0.55, zorder=5)
        ax.scatter(c[:, 0], c[:, 1], s=34, marker="o", fc="white", ec=col, lw=1.0, zorder=6)
    # highlight the template-source core-member contacts (the early electrodes)
    for nm, hi in (("A", core_a), ("B", core_b)):
        ec = HOMOGENEOUS_CORE if homogeneous_cores else ("crimson" if nm == "A" else "#1f4fd8")
        for n in hi:
            if n in names:
                xy = contacts[list(names).index(n)]
                ax.scatter([xy[0]], [xy[1]], s=70, marker="o", fc="none", ec=ec, lw=1.8, zorder=7)


def _plot_mechanism(
    ax, fd, core_a, core_b, shafts, title="mechanism", *, display=None, formal=False,
):
    pos = np.asarray(fd["posE"], float); v = np.asarray(fd["vth"], float)
    foci = np.asarray(fd["foci"], float); contacts = np.asarray(fd["contacts"], float)
    names = [str(x) for x in fd["names"]]; L = float(fd["L"]); core_r = float(fd["core_r"])
    pos = _display_xy(pos, display)
    foci = _display_xy(foci, display)
    contacts = _display_xy(contacts, display)
    core_r = _display_radius(core_r, display)
    axis_vec = foci[1] - foci[0]
    theta_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))
    ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(18.0 - v, 0.0, None), s=DOT, cmap="plasma",
               vmin=0.0, vmax=1.2, alpha=ALPHA, linewidths=0, rasterized=True, zorder=2)
    ax.add_patch(_ee_ellipse(foci, core_r, theta_deg))
    for i, f in enumerate(foci):
        core_col = HOMOGENEOUS_CORE if formal else CORE_COLS[i]
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec=core_col, lw=1.25, ls="--", zorder=7))
        if not formal:
            ax.text(f[0], f[1] + 1.0, "A" if i == 0 else "B", fontsize=9, color=core_col,
                    fontweight="bold", ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.plot([foci[0, 0], foci[1, 0]], [foci[0, 1], foci[1, 1]],
            color=AXIS_COL, lw=1.7, alpha=0.95, zorder=8)
    _draw_contacts(
        ax, contacts, names, shafts, core_a, core_b, homogeneous_cores=formal,
    )
    ax.set_title(title, fontsize=12.0 if formal else 9.5, fontweight="bold", pad=7)
    _style(ax, L, display=display, formal=formal, show_ylabel=True)


def _plot_event(
    ax,
    fd,
    rep,
    title,
    shafts,
    core_a,
    core_b,
    source_index,
    *,
    normalize_color=False,
    display=None,
    formal=False,
    show_ylabel=True,
):
    pos = np.asarray(fd["posE"], float); foci = np.asarray(fd["foci"], float)
    contacts = np.asarray(fd["contacts"], float); names = [str(x) for x in fd["names"]]
    L = float(fd["L"]); core_r = float(fd["core_r"])
    pos = _display_xy(pos, display)
    foci = _display_xy(foci, display)
    contacts = _display_xy(contacts, display)
    core_r = _display_radius(core_r, display)
    onset = np.asarray(rep["onset"], float) if rep else np.full(len(pos), np.nan)
    fin = np.isfinite(onset); bg = np.zeros(len(pos), bool); bg[::4] = True
    ax.scatter(pos[bg & ~fin, 0], pos[bg & ~fin, 1], s=1.2, c="0.86", alpha=0.35,
               linewidths=0, rasterized=True, zorder=1)
    event_mappable = None
    if fin.any():
        rel = onset.copy(); rel[fin] -= np.nanmin(rel[fin])
        vmax = max(1.0, float(np.nanpercentile(rel[fin], 98)))
        display_values = rel[fin] / vmax if normalize_color else rel[fin]
        display_vmax = 1.0 if normalize_color else vmax
        event_mappable = ax.scatter(
            pos[fin, 0],
            pos[fin, 1],
            c=display_values,
            s=DOT,
            cmap="viridis",
            vmin=0.0,
            vmax=display_vmax,
            alpha=ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    # The actual ignition core is passed explicitly; reader-facing titles are
    # free to change without silently changing which core receives the star.
    src_i = int(source_index)
    for i, f in enumerate(foci):
        core_col = HOMOGENEOUS_CORE if formal else CORE_COLS[i]
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec=core_col, lw=1.2, ls="--", zorder=5))
        if i == src_i:
            ax.scatter([f[0]], [f[1]], marker="*", s=150, c="black", ec="white", lw=0.8, zorder=7)
    ax.plot([foci[0, 0], foci[1, 0]], [foci[0, 1], foci[1, 1]], color="0.20", lw=1.2, alpha=0.7, zorder=4)
    _draw_contacts(
        ax, contacts, names, shafts, core_a, core_b, homogeneous_cores=formal,
    )
    title_color = TA_COLOR if source_index == 0 else TB_COLOR
    ax.set_title(title, fontsize=12.0 if formal else 9.5, fontweight="bold", pad=7,
                 color=title_color if formal else "black")
    _style(ax, L, display=display, formal=formal, show_ylabel=show_ylabel)
    return event_mappable


def _active_order(names, contacts, u, p, ev_lists):
    active = set()
    for evs in ev_lists:
        for e in evs:
            active.update(n for n, val in (e.get("ranks") or {}).items() if val is not None)
    keep = [i for i, n in enumerate(names) if n in active] or list(range(len(names)))
    pp = np.array([contacts[i] @ u for i in keep]); qq = np.array([contacts[i] @ p for i in keep])
    order = np.lexsort((qq, pp)); return [keep[i] for i in order]


def _plot_readout(
    ax, fd, events, order, names, shafts, window_ms=5000.0, reader_labels=False, *, formal=False,
):
    """Single twoend run: LFP train on the patient montage; forward events warm, reverse cool;
    peak-locus per clean readable event (matches the reference s3_brakeoff readout)."""
    lfp = np.abs(np.asarray(fd["lfp_trace"], float)); t = np.asarray(fd["times"], float)
    win_hi = min(float(window_ms), float(t[-1])); sel = (t >= 0.0) & (t <= win_hi); ts = t[sel]
    sub = lfp[sel][:, order].T
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(order)) * OFF
    for row, ci in enumerate(order):
        ax.plot(ts, zt[row] + y[row], color=_shaft_color(names[ci], shafts), lw=0.72, alpha=0.9, zorder=3)
    n_fwd = n_rev = 0
    for e in events:
        if e.get("sign") is None or e.get("n_part", 0) < 4 or e["t_on"] > win_hi:
            continue
        fwd = e["sign"] > 0; n_fwd += fwd; n_rev += not fwd
        shade = FWD_SHADE if fwd else REV_SHADE
        ax.axvspan(max(0.0, e["t_on"] - SHADE_PAD_MS), min(win_hi, e["t_off"] + SHADE_PAD_MS),
                   color=shade, alpha=0.26, lw=0, zorder=0)
        pts, ranks = [], (e.get("ranks") or {})
        for row, ci in enumerate(order):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[row][m]))]
            pts.append((ts[pi], zt[row][pi] + y[row]))
            ax.plot(ts[pi], zt[row][pi] + y[row], "o", ms=2.2, mfc="black", mec="white", mew=0.35, zorder=6)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts)); ax.plot(px, py, "-", color="black", lw=0.72, alpha=0.7, zorder=5)
    tick_fs = 10.0 if formal else 7.0
    label_fs = 11.5 if formal else 8.2
    legend_fs = 9.6 if formal else 7.4
    ax.set_xlim(0, win_hi); ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=tick_fs)
    for tick, ci in zip(ax.get_yticklabels(), order):
        tick.set_color(_shaft_color(names[ci], shafts))
    ax.tick_params(axis="y", length=3.0, labelsize=tick_fs, color="0.35")
    ax.tick_params(axis="x", length=3.2, labelsize=tick_fs, color="0.35")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.35"); ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel("time (ms)", fontsize=label_fs)
    label_a = "model forward event" if reader_labels else "tempA (forward) event"
    label_b = "model reverse event" if reader_labels else "tempB (reverse) event"
    peak_label = "contact peak order" if reader_labels else "peak order"
    ax.legend(handles=[Patch(facecolor=FWD_SHADE, alpha=0.4, label=label_a),
                       Patch(facecolor=REV_SHADE, alpha=0.4, label=label_b),
                       Line2D([0], [0], color="black", marker="o", lw=0.8, ms=3, mfc="black", mec="white", label=peak_label)],
              frameon=False, fontsize=legend_fs, loc="upper right", bbox_to_anchor=(1.0, 1.065),
              borderaxespad=0.0, ncol=3, handlelength=1.5, columnspacing=0.9)
    return {"forward_events": n_fwd, "reverse_events": n_rev}


def _plot_interictal_sample_readout(
    ax,
    fd,
    events,
    names,
    shafts,
    *,
    window_ms=1200.0,
):
    """Reference-style signed 30--80 Hz trace containing both TA and TB events.

    This follows the accepted Figure-5 readout grammar, but consumes the
    Figure-4 spontaneous interictal artifact only.  No runaway or ictal marker
    is drawn here.
    """
    clean = sorted(
        (
            event for event in events
            if event.get("sign") is not None and int(event.get("n_part", 0)) >= 4
        ),
        key=lambda event: float(event["t_on"]),
    )
    opposite_pairs = [
        (left, right)
        for i, left in enumerate(clean)
        for right in clean[i + 1:]
        if float(left["sign"]) * float(right["sign"]) < 0.0
    ]
    if not opposite_pairs:
        raise ValueError("formal interictal readout requires both TA and TB clean events")
    raw = np.asarray(fd["lfp_trace"], float)
    times = np.asarray(fd["times"], float)
    dt_ms = float(np.median(np.diff(times)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("virtual-SEEG time axis must have positive regular spacing")
    fs_hz = 1000.0 / dt_ms
    sos = butter(4, TRACE_BAND_HZ, btype="bandpass", fs=fs_hz, output="sos")
    signed = sosfiltfilt(sos, raw, axis=0)

    # Select the closest opposite-direction pair from one continuous run.  The
    # pair, rather than one hand-picked event, is the formal display contract.
    pair = min(
        opposite_pairs,
        key=lambda item: abs(
            0.5 * (float(item[1]["t_on"]) + float(item[1]["t_off"]))
            - 0.5 * (float(item[0]["t_on"]) + float(item[0]["t_off"]))
        ),
    )
    pair_start = min(float(event["t_on"]) for event in pair)
    pair_end = max(float(event["t_off"]) for event in pair)
    span = pair_end - pair_start
    window_ms = max(float(window_ms), span + 240.0)
    start = max(float(times[0]), 0.5 * (pair_start + pair_end - window_ms))
    end = min(float(times[-1]), start + window_ms)
    start = max(float(times[0]), end - float(window_ms))
    sel = (times >= start) & (times <= end)
    if int(sel.sum()) < 20:
        raise ValueError("formal interictal readout window is too short")
    ts = times[sel] - start
    trace = signed[sel]

    scale = np.percentile(np.abs(trace), TRACE_SCALE_PERCENTILE, axis=0)
    positive = scale[np.isfinite(scale) & (scale > 1e-12)]
    if positive.size == 0:
        raise ValueError("30-80 Hz virtual-SEEG trace is constant")
    scale = np.maximum(scale, 0.15 * float(np.median(positive)))
    trace = 0.70 * trace / scale[None, :]

    # Native artifact order is SCL6..9 then ICL1..11, so increasing vertical
    # offsets reproduce the reference ordering (ICL11 top, SCL6 bottom).
    order = list(range(len(names)))
    y = np.arange(len(order), dtype=float) * 1.28
    displayed = [
        event for event in clean
        if float(event["t_on"]) >= start and float(event["t_off"]) <= end
    ]
    for event in displayed:
        shade = FWD_SHADE if float(event["sign"]) > 0.0 else REV_SHADE
        ax.axvspan(
            max(0.0, float(event["t_on"]) - start),
            min(float(ts[-1]), float(event["t_off"]) - start),
            color=shade,
            alpha=0.18,
            lw=0,
            zorder=0,
        )
    for row, ci in enumerate(order):
        ax.plot(
            ts,
            trace[:, ci] + y[row],
            color=_shaft_color(names[ci], shafts),
            lw=0.90,
            alpha=0.94,
            zorder=3,
            clip_on=True,
        )

    ax.set_xlim(0.0, float(ts[-1]))
    ax.set_ylim(-0.70, float(y[-1] + 1.05))
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=10.5)
    for tick, ci in zip(ax.get_yticklabels(), order):
        tick.set_color(_shaft_color(names[ci], shafts))
    ax.set_xlabel("Simulation time (ms)", fontsize=12.5)
    ax.set_ylabel("Virtual-SEEG (30–80 Hz)", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11.0, length=3.2, color="0.35")
    ax.tick_params(axis="y", labelsize=10.5, length=2.5, color="0.35")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.35")
        ax.spines[side].set_linewidth(0.8)
    ax.legend(
        handles=[
            Patch(facecolor=FWD_SHADE, alpha=0.24, edgecolor="none", label="model forward event"),
            Patch(facecolor=REV_SHADE, alpha=0.24, edgecolor="none", label="model reverse event"),
        ],
        frameon=False,
        fontsize=10.8,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.105),
        borderaxespad=0.0,
        handlelength=1.6,
        ncol=2,
    )
    forward_events = int(sum(float(event["sign"]) > 0.0 for event in displayed))
    reverse_events = int(sum(float(event["sign"]) < 0.0 for event in displayed))
    return {
        "forward_events": forward_events,
        "reverse_events": reverse_events,
        "selected_pair": [
            {"t_on_ms": float(event["t_on"]), "t_off_ms": float(event["t_off"]),
             "type": "TA" if float(event["sign"]) > 0.0 else "TB"}
            for event in pair
        ],
        "display_window_t0_ms": start,
        "display_window_t1_ms": end,
        "trace_band_hz": list(TRACE_BAND_HZ),
        "trace_scaling": f"per-contact signed component / window absolute p{TRACE_SCALE_PERCENTILE:g}",
    }


def compose(
    twoend_tag,
    source_tag,
    sink_tag,
    fig_name,
    subject_label,
    readout_window_ms=5000.0,
    *,
    output_stem=None,
    panel_letter="A",
    formal_layout=False,
):
    to, tfd = _load(twoend_tag)            # twoend (spontaneous) -> mechanism + readout
    # single-run mode: if no dedicated source/sink runs, take the tempA/tempB event panels
    # from the twoend run's own best forward / best reverse spontaneous events (rep_fwd/rep_rev).
    single = not (source_tag and sink_tag)
    if single:
        so, sfd, ko, kfd = to, tfd, to, tfd
    else:
        so, sfd = _load(source_tag)        # source-core only -> clean tempA event panel
        ko, kfd = _load(sink_tag)          # sink-core only   -> clean tempB event panel
    reg = tfd["reg"].item(); core_a = list(reg["source_names"]); core_b = list(reg["sink_names"])
    names = [str(x) for x in tfd["names"]]; shafts = sorted(set(_shaft(n) for n in names))
    display = _registered_axis_display(tfd) if formal_layout else None
    if display is None:
        u, p = _axis(float(tfd["theta_deg"]))
        order_contacts = np.asarray(tfd["contacts"], float)
    else:
        u, p = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        order_contacts = _display_xy(np.asarray(tfd["contacts"], float), display)
    order = _active_order(names, order_contacts, u, p, [to["events"]])
    rep_s = sfd["rep_fwd"].item() or sfd["rep_rev"].item()
    rep_k = kfd["rep_rev"].item() or kfd["rep_fwd"].item()

    if formal_layout:
        fig_size = (19.2, 4.75)
        top, bottom = 0.88, 0.15
        titles = ("model forward", "model reverse")
    else:
        fig_size = (18.0, 4.45)
        width_ratios = [1.0, 1.0, 1.0, 2.75]
        top, bottom, wspace = 0.86, 0.16, 0.16
        titles = ("mechanism", "tempA source", "tempB source")
    fig = plt.figure(figsize=fig_size, facecolor="white")
    if formal_layout:
        # Panel B contains only the two event maps plus the continuous SEEG.
        # Keep the maps compact and reserve just enough gutter for their
        # colorbar and the readout y-label.
        outer = gridspec.GridSpec(
            1, 2, width_ratios=[2.0, 2.80],
            left=0.045, right=0.992, bottom=bottom, top=top, wspace=0.14,
        )
        spatial = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, 0], wspace=0.045,
        )
        ax_setup = None
        ax_fwd = fig.add_subplot(spatial[0, 0])
        ax_rev = fig.add_subplot(spatial[0, 1])
        ax_readout = fig.add_subplot(outer[0, 1])
    else:
        gs = gridspec.GridSpec(
            1, 4, width_ratios=width_ratios,
            left=0.045, right=0.992, bottom=bottom, top=top, wspace=wspace,
        )
        ax_setup = fig.add_subplot(gs[0, 0])
        ax_fwd = fig.add_subplot(gs[0, 1])
        ax_rev = fig.add_subplot(gs[0, 2])
        ax_readout = fig.add_subplot(gs[0, 3])
    if not formal_layout:
        _plot_mechanism(
            ax_setup, tfd, core_a, core_b, shafts, title=titles[0], display=display,
            formal=False,
        )
    _plot_event(
        ax_fwd,
        sfd,
        rep_s,
        titles[0] if formal_layout else titles[1],
        shafts,
        core_a,
        core_b,
        source_index=0,
        normalize_color=formal_layout,
        display=display,
        formal=formal_layout,
        show_ylabel=formal_layout,
    )
    event_mappable = _plot_event(
        ax_rev,
        kfd,
        rep_k,
        titles[1] if formal_layout else titles[2],
        shafts,
        core_a,
        core_b,
        source_index=1,
        normalize_color=formal_layout,
        display=display,
        formal=formal_layout,
        show_ylabel=False,
    )
    if formal_layout:
        # Tight grouping makes adjacent +10/-10 boundary labels redundant and
        # visually colliding.  Keep the ticks, but suppress the repeated right
        # edge label on the first two spatial panels.
        for spatial_ax in (ax_fwd,):
            right_labels = spatial_ax.get_xticklabels()
            if right_labels:
                right_labels[-1].set_visible(False)
    if formal_layout and event_mappable is not None:
        cax = ax_rev.inset_axes([1.035, 0.0, 0.052, 1.0])
        cb = fig.colorbar(event_mappable, cax=cax)
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["early", "late"])
        cb.ax.set_title("relative\nfiring onset", fontsize=10.0, pad=5.0)
        cb.ax.tick_params(labelsize=9.5, length=2.8)
    # driven-pooled runs supply time-adjusted readout_window_events matching the concatenated trace;
    # spontaneous runs fall back to the run's own events.
    readout_evs = to.get("readout_window_events", to["events"])
    if formal_layout:
        stats = _plot_interictal_sample_readout(
            ax_readout,
            tfd,
            readout_evs,
            names,
            shafts,
            window_ms=min(float(readout_window_ms), 1200.0),
        )
    else:
        stats = _plot_readout(
            ax_readout,
            tfd,
            readout_evs,
            order,
            names,
            shafts,
            window_ms=float(readout_window_ms),
            reader_labels=formal_layout,
            formal=formal_layout,
        )
    if formal_layout:
        # Force the continuous trace block to the same vertical box as the
        # equal-aspect spatial panels; the inset colorbar follows ax_rev exactly.
        fig.canvas.draw()
        spatial_box = ax_rev.get_position()
        readout_box = ax_readout.get_position()
        ax_readout.set_position(
            [readout_box.x0, spatial_box.y0, readout_box.width, spatial_box.height]
        )
    if panel_letter:
        fig.text(0.012, 0.93, str(panel_letter), fontsize=22, fontweight="bold")

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or fig_name
    png = outdir / f"{stem}.png"; pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.0, facecolor="white"); plt.close(fig)
    meta = dict(figure=stem, figure_group=fig_name, subject=subject_label,
                twoend_tag=twoend_tag, source_tag=source_tag, sink_tag=sink_tag,
                placement=(
                    "gradient_shared coordinates + frozen template_source core membership"
                    if reg.get("coordinate_frame") == "gradient_shared"
                    else "template_source (earliest-3 of each template = the two template sources)"
                ),
                source_core=core_a, sink_core=core_b,
                twoend_spontaneous_dir=f"{to['dir_forward']}/{to['dir_reverse']}",
                source_only_dir=(None if single else f"{so['dir_forward']}/{so['dir_reverse']}"),
                sink_only_dir=(None if single else f"{ko['dir_forward']}/{ko['dir_reverse']}"),
                single_run_mode=single,
                panel_letter=panel_letter,
                formal_layout=bool(formal_layout),
                registered_axis_display=(None if display is None else {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in display.items()
                }),
                event_color_scale=("within-event normalized early-to-late" if formal_layout else "relative onset time (ms)"),
                readout_window_ms=(
                    float(stats["display_window_t1_ms"] - stats["display_window_t0_ms"])
                    if formal_layout else float(readout_window_ms)
                ),
                readout_events=stats,
                notes=["Plotting-only; consumes the completed SNN artifact named above.",
                       (("Single-run formal mode: forward/reverse event maps and continuous readout all come from "
                          "one spontaneous twoend run; the separate substrate/mechanism map is omitted.")
                        if single and formal_layout else
                        ("Single-run mode: mechanism + readout + tempA/tempB event panels ALL from the one "
                         "spontaneous twoend run (tempA=best forward event, tempB=best reverse event).") if single
                        else "Mechanism + readout = spontaneous twoend run; tempA/tempB panels = source-only/sink-only runs."),
                       "k_dir=2 sparse-electrode readout; plane-fit registration (no core-anchoring).",
                       (("Formal layout uses the frozen template-gradient shared-plane orientation; "
                         "the display applies only a sheet-centering translation and no post-hoc rotation.")
                        if formal_layout and reg.get("coordinate_frame") == "gradient_shared"
                        else ("Formal layout uses the simulation's stored registered TA-axis frame; no post-hoc fit/rotation "
                              "onto a second patient-plane rendering." if formal_layout else "Native SNN sheet display.")),
                       ("Formal readout is one continuous interictal-only 30-80 Hz window; no ictal/runaway marker."
                        if formal_layout else "Diagnostic readout shows the requested full event train.")])
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {png}\nwrote {pdf}\nreadout {stats}")
    return outdir, meta


def main():
    os.chdir(ROOT)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--twoend-tag", default="epilepsiae_1146_twoend_equal_tsrc_s3")
    ap.add_argument("--source-tag", default=None, help="omit for single-run mode (tempA from twoend rep_fwd)")
    ap.add_argument("--sink-tag", default=None, help="omit for single-run mode (tempB from twoend rep_rev)")
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146")
    ap.add_argument("--label", default="epilepsiae_1146 (ICL strip)")
    ap.add_argument("--readout-window-ms", type=float, default=5000.0)
    ap.add_argument("--output-stem", default=None)
    ap.add_argument("--panel-letter", default="A")
    ap.add_argument("--formal-layout", action="store_true")
    a = ap.parse_args()
    compose(
        a.twoend_tag,
        a.source_tag,
        a.sink_tag,
        a.fig_name,
        a.label,
        a.readout_window_ms,
        output_stem=a.output_stem,
        panel_letter=a.panel_letter,
        formal_layout=a.formal_layout,
    )


if __name__ == "__main__":
    main()
