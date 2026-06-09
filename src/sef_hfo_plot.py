"""Shared paper-grade plotting for the SEF-HFO model section (house style).

Consolidates the spatial-substrate scatter (lifted from the spiking-validation
figure, scripts/plot_sef_hfo_spiking_validation.py) plus the canonical
TWO-ELECTRODE read-out figure used for BOTH the LIF rate field and the spiking
network, so every "virtual electrode reads the propagation direction" figure
reuses ONE styled function instead of a per-figure throwaway plotter
(user 2026-06-07; supersedes the tab-color plot_readout_diagnostic role).

Panel B/C show THE RECORDED SIGNAL each contact sees over time (per-contact
trace, offset, event window shaded, peak-time locus), in the style of
inc2_cm_currentLFP.png — an electrode laid ALONG the connectivity axis is crossed
in sequence (the per-contact peaks sweep → a slanted peak locus = reads
direction), one laid ACROSS is crossed together (peaks aligned → vertical locus).
The signal is substrate-specific (rate field = firing-rate-density envelope, NOT
a current; SNN = formal current-based LFP), passed in by the caller.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from src.plot_style import (  # noqa: E402
    style_panel, COL_CLUSTER_T0, COL_CLUSTER_T1, FS_LABEL, FS_TICK, FS_TITLE,
)

C_E = COL_CLUSTER_T1       # excitatory substrate — red (matches spiking-validation fig)
C_I = COL_CLUSTER_T0       # inhibitory substrate — blue
C_PAR = "#D2691E"          # electrode ALONG the axis — vivid rust, circle marker
C_PERP = "#2F5D62"         # electrode ACROSS the axis — deep teal, square marker
C_AXIS = "#2c2c2c"         # imposed connectivity axis — near-black
C_FOOT = "#7E6E84"         # activated footprint (single-color fallback) — Morandi plum
FIELD_CMAP = "viridis"
LEG_FS = 10


def draw_substrate(ax, E_xy=None, I_xy=None, alpha_E=0.10, alpha_I=0.12):
    """Faint E (red) / I (blue) neuron sheet for spatial context (SNN only).

    Lifted from scripts/plot_sef_hfo_spiking_validation.py::_substrate so the
    model section shares one substrate look."""
    if I_xy is not None:
        ax.scatter(I_xy[:, 0], I_xy[:, 1], s=1.2, c=C_I, alpha=alpha_I, linewidths=0)
    if E_xy is not None:
        ax.scatter(E_xy[:, 0], E_xy[:, 1], s=1.0, c=C_E, alpha=alpha_E, linewidths=0)


def _draw_electrode(ax, contacts, part, color, marker, label, names=None, name_dxy=(0, 9),
                    name_fs=9, endpoints_only=False, contact_c=None, ccmap=None, cnorm=None):
    """One shaft: a thin guide line through the contacts, filled markers for
    participating contacts, hollow gray for non-participating, per-contact name
    labels (so panel A contacts match the trace labels in panels B/C). Labels are
    drawn large with a white-stroke halo so they read over the dense field. When
    the montage is too dense to label every contact (crossing sub-mm shafts whose
    centre contacts coincide), endpoints_only labels just the two shaft ends — the
    intermediate contacts follow shaft order, matching the B/C trace stack.

    contact_c (+ ccmap/cnorm): colour the participating markers by a per-contact
    value (e.g. arrival time) on a shared scale — so the propagation order is
    visible ON the electrodes, not just the field (user 2026-06-08c). The shaft
    `color` then only outlines the guide line / non-participants / labels."""
    c = np.asarray(contacts, float)
    ax.plot(c[:, 0], c[:, 1], color=color, lw=1.0, alpha=0.6, zorder=3)
    part = np.asarray(part, bool)
    if (~part).any():
        ax.scatter(c[~part, 0], c[~part, 1], s=42, facecolor="white",
                   edgecolor="0.5", marker=marker, linewidths=1.0, zorder=4)
    if contact_c is not None and ccmap is not None and cnorm is not None and part.any():
        fc = ccmap(cnorm(np.asarray(contact_c, float)[part]))
    else:
        fc = color
    ax.scatter(c[part, 0], c[part, 1], s=66, facecolor=fc, edgecolor="black",
               marker=marker, linewidths=1.1, zorder=5, label=label)
    if names is not None:
        idxs = [0, len(names) - 1] if endpoints_only else range(len(names))
        for j in idxs:
            ax.annotate(names[j], c[j], fontsize=name_fs, color=color, fontweight="bold",
                        zorder=8, ha="center", va="center", xytext=name_dxy,
                        textcoords="offset points",
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def _axis_arrow(ax, kick_xy, axis_deg, extent):
    """Imposed connectivity axis: a short arrow through the seed (no text label —
    the elongated activated band + the arrow already carry the axis; user 2026-06-08)."""
    th = np.deg2rad(axis_deg)
    u = np.array([np.cos(th), np.sin(th)])
    span = 0.30 * (extent[1] - extent[0])
    k = np.asarray(kick_xy, float)
    ax.annotate("", xy=k + span * u, xytext=k - 0.18 * span * u,
                arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=2.2), zorder=6)


def _trace_panel(ax, e, t, event_window, cmap_name, panel_title, signal_ylabel,
                 show_ylabel=True):
    """Stacked per-contact recorded-signal traces, ordered by position along the
    electrode. Each trace baseline-subtracted + peak-normalised (so timing, not
    amplitude, is what's compared); event window shaded; the per-contact peak time
    marked + connected into a 'peak locus' (slanted = sweep = direction; vertical
    = simultaneous = none)."""
    sig = np.asarray(e["signal"], float)                 # (k, nt)
    s = np.asarray(e["s"], float)
    part = np.asarray(e["part"], bool)
    names = list(e["names"]) if e.get("names") is not None else None
    order = np.argsort(s)
    sig, part = sig[order], part[order]
    if names is not None:
        names = [names[j] for j in order]
    k, nt = sig.shape
    ev0, ev1 = event_window
    pre = t < ev0

    # per-contact peak = TRUE arrival over the FULL trace. The detected field-event
    # window (bulk supra-threshold pixel fraction) can be NARROWER than the time the
    # travelling wave takes to transit the whole shaft, so restricting argmax to it
    # clamps the first/last-crossed contacts to the window edge and fakes the locus
    # tails (user 2026-06-08: "why are peaks only in the gray?").
    peak_idx = np.array([int(np.argmax(sig[i])) for i in range(k)])
    peak_t = t[peak_idx]

    base = sig[:, pre].mean(axis=1) if pre.any() else sig[:, :max(1, nt // 10)].mean(axis=1)
    ev_peak = sig.max(axis=1)
    glob = np.median((ev_peak - base)[part]) if part.any() else 1.0
    denom = np.maximum(ev_peak - base, 0.25 * max(glob, 1e-9))
    z = (sig - base[:, None]) / denom[:, None]           # participating peaks ~1, baseline ~0
    off = 1.35

    # shade = the event AS THIS ELECTRODE RECORDS IT: first to last participating-contact
    # peak (union with the detected field window) -> no peak clamped, slant is pure propagation
    pk = peak_t[part] if part.any() else peak_t
    shade0 = min(ev0, float(pk.min())); shade1 = max(ev1, float(pk.max()))
    span = max(shade1 - shade0, 1e-6)
    xr = min(shade1 + 0.40 * span, float(t.max()))       # right edge for trace name labels

    ax.axvspan(shade0, shade1, color="0.86", alpha=0.6, lw=0, zorder=0)
    cmap = plt.get_cmap(cmap_name)
    locus_t, locus_y = [], []
    for i in range(k):
        col = cmap(0.12 + 0.76 * (i / max(k - 1, 1)))
        ax.plot(t, z[i] + i * off, color=col, lw=1.0, alpha=(0.95 if part[i] else 0.30),
                zorder=2)
        if names is not None:
            ax.text(xr, i * off + 0.15, f" {names[i]}", fontsize=7,
                    fontweight=("bold" if part[i] else "normal"), va="center", ha="left",
                    color=col, alpha=(0.95 if part[i] else 0.4), zorder=6)
        if part[i]:
            py = z[i][peak_idx[i]] + i * off
            ax.plot([peak_t[i]], [py], marker="o", ms=4.5, mfc="black", mec="white",
                    mew=0.6, zorder=5)
            locus_t.append(peak_t[i]); locus_y.append(py)
    if len(locus_t) >= 2:
        ax.plot(locus_t, locus_y, color="black", lw=1.3, ls="--", alpha=0.8, zorder=4,
                label="per-contact peak")

    ax.set_xlim(max(shade0 - 0.16 * span, float(t.min())), xr + 0.16 * span)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)", fontsize=FS_LABEL)
    if show_ylabel:
        ax.set_ylabel(f"contact (stacked along shaft)\n{signal_ylabel}", fontsize=FS_LABEL - 2)
    ax.set_title(panel_title, fontsize=FS_TITLE - 3, pad=6)


def _spatial_panel(fig, axA, cax, *, field_xy, kick_xy, axis_deg, extent, par, perp,
                   field_c=None, field_clabel=None, field_cmap=None, field_vlim=None,
                   color_contacts=False, E_xy=None, I_xy=None, name_fs=10,
                   label_endpoints_only=False, patch_circle=None, substrate_label="",
                   panel_letter="a", label_x=-0.20, show_legend=True):
    """One spatial event+electrodes panel (field scatter + colorbar + electrodes +
    seed + axis arrow + optional pathology-core outline). Shared by the 3-panel
    read-out and the 4-panel mechanism figure so both draw the map identically."""
    draw_substrate(axA, E_xy, I_xy)
    fxy = np.asarray(field_xy, float)
    _cmap_name = field_cmap or FIELD_CMAP
    _ccmap = _cnorm = None
    if field_c is not None:
        fc = np.asarray(field_c, float)
        fin = np.isfinite(fc)
        _vmin = field_vlim[0] if field_vlim else float(np.nanmin(fc[fin]))
        _vmax = field_vlim[1] if field_vlim else float(np.nanmax(fc[fin]))
        sc = axA.scatter(fxy[fin, 0], fxy[fin, 1], c=fc[fin], cmap=_cmap_name,
                         vmin=_vmin, vmax=_vmax, s=8, linewidths=0, alpha=0.9, zorder=1)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(field_clabel or "", fontsize=FS_LABEL - 3)
        cb.ax.tick_params(labelsize=FS_TICK - 4)
        if color_contacts:
            _ccmap = plt.get_cmap(_cmap_name)
            _cnorm = plt.Normalize(vmin=_vmin, vmax=_vmax)
    else:
        cax.set_visible(False)
        axA.scatter(fxy[:, 0], fxy[:, 1], c=C_FOOT, s=7, linewidths=0, alpha=0.45, zorder=1)

    if patch_circle is not None:
        px, py, pr = patch_circle
        axA.add_patch(plt.Circle((px, py), pr, fill=False, ls="--", ec="crimson",
                                 lw=1.6, zorder=4, label="pathology core"))

    _axis_arrow(axA, kick_xy, axis_deg, extent)
    _cc_par = par.get("contact_c") if color_contacts else None
    _cc_perp = perp.get("contact_c") if color_contacts else None
    _draw_electrode(axA, par["contacts"], par["part"], C_PAR, "o", "electrode ∥ axis",
                    names=par.get("names"), name_dxy=(7, 7), name_fs=name_fs,
                    endpoints_only=label_endpoints_only,
                    contact_c=_cc_par, ccmap=_ccmap, cnorm=_cnorm)
    _draw_electrode(axA, perp["contacts"], perp["part"], C_PERP, "s", "electrode ⊥ axis",
                    names=perp.get("names"), name_dxy=(-7, -9), name_fs=name_fs,
                    endpoints_only=label_endpoints_only,
                    contact_c=_cc_perp, ccmap=_ccmap, cnorm=_cnorm)
    k = np.asarray(kick_xy, float)
    axA.scatter([k[0]], [k[1]], marker="*", s=280, c="black", edgecolor="white",
                linewidths=1.0, zorder=7, label="event seed")

    axA.set_xlim(extent[0], extent[1]); axA.set_ylim(extent[2], extent[3])
    axA.set_aspect("equal")
    axA.set_xlabel("x (mm)", fontsize=FS_LABEL); axA.set_ylabel("y (mm)", fontsize=FS_LABEL)
    axA.set_title(substrate_label, fontsize=FS_TITLE - 3, pad=6)
    style_panel(axA, panel_letter, label_x=label_x, label_y=1.02)
    if show_legend:
        axA.legend(fontsize=LEG_FS - 1, loc="upper left", framealpha=0.92)


def two_electrode_readout(
    out_path, *,
    field_xy, kick_xy, axis_deg, extent,     # panel A: points; seed; axis deg; (xmin,xmax,ymin,ymax)
    par, perp,                               # dict(contacts,part,s, signal(k,nt), label, panel_title)
    t, event_window, signal_ylabel,          # shared time vector (ms); event window (t0,t1); signal name
    substrate_label, contact_note,           # panel-A title; spacing caption
    field_c=None, field_clabel=None,         # color panel-A points (rate=onset / SNN=density); None=single
    E_xy=None, I_xy=None,                    # faint substrate (SNN); None for rate field
    name_fs=10,                              # panel-A contact-label font (smaller for dense montages)
    label_endpoints_only=False,              # dense crossing montage: label only the shaft ends
    patch_circle=None,                       # (x,y,r) mm: dashed outline of a pathology core on panel A
    field_cmap=None,                         # panel-A field colormap (default viridis; e.g. 'plasma' for spread)
    field_vlim=None,                         # (vmin,vmax) clip for panel-A field colour (e.g. 5–95th pct so outliers don't blow out the gradient)
    color_contacts=False,                    # colour electrode markers by par/perp 'contact_c' on the field scale
    title=None,
):
    """3-panel two-electrode read-out (A geometry | B ∥-traces | C ⊥-traces),
    house style, manual fixed-fraction axes (robust to square-box + colorbar
    collapse)."""
    fig = plt.figure(figsize=(17.0, 5.6))
    fig.patch.set_facecolor("white")
    axA = fig.add_axes([0.040, 0.150, 0.250, 0.700])    # square (0.250*17 ≈ 0.700*5.6 ... ≈4.0)
    cax = fig.add_axes([0.296, 0.150, 0.011, 0.700])
    axB = fig.add_axes([0.405, 0.150, 0.255, 0.700])
    axC = fig.add_axes([0.715, 0.150, 0.255, 0.700])

    # ---------------- Panel A: spatial event + electrodes ----------------
    _spatial_panel(fig, axA, cax, field_xy=field_xy, kick_xy=kick_xy, axis_deg=axis_deg,
                   extent=extent, par=par, perp=perp, field_c=field_c,
                   field_clabel=field_clabel, field_cmap=field_cmap, field_vlim=field_vlim,
                   color_contacts=color_contacts, E_xy=E_xy, I_xy=I_xy, name_fs=name_fs,
                   label_endpoints_only=label_endpoints_only, patch_circle=patch_circle,
                   substrate_label=substrate_label, panel_letter="a", label_x=-0.20)

    # ---------------- Panel B/C: per-contact recorded signal ----------------
    _trace_panel(axB, par, t, event_window, "Oranges", par["panel_title"], signal_ylabel,
                 show_ylabel=True)
    _trace_panel(axC, perp, t, event_window, "GnBu", perp["panel_title"], signal_ylabel,
                 show_ylabel=False)
    style_panel(axB, "b", label_x=-0.10, label_y=1.02)
    style_panel(axC, "c", label_x=-0.08, label_y=1.02)
    axB.legend(fontsize=LEG_FS - 1, loc="upper right", framealpha=0.9)
    fig.text(0.5, 0.018, contact_note, ha="center", va="bottom",
             fontsize=FS_TICK - 4, color="0.35")

    if title:
        fig.suptitle(title, fontsize=FS_TITLE, y=0.99)
    fig.savefig(out_path, dpi=300, facecolor="white")
    fig.savefig(out_path.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def mechanism_4panel(
    out_path, *,
    field_xy, kick_xy, axis_deg, extent,
    map_a, map_b,                            # each: dict(field_c, clabel, cmap, vlim, color_contacts, title)
    par, perp, t, event_window, signal_ylabel, contact_note,
    E_xy=None, I_xy=None, name_fs=10, label_endpoints_only=False,
    patch_circle=None, title=None,
):
    """4-panel mechanism figure: (a) heterogeneity map | (b) onset/propagation map |
    (c) ∥-axis read-out | (d) ⊥-axis read-out. The two maps share the same geometry
    (electrodes / seed / axis / core); the read-out (c/d) is SHARED, not duplicated
    across two figures (CLAUDE.md §7)."""
    fig = plt.figure(figsize=(20.5, 5.6))
    fig.patch.set_facecolor("white")
    axA = fig.add_axes([0.030, 0.150, 0.185, 0.700]); caxA = fig.add_axes([0.220, 0.150, 0.008, 0.700])
    axB = fig.add_axes([0.285, 0.150, 0.185, 0.700]); caxB = fig.add_axes([0.475, 0.150, 0.008, 0.700])
    axC = fig.add_axes([0.560, 0.150, 0.185, 0.700])
    axD = fig.add_axes([0.795, 0.150, 0.185, 0.700])

    def _map(ax, cax, m, letter, label_x, show_legend):
        _spatial_panel(fig, ax, cax, field_xy=field_xy, kick_xy=kick_xy, axis_deg=axis_deg,
                       extent=extent, par=par, perp=perp, field_c=m.get("field_c"),
                       field_clabel=m.get("clabel"), field_cmap=m.get("cmap"),
                       field_vlim=m.get("vlim"), color_contacts=m.get("color_contacts", False),
                       E_xy=E_xy, I_xy=I_xy, name_fs=name_fs,
                       label_endpoints_only=label_endpoints_only, patch_circle=patch_circle,
                       substrate_label=m.get("title", ""), panel_letter=letter,
                       label_x=label_x, show_legend=show_legend)

    _map(axA, caxA, map_a, "a", -0.22, True)
    _map(axB, caxB, map_b, "b", -0.18, False)
    _trace_panel(axC, par, t, event_window, "Oranges", par["panel_title"], signal_ylabel,
                 show_ylabel=True)
    _trace_panel(axD, perp, t, event_window, "GnBu", perp["panel_title"], signal_ylabel,
                 show_ylabel=False)
    style_panel(axC, "c", label_x=-0.10, label_y=1.02)
    style_panel(axD, "d", label_x=-0.08, label_y=1.02)
    axC.legend(fontsize=LEG_FS - 1, loc="upper right", framealpha=0.9)
    fig.text(0.5, 0.018, contact_note, ha="center", va="bottom",
             fontsize=FS_TICK - 4, color="0.35")
    if title:
        fig.suptitle(title, fontsize=FS_TITLE, y=0.99)
    # dpi 150 keeps each ~16k-neuron 4-panel inspection figure ~1MB (vs ~4MB at 300);
    # one per (kick×core×cond) combination is committed, so the footprint matters. No
    # PDF (the vector copy of a 16k-point scatter is large and was never committed).
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path
