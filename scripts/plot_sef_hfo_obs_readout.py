"""Diagnostic figure for ANY "SEEG read-out" claim (spec 2026-06-06 §13.2-13.3, user 2026-06-07).

Every figure that claims to read direction/template through virtual SEEG MUST make the chain
visible + diagnosable, so a failure is attributable to model vs estimator vs electrode placement:

  Panel A (geometry overlay): model plane + event footprint + KICK point + imposed θ_EE arrow +
           RECOVERED axis arrow + shaft/contact positions; annotates contact spacing + sheet size.
  Panel B (per-contact read-out): each contact's activity TIME TRACE, offset by contact, labeled
           by shaft; the event window shaded.
  Panel C (derived chain): per-contact first-crossing LAG vs position-along-recovered-axis (the
           model-event -> electrode-observation -> direction-estimate chain, made explicit).

Reusable: pass a source frame (rate field or binned spikes) on grid_xy for the footprint, the
VirtualMontage, the per-contact envelopes + frame_dt, the LagPatArtifact, kick xy, θ_EE, recovered
axis, sheet L, contact spacing. Works for the rate field (cm-scale) and the SNN alike.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_readout_diagnostic(out_path, montage, envelopes, frame_dt, artifact,
                            kick_xy, theta_EE_deg, recovered_axis, sheet_L,
                            contact_spacing, source_frame=None, grid_xy=None,
                            event_window=None, title=""):
    """Render the 3-panel SEEG-read-out diagnostic. recovered_axis = unit 2-vec or None.
    source_frame (n_pix,) + grid_xy (n_pix,2) optional (event-footprint background)."""
    contacts = np.asarray(montage.contacts, float)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Panel A: geometry overlay (the model plane + electrodes + directions) ----
    a = ax[0]
    if source_frame is not None and grid_xy is not None:
        gx = np.asarray(grid_xy, float)
        sc = a.scatter(gx[:, 0], gx[:, 1], c=np.asarray(source_frame, float),
                       s=6, cmap="Greys", alpha=0.6)
        fig.colorbar(sc, ax=a, fraction=0.046, label="event footprint")
    part = artifact.bools[:, 0] if artifact is not None else np.ones(len(contacts), bool)
    a.scatter(contacts[~part, 0], contacts[~part, 1], c="lightgray", s=70, edgecolor="k",
              label="contact (no part.)", zorder=3)
    a.scatter(contacts[part, 0], contacts[part, 1], c="tab:blue", s=70, edgecolor="k",
              label="contact (participating)", zorder=3)
    for i, c in enumerate(contacts):
        a.annotate(montage.names[i], c, fontsize=6, ha="center", va="center", zorder=4)
    if kick_xy is not None:
        a.scatter([kick_xy[0]], [kick_xy[1]], marker="*", s=260, c="tab:red",
                  edgecolor="k", label="kick", zorder=5)
    cen = contacts.mean(0)
    L = 0.4 * sheet_L
    th = np.deg2rad(theta_EE_deg)
    a.annotate("", xy=cen + L * np.array([np.cos(th), np.sin(th)]), xytext=cen,
               arrowprops=dict(arrowstyle="->", color="tab:green", lw=2.5))
    a.text(*(cen + L * np.array([np.cos(th), np.sin(th)])), f" θ_EE={theta_EE_deg:g}°",
           color="tab:green", fontsize=9)
    if recovered_axis is not None:
        rv = np.asarray(recovered_axis, float)
        a.annotate("", xy=cen + 0.9 * L * rv, xytext=cen - 0.9 * L * rv,
                   arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=2.5, ls="--"))
        a.text(*(cen + 0.9 * L * rv), " recovered axis", color="tab:orange", fontsize=9)
    # data-driven limits (works for centered [-L/2,L/2] and [0,L] frames alike)
    if grid_xy is not None:
        pts = np.vstack([np.asarray(grid_xy, float), contacts])
    else:
        pts = contacts
    pad = 0.05 * (pts.max(0) - pts.min(0) + 1e-9)
    a.set_xlim(pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    a.set_ylim(pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])
    a.set_aspect("equal")
    a.set_title(f"A. geometry  (sheet {sheet_L:g} mm, contact spacing {contact_spacing:g} mm)")
    a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    a.legend(fontsize=7, loc="upper right")

    # ---- Panel B: per-contact time traces ----
    b = ax[1]
    env = np.asarray(envelopes, float)
    t = np.arange(env.shape[1]) * frame_dt
    off = 1.15 * (np.nanmax(env) - np.nanmin(env) + 1e-9)
    for i in range(env.shape[0]):
        b.plot(t, env[i] + i * off, lw=0.9)
        b.text(t[-1], i * off, f" {montage.names[i]}", fontsize=6, va="center")
    if event_window is not None:
        b.axvspan(event_window[0], event_window[1], color="tab:red", alpha=0.08)
    b.set_title("B. per-contact activity traces (offset; event window shaded)")
    b.set_xlabel("time (ms)"); b.set_yticks([])

    # ---- Panel C: derived chain (lag vs projection on recovered axis) ----
    c = ax[2]
    if artifact is not None and recovered_axis is not None:
        lag = artifact.lag_raw[:, 0]
        m = part & np.isfinite(lag)
        proj = (contacts[m] - cen) @ np.asarray(recovered_axis, float)
        c.scatter(proj, lag[m], c="tab:blue", s=60, edgecolor="k")
        for j, idx in enumerate(np.flatnonzero(m)):
            c.annotate(montage.names[idx], (proj[j], lag[m][j]), fontsize=6)
        c.set_xlabel("position along recovered axis (mm)")
        c.set_ylabel("first-crossing lag (ms)")
        c.set_title("C. model event → electrode lag → direction")
    else:
        c.text(0.5, 0.5, "no readable axis\n(degenerate / <7 contacts)", ha="center",
               va="center", transform=c.transAxes)
        c.set_title("C. (no axis)")

    fig.suptitle(title or "virtual-SEEG read-out diagnostic", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _demo():
    """Render on a synthetic toy wave (validates the tool; not real model output)."""
    import os
    from src.sef_hfo_toywave import traveling_wave
    from src.sef_hfo_observation import (build_shaft, merge_montages, sample_envelopes,
                                         extract_lagpat, attach_geometry, endpoint_centroid_axis)
    L = 64.0
    src = traveling_wave(64, L, np.deg2rad(30.0), c=0.4, dt=0.25, t_max=200.0, width=8.0)
    spacing = 3.5
    # toy-wave grid is CENTERED at (0,0) -> montage at (0,0) so the wave sweeps it
    m = merge_montages([build_shaft(np.deg2rad(10.0), spacing, 8, (0.0, 0.0), "A"),
                        build_shaft(np.deg2rad(100.0), spacing, 8, (0.0, 0.0), "B")])
    env = sample_envelopes(src["frames"], src["grid_xy"], m, kernel_width=3.0)
    art = extract_lagpat(env, src["dt"], [src["window"]], float(env.min()),
                         0.5 * (float(env.max()) - float(env.min())), 0.5, src["dt"])
    art = attach_geometry(art, m)
    axis = endpoint_centroid_axis(art.ranks[:, 0], art.bools[:, 0], art.contact_coords,
                                  k_dir=3, eps_deg=0.5 * spacing)
    foot = src["frames"][int(0.5 * src["frames"].shape[0])]
    out = "results/topic4_sef_hfo/observation_layer/figures"
    os.makedirs(out, exist_ok=True)
    nhat = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
    kick_xy = -14.0 * nhat        # wave-entry side (low-s end of the 30° axis), grid centered at 0
    p = plot_readout_diagnostic(os.path.join(out, "readout_diagnostic_demo.png"), m, env,
                                src["dt"], art, kick_xy, 30.0, axis, L, spacing,
                                source_frame=foot, grid_xy=src["grid_xy"],
                                event_window=src["window"],
                                title="DEMO (synthetic toy wave, NOT a model read; firing-density envelope, NOT LFP)")
    print("wrote", p)


if __name__ == "__main__":
    _demo()
