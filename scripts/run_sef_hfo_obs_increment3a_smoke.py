"""Rate-field θ=45° smoke — Increment-3a Task 4 (Increment-3 plan 2026-06-07).

Validates the full endpoint_centroid_axis read-out chain on the LIF rate field:
  pulse_stim_fn → integrate_lif_field(return_frames) → rate_event_envelope →
  extract_lagpat → endpoint_centroid_axis → plot_readout_diagnostic

Locked parameters (do NOT tune until the smoke passes, then freeze):
  AXIS_ERR_MAX = 25°   (direction accuracy gate)
  KDIR         = 3     (endpoint centroid contacts per side)
  PART_MIN     = 7     (minimum participating contacts)
  RATIO        = 0.6   (interictal drive — Step-0b self_limited_propagation)
  THETA_EE     = 45°   (connectivity orientation, in degrees)
  L, N         = 12mm, 96  (default sheet; event advances ~6.9mm, spans ~11mm)
  STIM_R       = 2.0mm, STIM_DIST = 3.0mm  (kick at θ_EE-end, 3mm from center)
  SPACING      = 3.0mm, NC = 5 per shaft (10 total contacts)
  SHAFT_LAT_OFFSET = 2.0mm (lateral offset ±2mm from the θ_EE axis)

Montage design rationale: the event footprint is a narrow stripe (~3-4mm wide) along θ_EE.
Oblique shafts (e.g. 10° and 100°) cross the stripe for only ~2 contacts each and have
endpoint contacts outside the L=12mm grid (wrap artifact). Fix: both shafts run PARALLEL
to θ_EE=45°, laterally offset ±2mm (perpendicular direction). All 10 contacts stay within
the sheet AND within the event stripe → ≥7 participate. The estimator is endpoint_centroid_axis
(sparse-friendly; shafts need not be perpendicular to θ_EE).
"""
import os
import numpy as np

from src.sef_hfo_lif import mean_field, integrate_lif_field, _DEFAULT_N, _DEFAULT_L
from src.sef_hfo_rate_adapter import pulse_stim_fn, rate_event_envelope
from src.sef_hfo_observation import (
    build_shaft, merge_montages, grid_coords,
    extract_lagpat, attach_geometry, endpoint_centroid_axis, axis_angle_error_deg,
)
from scripts.plot_sef_hfo_obs_readout import plot_readout_diagnostic

# ---------- locked params ----------
RATIO        = 0.6
THETA_DEG    = 45.0
L, N         = _DEFAULT_L, _DEFAULT_N   # 12mm, 96
DT           = 0.25                      # ms
T_MAX        = 280.0                     # ms (event dur ~100ms well within)
STIM_R       = 2.0                       # mm
STIM_DIST    = 3.0                       # mm from center toward -θ_EE end
STIM_AMP     = 8.0                       # matched to sef_hfo_lif_kick_sweep.py
STIM_T_ON    = 0.0
STIM_T_OFF   = 30.0                      # ms = _STIM_T
KERNEL_W     = 1.0                       # mm sampling kernel (< spacing/2)
SPACING      = 3.0                       # mm (spans 12mm per shaft; stays within L=12mm)
NC           = 5                         # contacts per shaft → 10 total
SHAFT_LAT_OFFSET = 2.0                   # mm lateral (perp-to-θ_EE) offset for each shaft
AXIS_ERR_MAX = 25.0                      # degrees
KDIR         = 3
PART_MIN     = 7
OUT = "results/topic4_sef_hfo/observation_layer/figures"


def main():
    os.makedirs(OUT, exist_ok=True)
    theta_rad = np.deg2rad(THETA_DEG)

    op = mean_field(RATIO)
    print(f"op: nuE={op['nuE']:.4g} kHz, nuI={op['nuI']:.4g} kHz", flush=True)

    # kick at the -θ_EE side of the sheet center (event propagates toward +θ_EE)
    kick_center = (-STIM_DIST * np.cos(theta_rad), -STIM_DIST * np.sin(theta_rad))
    stim = pulse_stim_fn(kick_center, STIM_R, STIM_AMP, STIM_T_ON, STIM_T_OFF, N, L)

    print("integrating rate field...", flush=True)
    out = integrate_lif_field(
        op, stim, dt=DT, t_max=T_MAX,
        theta_EE=theta_rad, n=N, L=L,
        return_frames=True)
    ext, front = out[0], out[1]
    rE_frames = out[-1]   # LAST element per tuple-position lock
    print(f"  max_ext={ext.max():.3f}, nsteps={rE_frames.shape[0]}", flush=True)

    # event window: from stim offset to where ext drops back below detection threshold
    thr_ext = 0.02                       # active fraction threshold for event-on
    i_off = int(STIM_T_OFF / DT)
    post = ext[i_off:]
    on_idx = np.where(post > thr_ext)[0]
    if on_idx.size == 0:
        print("WARNING: no active pixels detected after stim — check operating point / stim")
        event_win = (STIM_T_OFF, T_MAX)
    else:
        t_win_end = STIM_T_OFF + (on_idx[-1] + 1) * DT
        event_win = (STIM_T_OFF, min(t_win_end, T_MAX))
    print(f"  event_window={event_win[0]:.1f}–{event_win[1]:.1f} ms", flush=True)

    # montage: two shafts PARALLEL to θ_EE, offset ±SHAFT_LAT_OFFSET mm in the
    # perpendicular direction. All contacts within the event stripe + L=12mm sheet.
    # perp direction to θ_EE = (-sinθ, cosθ)
    perp = np.array([-np.sin(theta_rad), np.cos(theta_rad)])
    originA = tuple( SHAFT_LAT_OFFSET * perp)
    originB = tuple(-SHAFT_LAT_OFFSET * perp)
    montage = merge_montages([
        build_shaft(theta_rad, SPACING, NC, origin=originA, name_prefix="A"),
        build_shaft(theta_rad, SPACING, NC, origin=originB, name_prefix="B"),
    ])

    # sample field → per-contact envelopes
    env = rate_event_envelope(rE_frames, N, L, montage, kernel_width=KERNEL_W)
    print(f"  env shape={env.shape}, max={env.max():.4g} kHz, min={env.min():.4g} kHz", flush=True)

    # participation threshold: background + 20% of event amplitude above background
    floor = float(op["nuE"])
    margin = 0.2 * (float(env.max()) - floor)
    print(f"  participation threshold floor={floor:.4g} margin={margin:.4g}", flush=True)
    art = extract_lagpat(env, DT, [event_win], floor, margin, timing_frac=0.5, tie_tol=DT)
    art = attach_geometry(art, montage)

    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    axis = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5 * SPACING)
    err = None if axis is None else round(float(axis_angle_error_deg(axis, theta_rad)), 1)

    passed = (axis is not None) and (err <= AXIS_ERR_MAX) and (n_part >= PART_MIN)
    verdict = "PASS" if passed else "FAIL"
    print(f"\n=== SMOKE {verdict} ===", flush=True)
    print(f"  n_part={n_part} (need >={PART_MIN})", flush=True)
    print(f"  endpoint-axis err_vs_theta_EE={err}° (need <={AXIS_ERR_MAX}°)", flush=True)

    # figure: peak-activity frame as footprint background
    peak_t = int(np.argmax(ext))
    foot_frame = rE_frames[peak_t]
    gxy = grid_coords(N, L)
    foot = foot_frame.ravel()

    # print contact positions for audit (grid-wrap check)
    contacts = np.asarray(montage.contacts)
    print(f"  contact range x=[{contacts[:,0].min():.2f},{contacts[:,0].max():.2f}] "
          f"y=[{contacts[:,1].min():.2f},{contacts[:,1].max():.2f}] (sheet ±{L/2:.1f}mm)", flush=True)

    p_out = plot_readout_diagnostic(
        os.path.join(OUT, "inc3a_rate_smoke_theta45.png"),
        montage, env, DT, art,
        kick_xy=kick_center, theta_EE_deg=THETA_DEG, recovered_axis=axis,
        sheet_L=L, contact_spacing=SPACING,
        source_frame=foot, grid_xy=gxy, event_window=event_win,
        title=(f"Rate-field smoke (L={L:g}mm, ratio={RATIO}, θ_EE={THETA_DEG}°) | "
               f"endpoint_centroid_axis err={err}° n_part={n_part} → {verdict}"))
    print("wrote", p_out, flush=True)


if __name__ == "__main__":
    main()
