"""Offline oracle + per-contact-baseline LFP read-out + §13 figure from a saved cm-scale SNN npz.

advisor 2026-06-07: the previous "on - off" causal isolation has NO real-SEEG analog (the real PR-2
detector thresholds each channel against its OWN pre-event baseline, there is no no-kick control
trial) AND is swamped by chaotic trajectory divergence (same RNG seed, but the kick perturbs the
state so the two runs decorrelate everywhere within ms). Correct, pipeline-faithful read-out =
SINGLE-TRIAL, per-contact baseline z-score of the ACTUAL kicked LFP.

Before trusting a NULL read, confirm a front EXISTS with the dense-neuron oracle
(anisotropy_front.principal_axis on the EXCESS front, ON & ~OFF — the quiet-drive background guard).
The off run is used ONLY here, in neuron-onset space, for that guard — never for the LFP read-out.

Saving raw arrays once + iterating in this file avoids re-paying the 15-25 min sim per read-out
variant (z-threshold / event window).
"""
import sys
import os
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
from anisotropy_front import (onset_times, front_mask, principal_axis,        # noqa: E402
                              axis_error, max_active_frac)

from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis, axis_angle_error_deg)
from scripts.plot_sef_hfo_obs_readout import plot_readout_diagnostic            # noqa: E402


def _montage_from_params(d):
    """Deterministically rebuild the montage from saved params (no pickle)."""
    sp = float(d["spacing"]); nc = int(d["NC"]); cen = tuple(np.asarray(d["center"], float))
    angs = np.asarray(d["shaft_angles_deg"], float)
    return merge_montages([build_shaft(np.deg2rad(a), sp, nc, cen, chr(ord("A") + i))
                           for i, a in enumerate(angs)])


def analyze_cm(npz_path, zthr=4.0, out_png=None, win=None, verbose=True):
    d = np.load(npz_path, allow_pickle=False)
    dt = float(d["dt"]); L = float(d["L"]); theta = float(d["theta"]); T = float(d["T"])
    T_KICK = float(d["T_KICK"]); DUR_KICK = float(d["DUR_KICK"]); spacing = float(d["spacing"])
    end = np.asarray(d["end"], float); CENTER = np.asarray(d["center"], float)
    posE = d["posE"]; on_lfp = d["on_lfp"]; on_spk = d["on_spk"]; off_spk = d["off_spk"]
    m = _montage_from_params(d)
    FRONT_LO = T_KICK + DUR_KICK
    nhat = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])

    # ---- ORACLE: does an anisotropic front exist at all? (excess = ON & ~OFF) ----
    on_on = onset_times(on_spk, dt, FRONT_LO)
    off_on = onset_times(off_spk, dt, FRONT_LO)
    oracle = {}
    for W in (8.0, 20.0, 50.0):
        mex = front_mask(on_on, FRONT_LO, FRONT_LO + W) & ~front_mask(off_on, FRONT_LO, FRONT_LO + W)
        a, r, n = principal_axis(mex, posE, CENTER)
        oracle[W] = dict(angle=None if not np.isfinite(a) else round(a, 1),
                         ratio=None if not np.isfinite(r) else round(r, 2), n=int(n),
                         err=None if not np.isfinite(a) else round(axis_error(a, theta), 1))
    maf = float(max_active_frac(on_spk, dt, FRONT_LO, min(FRONT_LO + 50.0, T)))
    mex50 = (front_mask(on_on, FRONT_LO, FRONT_LO + 50.0)
             & ~front_mask(off_on, FRONT_LO, FRONT_LO + 50.0))
    extent = 0.0
    if int(mex50.sum()) >= 3:
        proj = (posE[mex50] - CENTER) @ nhat
        extent = float(proj.max() - proj.min())

    # ---- READOUT: per-contact baseline z-score on the ACTUAL kicked LFP (single trial) ----
    lfp = np.asarray(on_lfp, float).T                  # (n_contact, nsteps)
    i_k = int(round(T_KICK / dt))
    mu = lfp[:, :i_k].mean(1, keepdims=True)
    sd = lfp[:, :i_k].std(1, keepdims=True) + 1e-9
    env = np.clip((lfp - mu) / sd, 0, None)            # per-contact z (positive deflection)
    if win is None:
        win = (FRONT_LO, T)                            # post-kick-offset propagation window
    art = extract_lagpat(env, dt, [win], 0.0, float(zthr), 0.5, dt)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    axis = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=3, eps_deg=0.5 * spacing)
    n_part = int(b0.sum())
    err = None if axis is None else round(float(axis_angle_error_deg(axis, np.deg2rad(theta))), 1)

    # ---- figure background: early-front spike-density footprint (numpy histogram2d) ----
    mid = on_spk[i_k:i_k + int(80 / dt)].sum(0)
    H, xe, ye = np.histogram2d(posE[:, 0], posE[:, 1], bins=40, range=[[0, L], [0, L]], weights=mid)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(xc, yc, indexing="ij")
    gxy = np.column_stack([GX.ravel(), GY.ravel()]); foot = H.ravel()

    if verbose:
        print("[oracle] excess-front (ON&~OFF) principal axis per window:")
        for W in (8.0, 20.0, 50.0):
            print(f"   W={W:>4g}ms -> {oracle[W]}")
        print(f"[oracle] max_active_frac(post-kick 50ms)={maf:.3f}  "
              f"front_extent_along_theta={extent:.1f}mm")
        print(f"[readout] zthr={zthr} n_part={n_part} endpoint-axis err_vs_theta={err} "
              f"(theta_EE={theta})")

    if out_png is None:
        out_png = os.path.join("results", "topic4_sef_hfo", "observation_layer",
                               "figures", "inc2_cm_currentLFP.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plot_readout_diagnostic(
        out_png, m, env, dt, art, end, theta, axis, L, spacing,
        source_frame=foot, grid_xy=gxy, event_window=win,
        title=(f"cm-scale SNN (L={L:g}mm) CURRENT-LFP z-scored vs pre-kick baseline (formal §13.4) "
               f"@ {spacing:g}mm contacts; theta_EE={theta:g}°  "
               f"[oracle excess-axis err W50={oracle[50.0]['err']}°]"))
    print("wrote", out_png, flush=True)
    return dict(oracle=oracle, maf=maf, extent=extent, n_part=n_part, readout_err=err)


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else \
        "results/topic4_sef_hfo/observation_layer/inc2_cm_raw.npz"
    zt = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    analyze_cm(npz, zthr=zt)
