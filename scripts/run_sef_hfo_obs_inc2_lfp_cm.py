"""cm-scale SNN + FORMAL current-based LFP read-out (user 2026-06-07; spec §13.1/§13.4).

Builds a cm-scale SNN (L~20mm, low density), single-end-kicks at the theta_EE end, records the
current-based LFP (|I_E|+|I_I|, engine lfp.py forward model) at a virtual montage with REAL depth-
electrode spacing (~4mm), and renders the §13 diagnostic figure: electrode overlay + per-electrode
CURRENT-LFP traces (formal LFP, NOT firing envelope) + the lag/axis chain. ONE condition (theta_EE=45)
to validate the cm-scale current-LFP read-out end-to-end before the full four-control.
"""
import sys
import os
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta            # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick, T_KICK            # noqa: E402 (patched: kick_center, lfp_recorder)
from lfp import LFPRecorder                              # noqa: E402 (patched: sites=)

from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis, axis_angle_error_deg)
from scripts.plot_sef_hfo_obs_readout import plot_readout_diagnostic                # noqa: E402

L, DENSITY, T, DT, DRIVE = 20.0, 100.0, 320.0, 0.1, 0.6   # cm-scale, low density (N~40k)
SPACING, NC, THETA = 4.0, 7, 45.0                         # ~real depth-electrode pitch
CENTER = np.array([L / 2, L / 2])
OUT = "results/topic4_sef_hfo/observation_layer/figures"


def main():
    os.makedirs(OUT, exist_ok=True)
    p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(THETA), AR=2.0, verbose=True)
    posE = net["pos"][:NE]
    nu_theta = compute_nu_theta(p)[0]
    m = merge_montages([build_shaft(np.deg2rad(10.0), SPACING, NC, tuple(CENTER), "A"),
                        build_shaft(np.deg2rad(100.0), SPACING, NC, tuple(CENTER), "B")])
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    end = CENTER + 0.5 * (L / 2) * np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])

    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(end), lfp_recorder=rec)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(end), lfp_recorder=rec)

    dt = DT
    i_k = int(round(T_KICK / dt))
    # event-driven current-LFP = kick - no-kick, baseline-subtracted (pre-kick mean), positive part
    base = on["lfp_trace"][:i_k].mean(0)
    env = np.clip((on["lfp_trace"] - off["lfp_trace"]).T, 0, None)   # (n_contact, nsteps)
    fdt = dt
    agg = env.mean(0)
    # simple event window: from kick onset to where aggregate returns near baseline
    thr = 0.2 * agg[i_k:].max()
    on_idx = np.where(agg[i_k:] > thr)[0]
    win = (i_k * dt, (i_k + (on_idx[-1] if on_idx.size else 100)) * dt)

    art = extract_lagpat(env, fdt, [win], float(env.min()),
                         0.3 * (float(env.max()) - float(env.min())), 0.5, fdt)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    axis = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=3, eps_deg=0.5 * SPACING)
    n_part = int(b0.sum())
    err = None if axis is None else round(float(axis_angle_error_deg(axis, np.deg2rad(THETA))), 1)
    # coarse E-spike-density footprint background (numpy histogram2d, no scipy)
    mid = on["E_spk_bool"][i_k:i_k + int(80 / dt)].sum(0)        # spikes/E-neuron during early event
    nb = 40
    H, xe, ye = np.histogram2d(posE[:, 0], posE[:, 1], bins=nb, range=[[0, L], [0, L]], weights=mid)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(xc, yc, indexing="ij")
    gxy = np.column_stack([GX.ravel(), GY.ravel()])
    foot = H.ravel()

    print(f"cm-scale SNN current-LFP: theta_EE={THETA} n_part={n_part} endpoint-axis err_vs_theta={err}", flush=True)
    p_out = plot_readout_diagnostic(
        os.path.join(OUT, "inc2_cm_currentLFP.png"), m, env, fdt, art,
        end, THETA, axis, L, SPACING, source_frame=foot, grid_xy=gxy, event_window=win,
        title=f"cm-scale SNN (L={L:g}mm) + CURRENT-based LFP (formal §13.4) @ {SPACING:g}mm contacts; theta_EE={THETA:g}°")
    print("wrote", p_out, flush=True)


if __name__ == "__main__":
    main()
