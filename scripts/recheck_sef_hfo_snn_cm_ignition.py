"""SNN cm-scale IGNITION recheck (user 2026-06-07 '后面两个'): does a cm event ignite when
density is set adequately (the prior cm run used density=100 -> ~6-7 seeded neurons -> no
ignition; the validated 2mm runs use density=1000 -> ~70 seeded neurons -> ignites)?

This is the '同尺度复核': L=12mm (cm-ish, fits 4mm real-SEEG pitch), density=1000 (igniting),
single-end kick at the theta_EE end, current-LFP recorded at a real-4mm-pitch 3-shaft montage.
Reports (1) whether it ignites, (2) whether the current-LFP read recovers theta_EE at REAL
spacing. NOT written as 'SNN supported' regardless of outcome — this is a feasibility recheck.

N = density * L^2 = 1000 * 144 = 144k neurons -> heavier build than 2mm (N=9k); run in background.
"""
import sys
import os
import json
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick                         # noqa: E402
from lfp import LFPRecorder                                  # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run         # noqa: E402

L, DENSITY, T, DT, DRIVE = 12.0, 1000.0, 220.0, 0.1, 0.6   # cm-ish + igniting density
CENTER = np.array([L / 2, L / 2])
PITCH, NC, THETA = 4.0, 4, 45.0                             # locked real-SEEG pitch
SHAFTS = (10.0, 70.0, 130.0)                                # 3 non-parallel (D6)
KDIR, AXIS_ERR_MAX, PART_MIN = 3, 25.0, 7


def read_chain(env, env_ref, fdt, m, theta_rad, label):
    win = event_window_for_run(env.mean(0), env_ref.mean(0), fdt)
    if win is None:
        return {"signal": label, "n_part": 0, "axis_err": None, "readability": None, "win": None}
    art = extract_lagpat(env, fdt, [win], float(env.min()),
                         0.5 * (float(env.max()) - float(env.min())), 0.5, fdt)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5 * PITCH)
    rd = direction_readability(r0, b0, art.contact_coords)
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_rad)), 1)
    return {"signal": label, "n_part": int(b0.sum()), "axis_err": err,
            "readability": (None if rd is None or rd != rd else round(float(rd), 3)),
            "win": [round(win[0], 1), round(win[1], 1)]}


def main():
    theta_rad = np.deg2rad(THETA)
    p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    print(f"placing neurons (N~{int(DENSITY * L * L)})...", flush=True)
    pos, labels, NE, NI = place_neurons(p, rng)
    print(f"building connectivity (NE={NE}, NI={NI})...", flush=True)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=theta_rad, AR=2.0, verbose=False)
    posE = net["pos"][:NE]
    nut = compute_nu_theta(p)[0]
    m = merge_montages([build_shaft(np.deg2rad(a), PITCH, NC, tuple(CENTER), chr(65 + i))
                        for i, a in enumerate(SHAFTS)])
    end = CENTER + 0.6 * (L / 2) * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    print("simulating kick (on)...", flush=True)
    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(end), lfp_recorder=rec)
    print("simulating ref (off)...", flush=True)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(end), lfp_recorder=rec)

    ignited = float(on["E_spk_bool"].mean(axis=1).max())
    base = float(off["E_spk_bool"].mean(axis=1).max())
    print(f"IGNITION: on max_spk_frac={ignited:.4f} (off={base:.4f}) "
          f"-> {'IGNITED' if ignited > 3 * base + 0.005 else 'NO EVENT'}", flush=True)

    env_f, fdt, _ = snn_event_envelope(on["E_spk_bool"], posE, m, DT)
    env_f_ref, _, _ = snn_event_envelope(off["E_spk_bool"], posE, m, DT)
    res_fire = read_chain(env_f, env_f_ref, fdt, m, theta_rad, "firing_envelope")
    res_lfp = read_chain(on["lfp_trace"].T, off["lfp_trace"].T, DT, m, theta_rad, "current_lfp")

    def ok(r):
        return r["axis_err"] is not None and r["n_part"] >= PART_MIN and r["axis_err"] < AXIS_ERR_MAX
    verdict = {"ignited": bool(ignited > 3 * base + 0.005),
               "on_max_spk_frac": ignited, "off_max_spk_frac": base,
               "firing": res_fire, "current_lfp": res_lfp,
               "cm_4mm_read_passes": bool(ok(res_lfp)),
               "config": dict(L=L, density=DENSITY, N=int(DENSITY * L * L), pitch=PITCH,
                              shafts=SHAFTS, theta=THETA)}
    out = "results/topic4_sef_hfo/observation_layer/lfp_forward_validation"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cm_ignition_recheck_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=lambda o: None)
    print("firing     :", res_fire, flush=True)
    print("current_lfp:", res_lfp, flush=True)
    print(f"cm 4mm read passes = {verdict['cm_4mm_read_passes']}", flush=True)


if __name__ == "__main__":
    main()
