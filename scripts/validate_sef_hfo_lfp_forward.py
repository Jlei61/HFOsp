"""Validate the current-based LFP forward model on a REAL ignited event (user 2026-06-07).

Q1: is the §13.4 current-LFP forward model (engine/lfp.py |I_E|+|I_I| at sites) sound —
does it, on a real propagating event, recover the connectivity direction θ_EE through the
SAME read-out pipeline as the firing envelope? The cm run could not answer this (no event
ignited). This uses the RELIABLY-IGNITING 2mm config (_diag_singleend.py: L=3, density=1000,
single-end kick → firing-envelope endpoint err 6.4°) and reads the SAME event TWO ways:
  (a) firing-density envelope (snn_event_envelope)         — the already-validated read
  (b) current-based LFP (lfp_recorder, |I_E|+|I_I|)        — the forward model under test
Both go through event_window_for_run → extract_lagpat → endpoint_centroid_axis. If (b)
recovers θ_EE (err<25°, n_part≥7) AND agrees with (a), the LFP forward model is validated
AT THIS SCALE. (Real-4mm-pitch cm-scale read-out is a separate, heavier '后调' question.)
"""
import sys
import os
import json
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick                         # noqa: E402 (patched: kick_center, lfp_recorder)
from lfp import LFPRecorder                                  # noqa: E402 (patched: sites=)

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run         # noqa: E402

L, DENSITY, T, DT, DRIVE = 3.0, 1000.0, 220.0, 0.1, 0.6     # validated igniting 2mm config
CENTER = np.array([L / 2, L / 2])
PITCH, NC, THETA = 0.45, 8, 45.0
KDIR, AXIS_ERR_MAX, PART_MIN = 3, 25.0, 7


def read_chain(env, env_ref, fdt, m, theta_rad, label):
    """SAME read-out pipeline for either signal: window -> lagpat -> endpoint axis."""
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
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=theta_rad, AR=2.0, verbose=False)
    posE = net["pos"][:NE]
    nut = compute_nu_theta(p)[0]
    m = merge_montages([build_shaft(np.deg2rad(10.0), PITCH, NC, tuple(CENTER), "A"),
                        build_shaft(np.deg2rad(100.0), PITCH, NC, tuple(CENTER), "B")])
    end = CENTER + 0.6 * (L / 2) * np.array([np.cos(theta_rad), np.sin(theta_rad)])   # single-end
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)

    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(end), lfp_recorder=rec)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(end), lfp_recorder=rec)

    print(f"event ignited: on max_spk_frac={on['E_spk_bool'].mean(axis=1).max():.4f} "
          f"(off={off['E_spk_bool'].mean(axis=1).max():.4f})", flush=True)

    # (a) firing-density envelope (validated reference)
    env_f, fdt, _ = snn_event_envelope(on["E_spk_bool"], posE, m, DT)
    env_f_ref, _, _ = snn_event_envelope(off["E_spk_bool"], posE, m, DT)
    res_fire = read_chain(env_f, env_f_ref, fdt, m, theta_rad, "firing_envelope")

    # (b) current-based LFP (forward model under test): |I_E|+|I_I| at the same contacts
    env_l = on["lfp_trace"].T                     # (n_contact, nsteps)
    env_l_ref = off["lfp_trace"].T
    res_lfp = read_chain(env_l, env_l_ref, DT, m, theta_rad, "current_lfp")

    def ok(r):
        return r["axis_err"] is not None and r["n_part"] >= PART_MIN and r["axis_err"] < AXIS_ERR_MAX
    lfp_validated = ok(res_lfp)
    agree = (res_fire["axis_err"] is not None and res_lfp["axis_err"] is not None
             and abs(res_fire["axis_err"] - res_lfp["axis_err"]) < 15.0)
    verdict = {"firing": res_fire, "current_lfp": res_lfp,
               "LFP_FORWARD_MODEL_VALIDATED_2mm": bool(lfp_validated),
               "fire_lfp_agree": bool(agree),
               "config": dict(L=L, density=DENSITY, theta=THETA, AR=2.0, scale="2mm mesoscopic")}
    out = "results/topic4_sef_hfo/observation_layer/lfp_forward_validation"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "lfp_forward_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=lambda o: None)
    print("firing     :", res_fire, flush=True)
    print("current_lfp:", res_lfp, flush=True)
    print(f"LFP forward model validated @2mm = {lfp_validated} | firing~lfp agree = {agree}", flush=True)


if __name__ == "__main__":
    main()
