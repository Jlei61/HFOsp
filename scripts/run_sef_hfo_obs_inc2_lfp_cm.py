"""cm-scale SNN + FORMAL current-based LFP — RUN ONCE, SAVE RAW (user 2026-06-07; advisor).

Runs ONE cm-scale SNN (L~20mm, low density), single-end-kicks at the theta_EE end, records the
current-based LFP (engine lfp.py |I_E|+|I_I| forward model) at a virtual montage with REAL depth-
electrode spacing (~4mm). Saves ALL raw arrays (on/off E_spk_bool + LFP, posE, montage params) to
npz, then hands off to scripts.analyze_sef_hfo_obs_cm_offline.analyze_cm for the oracle + the
per-contact-baseline read-out + the §13 figure. Iterate the read-out OFFLINE from the npz (no need
to re-pay this 15-25 min sim). The off run is used ONLY by the oracle's EXCESS-front guard (neuron
space), NOT by the LFP read-out (single-trial, per-contact baseline — advisor 2026-06-07).
"""
import sys
import os
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick, T_KICK, DUR_KICK      # noqa: E402 (patched: kick_center, lfp_recorder)
from lfp import LFPRecorder                                  # noqa: E402 (patched: sites=)

from src.sef_hfo_observation import build_shaft, merge_montages          # noqa: E402
from scripts.analyze_sef_hfo_obs_cm_offline import analyze_cm            # noqa: E402

L, DENSITY, T, DT, DRIVE = 20.0, 100.0, 320.0, 0.1, 0.6     # cm-scale, low density (N~40k)
SPACING, NC, THETA = 4.0, 7, 45.0                           # ~real depth-electrode pitch
SHAFT_ANGLES = [10.0, 100.0]                                # two non-parallel shafts, centered
CENTER = np.array([L / 2, L / 2])
NPZ = "results/topic4_sef_hfo/observation_layer/inc2_cm_raw.npz"


def main():
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(THETA), AR=2.0, verbose=True)
    posE = net["pos"][:NE]
    nu_theta = compute_nu_theta(p)[0]
    m = merge_montages([build_shaft(np.deg2rad(a), SPACING, NC, tuple(CENTER), chr(ord("A") + i))
                        for i, a in enumerate(SHAFT_ANGLES)])
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    end = CENTER + 0.5 * (L / 2) * np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])

    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(end), lfp_recorder=rec)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(end), lfp_recorder=rec)

    np.savez_compressed(
        NPZ,
        on_spk=on["E_spk_bool"], off_spk=off["E_spk_bool"],
        on_lfp=on["lfp_trace"], off_lfp=off["lfp_trace"],
        posE=posE, contacts=m.contacts, names=np.array(m.names),
        dt=DT, L=L, T=T, theta=THETA, T_KICK=T_KICK, DUR_KICK=DUR_KICK,
        spacing=SPACING, NC=NC, shaft_angles_deg=np.array(SHAFT_ANGLES),
        center=CENTER, end=end, NE=NE)
    print("saved raw arrays ->", NPZ, flush=True)
    analyze_cm(NPZ, zthr=4.0)


if __name__ == "__main__":
    main()
