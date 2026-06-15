"""SNN current-LFP at ∥/⊥ electrodes — 3 mm IGNITING sheet, end-kick (user 2026-06-07).

The saved 3 mm artifacts (front_shapes) hold only per-neuron ONSET, no time-resolved
signal. To show "the electrical signal each contact records" (user, referencing
inc2_cm_currentLFP.png panel B) we must re-run an igniting SNN and record the formal
current-based LFP (engine/lfp.py |I_E|+|I_I|) at two virtual electrodes: one ∥ the E→E
connectivity axis, one ⊥. End-kick (NOT the center-kick of front_shapes) → a unidirectional
event, so a ∥ electrode should see the LFP peak sweep contact-to-contact.

Scale honesty (user LFP lock): 3 mm is the small scale where the current-LFP forward is
validated (~2 mm); contacts are SCALED (sub-mm), NOT real 4 mm SEEG. cm scale does not ignite.

Runs ONLY the kick-ON trial (per-contact PRE-KICK baseline is the reference, no off-run needed),
saves raw arrays, then the offline plotter reads them. Run in background (~minutes):
  PYTHONPATH="$PWD" python scripts/run_sef_hfo_obs_two_electrode_snn_lfp.py
"""
import sys
import os
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                 # noqa: E402
from connectivity import place_neurons                       # noqa: E402
from connectivity_rot import build_connectivity_rot          # noqa: E402
from kick_probe import simulate_kick, T_KICK, DUR_KICK       # noqa: E402
from lfp import LFPRecorder                                   # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import build_shaft, merge_montages          # noqa: E402

# ---- igniting small-scale params (= anisotropy_front: L=3, density=1800) ----
L, DENSITY, T, DT, DRIVE = 3.0, 1800.0, 220.0, 0.1, 0.6
THETA = 45.0                                  # E->E connectivity long axis
PITCH_PAR, NC_PAR = 0.26, 9                   # ∥ electrode (scaled, sub-mm)
PITCH_PERP, NC_PERP = 0.26, 9                 # ⊥ electrode
CENTER = np.array([L / 2, L / 2])
# electrodes sit on the DOWN-axis propagation arm (the event nucleates centrally and
# the front travels toward [0,0]); placing them off the kick/boundary side gives a clean
# one-way front rather than straddling the nucleation (which reads as a V).
ELEC_CENTER = np.array([0.85, 0.85])
NPZ = "results/topic4_sef_hfo/observation_layer/two_electrode_snn_raw.npz"


def main():
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(THETA), AR=2.0, verbose=True)
    posE = net["pos"][:NE]
    nu_theta = compute_nu_theta(p)[0]

    u = np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    m_par = build_shaft(np.deg2rad(THETA), PITCH_PAR, NC_PAR, tuple(ELEC_CENTER), "P")
    m_perp = build_shaft(np.deg2rad(THETA + 90.0), PITCH_PERP, NC_PERP, tuple(ELEC_CENTER), "Q")
    m = merge_montages([m_par, m_perp])           # first NC_PAR = ∥, next NC_PERP = ⊥
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)

    # end-kick: seed near the +axis end so the event travels back along the diagonal
    end = CENTER + 0.6 * (L / 2) * u
    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(end), lfp_recorder=rec)

    # per-E-neuron first-spike time after the kick (for panel-A activation map)
    spk = on["E_spk_bool"]                          # (nsteps, NE)
    i_kick = int(round(T_KICK / DT))
    post = spk[i_kick:]
    ever = post.any(axis=0)
    onset_E = np.where(ever, (np.argmax(post, axis=0) + i_kick) * DT, np.nan)

    np.savez_compressed(
        NPZ,
        on_lfp=on["lfp_trace"], times=on["times"], rate_E=on["rate_E"], onset_E=onset_E,
        posE=posE, contacts=m.contacts, names=np.array(m.names),
        nc_par=NC_PAR, nc_perp=NC_PERP, pitch_par=PITCH_PAR, pitch_perp=PITCH_PERP,
        dt=DT, L=L, T=T, theta=THETA, T_KICK=T_KICK, DUR_KICK=DUR_KICK,
        center=CENTER, end=end, NE=NE, peak_rate_Hz=float(np.max(on["rate_E"])))
    print(f"saved -> {NPZ}", flush=True)
    print(f"  NE={NE} peak E-rate={np.max(on['rate_E']):.1f} Hz "
          f"(baseline {np.mean(on['rate_E'][:int(120/DT)]):.1f} Hz) "
          f"-> ignited={np.max(on['rate_E']) > 5 * max(np.mean(on['rate_E'][:int(120/DT)]), 1e-3)}",
          flush=True)


if __name__ == "__main__":
    main()
