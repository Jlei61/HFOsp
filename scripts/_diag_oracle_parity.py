"""DIAGNOSTIC (not the full run): oracle (dense neurons) vs virtual (sparse montage)
onset-front axis on the SAME AR=2/theta_EE=45 center-kick event. Discriminates
estimator<->sparse-montage mismatch vs substrate issue (advisor 2026-06-07)."""
import sys, os, numpy as np
ENG = os.path.join("results","topic4_sef_hfo","lif_snn","engine"); sys.path.insert(0, ENG)
from params import Params, compute_nu_theta
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick, T_KICK, DUR_KICK
from anisotropy_front import onset_times, front_mask, principal_axis, FRONT_LO, W_PRIMARY
from src.sef_hfo_observation import build_shaft, merge_montages, extract_lagpat, attach_geometry, onset_front_axis, angle_error_deg
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run

L, DENSITY, T, DT, DRIVE = 3.0, 1000.0, 200.0, 0.1, 0.6
CENTER = np.array([L/2, L/2]); THETA = 45.0
p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
rng = np.random.default_rng(1)
pos, labels, NE, NI = place_neurons(p, rng)
net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.deg2rad(THETA), AR=2.0, verbose=False)
posE = net["pos"][:NE]; nu_theta = compute_nu_theta(p)[0]
net["rng"] = np.random.default_rng(1); on  = simulate_kick(p, net, KICK_BOOST=2*nu_theta, kick_center=list(CENTER))
net["rng"] = np.random.default_rng(1); off = simulate_kick(p, net, KICK_BOOST=0.0,        kick_center=list(CENTER))

# ORACLE (dense neuron positions, excess front [FRONT_LO, FRONT_LO+W])
on_t  = onset_times(on["E_spk_bool"],  DT, T_KICK)
off_t = onset_times(off["E_spk_bool"], DT, T_KICK)
fo = front_mask(on_t, FRONT_LO, FRONT_LO+W_PRIMARY)
ff = front_mask(off_t, FRONT_LO, FRONT_LO+W_PRIMARY)
excess = fo & ~ff
oa, orr, on_n = principal_axis(excess, posE, CENTER)
print(f"ORACLE (dense, n={on_n}): angle={oa:.1f} ratio={orr:.2f}  err_vs45={angle_error_deg(oa,45.0):.1f}")

# VIRTUAL (sparse 2-shaft montage, pitch 0.45)
m = merge_montages([build_shaft(np.deg2rad(10.0),0.45,8,tuple(CENTER),"A"),
                    build_shaft(np.deg2rad(100.0),0.45,8,tuple(CENTER),"B")])
env,fdt,agg = snn_event_envelope(on["E_spk_bool"], posE, m, DT, bin_ms=2.0, smooth_ms=5.0, kernel_width=0.3)
_,_,aggr   = snn_event_envelope(off["E_spk_bool"], posE, m, DT, bin_ms=2.0, smooth_ms=5.0, kernel_width=0.3)
win = event_window_for_run(agg, aggr, fdt)
art = extract_lagpat(env, fdt, [win], float(env.min()), 0.5*(float(env.max())-float(env.min())), 0.5, fdt)
art = attach_geometry(art, m)
va,vr,vn = onset_front_axis(art.lag_raw[:,0], art.bools[:,0], art.contact_coords, 8.0)
ve = None if va is None else round(angle_error_deg(va,45.0),1)
print(f"VIRTUAL (sparse 16c, n_front={vn}): angle={va} ratio={vr}  err_vs45={ve}")
