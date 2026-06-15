"""Generate kick_snapshots.npz (panel c) + front_shapes.npz (panel d) for the supplementary.

Reuses kick_probe.simulate_kick (verbatim kick) + anisotropy_front's build_network_rot /
onset_times / front_mask. L=3 / density 1800 (sub-global event). Run from repo root:
  python src/snn_engine/gen_figure_panels.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import Params, compute_nu_theta                       # noqa: E402
from model import build_network                                   # noqa: E402
from kick_probe import simulate_kick, T_KICK, DUR_KICK            # noqa: E402
from anisotropy_front import build_network_rot, onset_times, front_mask  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def gen_snapshots():
    p = Params(g=3.6, L=3.0, density=1800.0, T=320.0, nu_ext_ratio=0.6, seed=1)
    nu_theta = compute_nu_theta(p)[0]
    net = build_network(p, verbose=False)                         # default rho_EE=0.6 anisotropy
    net["rng"] = np.random.default_rng(p.seed)
    res = simulate_kick(p, net, KICK_BOOST=2.0 * nu_theta)
    E, NE, dt = res["E_spk_bool"], res["NE"], p.dt
    posE = net["pos"][:NE]
    wins = [(150, 158), (158, 168), (168, 178), (195, 210)]
    out = {}
    for j, (lo, hi) in enumerate(wins):
        m = E[int(round(lo / dt)):int(round(hi / dt))].any(axis=0)
        out[f"w{j}_xy"] = posE[m]
    best = 0.0
    for b0 in np.arange(T_KICK, p.T, 2.0):
        i0, i1 = int(round(b0 / dt)), int(round((b0 + 2.0) / dt))
        best = max(best, float(E[i0:i1].any(axis=0).mean()))
    np.savez(os.path.join(DATA, "kick_snapshots.npz"),
             windows=np.array(wins, float), L=float(p.L),
             kick_center=np.array([p.L / 2, p.L / 2]), kick_R=0.15,
             peak_active_frac=best, **out)
    print("snapshots: peak_frac=%.3f  n_per_window=%s" %
          (best, [out[f"w{j}_xy"].shape[0] for j in range(4)]))


def gen_fronts():
    p = Params(g=3.6, L=3.0, density=1800.0, T=320.0, nu_ext_ratio=0.6, seed=1)
    nu_theta = compute_nu_theta(p)[0]
    conds = [("c0", 0.0, 2.0), ("c45", 45.0, 2.0), ("c90", 90.0, 2.0), ("ciso", 0.0, 1.0)]
    out, ns = {}, []
    for key, th, AR in conds:
        net = build_network_rot(p, np.radians(th), AR)
        net["rng"] = np.random.default_rng(p.seed)
        res = simulate_kick(p, net, KICK_BOOST=2.0 * nu_theta)
        NE = res["NE"]
        posE = net["pos"][:NE]
        onset = onset_times(res["E_spk_bool"], p.dt, T_KICK)
        m = front_mask(onset, T_KICK + DUR_KICK, T_KICK + DUR_KICK + 12.0)   # [168,180)
        out[f"{key}_xy"] = posE[m]
        out[f"{key}_onset"] = onset[m]
        ns.append(int(m.sum()))
    np.savez(os.path.join(DATA, "front_shapes.npz"),
             L=float(p.L), kick_center=np.array([p.L / 2, p.L / 2]),
             labels=np.array(["theta=0", "theta=45", "theta=90", "isotropic"]), **out)
    print("fronts: n_per_condition=%s" % ns)


if __name__ == "__main__":
    gen_snapshots()
    gen_fronts()
