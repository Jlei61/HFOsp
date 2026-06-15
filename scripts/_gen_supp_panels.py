"""
Data-generation for the 3 pictorial supplementary-figure panels.
Pure data-gen: no plotting, no interpretation. Reuses engine helpers verbatim.

Outputs (results/topic4_sef_hfo/lif_snn/data/):
  regime_timeseries.json   (panel b: quiet vs oscillating base-net rate)
  kick_snapshots.npz       (panel c: kick spreads then self-terminates)
  front_shapes.npz         (panel d: onset front tracks rotated connectivity axis)
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np

ENGINE = os.path.join("src", "snn_engine")

sys.path.insert(0, ENGINE)

from params import Params, compute_nu_theta                       # noqa: E402
from model import build_network, simulate                         # noqa: E402
from connectivity import place_neurons                            # noqa: E402
from connectivity_rot import build_connectivity_rot               # noqa: E402
from kick_probe import (simulate_kick, peak_active_fraction,      # noqa: E402
                        T_KICK, DUR_KICK, R_KICK)

OUT = os.path.join("results", "topic4_sef_hfo", "lif_snn", "data")
os.makedirs(OUT, exist_ok=True)


# ======================================================================
# File 1 -- regime_timeseries.json
# ======================================================================
def gen_regime_timeseries():
    print("\n=== File 1: regime_timeseries.json ===", flush=True)
    base = dict(g=3.6, L=1.0, density=4000.0, T=800.0, seed=1)
    dt = Params(**base).dt          # 0.1 ms
    bin_steps = int(round(1.0 / dt))  # 10 dt-steps -> 1 ms

    def run(ratio):
        p = Params(nu_ext_ratio=ratio, **base)
        net = build_network(p, verbose=False)   # consumes seed via rng
        res = simulate(p, net, verbose=False)
        r = res["rate_E"]                        # per-dt, Hz
        n_bins = len(r) // bin_steps
        r1ms = r[:n_bins * bin_steps].reshape(n_bins, bin_steps).mean(axis=1)
        t1ms = (np.arange(n_bins) + 0.5) * 1.0   # ms, bin centers
        print(f"  ratio={ratio}: mean={r1ms.mean():.2f} Hz  "
              f"max={r1ms.max():.2f} Hz  std={r1ms.std():.2f}", flush=True)
        return t1ms, r1ms

    t_quiet, rateE_quiet = run(0.6)
    t_osc, rateE_osc = run(1.0)
    assert len(t_quiet) == len(t_osc)

    doc = dict(
        t_ms=t_quiet.tolist(),
        rateE_quiet=rateE_quiet.tolist(),
        rateE_osc=rateE_osc.tolist(),
        drive_quiet=0.6,
        drive_osc=1.0,
    )
    path = os.path.join(OUT, "regime_timeseries.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    print(f"  saved {path}", flush=True)
    return dict(path=path,
                quiet=dict(mean=float(rateE_quiet.mean()),
                           max=float(rateE_quiet.max()),
                           std=float(rateE_quiet.std())),
                osc=dict(mean=float(rateE_osc.mean()),
                         max=float(rateE_osc.max()),
                         std=float(rateE_osc.std())))


# ======================================================================
# File 2 -- kick_snapshots.npz
# ======================================================================
def gen_kick_snapshots():
    print("\n=== File 2: kick_snapshots.npz ===", flush=True)
    p = Params(g=3.6, L=3.0, density=1800.0, T=320.0, nu_ext_ratio=0.6, seed=1)
    dt = p.dt
    net = build_network(p, verbose=False)
    net["rng"] = np.random.default_rng(1)
    KICK_BOOST = 2 * compute_nu_theta(p)[0]
    res = simulate_kick(p, net, KICK_BOOST=KICK_BOOST)

    E_spk_bool = res["E_spk_bool"]                # (nsteps, NE)
    NE = res["NE"]
    posE = net["pos"][:NE]

    windows = [(150.0, 158.0), (158.0, 168.0), (168.0, 178.0), (195.0, 210.0)]

    def window_xy(lo, hi):
        i_lo = int(round(lo / dt)); i_hi = int(round(hi / dt))
        active = E_spk_bool[i_lo:i_hi].any(axis=0)   # any spike in window
        return posE[active].astype(float)

    w_xy = [window_xy(lo, hi) for lo, hi in windows]

    # peak active fraction over 2-ms bins (spec), across the whole kick->end window
    paf = peak_active_fraction(E_spk_bool, dt, T_KICK, p.T, bin_ms=2.0)

    path = os.path.join(OUT, "kick_snapshots.npz")
    np.savez(
        path,
        w0_xy=w_xy[0], w1_xy=w_xy[1], w2_xy=w_xy[2], w3_xy=w_xy[3],
        windows=np.array(windows, dtype=float),
        L=3.0,
        kick_center=np.array([1.5, 1.5], dtype=float),
        kick_R=R_KICK,
        peak_active_frac=float(paf),
    )
    ns = [len(x) for x in w_xy]
    print(f"  NE={NE}  peak_active_frac={paf:.4f}", flush=True)
    print(f"  n per window {windows} = {ns}", flush=True)
    print(f"  saved {path}", flush=True)
    return dict(path=path, peak_active_frac=float(paf),
                n_per_window=ns, windows=windows)


# ======================================================================
# File 3 -- front_shapes.npz
# ======================================================================
def gen_front_shapes():
    print("\n=== File 3: front_shapes.npz ===", flush=True)
    base = dict(g=3.6, L=3.0, density=1800.0, T=320.0, nu_ext_ratio=0.6)
    conds = [("c0", 0.0, 2.0), ("c45", 45.0, 2.0),
             ("c90", 90.0, 2.0), ("ciso", 0.0, 1.0)]
    front_lo, front_hi = 168.0, 180.0   # EARLY front window (pre-saturation)
    center = np.array([1.5, 1.5])

    save_kw = dict(L=3.0, kick_center=center.astype(float),
                   labels=np.array(["theta=0", "theta=45", "theta=90",
                                    "isotropic"]))
    report = {}

    for key, theta_deg, AR in conds:
        p = Params(seed=1, **base)
        dt = p.dt
        # build pattern mirrors anisotropy_front.build_network_rot: ONE fresh rng
        # consumed by placement AND kernel sampling.
        rng = np.random.default_rng(p.seed)
        pos, labels, NE, NI = place_neurons(p, rng)
        net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                     theta_EE=np.radians(theta_deg), AR=AR,
                                     verbose=False)
        net["rng"] = rng
        # fresh_run_rot: reseed rng each run then simulate_kick
        net["rng"] = np.random.default_rng(p.seed)
        KICK_BOOST = 2 * compute_nu_theta(p)[0]
        res = simulate_kick(p, net, KICK_BOOST=KICK_BOOST)

        E_spk_bool = res["E_spk_bool"]
        posE = net["pos"][:NE]

        # onset = first-spike time (ms) AT OR AFTER T_KICK; inf if never
        i0 = int(round(T_KICK / dt))
        post = E_spk_bool[i0:]
        ever = post.any(axis=0)
        first_idx = post.argmax(axis=0)
        onset = np.full(NE, np.inf)
        onset[ever] = (i0 + first_idx[ever]) * dt

        keep = (onset >= front_lo) & (onset < front_hi)
        xy = posE[keep].astype(float)
        on = onset[keep].astype(float)

        save_kw[f"{key}_xy"] = xy
        save_kw[f"{key}_onset"] = on

        # rough elongation direction (cov major axis of front pts about center)
        n = int(keep.sum())
        if n >= 3:
            pts = xy - center
            C = np.cov(pts, rowvar=False)
            evals, evecs = np.linalg.eigh(C)
            vmaj = evecs[:, 1]
            ang = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
            ratio = float(np.sqrt(evals[1] / evals[0])) if evals[0] > 0 else float("inf")
        else:
            ang, ratio = float("nan"), float("nan")
        print(f"  {key} (theta={theta_deg} AR={AR}): n_front={n}  "
              f"elong_angle={ang:.1f}deg  ratio={ratio:.2f}", flush=True)
        report[key] = dict(n_front=n, elong_angle_deg=ang, elong_ratio=ratio,
                           theta_deg=theta_deg, AR=AR)
        del res, net

    path = os.path.join(OUT, "front_shapes.npz")
    np.savez(path, **save_kw)
    print(f"  saved {path}", flush=True)
    return dict(path=path, per_cond=report)


if __name__ == "__main__":
    r1 = gen_regime_timeseries()
    r2 = gen_kick_snapshots()
    r3 = gen_front_shapes()
    print("\n===== SUMMARY =====")
    print("File1:", r1)
    print("File2:", r2)
    print("File3:", r3)
