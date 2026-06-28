"""M3A-v2 SUBSTRATE QUALIFICATION (slow vars OFF) — does this SNN substrate produce a LOCAL,
SELF-LIMITED, AXIALLY-PROPAGATING interictal event at all? Only if YES does M3A-v2 have testing
meaning (the closed-loop pilot failed because the substrate jumps from no-event to global event,
with no localized axial window). Single-source core (user 2026-06-28: do single-core first).

This is Step 1 of the next-round plan: q_I=1, g_K=0, D_EE=1 (slow=None). Scan the substrate levers
and find configs that pass the substrate PASS CRITERION. NOT looking for a seizure — looking for an
interictal axial event.

PASS criterion (all must hold):
  1. local:        0.05 < R_area < 0.50           (not a whole-sheet recruitment)
  2. axial:        S_axis > 0.70                  (onset gradient along the E->E scaffold)
  3. contained:    F_offaxis < 0.25               (mass stays in the axial corridor)
  4. self-limited: returned == True               (rate falls back to baseline, no tonic high)
  5. propagating:  onset_span_ms > 8 AND |r_axial| > 0.5
        (onset times are ORDERED along the axis with real transit time -- a traveling wave, not a
         near-synchronous co-ignition that merely happens to have a fitted gradient direction)
  + clean:         pre_kick_ignited == False       (quiescent before the kick; the event is evoked)

Levers scanned: AR (anisotropy), g (inhibition/containment), w_EE_scale (recurrent excitation),
nu_ext_ratio (background excitability), KICK_BOOST (min-propagating kick). Fixed: single small core.

Output JSON -> results/topic4_m3a_v2_substrate_qual/qualification_results.json. DESCRIPTIVE screen.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # single-threaded BLAS per worker (before numpy)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                          # noqa: E402
import itertools                                         # noqa: E402
import json                                              # noqa: E402
import multiprocessing as mp                             # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402

import numpy as np                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                                # noqa: E402
from connectivity import place_neurons                   # noqa: E402
from connectivity_rot import build_connectivity_rot      # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.sef_hfo_snn_metrics import self_limit, pre_kick_ignition  # noqa: E402
from slow_field import firing_rate_field                 # noqa: E402
from src.topic4_m3a_v2_phenotype import (                # noqa: E402
    recruitment_area, axis_score, offaxis_fraction, participation_ratio, make_field_grid_xy)

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_substrate_qual")
N_GRID = 32
CORRIDOR_HW = 1.5
THETA_A_FRAC = 0.20
T_KICK = 120.0
EVENT_DUR = 120.0


def _round(x, nd=4):
    return None if (x is None or (isinstance(x, float) and x != x)) else round(float(x), nd)


def run_one(cfg):
    """Build a single-core anisotropic substrate, kick the core locally (slow OFF), return the
    5-criterion source-space readout + PASS flag. cfg: AR, g, w_EE_scale, nu, kick, plus fixed."""
    L = cfg["L"]; theta = np.radians(cfg["theta_deg"]); seed = cfg["seed"]
    axis_unit = np.array([np.cos(theta), np.sin(theta)])
    p = Params(g=cfg["g"], L=L, density=cfg["density"], T=cfg["T"], dt=0.1,
               nu_ext_ratio=cfg["nu"], seed=seed,
               w_EE=0.1575 * cfg["w_EE_scale"], l_EE=cfg["l_EE"], C_EE=cfg["C_EE"])
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta, AR=cfg["AR"],
                                 verbose=False)
    pos = net["pos"]; N = NE + NI
    is_E = np.zeros(N, bool); is_E[:NE] = True
    center = np.array([L / 2, L / 2]); half = L / 2
    core_xy = center - cfg["sep_frac"] * half * axis_unit          # single core at the -axis end
    cf = sample_core_field(pos, is_E, core_xy, cfg["core_r"], np.random.default_rng(seed + 7),
                           core_mean=cfg["core_mean"], core_std=cfg["core_std"], base_mean=18.0)
    vth = cf["vth"]

    net["rng"] = np.random.default_rng(seed)
    res = simulate_kick(p, net, KICK_BOOST=cfg["kick"], slow=None, kick_center=core_xy,
                        r_kick=cfg["r_kick"], t_kick=T_KICK, V_th_per_neuron=vth)

    dt = p.dt; posE = pos[:NE]; rate_E = res["rate_E"]; E_spk = res["E_spk_bool"]
    i_lo, i_hi = int(round(T_KICK / dt)), int(round((T_KICK + EVENT_DUR) / dt))
    post = E_spk[i_lo:i_hi]; ever = post.any(axis=0)
    onset_win = np.where(ever, post.argmax(axis=0) * dt + T_KICK, np.nan)
    A = firing_rate_field(ever, posE, L, N_GRID, sigma=0.5)
    grid_xy = make_field_grid_xy(L, N_GRID)
    A_thr = THETA_A_FRAC * A.max() if A.max() > 0 else 0.0

    # criterion 5: real propagation (onset ordered along axis, with transit time)
    fin = np.isfinite(onset_win)
    onset_span = float(np.ptp(onset_win[fin])) if fin.sum() >= 2 else 0.0
    r_axial = 0.0
    if fin.sum() >= 20:
        ax_pos = (posE[fin] - center) @ axis_unit
        ot = onset_win[fin]
        if ot.std() > 1e-9 and ax_pos.std() > 1e-9:
            r_axial = float(abs(np.corrcoef(ot, ax_pos)[0, 1]))

    sl = self_limit(rate_E, dt, T_KICK)
    ignited, ig_lat = pre_kick_ignition(rate_E, dt, T_KICK)
    R_area = recruitment_area(A, A_thr)
    S_axis = axis_score(posE, onset_win, axis_unit)
    F_off = offaxis_fraction(A, grid_xy, center, axis_unit, CORRIDOR_HW)
    G_PR = participation_ratio(A)

    crit = dict(c1_local=bool(0.05 < R_area < 0.50),
                c2_axial=bool(S_axis == S_axis and S_axis > 0.70),
                c3_contained=bool(F_off == F_off and F_off < 0.25),
                c4_returned=bool(sl["returned"] and sl["tail_complete"]),
                c5_propagating=bool(onset_span > 8.0 and r_axial > 0.5),
                clean=bool(not ignited))
    return dict(**{k: cfg[k] for k in ("AR", "g", "w_EE_scale", "nu", "kick")},
                n_onsets=int(ever.sum()), R_area=_round(R_area), S_axis=_round(S_axis),
                F_offaxis=_round(F_off), G_PR=_round(G_PR), returned=bool(sl["returned"]),
                tail_complete=bool(sl["tail_complete"]), onset_span_ms=_round(onset_span, 1),
                r_axial=_round(r_axial), peak_rate=_round(sl["peak"], 1),
                pre_kick_ignited=bool(ignited), pre_kick_latency=_round(ig_lat, 1),
                PASS=bool(all(crit.values())), **crit)


def build_grid(args):
    fixed = dict(L=args.L, density=args.density, theta_deg=45.0, T=args.T, seed=args.seed,
                 sep_frac=0.6, core_mean=args.core_mean, core_std=1.0, core_r=args.core_r,
                 r_kick=args.r_kick, l_EE=0.380, C_EE=800)
    grid = list(itertools.product(args.AR, args.g, args.w_EE_scale, args.nu, args.kick))
    return [dict(fixed, AR=AR, g=g, w_EE_scale=w, nu=nu, kick=k) for (AR, g, w, nu, k) in grid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--AR", type=float, nargs="+", default=[4.0, 6.0, 8.0, 10.0])
    ap.add_argument("--g", type=float, nargs="+", default=[3.6, 5.0, 6.5])
    ap.add_argument("--w_EE_scale", type=float, nargs="+", default=[0.7, 1.0])
    ap.add_argument("--nu", type=float, nargs="+", default=[0.35, 0.50])
    ap.add_argument("--kick", type=float, nargs="+", default=[1.0, 1.5, 2.5, 4.0])
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=800.0)   # long enough for slow high-AR waves to return
    ap.add_argument("--core-mean", dest="core_mean", type=float, default=16.5)
    ap.add_argument("--core-r", dest="core_r", type=float, default=1.0)
    ap.add_argument("--r-kick", dest="r_kick", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    configs = build_grid(a)
    n_workers = min(a.workers, len(configs))
    print(f"substrate qualification: {len(configs)} configs x {n_workers} workers "
          f"(AR{a.AR} g{a.g} wEE{a.w_EE_scale} nu{a.nu} kick{a.kick})", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        rows = pool.map(run_one, configs)
    wall = time.time() - t0

    passes = [r for r in rows if r["PASS"]]
    payload = dict(meta=dict(date="2026-06-28", step="substrate qualification (slow OFF, single core)",
                             n_configs=len(configs), n_pass=len(passes), wall_s=round(wall, 1),
                             criterion="0.05<R_area<0.5; S_axis>0.7; F_off<0.25; returned; "
                                       "onset_span>8 & |r_axial|>0.5; not pre-igniting"),
                   passes=passes, all_rows=rows)
    out_path = os.path.join(a.out, "qualification_results.json")
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"\n{len(configs)} configs in {wall:.0f}s. PASS={len(passes)}/{len(configs)}", flush=True)
    if passes:
        print("PASS configs (AR,g,wEE,nu,kick | R_area S_axis F_off span r_axial):")
        for r in sorted(passes, key=lambda r: -r["S_axis"]):
            print(f"  AR={r['AR']} g={r['g']} wEE={r['w_EE_scale']} nu={r['nu']} kick={r['kick']} | "
                  f"R={r['R_area']} S={r['S_axis']} Foff={r['F_offaxis']} span={r['onset_span_ms']} "
                  f"r={r['r_axial']}")
    else:
        # no pass: show the near-misses (which criterion fails most) to guide the next scan
        print("NO PASS. criterion fail counts:")
        for c in ("c1_local", "c2_axial", "c3_contained", "c4_returned", "c5_propagating", "clean"):
            print(f"  {c}: {sum(1 for r in rows if not r[c])}/{len(rows)} fail")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
