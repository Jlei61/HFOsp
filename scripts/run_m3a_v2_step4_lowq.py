"""M3A-v2 Step 4 (fork A) — LOW-Q INITIAL STATE via PRELOAD -> WASHOUT -> PROBE.

Steps 2-3 negative reason: a single ~55ms event from q_init=1 can't deplete off-axis q_I, so the
off-axis permissiveness never forms. This tests the regime that was NEVER tested: deplete q_I over
MANY events first (preload), let the network go quiet (washout), THEN probe a standard kick -- does
the probe event recruit OFF-AXIS / globally and still return, *because* q_I is already depleted?

Phases (per review gate):
  PRELOAD: N repeated kicks on the SAME slow field -> q_I depletes (spatially, axis most + sigma_q
           spread off-axis). full version also builds g_K.
  WASHOUT: the probe is a FRESH run (V reset); its pre-kick window [20,120) is the quiet check.
  PROBE:   one standard kick; read out + the recorded probe-START state.

Conditions (this first SMALL scan):
  - baseline      : q_init=1, no field (the normal axial event).
  - spatial_qonly : preload depletes q (k_K=0), RESET g_K=0, probe -> 'is depleted-q permissiveness
                    enough for off-axis?'
  - spatial_full  : preload depletes q AND builds g_K, keep both, probe -> 'does the real slow state work?'
  (uniform-low-q clamp control added when expanding, matched to the achieved q_global.)

Every probe row records q_{axis,off,global}_init, gK_{axis,off}_init, rate_pre_probe,
pre_probe_ignited -- without these we cannot attribute an effect to the low-q state (review gate).
Success (strict, NOT just S_axis down): returned AND F_off up (or G_PR up) AND R_area up AND not
tonic-pinned AND not pre-igniting.

SMALL first (2 seeds x 2 depths x 3 conditions x 2 substrates). Expand only once q_off_init<0.7 with a
quiet pre-probe is observed. DESCRIPTIVE screen.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                          # noqa: E402
import json                                              # noqa: E402
import multiprocessing as mp                             # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402

import numpy as np                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m3a_v2_step2_qI as S2                          # noqa: E402
import run_m3a_v2_step3_qI_gK as S3                       # noqa: E402  (S3Field records q + g_K)
from kick_probe import simulate_kick                     # noqa: E402
from slow_field import SpatialSlowFieldConfig            # noqa: E402
from src.sef_hfo_snn_metrics import pre_kick_ignition    # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_step4_lowq")
SIGMA_Q, SIGMA_K = 1.5, 0.5
N_PRELOAD = 4             # preload kicks (deplete q; k_K=0 so g_K doesn't pre-suppress the preload)
KQ_LEVELS = [0.02, 0.05, 0.10]   # preload depletion rate (sweep -> find q_global ~ 0.5-0.7)
KK_PROBE = 1.0           # g_K build during the PROBE (braked version; g_K builds where activity goes)
ETA_K_PROBE = 60.0       # g_K brake coupling (braked version), ~ Step-3 disinhibition scale
_round = S2._round


def _cfg(k_q, k_K, eta_K):
    return SpatialSlowFieldConfig(use_qI=True, use_gK=(k_K > 0), k_q=k_q, k_K=k_K, sigma_q=SIGMA_Q,
                                  sigma_K=SIGMA_K, q_min=0.0, eta_K=eta_K, gK_max=1.0, tau_q=S2.TAU_Q,
                                  tau_K=S2.TAU_Q, tau_a=S2.TAU_A, q_init=1.0)


def _new_field(S, cfg):
    f = S3.S3Field(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    f.set_masks(S["masks"]["axis"], S["masks"]["offaxis"])
    return f


def _state(f, S):
    am, om = S["masks"]["axis"], S["masks"]["offaxis"]
    return dict(q_axis_init=_round(float(f.q_I[am].mean())), q_off_init=_round(float(f.q_I[om].mean())),
                q_global_init=_round(float(f.q_I.mean())), gK_axis_init=_round(float(f.g_K[am].mean())),
                gK_off_init=_round(float(f.g_K[om].mean())))


def _probe(S, field, seed):
    """Fresh kick run with `field` (None=baseline). Returns readout + pre-probe quiet check."""
    S["net"]["rng"] = np.random.default_rng(seed)
    res = simulate_kick(S["p"], S["net"], KICK_BOOST=S2.KICK, slow=field, kick_center=S["core_xy"],
                        r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"])
    r = S2._readout(res, S, slow=field)
    dt = S["p"].dt
    ig, _ = pre_kick_ignition(res["rate_E"], dt, S2.T_KICK)
    i0, i1 = int(round(20 / dt)), int(round(S2.T_KICK / dt))
    r["rate_pre_probe"] = _round(float(res["rate_E"][i0:i1].mean()), 2)
    r["pre_probe_ignited"] = bool(ig)
    return r


def _probe_field(S, q_profile, k_K, eta_K):
    """Probe field: q_I FROZEN at the depleted profile (k_q=0), g_K starts 0 (dynamic if k_K>0)."""
    f = _new_field(S, _cfg(0.0, k_K, eta_K))
    f.q_I[:] = q_profile
    return f


def worker(task):
    sub_id, seed, kq = task
    S = S2.build(S2.SUBSTRATES[sub_id], seed)
    meta = dict(substrate=sub_id, seed=seed, kq_preload=kq, n_preload=N_PRELOAD)
    out = []

    # --- baseline (q_init=1, no field) ---
    b = _probe(S, None, seed)
    out.append(dict(**meta, condition="baseline", **{k: None for k in
               ("q_axis_init", "q_off_init", "q_global_init", "gK_axis_init", "gK_off_init")}, **b))

    def _success(r):
        return (r["returned"] and not r["pre_probe_ignited"] and r["R_area"] is not None
                and r["R_area"] > (b["R_area"] or 0) + 0.05
                and r["F_off"] is not None and ((r["F_off"] - (b["F_off"] or 0)) > 0.05 or r["F_off"] > 0.30))

    # --- PRELOAD: N kicks deplete q (k_K=0 so g_K does not pre-suppress) ---
    pf = _new_field(S, _cfg(kq, 0.0, 0.0))
    for i in range(N_PRELOAD):
        S["net"]["rng"] = np.random.default_rng(seed * 100 + i + 1)
        simulate_kick(S["p"], S["net"], KICK_BOOST=S2.KICK, slow=pf, kick_center=S["core_xy"],
                      r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"])
    st = _state(pf, S)                                       # probe-start q state (g_K=0 from preload)
    q_depl = pf.q_I.copy()

    # --- PROBE on the depleted-q state: qonly (no brake) vs braked (dynamic g_K) ---
    for cond, k_K, eta_K in (("qonly", 0.0, 0.0), ("braked", KK_PROBE, ETA_K_PROBE)):
        f = _probe_field(S, q_depl, k_K, eta_K)
        r = _probe(S, f, seed)
        out.append(dict(**meta, condition=cond, **st, **r,
                        dR=_round((r["R_area"] or 0) - (b["R_area"] or 0)),
                        dF=_round((r["F_off"] or 0) - (b["F_off"] or 0)),
                        dS=_round((r["S_axis"] or 0) - (b["S_axis"] or 0)), success=_success(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=["primary", "sensitivity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--kq", type=float, nargs="+", default=KQ_LEVELS)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(s, sd, kq) for s in a.substrates for sd in a.seeds for kq in a.kq]
    print(f"Step 4 low-q (SMALL): {len(tasks)} (substrate,seed,kq) tasks, N_preload={N_PRELOAD}", flush=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        raw = [r for rows in pool.map(worker, tasks) for r in rows]
    wall = time.time() - t0
    probes = [r for r in raw if r["condition"] != "baseline"]
    n_success = sum(r.get("success", False) for r in probes)
    json.dump(dict(meta=dict(date="2026-06-28", step="Step 4 fork A low-q preload->washout->probe (small)",
                             kq_levels=KQ_LEVELS, n_preload=N_PRELOAD, kk_probe=KK_PROBE,
                             eta_K_probe=ETA_K_PROBE, n_runs=len(raw), wall_s=round(wall, 1),
                             n_success=n_success),
                   raw_rows=raw), open(os.path.join(a.out, "step4_lowq_small.json"), "w"), indent=2)
    print(f"\n{len(raw)} rows in {wall:.0f}s. SUCCESS={n_success}/{len(probes)}", flush=True)
    print("GATE CHECK — did preload reach q_off_init<0.7 with a QUIET pre-probe?")
    print(f"{'sub':11} {'sd':>2} {'kq':>5} {'cond':9} | {'q[ax/off/glob]_init':>22} {'gK[ax]':>6} "
          f"{'rate_pre':>8} {'preIg':>5} | {'R':>6} {'S':>6} {'F':>6} {'ret':>4} {'dF':>7} succ")
    for r in raw:
        qi = (f"{r['q_axis_init']}/{r['q_off_init']}/{r['q_global_init']}"
              if r["q_axis_init"] is not None else "-")
        print(f"{r['substrate']:11} {r['seed']:>2} {r['kq_preload']:>5} {r['condition']:9} | {qi:>22} "
              f"{str(r.get('gK_axis_init')):>6} {str(r.get('rate_pre_probe')):>8} "
              f"{str(r.get('pre_probe_ignited'))[0]:>5} | {r['R_area']} {r['S_axis']} {r['F_off']} "
              f"{str(r['returned'])[0]:>4} {str(r.get('dF','')):>6} {r.get('success','')}")
    print(f"\nwrote {os.path.join(a.out, 'step4_lowq_small.json')}", flush=True)


if __name__ == "__main__":
    main()
