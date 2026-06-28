"""M3A-v2 substrate BROAD parameter sweep (slow vars OFF, single core) — Step 1 refinement.

Step 1 found local axial self-limited events EXIST but self-limitation is MARGINAL (best 3/4 seeds).
This sweep adds the unscanned Lever 2 (surround inhibition WIDTH l_EI -- wider I->E than E->E ->
spatial containment) and scans AR x g x l_EI x nu x kick MULTI-SEED, so the report is about the whole
landscape (which lever moves which criterion, where the ROBUST self-limit region is), not the first
config that passes. Reuses run_one from run_m3a_v2_substrate_qualification.

PASS criterion (per run, all 5 + clean): 0.05<R_area<0.5; S_axis>0.7; F_off<0.25; returned&tail;
onset_span>8 & |r_axial|>0.5; not pre-igniting. A CONFIG's robustness = #seeds PASS / #seeds.

Output JSON -> results/topic4_m3a_v2_substrate_qual/sweep_results.json. DESCRIPTIVE screen.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                          # noqa: E402
import itertools                                         # noqa: E402
import json                                              # noqa: E402
import multiprocessing as mp                             # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402
from collections import defaultdict                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_m3a_v2_substrate_qualification import run_one   # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_substrate_qual")
LEVERS = ("AR", "g", "l_EI", "C_EI", "nu", "kick")
CRIT = ("c1_local", "c2_axial", "c3_contained", "c4_returned", "c5_propagating", "clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--AR", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--g", type=float, nargs="+", default=[5.0, 8.0, 12.0])
    ap.add_argument("--l_EI", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    ap.add_argument("--C_EI", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--nu", type=float, nargs="+", default=[0.46, 0.48, 0.50])
    ap.add_argument("--kick", type=float, nargs="+", default=[3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=800.0)
    ap.add_argument("--core-mean", dest="core_mean", type=float, default=16.5)
    ap.add_argument("--core-r", dest="core_r", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fixed = dict(L=a.L, density=a.density, theta_deg=45.0, T=a.T, sep_frac=0.6,
                 core_mean=a.core_mean, core_std=1.0, core_r=a.core_r, r_kick=0.3,
                 l_EE=0.380, C_EE=800, w_EE_scale=1.0)
    combos = list(itertools.product(a.AR, a.g, a.l_EI, a.C_EI, a.nu, a.kick))
    configs = [dict(fixed, AR=AR, g=g, l_EI=lEI, C_EI=cEI, nu=nu, kick=k, seed=s)
               for (AR, g, lEI, cEI, nu, k) in combos for s in a.seeds]
    n_workers = min(a.workers, len(configs))
    print(f"broad sweep: {len(combos)} configs x {len(a.seeds)} seeds = {len(configs)} runs "
          f"x {n_workers} workers", flush=True)
    print(f"  AR{a.AR} g{a.g} l_EI{a.l_EI} C_EI{a.C_EI} nu{a.nu} kick{a.kick}", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        rows = pool.map(run_one, configs)
    wall = time.time() - t0

    # aggregate by lever-tuple (across seeds)
    by_combo = defaultdict(list)
    for cf, r in zip(configs, rows):
        by_combo[tuple(cf[k] for k in LEVERS)].append(r)
    agg = []
    for combo, rs in by_combo.items():
        n = len(rs)
        rec = dict(zip(LEVERS, combo), n_seeds=n, n_pass=sum(r["PASS"] for r in rs),
                   pass_rate=round(sum(r["PASS"] for r in rs) / n, 3),
                   **{f"{c}_rate": round(sum(r[c] for r in rs) / n, 3) for c in CRIT},
                   R_area_mean=round(sum(r["R_area"] for r in rs) / n, 3),
                   S_axis_mean=round(sum((r["S_axis"] or 0) for r in rs) / n, 3),
                   F_off_mean=round(sum((r["F_offaxis"] or 0) for r in rs) / n, 3),
                   peak_mean=round(sum(r["peak_rate"] for r in rs) / n, 1))
        agg.append(rec)
    agg.sort(key=lambda r: (-r["pass_rate"], -r["c4_returned_rate"]))

    # per-lever marginal pass-rate (mean over all configs at each value)
    marg = {}
    for lv in LEVERS:
        d = defaultdict(list)
        for r in agg:
            d[r[lv]].append(r["pass_rate"])
        marg[lv] = {v: round(sum(x) / len(x), 3) for v, x in sorted(d.items())}

    robust = [r for r in agg if r["n_pass"] >= max(3, r["n_seeds"] - 1)]   # >=3/4 (or n-1)
    full = [r for r in agg if r["n_pass"] == r["n_seeds"]]                   # 4/4 robust

    payload = dict(meta=dict(date="2026-06-28", step="substrate sweep (slow OFF, single core, Lever 2)",
                             n_configs=len(combos), n_runs=len(configs), seeds=a.seeds,
                             wall_s=round(wall, 1), n_full_robust=len(full), n_robust=len(robust)),
                   per_lever_marginal_pass_rate=marg, aggregates=agg)
    out_path = os.path.join(a.out, "sweep_results.json")
    json.dump(payload, open(out_path, "w"), indent=2)

    # ---- landscape report ----
    print(f"\n{len(configs)} runs in {wall:.0f}s. configs 4/4-robust={len(full)}, "
          f">=3/4-robust={len(robust)} / {len(combos)}", flush=True)
    print("\nPER-LEVER marginal mean pass-rate (which lever value helps):")
    for lv in LEVERS:
        print(f"  {lv:5}: " + "  ".join(f"{v}->{r}" for v, r in marg[lv].items()))
    print("\nTOP configs by robustness (AR,g,l_EI,nu,kick | pass_rate c4_rate | R S Foff peak):")
    for r in agg[:12]:
        print(f"  AR={r['AR']} g={r['g']} l_EI={r['l_EI']} nu={r['nu']} kick={r['kick']} | "
              f"pass={r['pass_rate']} c4={r['c4_returned_rate']} | R={r['R_area_mean']} "
              f"S={r['S_axis_mean']} Foff={r['F_off_mean']} peak={r['peak_mean']}")
    print("\nPER-CRITERION mean pass-rate across ALL configs (the bottleneck):")
    for c in CRIT:
        print(f"  {c}: {round(sum(r[c + '_rate'] for r in agg) / len(agg), 3)}")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
