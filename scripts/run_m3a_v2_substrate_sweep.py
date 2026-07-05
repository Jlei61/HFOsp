"""M3A-v2 substrate BROAD parameter sweep (slow vars OFF, single core) — Step 1 refinement.

Step 1 found local axial self-limited events EXIST but self-limitation was MARGINAL in the narrow
search. This sweep adds the unscanned Lever 2 (surround inhibition l_EI width x C_EI count) and scans
AR x g x l_EI x C_EI x nu MULTI-SEED, so the report is the whole landscape (which lever moves which
criterion, where the ROBUST self-limit region is), not the first config that passes.

AUDITABILITY (review 2026-06-28): the JSON keeps PER-SEED raw rows (each carries the 6 gate flags),
a fresh-seed (5-8) validation of the named canonical candidates, an AR=2 boundary probe (so the AR
claim is a measured gradient, not an assertion), and the canonical candidates' per-seed 1-8 metrics.
A separate multiseed_results.json holds the PRIMARY canonical x seeds 1-8.

PASS (per run): 0.05<R_area<0.5; S_axis>0.7; F_off<0.25; returned&tail; onset_span>8 & |r_axial|>0.5;
not pre-igniting. A CONFIG's robustness = #seeds PASS / #seeds.

Output -> results/topic4_m3a_v2_substrate_qual/{sweep_results.json, multiseed_results.json}. DESCRIPTIVE.
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
FIXED = dict(L=10.0, density=100.0, theta_deg=45.0, T=800.0, sep_frac=0.6, core_mean=16.5,
             core_std=1.0, core_r=1.0, r_kick=0.3, l_EE=0.380, C_EE=800, w_EE_scale=1.0)

# Named canonical candidates (review P1-5): PRIMARY uses the DEFAULT I->E structure (l_EI=0.25) so it
# carries no Lever-2 confound; matched = same but l_EI=0.5; headroom = strong-surround backup only.
CANON = {
    "primary":  dict(AR=4.0, g=8.0, l_EI=0.25, C_EI=200, nu=0.46, kick=3.0),   # default I->E, main baseline
    "matched":  dict(AR=4.0, g=8.0, l_EI=0.50, C_EI=200, nu=0.46, kick=3.0),   # Lever-2 sensitivity
    "headroom": dict(AR=4.0, g=5.0, l_EI=1.00, C_EI=400, nu=0.46, kick=3.0),   # strong surround, backup
}


def _run(combos, seeds, workers):
    configs = [dict(FIXED, **dict(zip(LEVERS, c)), seed=s) for c in combos for s in seeds]
    with mp.Pool(min(workers, len(configs))) as pool:
        return pool.map(run_one, configs)


def _aggregate(rows):
    by = defaultdict(list)
    for r in rows:
        by[tuple(r[k] for k in LEVERS)].append(r)
    agg = []
    for combo, rs in by.items():
        n = len(rs)
        agg.append(dict(zip(LEVERS, combo), n_seeds=n, n_pass=sum(r["PASS"] for r in rs),
                        pass_rate=round(sum(r["PASS"] for r in rs) / n, 3),
                        **{f"{c}_rate": round(sum(r[c] for r in rs) / n, 3) for c in CRIT},
                        R_area_mean=round(sum(r["R_area"] for r in rs) / n, 3),
                        S_axis_mean=round(sum((r["S_axis"] or 0) for r in rs) / n, 3),
                        F_off_mean=round(sum((r["F_offaxis"] or 0) for r in rs) / n, 3),
                        peak_mean=round(sum(r["peak_rate"] for r in rs) / n, 1)))
    agg.sort(key=lambda r: (-r["pass_rate"], -r["c4_returned_rate"]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--AR", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--g", type=float, nargs="+", default=[5.0, 8.0, 12.0])
    ap.add_argument("--l_EI", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    ap.add_argument("--C_EI", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--nu", type=float, nargs="+", default=[0.46, 0.48, 0.50])
    ap.add_argument("--kick", type=float, nargs="+", default=[3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--fresh-seeds", dest="fresh_seeds", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # ---- phase 1: main grid (per-seed raw rows kept) ----
    main_combos = list(itertools.product(a.AR, a.g, a.l_EI, a.C_EI, a.nu, a.kick))
    print(f"main grid: {len(main_combos)} configs x {len(a.seeds)} seeds = "
          f"{len(main_combos) * len(a.seeds)} runs", flush=True)
    raw_rows = _run(main_combos, a.seeds, a.workers)
    agg = _aggregate(raw_rows)
    marg = {}                                              # per-lever marginal mean pass-rate
    for lv in LEVERS:
        d = defaultdict(list)
        for r in agg:
            d[r[lv]].append(r["pass_rate"])
        marg[lv] = {v: round(sum(x) / len(x), 3) for v, x in sorted(d.items())}

    # ---- phase 2: fresh-seed (5-8) validation of the named canonical candidates ----
    canon_combos = [tuple(c[k] for k in LEVERS) for c in CANON.values()]
    fresh_rows = _run(canon_combos, a.fresh_seeds, a.workers)

    # ---- phase 3: AR=2 boundary probe (review P1-1: the AR claim is a measured gradient) ----
    ar2_combos = list(itertools.product([2.0], [3.6, 5.0, 6.5], [0.25], [200], [0.46, 0.50], [1.5, 2.5, 4.0]))
    ar2_rows = _run(ar2_combos, a.seeds, a.workers)
    ar2_agg = _aggregate(ar2_rows)

    # ---- canonical candidates: per-seed 1-8 metrics ----
    def _rows_for(combo, pool_rows):
        return [r for r in pool_rows if all(r[k] == v for k, v in zip(LEVERS, combo))]
    canonical = {}
    for name, c in CANON.items():
        combo = tuple(c[k] for k in LEVERS)
        rs = _rows_for(combo, raw_rows) + _rows_for(combo, fresh_rows)
        canonical[name] = dict(config=c, n_seeds=len(rs), n_pass=sum(r["PASS"] for r in rs),
                               per_seed=[dict(seed=r["seed"], PASS=r["PASS"],
                                              **{k: r[k] for k in CRIT},
                                              R_area=r["R_area"], S_axis=r["S_axis"],
                                              F_offaxis=r["F_offaxis"], returned=r["returned"],
                                              peak_rate=r["peak_rate"]) for r in sorted(rs, key=lambda r: r["seed"])])
    wall = time.time() - t0

    full = [r for r in agg if r["n_pass"] == r["n_seeds"]]
    robust = [r for r in agg if r["n_pass"] >= max(3, r["n_seeds"] - 1)]
    payload = dict(meta=dict(date="2026-06-28", step="substrate sweep (slow OFF, single core, Lever 2)",
                             n_main_configs=len(main_combos), n_runs=len(raw_rows), seeds=a.seeds,
                             fresh_seeds=a.fresh_seeds, wall_s=round(wall, 1),
                             n_full_robust=len(full), n_robust=len(robust),
                             ar2_n_pass_configs=sum(1 for r in ar2_agg if r["n_pass"] > 0)),
                   per_lever_marginal_pass_rate=marg, aggregates=agg,
                   canonical_candidates=canonical, ar2_boundary_probe=ar2_agg,
                   raw_rows=raw_rows, fresh_seed_rows=fresh_rows, ar2_probe_rows=ar2_rows)
    json.dump(payload, open(os.path.join(a.out, "sweep_results.json"), "w"), indent=2)
    # explicit per-seed artifact for the PRIMARY canonical (review P1-2)
    json.dump(dict(meta=dict(canonical="primary", config=CANON["primary"]),
                   per_seed=canonical["primary"]["per_seed"]),
              open(os.path.join(a.out, "multiseed_results.json"), "w"), indent=2)

    # ---- landscape report ----
    print(f"\n{len(raw_rows)} main runs in {wall:.0f}s. 4/4-robust={len(full)}, >=3/4={len(robust)} "
          f"/ {len(main_combos)}", flush=True)
    print("PER-LEVER marginal mean pass-rate:")
    for lv in LEVERS:
        print(f"  {lv:5}: " + "  ".join(f"{v}->{r}" for v, r in marg[lv].items()))
    print("\nAR=2 boundary probe (P1-1): pass configs = "
          f"{sum(1 for r in ar2_agg if r['n_pass'] > 0)}/{len(ar2_agg)}")
    for r in [r for r in ar2_agg if r["n_pass"] > 0][:5]:
        print(f"  AR=2 g={r['g']} nu={r['nu']} kick={r['kick']}: {r['n_pass']}/{r['n_seeds']} "
              f"R={r['R_area_mean']} S={r['S_axis_mean']}")
    print("\nCANONICAL candidates (per-seed 1-8 PASS):")
    for name, c in canonical.items():
        print(f"  {name:8} {CANON[name]}: {c['n_pass']}/{c['n_seeds']} PASS")
    print("\nPER-CRITERION mean pass-rate (bottleneck):")
    for cc in CRIT:
        print(f"  {cc}: {round(sum(r[cc + '_rate'] for r in agg) / len(agg), 3)}")
    print(f"\nwrote sweep_results.json + multiseed_results.json", flush=True)


if __name__ == "__main__":
    main()
