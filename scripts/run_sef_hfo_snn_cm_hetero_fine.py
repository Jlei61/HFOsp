"""Step-3 cm heterogeneity FINE scan — resolve the mean-gated ignition boundary in the
(17,18) mV band the coarse scan stepped over (advisor 2026-06-10).

The coarse scan bracketed: core_mean=18 -> off_peak 0.0004 (quiet); =17 -> 0.070 (full
self-ignite). The whole quiet->spontaneous transition is in (0,1)mV — and THAT band is
where the L=3 Step-3 deliverable lives (graded ignition law + wide-vs-narrow contrast AT
the boundary). This fine scan resolves it at multiple seeds.

Discriminating questions (NOT "does it read 45deg" — the read-out already serves):
  1. is off_peak GRADED across core_mean in (17,18), or a SHARP step (= qualitative
     departure from L=3's graded 0.6mV boundary)?
  2. do WIDE vs NARROW spreads SEPARATE at the cm boundary (wide's low-threshold tail
     self-ignites at higher core_mean — the L=3 11/12-vs-3/12 contrast)?
  3. is the boundary LOCATION seed-robust (2-3 seeds)?

Runs concurrently (each sim ~13GB / thread-capped); aggregates to cm_hetero_fine_summary.json.
"""
import os
import sys
import json
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_wave"
RUNNER = "scripts/run_sef_hfo_snn_cm_wave_read.py"
FIXED = ["--L", "20", "--density", "100", "--nc", "6", "--r-kick", "0.6",
         "--theta", "45", "--kick-mode", "negend", "--core", "--core-r", "1.5",
         "--core-at", "kick"]
MEANS = [17.8, 17.6, 17.4, 17.2]
SPREADS = [("wide", 1.5), ("narrow", 0.5)]
SEEDS = [1, 2, 3]
MAXPAR = 8        # 8 x 13GB = 104GB (< 241 free); thread-capped below


def run_cell(mean, sname, std, seed):
    tag = f"L20d100_FINE_m{mean:g}_{sname}_s{seed}"
    cmd = ["python3", RUNNER, *FIXED, "--core-mean", str(mean), "--core-std", str(std),
           "--seed", str(seed), "--tag", tag]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with open(os.path.join(OUT, f"{tag}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False, env=env)
    p = os.path.join(OUT, f"wave_read_{tag}.json")
    if not os.path.exists(p):
        return dict(core_mean=mean, spread=sname, seed=seed, ok=False)
    j = json.load(open(p))
    ig = j["ignition"]
    return dict(core_mean=mean, spread=sname, seed=seed, ok=True,
                ign_on=ig["peak_inst"], ign_off=ig["off_peak_inst"],
                self_ignite=ig["core_self_ignite"], evoked_clean=ig["evoked_clean"])


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(m, sn, std, s) for (sn, std) in SPREADS for m in MEANS for s in SEEDS]
    with ThreadPoolExecutor(max_workers=MAXPAR) as ex:
        cells = list(ex.map(lambda a: run_cell(*a), jobs))
    cells = [c for c in cells if c.get("ok")]

    # aggregate: per (mean, spread) the off_peak across seeds + self-ignite fraction
    agg = {}
    for c in cells:
        k = (c["core_mean"], c["spread"])
        agg.setdefault(k, []).append(c)
    table = []
    for (mean, sname), cs in sorted(agg.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        offs = sorted(round(c["ign_off"], 4) for c in cs)
        n_self = sum(c["self_ignite"] for c in cs)
        table.append(dict(core_mean=mean, spread=sname, n_seeds=len(cs),
                          off_peak_by_seed=offs, off_peak_min=min(offs), off_peak_max=max(offs),
                          self_ignite_frac=f"{n_self}/{len(cs)}"))
        print(f"  m={mean:g} {sname:6s} off_peak(seeds)={offs} self={n_self}/{len(cs)}", flush=True)

    summary = dict(
        config=dict(L=20, density=100, nc=6, core_r=1.5, core_at="kick",
                    means=MEANS, spreads=[s[0] for s in SPREADS], seeds=SEEDS,
                    known_endpoints=dict(m18_off_peak=0.0004, m17_off_peak=0.070)),
        table=table, cells=cells)
    json.dump(summary, open(os.path.join(OUT, "cm_hetero_fine_summary.json"), "w"), indent=2)
    print("\nwrote cm_hetero_fine_summary.json")
    with open(os.path.join(OUT, "cm_hetero_fine.done"), "w") as f:
        f.write("DONE")


if __name__ == "__main__":
    main()
