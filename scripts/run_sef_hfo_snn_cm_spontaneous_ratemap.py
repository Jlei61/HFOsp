"""Phase-1 spontaneous EVENT-RATE map across lesion (mean x var), BEFORE the read-out sweep
(user 2026-06-11): the read-out sweep must run where the lesion abnormality GRADES the spontaneous
activity ("more abnormal -> more spontaneous events"), NOT where everything already saturates.

The earlier cm scan only measured "does it self-ignite at all" (peak active fraction / off_peak).
This measures the SEPARABLE self-terminated EVENT RATE (events/s via the locked detect_events) plus
separability (returned fraction, IEI) as a function of core mean x var at the -end lesion — so we
can SEE the graded regime and pick a read-out-suitable band (enough separable events, not merged).

Reuses diag_sef_hfo_snn_cm_spontaneous_train.py (one sim per cell, --core-at negend). Aggregates
to ratemap_summary.json + a rate-vs-mean figure (wide vs narrow). NO read-out / montage here.
"""
import os
import glob
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DIAG = "scripts/diag_sef_hfo_snn_cm_spontaneous_train.py"
OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = os.path.join(OUT, "figures")
MEANS = [18.0, 17.5, 17.0, 16.5, 16.0]       # quiet(18) -> boundary -> saturated(16)
SPREADS = [("wide", 1.5), ("narrow", 0.5)]
SEEDS = [1, 2]
T = 1500.0
MAXPAR = 10
# A "large directional" event = a real sheet-spanning propagating wave, NOT a tiny flat fluctuation.
# (review 2026-06-11: narrow mean>=17.5 produced 46-118-neuron, corr~0, peak~3e-4 "events" that the
# bare returned-count scored as separable — they must not count as readable spontaneous events.)
MIN_EVENT_N, MIN_ABSCORR, MIN_PEAK = 5000, 0.5, 0.02


def _gated_metrics(json_path, T_s):
    """Compute, from a spont_<tag>.json (+ its active_fraction npy): the bare returned-event rate AND
    the LARGE-DIRECTIONAL-event rate (returned & >=MIN_EVENT_N neurons & |corr|>=MIN_ABSCORR & peak
    >=MIN_PEAK), plus a TRUE inter-event baseline = p95 active fraction OUTSIDE detected event windows
    (the detector's calibration 'floor' is only the first 5-50ms, not whole-trace quietness)."""
    j = json.load(open(json_path))
    evs = j["events"]; ne = len(evs); nret = sum(1 for e in evs if e["returned"])
    large = [e for e in evs if e["returned"] and e.get("direction")
             and e["direction"].get("n", 0) >= MIN_EVENT_N
             and abs(e["direction"].get("corr", 0.0)) >= MIN_ABSCORR
             and e.get("peak_ext", 0.0) >= MIN_PEAK]
    nn = [e["direction"]["n"] for e in evs if e.get("direction")]
    # true inter-event baseline from the saved active-fraction trace
    tag = os.path.basename(json_path)[len("spont_"):-len(".json")]
    af_p = os.path.join(os.path.dirname(json_path), f"active_fraction_{tag}.npy")
    true_floor = None
    if os.path.exists(af_p):
        af = np.load(af_p); bin_ms = j["config"].get("bin_ms", 1.0)
        mask = np.ones(len(af), bool)
        for e in evs:                                    # blank out event windows (+/-10ms pad)
            s = max(0, int((e["t_on"] - 10) / bin_ms)); o = int((e["t_off"] + 10) / bin_ms)
            mask[s:o] = False
        inter = af[mask]
        true_floor = round(float(np.percentile(inter, 95)), 5) if inter.size else None
    return dict(n_events=ne, n_returned=nret, n_large_dir=len(large),
                sep_rate_per_s=round(nret / T_s, 2),
                large_dir_rate_per_s=round(len(large) / T_s, 2),
                median_event_size=int(np.median(nn)) if nn else 0,
                cal_floor=j["detector"]["floor"], true_inter_event_floor=true_floor,
                peak=j["detector"]["peak"])


def run_cell(mean, sname, std, seed):
    tag = f"RATE_m{mean:g}_{sname}_s{seed}"
    cmd = ["python3", DIAG, "--core-at", "negend", "--core-mean", str(mean), "--core-std", str(std),
           "--T", str(T), "--seed", str(seed), "--tag", tag]
    env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6")
    with open(os.path.join(OUT, f"{tag}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False, env=env)
    p = os.path.join(OUT, f"spont_{tag}.json")
    if not os.path.exists(p):
        return dict(mean=mean, spread=sname, seed=seed, ok=False)
    return dict(mean=mean, spread=sname, seed=seed, ok=True, **_gated_metrics(p, T / 1000.0))


def _reaggregate_cells():
    """Rebuild cells from existing spont_RATE_*.json (NO re-sim) — recompute the gated metrics."""
    import re
    cells = []
    for p in sorted(glob.glob(os.path.join(OUT, "spont_RATE_*.json"))):
        m = re.match(r"spont_RATE_m([0-9.]+)_(wide|narrow)_s(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        mean, sname, seed = float(m.group(1)), m.group(2), int(m.group(3))
        cells.append(dict(mean=mean, spread=sname, seed=seed, ok=True, **_gated_metrics(p, T / 1000.0)))
    return cells


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaggregate", action="store_true",
                    help="recompute metrics from existing spont_RATE_*.json without re-running sims")
    ap.add_argument("--maxpar", type=int, default=MAXPAR)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    jobs = [(m, sn, std, s) for (sn, std) in SPREADS for m in MEANS for s in SEEDS]
    if a.reaggregate:
        cells = _reaggregate_cells()
        failed = []
    else:
        with ThreadPoolExecutor(max_workers=a.maxpar) as ex:
            all_cells = list(ex.map(lambda x: run_cell(*x), jobs))
        cells = [c for c in all_cells if c.get("ok")]
        failed = [dict(mean=c["mean"], spread=c["spread"], seed=c["seed"]) for c in all_cells if not c.get("ok")]

    agg = {}
    for c in cells:
        agg.setdefault((c["mean"], c["spread"]), []).append(c)
    table = []
    for (mean, sname), cs in sorted(agg.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        sep = [c["sep_rate_per_s"] for c in cs]
        large = [c["large_dir_rate_per_s"] for c in cs]
        tf = [c["true_inter_event_floor"] for c in cs if c["true_inter_event_floor"] is not None]
        sz = [c["median_event_size"] for c in cs]
        row = dict(mean=mean, spread=sname, n_seeds=len(cs),
                   sep_rate_per_s_mean=round(float(np.mean(sep)), 2), sep_rate_per_s_by_seed=sep,
                   large_dir_rate_per_s_mean=round(float(np.mean(large)), 2), large_dir_rate_by_seed=large,
                   true_inter_event_floor_mean=round(float(np.mean(tf)), 5) if tf else None,
                   median_event_size=int(np.median(sz)) if sz else 0)
        table.append(row)
        print(f"  mean={mean:g} {sname:6s} large_dir_rate={large} (sep={sep}) "
              f"true_floor={row['true_inter_event_floor_mean']} med_size={row['median_event_size']}", flush=True)

    json.dump(dict(config=dict(L=20, density=100, lesion="negend", T=T, means=MEANS,
                               spreads=[s[0] for s in SPREADS], seeds=SEEDS,
                               large_event_gate=dict(min_n=MIN_EVENT_N, min_abscorr=MIN_ABSCORR, min_peak=MIN_PEAK)),
                   table=table, cells=cells, failed_cells=failed),
              open(os.path.join(OUT, "ratemap_summary.json"), "w"), indent=2)
    if failed:
        print(f"  !! {len(failed)} FAILED cells: {failed}", flush=True)
    print("\nwrote ratemap_summary.json (use plot_sef_hfo_snn_cm_spontaneous_regime.py for the 3-panel figure)")


if __name__ == "__main__":
    main()
