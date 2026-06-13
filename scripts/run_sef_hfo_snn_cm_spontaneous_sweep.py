"""Unattended DIAGNOSTIC sweep: twoend_equal across drive x core_mean x core-spacing x seeds.

GOAL: find a working point where ONE record has BOTH neg AND pos clean events (>= COEXIST_MIN
each) — the prerequisite Stage 3 needs to test label-timing in a single network. The pilot showed
one core monopolizes ignition at drive=0.6/mean=17.0/sep=0.6 (events too global -> whoever fires
first suppresses the other). Two levers probed here to relieve that mutual suppression:
  * Pass 1 (USER-CONFIRMED): drive {0.40..0.60} x core_mean {17.0, 17.5}, 7 seeds, at pilot spacing
    sep_frac=0.6. Lower drive / higher mean -> sparser, more local events.
  * Pass 2 (sep_frac lever, user wired --sep-frac while leaving 2026-06-14): widen the core spacing
    sep_frac {0.75, 0.90} at the pilot op-point (drive 0.6, mean 17.0). Farther apart = weaker
    coupling between the two ends = the most direct decoupling lever.

DIAGNOSTIC ONLY: finds candidate working points; draws NO scientific conclusion, makes NO figure,
runs NO multi-seed long-record conclusion. Output = summary table + candidate list for user review.
Coexist gate = neg_clean >= 5 AND pos_clean >= 5 in ONE record (user-confirmed).
Idempotent (skips a cell whose readout exists; safe to resume). --collect-only / --dry-run available.
"""
import os
import json
import csv
import argparse
import subprocess
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed

RUNNER = "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
SWEEP_DIR = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/twoend_sweep"
T = 4000.0
COEXIST_MIN = 5
MAX_WORKERS = 9                 # each sim uses OMP=8 -> 9*8=72 threads < 80 cores (headroom)
PER_RUN_TIMEOUT = 90 * 60       # one L=20/T=4000 run is ~35 min; 90 min = generous backstop
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")

# job = (drive, core_mean, sep_frac, seed)
SEEDS = list(range(1, 8))
JOBS = (
    [(d, m, 0.6, s) for d in [0.40, 0.45, 0.50, 0.55, 0.60] for m in [17.0, 17.5] for s in SEEDS]  # Pass 1
    + [(0.60, 17.0, sf, s) for sf in [0.75, 0.90] for s in SEEDS]                                   # Pass 2
)


def tag_for(d, m, sf, s):
    return f"sweep_d{d:.2f}_m{m:.1f}_sf{sf:.2f}_s{s}"


def run_one(job, dry=False):
    d, m, sf, s = job
    tag = tag_for(d, m, sf, s)
    readout = os.path.join(SWEEP_DIR, f"readout_{tag}.json")
    if os.path.exists(readout):
        return tag, "skip(exists)"
    cmd = ["python3", RUNNER, "--L", "20", "--density", "100", "--T", str(T),
           "--lesion", "twoend_equal", "--core-mean", str(m), "--core-std", "1.5", "--core-r", "1.5",
           "--drive", str(d), "--sep-frac", str(sf), "--seed", str(s),
           "--delta-onset", "30", "--n-min", "5", "--tag", tag, "--out", SWEEP_DIR]
    if dry:
        return tag, "DRY: " + " ".join(cmd)
    log = os.path.join(SWEEP_DIR, f"{tag}.log")
    try:
        with open(log, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False,
                           env=ENV, timeout=PER_RUN_TIMEOUT)
        return tag, ("ok" if os.path.exists(readout) else "no_readout")
    except subprocess.TimeoutExpired:
        return tag, "timeout"
    except Exception as e:                      # noqa: BLE001 - one bad run must not kill the sweep
        return tag, f"err:{e!r}"


def collect():
    rows = []
    for (d, m, sf, s) in JOBS:
        tag = tag_for(d, m, sf, s)
        p = os.path.join(SWEEP_DIR, f"readout_{tag}.json")
        row = dict(drive=d, core_mean=m, sep_frac=sf, seed=s, status="missing", n_events=None,
                   neg_clean=None, pos_clean=None, collision=None, ambiguous=None,
                   collision_rate=None, true_floor=None, coexist=False)
        if os.path.exists(p):
            try:
                j = json.load(open(p))
                sc = j.get("stage3_source_counts") or {}
                nc, pc = sc.get("neg_clean"), sc.get("pos_clean")
                row.update(status="ok", n_events=j.get("n_events"),
                           neg_clean=nc, pos_clean=pc, collision=sc.get("collision"),
                           ambiguous=sc.get("ambiguous"), collision_rate=sc.get("collision_rate"),
                           true_floor=j.get("detector", {}).get("true_inter_event_floor"),
                           coexist=bool((nc or 0) >= COEXIST_MIN and (pc or 0) >= COEXIST_MIN))
            except Exception as e:              # noqa: BLE001
                row["status"] = f"parse_err:{e!r}"
        rows.append(row)
    return rows


def write_summary(rows):
    csvp = os.path.join(SWEEP_DIR, "sweep_summary.csv")
    with open(csvp, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cells = {}
    for r in rows:
        cells.setdefault((r["drive"], r["core_mean"], r["sep_frac"]), []).append(r)
    cand = [r for r in rows if r["coexist"]]
    done = sum(1 for r in rows if r["status"] == "ok")

    L = ["# twoend_equal diagnostic sweep — candidate working points\n",
         f"Grid: Pass 1 drive{{0.40..0.60}} x core_mean{{17.0,17.5}} @ sep_frac=0.6 (70) + "
         f"Pass 2 sep_frac{{0.75,0.90}} @ drive0.6/mean17.0 (14); 7 seeds; T={T}. "
         f"Coexist gate: neg_clean >= {COEXIST_MIN} AND pos_clean >= {COEXIST_MIN} in ONE record.\n",
         "**DIAGNOSTIC ONLY** — candidate working points for the user to review; no scientific "
         "conclusion, no figure, no multi-seed long run. A 'coexist' cell = the prerequisite Stage 3 "
         "needs (both ends fire their own events in one network).\n",
         f"\nRuns with readout: {done}/{len(rows)}. Coexist runs: **{len(cand)}**.\n",
         "\n## Per-cell summary (drive x core_mean x sep_frac)\n",
         "| drive | core_mean | sep_frac | seeds_ok | coexist | best min(neg,pos) | median coll_rate | n hot(floor>0.04) | median n_events |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(cells):
        cr = cells[k]
        ok = [r for r in cr if r["status"] == "ok"]
        n_co = sum(1 for r in ok if r["coexist"])
        best = max((min(r["neg_clean"] or 0, r["pos_clean"] or 0) for r in ok), default=0)
        colls = [r["collision_rate"] for r in ok if r["collision_rate"] is not None]
        floors = [r["true_floor"] for r in ok if r["true_floor"] is not None]
        nev = [r["n_events"] for r in ok if r["n_events"] is not None]
        n_hot = sum(1 for f in floors if f > 0.04)
        L.append(f"| {k[0]:.2f} | {k[1]:.1f} | {k[2]:.2f} | {len(ok)}/{len(cr)} | {n_co} | {best} | "
                 f"{round(st.median(colls), 3) if colls else '-'} | {n_hot} | "
                 f"{int(st.median(nev)) if nev else '-'} |")

    L.append("\n## Coexist candidate runs (neg_clean >= 5 AND pos_clean >= 5)\n")
    if cand:
        L.append("| drive | core_mean | sep_frac | seed | neg | pos | coll | amb | coll_rate | n_events |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(cand, key=lambda r: -min(r["neg_clean"], r["pos_clean"])):
            L.append(f"| {r['drive']:.2f} | {r['core_mean']:.1f} | {r['sep_frac']:.2f} | {r['seed']} | "
                     f"{r['neg_clean']} | {r['pos_clean']} | {r['collision']} | {r['ambiguous']} | "
                     f"{r['collision_rate']} | {r['n_events']} |")
    else:
        L.append("**NONE** — no working point produced a record with both ends >= 5 clean events. "
                 "Then: drive / core_mean / core-spacing in this grid did not relieve the one-network "
                 "mutual-suppression blocker; report the near-misses (highest best-min cells above) "
                 "and the next levers (even wider spacing / different design) for the user to decide.")
    L.append("\n*(internal: stage3_source_counts gate; drive=nu_ext_ratio; sep_frac=core spacing as "
             "fraction of half-axis; pilot archive docs/archive/topic4/sef_hfo/stage3_twoend_equal_pilot_2026-06-13.md.)*\n")
    open(os.path.join(SWEEP_DIR, "SWEEP_SUMMARY.md"), "w").write("\n".join(L))
    print(f"[sweep] wrote sweep_summary.csv + SWEEP_SUMMARY.md; coexist candidates: {len(cand)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(SWEEP_DIR, exist_ok=True)

    if a.collect_only:
        write_summary(collect())
        return
    if a.dry_run:
        for j in JOBS[:3] + JOBS[-2:]:
            print(run_one(j, dry=True)[1])
        print(f"[sweep] DRY: {len(JOBS)} total runs ({MAX_WORKERS} parallel) — Pass1 70 + Pass2 14")
        return

    print(f"[sweep] {len(JOBS)} runs, {MAX_WORKERS} parallel, T={T}, out={SWEEP_DIR}", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, j): j for j in JOBS}
        done = 0
        for f in as_completed(futs):
            tag, status = f.result()
            done += 1
            print(f"[sweep] {done}/{len(JOBS)} {tag} -> {status}", flush=True)
    write_summary(collect())
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
