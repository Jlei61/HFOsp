"""Unattended DIAGNOSTIC sweep: twoend_equal across drive x core_mean x seeds.

GOAL: find a working point where ONE record has BOTH neg AND pos clean events (>= COEXIST_MIN
each) — the prerequisite Stage 3 needs to test label-timing in a single network. The pilot showed
one core monopolizes ignition at drive=0.6/mean=17.0 (events too global -> whoever fires first
suppresses the other). Hypothesis: lower drive / higher core mean -> sparser, more local events ->
the two ends nucleate independently and can coexist in one record.

DIAGNOSTIC ONLY (user-confirmed 2026-06-14): finds candidate working points; draws NO scientific
conclusion, makes NO figure, runs NO multi-seed long-record conclusion. Output = a summary table +
candidate list for the user to review and decide the next formal step.

Grid (user-confirmed): drive x core_mean x 7 seeds; coexist gate = neg_clean >= 5 AND pos_clean >= 5.
Idempotent: skips a cell whose readout already exists (safe to resume). `--collect-only` rebuilds the
summary from existing readouts without running; `--dry-run` prints the commands only.
"""
import os
import sys
import json
import csv
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

RUNNER = "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
SWEEP_DIR = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/twoend_sweep"
DRIVES = [0.40, 0.45, 0.50, 0.55, 0.60]
MEANS = [17.0, 17.5]
SEEDS = [1, 2, 3, 4, 5, 6, 7]
T = 4000.0
COEXIST_MIN = 5
MAX_WORKERS = 9                 # each sim uses OMP=8 -> 9*8=72 threads < 80 cores (headroom)
PER_RUN_TIMEOUT = 90 * 60       # a single L=20/T=4000 run is ~35 min; 90 min = generous backstop
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")


def tag_for(d, m, s):
    return f"sweep_d{d:.2f}_m{m:.1f}_s{s}"


def run_one(d, m, s, dry=False):
    tag = tag_for(d, m, s)
    readout = os.path.join(SWEEP_DIR, f"readout_{tag}.json")
    if os.path.exists(readout):
        return tag, "skip(exists)"
    cmd = ["python3", RUNNER, "--L", "20", "--density", "100", "--T", str(T),
           "--lesion", "twoend_equal", "--core-mean", str(m), "--core-std", "1.5", "--core-r", "1.5",
           "--drive", str(d), "--seed", str(s), "--delta-onset", "30", "--n-min", "5",
           "--tag", tag, "--out", SWEEP_DIR]
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
    for d in DRIVES:
        for m in MEANS:
            for s in SEEDS:
                tag = tag_for(d, m, s)
                p = os.path.join(SWEEP_DIR, f"readout_{tag}.json")
                row = dict(drive=d, core_mean=m, seed=s, status="missing", n_events=None,
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
                    except Exception as e:      # noqa: BLE001
                        row["status"] = f"parse_err:{e!r}"
                rows.append(row)
    return rows


def write_summary(rows):
    csvp = os.path.join(SWEEP_DIR, "sweep_summary.csv")
    with open(csvp, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-cell aggregate
    cells = {}
    for r in rows:
        cells.setdefault((r["drive"], r["core_mean"]), []).append(r)
    cand = [r for r in rows if r["coexist"]]

    L = []
    L.append("# twoend_equal diagnostic sweep — candidate working points\n")
    L.append(f"Grid: drive {DRIVES} x core_mean {MEANS} x seeds {SEEDS} (T={T}). "
             f"Coexist gate: neg_clean >= {COEXIST_MIN} AND pos_clean >= {COEXIST_MIN} in ONE record.\n")
    L.append("**DIAGNOSTIC ONLY** — candidate working points for the user to review; no scientific "
             "conclusion, no figure, no multi-seed long run. A 'coexist' cell is the prerequisite "
             "Stage 3 needs (both ends fire their own events in one network).\n")
    done = sum(1 for r in rows if r["status"] == "ok")
    L.append(f"\nRuns with readout: {done}/{len(rows)}. Coexist runs: **{len(cand)}**.\n")

    L.append("\n## Per-cell summary (drive x core_mean)\n")
    L.append("| drive | core_mean | seeds_ok | coexist/seeds | best min(neg,pos) | median collision_rate | n overheated(floor>0.04) | median n_events |")
    L.append("|---|---|---|---|---|---|---|---|")
    import statistics as st
    for d in DRIVES:
        for m in MEANS:
            cr = cells[(d, m)]
            ok = [r for r in cr if r["status"] == "ok"]
            n_co = sum(1 for r in ok if r["coexist"])
            best = max((min(r["neg_clean"] or 0, r["pos_clean"] or 0) for r in ok), default=0)
            colls = [r["collision_rate"] for r in ok if r["collision_rate"] is not None]
            floors = [r["true_floor"] for r in ok if r["true_floor"] is not None]
            nev = [r["n_events"] for r in ok if r["n_events"] is not None]
            n_hot = sum(1 for f in floors if f > 0.04)
            L.append(f"| {d:.2f} | {m:.1f} | {len(ok)}/{len(cr)} | {n_co}/{len(ok) or '-'} | {best} | "
                     f"{round(st.median(colls), 3) if colls else '-'} | {n_hot} | "
                     f"{int(st.median(nev)) if nev else '-'} |")

    L.append("\n## Coexist candidate runs (neg_clean >= 5 AND pos_clean >= 5)\n")
    if cand:
        L.append("| drive | core_mean | seed | neg_clean | pos_clean | collision | ambiguous | collision_rate | n_events |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(cand, key=lambda r: -min(r["neg_clean"], r["pos_clean"])):
            L.append(f"| {r['drive']:.2f} | {r['core_mean']:.1f} | {r['seed']} | {r['neg_clean']} | "
                     f"{r['pos_clean']} | {r['collision']} | {r['ambiguous']} | {r['collision_rate']} | {r['n_events']} |")
    else:
        L.append("**NONE** — no working point produced a record with both ends >= 5 clean events. "
                 "If so: the one-network mutual-suppression blocker is not relieved by drive/core_mean "
                 "alone in this grid; next levers = core spacing / local-event regime (option C) or "
                 "rethink the design. (Report the near-misses: highest best-min cells above.)")
    L.append("\n*(internal: stage3_source_counts gate; drive=nu_ext_ratio; "
             "see docs/archive/topic4/sef_hfo/stage3_twoend_equal_pilot_2026-06-13.md for the pilot.)*\n")

    mdp = os.path.join(SWEEP_DIR, "SWEEP_SUMMARY.md")
    open(mdp, "w").write("\n".join(L))
    print(f"[sweep] wrote {csvp} and {mdp}; coexist candidates: {len(cand)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(SWEEP_DIR, exist_ok=True)
    jobs = [(d, m, s) for d in DRIVES for m in MEANS for s in SEEDS]

    if a.collect_only:
        write_summary(collect())
        return
    if a.dry_run:
        for j in jobs[:4] + jobs[-1:]:
            print(run_one(*j, dry=True)[1])
        print(f"[sweep] DRY: {len(jobs)} total runs, {MAX_WORKERS} parallel")
        return

    print(f"[sweep] {len(jobs)} runs, {MAX_WORKERS} parallel, T={T}, out={SWEEP_DIR}", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            tag, status = f.result()
            done += 1
            print(f"[sweep] {done}/{len(jobs)} {tag} -> {status}", flush=True)
    write_summary(collect())
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
