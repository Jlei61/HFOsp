#!/usr/bin/env python3
"""B3 -- run B1 and B2 over every (subject, producer, seed) the registry offers.

Nothing here selects producers or patients by their result: the job list is the
cross product of "has an early ictal field and a held-out episode" with
"registered in the registry". Cells that cannot be estimated are recorded as
such, so the denominator of every later statement is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_h2b_transfer.registry import read_registry  # noqa: E402
from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    DEFAULT_POSTICTAL_EXCLUSION_SECONDS, group_seizure_episodes)

PY = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")
LEADS = ("5min", "30min", "2h", "6h")


def job_list(crosswalk: Path, data_root: Path, registry):
    sz: dict[str, list] = {}
    for r in csv.DictReader(crosswalk.open()):
        if r["disposition"] == "matched":
            sz.setdefault(r["subject"], []).append(
                {"seizure_id": r["seizure_id"], "onset_epoch": float(r["onset_epoch"]),
                 "offset_epoch": float(r["offset_epoch"])})
    jobs = []
    for subject, v in sorted(sz.items()):
        if not (data_root / "early_field" / f"{subject}.json").exists():
            continue
        v.sort(key=lambda s: s["onset_epoch"])
        eps = group_seizure_episodes(v, gap_seconds=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
        if len(eps) - max(1, math.ceil(len(eps) / 2)) < 1:
            continue  # no held-out episode under the rolling origin
        for pid, entry in registry.producers.items():
            seeds = entry.get("subjects", {}).get(subject)
            if not isinstance(seeds, dict):
                continue
            for seed in sorted(k for k in seeds if str(k).isdigit()):
                jobs.append((subject, pid, seed))
    return jobs


def run_one(args):
    subject, pid, seed, task, out_root = args
    script = {"b1": "run_b1_plumbing.py", "b2": "run_b2_field_transfer.py"}[task]
    cmd = [PY, str(ROOT / "scripts/topic5_h2b_transfer" / script),
           "--subject", subject, "--producer", pid, "--seed", str(seed),
           "--out-root", str(out_root)]
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return (subject, pid, seed, task, p.returncode, p.stderr[-300:] if p.returncode else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tasks", nargs="*", default=["b1", "b2"])
    args = ap.parse_args()

    reg = read_registry()
    jobs = job_list(args.crosswalk, args.data_root, reg)
    work = [(s, p, sd, t, args.out_root) for (s, p, sd) in jobs for t in args.tasks]
    print(f"registry: {reg.version}  producers={sorted(reg.producers)}")
    print(f"cells: {len(jobs)}  runs: {len(work)}  workers: {args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fails = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, w) for w in work]
        for f in as_completed(futs):
            s, pid, seed, task, rc, err = f.result()
            done += 1
            if rc != 0:
                fails.append((s, pid, seed, task, err))
            if done % 50 == 0:
                print(f"  {done}/{len(work)}", flush=True)
    print(f"done: {done}, failures: {len(fails)}")
    for f in fails[:10]:
        print("  FAIL", f[:4], f[4][:150])


if __name__ == "__main__":
    main()
