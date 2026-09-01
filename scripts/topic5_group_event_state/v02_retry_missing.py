#!/usr/bin/env python3
"""Re-run only the (subject, producer, seed) cells that have no result yet.

``config_hash`` deliberately includes the source commit, so a result always
matches the code that produced it -- but that also means a plain resume after any
commit would re-run the whole queue.  This helper scopes a resume to the cells
that are genuinely missing, and prints what it is about to do before doing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main")
DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--producers", nargs="+", default=["P_local", "P_slow"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists())
    missing: list[tuple[str, str, int]] = []
    for subject in subjects:
        for producer in args.producers:
            for seed in args.seeds:
                run = (args.producer_root / "runs" / subject / producer / f"seed{seed}")
                if not (run / "result.json").exists():
                    missing.append((subject, producer, seed))
    print(json.dumps({"n_missing": len(missing), "missing": missing}, indent=2))
    if not missing or args.dry_run:
        return

    # One subprocess per missing cell, sequential across cells but pinned across
    # cards; the volume here is small by construction.
    failures = []
    for i, (subject, producer, seed) in enumerate(missing):
        gpu = args.gpus[i % len(args.gpus)]
        cmd = [sys.executable,
               str(ROOT / "scripts/topic5_group_event_state/v02_train_producers.py"),
               "--job", "--subject", subject, "--producer", producer,
               "--seed", str(seed), "--gpu", str(gpu),
               "--out-root", str(args.producer_root.parent), "--tag", args.tag]
        print(f"[{i + 1}/{len(missing)}] {subject}/{producer}/seed{seed} gpu{gpu}",
              flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append({"subject": subject, "producer": producer, "seed": seed,
                             "returncode": rc})
    print(json.dumps({"n_retried": len(missing), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
