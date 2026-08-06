"""Run (patient, variant, seed) units in parallel, with a concurrency the box holds.

v0.1 lost fourteen units to GPU exhaustion because four launchers queued over the
same cohort at once.  This one is CPU-only and single-owner: one process, a fixed
worker count, and units that already have a DONE.json are skipped so a rerun
resumes instead of repeating.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
PY = sys.executable

VARIANTS = ("STATIC", "FIELD_NULL", "ISOTROPIC_DIFFUSION",
            "ANISOTROPIC_DRIFT", "ANISOTROPIC_RECOVERY")


def unit_dir(root: Path, subject: str, variant: str, seed: int) -> Path:
    return root / subject / variant / f"seed{seed}"


def is_running(subject: str, variant: str, seed: int) -> bool:
    """Is a trainer already working on this exact unit?"""
    pattern = (f"train_topic5_spo_unit.py --subject {subject} "
               f"--variant {variant} --seed {seed}")
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--out-root", type=Path, default=OUT / "per_subject")
    parser.add_argument("--log", type=Path, default=OUT / "cohort_run.log")
    args = parser.parse_args()

    subjects = args.subjects or json.loads(
        (OUT / "INPUT_MANIFEST.json").read_text())["frozen_cohort"]["primary"]

    plan = [(s, v, seed) for s in subjects for v in args.variants for seed in args.seeds]
    pending = [u for u in plan
               if not (unit_dir(args.out_root, *u) / "DONE.json").exists()]
    # A unit with no DONE.json is not necessarily free -- a launcher from an
    # earlier invocation may still have it in flight. Starting a second trainer
    # on it wastes a core and lets two processes write the same DONE.json. This
    # already happened once, and it surfaced as failure markers sitting beside
    # completed units rather than as an error.
    inflight = {u for u in pending if is_running(*u)}
    todo = [u for u in pending if u not in inflight]
    print(f"planned {len(plan)} units, {len(plan) - len(pending)} already done, "
          f"{len(inflight)} already in flight elsewhere, "
          f"{len(todo)} to run, {args.workers} workers", flush=True)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    done = {"n": 0, "failed": 0}

    def run(unit) -> None:
        subject, variant, seed = unit
        command = [PY, str(ROOT / "scripts/train_topic5_spo_unit.py"),
                   "--subject", subject, "--variant", variant, "--seed", str(seed),
                   "--out-root", str(args.out_root)]
        if args.config:
            command += ["--config", str(args.config)]
        if args.holdout_fraction > 0:
            command += ["--holdout-fraction", str(args.holdout_fraction)]
        started = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        done["n"] += 1
        if result.returncode != 0:
            done["failed"] += 1
        with args.log.open("a") as handle:
            handle.write(f"{subject}\t{variant}\tseed{seed}\t"
                         f"{time.time() - started:.0f}s\trc={result.returncode}\t"
                         f"{result.stdout.strip()[-160:]}\n")
        print(f"[{done['n']}/{len(todo)}] {subject} {variant} seed{seed} "
              f"({time.time() - started:.0f}s) rc={result.returncode}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, todo))

    print(f"complete: {done['n']}/{len(todo)}   failed: {done['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
