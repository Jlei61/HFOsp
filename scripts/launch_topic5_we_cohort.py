"""Run WE-SLP-RNN v0.3 training units with a bounded worker pool.

A python orchestrator, not a shell script: bash resumes a running script by byte
offset, so editing one mid-run shifts every offset after the edit and the shell
silently restarts inside the middle of a command.  On 2026-08-07 that skipped
three analysis stages of the v0.2 run and every acceptance check still passed,
because none of them asked whether the outputs were current.

The duplicate guard compares absolute output directories, and checks both
finished work and work already in flight.  The v0.2 guard had to be tightened
three times: it did not look at in-flight jobs, then it looked but ignored the
output directory, then it compared strings where one side was relative.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

RNN_ARMS = {"STATIC_CONTACT": [0], "DENSE_TISSUE": [0],
            "RANDOM_SET": [0, 1, 2], "SPATIAL_SET": [0, 1, 2]}
# The two off-diagonal cells of the growth x cost square, one seed each: they
# only have to say which half of wiring economy did the work, not carry a claim.
FACTORIAL_ARMS = {"RANDOM_SET_COST": [0], "SPATIAL_SET_NOCOST": [0]}
GRU_ARMS = {"DENSE_TISSUE": [0], "RANDOM_SET": [0], "SPATIAL_SET": [0]}


def fits(out_root: Path) -> List[Dict[str, Any]]:
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    return manifest["fits"]


def development_subjects(out_root: Path, n: int = 8) -> List[str]:
    """Patients used to freeze eta, chosen by montage size only.

    Contact count is a property of the electrode layout, fixed long before any
    model was trained, so selecting on it cannot be informed by a result.
    """
    rows = sorted(fits(out_root), key=lambda r: (r["n_contacts"], r["fit_id"]))
    seen, per_subject = set(), []
    for row in rows:
        if row["subject"] not in seen:
            seen.add(row["subject"])
            per_subject.append(row["fit_id"])
    # Evenly spaced across the whole montage-size range, so the largest networks
    # -- where a wiring penalty has the most edges to act on -- are represented.
    positions = [round(i * (len(per_subject) - 1) / (n - 1)) for i in range(n)]
    return [per_subject[p] for p in sorted(set(positions))]


def job_directory(out_root: Path, fit_id: str, arm: str, cell: str, seed: int,
                  shuffled: bool, tag: str = "") -> Path:
    suffix = "_shuffled" if shuffled else ""
    return out_root / "per_subject" / fit_id / f"{arm}{suffix}_{cell}{tag}" / f"seed{seed}"


def build_jobs(out_root: Path, batch: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    all_fits = [r["fit_id"] for r in fits(out_root)]
    if batch == "eta_sweep":
        for fit_id in development_subjects(out_root):
            for eta in (0.003, 0.01, 0.03, 0.1, 0.3):
                jobs.append({"fit_id": fit_id, "arm": "SPATIAL_SET", "cell": "rnn",
                             "seed": 0, "eta": eta,
                             "tag": f"__eta{eta:g}".replace(".", "p")})
    elif batch == "state_probe":
        for fit_id in development_subjects(out_root):
            jobs.append({"fit_id": fit_id, "arm": "SPATIAL_SET", "cell": "rnn",
                         "seed": 0, "state_dim": 2, "tag": "__dim2"})
    elif batch == "rnn_main":
        for fit_id in all_fits:
            for arm, seeds in RNN_ARMS.items():
                for seed in seeds:
                    jobs.append({"fit_id": fit_id, "arm": arm, "cell": "rnn", "seed": seed})
    elif batch == "factorial":
        for fit_id in all_fits:
            for arm, seeds in FACTORIAL_ARMS.items():
                for seed in seeds:
                    jobs.append({"fit_id": fit_id, "arm": arm, "cell": "rnn", "seed": seed})
    elif batch == "shuffled":
        for fit_id in all_fits:
            jobs.append({"fit_id": fit_id, "arm": "SPATIAL_SET", "cell": "rnn",
                         "seed": 0, "shuffled": True})
    elif batch == "gru":
        for fit_id in all_fits:
            for arm, seeds in GRU_ARMS.items():
                for seed in seeds:
                    jobs.append({"fit_id": fit_id, "arm": arm, "cell": "gru", "seed": seed})
    elif batch == "gru_seeds":
        # Pre-registered escalation: the one-seed gated-cell replication
        # disagreed in direction with the primary cell, so the contrast it
        # disagreed about is rerun at the primary cell's three seeds before the
        # disagreement is called anything.
        for fit_id in all_fits:
            for arm in ("RANDOM_SET", "SPATIAL_SET"):
                for seed in (1, 2):
                    jobs.append({"fit_id": fit_id, "arm": arm, "cell": "gru", "seed": seed})
    elif batch == "density":
        for fit_id in all_fits:
            for arm in ("RANDOM_SET", "SPATIAL_SET"):
                for density in (0.05, 0.20):
                    jobs.append({"fit_id": fit_id, "arm": arm, "cell": "rnn", "seed": 0,
                                 "density": density,
                                 "tag": f"__rho{density:g}".replace(".", "p")})
    else:
        raise ValueError(f"unknown batch {batch!r}")
    for job in jobs:
        job["out_dir"] = job_directory(out_root, job["fit_id"], job["arm"], job["cell"],
                                       job["seed"], job.get("shuffled", False),
                                       job.get("tag", ""))
    return jobs


def command(job: Dict[str, Any], out_root: Path, device: str, eta: float | None) -> List[str]:
    cmd = [PYTHON, str(ROOT / "scripts/train_topic5_we_unit.py"),
           "--fit-id", job["fit_id"], "--arm", job["arm"], "--cell", job["cell"],
           "--seed", str(job["seed"]), "--device", device,
           "--out-root", str(out_root), "--out-tag", job.get("tag", "")]
    if job.get("shuffled"):
        cmd.append("--shuffled")
    if "eta" in job:
        cmd += ["--eta", str(job["eta"])]
    elif eta is not None:
        cmd += ["--eta", str(eta)]
    if "density" in job:
        cmd += ["--density", str(job["density"])]
    if "state_dim" in job:
        cmd += ["--state-dim", str(job["state_dim"])]
    return cmd


def in_flight() -> set[str]:
    """Absolute output directories of training units already running."""
    try:
        out = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return set()
    running = set()
    for line in out.splitlines():
        if "train_topic5_we_unit.py" not in line:
            continue
        parts = line.split()
        fields = {}
        for i, token in enumerate(parts):
            if token.startswith("--") and i + 1 < len(parts):
                fields[token] = parts[i + 1]
        if "--fit-id" in fields and "--arm" in fields:
            root = Path(fields.get("--out-root", str(OUT_ROOT))).resolve()
            running.add(str(job_directory(
                root, fields["--fit-id"], fields["--arm"], fields.get("--cell", "rnn"),
                int(fields.get("--seed", 0)), "--shuffled" in line,
                fields.get("--out-tag", ""))))
    return running


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    jobs = build_jobs(out_root, args.batch)
    running_dirs = in_flight()
    todo = [j for j in jobs
            if not (j["out_dir"] / "DONE.json").exists()
            and str(j["out_dir"].resolve()) not in running_dirs]
    print(f"batch={args.batch}: {len(jobs)} units, {len(jobs) - len(todo)} already done "
          f"or in flight, {len(todo)} to run on {args.device} with {args.max_workers} workers")
    if args.dry_run:
        for job in todo[:10]:
            print("  ", " ".join(command(job, out_root, args.device, args.eta)))
        if len(todo) > 10:
            print(f"   ... and {len(todo) - 10} more")
        return 0

    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log_dir = out_root / "logs" / args.batch
    log_dir.mkdir(parents=True, exist_ok=True)
    active: List[tuple[subprocess.Popen, Dict[str, Any], Any, float]] = []
    done = failed = 0
    started = time.time()

    def reap() -> None:
        nonlocal done, failed
        for entry in list(active):
            process, job, handle, t0 = entry
            if process.poll() is None:
                continue
            handle.close()
            active.remove(entry)
            if process.returncode == 0 and (job["out_dir"] / "DONE.json").exists():
                done += 1
            else:
                failed += 1
                print(f"  FAILED rc={process.returncode} {job['fit_id']} {job['arm']} "
                      f"{job['cell']} seed{job['seed']}{job.get('tag', '')}")

    for job in todo:
        while len(active) >= args.max_workers:
            reap()
            time.sleep(2.0)
        name = (f"{job['fit_id']}__{job['arm']}{'_shuffled' if job.get('shuffled') else ''}"
                f"_{job['cell']}{job.get('tag', '')}_seed{job['seed']}")
        handle = (log_dir / f"{name}.log").open("w")
        process = subprocess.Popen(command(job, out_root, args.device, args.eta),
                                   stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        active.append((process, job, handle, time.time()))
        if (done + failed) % 10 == 0 and done + failed:
            rate = (done + failed) / max(1e-9, time.time() - started)
            print(f"  {done} done, {failed} failed, {len(todo) - done - failed} left, "
                  f"{rate * 3600:.0f}/h", flush=True)

    while active:
        reap()
        time.sleep(2.0)
    print(f"batch={args.batch}: {done} done, {failed} failed in "
          f"{(time.time() - started) / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
