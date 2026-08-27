#!/usr/bin/env python3
"""Prefix + fit queue for the exploratory R1.7B extended cohort.

R1.7B widens the frozen R1.7A replication along the two axes that were the
stated limitation of the previous rounds: more development subjects (the
top-five-per-dataset cap is removed, 10 -> 17) and more seeds per subject
(5 -> 10).  Everything else -- the model, the frozen R1.6 optimiser
configuration, the 60/40 recorded-time split, the stability criteria and the
matched wrong-time construction -- is unchanged.

R1.7B is an exploratory extension.  It never rewrites the pre-registered R1.7A
result, and it writes into its own output root so R1.7A keeps exactly 50 cells
with a single source payload.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
R1_7B_REVISION = "r1_7b_extended_development_cohort_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT), "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def gpu_free_mib() -> float:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        return min(float(value) for value in output.splitlines())
    except Exception:
        return 0.0


def run(command: list[str], log: Path, *, min_gpu_mib: float) -> dict:
    while available_gib() < 40.0 or gpu_free_mib() < min_gpu_mib:
        time.sleep(20.0)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n"); handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, text=True,
        )
    return {"command": command, "log": str(log), "started": started,
            "finished": now(), "returncode": int(process.returncode)}


def complete(path: Path, revision: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and (revision is None or value.get("revision") == revision)
        and value.get("formal_test_partition_opened", False) is False
        and value.get("sealed_opened") is False
    )


def prefix(subject: str, seed: int, root: Path, upstream: Path, cfg: dict) -> dict:
    cid = cfg["config_id"]
    output = root / "prefix_initialisation" / cid / subject / f"seed_{seed}/result.json"
    if complete(output):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/build_r1_6_prefix_initialisation.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", str(cfg["epochs"]), "--learning-rate", str(cfg["lr"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--warmup-fraction", str(cfg["warmup"]),
        "--grad-clip-norm", str(cfg["clip"]), "--chunk-anchors", str(cfg["chunk"]),
        "--optimizer", str(cfg["optimizer"]), "--config-id", cid,
        "--selection-min-delta", str(cfg["min_delta"]),
        "--early-stopping-patience", str(cfg["patience"]),
        "--r1-2-root", str(upstream), "--output-root", str(root),
    ]
    value = run(command, root / "logs/prefix" / f"{subject}_seed_{seed}.log",
                min_gpu_mib=5000.0)
    value["output"] = str(output)
    value["status"] = "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    return value


def fit(subject: str, seed: int, root: Path, upstream: Path, r1_6: Path) -> dict:
    output = root / "fits" / subject / f"seed_{seed}/result.json"
    if complete(output, R1_7B_REVISION):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_7a_cell.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--r1-2-root", str(upstream), "--output-root", str(root),
        "--r1-6-root", str(r1_6), "--revision", R1_7B_REVISION,
    ]
    value = run(command, root / "logs/fits" / f"{subject}_seed_{seed}.log",
                min_gpu_mib=6500.0)
    value["output"] = str(output)
    value["status"] = ("COMPLETE" if value["returncode"] == 0
                       and complete(output, R1_7B_REVISION) else "FAIL")
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({"status": "FAIL", "task": list(map(str, futures[future])),
                             "error": repr(error)})
    return rows


def record(rows: list[dict], stage: str, status_path: Path) -> None:
    contract.atomic_json(status_path, {
        "status": "COMPLETE" if all(r.get("status") == "COMPLETE" for r in rows) else "FAIL",
        "stage": stage, "updated_utc": now(), "rows": rows,
        "revision": R1_7B_REVISION,
        "exploratory_extension_not_preregistered_replication": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })
    if any(row.get("status") != "COMPLETE" for row in rows):
        raise RuntimeError(f"R1.7B {stage} failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "r1_7b_cohort_extension")
    parser.add_argument("--r1-6-root", type=Path,
                        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6")
    parser.add_argument("--prefix-only", action="store_true",
                        help="run only the prefix stage (needs no R1.7B fit runner)")
    args = parser.parse_args()
    lock = (args.output_root / "QUEUE.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.7B queue already running") from error
    inventory = json.loads(
        (args.output_root / "manifests/cohort_inventory.json").read_text()
    )
    subjects = tuple(inventory["selected_subjects"])
    seeds = tuple(range(int(args.seeds)))
    upstream = args.output_root / "upstream_r1_2"
    (upstream / "coverage").mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for suffix in (".npz", ".manifest.json"):
            source = contract.RESULT_ROOT / "r1_2" / "coverage" / f"{subject}{suffix}"
            target = upstream / "coverage" / source.name
            if not target.exists():
                target.symlink_to(source.resolve())
    frozen = json.loads(
        (args.r1_6_root / "reports/recommended_optimizer_config.json").read_text()
    )
    status = args.output_root / "QUEUE_STATUS.json"
    rows = parallel(prefix, [
        (s, seed, args.output_root, upstream, frozen["prefix_core"])
        for s in subjects for seed in seeds
    ], args.workers)
    record(rows, "prefix", status)
    if args.prefix_only:
        return
    rows = parallel(fit, [
        (s, seed, args.output_root, upstream, args.r1_6_root)
        for s in subjects for seed in seeds
    ], args.workers)
    record(rows, "fits", status)
    contract.atomic_json(status, {
        "status": "COMPLETE", "stage": "R1.7B_COMPLETE", "updated_utc": now(),
        "revision": R1_7B_REVISION,
        "n_subjects": len(subjects), "n_seeds": len(seeds),
        "scheduled_cells": len(subjects) * len(seeds),
        "exploratory_extension_not_preregistered_replication": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


if __name__ == "__main__":
    main()
