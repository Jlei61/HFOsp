#!/usr/bin/env python3
"""Durable preparation and five-seed R1.7A prospective replication queue."""
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
from src.topic5_continuous_marked_state_r1.r1_7 import R1_7A_REVISION


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SEEDS = tuple(range(5))


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


def prepare(subject: str, root: Path, upstream: Path) -> dict:
    commands = [
        (
            upstream / "baselines" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_baseline.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--mark-batch-size", "512", "--output-root", str(upstream)],
            6500.0, "baseline",
        ),
        (
            upstream / "bridge_e1" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_bridge.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--anchor-batch-size", "2", "--max-train-anchors", "64",
             "--max-validation-anchors", "32", "--output-root", str(upstream)],
            7000.0, "bridge",
        ),
        (
            upstream / "cache" / subject / "manifest.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_cache.py",
             "--subject", subject, "--device", "cuda", "--anchor-batch-size", "4",
             "--output-root", str(upstream)],
            7000.0, "anchor_cache",
        ),
        (
            root / "cache" / subject / "manifest.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/build_r1_3_observation_cache.py",
             "--subject", subject, "--r1-2-root", str(upstream),
             "--output-root", str(root)],
            1000.0, "observation_cache",
        ),
    ]
    history = []
    for output, command, memory, label in commands:
        if complete(output):
            history.append({"stage": label, "skipped": True, "output": str(output)})
            continue
        value = run(command, root / "logs/preparation" / f"{subject}_{label}.log",
                    min_gpu_mib=memory)
        value.update({"stage": label, "output": str(output)})
        history.append(value)
        if value["returncode"] != 0 or not complete(output):
            return {"status": "FAIL", "subject": subject, "history": history}
    return {"status": "COMPLETE", "subject": subject, "history": history}


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
    if complete(output, R1_7A_REVISION):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_7a_cell.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--r1-2-root", str(upstream), "--output-root", str(root),
        "--r1-6-root", str(r1_6),
    ]
    value = run(command, root / "logs/fits" / f"{subject}_seed_{seed}.log",
                min_gpu_mib=6500.0)
    value["output"] = str(output)
    value["status"] = "COMPLETE" if value["returncode"] == 0 and complete(output, R1_7A_REVISION) else "FAIL"
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


def require(rows: list[dict], stage: str, status_path: Path) -> None:
    contract.atomic_json(status_path, {
        "status": "COMPLETE" if all(r.get("status") == "COMPLETE" for r in rows) else "FAIL",
        "stage": stage, "updated_utc": now(), "rows": rows,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })
    if any(row.get("status") != "COMPLETE" for row in rows):
        raise RuntimeError(f"R1.7A {stage} failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    parser.add_argument("--source-r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2")
    parser.add_argument("--r1-6-root", type=Path,
                        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / "QUEUE.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.7A queue already running") from error
    inventory = json.loads((args.output_root / "manifests/cohort_inventory.json").read_text())
    subjects = tuple(inventory["selected_subjects"])
    upstream = args.output_root / "upstream_r1_2"
    (upstream / "coverage").mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for suffix in (".npz", ".manifest.json"):
            source = args.source_r1_2_root / "coverage" / f"{subject}{suffix}"
            target = upstream / "coverage" / source.name
            if not target.exists():
                target.symlink_to(source.resolve())
    frozen = json.loads((args.r1_6_root / "reports/recommended_optimizer_config.json").read_text())
    status = args.output_root / "QUEUE_STATUS.json"
    rows = parallel(prepare, [(s, args.output_root, upstream) for s in subjects], args.workers)
    require(rows, "preparation", status)
    rows = parallel(prefix, [
        (s, seed, args.output_root, upstream, frozen["prefix_core"])
        for s in subjects for seed in SEEDS
    ], args.workers)
    require(rows, "prefix", status)
    rows = parallel(fit, [
        (s, seed, args.output_root, upstream, args.r1_6_root)
        for s in subjects for seed in SEEDS
    ], args.workers)
    require(rows, "fits", status)
    contract.atomic_json(status, {
        "status": "COMPLETE", "stage": "R1.7A_COMPLETE", "updated_utc": now(),
        "n_subjects": len(subjects), "n_seeds": len(SEEDS),
        "scheduled_cells": len(subjects) * len(SEEDS),
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


if __name__ == "__main__":
    main()
