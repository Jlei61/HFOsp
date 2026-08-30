#!/usr/bin/env python3
"""Prepare upstream artifacts for the R1.7B extended-cohort subjects only.

This stage is identical to the R1.7A queue's preparation step and writes into a
separate output root, so it can run alongside the frozen R1.7A fits without
touching any R1.7A artifact or any hashed source file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


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


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "r1_7b_cohort_extension")
    parser.add_argument("--source-r1-2-root", type=Path,
                        default=contract.RESULT_ROOT / "r1_2")
    parser.add_argument("--only-added", action="store_true",
                        help="prepare only subjects absent from frozen R1.7A")
    args = parser.parse_args()
    inventory = json.loads(
        (args.output_root / "manifests/cohort_inventory.json").read_text()
    )
    subjects = tuple(inventory["added_subjects"] if args.only_added
                     else inventory["selected_subjects"])
    upstream = args.output_root / "upstream_r1_2"
    (upstream / "coverage").mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for suffix in (".npz", ".manifest.json"):
            source = args.source_r1_2_root / "coverage" / f"{subject}{suffix}"
            target = upstream / "coverage" / source.name
            if not target.exists():
                target.symlink_to(source.resolve())
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, s, args.output_root, upstream): s
                   for s in subjects}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({"status": "FAIL", "subject": futures[future],
                             "error": repr(error)})
    status = {
        "status": "COMPLETE" if all(r.get("status") == "COMPLETE" for r in rows) else "FAIL",
        "stage": "R1.7B_PREPARATION", "updated_utc": now(),
        "n_subjects": len(subjects), "subjects": list(subjects), "rows": rows,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(args.output_root / "PREPARATION_STATUS.json", status)
    if status["status"] != "COMPLETE":
        raise RuntimeError("R1.7B preparation failed")


if __name__ == "__main__":
    main()
