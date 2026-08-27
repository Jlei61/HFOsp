#!/usr/bin/env python3
"""Durable, recoverable H3-long queue launched after R1.5 completes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_REVISION,
    H3_LONG_SUPPORT_REVISION,
    SOURCES,
    SCALES,
    SYNTHETIC_TRUTHS,
)
from src.topic5_continuous_marked_state_r1.h3_long_human import (
    R1_5_REVISION,
    cell_package_fingerprint,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SEEDS = (0, 1, 2, 3, 4)
SYNTHETIC_SEEDS = (0, 1, 2)
_GPU_LAUNCH_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def result_complete(path: Path, *, subject: str, seed: int, source: str,
                    cell: dict, root: Path) -> bool:
    value = read(path) or {}
    try:
        fingerprint, components = cell_package_fingerprint(
            subject, seed, source, int(cell["scale_events"]), cell["role"],
            support_path=root / "support/summary.json",
            r1_5_root=contract.RESULT_ROOT / "r1_5",
            runner_path=(
                contract.REPO_ROOT
                / "scripts/topic5_continuous_marked_state_r1/run_h3_long_human.py"
            ),
        )
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == H3_LONG_REVISION
        and value.get("sealed_opened") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("subject") == subject
        and value.get("seed") == int(seed)
        and value.get("source") == source
        and value.get("scale_events") == int(cell["scale_events"])
        and value.get("support_role") == cell["role"]
        and value.get("package_fingerprint") == fingerprint
        and value.get("package_components") == components
    )


def instrument_ready(root: Path) -> tuple[bool, dict]:
    support_path = root / "support/summary.json"
    synthetic_path = root / "synthetic/synthetic_recovery.json"
    support = read(support_path) or {}
    synthetic = read(synthetic_path) or {}
    expected_support_hashes = {
        "producer": contract.sha256_file(
            contract.REPO_ROOT
            / "scripts/topic5_continuous_marked_state_r1/audit_r1_5_h3_long_support.py"
        ),
        "h3_long": contract.sha256_file(
            contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long.py"
        ),
        "contract": contract.sha256_file(
            contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/contract.py"
        ),
        "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
    }
    expected_synthetic_hashes = {
        "producer": contract.sha256_file(
            contract.REPO_ROOT
            / "scripts/topic5_continuous_marked_state_r1/run_h3_long_synthetic.py"
        ),
        "h3_long": expected_support_hashes["h3_long"],
    }
    actual_cells = {
        (int(row.get("scale_events", -1)), row.get("truth"),
         int(row.get("seed", -1)))
        for row in synthetic.get("rows", [])
    }
    expected_cells = {
        (scale, truth, seed) for scale in SCALES
        for truth in SYNTHETIC_TRUTHS for seed in SYNTHETIC_SEEDS
    }
    scheduled_identity = [
        (row.get("subject"), int(row.get("scale_events", -1)))
        for row in support.get("scheduled_cells", [])
    ]
    checks = {
        "support_status": support.get("status") == "COMPLETE",
        "support_revision": support.get("revision") == H3_LONG_SUPPORT_REVISION,
        "support_unsealed": support.get("sealed_opened") is False,
        "support_development_only": (
            support.get("formal_test_partition_opened") is False
            and support.get("development_time_contract_verified") is True
        ),
        "support_hashes": support.get("source_hashes") == expected_support_hashes,
        "support_cells_unique": len(scheduled_identity) == len(set(scheduled_identity)),
        "support_subjects_frozen": set(
            row[0] for row in scheduled_identity
        ) == set(contract.R1_5_EXTENSION_SUBJECTS),
        "synthetic_status": synthetic.get("status") == "COMPLETE",
        "synthetic_revision": synthetic.get("revision") == H3_LONG_REVISION,
        "synthetic_unsealed": (
            synthetic.get("sealed_opened") is False
            and synthetic.get("formal_test_partition_opened") is False
        ),
        "synthetic_hashes": synthetic.get("source_hashes") == expected_synthetic_hashes,
        "synthetic_cells_exact": actual_cells == expected_cells,
        "synthetic_all_pass": synthetic.get("all_cells_pass") is True,
    }
    return bool(all(checks.values())), checks


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
        return min(float(value.strip()) for value in output.splitlines())
    except Exception:
        return 0.0


def wait_resources() -> None:
    while available_gib() < 48.0 or gpu_free_mib() < 7000.0:
        time.sleep(20)


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY", "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def run(command: list[str], log: Path, *, gpu: bool = True,
        lock: Path | None = None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    wrapped = command
    if lock is not None:
        lock.parent.mkdir(parents=True, exist_ok=True)
        wrapped = ["flock", "-n", "-E", "75", str(lock), *command]
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        if gpu:
            with _GPU_LAUNCH_LOCK:
                wait_resources()
                process = subprocess.Popen(
                    wrapped, cwd=contract.REPO_ROOT, env=environment(),
                    stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=True,
                    start_new_session=True,
                )
                time.sleep(3.0)
        else:
            process = subprocess.Popen(
                wrapped, cwd=contract.REPO_ROOT, env=environment(),
                stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
        returncode = process.wait()
    return {
        "command": command, "log": str(log), "started": started,
        "finished": now(), "returncode": int(returncode),
    }


def task_complete(subject: str, seed: int, source: str,
                  root: Path, support: dict) -> bool:
    cells = [
        value for value in support["scheduled_cells"]
        if value["subject"] == subject
    ]
    return bool(cells) and all(result_complete(
        root / "human" / subject / source
        / f"seed_{seed}_n_{cell['scale_events']}/result.json",
        subject=subject, seed=seed, source=source, cell=cell, root=root,
    ) for cell in cells)


def fit(subject: str, seed: int, source: str,
        root: Path, support: dict) -> dict:
    if task_complete(subject, seed, source, root, support):
        return {"subject": subject, "seed": seed, "source": source,
                "status": "COMPLETE", "skipped": True}
    log = root / "logs/human" / f"{subject}_{source}_seed_{seed}.log"
    lock = root / "locks" / f"{subject}_{source}_seed_{seed}.lock"
    value = None
    retries = []
    for batch_size in (4096, 2048, 1024, 512):
        value = run([
            str(PYTHON),
            "scripts/topic5_continuous_marked_state_r1/run_h3_long_human.py",
            "--subject", subject, "--seed", str(seed), "--source", source,
            "--device", "cuda", "--batch-size", str(batch_size),
            "--r1-5-root", str(contract.RESULT_ROOT / "r1_5"),
            "--support", str(root / "support/summary.json"),
            "--output-root", str(root),
        ], log, lock=lock)
        retries.append({"batch_size": batch_size,
                        "returncode": value["returncode"]})
        if value["returncode"] == 0:
            break
        if value["returncode"] == 75:
            # Another durable queue owns this exact task.  Wait for its
            # fingerprinted result rather than launching a duplicate.
            deadline = time.time() + 12 * 3600
            while time.time() < deadline and not task_complete(
                subject, seed, source, root, support
            ):
                time.sleep(60)
            break
        tail = log.read_text(errors="replace")[-20000:]
        if not any(token in tail for token in (
            "CUDA out of memory", "torch.OutOfMemoryError",
            "CUBLAS_STATUS_ALLOC_FAILED",
        )):
            break
    assert value is not None
    value.update({"subject": subject, "seed": seed, "source": source})
    value["batch_retries"] = retries
    value["status"] = (
        "COMPLETE" if task_complete(subject, seed, source, root, support)
        else "FAIL"
    )
    return value


def write_status(root: Path, stage: str, *, rows=None, state="RUNNING") -> None:
    support = read(root / "support/summary.json") or {"scheduled_cells": []}
    completed = sum(
        result_complete(
            root / "human" / cell["subject"] / source
            / f"seed_{seed}_n_{cell['scale_events']}/result.json",
            subject=cell["subject"], seed=seed, source=source,
            cell=cell, root=root,
        )
        for cell in support["scheduled_cells"]
        for seed in SEEDS for source in SOURCES
    )
    expected = len(support["scheduled_cells"]) * len(SEEDS) * len(SOURCES)
    contract.atomic_json(root / "STATUS.json", {
        "status": state, "stage": stage, "revision": H3_LONG_REVISION,
        "completed_cells": completed, "expected_cells": expected,
        "last_rows": rows or [], "updated_at": now(),
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long",
    )
    args = parser.parse_args()
    write_status(args.root, "waiting_for_r1_5")
    while True:
        r1 = read(contract.RESULT_ROOT / "r1_5/STATUS.json") or {}
        if (
            r1.get("status") == "COMPLETE"
            and r1.get("revision") == R1_5_REVISION
            and r1.get("formal_test_partition_opened") is False
            and r1.get("sealed_opened") is False
        ):
            break
        if r1.get("stage", "").endswith("_fail"):
            write_status(args.root, "r1_5_failed", rows=[r1], state="FAIL")
            raise RuntimeError("R1.5 failed")
        # R1.5 can legitimately outlive one wall-clock day under shared-GPU
        # contention.  Keep the durable queue alive and refresh the handoff
        # instead of turning resource contention into a scientific failure.
        write_status(args.root, "waiting_for_r1_5", rows=[r1])
        time.sleep(60)
    support = read(args.root / "support/summary.json") or {}
    ready, instrument_checks = instrument_ready(args.root)
    r1_summary = read(
        contract.RESULT_ROOT / "r1_5/reports/r1_5_summary.json"
    ) or {}
    r1_ready = bool(
        r1_summary.get("status") == "COMPLETE"
        and r1_summary.get("revision") == R1_5_REVISION
        and r1_summary.get("formal_test_partition_opened") is False
        and r1_summary.get("sealed_opened") is False
    )
    if not ready or not r1_ready:
        write_status(
            args.root, "instrument_not_ready",
            rows=[{"instrument_checks": instrument_checks,
                   "r1_summary_ready": r1_ready}], state="FAIL",
        )
        raise RuntimeError("H3-long instrument not ready")
    subjects = sorted({
        value["subject"] for value in support["scheduled_cells"]
    })
    tasks = [
        (subject, seed, source, args.root, support)
        for subject in subjects for seed in SEEDS for source in SOURCES
    ]
    write_status(args.root, "human")
    rows = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        future = {pool.submit(fit, *task): task for task in tasks}
        for item in as_completed(future):
            try:
                rows.append(item.result())
            except Exception as error:
                rows.append({"task": list(future[item]), "status": "FAIL",
                             "error": repr(error)})
            write_status(args.root, "human", rows=rows[-12:])
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(args.root, "human_fail", rows=rows, state="FAIL")
        raise RuntimeError("H3-long human queue failed")
    write_status(args.root, "aggregate", rows=rows[-12:])
    aggregate = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/aggregate_h3_long.py",
        "--root", str(args.root), "--r1-5-root",
        str(contract.RESULT_ROOT / "r1_5"),
    ], args.root / "logs/aggregate.log", gpu=False)
    if aggregate["returncode"]:
        write_status(args.root, "aggregate_fail", rows=[aggregate], state="FAIL")
        raise RuntimeError("H3-long aggregation failed")
    write_status(args.root, "complete", rows=[aggregate], state="COMPLETE")


if __name__ == "__main__":
    main()
