#!/usr/bin/env python3
"""Recoverable frozen-config confirmation queue for R1.6."""
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
from src.topic5_continuous_marked_state_r1.optimizer_audit import R1_6_REVISION
from scripts.topic5_continuous_marked_state_r1.run_r1_6_optimizer_confirmation_cell import (
    CONFIRMATION_REVISION,
    FIXED_SUBJECTS,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
CONFIRMATION_SEEDS = (3, 4)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def wait_for_resources() -> None:
    while available_gib() < 32.0 or gpu_free_mib() < 2500.0:
        time.sleep(15.0)


def run(command: list[str], log: Path) -> dict:
    wait_for_resources()
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, start_new_session=True,
        )
    return {
        "command": command, "log": str(log), "started": started,
        "finished": now(), "returncode": int(process.returncode),
    }


def valid_prefix(path: Path, prefix_config: str,
                 subject: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_REVISION
        and value.get("stage") == "prefix_initialisation"
        and value.get("config_id") == prefix_config
        and value.get("subject") == subject and value.get("seed") == seed
        and value.get("epoch_zero_seen_alignment_selection") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
    )


def valid_confirmation(path: Path, selected_prefix_config: str,
                       selected_config: str,
                       subject: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_REVISION
        and value.get("confirmation_revision") == CONFIRMATION_REVISION
        and value.get("selected_prefix_config") == selected_prefix_config
        and value.get("selected_config") == selected_config
        and value.get("subject") == subject and value.get("seed") == seed
        and value.get("development_validation_scored") is True
        and value.get("development_validation_used_for_selection") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
    )


def prefix_task(root: Path, selected_prefix_config: str,
                prefix_config: dict, subject: str, seed: int) -> dict:
    output = (
        root / "prefix_initialisation" / selected_prefix_config
        / subject / f"seed_{seed}/result.json"
    )
    if valid_prefix(output, selected_prefix_config, subject, seed):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_6_prefix_initialisation.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", str(prefix_config["epochs"]),
        "--learning-rate", str(prefix_config["lr"]),
        "--weight-decay", str(prefix_config["weight_decay"]),
        "--warmup-fraction", str(prefix_config["warmup"]),
        "--grad-clip-norm", str(prefix_config["clip"]),
        "--chunk-anchors", str(prefix_config["chunk"]),
        "--optimizer", str(prefix_config["optimizer"]),
        "--config-id", selected_prefix_config, "--output-root", str(root),
    ]
    value = run(
        command, root / "logs/confirmation_prefix" / f"{subject}_seed_{seed}.log"
    )
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0 and valid_prefix(
            output, selected_prefix_config, subject, seed
        )
    ) else "FAIL"
    return value


def confirmation_task(root: Path, selected_prefix_config: str,
                      selected_config: str,
                      subject: str, seed: int) -> dict:
    output = (
        root / "confirmation" / selected_prefix_config / selected_config
        / subject / f"seed_{seed}/result.json"
    )
    if valid_confirmation(
        output, selected_prefix_config, selected_config, subject, seed
    ):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_confirmation_cell.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--output-root", str(root),
    ]
    value = run(
        command,
        root / "logs/confirmation" / selected_config
        / f"{subject}_seed_{seed}.log",
    )
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0
        and valid_confirmation(
            output, selected_prefix_config, selected_config, subject, seed
        )
    ) else "FAIL"
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "status": "FAIL", "task": list(futures[future]),
                    "error": repr(error),
                })
    return rows


def require(rows: list[dict], stage: str, root: Path,
            selected_prefix_config: str | None = None,
            selected_config: str | None = None) -> None:
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(
            root, f"{stage}_fail", selected_prefix_config,
            selected_config, rows,
        )
        raise RuntimeError(f"R1.6 confirmation {stage} failed")


def write_status(root: Path, stage: str,
                 selected_prefix_config: str | None,
                 selected_config: str | None,
                 rows: list[dict] | None = None) -> None:
    completed_prefix = sum(
        valid_prefix(
            root / "prefix_initialisation" / str(selected_prefix_config)
            / subject / f"seed_{seed}/result.json",
            str(selected_prefix_config), subject, seed,
        )
        for subject in FIXED_SUBJECTS for seed in CONFIRMATION_SEEDS
    ) if selected_prefix_config is not None else 0
    completed_confirmation = 0
    if selected_config is not None:
        completed_confirmation = sum(
            valid_confirmation(
                root / "confirmation" / str(selected_prefix_config)
                / selected_config
                / subject / f"seed_{seed}/result.json",
                str(selected_prefix_config), selected_config, subject, seed,
            )
            for subject in FIXED_SUBJECTS for seed in CONFIRMATION_SEEDS
        )
    contract.atomic_json(root / "CONFIRMATION_STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage, "revision": R1_6_REVISION,
        "confirmation_revision": CONFIRMATION_REVISION,
        "selected_prefix_config": selected_prefix_config,
        "selected_config": selected_config,
        "subjects": list(FIXED_SUBJECTS),
        "seeds": list(CONFIRMATION_SEEDS),
        "completed_prefix": int(completed_prefix),
        "expected_prefix": len(FIXED_SUBJECTS) * len(CONFIRMATION_SEEDS),
        "completed_confirmation": int(completed_confirmation),
        "expected_confirmation": len(FIXED_SUBJECTS) * len(CONFIRMATION_SEEDS),
        "last_rows": rows or [], "updated_at": now(),
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.root / "confirmation_queue.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.6 confirmation queue is already running") from error
    lock_handle.write(f"pid={os.getpid()} started={now()}\n")
    lock_handle.flush()

    tuning_path = args.root / "reports/tuning_summary.json"
    tuning = json.loads(tuning_path.read_text())
    selected_config = str(tuning["selected_config"])
    selected_prefix_config = str(tuning["selected_prefix_config"])
    prefix_status = json.loads(
        (args.root / "PREFIX_TUNING_STATUS.json").read_text()
    )
    prefix_config = prefix_status["configs"][selected_prefix_config]
    write_status(
        args.root, "confirmation_prefix",
        selected_prefix_config, selected_config,
    )
    rows = parallel(
        prefix_task,
        [(args.root, selected_prefix_config, prefix_config, subject, seed)
         for subject in FIXED_SUBJECTS for seed in CONFIRMATION_SEEDS],
        args.workers,
    )
    require(
        rows, "confirmation_prefix", args.root,
        selected_prefix_config, selected_config,
    )
    write_status(
        args.root, "confirmation", selected_prefix_config,
        selected_config, rows,
    )
    rows = parallel(
        confirmation_task,
        [(args.root, selected_prefix_config, selected_config, subject, seed)
         for subject in FIXED_SUBJECTS for seed in CONFIRMATION_SEEDS],
        args.workers,
    )
    require(
        rows, "confirmation", args.root,
        selected_prefix_config, selected_config,
    )
    write_status(
        args.root, "complete", selected_prefix_config,
        selected_config, rows,
    )


if __name__ == "__main__":
    main()
