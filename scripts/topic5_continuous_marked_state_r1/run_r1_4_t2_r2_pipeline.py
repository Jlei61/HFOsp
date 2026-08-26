#!/usr/bin/env python3
"""Wait for R1.4, run same-checkpoint sensitivity, then stable-T1 T2-R2.0."""
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
from src.topic5_continuous_marked_state_r1.t2_r2 import T2_R2_REVISION


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
R1_ROOT = contract.RESULT_ROOT / "r1_4"
T2_ROOT = contract.RESULT_ROOT / "t2_r2"
SUBJECTS = (
    "epilepsiae_620", "epilepsiae_958", "yuquan_huanghanwen",
    "epilepsiae_922", "yuquan_pengzihang", "yuquan_hanyuxuan",
)
SEEDS = (0, 1, 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT), "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
    })
    return value


def status(stage: str, state: str = "RUNNING", **extra) -> None:
    T2_ROOT.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(T2_ROOT / "PIPELINE_STATUS.json", {
        "status": state, "stage": stage, "updated_at": now(),
        "revision": T2_R2_REVISION,
        "formal_test_partition_opened": False, "sealed_opened": False,
        **extra,
    })


def run(command: list[str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"[{now()}] {' '.join(command)}\n")
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
    return {"command": command, "log": str(log), "returncode": process.returncode}


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({"task": list(futures[future]), "returncode": -1,
                             "error": repr(error)})
    return rows


def sensitivity(subject: str, seed: int) -> dict:
    output = R1_ROOT / "sensitivity_10_donor" / subject / f"explicit_seed_{seed}.json"
    if output.exists():
        try:
            value = json.loads(output.read_text())
            if value.get("status") == "COMPLETE" and value.get("sealed_opened") is False:
                return {"subject": subject, "seed": seed, "returncode": 0,
                        "skipped": True, "output": str(output)}
        except Exception:
            pass
    result = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/evaluate_r1_4_10_donor.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--root", str(R1_ROOT),
    ], T2_ROOT / "logs/r1_4_sensitivity" / f"{subject}_seed_{seed}.log")
    result.update({"subject": subject, "seed": seed, "output": str(output)})
    return result


def t2_fit(subject: str, source: str, seed: int) -> dict:
    output = (
        T2_ROOT / "human" / subject / f"{source}_seed_{seed}_n_100/result.json"
    )
    if output.exists():
        try:
            value = json.loads(output.read_text())
            if (value.get("status") == "COMPLETE"
                    and value.get("revision") == T2_R2_REVISION
                    and value.get("sealed_opened") is False):
                return {"subject": subject, "source": source, "seed": seed,
                        "returncode": 0, "skipped": True, "output": str(output)}
        except Exception:
            pass
    result = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_t2_r2_human.py",
        "--subject", subject, "--source", source, "--seed", str(seed),
        "--device", "cuda", "--epochs", "30", "--batch-size", "4096",
        "--r1-4-root", str(R1_ROOT), "--output-root", str(T2_ROOT),
    ], T2_ROOT / "logs/human" / f"{subject}_{source}_seed_{seed}.log")
    result.update({"subject": subject, "source": source, "seed": seed,
                   "output": str(output)})
    return result


def required(rows: list[dict], label: str) -> None:
    failed = [row for row in rows if row.get("returncode") != 0]
    if failed:
        raise RuntimeError(f"{label} failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    deadline = time.time() + args.timeout_hours * 3600
    summary_path = R1_ROOT / "reports/r1_4_summary.json"
    status("waiting_for_r1_4")
    while not summary_path.exists():
        queue_status = R1_ROOT / "STATUS.json"
        if queue_status.exists():
            value = json.loads(queue_status.read_text())
            if value.get("stage", "").endswith("_fail"):
                status("r1_4_failed", "FAIL", r1_4_status=value)
                raise RuntimeError("R1.4 queue failed")
        if time.time() >= deadline:
            status("r1_4_timeout", "FAIL")
            raise TimeoutError("R1.4 did not finish inside the supervisor window")
        time.sleep(max(5.0, args.poll_seconds))

    status("ten_donor_sensitivity")
    sensitivity_rows = parallel(
        sensitivity,
        [(subject, seed) for subject in SUBJECTS for seed in SEEDS],
        args.workers,
    )
    required(sensitivity_rows, "R1.4 10-donor sensitivity")
    aggregate_r1 = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/aggregate_r1_4.py",
        "--root", str(R1_ROOT),
    ], T2_ROOT / "logs/aggregate_r1_4_after_sensitivity.log")
    required([aggregate_r1], "R1.4 re-aggregation")
    r1 = json.loads(summary_path.read_text())
    stable = [
        subject for subject, value in r1["by_subject"].items()
        if value["stable_explicit_t1_for_t2"]
    ]
    status("t2_load", stable_t1_subjects=stable)
    load_rows = parallel(
        t2_fit,
        [(subject, "load", seed) for subject in stable for seed in SEEDS],
        args.workers,
    )
    required(load_rows, "T2 load")
    status("t2_participation", stable_t1_subjects=stable)
    participation_rows = parallel(
        t2_fit,
        [(subject, "participation", seed) for subject in stable for seed in SEEDS],
        args.workers,
    )
    required(participation_rows, "T2 participation")
    status("aggregate", stable_t1_subjects=stable)
    aggregate = run([
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/aggregate_t2_r2.py",
        "--r1-4-root", str(R1_ROOT), "--root", str(T2_ROOT),
    ], T2_ROOT / "logs/aggregate_t2_r2.log")
    required([aggregate], "T2 aggregation")
    status(
        "complete", "COMPLETE", stable_t1_subjects=stable,
        completed_human_fits=len(load_rows) + len(participation_rows),
        r1_4_summary=str(summary_path),
        t2_summary=str(T2_ROOT / "reports/t2_r2_summary.json"),
    )


if __name__ == "__main__":
    main()
