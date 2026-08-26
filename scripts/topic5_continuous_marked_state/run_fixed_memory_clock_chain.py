#!/usr/bin/env python3
"""Persistently chain count grid -> matched physical grid -> aggregation."""
from __future__ import annotations

import json
import os
import subprocess
import time

from src.topic5_continuous_marked_state import contract


PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
COUNT_STATUS = contract.RESULT_ROOT / "EVENT_COUNT_GRID_STATUS.json"
CHAIN_STATUS = contract.RESULT_ROOT / "FIXED_MEMORY_CLOCK_CHAIN_STATUS.json"


def _write(stage: str, **extra) -> None:
    row = {
        "contract": contract.REVISION,
        "stage": stage,
        "pid": os.getpid(),
        "updated": time.time(),
        "sealed_opened": False,
        **extra,
    }
    temporary = CHAIN_STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True))
    os.replace(temporary, CHAIN_STATUS)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(contract.REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    _write("WAITING_FOR_EVENT_COUNT")
    while True:
        row = json.loads(COUNT_STATUS.read_text())
        stage = row.get("stage")
        if stage == "COMPLETE":
            if row.get("failures"):
                raise RuntimeError("count grid reports failures despite COMPLETE")
            break
        if stage == "COMPLETE_WITH_FAILURES":
            raise RuntimeError(f"count grid failed: {row.get('failures')}")
        if not _alive(int(row["pid"])) and time.time() - float(row["updated"]) > 120:
            raise RuntimeError("count grid owner died with stale incomplete status")
        _write(
            "WAITING_FOR_EVENT_COUNT",
            count_completed=row.get("n_completed"),
            count_jobs=row.get("n_jobs"),
        )
        time.sleep(30)

    _write("RUNNING_MATCHED_PHYSICAL")
    command = [
        PYTHON,
        "scripts/topic5_continuous_marked_state/run_fixed_event_count_grid.py",
        "--workers", "6",
        "--memories", "25", "50", "100", "200", "400",
        "--kinds", "load", "participation",
        "--decay-clocks", "physical_time",
        "--status-tag", "physical",
    ]
    done = subprocess.run(
        command, cwd=contract.REPO_ROOT, env=_environment(), check=False
    )
    if done.returncode:
        _write("FAILED_MATCHED_PHYSICAL", exit_code=int(done.returncode))
        raise SystemExit(done.returncode)

    _write("AGGREGATING")
    done = subprocess.run([
        PYTHON,
        "scripts/topic5_continuous_marked_state/analyze_fixed_event_count_grid.py",
    ], cwd=contract.REPO_ROOT, env=_environment(), check=False)
    if done.returncode:
        _write("FAILED_AGGREGATION", exit_code=int(done.returncode))
        raise SystemExit(done.returncode)
    _write("COMPLETE")


if __name__ == "__main__":
    main()
