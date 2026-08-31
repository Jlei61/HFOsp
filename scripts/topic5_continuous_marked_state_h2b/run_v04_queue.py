#!/usr/bin/env python3
"""Memory-aware resumable CPU queue for all audited H2b v0.4 cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    V0_4_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


PRODUCER = Path(__file__).resolve()
CELL_RUNNER = PRODUCER.with_name("run_v04_cell.py")
MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v04_heterogeneous.py"
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RSS_PATTERN = re.compile(r"__MAX_RSS_KB__\s+(\d+)")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_kb() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is unavailable")


def _valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = _json(path)
    except (OSError, json.JSONDecodeError):
        return False
    source = value.get("source", {})
    return (
        value.get("revision") == "h2b_v0_4_heterogeneous_route_cell_v6_direct_history_contrast_conditional_risk_sets"
        and source.get("producer_sha256") == sha256_file(CELL_RUNNER)
        and source.get("heterogeneous_module_sha256") == sha256_file(MODULE)
    )


def _run_one(
    cell: dict,
    *,
    v02_root: Path,
    v03_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    subject, seed = str(cell["subject"]), int(cell["seed"])
    output = result_root / "per_cell" / subject / f"seed_{seed}" / "result.json"
    log = result_root / "logs/cells" / subject / f"seed_{seed}.log"
    claim = result_root / "claims" / subject / f"seed_{seed}.claim"
    if _valid_existing(output):
        value = _json(output)
        return {
            "subject": subject, "seed": seed, "status": "SKIPPED_VALID",
            "result_status": value["status"], "max_rss_kb": None,
            "returncode": 0, "log": str(log),
        }
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return {
            "subject": subject, "seed": seed, "status": "ALREADY_CLAIMED",
            "max_rss_kb": None, "returncode": None, "log": str(log),
        }
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "pid": os.getpid(), "created_utc": utc_now(),
                "subject": subject, "seed": seed,
            }) + "\n")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "/usr/bin/time", "-f", "__MAX_RSS_KB__ %M",
            str(PYTHON), str(CELL_RUNNER),
            "--subject", subject,
            "--seed", str(seed),
            "--v0-2-root", str(v02_root),
            "--v0-3-root", str(v03_root),
            "--result-root", str(result_root),
        ]
        environment = dict(os.environ)
        environment.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
        })
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=REPO,
                check=False,
                text=True,
            )
        text = log.read_text(encoding="utf-8", errors="replace")
        match = RSS_PATTERN.search(text)
        rss = int(match.group(1)) if match else None
        status = "COMPLETE" if completed.returncode == 0 and _valid_existing(output) else "FAILED"
        result_status = _json(output)["status"] if _valid_existing(output) else None
        return {
            "subject": subject, "seed": seed, "status": status,
            "result_status": result_status, "max_rss_kb": rss,
            "returncode": completed.returncode, "log": str(log),
            "elapsed_seconds": time.monotonic() - started,
            "oom_like": completed.returncode in (-9, 9, 137),
        }
    finally:
        claim.unlink(missing_ok=True)


def run_queue(
    *,
    v02_root: Path,
    v03_root: Path,
    result_root: Path,
    requested_workers: int,
    measured_rss_kb: int,
) -> dict:
    inventory_path = result_root / "manifests/source_cells.json"
    inventory = _json(inventory_path)
    cells = inventory["cells"]
    available_before = _available_kb()
    # Keep 35% of available RAM outside the queue and add a 35% per-worker
    # margin over the measured sentinel RSS.
    memory_workers = max(1, int(
        (0.65 * available_before) // max(1.35 * int(measured_rss_kb), 1)
    ))
    workers = max(1, min(int(requested_workers), memory_workers, len(cells)))
    started_utc = utc_now()
    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    def write_status(state: str) -> None:
        complete = sum(row["status"] in ("COMPLETE", "SKIPPED_VALID") for row in rows)
        failed = sum(row["status"] == "FAILED" for row in rows)
        payload = {
            "status": state,
            "revision": "h2b_v0_4_cpu_queue_v6_direct_history_contrast_conditional_risk_sets",
            "started_utc": started_utc,
            "updated_utc": utc_now(),
            "pid": os.getpid(),
            "expected_cells": len(cells),
            "finished_cells": len(rows),
            "complete_or_valid_cells": complete,
            "failed_cells": failed,
            "workers": workers,
            "requested_workers": int(requested_workers),
            "available_memory_before_kb": available_before,
            "measured_sentinel_rss_kb": int(measured_rss_kb),
            "memory_worker_limit": memory_workers,
            "thread_limits": 1,
            "gpu": "disabled_cpu_only",
            "source_inventory": str(inventory_path),
            "source_inventory_sha256": sha256_file(inventory_path),
            "producer_sha256": sha256_file(PRODUCER),
            "cell_runner_sha256": sha256_file(CELL_RUNNER),
            "rows": sorted(rows, key=lambda row: (row["subject"], row["seed"])),
        }
        atomic_json(result_root / "QUEUE_STATUS.json", payload)
        atomic_json(result_root / "CURRENT_HANDOFF.json", {
            "status": state,
            "updated_utc": utc_now(),
            "queue_pid": os.getpid(),
            "workers": workers,
            "complete_or_valid_cells": complete,
            "expected_cells": len(cells),
            "failed_cells": failed,
        })

    write_status("RUNNING")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                cell,
                v02_root=v02_root,
                v03_root=v03_root,
                result_root=result_root,
            ): cell
            for cell in cells
        }
        for future in as_completed(futures):
            rows.append(future.result())
            write_status("RUNNING")

    oom_failures = [row for row in rows if row.get("oom_like")]
    if oom_failures:
        # The first-pass output is atomic.  Retry only OOM-like failures one at
        # a time without dropping any patient or changing model parameters.
        retry_cells = [
            cell for cell in cells
            if any(cell["subject"] == row["subject"] and int(cell["seed"]) == row["seed"]
                   for row in oom_failures)
        ]
        for cell in retry_cells:
            rows.append(_run_one(
                cell, v02_root=v02_root, v03_root=v03_root, result_root=result_root,
            ))
            write_status("RUNNING_SERIAL_OOM_RETRY")

    valid = []
    for cell in cells:
        output = (
            result_root / "per_cell" / str(cell["subject"])
            / f"seed_{int(cell['seed'])}" / "result.json"
        )
        if _valid_existing(output):
            valid.append(_json(output))
    final_status = "PASS_COMPLETE" if len(valid) == len(cells) else "FAILED_INCOMPLETE"
    final = {
        "status": final_status,
        "revision": "h2b_v0_4_cpu_queue_v6_direct_history_contrast_conditional_risk_sets",
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        "expected_cells": len(cells),
        "valid_result_cells": len(valid),
        "complete_development_cells": sum(
            row["status"] == "COMPLETE_DEVELOPMENT" for row in valid
        ),
        "not_estimable_primary_cells": sum(
            row["status"] == "NOT_ESTIMABLE_PRIMARY_LEAD" for row in valid
        ),
        "workers": workers,
        "requested_workers": int(requested_workers),
        "available_memory_before_kb": available_before,
        "available_memory_after_kb": _available_kb(),
        "measured_sentinel_rss_kb": int(measured_rss_kb),
        "max_observed_worker_rss_kb": max(
            (row["max_rss_kb"] for row in rows if row.get("max_rss_kb") is not None),
            default=None,
        ),
        "oom_failures_first_pass": len(oom_failures),
        "serial_oom_retry_policy": True,
        "thread_limits": 1,
        "gpu": "disabled_cpu_only",
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "rows": sorted(rows, key=lambda row: (row["subject"], row["seed"])),
        "source": {
            "inventory": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "producer_sha256": sha256_file(PRODUCER),
            "cell_runner_sha256": sha256_file(CELL_RUNNER),
            "module_sha256": sha256_file(MODULE),
        },
    }
    atomic_json(result_root / "QUEUE_STATUS.json", final)
    atomic_json(result_root / "CURRENT_HANDOFF.json", {
        "status": final_status,
        "updated_utc": utc_now(),
        "valid_result_cells": len(valid),
        "expected_cells": len(cells),
        "next": "run assay and aggregate patient-first results",
    })
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--v0-3-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--measured-rss-kb", type=int, default=500000)
    args = parser.parse_args()
    result = run_queue(
        v02_root=args.v0_2_root.resolve(),
        v03_root=args.v0_3_root.resolve(),
        result_root=args.result_root.resolve(),
        requested_workers=int(args.max_workers),
        measured_rss_kb=int(args.measured_rss_kb),
    )
    print(result["status"], result["valid_result_cells"], result["workers"])


if __name__ == "__main__":
    main()
