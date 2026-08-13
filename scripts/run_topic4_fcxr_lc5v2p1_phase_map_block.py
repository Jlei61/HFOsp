#!/usr/bin/env python3
"""Resource-gated detached block launcher for the locked LC5v2.1 3x3 map."""

from __future__ import annotations

import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5v2p1_phase_map as MAP  # noqa: E402


OUT = MAP.PREFIX.U2.OUT
PYTHON = "/home/honglab/leijiaxin/anaconda3/bin/python"
SELF_RSS_GIB = 6.8
CHILD_WALL_LIMIT_S = 30000


def _write_json(path, payload):
    MAP.PREFIX._write_json(path, payload)


def _meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = float(rest.strip().split()[0])
    return {
        "mem_available_gib": values["MemAvailable"] / 1024.0 / 1024.0,
        "swap_used_mib": (values["SwapTotal"] - values["SwapFree"]) / 1024.0,
    }


def pending_cells(manifest, cells):
    pending = []
    for cell in cells:
        if manifest["reuse"]["eligible_cells"][cell] is not None:
            continue
        tau_ms, gamma = cells[cell]
        tag = MAP.PREFIX._tag(
            gamma, "q099", tau_ms, manifest["experiment_id"]
        )
        if not (OUT / tag / "summary.json").is_file():
            pending.append(cell)
    return pending


def required_memavailable_gib(n_workers, manifest):
    resource = manifest["resource"]
    total_budget = (
        float(n_workers) * SELF_RSS_GIB * float(resource["rss_budget_multiplier"])
    )
    return total_budget * float(resource["memavailable_to_total_budget_ratio"])


def _block_stem(experiment_id):
    if experiment_id == MAP.BASE_EXPERIMENT:
        return "lc5v2p1_phase_map"
    return str(experiment_id)


def _run_batch(batch, manifest_path, batch_index, swap_baseline, block_stem):
    manifest_path = Path(manifest_path).resolve()
    logs = OUT / "lc5v2p1_phase_map_logs"
    logs.mkdir(parents=True, exist_ok=True)
    resource = _meminfo()
    required = required_memavailable_gib(len(batch), MAP.load_manifest(manifest_path)[1])
    if resource["mem_available_gib"] < required:
        raise RuntimeError(
            f"RESOURCE_PREFLIGHT_FAIL: MemAvailable={resource['mem_available_gib']:.2f} GiB "
            f"< required={required:.2f} GiB"
        )
    if resource["swap_used_mib"] - swap_baseline > 256.0:
        raise RuntimeError("RESOURCE_PREFLIGHT_FAIL: swap delta exceeds 256 MiB")
    children = []
    for cell in batch:
        log_path = logs / f"{cell}.log"
        log = log_path.open("ab", buffering=0)
        cmd = [
            "timeout", "--signal=TERM", "--kill-after=300s", str(CHILD_WALL_LIMIT_S),
            PYTHON, str(ROOT / "scripts/run_topic4_fcxr_lc5v2p1_phase_map.py"),
            "--manifest", str(manifest_path), "--cell", cell, "--confirm-run",
        ]
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        children.append((cell, process, log, str(log_path)))
    _write_json(OUT / f"{block_stem}_batch_{batch_index}_RUNNING.json", {
        "status": "RUNNING", "batch_index": batch_index, "cells": list(batch),
        "children": [{"cell": c, "pid": p.pid, "log": log} for c, p, _, log in children],
        "resource_preflight": resource, "required_memavailable_gib": required,
    })
    failures = []
    for cell, process, log, log_path in children:
        code = process.wait()
        log.close()
        if code != 0:
            failures.append({"cell": cell, "exit_code": code, "log": log_path})
    status = "DONE" if not failures else "FAILED"
    _write_json(OUT / f"{block_stem}_batch_{batch_index}_{status}.json", {
        "status": status, "batch_index": batch_index, "cells": list(batch),
        "failures": failures, "finished_epoch_s": time.time(),
    })
    (OUT / f"{block_stem}_batch_{batch_index}_RUNNING.json").unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f"batch {batch_index} failed: {failures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MAP.DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path, manifest, cells = MAP.load_manifest(args.manifest)
    block_stem = _block_stem(manifest["experiment_id"])
    lock_path = OUT / f".{block_stem}_block.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("LC5v2.1 phase-map block is already running") from exc
    try:
        # Publish all no-compute reuse receipts before expensive work.
        MAP.run_cell("control", manifest_path)
        for cell, source in manifest["reuse"]["eligible_cells"].items():
            if source is not None:
                MAP.run_cell(cell, manifest_path)
        pending = pending_cells(manifest, cells)
        max_parallel = int(manifest["resource"]["max_parallel_arms"])
        swap_baseline = _meminfo()["swap_used_mib"]
        _write_json(OUT / f"{block_stem}_block_RUNNING.json", {
            "status": "RUNNING", "pid": os.getpid(), "pending_cells": pending,
            "manifest_sha256": MAP._sha(manifest_path), "swap_baseline_mib": swap_baseline,
        })
        for batch_index, start in enumerate(range(0, len(pending), max_parallel), 1):
            _run_batch(
                pending[start:start + max_parallel], manifest_path, batch_index, swap_baseline,
                block_stem,
            )
        _write_json(OUT / f"{block_stem}_block_DONE.json", {
            "status": "DONE", "cells": list(cells), "finished_epoch_s": time.time(),
        })
        (OUT / f"{block_stem}_block_RUNNING.json").unlink(missing_ok=True)
    except BaseException as exc:
        _write_json(OUT / f"{block_stem}_block_FAILED.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "finished_epoch_s": time.time(),
        })
        (OUT / f"{block_stem}_block_RUNNING.json").unlink(missing_ok=True)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
