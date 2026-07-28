#!/usr/bin/env python3
"""Conditional Phase-C dt/2 production coordinator.

``c0`` delegates the complete two-seed identity+gain matrix to the generic C0
coordinator after its native-positive gate.  ``c1`` launches only arms recorded
in the immutable native-window-derived confirmation manifest.  Both routes use
independent dt/2 configs/checkpoints; native states are never interpolated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_topic4_zm_phasec0_parallel as C0  # noqa: E402
import scripts.run_topic4_zm_phasec1_parallel as C1  # noqa: E402
import scripts.run_topic4_zm_phasec_cell as CELL  # noqa: E402
import scripts.lock_topic4_zm_phasec1_dt2_confirmation as LOCK  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
PHASEC_MANIFEST_PATH = OUT / "phasec_manifest.json"
CONFIRMATION_PATH = OUT / "phasec1_dt2_confirmation_manifest.json"
CELL_SCRIPT = ROOT / "scripts/run_topic4_zm_phasec_cell.py"
TERMINAL = {"complete", "scientific_failure"}


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _manifest_and_selection():
    manifest = _read(PHASEC_MANIFEST_PATH)
    PCC.require_production_manifest(manifest)
    producer_locks = C1._validate_live_producers(manifest)
    selection = _read(CONFIRMATION_PATH)
    CELL._validate_self_hash(
        selection, label="C1 dt2 confirmation manifest"
    )
    if (
        selection.get("schema") != LOCK.SCHEMA
        or selection.get("resolution") != "dt2"
        or selection.get("selection_is_closed") is not True
        or selection.get("final_phasec", {}).get("manifest_sha256")
        != manifest["manifest_sha256"]
        or selection.get("final_phasec", {}).get("file_sha256")
        != C1._sha(PHASEC_MANIFEST_PATH)
    ):
        raise RuntimeError("C1 dt2 confirmation parent lock mismatch")
    dt2_coord = manifest["c1"]["coordinate_manifests"]["dt2"]
    if selection.get("coordinate_manifests", {}).get("dt2") != dt2_coord:
        raise RuntimeError("C1 dt2 confirmation/coordinate forward lock mismatch")
    return manifest, selection, producer_locks


def c1_tasks(manifest, selection):
    coordinate_path = ROOT / manifest["c1"]["coordinate_manifests"]["dt2"]["path"]
    rows = []
    for expected in selection["expected_base_arms"]:
        if expected.get("resolution") != "dt2":
            raise RuntimeError("non-dt2 arm in dt2 confirmation manifest")
        output = ROOT / expected["path"]
        task_expected = {
            key: value for key, value in expected.items() if key != "path"
        }
        task_expected.update({
            "dt2_confirmation_manifest_sha256": selection["manifest_sha256"],
            "dt2_confirmation_manifest_file_sha256": C1._sha(
                CONFIRMATION_PATH
            ),
        })
        rows.append({
            "kind": "c1_base",
            "key": (
                f"dt2|s{expected['seed']}|{expected['tier']}|"
                f"{expected['cell_id']}|{expected['phase']}|"
                f"{expected['noise']}"
            ),
            "output": str(output),
            "expected": task_expected,
            "coordinate_producer_locks": selection[
                "coordinate_producer_file_sha256"
            ],
            "cmd": [
                sys.executable, str(CELL_SCRIPT),
                "--mode", "c1_base",
                "--resolution", "dt2",
                "--seed", str(expected["seed"]),
                "--tier", expected["tier"],
                "--cell-id", expected["cell_id"],
                "--phase", expected["phase"],
                "--replicate", expected["noise"],
                "--manifest", str(PHASEC_MANIFEST_PATH),
                "--coordinate-manifest", str(coordinate_path),
                "--dt2-confirmation-manifest", str(CONFIRMATION_PATH),
                "--confirm-run",
            ],
        })
    if not rows or len({row["output"] for row in rows}) != len(rows):
        raise RuntimeError("empty or duplicate dt2 C1 expected matrix")
    return rows


def validate_c1_output(path, task, *, producer_locks):
    valid, reason, payload = C1.validate_terminal_output(
        path, task, producer_locks=producer_locks
    )
    if not valid:
        return valid, reason, payload
    provenance = payload.get("runtime_provenance", {})
    if (
        provenance.get("dt2_confirmation_manifest_sha256")
        != task["expected"]["dt2_confirmation_manifest_sha256"]
        or provenance.get("dt2_confirmation_manifest_file_sha256")
        != task["expected"]["dt2_confirmation_manifest_file_sha256"]
        or provenance.get("coordinate_npz_file_sha256")
        != task["expected"]["coordinate_npz_file_sha256"]
        or provenance.get("coordinate_npz_semantic_sha256")
        != task["expected"]["coordinate_npz_semantic_sha256"]
    ):
        return False, "dt2_runtime_provenance_mismatch", payload
    if payload.get("resolution") != "dt2" or float(
        payload.get("dt_ms", np.nan)
    ) != 0.05:
        return False, "dt2_resolution_or_step_mismatch", payload
    return True, "valid", payload


def _run_c1(args):
    if args.max_workers > 12 or args.wave_size > 12:
        raise SystemExit("Phase-C dt2 concurrency is capped at 12 workers")
    if args.reserve_gb < 96 or args.reserve_cpus < 8:
        raise SystemExit("Phase-C dt2 requires >=96GB and >=8 CPU reserve")
    if not math.isfinite(args.worker_rss_gb) or args.worker_rss_gb <= 0:
        raise SystemExit("--worker-rss-gb must be a measured positive value")
    if (
        not math.isfinite(args.max_swap_growth_mb)
        or args.max_swap_growth_mb < 0
        or args.max_swap_growth_mb > 256
    ):
        raise SystemExit("--max-swap-growth-mb must be within [0,256]")
    manifest, selection, producer_locks = _manifest_and_selection()
    all_tasks = c1_tasks(manifest, selection)
    pending, skipped, conflicts = [], [], []
    for task in all_tasks:
        if not os.path.isfile(task["output"]):
            pending.append(task)
            continue
        valid, reason, payload = validate_c1_output(
            task["output"], task, producer_locks=producer_locks
        )
        if valid:
            if getattr(args, "resume", False):
                skipped.append({
                    "key": task["key"],
                    "artifact_sha256": C1._sha(task["output"]),
                    "peak_rss_gb": payload.get("peak_rss_gb"),
                })
            else:
                conflicts.append({
                    "path": task["output"],
                    "reason": "valid_terminal_requires_--resume",
                })
        else:
            conflicts.append({"path": task["output"], "reason": reason})
    if conflicts:
        raise SystemExit(
            "existing conflicting dt2 parts require explicit invalidation: "
            + json.dumps(conflicts[:5], sort_keys=True)
        )
    if pending and C1._resource_cap(args) < 1:
        raise SystemExit("resource guard authorizes no dt2 worker")

    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_p{os.getpid()}"
    logs_root = OUT / "logs/phasec1/dt2" / run_id
    logs_root.mkdir(parents=True, exist_ok=True)
    resource_log = logs_root / "resource_log.jsonl"
    swap0 = C1.swap_used_kb()
    completed, failures = [], []
    launched = 0
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    print(
        f"[phasec1 dt2] expected={len(all_tasks)} skipped={len(skipped)} "
        f"pending={len(pending)} cap={C1._resource_cap(args)} "
        f"MemAvailable={C1.mem_available_gb():.1f}GB swap0={swap0}kB",
        flush=True,
    )
    while pending:
        if C1.swap_growth_exceeded(swap0, args.max_swap_growth_mb):
            raise SystemExit(
                "swap growth exceeded tolerance before next dt2 wave"
            )
        cap = C1._resource_cap(args)
        if cap < 1:
            raise SystemExit("resource guard authorizes no next dt2 wave")
        wave = [
            pending.pop(0) for _ in range(
                min(len(pending), cap, args.wave_size)
            )
        ]
        running = []
        for task in wave:
            if (
                C1.mem_available_gb() < args.reserve_gb
                or C1.swap_growth_exceeded(
                    swap0, args.max_swap_growth_mb
                )
            ):
                pending[:0] = wave[len(running):]
                break
            log_path = logs_root / (
                task["key"].replace("|", "__").replace("/", "_") + ".log"
            )
            handle = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                task["cmd"], cwd=ROOT, stdout=handle,
                stderr=subprocess.STDOUT, env=env,
            )
            launched += 1
            running.append({
                **task, "process": process, "handle": handle,
                "log": str(log_path), "started": time.time(),
            })
            C1._append(resource_log, {
                "event": "launch", "time": time.time(), "pid": process.pid,
                "key": task["key"], "cmd": task["cmd"],
                "mem_available_gb": C1.mem_available_gb(),
                "swap_used_kb": C1.swap_used_kb(),
            })
        if not running:
            raise SystemExit("resource guard blocked the entire next dt2 wave")
        last_heartbeat = 0.0
        while running:
            if C1.swap_growth_exceeded(
                swap0, args.max_swap_growth_mb
            ):
                for task in running:
                    task["process"].terminate()
                for task in running:
                    try:
                        task["process"].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        task["process"].kill()
                        task["process"].wait()
                    task["handle"].close()
                raise SystemExit(
                    "swap growth exceeded tolerance; stopped only this "
                    "coordinator's workers"
                )
            next_running = []
            for task in running:
                code = task["process"].poll()
                if code is None:
                    next_running.append(task)
                    continue
                task["handle"].close()
                valid, reason, payload = validate_c1_output(
                    task["output"], task, producer_locks=producer_locks
                )
                finished = {
                    "event": "finish", "time": time.time(),
                    "pid": task["process"].pid, "key": task["key"],
                    "exit_code": code, "valid_terminal": valid,
                    "validation_reason": reason,
                    "wall_s": round(time.time() - task["started"], 2),
                    "artifact_sha256": (
                        C1._sha(task["output"])
                        if os.path.isfile(task["output"]) else None
                    ),
                    "child_peak_rss_gb": (
                        payload.get("peak_rss_gb")
                        if isinstance(payload, dict) else None
                    ),
                    "mem_available_gb": C1.mem_available_gb(),
                    "swap_used_kb": C1.swap_used_kb(),
                }
                C1._append(resource_log, finished)
                if code == 0 and valid:
                    completed.append(finished)
                else:
                    failures.append({
                        "key": task["key"], "exit_code": code,
                        "validation_reason": reason, "log": task["log"],
                    })
            running = next_running
            if running:
                now = time.time()
                if now - last_heartbeat >= 30.0:
                    C1._append(resource_log, {
                        "event": "heartbeat", "time": now,
                        "running_pids": [
                            row["process"].pid for row in running
                        ],
                        "n_running": len(running),
                        "n_pending": len(pending),
                        "mem_available_gb": C1.mem_available_gb(),
                        "swap_used_kb": C1.swap_used_kb(),
                    })
                    last_heartbeat = now
                time.sleep(args.poll_s)
        if failures:
            break
    summary = {
        "schema": "zm_phasec1_dt2_coordinator_v1_2026-07-28",
        "run_id": run_id,
        "resolution": "dt2",
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "dt2_confirmation_manifest_sha256": selection["manifest_sha256"],
        "coordinate_manifest_sha256": manifest["c1"][
            "coordinate_manifests"
        ]["dt2"]["manifest_sha256"],
        "n_expected_simulations": len(all_tasks),
        "n_skipped_valid": len(skipped),
        "n_launched": launched,
        "n_completed_this_run": len(completed),
        "n_pending_after_stop": len(pending),
        "n_failures": len(failures),
        "failures": failures,
        "worker_rss_gb": args.worker_rss_gb,
        "max_workers": args.max_workers,
        "wave_size": args.wave_size,
        "reserve_gb": args.reserve_gb,
        "reserve_cpus": args.reserve_cpus,
        "swap_baseline_kb": swap0,
        "swap_final_kb": C1.swap_used_kb(),
        "max_swap_growth_mb": args.max_swap_growth_mb,
        "resource_log_path": str(resource_log.relative_to(ROOT)),
        "claim_boundary": (
            "conditional homologous source-space dt2 confirmation only; "
            "not observation match, entry, offset, recovery, or lifecycle"
        ),
    }
    summary_path = OUT / "coordinator_runs/dt2" / (
        f"phasec1_summary_{run_id}.json"
    )
    C1._publish_json_once(summary_path, summary)
    if failures or pending:
        raise SystemExit(f"Phase-C1 dt2 stopped; see {summary_path}")
    print(f"[phasec1 dt2] complete -> {summary_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("c0", "c1"), required=True)
    parser.add_argument("--worker-rss-gb", type=float, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--wave-size", type=int, default=12)
    parser.add_argument("--reserve-gb", type=float, default=96.0)
    parser.add_argument("--reserve-cpus", type=int, default=8)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument(
        "--max-swap-growth-mb", type=float, default=64.0,
        help="bounded shared-host swap jitter allowance before fail-close",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        raise SystemExit("Phase-C dt2 production requires --confirm-run")
    if args.phase == "c0":
        args.phases = "identity,gain"
        args.resolution = "dt2"
        args.seeds = "1,3"
        C0.run(args)
    else:
        _run_c1(args)


if __name__ == "__main__":
    main()
