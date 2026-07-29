#!/usr/bin/env python3
"""Conditional Phase-C dt/2 production coordinator.

``c0`` delegates the complete two-seed identity+gain matrix to the generic C0
coordinator after its native-positive gate.  ``c1`` launches only arms recorded
in the immutable native-window-derived confirmation manifest.  Both routes use
independent dt/2 configs/checkpoints; native states are never interpolated.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

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
    expected_phasec = {
        "path": LOCK._rel(PHASEC_MANIFEST_PATH),
        "file_sha256": C1._sha(PHASEC_MANIFEST_PATH),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    if (
        selection.get("schema") != LOCK.SCHEMA
        or selection.get("resolution") != "dt2"
        or selection.get("selection_is_closed") is not True
        or selection.get("final_phasec") != expected_phasec
    ):
        raise RuntimeError("C1 dt2 confirmation parent lock mismatch")
    coordinate, dt2_ref = LOCK._dt2_coordinate(manifest)
    expected_coordinates = {
        "dt": LOCK._coordinate_ref(manifest, "dt"),
        "dt2": dt2_ref,
    }
    if selection.get("coordinate_manifests") != expected_coordinates:
        raise RuntimeError("C1 dt2 confirmation/coordinate forward lock mismatch")
    if (
        selection.get("coordinate_producer_file_sha256")
        != coordinate.get("producer_file_sha256")
    ):
        raise RuntimeError("C1 dt2 confirmation/coordinate producer mismatch")
    trigger_ref = selection.get("gain_trigger_manifest")
    if not isinstance(trigger_ref, dict):
        raise RuntimeError("C1 dt2 confirmation lacks gain-trigger provenance")
    trigger_path = ROOT / str(trigger_ref.get("path", ""))
    trigger = LOCK._validate_gain_trigger(
        trigger_path, manifest, PHASEC_MANIFEST_PATH
    )
    expected_trigger = {
        "path": LOCK._rel(trigger_path),
        "file_sha256": C1._sha(trigger_path),
        "manifest_sha256": trigger["manifest_sha256"],
        "selection_is_closed": True,
    }
    if trigger_ref != expected_trigger:
        raise RuntimeError("C1 dt2 confirmation/gain-trigger mismatch")
    native_ref = selection.get("native_summary")
    if not isinstance(native_ref, dict):
        raise RuntimeError("C1 dt2 confirmation lacks native-summary provenance")
    native_path = ROOT / str(native_ref.get("path", ""))
    if (
        not native_path.is_file()
        or C1._sha(native_path) != native_ref.get("file_sha256")
    ):
        raise RuntimeError("C1 dt2 confirmation/native-summary file drift")
    native = _read(native_path)
    if (
        native_ref.get("schema") != native.get("schema")
        or native.get("resolution") != "dt"
        or native.get("phasec_manifest_sha256")
        != manifest["manifest_sha256"]
        or native.get("phasec_manifest_file_sha256")
        != expected_phasec["file_sha256"]
        or native.get("coordinate_manifest_sha256")
        != expected_coordinates["dt"]["manifest_sha256"]
        or native.get("coordinate_manifest_semantic_sha256")
        != expected_coordinates["dt"]["semantic_sha256"]
        or native.get("coordinate_manifest_file_sha256")
        != expected_coordinates["dt"]["file_sha256"]
        or native.get("gain_trigger_manifest_sha256")
        != trigger["manifest_sha256"]
    ):
        raise RuntimeError("C1 dt2 confirmation/native-summary parent drift")
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
    if float(args.worker_rss_gb) < C1.MIN_MEASURED_WORKER_RSS_GB["base"]:
        raise SystemExit(
            "--worker-rss-gb cannot be lower than the locked measured "
            f"{C1.MIN_MEASURED_WORKER_RSS_GB['base']:g} GB for dt2 C1"
        )
    if (
        not math.isfinite(args.max_swap_growth_mb)
        or args.max_swap_growth_mb < 0
        or args.max_swap_growth_mb > 256
    ):
        raise SystemExit("--max-swap-growth-mb must be within [0,256]")
    manifest, selection, producer_locks = _manifest_and_selection()
    resources = manifest["resources"]
    expected_host_swap_mb = (
        float(resources["host_swap_growth_tolerance_bytes"])
        / (1024.0 * 1024.0)
    )
    if not math.isclose(
        float(args.max_swap_growth_mb), expected_host_swap_mb,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise SystemExit(
            "--max-swap-growth-mb must equal the locked shared-host "
            f"tolerance ({expected_host_swap_mb:g} MiB)"
        )
    worker_swap_allowed = int(
        resources["worker_swap_sampled_allowed_bytes"]
    )
    if (
        not math.isfinite(args.poll_s)
        or args.poll_s <= 0
        or args.poll_s > float(resources["worker_swap_poll_max_s"])
    ):
        raise SystemExit(
            "--poll-s must be positive and no slower than the locked "
            f"{resources['worker_swap_poll_max_s']:g}s worker-swap cadence"
        )
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
                receipt_path = C1.PRES.resource_receipt_path(task["output"])
                audit_ok, audit_reason, receipt = (
                    C1.PRES.validate_resource_receipt(
                        receipt_path,
                        artifact_path=task["output"],
                        artifact_root=ROOT,
                        manifest_sha256=manifest["manifest_sha256"],
                        task_key=task["key"],
                    )
                )
                if audit_ok:
                    skipped.append({
                        "key": task["key"],
                        "artifact_sha256": C1._sha(task["output"]),
                        "peak_rss_gb": payload.get("peak_rss_gb"),
                        "resource_receipt_path": str(
                            receipt_path.relative_to(ROOT)
                        ),
                        "resource_receipt_sha256": receipt["receipt_sha256"],
                    })
                else:
                    conflicts.append({
                        "path": task["output"], "reason": audit_reason
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
    worker_swap_peak_kb = 0
    worker_swap_audit = {}
    owned_running = []
    cleanup_state = {"finalized": False}
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"

    def _cleanup_at_exit():
        if cleanup_state["finalized"]:
            return
        try:
            if owned_running:
                C1._append(resource_log, {
                    "event": "abort_cleanup",
                    "run_id": run_id,
                    "time": time.time(),
                    "reason": "coordinator_exit_or_exception",
                    "owned_pids": [
                        row["process"].pid for row in owned_running
                    ],
                    "mem_available_gb": C1.mem_available_gb(),
                    "swap_used_kb": C1.swap_used_kb(),
                })
                C1.PRES.terminate_owned_workers(owned_running)
            partial_path = OUT / "coordinator_runs/dt2" / (
                f"phasec1_partial_abort_{run_id}.json"
            )
            C1._publish_json_once(partial_path, {
                "schema": "zm_phasec1_dt2_coordinator_partial_abort_v1",
                "run_id": run_id,
                "phasec_manifest_sha256": manifest["manifest_sha256"],
                "dt2_confirmation_manifest_sha256": selection[
                    "manifest_sha256"
                ],
                "n_expected_simulations": len(all_tasks),
                "n_skipped_valid": len(skipped),
                "n_launched": launched,
                "n_completed_before_abort": len(completed),
                "n_pending_at_abort": max(
                    0, len(all_tasks) - len(skipped) - len(completed)
                ),
                "n_queue_pending_at_abort": len(pending),
                "n_owned_inflight_at_abort": len(owned_running),
                "owned_pids_at_cleanup": [
                    row["process"].pid for row in owned_running
                ],
                "worker_swap_sampled_observed_max_kb": worker_swap_peak_kb,
                "worker_swap_audit_by_launch_token": worker_swap_audit,
                "resource_log_path": str(resource_log.relative_to(ROOT)),
                "evidence_scope": (
                    "partial abort record; sampled VmSwap plus any available "
                    "pre-publish child self snapshots, not kernel peaks"
                ),
                "finished_at": time.time(),
            })
        except Exception:
            C1.PRES.terminate_owned_workers(owned_running)

    atexit.register(_cleanup_at_exit)
    previous_signal_handlers = (
        C1.PRES.install_coordinator_signal_handlers()
    )
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
        owned_running[:] = running
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
            launch_token = uuid.uuid4().hex
            child_env = dict(env)
            child_env[C1.PRES.COORDINATOR_RUN_ENV] = run_id
            child_env[C1.PRES.COORDINATOR_TOKEN_ENV] = launch_token
            previous_mask = (
                C1.PRES.block_coordinator_termination_signals()
            )
            process = None
            item = None
            try:
                process = subprocess.Popen(
                    task["cmd"], cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env=child_env,
                )
                launched_at = time.time()
                item = {
                    **task, "process": process, "handle": handle,
                    "log": str(log_path), "started": launched_at,
                    "coordinator_run_id": run_id,
                    "coordinator_launch_token": launch_token,
                }
                running.append(item)
                owned_running[:] = running
                C1.PRES.register_worker_swap_audit(
                    worker_swap_audit,
                    pid=process.pid,
                    task_key=task["key"],
                    run_id=run_id,
                    launch_token=launch_token,
                    launched_at=launched_at,
                )
                launched += 1
            except BaseException:
                if process is None:
                    handle.close()
                else:
                    C1.PRES.terminate_owned_workers([
                        item or {"process": process, "handle": handle}
                    ])
                C1._append(resource_log, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(), "reason": "worker_launch_failed",
                    "key": task["key"],
                })
                C1.PRES.terminate_owned_workers(running)
                raise
            finally:
                C1.PRES.restore_coordinator_signal_mask(previous_mask)
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
            worker_swap = C1.worker_swap_snapshot(running)
            sampled_at = time.time()
            C1.PRES.update_worker_swap_audit(
                worker_swap_audit,
                worker_swap,
                sampled_at=sampled_at,
                audit_key_by_pid={
                    str(task["process"].pid):
                    task["coordinator_launch_token"]
                    for task in running
                },
            )
            worker_swap_peak_kb = max(
                worker_swap_peak_kb,
                int(worker_swap["worker_swap_max_kb"]),
            )
            if worker_swap["worker_swap_max_kb"] * 1024 > worker_swap_allowed:
                C1._append(resource_log, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "worker_sampled_swap_nonzero",
                    **worker_swap,
                })
                C1.PRES.terminate_owned_workers(running)
                raise SystemExit(
                    "a Phase-C worker acquired VmSwap; run invalidated"
                )
            if C1.mem_available_gb() < float(args.reserve_gb):
                C1._append(resource_log, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "mem_available_below_reserve",
                    "mem_available_gb": C1.mem_available_gb(),
                })
                C1.PRES.terminate_owned_workers(running)
                raise SystemExit(
                    "MemAvailable fell below the locked running-wave reserve"
                )
            if C1.swap_growth_exceeded(
                swap0, args.max_swap_growth_mb
            ):
                C1._append(resource_log, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "shared_host_swap_growth",
                    "swap_used_kb": C1.swap_used_kb(),
                })
                C1.PRES.terminate_owned_workers(running)
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
                final_swap = (
                    payload.get("runtime_provenance", {}).get(
                        "self_vm_swap_kb_at_publish"
                    )
                    if isinstance(payload, dict) else None
                )
                if isinstance(final_swap, int):
                    worker_swap_peak_kb = max(
                        worker_swap_peak_kb, final_swap
                    )
                    C1.PRES.record_final_worker_swap(
                        worker_swap_audit,
                        pid=task["process"].pid,
                        launch_token=task["coordinator_launch_token"],
                        value_kb=final_swap,
                        sampled_at=time.time(),
                    )
                ok = code == 0 and valid
                receipt_path = C1.PRES.resource_receipt_path(task["output"])
                if ok:
                    try:
                        receipt = C1.PRES.build_resource_receipt(
                            artifact_path=task["output"],
                            artifact_root=ROOT,
                            artifact_sha256=C1._sha(task["output"]),
                            manifest_sha256=manifest["manifest_sha256"],
                            task_key=task["key"],
                            run_id=run_id,
                            launch_token=task["coordinator_launch_token"],
                            pid=task["process"].pid,
                            audit_row=worker_swap_audit.get(
                                task["coordinator_launch_token"], {}
                            ),
                            sampled_allowed_bytes=worker_swap_allowed,
                        )
                        C1.PRES.publish_resource_receipt_once(
                            receipt_path, receipt
                        )
                    except Exception as exc:
                        ok = False
                        valid = False
                        reason = (
                            "resource_receipt_failure:"
                            f"{type(exc).__name__}:{exc}"
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
                if ok:
                    completed.append(finished)
                else:
                    failures.append({
                        "key": task["key"], "exit_code": code,
                        "validation_reason": reason, "log": task["log"],
                    })
            running = next_running
            owned_running[:] = running
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
                        **worker_swap,
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
        "skipped_valid": skipped,
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
        "host_swap_growth_tolerance_mb": args.max_swap_growth_mb,
        "worker_swap_sampled_allowed_bytes": worker_swap_allowed,
        "worker_swap_observed_max_kb": worker_swap_peak_kb,
        "worker_swap_poll_s": args.poll_s,
        "worker_swap_audit_by_launch_token": worker_swap_audit,
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
    cleanup_state["finalized"] = True
    owned_running.clear()
    C1.PRES.restore_signal_handlers(previous_signal_handlers)
    atexit.unregister(_cleanup_at_exit)
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
