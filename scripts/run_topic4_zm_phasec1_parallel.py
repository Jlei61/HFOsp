#!/usr/bin/env python3
"""Crash-safe coordinator for the locked Phase-C1 base and conditional gain.

Invalid physical coordinates are explicit coverage entries and are never
launched.  Conditional gain can only be enumerated from the canonical
write-once trigger manifest.  This coordinator never creates either manifest.
"""
from __future__ import annotations

import argparse
import hashlib
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

import scripts.run_topic4_zm_phasec_cell as CELL  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
MANIFEST_PATH = OUT / "phasec_manifest.json"
COORDINATE_MANIFEST_PATH = OUT / "phasec1_coordinate_manifest_dt.json"
TRIGGER_MANIFEST_PATH = OUT / "c1_gain_trigger_manifest.json"
CELL_SCRIPT = ROOT / "scripts/run_topic4_zm_phasec_cell.py"
PHASES = ("rising", "peak")
NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")
DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
TERMINAL = {"complete", "scientific_failure"}
SCIENTIFIC_ENDS = {
    "runaway", "whole_sheet_plateau", "empirical_rest_dwell"
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_live_producers(manifest):
    locks = manifest.get("provenance", {}).get("producer_file_sha256")
    if not isinstance(locks, dict) or not locks:
        raise RuntimeError("Phase-C manifest lacks producer-file locks")
    for relative, expected in sorted(locks.items()):
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"live producer hash mismatch: {relative}")
    return locks


def _load_contracts(*, require_trigger=False):
    manifest = _read_json(MANIFEST_PATH)
    PCC.require_production_manifest(manifest)
    producers = _validate_live_producers(manifest)
    coordinate_ref = manifest["c1"]["coordinate_manifests"]["dt"]
    coordinate_path = ROOT / coordinate_ref["path"]
    coordinate = CELL._coordinate_contract(
        manifest, str(coordinate_path), resolution="dt"
    )
    trigger = None
    if require_trigger:
        trigger = _read_json(TRIGGER_MANIFEST_PATH)
        CELL._validate_self_hash(
            trigger, label="C1 conditional-gain trigger manifest"
        )
        if (
            trigger.get("phasec_manifest_sha256")
            != manifest["manifest_sha256"]
            or trigger.get("phasec_manifest_file_sha256")
            != _sha(MANIFEST_PATH)
            or trigger.get("coordinate_manifest_sha256")
            != coordinate["manifest_sha256"]
            or trigger.get("coordinate_manifest_file_sha256")
            != _sha(coordinate_path)
            or trigger.get("resolution") != "dt"
            or trigger.get("selection_is_closed") is not True
        ):
            raise RuntimeError("conditional-gain trigger parent/closure mismatch")
    return manifest, coordinate, coordinate_path, trigger, producers


def _cells(coordinate):
    valid, invalid = [], []
    for seed_text, seed_row in sorted(coordinate["seeds"].items()):
        seed = int(seed_text)
        for cell in seed_row["cells"]:
            row = {**cell, "seed": seed}
            (valid if cell["status"] == "valid" else invalid).append(row)
    return valid, invalid


def _base_output(seed, tier, cell_id, phase, noise):
    return ROOT / CELL._c1_base_relative_path(
        "dt", seed, tier, cell_id, phase, noise
    )


def _gain_output(seed, tier, cell_id, phase, noise, delta):
    return ROOT / CELL._c1_gain_relative_path(
        "dt", seed, tier, cell_id, phase, noise, delta
    )


def base_tasks(
    manifest, coordinate, coordinate_path=None
):
    coordinate_path = coordinate_path or COORDINATE_MANIFEST_PATH
    valid, invalid = _cells(coordinate)
    rows = []
    for cell in valid:
        for phase in PHASES:
            for noise in NOISES:
                output = _base_output(
                    cell["seed"], cell["tier"], cell["cell_id"], phase, noise
                )
                expected = {
                    "schema": CELL.C1_BASE_PART_SCHEMA,
                    "phasec_manifest_sha256": manifest["manifest_sha256"],
                    "coordinate_manifest_sha256": coordinate["manifest_sha256"],
                    "coordinate_manifest_semantic_sha256": coordinate[
                        "semantic_sha256"
                    ],
                    "phasec_manifest_file_sha256": _sha(MANIFEST_PATH),
                    "coordinate_manifest_file_sha256": _sha(
                        coordinate_path
                    ),
                    "seed": cell["seed"],
                    "tier": cell["tier"],
                    "cell_id": cell["cell_id"],
                    "trajectory_id": cell["trajectory_id"],
                    "path_index": int(cell["path_index"]),
                    "path_direction": cell["path_direction"],
                    "phase": phase,
                    "noise": noise,
                    "resolution": "dt",
                    "slow_state_sha256": cell["state_sha256"],
                    "coordinate_npz_file_sha256": coordinate["seeds"][
                        str(cell["seed"])
                    ]["npz_file_sha256"],
                    "coordinate_npz_semantic_sha256": coordinate["seeds"][
                        str(cell["seed"])
                    ]["npz_semantic_sha256"],
                    "config_sha": coordinate["seeds"][
                        str(cell["seed"])
                    ]["config_sha"],
                    "burn_in_ms": 500.0,
                    "measure_ms": 8000.0,
                }
                rows.append({
                    "kind": "c1_base",
                    "key": (
                        f"base|s{cell['seed']}|{cell['tier']}|"
                        f"{cell['cell_id']}|{phase}|{noise}"
                    ),
                    "output": str(output),
                    "expected": expected,
                    "coordinate_producer_locks": coordinate[
                        "producer_file_sha256"
                    ],
                    "cmd": [
                        sys.executable, str(CELL_SCRIPT),
                        "--mode", "c1_base",
                        "--seed", str(cell["seed"]),
                        "--tier", cell["tier"],
                        "--cell-id", cell["cell_id"],
                        "--phase", phase,
                        "--replicate", noise,
                        "--manifest", str(MANIFEST_PATH),
                        "--coordinate-manifest", str(
                            coordinate_path
                        ),
                        "--confirm-run",
                    ],
                })
    return rows, invalid


def gain_tasks(
    manifest, coordinate, trigger,
    coordinate_path=None,
):
    coordinate_path = coordinate_path or COORDINATE_MANIFEST_PATH
    if trigger is None:
        raise RuntimeError("gain task enumeration requires locked trigger")
    coordinate_cells = {
        (row["seed"], row["tier"], row["cell_id"]): row
        for row in _cells(coordinate)[0]
    }
    rows = []
    for selected in trigger.get("triggered_cells", []):
        key = (
            int(selected["seed"]), selected["tier"], selected["cell_id"]
        )
        coordinate_cell = coordinate_cells.get(key)
        if coordinate_cell is None:
            raise RuntimeError(f"trigger selects absent/invalid coordinate {key}")
        if selected.get("slow_state_sha256") != coordinate_cell[
            "state_sha256"
        ]:
            raise RuntimeError(f"trigger slow-state SHA mismatch: {key}")
        base_refs = selected.get("triggering_base_parts", [])
        if len(base_refs) != 6:
            raise RuntimeError(f"trigger lacks six base references: {key}")
        for ref in base_refs:
            path = ROOT / ref["part_path"]
            if not path.is_file() or _sha(path) != ref["part_sha256"]:
                raise RuntimeError(f"triggering base evidence drift: {path}")
        expected_paths = {
            row["path"] for row in selected.get(
                "expected_carrier_gain_arms", []
            )
        }
        if len(expected_paths) != 30:
            raise RuntimeError(f"trigger arm coverage is not 2x3x5: {key}")
        for phase in PHASES:
            for noise in NOISES:
                for delta in DELTAS:
                    output = _gain_output(*key, phase, noise, delta)
                    if str(output.relative_to(ROOT)) not in expected_paths:
                        raise RuntimeError(
                            f"trigger expected path drift: {output}"
                        )
                    sign = 0 if delta == 0 else (1 if delta > 0 else -1)
                    expected = {
                        "schema": CELL.C1_GAIN_PART_SCHEMA,
                        "trigger_manifest_sha256": trigger[
                            "manifest_sha256"
                        ],
                        "phasec_manifest_sha256": manifest[
                            "manifest_sha256"
                        ],
                        "coordinate_manifest_sha256": coordinate[
                            "manifest_sha256"
                        ],
                        "coordinate_manifest_semantic_sha256": coordinate[
                            "semantic_sha256"
                        ],
                        "phasec_manifest_file_sha256": _sha(MANIFEST_PATH),
                        "coordinate_manifest_file_sha256": _sha(
                            coordinate_path
                        ),
                        "trigger_manifest_file_sha256": _sha(
                            TRIGGER_MANIFEST_PATH
                        ),
                        "seed": key[0],
                        "tier": key[1],
                        "cell_id": key[2],
                        "trajectory_id": coordinate_cell["trajectory_id"],
                        "path_index": int(coordinate_cell["path_index"]),
                        "path_direction": coordinate_cell["path_direction"],
                        "phase": phase,
                        "noise": noise,
                        "resolution": "dt",
                        "slow_state_sha256": coordinate_cell[
                            "state_sha256"
                        ],
                        "coordinate_npz_file_sha256": coordinate["seeds"][
                            str(key[0])
                        ]["npz_file_sha256"],
                        "coordinate_npz_semantic_sha256": coordinate["seeds"][
                            str(key[0])
                        ]["npz_semantic_sha256"],
                        "delta_mV": float(delta),
                        "config_sha": coordinate["seeds"][
                            str(key[0])
                        ]["config_sha"],
                        "burn_in_ms": 500.0,
                        "measure_ms": 1000.0,
                    }
                    rows.append({
                        "kind": "c1_gain",
                        "key": (
                            f"gain|s{key[0]}|{key[1]}|{key[2]}|"
                            f"{phase}|{noise}|{delta:+g}"
                        ),
                        "output": str(output),
                        "expected": expected,
                        "coordinate_producer_locks": coordinate[
                            "producer_file_sha256"
                        ],
                        "trigger_producer_locks": trigger[
                            "producer_file_sha256"
                        ],
                        "cmd": [
                            sys.executable, str(CELL_SCRIPT),
                            "--mode", "c1_gain",
                            "--seed", str(key[0]),
                            "--tier", key[1],
                            "--cell-id", key[2],
                            "--phase", phase,
                            "--replicate", noise,
                            "--delta-mV", str(abs(delta)),
                            "--sign", str(sign),
                            "--manifest", str(MANIFEST_PATH),
                            "--coordinate-manifest", str(
                                coordinate_path
                            ),
                            "--trigger-manifest", str(
                                TRIGGER_MANIFEST_PATH
                            ),
                            "--confirm-run",
                        ],
                    })
    if len({row["output"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate conditional-gain output path")
    return rows


def validate_terminal_output(path, task, *, producer_locks):
    path = Path(path)
    if not path.is_file():
        return False, "missing", None
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_json:{exc}", None
    mismatches = [
        key for key, expected in task["expected"].items()
        if payload.get(key) != expected
    ]
    if mismatches:
        return False, "identity_mismatch:" + ",".join(mismatches), payload
    if payload.get("status") not in TERMINAL:
        return False, "nonterminal_status", payload
    if (
        payload["status"] == "scientific_failure"
        and payload.get("scientific_end_reason") not in SCIENTIFIC_ENDS
    ):
        return False, "unknown_scientific_end", payload
    provenance = payload.get("runtime_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("producer_sha256") != producer_locks
        or provenance.get("manifest_sha256")
        != task["expected"]["phasec_manifest_sha256"]
        or provenance.get("coordinate_manifest_sha256")
        != task["expected"]["coordinate_manifest_sha256"]
    ):
        return False, "runtime_provenance_mismatch", payload
    if (
        provenance.get("coordinate_producer_sha256")
        != task.get("coordinate_producer_locks")
    ):
        return False, "runtime_coordinate_producer_mismatch", payload
    if task["kind"] == "c1_gain":
        if (
            provenance.get("trigger_manifest_sha256")
            != task["expected"]["trigger_manifest_sha256"]
            or provenance.get("trigger_producer_sha256")
            != task.get("trigger_producer_locks")
        ):
            return False, "runtime_trigger_provenance_mismatch", payload
        if payload["status"] == "complete" and (
            not isinstance(payload.get("core_rate_500ms_hz"), list)
            or len(payload["core_rate_500ms_hz"]) != 2
            or not isinstance(payload.get("gain_plateau_gate_pass"), bool)
        ):
            return False, "invalid_gain_observables", payload
        return True, "valid", payload
    obs_value = payload.get("observables_path")
    obs_sha = payload.get("observables_sha256")
    if not isinstance(obs_value, str) or not isinstance(obs_sha, str):
        return False, "missing_observables_provenance", payload
    obs_path = ROOT / obs_value
    if not obs_path.is_file() or _sha(obs_path) != obs_sha:
        return False, "observables_sha_mismatch", payload
    if payload["status"] == "complete":
        required = {
            "phasec1_observables_schema", "hierarchical_schema", "bin_ms",
            "E_rate_grid", "I_rate_grid", "source_rate_hz", "rest_mask",
            "active_area_fraction", "kymograph", "axis_positions",
            "rho80_active_core_by_block_window",
            "block_isi_cv2_by_panel_neuron",
            "block_refractory_isi_fraction_by_panel_neuron",
            "pair_corr_by_block_and_pair",
            "pair_null_median_by_block_and_draw",
            "active_area_fraction_by_block_window",
        }
        try:
            with np.load(obs_path, allow_pickle=False) as data:
                if not required.issubset(data.files):
                    return False, "missing_C1_observables", payload
                if str(np.asarray(
                    data["phasec1_observables_schema"]
                ).reshape(()).item()) != CELL.C1_OBSERVABLES_SCHEMA:
                    return False, "C1_observables_schema_mismatch", payload
        except (OSError, TypeError, ValueError) as exc:
            return False, f"invalid_C1_observables:{exc}", payload
    return True, "valid", payload


def _meminfo_kb():
    with open("/proc/meminfo", encoding="utf-8") as handle:
        return {
            key: int(value.split()[0])
            for key, value in (
                line.split(":", 1) for line in handle if ":" in line
            )
        }


def mem_available_gb():
    return _meminfo_kb()["MemAvailable"] / 1024 ** 2


def swap_used_kb():
    row = _meminfo_kb()
    return row["SwapTotal"] - row["SwapFree"]


def swap_growth_exceeded(baseline_kb, limit_mb):
    limit_kb = int(round(float(limit_mb) * 1024.0))
    return swap_used_kb() - int(baseline_kb) > limit_kb


def _resource_cap(args):
    cpu_cap = max(0, (os.cpu_count() or 1) - int(args.reserve_cpus))
    budget = max(0.0, mem_available_gb() - float(args.reserve_gb))
    mem_cap = max(
        0, math.floor(budget / (1.25 * float(args.worker_rss_gb)))
    )
    return min(cpu_cap, mem_cap, int(args.max_workers), 12)


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_json_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite coordinator summary: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def run(args):
    if args.max_workers > 12 or args.wave_size > 12:
        raise SystemExit("Phase-C1 concurrency is capped at 12 workers")
    if args.reserve_gb < 96 or args.reserve_cpus < 8:
        raise SystemExit("Phase-C1 requires >=96GB and >=8 CPU reserve")
    if not math.isfinite(args.worker_rss_gb) or args.worker_rss_gb <= 0:
        raise SystemExit("--worker-rss-gb must be a measured positive value")
    if (
        not math.isfinite(args.max_swap_growth_mb)
        or args.max_swap_growth_mb < 0
        or args.max_swap_growth_mb > 256
    ):
        raise SystemExit("--max-swap-growth-mb must be within [0,256]")
    manifest, coordinate, coordinate_path, trigger, producer_locks = (
        _load_contracts(
        require_trigger=args.phase == "gain"
        )
    )
    if args.phase == "base":
        all_tasks, invalid = base_tasks(
            manifest, coordinate, coordinate_path
        )
    else:
        all_tasks = gain_tasks(
            manifest, coordinate, trigger, coordinate_path
        )
        invalid = _cells(coordinate)[1]
    pending, skipped, conflicts = [], [], []
    for task in all_tasks:
        if not os.path.exists(task["output"]):
            pending.append(task)
            continue
        valid, reason, payload = validate_terminal_output(
            task["output"], task, producer_locks=producer_locks
        )
        if valid:
            if getattr(args, "resume", False):
                skipped.append({
                    "key": task["key"],
                    "artifact_sha256": _sha(task["output"]),
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
            "existing conflicting/nonterminal parts require explicit "
            "invalidation: " + json.dumps(conflicts[:5], sort_keys=True)
        )
    if pending and _resource_cap(args) < 1:
        raise SystemExit("resource guard authorizes no worker")

    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_p{os.getpid()}"
    logs_root = OUT / "logs/phasec1" / run_id
    logs_root.mkdir(parents=True, exist_ok=True)
    resource_log = logs_root / "resource_log.jsonl"
    swap0 = swap_used_kb()
    completed, failures = [], []
    launched = 0
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    print(
        f"[phasec1] phase={args.phase} expected={len(all_tasks)} "
        f"invalid_physical_skipped={len(invalid)} skipped={len(skipped)} "
        f"pending={len(pending)} cap={_resource_cap(args)} "
        f"MemAvailable={mem_available_gb():.1f}GB swap0={swap0}kB",
        flush=True,
    )
    while pending:
        if swap_growth_exceeded(swap0, args.max_swap_growth_mb):
            raise SystemExit("swap growth exceeded tolerance before next wave")
        cap = _resource_cap(args)
        if cap < 1:
            raise SystemExit("resource guard authorizes no next wave")
        wave = [pending.pop(0) for _ in range(
            min(len(pending), cap, args.wave_size)
        )]
        running = []
        for task in wave:
            if (
                mem_available_gb() < args.reserve_gb
                or swap_growth_exceeded(swap0, args.max_swap_growth_mb)
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
            item = {
                **task, "process": process, "handle": handle,
                "log": str(log_path), "started": time.time(),
            }
            running.append(item)
            _append(resource_log, {
                "event": "launch", "time": time.time(),
                "pid": process.pid, "key": task["key"],
                "cmd": task["cmd"], "mem_available_gb": mem_available_gb(),
                "swap_used_kb": swap_used_kb(),
            })
        if not running:
            raise SystemExit("resource guard blocked the entire next wave")
        last_heartbeat = 0.0
        while running:
            if swap_growth_exceeded(swap0, args.max_swap_growth_mb):
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
                    "swap growth exceeded tolerance; stopped this "
                    "coordinator's workers"
                )
            next_running = []
            for task in running:
                code = task["process"].poll()
                if code is None:
                    next_running.append(task)
                    continue
                task["handle"].close()
                valid, reason, payload = validate_terminal_output(
                    task["output"], task, producer_locks=producer_locks
                )
                row = {
                    "event": "finish", "time": time.time(),
                    "pid": task["process"].pid, "key": task["key"],
                    "exit_code": code, "valid_terminal": valid,
                    "validation_reason": reason,
                    "wall_s": round(time.time() - task["started"], 2),
                    "artifact_sha256": (
                        _sha(task["output"])
                        if os.path.exists(task["output"]) else None
                    ),
                    "child_peak_rss_gb": (
                        payload.get("peak_rss_gb")
                        if isinstance(payload, dict) else None
                    ),
                    "mem_available_gb": mem_available_gb(),
                    "swap_used_kb": swap_used_kb(),
                }
                _append(resource_log, row)
                if code == 0 and valid:
                    completed.append(row)
                else:
                    failures.append({
                        "key": task["key"], "exit_code": code,
                        "validation_reason": reason, "log": task["log"],
                    })
                print(
                    f"[phasec1] {'ok' if code == 0 and valid else 'FAIL'} "
                    f"{task['key']} running={len(running)-1} "
                    f"pending={len(pending)}",
                    flush=True,
                )
            running = next_running
            if running:
                now = time.time()
                if now - last_heartbeat >= 30.0:
                    _append(resource_log, {
                        "event": "heartbeat", "time": now,
                        "running_pids": [
                            row["process"].pid for row in running
                        ],
                        "n_running": len(running),
                        "n_pending": len(pending),
                        "mem_available_gb": mem_available_gb(),
                        "swap_used_kb": swap_used_kb(),
                    })
                    last_heartbeat = now
                time.sleep(args.poll_s)
        if failures:
            break

    gain_analysis = None
    if args.phase == "gain" and not failures and not pending:
        import scripts.analyze_topic4_zm_phasec1_gain as GAIN_ANALYZER
        try:
            gain_analysis = GAIN_ANALYZER.analyze_all(
                TRIGGER_MANIFEST_PATH, write=True
            )
        except Exception as exc:  # fail closed; raw parts remain reusable
            failures.append({
                "key": "conditional_gain_pure_analysis",
                "exit_code": None,
                "validation_reason": f"{type(exc).__name__}:{exc}",
                "log": None,
            })

    summary = {
        "schema": "zm_phasec1_coordinator_v1_2026-07-28",
        "run_id": run_id,
        "phase": args.phase,
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "trigger_manifest_sha256": (
            None if trigger is None else trigger["manifest_sha256"]
        ),
        "n_expected_simulations": len(all_tasks),
        "n_invalid_physical_skipped": len(invalid),
        "invalid_physical_cells": [
            {
                "seed": row["seed"], "tier": row["tier"],
                "cell_id": row["cell_id"], "reasons": row.get("reasons", []),
            }
            for row in invalid
        ],
        "n_skipped_valid": len(skipped),
        "n_launched": launched,
        "n_completed_this_run": len(completed),
        "n_pending_after_stop": len(pending),
        "n_failures": len(failures),
        "failures": failures,
        "conditional_gain_analysis": gain_analysis,
        "max_workers": args.max_workers,
        "wave_size": args.wave_size,
        "worker_rss_gb": args.worker_rss_gb,
        "reserve_gb": args.reserve_gb,
        "reserve_cpus": args.reserve_cpus,
        "swap_baseline_kb": swap0,
        "swap_final_kb": swap_used_kb(),
        "max_swap_growth_mb": args.max_swap_growth_mb,
        "resource_log_path": str(resource_log.relative_to(ROOT)),
        "claim_boundary": (
            "C1 frozen slow-field identity/maturation only; invalid physical "
            "coordinates are coverage, not simulated negatives; not lifecycle"
        ),
    }
    summary_path = OUT / "coordinator_runs" / (
        f"phasec1_{args.phase}_summary_{run_id}.json"
    )
    _publish_json_once(summary_path, summary)
    if failures or pending:
        raise SystemExit(f"Phase-C1 stopped; see {summary_path}")
    print(f"[phasec1] complete -> {summary_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("base", "gain"), required=True)
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
        raise SystemExit("Phase-C1 production requires --confirm-run")
    run(args)


if __name__ == "__main__":
    main()
