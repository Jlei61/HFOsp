#!/usr/bin/env python
"""Crash-safe, provenance-strict coordinator for the locked Phase-C0 matrix."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_metrics as PCM  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402


OUT = os.path.join(
    ROOT, "results", "topic4_sef_hfo", "zm_phase_c_tonic_identity"
)
MANIFEST_PATH = os.path.join(OUT, "phasec_manifest.json")
CELL = os.path.join(ROOT, "scripts", "run_topic4_zm_phasec_cell.py")
SEEDS_BY_RESOLUTION = {"dt": (1, 3, 4), "dt2": (1, 3)}
PHASES = ("bounded_mid__rising", "bounded_mid__peak")
NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")
GAIN_STATES = ("pre_entry__natural",) + PHASES
DELTAS = (0.05, 0.10)
TERMINAL_STATUSES = {"complete", "scientific_failure"}
MIN_MEASURED_WORKER_RSS_GB = {"identity": 7.23, "gain": 6.90}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _validate_live_producers(manifest):
    locks = manifest.get("provenance", {}).get("producer_file_sha256")
    if not isinstance(locks, dict) or not locks:
        raise RuntimeError("Phase-C manifest lacks producer-file hash locks")
    for relative_path, expected_sha in sorted(locks.items()):
        live_path = os.path.join(ROOT, relative_path)
        if not os.path.isfile(live_path) or _sha(live_path) != expected_sha:
            raise RuntimeError(
                f"Phase-C live producer hash mismatch: {relative_path}"
            )
    return locks


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
    rows = _meminfo_kb()
    return rows["SwapTotal"] - rows["SwapFree"]


def _identity_output(seed, phase, noise, *, resolution="dt"):
    return os.path.join(
        OUT, "parts", "c0_identity", resolution, f"seed{seed}",
        phase, noise, "identity.json",
    )


def _gain_output(seed, state, noise, delta, sign, *, resolution="dt"):
    label = (
        "d0_zero" if sign == 0
        else f"d{delta:g}_{'plus' if sign > 0 else 'minus'}"
    )
    return os.path.join(
        OUT, "parts", "c0_gain", resolution, f"seed{seed}",
        state, noise, label, "gain.json",
    )


def _resolution_seed_row(manifest, seed, resolution):
    seed_row = manifest["per_seed"][str(seed)]
    if resolution == "dt":
        return seed_row
    if resolution != "dt2":
        raise RuntimeError(f"unsupported Phase-C resolution: {resolution}")
    confirmations = seed_row.get("resolution_confirmations")
    row = (
        confirmations.get("dt2")
        if isinstance(confirmations, dict) else None
    )
    if not isinstance(row, dict):
        raise RuntimeError(
            f"final manifest lacks independent dt2 contract for seed {seed}"
        )
    if (
        row.get("resolution") != "dt2"
        or row.get("parent_config_sha")
        != seed_row.get("canonical_config_sha")
        or row.get("panel_selection_config_sha")
        != seed_row.get("canonical_config_sha")
        or row.get("panel_selection_resolution") != "parent_native_dt"
        or row.get("fixed_panels") != seed_row.get("fixed_panels")
    ):
        raise RuntimeError(
            f"dt2 parent-config/panel lineage mismatch for seed {seed}"
        )
    return row


def _source_lock(manifest, seed, state, noise, *, resolution="dt"):
    seed_row = manifest["per_seed"][str(seed)]
    resolution_row = _resolution_seed_row(manifest, seed, resolution)
    family = (
        resolution_row["c0_pre_entry_gain_control"]
        if state == "pre_entry__natural"
        else resolution_row["c0_carrier_states"][state.rsplit("__", 1)[-1]]
    )
    bank = {
        row["replicate"]: row for row in family["noise_banks"]
    }[noise]
    return {
        "config_sha": (
            seed_row["canonical_config_sha"]
            if resolution == "dt"
            else resolution_row["config_sha"]
        ),
        "state_hash": family["state"]["state_hash"],
        "state_file_sha256": family["state"]["file_sha256"],
        "noise_bank_sha": bank["bank_sha"],
        "panel_sha256": seed_row["fixed_panels"]["panel_sha256"],
    }


def tasks(phases, manifest=None, *, resolution="dt", seeds=None):
    rows = []
    manifest = manifest or {}
    manifest_sha = manifest.get("manifest_sha256")
    seeds = tuple(SEEDS_BY_RESOLUTION[resolution] if seeds is None else seeds)
    if not seeds or any(seed not in SEEDS_BY_RESOLUTION[resolution] for seed in seeds):
        raise RuntimeError(
            f"invalid {resolution} seed selection: {seeds}"
        )
    if "identity" in phases:
        for seed in seeds:
            for state in PHASES:
                for noise in NOISES:
                    expected = {
                        "schema": "zm_phasec_identity_cell_v1",
                        "manifest_sha256": manifest_sha,
                        "seed": seed,
                        "resolution": resolution,
                        "state_tag": state,
                        "replicate": noise,
                        "burn_in_ms": 500.0,
                        "measure_ms": 8000.0,
                    }
                    if manifest:
                        expected.update(_source_lock(
                            manifest, seed, state, noise
                            , resolution=resolution
                        ))
                    rows.append({
                        "kind": "identity",
                        "key": f"identity|s{seed}|{state}|{noise}",
                        "output": _identity_output(
                            seed, state, noise, resolution=resolution
                        ),
                        "expected": expected,
                        "cmd": [
                            sys.executable, CELL, "--mode", "identity",
                            "--seed", str(seed), "--state-tag", state,
                            "--replicate", noise, "--manifest", MANIFEST_PATH,
                            "--resolution", resolution,
                            "--confirm-run",
                        ],
                    })
    if "gain" in phases:
        for seed in seeds:
            for state in GAIN_STATES:
                for noise in NOISES:
                    arms = [(0.0, 0)]
                    arms.extend(
                        (delta, sign)
                        for delta in DELTAS for sign in (-1, 1)
                    )
                    for delta, sign in arms:
                        expected = {
                            "schema": "zm_phasec_gain_cell_v1",
                            "manifest_sha256": manifest_sha,
                            "seed": seed,
                            "resolution": resolution,
                            "state_tag": state,
                            "replicate": noise,
                            "delta_mV": float(delta),
                            "sign": int(sign),
                            "burn_in_ms": 500.0,
                            "measure_ms": 1000.0,
                        }
                        if manifest:
                            expected.update(_source_lock(
                                manifest, seed, state, noise
                                , resolution=resolution
                            ))
                        rows.append({
                            "kind": "gain",
                            "key": (
                                f"gain|s{seed}|{state}|{noise}|"
                                f"d{delta:g}|{sign:+d}"
                            ),
                            "output": _gain_output(
                                seed, state, noise, delta, sign
                                , resolution=resolution
                            ),
                            "expected": expected,
                            "cmd": [
                                sys.executable, CELL, "--mode", "gain",
                                "--seed", str(seed), "--state-tag", state,
                                "--replicate", noise,
                                "--delta-mV", str(delta),
                                "--sign", str(sign),
                                "--manifest", MANIFEST_PATH,
                                "--resolution", resolution,
                                "--confirm-run",
                            ],
                        })
    return rows


def validate_terminal_output(path, task):
    """Validate scientific/technical identity, hashes, and required observables."""
    if not os.path.exists(path):
        return False, "missing", None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_json:{exc}", None
    mismatches = []
    for key, expected in task["expected"].items():
        if expected is not None and payload.get(key) != expected:
            mismatches.append(key)
    if mismatches:
        return False, "identity_mismatch:" + ",".join(mismatches), payload
    if payload.get("status") not in TERMINAL_STATUSES:
        return False, "nonterminal_status", payload
    gates = payload.get("carrier_gates")
    if not isinstance(gates, dict) or any(
        not isinstance(gates.get(key), bool)
        for key in ("runaway", "whole_sheet_plateau", "empirical_rest_dwell")
    ):
        return False, "missing_carrier_gates", payload
    provenance = payload.get("runtime_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("manifest_sha256")
        != task["expected"].get("manifest_sha256")
        or not isinstance(provenance.get("producer_sha256"), dict)
        or provenance.get("self_vm_swap_kb_at_publish") != 0
    ):
        return False, "invalid_runtime_provenance", payload
    producer_locks = task.get("producer_locks")
    if (
        producer_locks is not None
        and provenance.get("producer_sha256") != producer_locks
    ):
        return False, "runtime_producer_hash_mismatch", payload
    if task.get("coordinator_run_id") is not None and (
        provenance.get("coordinator_run_id") != task["coordinator_run_id"]
        or provenance.get("coordinator_launch_token")
        != task.get("coordinator_launch_token")
    ):
        return False, "runtime_coordinator_identity_mismatch", payload
    if task["kind"] == "identity":
        observables = payload.get("observables_path")
        expected_sha = payload.get("observables_sha256")
        if not isinstance(observables, str) or not isinstance(expected_sha, str):
            return False, "missing_observables_provenance", payload
        obs_path = (
            observables if os.path.isabs(observables)
            else os.path.join(ROOT, observables)
        )
        if not os.path.exists(obs_path) or _sha(obs_path) != expected_sha:
            return False, "observables_sha_mismatch", payload
        if payload["status"] == "complete":
            try:
                with __import__("numpy").load(
                    obs_path, allow_pickle=False
                ) as z:
                    required = {
                        "hierarchical_schema",
                        "rho80_active_core_by_block_window",
                        "block_isi_cv2_by_panel_neuron",
                        "block_refractory_isi_numerator_by_stratum",
                        "block_refractory_isi_denominator_by_stratum",
                        "refractory_isi_stratum_names",
                        "pair_corr_by_block_and_pair",
                        "pair_null_median_by_block_and_draw",
                        "active_area_fraction_by_block_window",
                        "spatial_grid_n_occupied_E",
                        "spatial_grid_all_E_bins_occupied",
                        "spatial_active_floor_hz",
                        "spatial_area_denominator",
                        "analysis_panel_E_ids",
                        "pairwise_panel_E_ids",
                    }
                    if not required.issubset(z.files):
                        return False, "missing_hierarchical_arrays", payload
                    if str(
                        __import__("numpy").asarray(
                            z["hierarchical_schema"]
                        ).reshape(()).item()
                    ) != PCM.HIERARCHICAL_STATS_VERSION:
                        return False, "hierarchical_schema_mismatch", payload
            except (OSError, ValueError) as exc:
                return False, f"invalid_observables:{exc}", payload
    else:
        blocks = payload.get("core_rate_500ms_hz")
        if (
            payload["status"] == "complete"
            and (
                not isinstance(blocks, list)
                or len(blocks) != 2
                or not isinstance(payload.get("gain_plateau_gate_pass"), bool)
            )
        ):
            return False, "invalid_gain_block_observables", payload
    return True, "valid_terminal", payload


def valid_terminal_output(path, manifest_sha, task=None):
    """Backward-compatible boolean helper retained for existing tests."""
    if task is None:
        if not os.path.exists(path):
            return False
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            payload.get("manifest_sha256") == manifest_sha
            and payload.get("status") in TERMINAL_STATUSES
        )
    valid, _reason, _payload = validate_terminal_output(path, task)
    return valid


def _append(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_json_once(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite coordinator result: {path}")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def _resource_cap(args):
    cpu_cap = max(0, (os.cpu_count() or 1) - int(args.reserve_cpus))
    available = mem_available_gb()
    memory_budget = max(0.0, available - float(args.reserve_gb))
    mem_cap = max(
        0, math.floor(memory_budget / (1.25 * float(args.worker_rss_gb)))
    )
    return min(cpu_cap, mem_cap, int(args.max_workers))


def _swap_growth_exceeded(baseline_kb, limit_mb):
    limit_kb = int(round(float(limit_mb) * 1024.0))
    return swap_used_kb() - int(baseline_kb) > limit_kb


def _worker_swap_snapshot(running):
    return PRES.worker_swap_snapshot(
        row["proc"].pid for row in running
    )


def run(args):
    manifest = _load_json(MANIFEST_PATH)
    PCC.require_production_manifest(manifest)
    producer_locks = _validate_live_producers(manifest)
    manifest_sha = manifest["manifest_sha256"]
    manifest_file_sha = _sha(MANIFEST_PATH)
    if not math.isfinite(args.worker_rss_gb) or args.worker_rss_gb <= 0:
        raise SystemExit("--worker-rss-gb must be a positive measured full-cell RSS")
    if args.max_workers > 16 or args.wave_size > 16:
        raise SystemExit("Phase-C0 hard cap is 16 workers per wave")
    if args.reserve_gb < 96 or args.reserve_cpus < 8:
        raise SystemExit("Phase-C0 requires reserve_gb>=96 and reserve_cpus>=8")
    if (
        not math.isfinite(args.max_swap_growth_mb)
        or args.max_swap_growth_mb < 0
        or args.max_swap_growth_mb > 256
    ):
        raise SystemExit("--max-swap-growth-mb must be within [0,256]")
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
    selected = tuple(x.strip() for x in args.phases.split(",") if x.strip())
    unknown = set(selected) - {"identity", "gain"}
    if unknown or not selected:
        raise SystemExit(f"unknown phases: {sorted(unknown)}")
    required_worker_rss = max(
        MIN_MEASURED_WORKER_RSS_GB[phase] for phase in selected
    )
    if float(args.worker_rss_gb) < required_worker_rss:
        raise SystemExit(
            "--worker-rss-gb cannot be lower than the locked measured "
            f"value for {selected}: {required_worker_rss:g} GB"
        )
    resolution = getattr(args, "resolution", "dt")
    if resolution not in SEEDS_BY_RESOLUTION:
        raise SystemExit(f"unknown Phase-C resolution: {resolution}")
    selected_seeds = tuple(
        int(value) for value in (
            getattr(args, "seeds", "") or
            ",".join(str(seed) for seed in SEEDS_BY_RESOLUTION[resolution])
        ).split(",") if value.strip()
    )
    if resolution == "dt2":
        if selected_seeds != SEEDS_BY_RESOLUTION["dt2"]:
            raise SystemExit(
                "dt2 C0 confirmation requires both independently anchored "
                "supporting seeds 1 and 3"
            )
        native_path = os.path.join(OUT, "c0_identity_summary_dt.json")
        if not os.path.isfile(native_path):
            raise SystemExit(
                "dt2 C0 is conditional on a complete native identity summary"
            )
        native = _load_json(native_path)
        aggregate = native.get("aggregate", {})
        supporting = tuple(sorted(aggregate.get("supporting_seeds", [])))
        if (
            native.get("schema") != "zm_phasec_c0_summary_v1"
            or native.get("resolution") != "dt"
            or native.get("manifest_sha256") != manifest_sha
            or not str(aggregate.get("verdict", "")).endswith("_supported")
            or not set(selected_seeds).issubset(supporting)
        ):
            raise SystemExit(
                "dt2 C0 requires a native supported identity in both "
                "independently anchored seeds 1 and 3"
            )
    all_tasks = tasks(
        selected, manifest, resolution=resolution, seeds=selected_seeds
    )
    for row in all_tasks:
        row["expected"]["manifest_file_sha256"] = manifest_file_sha
        row["producer_locks"] = producer_locks
    invalid_existing = []
    skipped = []
    pending = []
    for row in all_tasks:
        if not os.path.exists(row["output"]):
            pending.append(row)
            continue
        valid, reason, payload = validate_terminal_output(row["output"], row)
        if valid:
            if getattr(args, "resume", False):
                receipt_path = PRES.resource_receipt_path(row["output"])
                audit_ok, audit_reason, receipt = (
                    PRES.validate_resource_receipt(
                        receipt_path,
                        artifact_path=row["output"],
                        artifact_root=ROOT,
                        manifest_sha256=manifest_sha,
                        task_key=row["key"],
                    )
                )
                if audit_ok:
                    skipped.append({
                        "key": row["key"],
                        "artifact_sha256": _sha(row["output"]),
                        "peak_rss_gb": payload.get("peak_rss_gb"),
                        "resource_receipt_path": os.path.relpath(
                            receipt_path, ROOT
                        ),
                        "resource_receipt_sha256": receipt["receipt_sha256"],
                    })
                else:
                    invalid_existing.append({
                        "path": row["output"], "reason": audit_reason
                    })
            else:
                invalid_existing.append({
                    "path": row["output"],
                    "reason": "valid_terminal_requires_--resume",
                })
        else:
            invalid_existing.append({
                "path": row["output"], "reason": reason
            })
    if invalid_existing:
        raise SystemExit(
            "conflicting/nonterminal existing parts preserved; explicit "
            "invalidation is required: "
            + json.dumps(invalid_existing[:5], sort_keys=True)
        )
    initial_cap = _resource_cap(args)
    if pending and initial_cap < 1:
        raise SystemExit("resource guard authorizes no worker")
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"_p{os.getpid()}"
    log_path = os.path.join(
        OUT, "logs", "phasec0", resolution,
        f"resource_log_{run_id}.jsonl"
    )
    logs_root = os.path.join(OUT, "logs", "phasec0", resolution, run_id)
    os.makedirs(logs_root, exist_ok=True)
    swap0_kb = swap_used_kb()
    n_launched = 0
    failures = []
    completed = []
    worker_swap_peak_kb = 0
    worker_swap_audit = {}
    owned_running = []
    cleanup_state = {"finalized": False}
    print(
        f"[phasec0] expected={len(all_tasks)} skipped={len(skipped)} "
        f"pending={len(pending)} cap={initial_cap} "
        f"MemAvailable={mem_available_gb():.1f}GB swap0={swap0_kb}kB",
        flush=True,
    )
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    def _cleanup_at_exit():
        if cleanup_state["finalized"]:
            return
        try:
            if owned_running:
                _append(log_path, {
                    "event": "abort_cleanup",
                    "run_id": run_id,
                    "time": time.time(),
                    "reason": "coordinator_exit_or_exception",
                    "owned_pids": [
                        row["proc"].pid for row in owned_running
                    ],
                    "mem_available_gb": mem_available_gb(),
                    "swap_used_kb": swap_used_kb(),
                })
                PRES.terminate_owned_workers(owned_running)
            partial_path = os.path.join(
                OUT, "coordinator_runs", resolution,
                f"phasec0_partial_abort_{run_id}.json",
            )
            _publish_json_once(partial_path, {
                "schema": "zm_phasec0_coordinator_partial_abort_v1",
                "run_id": run_id,
                "manifest_sha256": manifest_sha,
                "resolution": resolution,
                "n_expected": len(all_tasks),
                "n_skipped_valid": len(skipped),
                "n_launched": n_launched,
                "n_completed_before_abort": len(completed),
                "n_pending_at_abort": max(
                    0, len(all_tasks) - len(skipped) - len(completed)
                ),
                "n_queue_pending_at_abort": len(pending),
                "n_owned_inflight_at_abort": len(owned_running),
                "owned_pids_at_cleanup": [
                    row["proc"].pid for row in owned_running
                ],
                "worker_swap_sampled_observed_max_kb": worker_swap_peak_kb,
                "worker_swap_audit_by_launch_token": worker_swap_audit,
                "resource_log_path": os.path.relpath(log_path, ROOT),
                "evidence_scope": (
                    "partial abort record; sampled VmSwap plus any available "
                    "pre-publish child self snapshots, not kernel peaks"
                ),
                "finished_at": time.time(),
            })
        except Exception:
            # Interpreter shutdown must still finish after best-effort cleanup.
            PRES.terminate_owned_workers(owned_running)

    atexit.register(_cleanup_at_exit)
    previous_signal_handlers = PRES.install_coordinator_signal_handlers()

    while pending:
        if _swap_growth_exceeded(swap0_kb, args.max_swap_growth_mb):
            raise SystemExit(
                "swap growth exceeded tolerance before the next wave"
            )
        wave_cap = _resource_cap(args)
        if wave_cap < 1:
            raise SystemExit("resource guard authorizes no next-wave worker")
        wave_n = min(len(pending), wave_cap, int(args.wave_size))
        wave = [pending.pop(0) for _ in range(wave_n)]
        running = []
        owned_running[:] = running
        for row in wave:
            if (
                mem_available_gb() < args.reserve_gb
                or _swap_growth_exceeded(
                    swap0_kb, args.max_swap_growth_mb
                )
            ):
                pending[:0] = wave[len(running):]
                break
            safe_key = row["key"].replace("|", "__").replace("/", "_")
            log_file = os.path.join(logs_root, safe_key + ".log")
            handle = open(log_file, "x", encoding="utf-8")
            launch_token = uuid.uuid4().hex
            child_env = dict(env)
            child_env[PRES.COORDINATOR_RUN_ENV] = run_id
            child_env[PRES.COORDINATOR_TOKEN_ENV] = launch_token
            previous_mask = PRES.block_coordinator_termination_signals()
            proc = None
            item = None
            try:
                proc = subprocess.Popen(
                    row["cmd"], cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env=child_env,
                )
                launched_at = time.time()
                item = {
                    **row, "proc": proc, "handle": handle,
                    "log": log_file, "started": launched_at,
                    "coordinator_run_id": run_id,
                    "coordinator_launch_token": launch_token,
                }
                running.append(item)
                owned_running[:] = running
                PRES.register_worker_swap_audit(
                    worker_swap_audit,
                    pid=proc.pid,
                    task_key=row["key"],
                    run_id=run_id,
                    launch_token=launch_token,
                    launched_at=launched_at,
                )
                n_launched += 1
            except BaseException:
                if proc is None:
                    handle.close()
                else:
                    PRES.terminate_owned_workers([
                        item or {"proc": proc, "handle": handle}
                    ])
                _append(log_path, {
                    "event": "abort",
                    "run_id": run_id,
                    "time": time.time(),
                    "reason": "worker_launch_failed",
                    "key": row["key"],
                })
                PRES.terminate_owned_workers(running)
                raise
            finally:
                PRES.restore_coordinator_signal_mask(previous_mask)
            _append(log_path, {
                "event": "launch",
                "run_id": run_id,
                "time": time.time(),
                "pid": proc.pid,
                "key": row["key"],
                "manifest_sha256": manifest_sha,
                "manifest_file_sha256": manifest_file_sha,
                "mem_available_gb": mem_available_gb(),
                "swap_used_kb": swap_used_kb(),
                "cmd": row["cmd"],
                "n_launched": n_launched,
            })
        if not running:
            raise SystemExit("resource guard blocked the entire next wave")
        last_heartbeat = 0.0
        while running:
            worker_swap = _worker_swap_snapshot(running)
            sampled_at = time.time()
            PRES.update_worker_swap_audit(
                worker_swap_audit,
                worker_swap,
                sampled_at=sampled_at,
                audit_key_by_pid={
                    str(row["proc"].pid): row["coordinator_launch_token"]
                    for row in running
                },
            )
            worker_swap_peak_kb = max(
                worker_swap_peak_kb,
                int(worker_swap["worker_swap_max_kb"]),
            )
            if worker_swap["worker_swap_max_kb"] * 1024 > worker_swap_allowed:
                _append(log_path, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "worker_sampled_swap_nonzero",
                    **worker_swap,
                })
                PRES.terminate_owned_workers(running)
                raise SystemExit(
                    "a Phase-C worker acquired VmSwap; run invalidated"
                )
            if mem_available_gb() < float(args.reserve_gb):
                _append(log_path, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "mem_available_below_reserve",
                    "mem_available_gb": mem_available_gb(),
                })
                PRES.terminate_owned_workers(running)
                raise SystemExit(
                    "MemAvailable fell below the locked running-wave reserve"
                )
            if _swap_growth_exceeded(
                swap0_kb, args.max_swap_growth_mb
            ):
                _append(log_path, {
                    "event": "abort", "run_id": run_id,
                    "time": time.time(),
                    "reason": "shared_host_swap_growth",
                    "swap_used_kb": swap_used_kb(),
                })
                PRES.terminate_owned_workers(running)
                raise SystemExit(
                    "swap growth exceeded tolerance while a wave was running"
                )
            next_running = []
            for row in running:
                code = row["proc"].poll()
                if code is None:
                    next_running.append(row)
                    continue
                row["handle"].close()
                valid, reason, payload = validate_terminal_output(
                    row["output"], row
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
                    PRES.record_final_worker_swap(
                        worker_swap_audit,
                        pid=row["proc"].pid,
                        launch_token=row["coordinator_launch_token"],
                        value_kb=final_swap,
                        sampled_at=time.time(),
                    )
                ok = code == 0 and valid
                receipt_path = PRES.resource_receipt_path(row["output"])
                if ok:
                    try:
                        receipt = PRES.build_resource_receipt(
                            artifact_path=row["output"],
                            artifact_root=ROOT,
                            artifact_sha256=_sha(row["output"]),
                            manifest_sha256=manifest_sha,
                            task_key=row["key"],
                            run_id=run_id,
                            launch_token=row["coordinator_launch_token"],
                            pid=row["proc"].pid,
                            audit_row=worker_swap_audit.get(
                                row["coordinator_launch_token"], {}
                            ),
                            sampled_allowed_bytes=worker_swap_allowed,
                        )
                        PRES.publish_resource_receipt_once(
                            receipt_path, receipt
                        )
                    except Exception as exc:
                        ok = False
                        valid = False
                        reason = (
                            "resource_receipt_failure:"
                            f"{type(exc).__name__}:{exc}"
                        )
                finish = {
                    "event": "finish",
                    "run_id": run_id,
                    "time": time.time(),
                    "pid": row["proc"].pid,
                    "key": row["key"],
                    "exit_code": code,
                    "valid_terminal": valid,
                    "validation_reason": reason,
                    "wall_s": round(time.time() - row["started"], 2),
                    "mem_available_gb": mem_available_gb(),
                    "swap_used_kb": swap_used_kb(),
                    "artifact_sha256": (
                        _sha(row["output"])
                        if os.path.exists(row["output"]) else None
                    ),
                    "observables_sha256": (
                        payload.get("observables_sha256")
                        if isinstance(payload, dict) else None
                    ),
                    "child_peak_rss_gb": (
                        payload.get("peak_rss_gb")
                        if isinstance(payload, dict) else None
                    ),
                    "n_launched": n_launched,
                }
                _append(log_path, finish)
                if ok:
                    completed.append(finish)
                else:
                    failures.append({
                        "key": row["key"],
                        "exit_code": code,
                        "validation_reason": reason,
                        "log": row["log"],
                    })
                print(
                    f"[phasec0] {'ok' if ok else 'FAIL'} {row['key']} "
                    f"wave_running={len(running)-1} pending={len(pending)}",
                    flush=True,
                )
            running = next_running
            owned_running[:] = running
            if running:
                now = time.time()
                if now - last_heartbeat >= 30.0:
                    _append(log_path, {
                        "event": "heartbeat",
                        "run_id": run_id,
                        "time": now,
                        "running_pids": [
                            row["proc"].pid for row in running
                        ],
                        "n_running": len(running),
                        "n_pending": len(pending),
                        "n_launched": n_launched,
                        "mem_available_gb": mem_available_gb(),
                        "swap_used_kb": swap_used_kb(),
                        **worker_swap,
                    })
                    last_heartbeat = now
                time.sleep(args.poll_s)
        if failures:
            break

    summary = {
        "schema": "zm_phasec0_coordinator_v2",
        "run_id": run_id,
        "manifest_path": os.path.relpath(MANIFEST_PATH, ROOT),
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_file_sha,
        "resolution": resolution,
        "seeds": list(selected_seeds),
        "phases": list(selected),
        "n_expected": len(all_tasks),
        "n_skipped_valid": len(skipped),
        "skipped_valid": skipped,
        "n_launched": n_launched,
        "n_completed_this_run": len(completed),
        "n_pending_after_stop": len(pending),
        "n_failures": len(failures),
        "failures": failures,
        "worker_rss_gb": args.worker_rss_gb,
        "max_workers": args.max_workers,
        "wave_size": args.wave_size,
        "reserve_cpus": args.reserve_cpus,
        "reserve_gb": args.reserve_gb,
        "swap_baseline_kb": swap0_kb,
        "swap_final_kb": swap_used_kb(),
        "host_swap_growth_tolerance_mb": args.max_swap_growth_mb,
        "worker_swap_sampled_allowed_bytes": worker_swap_allowed,
        "worker_swap_observed_max_kb": worker_swap_peak_kb,
        "worker_swap_poll_s": args.poll_s,
        "worker_swap_audit_by_launch_token": worker_swap_audit,
        "resource_log_path": os.path.relpath(log_path, ROOT),
        "finished_at": time.time(),
    }
    path = os.path.join(
        OUT, "coordinator_runs", resolution,
        f"phasec0_summary_{run_id}.json"
    )
    _publish_json_once(path, summary)
    cleanup_state["finalized"] = True
    owned_running.clear()
    PRES.restore_signal_handlers(previous_signal_handlers)
    atexit.unregister(_cleanup_at_exit)
    if failures or pending:
        raise SystemExit(
            f"Phase-C0 stopped with failures/pending; see {path}"
        )
    print(f"[phasec0] complete -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="identity,gain")
    parser.add_argument("--resolution", choices=("dt", "dt2"), default="dt")
    parser.add_argument(
        "--seeds", default="",
        help="comma-separated subset; defaults to all locked seeds at resolution",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--wave-size", type=int, default=12)
    parser.add_argument(
        "--worker-rss-gb", type=float, required=True,
        help="measured peak RSS of one full production cell",
    )
    parser.add_argument("--reserve-cpus", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=96.0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument(
        "--max-swap-growth-mb",
        type=float,
        default=64.0,
        help=(
            "bounded shared-host swap jitter allowance before fail-close"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("production coordinator requires --confirm-run")
    run(args)


if __name__ == "__main__":
    main()
