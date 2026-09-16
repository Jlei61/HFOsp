#!/usr/bin/env python3
"""Resumable G0/G1 physical controller for multi-event distribution v2.1."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from scripts.resume_topic4_multidimensional_pilot import dependency_paths


OUT = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1"
DESIGN_CONFIG = ROOT / "config/topic4_multievent_distribution_search_v2_1.json"
INHERITED = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2/initial_candidate_manifest.json"
REFERENCE_EXECUTION = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1/execution_config.json"
WORKER = ROOT / "scripts/run_topic4_multidimensional_worker.py"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
G0_IDS = [
    "v2_anchor_historical__baseline",
    "v2_pop0_sobol_000",
    "v2_pop1_sobol_000",
]


def repaired_observation(worker_json):
    """Apply the frozen repaired observer; worker lineage onsets are not training data."""
    from src.topic4_observation_repaired import observe
    worker_json = Path(worker_json)
    record = base.read(worker_json)
    arrays_path = Path(record["arrays"]["path"])
    contract_path = (
        ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/observation_contract.json"
    )
    qualification_path = contract_path.parent / "qualification.json"
    qualification = base.read(qualification_path)
    if base.sha(contract_path) != qualification["observer_sha256"]:
        raise RuntimeError("frozen repaired observation contract changed")
    output_dir = worker_json.parent.parent / "repaired_observation"
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / worker_json.name
    npz_path = json_path.with_suffix(".npz")
    if json_path.exists():
        metadata = base.read(json_path)
        if (metadata["worker_sha256"] == base.sha(worker_json)
                and metadata["arrays_sha256"] == base.sha(arrays_path)
                and metadata["observation_contract_sha256"] == base.sha(contract_path)
                and base.sha(npz_path) == metadata["observation_arrays_sha256"]):
            return json_path, npz_path, metadata
        raise RuntimeError(f"stale repaired observation: {json_path}")
    with np.load(arrays_path) as arrays:
        result = observe(
            arrays["contact_envelope"],
            float(arrays["contact_envelope_dt_ms"]),
            base.read(contract_path),
        )
    temporary = npz_path.with_suffix(".npz.tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            centroid_ms=np.asarray(result["centroid_ms"], np.float32),
            recruitment_ms=np.asarray(result["recruitment_ms"], np.float32),
            primary_event_indices=np.asarray(result["primary_event_indices"], np.int32),
            windows_ms=np.asarray(result["windows_ms"], np.float32).reshape(-1, 2),
        )
    temporary.replace(npz_path)
    metadata = {
        "status": "FROZEN_REPAIRED_OBSERVATION_COMPLETE",
        "worker_path": str(worker_json), "worker_sha256": base.sha(worker_json),
        "arrays_path": str(arrays_path), "arrays_sha256": base.sha(arrays_path),
        "observation_contract_path": str(contract_path),
        "observation_contract_sha256": base.sha(contract_path),
        "observation_arrays_path": str(npz_path),
        "observation_arrays_sha256": base.sha(npz_path),
        "n_detected_windows": int(result["n_groups"]),
        "n_primary_events": int(result["n_primary_events"]),
        "events": result["events"],
        "boundary_or_low_window_support": result["boundary_or_low_window_support"],
        "training_table_source": "centroid_ms at frozen primary_event_indices",
        "worker_lineage_onsets_used_for_training": False,
    }
    base.write(json_path, metadata)
    return json_path, npz_path, metadata


def _status(name, **payload):
    base.write(OUT / "status.json", {
        "status": name, "updated_unix": time.time(), **payload,
    })


def _prepare_execution():
    folder = OUT / "execution" / "training_24s"
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "execution_config.json"
    manifest_path = folder / "candidate_manifest.json"
    snapshot_path = folder / "runtime_snapshot.json"
    old_manifest = base.read(INHERITED)
    if (base.sha(INHERITED)
            != base.read(DESIGN_CONFIG)["inheritance"]["initial_candidate_manifest"]["sha256"]):
        raise RuntimeError("the inherited 48-candidate manifest changed")
    if not config_path.exists():
        cfg = copy.deepcopy(base.read(REFERENCE_EXECUTION))
        cfg.update({
            # Reuse the already-qualified physical worker role; the output root
            # and v2.1 manifest keep this execution scientifically separate.
            "scientific_role": "development_only_multidimensional_interictal_pilot",
            "output_root": str(folder),
            "candidate_manifest": str(manifest_path),
            "corrected_networks": {
                str(seed): base.network_record(seed) for seed in (2511, 2512)
            },
        })
        cfg["search"] = {
            "fit_network_seeds": [2511, 2512],
            "dynamics_seeds": [2511, 2512, 7101, 7102],
            "simulation": {
                "duration_ms": 24000.0,
                "early_stop_runaway": True,
                "late_runaway_is_invalid": True,
            },
            "contact_readout": copy.deepcopy(
                base.read(REFERENCE_EXECUTION)["search"]["contact_readout"]
            ),
        }
        base.write(config_path, cfg)
        base.write(manifest_path, {
            "config_sha256": base.sha(config_path),
            "candidates": old_manifest["candidates"],
            "source_manifest": str(INHERITED),
            "source_manifest_sha256": base.sha(INHERITED),
            "master_seed": old_manifest["master_seed"],
            "candidate_order_inherited_unchanged": True,
            "frozen_before_simulation": True,
        })
        seeds = {
            "scripts/run_topic4_multidimensional_worker.py",
            "scripts/run_topic4_rev12_node_worker.py",
            "src/topic4_xy_search.py",
            "src/topic4_multidimensional_parameters.py",
        }
        sources = dependency_paths(ROOT, seeds)
        lock = {path: base.sha(ROOT / path) for path in sorted(sources)}
        base.write(snapshot_path, {
            "source_hashes": lock,
            "input_hashes": {
                str(config_path.resolve()): base.sha(config_path),
                str(manifest_path.resolve()): base.sha(manifest_path),
            },
            "identity_kind": "dependency_scoped_source_hash_snapshot",
            "not_final_substrate_freeze": True,
        })
    for path in (config_path, manifest_path):
        expected = base.read(snapshot_path)["input_hashes"].get(str(path.resolve()))
        if expected != base.sha(path):
            raise RuntimeError(f"frozen execution input changed: {path}")
    for rel, digest in base.read(snapshot_path)["source_hashes"].items():
        if base.sha(ROOT / rel) != digest:
            raise RuntimeError(f"physical runtime dependency changed: {rel}")
    if base.read(manifest_path)["config_sha256"] != base.sha(config_path):
        raise RuntimeError("candidate manifest/config binding changed")
    return folder, config_path, manifest_path, snapshot_path


def _stem(candidate_id, topology_seed, dynamics_seed):
    return f"{candidate_id}_topo_{topology_seed}_dyn_{dynamics_seed}"


def _complete(path, snapshot_path):
    if not path.exists():
        return False
    record = base.read(path)
    arrays = Path(record.get("arrays", {}).get("path", ""))
    provenance_snapshot = record.get("provenance", {}).get(
        "source_hash_snapshot", {}
    ).get("sha256")
    snapshot_matches = provenance_snapshot == base.sha(snapshot_path)
    if not snapshot_matches:
        migration_path = Path(snapshot_path).parent / "runtime_migration.json"
        if migration_path.exists():
            migration = base.read(migration_path)
            migrated = migration.get("migrated_completed_workers", {}).get(
                path.name, {}
            )
            snapshot_matches = bool(
                provenance_snapshot == migration.get("old_snapshot_sha256")
                and migrated.get("new_worker_sha256") == base.sha(path)
                and migration.get("new_snapshot_sha256") == base.sha(snapshot_path)
                and migration.get("physics_or_observer_changed") is False
            )
    return bool(
        record.get("status") == "REV12ND_NODE_WORKER_COMPLETE"
        and record.get("execution_status") == "COMPLETE"
        and arrays.is_file()
        and base.sha(arrays) == record["arrays"]["sha256"]
        and snapshot_matches
    )


def _process_tree(root_pid):
    parent = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            values = (proc / "stat").read_text().split()
            parent[int(proc.name)] = int(values[3])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            pass
    selected = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected)


def _memory_sample(pid):
    rss = vms = pss = 0
    live = []
    for child in _process_tree(pid):
        try:
            fields = (Path(f"/proc/{child}/statm").read_text().split())
            page = os.sysconf("SC_PAGE_SIZE")
            vms += int(fields[0]) * page
            rss += int(fields[1]) * page
            rollup = Path(f"/proc/{child}/smaps_rollup")
            if rollup.exists():
                for line in rollup.read_text().splitlines():
                    if line.startswith("Pss:"):
                        pss += int(line.split()[1]) * 1024
                        break
            live.append(child)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            pass
    return {"pids": live, "rss_bytes": rss, "pss_bytes": pss,
            "vms_bytes": vms}


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _other_snn_reserve_gib(exclude):
    reserve = 0.0
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            if pid in exclude:
                continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(Path(os.fsdecode(arg)).name in {
                    "run_topic4_rev12_node_worker.py",
                    "run_topic4_multidimensional_worker.py",
            } for arg in args if arg):
                continue
            fields = (proc / "statm").read_text().split()
            rss = int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
            reserve += max(0.0, 18.0 - rss)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            pass
    return reserve


def _run_jobs(stage, jobs, *, maximum_workers, execution=None):
    folder, config_path, _, snapshot_path = (
        _prepare_execution() if execution is None else execution
    )
    workers = folder / "workers"
    logs = folder / "run_logs"
    resources = folder / "resource_logs"
    workers.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    resources.mkdir(exist_ok=True)
    pending, complete = [], []
    for candidate_id, topology_seed, dynamics_seed in jobs:
        output = workers / f"{_stem(candidate_id, topology_seed, dynamics_seed)}.json"
        (complete if _complete(output, snapshot_path) else pending).append(
            (candidate_id, topology_seed, dynamics_seed)
        )
    active = {}
    failures = []
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    while pending or active:
        for job, handle in list(active.items()):
            proc, stream, peaks = handle
            sample = _memory_sample(proc.pid)
            peaks["rss_bytes"] = max(peaks["rss_bytes"], sample["rss_bytes"])
            peaks["pss_bytes"] = max(peaks["pss_bytes"], sample["pss_bytes"])
            peaks["vms_bytes"] = max(peaks["vms_bytes"], sample["vms_bytes"])
            stem = _stem(*job)
            _append_jsonl(resources / f"{stem}.jsonl", {
                "unix": time.time(), **sample,
                "output_bytes": sum(
                    path.stat().st_size for path in workers.glob(f"{stem}*")
                ),
            })
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            del active[job]
            base.write(resources / f"{stem}_summary.json", {
                "candidate_id": job[0], "topology_seed": job[1],
                "dynamics_seed": job[2], "exit_code": code,
                "peak_process_tree": peaks,
            })
            output = workers / f"{stem}.json"
            if code != 0 or not _complete(output, snapshot_path):
                failures.append({"job": job, "exit_code": code,
                                 "log": str(logs / f"{stem}.log")})
            else:
                complete.append(job)
        if failures:
            _status("WORKER_FAILURE_DRAINING", stage=stage,
                    failures=failures, running=len(active))
            if not active:
                raise RuntimeError(f"worker failures: {failures}")
            time.sleep(10)
            continue
        if shutil.disk_usage(OUT).free < 30 * 1024 ** 3:
            raise RuntimeError("less than 30 GiB free disk")
        active_pids = set()
        for proc, _, _ in active.values():
            active_pids.update(_memory_sample(proc.pid)["pids"])
        # RLIMIT_AS is a failure boundary, not a claim that physical RAM is used.
        # Reserve it conservatively for admission until measured canary PSS exists.
        other_reserve = _other_snn_reserve_gib(active_pids)
        running_reserve = 18.0 * len(active)
        allowance = max(0, int((
            base.available_gib() - 40.0 - other_reserve - running_reserve
        ) / 18.0))
        slots = min(maximum_workers - len(active), allowance, len(pending))
        for _ in range(max(0, slots)):
            job = pending.pop(0)
            candidate_id, topology_seed, dynamics_seed = job
            stem = _stem(*job)
            output = workers / f"{stem}.json"
            stream = open(logs / f"{stem}.log", "w")
            command = [
                "/usr/bin/prlimit", f"--as={18 * 1024 ** 3}", "--",
                PYTHON, str(WORKER), "--config", str(config_path),
                "--candidate-id", candidate_id, "--seed", str(topology_seed),
                "--topology-seed", str(topology_seed),
                "--dynamics-seed", str(dynamics_seed),
                "--expected-commit", commit,
                "--runtime-manifest", str(snapshot_path),
                "--artifact-root", str(ARTIFACT_ROOT),
                "--out-json", str(output),
                "--out-npz", str(output.with_suffix(".npz")),
            ]
            proc = subprocess.Popen(
                command, cwd=ROOT, env=ENV, stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active[job] = (proc, stream, {
                "rss_bytes": 0, "pss_bytes": 0, "vms_bytes": 0,
            })
        _status(f"RUNNING_{stage}", stage=stage, total=len(jobs),
                complete=len(complete), running=len(active), pending=len(pending),
                maximum_workers=maximum_workers,
                memory_available_gib=base.available_gib(),
                disk_free_gib=shutil.disk_usage(OUT).free / 1024 ** 3,
                active=[{"candidate_id": job[0], "topology_seed": job[1],
                         "dynamics_seed": job[2], "pid": handle[0].pid}
                        for job, handle in active.items()])
        if pending or active:
            time.sleep(10)
    return complete


def _audit_g0(jobs):
    folder, _, _, _ = _prepare_execution()
    rows = []
    for job in jobs:
        stem = _stem(*job)
        path = folder / "workers" / f"{stem}.json"
        record = base.read(path)
        observation_json, observation_npz, observation = repaired_observation(path)
        arrays_path = Path(record["arrays"]["path"])
        audit = record["multidimensional_parameter_audit"]
        ellipse = record["mechanism_freeze"]["ellipse_audit"]
        with np.load(arrays_path) as arrays:
            checks = {
                "requested_parameters_effective": audit["requested"] == audit["effective"],
                "native_field_present": (
                    "sheet_activity_counts" in arrays.files
                    and arrays["sheet_activity_counts"].ndim == 3
                ),
                "raw_activity_and_contact_envelope_finite": bool(
                    np.isfinite(arrays["active_fraction"]).all()
                    and np.isfinite(arrays["contact_envelope"]).all()
                ),
                "frozen_repaired_readout": (
                    observation["status"] == "FROZEN_REPAIRED_OBSERVATION_COMPLETE"
                    and not observation["worker_lineage_onsets_used_for_training"]
                    and base.sha(observation_npz)
                    == observation["observation_arrays_sha256"]
                ),
                "seed_identity_recorded": (
                    int(arrays["topology_seed"]) == job[1]
                    and int(arrays["dynamics_seed"]) == job[2]
                ),
                "static_threshold_arrays_saved": all(
                    key in arrays.files for key in ("h", "delta_vtheta", "vtheta")
                ),
            }
        full_duration = np.isclose(record["simulation"]["actual_duration_ms"], 24000.0)
        known_runaway = record["physical_status"] == "RUNAWAY"
        checks.update({
            "duration_or_explicit_runaway": bool(full_duration or known_runaway),
            "execution_and_physical_status_separate": (
                record["execution_status"] == "COMPLETE"
                and record["physical_status"] in {
                    "RUNAWAY", "COMPLETE_NO_RUNAWAY_BY_EXISTING_GATE",
                }
            ),
            "EE_input_conservation_audited": (
                ellipse["maximum_abs_incoming_EE_error"] <= 1e-9
            ),
            "EE_weighted_geometry_before_after_present": all(
                key in ellipse for key in (
                    "weighted_geometry_before", "weighted_geometry_after",
                )
            ),
            "GABA_not_dose_matched_disclosed": "not dose-matched" in audit["kinetics_contract"],
        })
        resource = base.read(folder / "resource_logs" / f"{stem}_summary.json")
        rows.append({
            "candidate_id": job[0], "topology_seed": job[1],
            "dynamics_seed": job[2], "checks": checks,
            "pass": all(checks.values()),
            "actual_duration_ms": record["simulation"]["actual_duration_ms"],
            "wall_seconds": record["simulation"]["wall_seconds"],
            "physical_status": record["physical_status"],
            "execution_status": record["execution_status"],
            "peak_process_tree": resource["peak_process_tree"],
            "worker_path": str(path), "worker_sha256": base.sha(path),
            "arrays_sha256": record["arrays"]["sha256"],
            "repaired_observation_path": str(observation_json),
            "repaired_observation_sha256": base.sha(observation_json),
        })
    result = {
        "status": "G0_CANARY_PASS" if all(row["pass"] for row in rows)
        else "G0_CANARY_FAILED",
        "canaries": rows,
        "historical_baseline_completed_full_24s": bool(
            rows[0]["actual_duration_ms"] == 24000.0
            and rows[0]["physical_status"] != "RUNAWAY"
        ),
        "G0_does_not_require_fit_improvement_or_two_modes": True,
    }
    result["pass"] = bool(
        all(row["pass"] for row in rows)
        and result["historical_baseline_completed_full_24s"]
    )
    base.write(OUT / "g0_canary_audit.json", result)
    if not result["pass"]:
        raise RuntimeError("G0 canary contract failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-workers", type=int, default=6)
    parser.add_argument("--g0-workers", type=int, default=3)
    parser.add_argument("--through-g1", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.maximum_workers <= 8 or not 1 <= args.g0_workers <= 3:
        raise ValueError("worker limits are G0 1..3 and G1 1..8")
    OUT.mkdir(parents=True, exist_ok=True)
    guard = open(OUT / "controller.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder, config_path, manifest_path, snapshot_path = _prepare_execution()
    if args.prepare_only:
        print(json.dumps({
            "status": "G0_G1_EXECUTION_PREPARED",
            "config": str(config_path), "config_sha256": base.sha(config_path),
            "manifest": str(manifest_path), "manifest_sha256": base.sha(manifest_path),
            "runtime_snapshot": str(snapshot_path),
        }, indent=2))
        return
    g0_jobs = [(candidate_id, 2511, 2511) for candidate_id in G0_IDS]
    _run_jobs("G0", g0_jobs, maximum_workers=args.g0_workers)
    g0 = _audit_g0(g0_jobs)
    if not args.through_g1:
        _status("G0_COMPLETE_G1_NOT_STARTED", g0_pass=g0["pass"])
        return
    candidates = base.read(manifest_path)["candidates"]
    g1_jobs = [
        (row["candidate_id"], seed, seed)
        for row in candidates for seed in (2511, 2512)
    ]
    complete = _run_jobs("G1", g1_jobs, maximum_workers=args.maximum_workers)
    base.write(OUT / "g1_physical_completion.json", {
        "status": "G1_48_BY_2_PHYSICAL_TRAJECTORIES_COMPLETE",
        "jobs": len(g1_jobs), "completed": len(complete),
        "g0_reused_jobs": len(g0_jobs),
        "normal_low_event_and_runaway_retained": True,
        "scoring_pending": True,
    })
    _status("G1_PHYSICAL_COMPLETE_SCORING_PENDING", completed=len(complete))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _status("ERROR", error=repr(exc))
        raise
