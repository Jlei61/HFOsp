#!/usr/bin/env python3
"""Resource-bounded rev13 worker queue managed as a user service."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
TIME = Path("/usr/bin/time")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev13_node_zero_sum_worker.py"
COMPLETE_STATUS = "REV13_NODE_ZERO_SUM_WORKER_COMPLETE"
PARITY_SCHEMA = "topic4_rev13_exact_off_artifact_parity_v1"
DECISION_SCHEMA = "topic4_rev13_node_zero_sum_model_internal_decision_v2"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_token(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    text = path.read_text(encoding="utf-8").strip()
    return text.split(maxsplit=1)[0] if text else "EMPTY"


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _free_disk_gib(path: Path) -> float:
    return float(shutil.disk_usage(path).free) / 1024.0**3


def _peak_rss_kib(path: Path) -> int:
    matches = re.findall(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8"),
    )
    if not matches:
        raise RuntimeError(f"sentinel peak RSS is absent from {path}")
    peak = int(matches[-1])
    if peak <= 0:
        raise RuntimeError("sentinel peak RSS must be positive")
    return peak


def _worker_capacity(
    *, available_gib: float, peak_rss_kib: int, reserve_gib: float, cap: int,
) -> int:
    if peak_rss_kib <= 0 or reserve_gib < 0.0 or cap <= 0:
        raise ValueError("invalid worker-capacity contract")
    usable_kib = max(0.0, available_gib - reserve_gib) * 1024.0**2
    return min(int(cap), max(0, int(math.floor(usable_kib / peak_rss_kib))))


def _launch_slots(*, additional_capacity: int, active: int, cap: int) -> int:
    """Combine current MemAvailable capacity with the total worker cap."""
    if additional_capacity < 0 or active < 0 or cap < 1:
        raise ValueError("invalid worker-slot contract")
    return min(max(0, cap - active), additional_capacity)


def _validate_inputs(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN":
        raise RuntimeError("rev13 candidate manifest is not frozen")
    if manifest.get("schema_id") != (
        "topic4_rev13_node_zero_sum_recovery_manifest_v2"
    ):
        raise RuntimeError("rev13 candidate manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev13 candidate manifest uses another config")
    if manifest.get("provenance", {}).get("git_commit") != expected_commit:
        raise RuntimeError("rev13 manifest was frozen at another commit")
    config_ids = [row["arm_id"] for row in config["arms"]]
    manifest_ids = [row["candidate_id"] for row in manifest["candidates"]]
    if config_ids != manifest_ids:
        raise RuntimeError("rev13 config and manifest candidate order differ")
    return config, manifest


def _phase_jobs(
    config: dict, manifest: dict, phase: str, artifact_root: Path,
) -> list[dict[str, Any]]:
    candidate_ids = [str(row["candidate_id"]) for row in manifest["candidates"]]
    if phase == "sentinel":
        candidate_ids = ["exact_off"]
        seeds = [int(config["search"]["canary_network_seeds"][0])]
    elif phase == "canary":
        seeds = [int(seed) for seed in config["search"]["canary_network_seeds"]]
    elif phase == "fit":
        seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    else:
        raise ValueError(f"unknown phase {phase!r}")
    if not seeds:
        raise RuntimeError(f"rev13 {phase} seed pool is empty")
    output_root = artifact_root / config["output_root"]
    worker_dir = (
        output_root / "sentinel_workers"
        if phase == "sentinel" else output_root / "workers"
    )
    log_dir = output_root / "run_logs" / (
        "sentinel" if phase == "sentinel" else "workers"
    )
    jobs = []
    for candidate_id in candidate_ids:
        for seed in seeds:
            stem = f"{candidate_id}_seed_{seed}"
            jobs.append({
                "candidate_id": candidate_id,
                "seed": seed,
                "json": worker_dir / f"{stem}.json",
                "npz": worker_dir / f"{stem}.npz",
                "status": log_dir / f"{stem}.status",
                "log": log_dir / f"{stem}.log",
                "duration_ms": 2000.0 if phase == "sentinel" else None,
                "expected_duration_ms": float(
                    config["search"]["simulation"]["duration_ms"]
                ),
                "engineering_run_kind": (
                    "sentinel" if phase == "sentinel" else None
                ),
            })
    return jobs


def _worker_complete(job: dict[str, Any], expected_commit: str) -> bool:
    if not job["json"].exists() or not job["npz"].exists():
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    arrays = payload.get("arrays", {})
    execution = payload.get("execution_duration", {})
    expected_formal_duration = float(job["expected_duration_ms"])
    requested_duration = job.get("duration_ms")
    expected_actual_duration = (
        expected_formal_duration if requested_duration is None
        else float(requested_duration)
    )
    try:
        arrays_path_matches = (
            Path(arrays.get("path", "")).resolve() == job["npz"].resolve()
        )
        arrays_hash_matches = arrays.get("sha256") == _sha256(job["npz"])
        duration_matches = (
            float(payload.get("simulation", {}).get("duration_ms"))
            == expected_actual_duration
            and float(execution.get("frozen_duration_ms"))
            == expected_formal_duration
            and execution.get("requested_duration_ms") == requested_duration
            and execution.get("engineering_run_kind")
            == job.get("engineering_run_kind")
            and bool(execution.get("duration_override_used"))
            == (requested_duration is not None)
        )
    except (TypeError, ValueError, OSError):
        return False
    return bool(
        payload.get("status") == COMPLETE_STATUS
        and payload.get("candidate_id") == job["candidate_id"]
        and int(payload.get("seed", -1)) == int(job["seed"])
        and provenance.get("expected_git_commit") == expected_commit
        and bool(provenance.get("runtime_modules_match_expected_commit"))
        and not bool(provenance.get("runtime_modules_dirty"))
        and arrays_path_matches and arrays_hash_matches and duration_matches
    )


def _systemd_unit_active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
    ).returncode == 0


def _job_state(
    job: dict[str, Any], expected_commit: str, *, expected_unit: str | None = None,
) -> str:
    if _worker_complete(job, expected_commit):
        return "complete"
    token = _status_token(job["status"])
    if token == "RUNNING":
        if expected_unit is not None and not _systemd_unit_active(expected_unit):
            return "failed"
        return "active"
    if token in {"FAILED", "SUCCESS"}:
        return "failed"
    return "pending"


def _unit_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _worker_unit(
    job: dict[str, Any], *, expected_commit: str, unit_prefix: str,
) -> str:
    return (
        f"{_unit_token(unit_prefix)}-{_unit_token(job['candidate_id'])}-"
        f"s{int(job['seed'])}-{expected_commit[:8]}"
    )


def _worker_command(
    job: dict[str, Any], *, config_path: Path, artifact_root: Path,
    expected_commit: str, unit_prefix: str,
) -> tuple[str, list[str]]:
    unit = _worker_unit(
        job, expected_commit=expected_commit, unit_prefix=unit_prefix,
    )
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}", "--quiet",
        "--property=Type=exec", f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{job['log']}",
        f"--property=StandardError=append:{job['log']}",
        f"--setenv=REV13_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"rev13 {job['candidate_id']} seed={job['seed']}", expected_commit[:8],
        str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path),
        "--candidate-id", str(job["candidate_id"]),
        "--seed", str(job["seed"]),
        "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root),
        "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
    if job.get("duration_ms") is not None:
        if job.get("engineering_run_kind") != "sentinel":
            raise RuntimeError(
                "sentinel duration override requires engineering_run_kind=sentinel"
            )
        command.extend(["--duration-ms", str(float(job["duration_ms"]))])
    if job.get("engineering_run_kind") is not None:
        command.extend([
            "--engineering-run-kind", str(job["engineering_run_kind"]),
        ])
    return unit, command


def _launch_worker(
    job: dict[str, Any], *, config_path: Path, artifact_root: Path,
    expected_commit: str, unit_prefix: str,
) -> str:
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    unit, command = _worker_command(
        job, config_path=config_path, artifact_root=artifact_root,
        expected_commit=expected_commit, unit_prefix=unit_prefix,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def _notify(title: str, message: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", title, message], check=False)


def _sentinel_audit_path(output_root: Path) -> Path:
    return output_root / "status" / "sentinel_memory_audit.json"


def _parity_audit_path(output_root: Path) -> Path:
    return output_root / "parity" / "exact_off_parity_audit.json"


def _parity_log_path(output_root: Path) -> Path:
    return output_root / "run_logs" / "parity" / "exact_off_seed_2291.log"


def _require_parity_pass(output_root: Path, expected_commit: str) -> dict[str, Any]:
    audit_path = _parity_audit_path(output_root)
    if not audit_path.is_file():
        raise RuntimeError("run and audit full-duration exact-off parity first")
    audit = json.loads(audit_path.read_text())
    if audit.get("schema_id") != PARITY_SCHEMA or audit.get("status") != "PASS":
        raise RuntimeError("rev13 exact-off parity audit did not pass")
    rev13_json = Path(audit.get("inputs", {}).get("rev13_json", {}).get("path", ""))
    if not rev13_json.is_file():
        raise RuntimeError("rev13 parity JSON referenced by the audit is missing")
    payload = json.loads(rev13_json.read_text())
    if payload.get("provenance", {}).get("expected_git_commit") != expected_commit:
        raise RuntimeError("rev13 parity belongs to another commit")
    if _sha256(rev13_json) != audit["inputs"]["rev13_json"]["sha256"]:
        raise RuntimeError("rev13 parity JSON changed after audit")
    return audit


def _require_fit_precondition(output_root: Path, expected_commit: str) -> None:
    controller_path = output_root / "status" / "canary_controller.json"
    decision_path = output_root / "analysis" / "model_internal_decision.json"
    per_run_path = output_root / "aggregate" / "model_internal_per_run.json"
    for path in (controller_path, decision_path, per_run_path):
        if not path.is_file():
            raise RuntimeError("fit requires completed seed2311 aggregation")
    controller = json.loads(controller_path.read_text())
    decision = json.loads(decision_path.read_text())
    per_run = json.loads(per_run_path.read_text()).get("rows", [])
    if (
        controller.get("status") != "REV13_NODE_ZERO_SUM_QUEUE_COMPLETE"
        or controller.get("expected_git_commit") != expected_commit
        or int(controller.get("n_complete", -1)) != 6
    ):
        raise RuntimeError("seed2311 six-arm canary is not complete at this commit")
    if decision.get("schema_id") != DECISION_SCHEMA or not decision.get(
        "canary_complete", False
    ):
        raise RuntimeError("seed2311 model-internal aggregation is incomplete")
    if len(per_run) != 6:
        raise RuntimeError("seed2311 aggregation does not contain all six arms")
    if all(bool(row.get("runaway")) for row in per_run):
        raise RuntimeError("seed2311 is catastrophic: every arm ran away")
    if all(
        int(row.get("n_returned_evaluable_causal_families", 0)) == 0
        for row in per_run
    ):
        raise RuntimeError("seed2311 is catastrophic: every arm has zero causal families")


def _load_peak_rss(output_root: Path) -> int:
    path = _sentinel_audit_path(output_root)
    if not path.exists():
        raise RuntimeError("run phase=sentinel before canary or fit")
    payload = json.loads(path.read_text())
    if payload.get("status") != "REV13_SENTINEL_MEMORY_COMPLETE":
        raise RuntimeError("rev13 sentinel memory audit is incomplete")
    parity_peak = _peak_rss_kib(_parity_log_path(output_root))
    return max(int(payload["peak_rss_kib"]), parity_peak)


def _snapshot(
    *, phase: str, jobs: list[dict[str, Any]], expected_commit: str,
    peak_rss_kib: int | None, resources: dict, started_at: float,
    artifact_root: Path, unit_prefix: str,
) -> dict[str, Any]:
    states = [
        _job_state(
            job, expected_commit,
            expected_unit=_worker_unit(
                job, expected_commit=expected_commit,
                unit_prefix=f"{unit_prefix}-{phase}",
            ),
        )
        for job in jobs
    ]
    available = _available_memory_gib()
    reserve = float(resources["reserved_available_memory_gib"])
    cap = int(resources["maximum_workers"])
    capacity = (
        1 if peak_rss_kib is None
        else _worker_capacity(
            available_gib=available, peak_rss_kib=peak_rss_kib,
            reserve_gib=reserve, cap=cap,
        )
    )
    return {
        "status": "REV13_NODE_ZERO_SUM_QUEUE_RUNNING",
        "phase": phase,
        "expected_git_commit": expected_commit,
        "elapsed_minutes": (time.time() - started_at) / 60.0,
        "n_total": len(jobs),
        "n_complete": states.count("complete"),
        "n_active": states.count("active"),
        "n_pending": states.count("pending"),
        "n_failed": states.count("failed"),
        "peak_rss_kib": peak_rss_kib,
        "available_memory_gib": available,
        "free_disk_gib": _free_disk_gib(artifact_root),
        "selected_worker_capacity": capacity,
        "reserved_available_memory_gib": reserve,
        "maximum_workers": cap,
        "monitor_interval_seconds": int(resources["monitor_interval_seconds"]),
    }


def run_monitor(
    *, config_path: Path, phase: str, expected_commit: str,
    artifact_root: Path, unit_prefix: str, once: bool = False,
) -> dict[str, Any]:
    config, manifest = _validate_inputs(
        config_path, artifact_root, expected_commit
    )
    resources = config["resources"]
    interval = int(resources["monitor_interval_seconds"])
    if interval != 600:
        raise RuntimeError("rev13 monitor interval drifted from 600 seconds")
    output_root = artifact_root / config["output_root"]
    _require_parity_pass(output_root, expected_commit)
    if phase == "fit":
        _require_fit_precondition(output_root, expected_commit)
    status_path = output_root / "status" / f"{phase}_controller.json"
    jobs = _phase_jobs(config, manifest, phase, artifact_root)
    peak_rss_kib = None if phase == "sentinel" else _load_peak_rss(output_root)
    started_at = time.time()

    while True:
        states = [
            _job_state(
                job, expected_commit,
                expected_unit=_worker_unit(
                    job, expected_commit=expected_commit,
                    unit_prefix=f"{unit_prefix}-{phase}",
                ),
            )
            for job in jobs
        ]
        if "failed" in states:
            payload = _snapshot(
                phase=phase, jobs=jobs, expected_commit=expected_commit,
                peak_rss_kib=peak_rss_kib, resources=resources,
                started_at=started_at, artifact_root=artifact_root,
                unit_prefix=unit_prefix,
            )
            payload["status"] = "REV13_NODE_ZERO_SUM_QUEUE_FAILED"
            _atomic_json(status_path, payload)
            _notify("Topic 4 rev13", f"{phase} worker failed")
            raise RuntimeError(f"rev13 {phase} worker exited without valid artifacts")
        if all(state == "complete" for state in states):
            if phase == "sentinel":
                peak_rss_kib = _peak_rss_kib(jobs[0]["log"])
                _atomic_json(_sentinel_audit_path(output_root), {
                    "status": "REV13_SENTINEL_MEMORY_COMPLETE",
                    "candidate_id": jobs[0]["candidate_id"],
                    "seed": jobs[0]["seed"],
                    "peak_rss_kib": peak_rss_kib,
                    "peak_rss_gib": peak_rss_kib / 1024.0**2,
                    "log": str(jobs[0]["log"]),
                    "log_sha256": _sha256(jobs[0]["log"]),
                    "expected_git_commit": expected_commit,
                })
            payload = _snapshot(
                phase=phase, jobs=jobs, expected_commit=expected_commit,
                peak_rss_kib=peak_rss_kib, resources=resources,
                started_at=started_at, artifact_root=artifact_root,
                unit_prefix=unit_prefix,
            )
            payload["status"] = "REV13_NODE_ZERO_SUM_QUEUE_COMPLETE"
            _atomic_json(status_path, payload)
            _notify("Topic 4 rev13", f"{phase} complete: {len(jobs)}/{len(jobs)}")
            print(json.dumps(payload), flush=True)
            return payload

        snapshot = _snapshot(
            phase=phase, jobs=jobs, expected_commit=expected_commit,
            peak_rss_kib=peak_rss_kib, resources=resources,
            started_at=started_at, artifact_root=artifact_root,
            unit_prefix=unit_prefix,
        )
        active = states.count("active")
        pending_indices = [i for i, state in enumerate(states) if state == "pending"]
        disk_floor = float(resources["minimum_free_disk_gib"])
        if snapshot["free_disk_gib"] < disk_floor:
            snapshot["status"] = "REV13_WAITING_FOR_DISK"
        elif snapshot["available_memory_gib"] < float(
            resources["reserved_available_memory_gib"]
        ):
            snapshot["status"] = "REV13_WAITING_FOR_MEMORY"
        else:
            slots = _launch_slots(
                additional_capacity=max(
                    0, int(snapshot["selected_worker_capacity"])
                ),
                active=active,
                cap=int(resources["maximum_workers"]),
            )
            for index in pending_indices[:slots]:
                job = jobs[index]
                _launch_worker(
                    job, config_path=config_path, artifact_root=artifact_root,
                    expected_commit=expected_commit,
                    unit_prefix=f"{unit_prefix}-{phase}",
                )
            snapshot["n_launched_this_cycle"] = min(slots, len(pending_indices))
        _atomic_json(status_path, snapshot)
        print(json.dumps(snapshot), flush=True)
        if once:
            return snapshot
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("sentinel", "canary", "fit"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default="codex-t4-r13-node")
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    run_monitor(
        config_path=args.config.resolve(), phase=args.phase,
        expected_commit=expected_commit, artifact_root=args.artifact_root.resolve(),
        unit_prefix=args.unit_prefix, once=args.once,
    )


if __name__ == "__main__":
    main()
