#!/usr/bin/env python3
"""Resource-bounded controller for the frozen rev14 M3 canary queue.

The first queue item is the full-duration ``uniform_node`` run.  Its measured
RSS, with the frozen safety multiplier, is the only memory estimate used to
launch the remaining jobs.  Direct invocation is a dry run unless ``--execute``
is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
TIME = Path("/usr/bin/time")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev14_m3_canary_worker.py"
COMPLETE_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_WORKER_COMPLETE"
MANIFEST_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_FROZEN"
MANIFEST_SCHEMA = "topic4_rev14_m3_observation_free_canary_manifest_v1"
CONTROLLER_SCHEMA = "topic4_rev14_m3_queue_controller_v1"
DEFAULT_UNIT_PREFIX = "codex-t4-r14-m3"
PROTECTED_UNIT_MARKERS = ("topic4-ps-cohort-v2p1",)
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
OOM_PATTERNS = (
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"out of memory", r"oom[-_ ]kill", r"killed process",
        r"command terminated by signal 9", r"status=9/kill",
    )
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _validate_unit_prefix(prefix: str) -> str:
    token = _unit_token(prefix)
    if not token.startswith(DEFAULT_UNIT_PREFIX):
        raise ValueError(f"unit prefix must start with {DEFAULT_UNIT_PREFIX!r}")
    if any(marker in token for marker in PROTECTED_UNIT_MARKERS):
        raise ValueError("unit prefix overlaps a protected Topic 4 service")
    return token


def _resolve_commit(revision: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", revision], cwd=ROOT, text=True,
    ).strip()


def _require_clean_commit(expected_commit: str) -> None:
    head = _resolve_commit("HEAD")
    if head != expected_commit:
        raise RuntimeError("rev14 controller HEAD differs from expected commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("rev14 controller requires a completely clean worktree")


def _load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resources = config.get("resources", {})
    frozen_resources = {
        "maximum_workers": 10,
        "recommended_workers_after_full_prewarm": 9,
        "stop_launching_below_available_memory_gib": 64,
        "emergency_stop_below_available_memory_gib": 48,
        "worker_rss_safety_multiplier": 1.2,
        "minimum_free_disk_gib": 40,
        "monitor_interval_seconds": 600,
    }
    for key, expected in frozen_resources.items():
        if resources.get(key) != expected:
            raise RuntimeError(f"rev14 M3 resource contract changed: {key}")
    if int(resources.get("numerical_threads_per_worker", -1)) != 1:
        raise RuntimeError("rev14 M3 workers must use one numerical thread")
    seeds = [int(seed) for seed in config["search"]["active_network_seeds"]]
    if seeds != [2321]:
        raise RuntimeError("rev14 M3 active canary seed pool changed")
    if float(config["search"]["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev14 M3 prewarm and workers must run the full 20 s")

    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("rev14 M3 formal candidate manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError("rev14 M3 candidate manifest is not formally frozen")
    if manifest.get("schema_id") != MANIFEST_SCHEMA:
        raise RuntimeError("rev14 M3 candidate manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev14 M3 manifest was frozen from another config")
    provenance = manifest.get("provenance", {})
    if not provenance.get("formal_ready"):
        raise RuntimeError("rev14 M3 manifest provenance is not formal-ready")
    if provenance.get("git_commit") != expected_commit:
        raise RuntimeError("rev14 M3 manifest belongs to another commit")
    if not provenance.get("all_explicit_paths_clean") or not provenance.get(
        "all_explicit_paths_match_expected_commit"
    ):
        raise RuntimeError("rev14 M3 manifest runtime provenance is not clean")
    candidates = manifest.get("candidates", [])
    candidate_ids = [str(row.get("candidate_id")) for row in candidates]
    if len(candidate_ids) != int(config["m3_design"]["candidate_count"]):
        raise RuntimeError("rev14 M3 candidate count changed")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("rev14 M3 candidate IDs are not unique")
    if candidate_ids.count("uniform_node") != 1:
        raise RuntimeError("rev14 M3 full prewarm candidate is missing")
    return config, manifest


def _jobs(
    config: dict[str, Any], manifest: dict[str, Any], artifact_root: Path,
) -> list[dict[str, Any]]:
    output_root = artifact_root / str(config["output_root"])
    manifest_path = artifact_root / str(config["candidate_manifest"])
    manifest_sha256 = _sha256(manifest_path)
    config_sha256 = str(manifest["config_sha256"])
    worker_dir = output_root / "workers"
    log_dir = output_root / "run_logs" / "workers"
    duration_ms = float(config["search"]["simulation"]["duration_ms"])
    seeds = [int(seed) for seed in config["search"]["active_network_seeds"]]
    rows: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        for seed in seeds:
            candidate_id = str(candidate["candidate_id"])
            stem = f"{candidate_id}_seed_{seed}"
            rows.append({
                "candidate_id": candidate_id,
                "seed": seed,
                "selection_eligible": bool(candidate["selection_eligible"]),
                "json": worker_dir / f"{stem}.json",
                "npz": worker_dir / f"{stem}.npz",
                "log": log_dir / f"{stem}.log",
                "status": log_dir / f"{stem}.status",
                "expected_duration_ms": duration_ms,
                "expected_config_sha256": config_sha256,
                "expected_manifest_sha256": manifest_sha256,
            })
    return rows


def _prewarm_job(jobs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    matches = [job for job in jobs if job["candidate_id"] == "uniform_node"]
    if len(matches) != 1:
        raise RuntimeError("rev14 M3 requires exactly one full uniform prewarm")
    return matches[0]


def _status_token(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    text = path.read_text(encoding="utf-8").strip()
    return text.split(maxsplit=1)[0] if text else "EMPTY"


def _worker_complete(job: dict[str, Any], expected_commit: str) -> bool:
    if not job["json"].is_file() or not job["npz"].is_file():
        return False
    try:
        payload = json.loads(job["json"].read_text(encoding="utf-8"))
        arrays = payload.get("arrays", {})
        provenance = payload.get("provenance", {})
        explicit = provenance.get("rev14_explicit_runtime_freeze", {})
        manifest_audit = provenance.get("rev14_manifest_audit", {})
        return bool(
            payload.get("status") == COMPLETE_STATUS
            and payload.get("candidate_id") == job["candidate_id"]
            and int(payload.get("seed", -1)) == int(job["seed"])
            and float(payload.get("simulation", {}).get("duration_ms"))
            == float(job["expected_duration_ms"])
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("git_commit") == expected_commit
            and provenance.get("runtime_modules_match_expected_commit") is True
            and provenance.get("runtime_modules_dirty") is False
            and provenance.get("config_sha256")
            == job["expected_config_sha256"]
            and explicit.get("formal_ready") is True
            and explicit.get("git_commit") == expected_commit
            and manifest_audit.get("manifest_read") is True
            and manifest_audit.get("manifest_sha256")
            == job["expected_manifest_sha256"]
            and Path(arrays.get("path", "")).resolve() == job["npz"].resolve()
            and arrays.get("sha256") == _sha256(job["npz"])
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _full_prewarm_complete(job: dict[str, Any], expected_commit: str) -> bool:
    if not _worker_complete(job, expected_commit):
        return False
    try:
        payload = json.loads(job["json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("simulation", {}).get("runaway_early_stop_ms") is None


def _systemd_unit_active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
    ).returncode == 0


def _worker_unit(
    job: dict[str, Any], *, expected_commit: str, unit_prefix: str,
) -> str:
    prefix = _validate_unit_prefix(unit_prefix)
    return (
        f"{prefix}-{_unit_token(job['candidate_id'])}-"
        f"s{int(job['seed'])}-{expected_commit[:8]}"
    )


def _job_state(
    job: dict[str, Any], expected_commit: str, *, unit_prefix: str,
) -> str:
    if _worker_complete(job, expected_commit):
        return "complete"
    unit = _worker_unit(
        job, expected_commit=expected_commit, unit_prefix=unit_prefix,
    )
    token = _status_token(job["status"])
    if token == "RUNNING":
        return "active" if _systemd_unit_active(unit) else "failed"
    if token in {"FAILED", "SUCCESS"}:
        return "failed"
    if job["json"].exists() or job["npz"].exists():
        return "invalid_artifact"
    return "pending"


def _worker_command(
    job: dict[str, Any], *, config_path: Path, artifact_root: Path,
    expected_commit: str, unit_prefix: str,
) -> tuple[str, list[str]]:
    unit = _worker_unit(
        job, expected_commit=expected_commit, unit_prefix=unit_prefix,
    )
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{job['log']}",
        f"--property=StandardError=append:{job['log']}",
        f"--setenv=REV14_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"rev14 M3 {job['candidate_id']} seed={job['seed']}",
        expected_commit[:8], str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path),
        "--candidate-id", str(job["candidate_id"]),
        "--seed", str(job["seed"]),
        "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root),
        "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
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


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _free_disk_gib(path: Path) -> float:
    return float(shutil.disk_usage(path).free) / 1024.0**3


def _peak_rss_kib(log_path: Path) -> int:
    matches = re.findall(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        log_path.read_text(encoding="utf-8"),
    )
    if not matches or int(matches[-1]) <= 0:
        raise RuntimeError("full prewarm log has no valid peak RSS")
    return int(matches[-1])


def _safe_peak_rss_kib(peak_rss_kib: int, multiplier: float) -> int:
    if peak_rss_kib <= 0 or multiplier < 1.0:
        raise ValueError("invalid RSS safety contract")
    return int(math.ceil(float(peak_rss_kib) * float(multiplier)))


def _worker_capacity(
    *, available_gib: float, safe_peak_rss_kib: int, reserve_gib: float,
    requested_cap: int, recommended_cap: int, hard_cap: int,
) -> int:
    if not 1 <= requested_cap <= hard_cap or recommended_cap > hard_cap:
        raise ValueError("requested worker cap violates the frozen hard cap")
    usable_kib = max(0.0, available_gib - reserve_gib) * 1024.0**2
    memory_cap = int(math.floor(usable_kib / safe_peak_rss_kib))
    return max(0, min(memory_cap, requested_cap, hard_cap))


def _contains_oom(job: dict[str, Any]) -> bool:
    text = ""
    for key in ("status", "log"):
        path = job[key]
        if path.is_file():
            try:
                text += "\n" + path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return any(pattern.search(text) for pattern in OOM_PATTERNS)


def _resource_decision(
    *, available_gib: float, disk_gib: float, oom: bool,
    resources: dict[str, Any],
) -> str:
    if (
        available_gib
        < float(resources["emergency_stop_below_available_memory_gib"])
        or disk_gib < 35.0 or oom
    ):
        return "emergency_stop"
    if (
        available_gib
        < float(resources["stop_launching_below_available_memory_gib"])
        or disk_gib < float(resources["minimum_free_disk_gib"])
    ):
        return "hold"
    return "launch"


def _listed_rev14_units(unit_prefix: str) -> list[str]:
    prefix = _validate_unit_prefix(unit_prefix)
    output = subprocess.check_output(
        [
            "systemctl", "--user", "--no-legend", "--plain", "--all",
            "list-units", f"{prefix}-*.service",
        ],
        text=True,
    )
    units = []
    for line in output.splitlines():
        if not line.strip():
            continue
        unit = line.split()[0].lstrip("●")
        if not unit.startswith(prefix + "-"):
            continue
        if unit.startswith(prefix + "-controller"):
            continue
        if any(marker in unit for marker in PROTECTED_UNIT_MARKERS):
            raise RuntimeError("protected Topic 4 service appeared in rev14 unit list")
        units.append(unit)
    return sorted(set(units))


def _stop_rev14_units(unit_prefix: str) -> list[str]:
    units = _listed_rev14_units(unit_prefix)
    for unit in units:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    return units


def _notify(message: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "Topic 4 rev14 M3", message], check=False,
        )


def _state_counts(
    jobs: list[dict[str, Any]], expected_commit: str, unit_prefix: str,
) -> tuple[dict[str, int], dict[str, str]]:
    states = {
        f"{job['candidate_id']}_seed_{job['seed']}": _job_state(
            job, expected_commit, unit_prefix=unit_prefix,
        )
        for job in jobs
    }
    counts = {
        state: sum(value == state for value in states.values())
        for state in (
            "complete", "active", "pending", "failed", "invalid_artifact",
        )
    }
    return counts, states


def _snapshot(
    *, status: str, jobs: list[dict[str, Any]], expected_commit: str,
    unit_prefix: str, available_gib: float, disk_gib: float,
    peak_rss_kib: int | None, safe_peak_rss_kib: int | None,
    concurrency: int | None, launched: list[str], stopped: list[str],
) -> dict[str, Any]:
    counts, states = _state_counts(jobs, expected_commit, unit_prefix)
    return {
        "schema_id": CONTROLLER_SCHEMA,
        "status": status,
        "expected_git_commit": expected_commit,
        "unit_prefix": _validate_unit_prefix(unit_prefix),
        "monitor_interval_seconds": 600,
        "resources": {
            "available_memory_gib": available_gib,
            "free_disk_gib": disk_gib,
            "full_prewarm_peak_rss_kib": peak_rss_kib,
            "safe_peak_rss_kib": safe_peak_rss_kib,
            "worker_concurrency": concurrency,
        },
        "n_jobs": len(jobs),
        **{f"n_{key}": value for key, value in counts.items()},
        "states": states,
        "launched_this_cycle": launched,
        "stopped_this_cycle": stopped,
        "updated_at_epoch": time.time(),
    }


def run_controller(
    *, config_path: Path, artifact_root: Path, expected_commit: str,
    unit_prefix: str, execute: bool, worker_cap: int,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    expected_commit = _resolve_commit(expected_commit)
    prefix = _validate_unit_prefix(unit_prefix)
    _require_clean_commit(expected_commit)
    config, manifest = _load_contract(config_path, artifact_root, expected_commit)
    jobs = _jobs(config, manifest, artifact_root)
    prewarm = _prewarm_job(jobs)
    resources = config["resources"]
    hard_cap = int(resources["maximum_workers"])
    recommended_cap = int(resources["recommended_workers_after_full_prewarm"])
    if not 1 <= worker_cap <= hard_cap:
        raise ValueError("worker-cap must lie between 1 and the frozen hard cap")
    output_root = artifact_root / str(config["output_root"])
    controller_path = output_root / "status" / "m3_controller.json"

    while True:
        _require_clean_commit(expected_commit)
        _load_contract(config_path, artifact_root, expected_commit)
        available_gib = _available_memory_gib()
        disk_gib = _free_disk_gib(artifact_root / "results")
        counts, states = _state_counts(jobs, expected_commit, prefix)
        failed_jobs = [
            job for job in jobs
            if states[f"{job['candidate_id']}_seed_{job['seed']}"]
            in {"failed", "invalid_artifact"}
        ]
        prewarm_state = states[f"{prewarm['candidate_id']}_seed_{prewarm['seed']}"]
        oom = any(_contains_oom(job) for job in failed_jobs)
        resource_decision = _resource_decision(
            available_gib=available_gib, disk_gib=disk_gib, oom=oom,
            resources=resources,
        )
        launched: list[str] = []
        stopped: list[str] = []
        peak_rss: int | None = None
        safe_peak: int | None = None
        concurrency: int | None = None

        if resource_decision == "emergency_stop":
            stopped = _stop_rev14_units(prefix) if execute else []
            status = "REV14_M3_QUEUE_EMERGENCY_STOPPED"
            snapshot = _snapshot(
                status=status, jobs=jobs, expected_commit=expected_commit,
                unit_prefix=prefix, available_gib=available_gib,
                disk_gib=disk_gib, peak_rss_kib=None,
                safe_peak_rss_kib=None, concurrency=None, launched=launched,
                stopped=stopped,
            )
            if execute:
                _atomic_json(controller_path, snapshot)
                _notify("queue stopped: rev14-only resource/OOM emergency")
            return snapshot

        if prewarm_state == "complete" and not _full_prewarm_complete(
            prewarm, expected_commit,
        ):
            snapshot = _snapshot(
                status="REV14_M3_FULL_PREWARM_INCOMPLETE",
                jobs=jobs, expected_commit=expected_commit,
                unit_prefix=prefix, available_gib=available_gib,
                disk_gib=disk_gib, peak_rss_kib=None,
                safe_peak_rss_kib=None, concurrency=0,
                launched=[], stopped=[],
            )
            if execute:
                _atomic_json(controller_path, snapshot)
                _notify("queue stopped: full 20 s prewarm ended early")
            return snapshot

        if counts["complete"] == len(jobs):
            peak_rss = _peak_rss_kib(prewarm["log"])
            safe_peak = _safe_peak_rss_kib(
                peak_rss, float(resources["worker_rss_safety_multiplier"]),
            )
            status = "REV14_M3_QUEUE_COMPLETE"
            snapshot = _snapshot(
                status=status, jobs=jobs, expected_commit=expected_commit,
                unit_prefix=prefix, available_gib=available_gib,
                disk_gib=disk_gib, peak_rss_kib=peak_rss,
                safe_peak_rss_kib=safe_peak, concurrency=0,
                launched=launched, stopped=stopped,
            )
            if execute:
                _atomic_json(controller_path, snapshot)
                _notify(f"queue complete: {len(jobs)}/{len(jobs)} candidates")
            return snapshot

        if failed_jobs:
            status = (
                "REV14_M3_QUEUE_DRAINING_AFTER_FAILURE"
                if counts["active"] else "REV14_M3_QUEUE_FAILED"
            )
            snapshot = _snapshot(
                status=status, jobs=jobs, expected_commit=expected_commit,
                unit_prefix=prefix, available_gib=available_gib,
                disk_gib=disk_gib, peak_rss_kib=None,
                safe_peak_rss_kib=None, concurrency=0,
                launched=launched, stopped=stopped,
            )
            if execute:
                _atomic_json(controller_path, snapshot)
            if not counts["active"] or not execute:
                if execute:
                    _notify("queue failed closed; no incomplete artifact was overwritten")
                return snapshot
            sleep_fn(float(resources["monitor_interval_seconds"]))
            continue

        soft_block = resource_decision == "hold"
        if prewarm_state != "complete":
            concurrency = 1
            if prewarm_state == "pending" and counts["active"] == 0 and not soft_block:
                unit = _worker_unit(
                    prewarm, expected_commit=expected_commit, unit_prefix=prefix,
                )
                if execute:
                    _launch_worker(
                        prewarm, config_path=config_path,
                        artifact_root=artifact_root,
                        expected_commit=expected_commit, unit_prefix=prefix,
                    )
                launched.append(unit)
            status = (
                "REV14_M3_FULL_PREWARM_RESOURCE_WAIT"
                if soft_block else "REV14_M3_FULL_PREWARM_RUNNING"
            )
        else:
            peak_rss = _peak_rss_kib(prewarm["log"])
            safe_peak = _safe_peak_rss_kib(
                peak_rss, float(resources["worker_rss_safety_multiplier"]),
            )
            concurrency = _worker_capacity(
                available_gib=available_gib,
                safe_peak_rss_kib=safe_peak,
                reserve_gib=float(resources["stop_launching_below_available_memory_gib"]),
                requested_cap=worker_cap,
                recommended_cap=recommended_cap,
                hard_cap=hard_cap,
            )
            slots = max(0, concurrency - counts["active"])
            if not soft_block:
                pending = [
                    job for job in jobs
                    if states[f"{job['candidate_id']}_seed_{job['seed']}"] == "pending"
                ]
                for job in pending[:slots]:
                    unit = _worker_unit(
                        job, expected_commit=expected_commit, unit_prefix=prefix,
                    )
                    if execute:
                        _launch_worker(
                            job, config_path=config_path,
                            artifact_root=artifact_root,
                            expected_commit=expected_commit, unit_prefix=prefix,
                        )
                    launched.append(unit)
            status = (
                "REV14_M3_QUEUE_RESOURCE_WAIT"
                if soft_block or concurrency == 0 else "REV14_M3_QUEUE_RUNNING"
            )

        snapshot = _snapshot(
            status=status, jobs=jobs, expected_commit=expected_commit,
            unit_prefix=prefix, available_gib=available_gib,
            disk_gib=disk_gib, peak_rss_kib=peak_rss,
            safe_peak_rss_kib=safe_peak, concurrency=concurrency,
            launched=launched, stopped=stopped,
        )
        if execute:
            _atomic_json(controller_path, snapshot)
        else:
            snapshot["commands"] = [
                _worker_command(
                    job, config_path=config_path, artifact_root=artifact_root,
                    expected_commit=expected_commit, unit_prefix=prefix,
                )[1]
                for job in jobs
                if _worker_unit(
                    job, expected_commit=expected_commit, unit_prefix=prefix,
                ) in launched
            ]
            return snapshot
        sleep_fn(float(resources["monitor_interval_seconds"]))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=9)
    parser.add_argument(
        "--execute", action="store_true",
        help="start systemd workers; without this flag only print a plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_controller(
        config_path=args.config.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_commit=args.expected_commit,
        unit_prefix=args.unit_prefix,
        execute=bool(args.execute),
        worker_cap=int(args.worker_cap),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
