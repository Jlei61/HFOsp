#!/usr/bin/env python3
"""Resource-safe background controller for rev22-DCI worker phases.

This module only validates frozen execution contracts and schedules the existing
rev12 Node worker.  It deliberately does not import analysis, validation,
KMeans, OOD, held-out, or ictal code.
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
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
WORKER = ROOT / "scripts/run_topic4_rev12_node_worker.py"
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER_COMPLETE = "REV12ND_NODE_WORKER_COMPLETE"
PHASES = (
    "canary", "fit", "decomposition", "qualification", "confirmation",
    "structural_nulls",
)
FROZEN_SELECTION_PHASES = frozenset({"qualification", "confirmation"})
RESERVE_GIB = 32.0
MAXIMUM_WORKERS = 16
MONITOR_INTERVAL_SECONDS = 600
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_contract_path(value: str | Path, *, artifact_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    local = ROOT / path
    return (local if local.exists() else artifact_root / path).resolve()


def _safe_candidate_id(value: Any) -> str:
    if value is None:
        raise RuntimeError("candidate id is missing")
    candidate_id = str(value)
    if not candidate_id or re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id) is None:
        raise RuntimeError(f"unsafe candidate id: {candidate_id!r}")
    return candidate_id


def _candidate_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("execution candidate manifest has no candidates")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("candidate manifest rows must be objects")
        candidate_id = _safe_candidate_id(row.get("candidate_id"))
        if candidate_id in indexed:
            raise RuntimeError(f"duplicate candidate id: {candidate_id}")
        indexed[candidate_id] = row
    if manifest.get("candidate_count", len(indexed)) != len(indexed):
        raise RuntimeError("candidate manifest count is inconsistent")
    return indexed


def _seed_units(seed_manifest: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    key = {
        "fit": "fit",
        "qualification": "qualification",
        "confirmation": "confirmation",
    }.get(phase)
    if key is None:
        raise ValueError(f"phase has no direct seed block: {phase}")
    rows = seed_manifest.get(key, {}).get("units")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"seed manifest has no {key} units")
    units = []
    for row in rows:
        topology = int(row["topology_seed"])
        dynamics = int(row["dynamics_seed"])
        units.append({
            "topology_seed": topology,
            "dynamics_seed": dynamics,
            "seed_mode": "legacy" if topology == dynamics else "split",
        })
    return units


def _unit_coordinates(row: Mapping[str, Any]) -> list[float]:
    cube = row.get("unit_cube", {})
    names = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
    try:
        return [float(cube[name]) for name in names]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"candidate {row.get('candidate_id')} lacks unit-cube coordinates") from exc


def _nearest(rows: Sequence[Mapping[str, Any]], target: Sequence[float]) -> str:
    if not rows:
        raise RuntimeError("required canary design block is empty")
    target_values = [float(value) for value in target]
    ranked = []
    for row in rows:
        coordinates = _unit_coordinates(row)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(coordinates, target_values)))
        ranked.append((distance, _safe_candidate_id(row["candidate_id"])))
    return min(ranked)[1]


def _canary_candidate_ids(manifest: Mapping[str, Any]) -> list[str]:
    """Choose the Task 6 engineering sentinels from frozen design roles only."""
    rows = list(_candidate_index(manifest).values())
    references = [row for row in rows if bool(row.get("is_reference"))]
    if len(references) != 1:
        raise RuntimeError("canary requires exactly one frozen reference candidate")
    reference = _safe_candidate_id(references[0]["candidate_id"])
    full = [row for row in rows if row.get("block") == "full4d"]
    geometry = [row for row in rows if row.get("block") == "geometry_plane"]
    dose = [row for row in rows if row.get("block") == "dose_plane"]
    if full and geometry:
        interior = _nearest(full, (0.5, 0.5, 0.5, 0.5))
        # A boundary sentinel maximizes distance from the reference geometry.
        geometry_boundary = max(
            (
                math.hypot(_unit_coordinates(row)[2] - 0.5,
                           _unit_coordinates(row)[3] - 0.5),
                _safe_candidate_id(row["candidate_id"]),
            )
            for row in geometry
        )[1]
        joint_dose = _nearest(dose, (0.5, 0.5, 0.5, 0.5))
        selected = [reference, interior, geometry_boundary, joint_dose]
    else:
        # Dose-only fallback: reference plus two distinct interior joint-dose points.
        if len(dose) < 2:
            raise RuntimeError("dose-only canary requires at least two dose-plane points")
        ranked = sorted(
            (
                math.hypot(_unit_coordinates(row)[0] - 0.5,
                           _unit_coordinates(row)[1] - 0.5),
                _safe_candidate_id(row["candidate_id"]),
            )
            for row in dose
        )
        selected = [reference, ranked[0][1], ranked[1][1]]
    if len(set(selected)) != len(selected):
        raise RuntimeError("canary design roles did not produce distinct candidates")
    return selected


def _frozen_candidate_ids(
    freeze_path: Path | None,
    candidate_index: Mapping[str, Mapping[str, Any]],
    expected_bindings: Mapping[str, str] | None,
) -> list[str]:
    if freeze_path is None or not freeze_path.is_file():
        raise RuntimeError(
            "qualification/confirmation requires the Task 8 frozen candidate file"
        )
    payload = _read_json(freeze_path, "Task 8 frozen candidate file")
    if payload.get("schema_id") != "topic4_rev22_dci_frozen_candidates_v1":
        raise RuntimeError("Task 8 freeze file has an unexpected schema")
    if expected_bindings is None:
        raise RuntimeError("Task 8 freeze bindings were not supplied")
    for key, expected in expected_bindings.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Task 8 freeze binding mismatch: {key}")
    input_hashes = payload.get("input_hashes") or {}
    for label in ("fit_aggregate", "response_fit"):
        record = input_hashes.get(label) or {}
        path = Path(str(record.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = (ARTIFACT_ROOT / path).resolve()
        if not path.is_file() or record.get("sha256") != _sha256(path):
            raise RuntimeError(f"Task 8 freeze has stale {label} provenance")
    raw = payload.get("candidate_ids", payload.get("candidates"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("Task 8 freeze file has no frozen candidate list")
    candidate_ids = []
    for value in raw:
        if isinstance(value, dict):
            value = value.get("candidate_id")
        candidate_ids.append(_safe_candidate_id(value))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Task 8 freeze file repeats a candidate")
    outside = sorted(set(candidate_ids) - set(candidate_index))
    if outside:
        raise RuntimeError(f"Task 8 candidates are outside execution manifest: {outside}")
    return candidate_ids


def _job(
    phase: str,
    candidate_id: str,
    unit: Mapping[str, Any],
    output_root: Path,
    commit: str,
) -> dict[str, Any]:
    topology = int(unit["topology_seed"])
    dynamics = int(unit["dynamics_seed"])
    stem = f"{candidate_id}_topo_{topology}_dyn_{dynamics}"
    worker_root = output_root / phase / "workers"
    log_root = output_root / phase / "run_logs" / "workers"
    digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:8]
    short = re.sub(r"[^a-zA-Z0-9]+", "-", candidate_id).strip("-").lower()[:16]
    systemd_unit = (
        f"codex-t4-r22dci-{phase}-{short}-{digest}-t{topology}-d{dynamics}-{commit[:8]}"
    )
    return {
        "phase": phase,
        "candidate_id": candidate_id,
        "seed": topology,
        "topology_seed": topology,
        "dynamics_seed": dynamics,
        "seed_mode": "legacy" if topology == dynamics else "split",
        "stem": stem,
        "json": worker_root / f"{stem}.json",
        "npz": worker_root / f"{stem}.npz",
        "status": log_root / f"{stem}.status",
        "log": log_root / f"{stem}.log",
        "unit": systemd_unit,
    }


def expand_jobs(
    phase: str,
    candidate_manifest: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
    output_root: Path,
    commit: str,
    *,
    frozen_candidates_path: Path | None = None,
    frozen_bindings: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Pure phase expansion used by both the controller and unit tests."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    index = _candidate_index(candidate_manifest)
    if phase == "canary":
        candidate_ids = _canary_candidate_ids(candidate_manifest)
        units = [_seed_units(seed_manifest, "fit")[0]]
    elif phase == "fit":
        candidate_ids = list(index)
        units = _seed_units(seed_manifest, "fit")
    elif phase == "decomposition":
        block = seed_manifest.get("variance_decomposition_block", {})
        candidate_ids = [_safe_candidate_id(value) for value in block.get("candidates", [])]
        outside = sorted(set(candidate_ids) - set(index))
        if not candidate_ids or outside:
            raise RuntimeError(f"invalid decomposition candidates: {outside}")
        units = [
            {
                "topology_seed": int(row["topology_seed"]),
                "dynamics_seed": int(row["dynamics_seed"]),
            }
            for row in block.get("units", [])
            if row.get("new_run") is True
        ]
        expected = int(block.get("new_trajectories", -1))
        if len(candidate_ids) * len(units) != expected or expected != 16:
            raise RuntimeError("decomposition manifest does not freeze exactly 16 new runs")
    elif phase == "structural_nulls":
        candidate_ids = list(index)
        units = _seed_units(seed_manifest, "confirmation")[:6]
        if len(candidate_ids) != 18 or len(units) != 6:
            raise RuntimeError(
                "structural-null manifest must freeze 18 candidates on six seeds"
            )
        if len(candidate_ids) * len(units) != 108:
            raise RuntimeError("structural-null phase must contain exactly 108 jobs")
    else:
        candidate_ids = _frozen_candidate_ids(
            frozen_candidates_path, index, frozen_bindings,
        )
        units = _seed_units(seed_manifest, phase)
    jobs = [
        _job(phase, candidate_id, unit, output_root, commit)
        for candidate_id in candidate_ids
        for unit in units
    ]
    keys = [(job["candidate_id"], job["topology_seed"], job["dynamics_seed"]) for job in jobs]
    if len(set(keys)) != len(keys) or len({job["stem"] for job in jobs}) != len(jobs):
        raise RuntimeError("expanded jobs are not unique")
    return jobs


def _git_output(arguments: Sequence[str]) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _path_hash_at_commit(path: Path, commit: str) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT)
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (ValueError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(content).hexdigest()


def validate_contracts(
    config_path: Path,
    candidate_manifest_path: Path,
    seed_manifest_path: Path,
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_json(config_path, "rev22 execution config")
    manifest = _read_json(candidate_manifest_path, "execution candidate manifest")
    seeds = _read_json(seed_manifest_path, "seed manifest")
    if config.get("schema_id") != "topic4_rev22_dci_response_execution_v1":
        raise RuntimeError("unexpected rev22 execution config schema")
    if manifest.get("schema_id") != "topic4_rev22_dci_execution_candidate_manifest_v1":
        raise RuntimeError("unexpected execution candidate manifest schema")
    if seeds.get("schema_id") != "topic4_rev22_dci_seed_manifest_v1":
        raise RuntimeError("unexpected seed manifest schema")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("execution candidate manifest is bound to another config")
    if manifest.get("seed_manifest_sha256") != _sha256(seed_manifest_path):
        raise RuntimeError("execution candidate manifest is bound to another seed manifest")
    if manifest.get("response_design_manifest_sha256") != seeds.get(
        "response_design_manifest_sha256"
    ):
        raise RuntimeError("candidate and seed manifests use different response designs")
    frozen_seed = config.get("frozen_contracts", {}).get("seed_manifest", {})
    if frozen_seed.get("sha256") != _sha256(seed_manifest_path):
        raise RuntimeError("execution config seed-manifest hash drift")
    if _path_hash_at_commit(config_path, expected_commit) != _sha256(config_path):
        raise RuntimeError("execution config differs from expected commit")
    candidate_index = _candidate_index(manifest)
    if any(row.get("mechanisms", {}).get("Z_M") != "off" for row in candidate_index.values()):
        raise RuntimeError("rev22 candidate manifest does not keep Z/M off")
    if len(_seed_units(seeds, "fit")) != 4:
        raise RuntimeError("rev22 fit contract requires four units")
    if len(_seed_units(seeds, "qualification")) != 6:
        raise RuntimeError("rev22 qualification contract requires six units")
    if len(_seed_units(seeds, "confirmation")) != 12:
        raise RuntimeError("rev22 confirmation contract requires twelve units")
    return config, manifest, seeds


def _contracts_unchanged(expected: Mapping[Path, str]) -> bool:
    try:
        return all(path.is_file() and _sha256(path) == digest for path, digest in expected.items())
    except OSError:
        return False


def _artifact_complete(job: Mapping[str, Any], commit: str, config_sha256: str) -> bool:
    json_path, npz_path = Path(job["json"]), Path(job["npz"])
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        arrays = payload.get("arrays", {})
        return bool(
            payload.get("status") == WORKER_COMPLETE
            and payload.get("candidate_id") == job["candidate_id"]
            and int(payload.get("seed", -1)) == int(job["topology_seed"])
            and int(payload.get("topology_seed", -1)) == int(job["topology_seed"])
            and int(payload.get("dynamics_seed", -1)) == int(job["dynamics_seed"])
            and payload.get("seed_mode") == job["seed_mode"]
            and Path(arrays.get("path", "")).resolve() == npz_path.resolve()
            and arrays.get("sha256") == _sha256(npz_path)
            and provenance.get("expected_git_commit") == commit
            and provenance.get("config_sha256") == config_sha256
            and provenance.get("config_sha256_at_expected_commit") == config_sha256
            and bool(provenance.get("runtime_modules_match_expected_commit"))
            and not bool(provenance.get("runtime_modules_dirty"))
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _systemd_active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"{unit}.service"],
        check=False,
    ).returncode == 0


def _job_state(job: Mapping[str, Any], commit: str, config_sha256: str) -> str:
    if _artifact_complete(job, commit, config_sha256):
        return "complete"
    status_path = Path(job["status"])
    status_text = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else ""
    if status_text:
        if f"commit={commit}" not in status_text:
            return "hash_drift"
        if status_text.startswith("FAILED"):
            return "failed"
        if status_text.startswith("SUCCESS"):
            return "invalid_artifact"
        if status_text.startswith("RUNNING"):
            return "running" if _systemd_active(str(job["unit"])) else "orphaned"
        return "invalid_status"
    if Path(job["json"]).exists() or Path(job["npz"]).exists() or Path(job["log"]).exists():
        return "invalid_artifact"
    return "pending"


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _measured_peak_rss_gib(output_root: Path) -> float | None:
    pattern = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    peaks = []
    canary_logs = output_root / "canary" / "run_logs" / "workers"
    if canary_logs.is_dir():
        for path in canary_logs.glob("*.log"):
            matches = pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
            if matches:
                peaks.extend(int(value) / 1024.0**2 for value in matches if int(value) > 0)
    return max(peaks) if peaks else None


def _worker_limit(
    *, available_gib: float, worker_gib: float, configured_cap: int,
    running: int,
) -> int:
    if worker_gib <= 0.0 or configured_cap < 1 or running < 0:
        raise ValueError("invalid resource contract")
    total_cap = min(MAXIMUM_WORKERS, configured_cap)
    memory_slots = max(0, math.floor((available_gib - RESERVE_GIB) / worker_gib))
    return max(0, min(total_cap - running, memory_slots))


def _launch(
    job: Mapping[str, Any], *, config_path: Path, artifact_root: Path, commit: str,
) -> None:
    Path(job["status"]).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", f"--unit={job['unit']}", "--collect",
        f"--working-directory={ROOT}", "--property=OOMPolicy=stop",
    ]
    command.extend(f"--setenv={name}={value}" for name, value in NUMERIC_ENV.items())
    command.extend([
        f"--setenv=REV12ND_SYSTEMD_UNIT={job['unit']}.service",
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        (f"rev22-DCI {job['phase']} {job['candidate_id']} "
         f"topology={job['topology_seed']} dynamics={job['dynamics_seed']}"),
        commit, "/usr/bin/time", "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path),
        "--candidate-id", str(job["candidate_id"]),
        "--seed", str(job["topology_seed"]),
        "--topology-seed", str(job["topology_seed"]),
        "--dynamics-seed", str(job["dynamics_seed"]),
        "--expected-commit", commit,
        "--artifact-root", str(artifact_root),
        "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ])
    subprocess.run(command, cwd=ROOT, check=True)


def _notify(message: str) -> None:
    subprocess.run(["notify-send", "Topic 4 rev22-DCI", message], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--frozen-candidates", type=Path)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--maximum-workers-override", type=int)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    config_path = args.config.resolve()
    preliminary = _read_json(config_path, "rev22 execution config")
    candidate_path = (
        args.candidate_manifest.resolve() if args.candidate_manifest
        else _resolve_contract_path(preliminary["candidate_manifest"], artifact_root=artifact_root)
    )
    seed_record = preliminary.get("frozen_contracts", {}).get("seed_manifest", {})
    seed_path = (
        args.seed_manifest.resolve() if args.seed_manifest
        else _resolve_contract_path(seed_record.get("path", ""), artifact_root=artifact_root)
    )
    commit = _git_output(["rev-parse", args.expected_commit])
    if _git_output(["rev-parse", "HEAD"]) != commit:
        raise RuntimeError("controller HEAD differs from expected commit")
    config, manifest, seeds = validate_contracts(
        config_path, candidate_path, seed_path, commit,
    )
    output_root = artifact_root / config["output_root"]
    frozen_path = args.frozen_candidates
    if args.phase in FROZEN_SELECTION_PHASES and frozen_path is None:
        frozen_path = output_root / "response_fit" / "frozen_candidates.json"
    jobs = expand_jobs(
        args.phase, manifest, seeds, output_root, commit,
        frozen_candidates_path=frozen_path,
        frozen_bindings={
            "execution_candidate_manifest_sha256": _sha256(candidate_path),
            "seed_manifest_sha256": _sha256(seed_path),
            "response_design_manifest_sha256": str(
                manifest["response_design_manifest_sha256"]
            ),
        } if args.phase in FROZEN_SELECTION_PHASES else None,
    )
    resources = config["resources"]
    configured_cap = int(
        args.maximum_workers_override
        if args.maximum_workers_override is not None
        else resources.get("maximum_workers", MAXIMUM_WORKERS)
    )
    if not 1 <= configured_cap <= MAXIMUM_WORKERS:
        raise RuntimeError("worker cap must be between 1 and 16")
    if int(resources.get("monitor_interval_seconds", -1)) != MONITOR_INTERVAL_SECONDS:
        raise RuntimeError("rev22 monitor interval must remain 600 seconds")
    if float(resources.get("reserved_available_memory_gib", 0.0)) < RESERVE_GIB:
        raise RuntimeError("rev22 must reserve at least 32 GiB available RAM")
    minimum_disk = float(resources["minimum_free_disk_gib"])
    estimated_worker_gib = float(resources["estimated_worker_gib"])
    status_path = output_root / args.phase / "status" / "controller.json"
    config_sha = _sha256(config_path)
    contract_hashes = {
        config_path: config_sha,
        candidate_path: _sha256(candidate_path),
        seed_path: _sha256(seed_path),
    }
    if frozen_path is not None:
        contract_hashes[frozen_path.resolve()] = _sha256(frozen_path)

    while True:
        states = [_job_state(job, commit, config_sha) for job in jobs]
        bad_states = {
            "failed", "invalid_artifact", "orphaned", "hash_drift", "invalid_status",
        }
        failed = [
            (job, state) for job, state in zip(jobs, states) if state in bad_states
        ]
        available = _available_memory_gib()
        free_disk = shutil.disk_usage(artifact_root).free / 1024.0**3
        measured = _measured_peak_rss_gib(output_root)
        if args.phase != "canary" and measured is None:
            raise RuntimeError("non-canary phase requires measured canary peak RSS")
        worker_gib = measured if measured is not None else estimated_worker_gib
        counts = {name: states.count(name) for name in sorted(set(states))}
        payload: dict[str, Any] = {
            "schema_id": "topic4_rev22_dci_controller_v1",
            "phase": args.phase,
            "git_commit": commit,
            "config_sha256": config_sha,
            "candidate_manifest_sha256": contract_hashes[candidate_path],
            "seed_manifest_sha256": contract_hashes[seed_path],
            "frozen_candidates_sha256": (
                contract_hashes.get(frozen_path.resolve()) if frozen_path is not None else None
            ),
            "updated_unix": time.time(),
            "job_count": len(jobs),
            "state_counts": counts,
            "available_memory_gib": available,
            "reserved_available_memory_gib": RESERVE_GIB,
            "worker_rss_gib": worker_gib,
            "worker_rss_source": "canary_measured" if measured is not None else "config_estimate",
            "free_disk_gib": free_disk,
            "maximum_workers": min(configured_cap, MAXIMUM_WORKERS),
            "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "status": "RUNNING",
        }
        if not _contracts_unchanged(contract_hashes):
            payload["status"] = "FAILED_CONTRACT_HASH_DRIFT"
            _atomic_json(status_path, payload)
            _notify(f"{args.phase} stopped: contract hash drift")
            raise RuntimeError("rev22-DCI execution contract changed while running")
        if failed:
            payload["status"] = "FAILED"
            payload["failed_jobs"] = [
                {
                    "candidate_id": job["candidate_id"],
                    "topology_seed": job["topology_seed"],
                    "dynamics_seed": job["dynamics_seed"],
                    "state": state,
                }
                for job, state in failed
            ]
            _atomic_json(status_path, payload)
            _notify(f"{args.phase} failed: {len(failed)} job(s)")
            raise RuntimeError("rev22-DCI phase contains failed or stale jobs")
        if all(state == "complete" for state in states):
            payload["status"] = "COMPLETE"
            _atomic_json(status_path, payload)
            _notify(f"{args.phase} complete ({len(jobs)} jobs)")
            return
        if free_disk < minimum_disk:
            payload["status"] = "FAILED_LOW_DISK"
            _atomic_json(status_path, payload)
            _notify(f"{args.phase} stopped: low disk")
            raise RuntimeError("free disk is below the frozen minimum")
        running = states.count("running")
        slots = _worker_limit(
            available_gib=available,
            worker_gib=worker_gib,
            configured_cap=configured_cap,
            running=running,
        )
        if slots == 0 and running == 0:
            payload["status"] = "WAITING_FOR_MEMORY"
        for job, state in zip(jobs, states):
            if slots <= 0:
                break
            if state == "pending":
                _launch(job, config_path=config_path, artifact_root=artifact_root, commit=commit)
                slots -= 1
        _atomic_json(status_path, payload)
        time.sleep(MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
