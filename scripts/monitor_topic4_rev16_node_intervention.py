#!/usr/bin/env python3
"""Resource-safe controller for the three rev16 hotspot workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev15_node_intervention as validation  # noqa: E402
from scripts import aggregate_topic4_rev16_node_intervention as aggregate  # noqa: E402
from scripts import run_topic4_rev16_node_intervention_worker as worker  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
WORKER = ROOT / "scripts/run_topic4_rev16_node_intervention_worker.py"
MANAGED = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
DEFAULT_UNIT_PREFIX = "codex-t4-r16-node-intervention"
CONTROLLER_SCHEMA = "topic4_rev16_node_intervention_controller_v1"
COMPLETE = "REV16_NODE_INTERVENTION_QUEUE_COMPLETE"
FAILED = "REV16_NODE_INTERVENTION_QUEUE_FAILED"
RUNNING = "REV16_NODE_INTERVENTION_QUEUE_RUNNING"
RESOURCE_WAIT = "REV16_NODE_INTERVENTION_RESOURCE_WAIT"
NUMERIC_ENV = (
    "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _commit(value: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", value], cwd=ROOT, text=True,
    ).strip()


def _require_clean_commit(expected: str, config_path: Path) -> None:
    if _commit("HEAD") != expected:
        raise RuntimeError("intervention controller is on the wrong commit")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    if status:
        raise RuntimeError("intervention controller worktree is dirty")
    relative = config_path.resolve().relative_to(ROOT)
    committed = subprocess.check_output(
        ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
    )
    if hashlib.sha256(committed).hexdigest() != _sha256(config_path):
        raise RuntimeError("intervention config does not match the expected commit")


def _resolve(record: Mapping[str, Any], artifact_root: Path) -> Path:
    for root in (artifact_root, ROOT):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"intervention controller input changed: {record['path']}")


def load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> dict[str, Any]:
    _require_clean_commit(expected_commit, config_path)
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != worker.EXPECTED_CONFIG_SCHEMA:
        raise RuntimeError("rev16 intervention config schema changed")
    if config.get("network_seeds") != [2351, 2352, 2353]:
        raise RuntimeError("rev16 intervention network pool changed")
    if config.get("mechanism_freeze") != {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }:
        raise RuntimeError("rev16 intervention activated another mechanism")
    expected_resources = {
        "maximum_workers": 3,
        "numerical_threads_per_worker": 1,
        "safe_peak_rss_gib_per_worker": 16.0,
        "stop_launching_below_available_memory_gib": 80.0,
        "minimum_free_disk_gib": 40.0,
        "monitor_interval_seconds": 600,
        "long_run_launcher": "systemd-run --user plus nohup",
    }
    if config.get("resources") != expected_resources:
        raise RuntimeError("rev16 intervention resource contract changed")
    final_audit = json.loads(
        _resolve(config["inputs"]["final_science_audit"], artifact_root).read_text()
    )
    if final_audit.get("status") != "NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION":
        raise RuntimeError("rev16 final audit does not permit intervention")
    decision = final_audit.get("decision", {})
    if decision.get("accepted_for_same_checkpoint_intervention") is not True:
        raise RuntimeError("rev16 final-audit decision is incomplete")
    if decision.get("node_freeze_permitted") is not False:
        raise RuntimeError("rev16 Node was frozen before intervention")
    return config


def _token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()
    if not token or not token.startswith(DEFAULT_UNIT_PREFIX):
        raise ValueError("invalid rev16 intervention unit prefix")
    return token


def _unit(prefix: str, seed: int, commit: str) -> str:
    return f"{prefix}-s{int(seed)}-{commit[:8]}"


def _is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"{unit}.service"],
        check=False,
    )
    return result.returncode == 0


def _validate_worker(
    path: Path, *, seed: int, candidate_id: str, config_sha256: str,
) -> None:
    previous = validation.WORKER_STATUS
    validation.WORKER_STATUS = worker.WORKER_STATUS
    try:
        validation._validate_worker(
            path, seed=seed, candidate_id=candidate_id,
            config_sha256=config_sha256,
        )
    finally:
        validation.WORKER_STATUS = previous


def classify(
    *, config: Mapping[str, Any], config_path: Path,
    artifact_root: Path, expected_commit: str, unit_prefix: str,
) -> dict[str, Any]:
    output = artifact_root / str(config["output_root"])
    config_hash = _sha256(config_path)
    states, invalid = {}, []
    for seed in config["network_seeds"]:
        path = output / "workers" / f"intervention_seed_{int(seed)}.json"
        if path.is_file():
            try:
                _validate_worker(
                    path, seed=int(seed), candidate_id=str(config["candidate_id"]),
                    config_sha256=config_hash,
                )
                states[str(seed)] = "complete"
            except Exception as error:
                states[str(seed)] = "invalid"
                invalid.append(f"{seed}:{error}")
        elif _is_active(_unit(unit_prefix, int(seed), expected_commit)):
            states[str(seed)] = "active"
        else:
            status_path = output / "run_logs/workers" / f"intervention_seed_{int(seed)}.status"
            status = status_path.read_text().strip() if status_path.is_file() else ""
            if status.startswith("FAILED"):
                states[str(seed)] = "failed"
            elif status.startswith("SUCCESS") or status.startswith("RUNNING"):
                states[str(seed)] = "invalid"
                invalid.append(f"{seed}:stale status without a valid worker artifact")
            else:
                states[str(seed)] = "pending"
    return {"states": states, "invalid": invalid}


def _launch(
    *, config_path: Path, config: Mapping[str, Any], seed: int,
    artifact_root: Path, expected_commit: str, unit_prefix: str,
) -> str:
    output = artifact_root / str(config["output_root"])
    status = output / "run_logs/workers" / f"intervention_seed_{seed}.status"
    log = output / "run_logs/workers" / f"intervention_seed_{seed}.log"
    out_json = output / "workers" / f"intervention_seed_{seed}.json"
    out_npz = output / "workers" / f"intervention_seed_{seed}.npz"
    unit = _unit(unit_prefix, seed, expected_commit)
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        *[f"--setenv={name}=1" for name in NUMERIC_ENV],
        "/usr/bin/nohup", str(MANAGED), str(status), str(log),
        f"rev16 Node intervention seed={seed}", expected_commit,
        "/usr/bin/time", "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path), "--seed", str(seed),
        "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root),
        "--out-json", str(out_json), "--out-npz", str(out_npz),
    ]
    status.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def tick(
    *, config_path: Path, artifact_root: Path, expected_commit: str,
    unit_prefix: str, worker_cap: int, execute: bool,
) -> dict[str, Any]:
    config = load_contract(config_path, artifact_root, expected_commit)
    prefix = _token(unit_prefix)
    if not 1 <= worker_cap <= int(config["resources"]["maximum_workers"]):
        raise ValueError("rev16 intervention worker cap changed")
    snapshot = classify(
        config=config, config_path=config_path, artifact_root=artifact_root,
        expected_commit=expected_commit, unit_prefix=prefix,
    )
    states = snapshot["states"]
    failed = [seed for seed, state in states.items() if state in {"failed", "invalid"}]
    launched = []
    if failed:
        status = FAILED
    elif all(state == "complete" for state in states.values()):
        result = aggregate.aggregate(
            config_path=config_path, artifact_root=artifact_root,
        )
        status = COMPLETE
        subprocess.run([
            "notify-send", "Topic 4 rev16",
            f"Node intervention complete: {result['status']}",
        ], check=False)
    else:
        memory = psutil.virtual_memory().available / 2**30
        disk = shutil.disk_usage(artifact_root).free / 2**30
        active = sum(state == "active" for state in states.values())
        pending = [int(seed) for seed, state in states.items() if state == "pending"]
        reserve = float(
            config["resources"]["stop_launching_below_available_memory_gib"]
        )
        per_worker = float(config["resources"]["safe_peak_rss_gib_per_worker"])
        memory_slots = max(0, int((memory - reserve) // per_worker))
        resource_ok = (
            memory_slots > 0
            and disk >= float(config["resources"]["minimum_free_disk_gib"])
        )
        if execute and resource_ok:
            slots = min(max(0, worker_cap - active), memory_slots)
            for seed in pending[:slots]:
                launched.append(_launch(
                    config_path=config_path, config=config, seed=seed,
                    artifact_root=artifact_root, expected_commit=expected_commit,
                    unit_prefix=prefix,
                ))
                states[str(seed)] = "active"
        status = RUNNING if resource_ok else RESOURCE_WAIT
    output = artifact_root / str(config["output_root"])
    payload = {
        "schema_id": CONTROLLER_SCHEMA, "status": status,
        "expected_git_commit": expected_commit,
        "n_complete": sum(state == "complete" for state in states.values()),
        "n_active": sum(state == "active" for state in states.values()),
        "n_pending": sum(state == "pending" for state in states.values()),
        "failed_or_invalid": failed, "invalid_detail": snapshot["invalid"],
        "states": states, "launched_this_cycle": launched,
        "available_memory_gib": psutil.virtual_memory().available / 2**30,
        "free_disk_gib": shutil.disk_usage(artifact_root).free / 2**30,
        "updated_at_epoch": time.time(),
    }
    _atomic_json(output / "status/intervention_controller.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    expected = _commit(args.expected_commit)
    while True:
        payload = tick(
            config_path=config_path, artifact_root=artifact_root,
            expected_commit=expected, unit_prefix=args.unit_prefix,
            worker_cap=args.worker_cap, execute=bool(args.execute),
        )
        print(json.dumps(payload), flush=True)
        if payload["status"] in {COMPLETE, FAILED} or not args.execute:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
