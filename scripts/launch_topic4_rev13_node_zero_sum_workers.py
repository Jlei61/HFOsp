#!/usr/bin/env python3
"""Launch the persistent rev13 queue monitor through systemd and nohup."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MONITOR = ROOT / "scripts/monitor_topic4_rev13_node_zero_sum_workers.py"
NUMERIC_ENV = (
    "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _unit_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _phase_monitor_log_path(output_root: Path, phase: str) -> Path:
    namespace = "sentinel" if phase == "sentinel" else "workers"
    return output_root / "run_logs" / namespace / f"{phase}_monitor.log"


def _monitor_command(
    *, config_path: Path, phase: str, expected_commit: str,
    artifact_root: Path, unit_prefix: str, log_path: Path,
) -> tuple[str, list[str]]:
    unit = (
        f"{_unit_token(unit_prefix)}-monitor-{_unit_token(phase)}-"
        f"{expected_commit[:8]}"
    )
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}", "--quiet",
        "--property=Type=exec", f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        *[f"--setenv={name}=1" for name in NUMERIC_ENV],
        "/usr/bin/nohup", str(PYTHON), str(MONITOR),
        "--config", str(config_path), "--phase", phase,
        "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root),
        "--unit-prefix", unit_prefix,
    ]
    return unit, command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("sentinel", "canary", "fit"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default="codex-t4-r13-node")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    if int(config["resources"]["monitor_interval_seconds"]) != 600:
        raise RuntimeError("rev13 monitor interval drifted from 600 seconds")
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    output_root = artifact_root / config["output_root"]
    log_path = _phase_monitor_log_path(output_root, args.phase)
    status_path = output_root / "status" / f"{args.phase}_controller.json"
    launch_path = output_root / "status" / f"{args.phase}_launch.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    unit, command = _monitor_command(
        config_path=config_path, phase=args.phase,
        expected_commit=expected_commit, artifact_root=artifact_root,
        unit_prefix=args.unit_prefix, log_path=log_path,
    )
    payload = {
        "status": "REV13_NODE_ZERO_SUM_MONITOR_LAUNCH_PLANNED",
        "phase": args.phase,
        "expected_git_commit": expected_commit,
        "unit": unit,
        "log": str(log_path),
        "status_path": str(status_path),
        "monitor_interval_seconds": 600,
        "launcher": "systemd-run --user plus nohup",
        "launched_at_epoch": time.time(),
    }
    _atomic_json(launch_path, payload)
    if not args.dry_run:
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except Exception:
            payload["status"] = "REV13_NODE_ZERO_SUM_MONITOR_LAUNCH_FAILED"
            _atomic_json(launch_path, payload)
            raise
        payload["status"] = "REV13_NODE_ZERO_SUM_MONITOR_LAUNCHED"
        _atomic_json(launch_path, payload)
    print(json.dumps({**payload, "command": command if args.dry_run else None}, indent=2))


if __name__ == "__main__":
    main()
