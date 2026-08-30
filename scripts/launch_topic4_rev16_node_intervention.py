#!/usr/bin/env python3
"""Launch the rev16 hotspot intervention controller with systemd and nohup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import monitor_topic4_rev16_node_intervention as monitor  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=monitor.ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=monitor.DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    expected = monitor._commit(args.expected_commit)
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = monitor.load_contract(config_path, artifact_root, expected)
    if not 1 <= args.worker_cap <= int(config["resources"]["maximum_workers"]):
        raise ValueError("rev16 intervention worker cap changed")
    prefix = monitor._token(args.unit_prefix)
    unit = f"{prefix}-controller-{expected[:8]}"
    log = artifact_root / config["output_root"] / "run_logs/controller.log"
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log}",
        f"--property=StandardError=append:{log}",
        *[f"--setenv={name}=1" for name in monitor.NUMERIC_ENV],
        "/usr/bin/nohup", str(monitor.PYTHON), str(Path(monitor.__file__).resolve()),
        "--config", str(config_path), "--expected-commit", expected,
        "--artifact-root", str(artifact_root), "--unit-prefix", prefix,
        "--worker-cap", str(args.worker_cap), "--execute",
    ]
    payload = {
        "status": "REV16_NODE_INTERVENTION_CONTROLLER_LAUNCHED"
        if args.execute else "REV16_NODE_INTERVENTION_CONTROLLER_DRY_RUN",
        "unit": unit, "expected_git_commit": expected,
        "worker_cap": args.worker_cap, "network_seeds": config["network_seeds"],
        "monitor_interval_seconds": config["resources"]["monitor_interval_seconds"],
        "log": str(log), "command": command,
    }
    if args.execute:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.touch(exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
