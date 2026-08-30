#!/usr/bin/env python3
"""Launch rev16 unseen-network Node confirmation with systemd and nohup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_topic4_rev16_joint_m3_m4_candidates as base  # noqa: E402
from scripts import monitor_topic4_rev16_node_confirmation as monitor  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=monitor.DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    monitor.configure_base()
    monitor.base._load_contract = monitor._load_contract
    expected_commit = monitor.base.base._resolve_commit(args.expected_commit)
    monitor.base.base._require_clean_commit(expected_commit)
    config, manifest = monitor._load_contract(
        args.config.resolve(), args.artifact_root.resolve(), expected_commit,
    )
    output_root = args.artifact_root.resolve() / config["output_root"]
    log_path = output_root / "run_logs/controller.log"
    prefix = monitor._validate_unit_prefix(args.unit_prefix)
    unit = f"{prefix}-controller-{expected_commit[:8]}"
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        *[f"--setenv={name}=1" for name in base.NUMERIC_ENV],
        "/usr/bin/nohup", str(base.PYTHON), str(Path(monitor.__file__).resolve()),
        "--config", str(args.config.resolve()), "--expected-commit", expected_commit,
        "--artifact-root", str(args.artifact_root.resolve()),
        "--unit-prefix", prefix, "--worker-cap", str(args.worker_cap), "--execute",
    ]
    payload = {
        "status": "REV16_NODE_CONFIRMATION_CONTROLLER_LAUNCHED"
        if args.execute else "REV16_NODE_CONFIRMATION_CONTROLLER_DRY_RUN",
        "unit": unit, "expected_git_commit": expected_commit,
        "worker_cap": int(args.worker_cap), "n_jobs": 6,
        "monitor_interval_seconds": config["resources"]["monitor_interval_seconds"],
        "log": str(log_path), "command": command,
    }
    if args.execute:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
