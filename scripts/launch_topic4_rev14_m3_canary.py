#!/usr/bin/env python3
"""Launch the rev14 M3 controller through systemd and nohup.

This command is a dry run by default.  ``--execute`` is required before it can
start the controller; the controller separately requires ``--execute`` before
it can start any SNN worker.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MONITOR = ROOT / "scripts/monitor_topic4_rev14_m3_canary.py"
NUMERIC_ENV = (
    "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _monitor_command(
    *, config_path: Path, expected_commit: str, artifact_root: Path,
    unit_prefix: str, worker_cap: int, log_path: Path,
) -> tuple[str, list[str]]:
    from scripts import monitor_topic4_rev14_m3_canary as monitor

    prefix = monitor._validate_unit_prefix(unit_prefix)
    unit = f"{prefix}-controller-{expected_commit[:8]}"
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        *[f"--setenv={name}=1" for name in NUMERIC_ENV],
        "/usr/bin/nohup", str(PYTHON), str(MONITOR),
        "--config", str(config_path),
        "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root),
        "--unit-prefix", prefix,
        "--worker-cap", str(worker_cap),
        "--execute",
    ]
    return unit, command


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default="codex-t4-r14-m3")
    parser.add_argument("--worker-cap", type=int, default=9)
    parser.add_argument(
        "--execute", action="store_true",
        help="start the controller; otherwise print the exact launch command",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from scripts import monitor_topic4_rev14_m3_canary as monitor

    args = _parse_args(argv)
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    expected_commit = monitor._resolve_commit(args.expected_commit)
    monitor._validate_unit_prefix(args.unit_prefix)
    monitor._require_clean_commit(expected_commit)
    config, _ = monitor._load_contract(
        config_path, artifact_root, expected_commit,
    )
    hard_cap = int(config["resources"]["maximum_workers"])
    if not 1 <= int(args.worker_cap) <= hard_cap:
        raise ValueError("worker-cap exceeds the frozen rev14 M3 hard cap")
    output_root = artifact_root / str(config["output_root"])
    log_path = output_root / "run_logs" / "controller.log"
    unit, command = _monitor_command(
        config_path=config_path, expected_commit=expected_commit,
        artifact_root=artifact_root, unit_prefix=args.unit_prefix,
        worker_cap=int(args.worker_cap), log_path=log_path,
    )
    payload = {
        "status": (
            "REV14_M3_CONTROLLER_LAUNCHED"
            if args.execute else "REV14_M3_CONTROLLER_DRY_RUN"
        ),
        "unit": unit,
        "expected_git_commit": expected_commit,
        "worker_cap": int(args.worker_cap),
        "hard_cap": hard_cap,
        "recommended_cap": int(
            config["resources"]["recommended_workers_after_full_prewarm"]
        ),
        "monitor_interval_seconds": int(
            config["resources"]["monitor_interval_seconds"]
        ),
        "log": str(log_path),
        "command": command,
    }
    if args.execute:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
