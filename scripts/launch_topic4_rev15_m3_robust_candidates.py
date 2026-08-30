#!/usr/bin/env python3
"""Launch the frozen rev15 robust-candidate controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import monitor_topic4_rev15_m3_robust_candidates as monitor


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
NUMERIC_ENV = (
    "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _command(
    *, config_path: Path, expected_commit: str, artifact_root: Path,
    unit_prefix: str, worker_cap: int, log_path: Path,
) -> tuple[str, list[str]]:
    prefix = monitor._validate_unit_prefix(unit_prefix)
    unit = f"{prefix}-controller-{expected_commit[:8]}"
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        *[f"--setenv={name}=1" for name in NUMERIC_ENV],
        "/usr/bin/nohup", str(PYTHON), str(Path(monitor.__file__).resolve()),
        "--config", str(config_path), "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root), "--unit-prefix", prefix,
        "--worker-cap", str(worker_cap), "--execute",
    ]
    return unit, command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=monitor.DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=9)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    monitor.configure_base()
    monitor.base._load_contract = monitor._load_contract
    expected_commit = monitor.base.base._resolve_commit(args.expected_commit)
    monitor.base.base._require_clean_commit(expected_commit)
    config, manifest = monitor._load_contract(
        args.config.resolve(), args.artifact_root.resolve(), expected_commit,
    )
    if not 1 <= args.worker_cap <= int(config["resources"]["maximum_workers"]):
        raise ValueError("worker cap exceeds rev15 robust-candidate limit")
    output_root = args.artifact_root.resolve() / config["output_root"]
    log_path = output_root / "run_logs" / "controller.log"
    unit, command = _command(
        config_path=args.config.resolve(), expected_commit=expected_commit,
        artifact_root=args.artifact_root.resolve(),
        unit_prefix=args.unit_prefix, worker_cap=args.worker_cap,
        log_path=log_path,
    )
    n_jobs = len(manifest["candidates"]) * len(
        config["search"]["active_network_seeds"]
    )
    payload = {
        "status": "REV15_M3_ROBUST_CONTROLLER_LAUNCHED" if args.execute
        else "REV15_M3_ROBUST_CONTROLLER_DRY_RUN",
        "unit": unit, "expected_git_commit": expected_commit,
        "worker_cap": args.worker_cap, "n_jobs": n_jobs,
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
