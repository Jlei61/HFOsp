"""Launch a ZM1.1 phase controller through systemd-run and nohup."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
CONTROLLER = ROOT / "scripts/run_topic4_rev10_zm1_1_tau_phase_controller.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--upstream-decision")
    parser.add_argument("--unit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    phase = config["search"]["phase"]
    root = ROOT / config["output_root"]
    logs = root / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = logs / "controller.status"
    log = logs / "controller.log"
    unit = args.unit or f"topic4-rev10-zm1-1-{phase}-controller"
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}",
        "--property=Type=exec", "--property=MemoryMax=4G",
        "--property=MemoryHigh=3G", f"--working-directory={ROOT}",
        "/usr/bin/nohup", str(MANAGER), str(status), str(log),
        f"rev10-ZM1.1 tau {phase} controller", args.commit[:8],
        str(PYTHON), str(CONTROLLER), "--config", str(config_path),
        "--commit", args.commit,
    ]
    if args.upstream_decision:
        command.extend(["--upstream-decision", args.upstream_decision])
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({
        "status": "REV10ZM1_1_TAU_PHASE_CONTROLLER_LAUNCHED",
        "phase": phase,
        "unit": unit,
        "controller_status": str(status),
        "controller_log": str(log),
        "worker_launcher": "systemd-run --user -> nohup",
    }, indent=2))


if __name__ == "__main__":
    main()
