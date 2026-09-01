"""Launch the complete ZM1 controller through systemd-run and nohup."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
CONTROLLER = ROOT / "scripts/run_topic4_rev10_zm1_controller.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--unit", default="topic4-rev10-zm1-controller")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    run_logs = output_root / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    status = run_logs / "controller.status"
    log = run_logs / "controller.log"
    command = [
        "systemd-run", "--user", "--collect", f"--unit={args.unit}",
        "--property=Type=exec", "--property=MemoryMax=4G",
        "--property=MemoryHigh=3G", f"--working-directory={ROOT}",
        "/usr/bin/nohup", str(MANAGER), str(status), str(log),
        "rev10-ZM1 h+Z+M controller", args.commit[:8],
        str(PYTHON), str(CONTROLLER), "--config", str(config_path),
        "--commit", args.commit,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({
        "status": "REV10ZM1_CONTROLLER_LAUNCHED",
        "unit": args.unit,
        "controller_status": str(status),
        "controller_log": str(log),
        "worker_launcher": "systemd-run --user -> nohup",
    }, indent=2))


if __name__ == "__main__":
    main()
