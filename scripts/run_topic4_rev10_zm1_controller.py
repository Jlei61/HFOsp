"""Run the frozen ZM1 screen, paired audit, and canonical figure producers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from src.topic4_core_field_runner import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
GENERIC_LAUNCHER = ROOT / "scripts/launch_topic4_rev10_r_edge_flow_screen.py"
AUDITOR = ROOT / "scripts/audit_topic4_rev10_zm1_data_driven_h_zm.py"
PLOTTER = ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    env = {**os.environ, **NUMERIC_ENV}
    subprocess.run([
        str(PYTHON), str(GENERIC_LAUNCHER), "--config", str(config_path),
        "--commit", args.commit, "--unit-prefix", "topic4-rev10-zm1",
    ], cwd=ROOT, check=True, env=env)
    subprocess.run([
        str(PYTHON), str(AUDITOR), "--config", str(config_path),
        "--expected-commit", args.commit,
    ], cwd=ROOT, check=True, env=env)
    subprocess.run([
        str(PYTHON), str(PLOTTER), "--config", str(config_path),
        "--expected-commit", args.commit,
    ], cwd=ROOT, check=True, env=env)
    output_root = ROOT / config["output_root"]
    audit = json.loads((output_root / "zm_transfer_audit.json").read_text())
    completion = {
        "status": "REV10ZM1_CONTROLLER_COMPLETE",
        "audit_status": audit["status"],
        "config": str(config_path.relative_to(ROOT)),
        "commit": args.commit,
        "figures": sorted(str(path.relative_to(ROOT)) for path in (
            output_root / "figures"
        ).glob("*.png")),
    }
    atomic_write_json(completion, output_root / "DONE.json")
    subprocess.run([
        "notify-send", "Topic 4 rev10-ZM1",
        f"{audit['status']}; figures and audit complete",
    ], check=False)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
