"""Run one frozen ZM1.1 tau phase and optionally render confirmation figures."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
FREEZER = ROOT / "scripts/freeze_topic4_rev10_zm1_1_tau_library.py"
LAUNCHER = ROOT / "scripts/launch_topic4_rev10_r_edge_flow_screen.py"
AUDITOR = ROOT / "scripts/audit_topic4_rev10_zm1_1_tau_phase.py"
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
    parser.add_argument("--upstream-decision")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    phase = config["search"]["phase"]
    env = {**os.environ, **NUMERIC_ENV}
    freeze_command = [
        str(PYTHON), str(FREEZER), "--config", str(config_path),
        "--expected-commit", args.commit,
    ]
    if args.upstream_decision:
        freeze_command.extend(["--upstream-decision", args.upstream_decision])
    subprocess.run(freeze_command, cwd=ROOT, check=True, env=env)
    subprocess.run([
        str(PYTHON), str(LAUNCHER), "--config", str(config_path),
        "--commit", args.commit,
        "--unit-prefix", f"topic4-rev10-zm1-1-{phase}",
    ], cwd=ROOT, check=True, env=env)
    subprocess.run([
        str(PYTHON), str(AUDITOR), "--config", str(config_path),
        "--expected-commit", args.commit,
    ], cwd=ROOT, check=True, env=env)
    root = ROOT / config["output_root"]
    if phase == "confirmation":
        subprocess.run([
            str(PYTHON), str(PLOTTER), "--config", str(config_path),
            "--expected-commit", args.commit,
        ], cwd=ROOT, check=True, env=env)
    done = {
        "status": f"REV10ZM1_1_TAU_{phase.upper()}_CONTROLLER_COMPLETE",
        "phase": phase,
        "config": str(config_path.relative_to(ROOT)),
        "commit": args.commit,
        "upstream_decision": args.upstream_decision,
        "figures": sorted(str(path.relative_to(ROOT)) for path in (
            root / "figures"
        ).glob("*.png")) if phase == "confirmation" else [],
    }
    from src.topic4_core_field_runner import atomic_write_json
    atomic_write_json(done, root / "DONE.json")
    subprocess.run([
        "notify-send", f"Topic 4 rev10-ZM1.1 {phase}", done["status"],
    ], check=False)
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
