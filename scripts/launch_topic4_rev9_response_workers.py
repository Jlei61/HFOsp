"""Launch independent rev9 response workers through systemd and nohup."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9_managed_command.sh"
PRODUCER = ROOT / "scripts/run_topic4_rev9_node_kick_canary.py"


def _alpha_tag(alpha):
    return f"{float(alpha):g}".replace("-", "m").replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("Node", "Edge"), required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--unit-prefix", default="topic4-rev9-response")
    args = parser.parse_args()

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")
    if args.arm == "Node" and args.alphas != [0.0]:
        raise ValueError("Node workers require the default alpha=0")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launched, skipped = [], []
    for alpha in args.alphas:
        for seed in args.seeds:
            tag = f"{args.arm.lower()}_alpha_{_alpha_tag(alpha)}_seed_{seed}"
            status = output_dir / f"{tag}.status"
            log = output_dir / f"{tag}.log"
            output_json = output_dir / f"{tag}.json"
            output_npz = output_dir / f"{tag}.npz"
            if status.exists() and status.read_text().startswith(("RUNNING", "SUCCESS")):
                skipped.append(tag)
                continue
            unit = f"{args.unit_prefix}-{args.arm.lower()}-a{_alpha_tag(alpha)}-s{seed}-{commit[:8]}"
            title = f"{args.arm} alpha={alpha:g} seed={seed}"
            command = [
                "systemd-run", "--user", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(status), str(log), title,
                commit[:8], str(PYTHON), str(PRODUCER), "--arm", args.arm,
                "--alpha", str(alpha), "--seeds", str(seed),
                "--out-json", str(output_json), "--out-npz", str(output_npz),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            launched.append(dict(tag=tag, unit=unit, status=str(status)))
    print(json.dumps(dict(
        status="REV9_RESPONSE_WORKERS_LAUNCHED", arm=args.arm,
        commit=commit, launched=launched, skipped=skipped), indent=2))


if __name__ == "__main__":
    main()
