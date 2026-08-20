#!/usr/bin/env python3
"""Wait sparsely for the frozen Z/M confirmation batch and aggregate it."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _unit_state(unit: str) -> tuple[str, str]:
    process = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState,Result"],
        check=False, capture_output=True, text=True,
    )
    values = {}
    for line in process.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values.get("ActiveState", "unknown"), values.get("Result", "unknown")


def _notify(title: str, body: str) -> None:
    subprocess.run(["notify-send", title, body], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    expected = {
        seed: root / f"ith080_s{seed}.json" for seed in args.seed
    }
    while True:
        missing = [seed for seed, path in expected.items() if not path.exists()]
        if not missing:
            break
        failures = []
        for seed in missing:
            unit = f"{args.unit_prefix}-s{seed}.service"
            active, result = _unit_state(unit)
            if active in {"failed", "inactive"}:
                failures.append({
                    "seed": seed, "unit": unit,
                    "active_state": active, "result": result,
                })
        if failures:
            payload = {"status": "WORKER_FAILURE", "failures": failures}
            (root / "confirmation_failed.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
            _notify("Topic4 Fig5 Z/M confirmation failed", str(failures))
            raise SystemExit(1)
        time.sleep(float(args.wait_seconds))

    out = root / "confirmation_summary.json"
    subprocess.run([
        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
        str(ROOT / "scripts" / "aggregate_topic4_zm_joint_morphology_confirmation.py"),
        "--root", str(root),
        "--calibration-summary", str(Path(args.calibration_summary).resolve()),
        "--out", str(out),
    ], check=True)
    summary = json.loads(out.read_text())
    _notify(
        "Topic4 Fig5 Z/M confirmation complete",
        f"{summary['n_pass']}/{summary['n_total']} networks passed",
    )


if __name__ == "__main__":
    main()
