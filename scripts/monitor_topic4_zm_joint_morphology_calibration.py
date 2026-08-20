#!/usr/bin/env python3
"""Wait sparsely for a fixed Z/M calibration batch and aggregate it."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _unit_state(unit: str) -> tuple[str, str]:
    proc = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState,Result"],
        check=False,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values.get("ActiveState", "unknown"), values.get("Result", "unknown")


def _notify(title: str, body: str) -> None:
    subprocess.run(["notify-send", title, body], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    expected = {tag: root / f"ith080_{tag}.json" for tag in args.tag}
    while True:
        missing = [tag for tag, path in expected.items() if not path.exists()]
        if not missing:
            break
        failures = []
        for tag in missing:
            unit = f"{args.unit_prefix}-{tag}.service"
            active, result = _unit_state(unit)
            if active in {"failed", "inactive"}:
                failures.append({
                    "tag": tag,
                    "unit": unit,
                    "active_state": active,
                    "result": result,
                })
        if failures:
            payload = {"status": "WORKER_FAILURE", "failures": failures}
            (root / "calibration_failed.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
            _notify("Topic4 Fig5 Z/M calibration failed", str(failures))
            raise SystemExit(1)
        time.sleep(float(args.wait_seconds))

    out = root / "calibration_summary.json"
    subprocess.run(
        [
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
            "scripts/aggregate_topic4_zm_joint_morphology_canary.py",
            "--root", str(root),
            "--out", str(out),
        ],
        check=True,
    )
    _notify(
        "Topic4 Fig5 Z/M calibration complete",
        "Four constant-gain M-timescale canaries aggregated",
    )


if __name__ == "__main__":
    main()
