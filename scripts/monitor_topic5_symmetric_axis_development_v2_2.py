#!/usr/bin/env python3
"""Compact progress/resource monitor for v2.2 development runs."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2/development"
)


def _gpu() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in output.strip().splitlines():
        index, used, free, utilization = [value.strip() for value in line.split(",")]
        rows.append(
            {
                "index": index,
                "memory_used_mb": used,
                "memory_free_mb": free,
                "utilization_percent": utilization,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_kind = "smoke" if args.smoke else "runs"
    expected = 9 if args.smoke else 27
    root = BASE / run_kind
    states = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "MISSING": 0}
    failures = []
    for subject in ("epilepsiae_1077", "epilepsiae_1146", "yuquan_chengshuai"):
        for objective in (
            "next_only",
            "next_plus_rollout_h3",
            "next_plus_rollout_h5",
        ):
            for seed in ((17,) if args.smoke else (17, 29, 43)):
                path = root / subject / objective / f"seed_{seed}" / "run_state.json"
                if not path.is_file():
                    states["MISSING"] += 1
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                status = str(record.get("status", "MISSING"))
                states[status] = states.get(status, 0) + 1
                if status == "FAILED":
                    failures.append(
                        {
                            "subject": subject,
                            "objective": objective,
                            "seed": seed,
                            "error": record.get("error"),
                        }
                    )
    print(
        json.dumps(
            {
                "mode": run_kind,
                "expected": expected,
                "states": states,
                "failures": failures,
                "gpu": _gpu(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
