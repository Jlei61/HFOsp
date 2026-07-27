#!/usr/bin/env python3
"""Monitor the 22-fold x 3-seed formal Claim-2 grid."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2/formal/claim2_runs"
)


def gpu_status() -> list[dict[str, str]]:
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
        index, used, free, utilization = [item.strip() for item in line.split(",")]
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
    cohort = json.loads(
        (
            ROOT
            / "results/topic5_symmetric_axis_propagation_state_v2_2/input_audit/"
            "physical_axis_formal_cohort.json"
        ).read_text(encoding="utf-8")
    )["subjects"]
    states = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "MISSING": 0}
    failures = []
    for subject in cohort:
        for seed in (17, 29, 43):
            path = BASE / subject / f"seed_{seed}/run_state.json"
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
                        "seed": seed,
                        "error": record.get("error"),
                    }
                )
    print(
        json.dumps(
            {
                "expected": len(cohort) * 3,
                "states": states,
                "failures": failures,
                "gpu": gpu_status(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
