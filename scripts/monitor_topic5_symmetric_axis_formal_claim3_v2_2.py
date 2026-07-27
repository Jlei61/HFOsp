#!/usr/bin/env python3
"""Monitor the formal 22 x 3 x 8-chunk random-axis Claim-3 grid."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_propagation_state_v2_2/formal/"
    "claim3_random_axis_runs"
)
CHUNKS = tuple((start, start + 32) for start in range(0, 256, 32))


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
    lock = json.loads(
        (
            ROOT
            / "results/topic5_symmetric_axis_propagation_state_v2_2/formal/"
            "PHYSICAL_AXIS_FORMAL_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    subjects = list(map(str, lock["subjects"]))
    seeds = list(map(int, lock["seeds"]))
    states = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "MISSING": 0}
    failures = []
    for subject in subjects:
        for seed in seeds:
            for start, stop in CHUNKS:
                chunk = f"chunk_{start:03d}_{stop - 1:03d}"
                path = BASE / subject / f"seed_{seed}" / chunk / "run_state.json"
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
                            "chunk": chunk,
                            "error": record.get("error"),
                        }
                    )
    expected = len(subjects) * len(seeds) * len(CHUNKS)
    print(
        json.dumps(
            {
                "expected": expected,
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
