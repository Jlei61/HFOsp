#!/usr/bin/env python3
"""Report compact progress for the persistent path-mode pilot."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SUBJECTS = (
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
)
SEEDS = (20260726, 20260727, 20260728)
SPECS = (
    (0, "no_history"),
    (1, "merged_path"),
    (1, "intact"),
    (1, "weight_shuffle"),
    (2, "intact"),
    (2, "weight_shuffle"),
    (2, "mode_shuffle"),
    (3, "intact"),
    (3, "weight_shuffle"),
    (3, "mode_shuffle"),
    (4, "intact"),
    (4, "weight_shuffle"),
    (4, "mode_shuffle"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/topic5_structured_axis_graph/"
            "screen_persistent_path_mode_v0_9"
        ),
    )
    args = parser.parse_args()
    expected = len(SUBJECTS) * len(SEEDS) * len(SPECS)
    status_counts = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "PENDING": 0}
    failed = []
    for seed in SEEDS:
        for subject in SUBJECTS:
            for mode_count, control in SPECS:
                run_dir = (
                    args.root
                    / f"seed_{seed}"
                    / f"k_{mode_count}"
                    / control
                    / subject
                )
                state_path = run_dir / "run_state.json"
                if not state_path.exists():
                    status_counts["PENDING"] += 1
                    continue
                try:
                    state = json.loads(state_path.read_text())
                    status = str(state.get("status", "FAILED")).upper()
                except Exception:
                    status = "FAILED"
                if status not in status_counts:
                    status = "FAILED"
                status_counts[status] += 1
                if status == "FAILED":
                    failed.append(str(run_dir))
    gpu = ""
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        gpu = "unavailable"
    print(
        json.dumps(
            {
                "expected": expected,
                "status_counts": status_counts,
                "percent_complete": round(
                    100.0 * status_counts["COMPLETE"] / expected, 1
                ),
                "failed": failed,
                "gpu_index_used_total_mib_util_percent": gpu,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
