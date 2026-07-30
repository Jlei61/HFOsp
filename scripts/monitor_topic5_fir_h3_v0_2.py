#!/usr/bin/env python3
"""Compact monitor for the formal FIR-H3 LOSO run."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/topic5_minimal_sequence_kernel_closeout/"
            "fir_h3_formal_v0_2"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summaries = list(root.glob("seed_*/*/summary.json"))
    logs = list((root / "logs").glob("*.log"))
    failures = []
    for path in logs:
        text = path.read_text(errors="replace")
        if "Traceback (most recent call last)" in text:
            failures.append(path.name)
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
                "root": str(root),
                "expected_cells": 102,
                "complete_cells": len(summaries),
                "log_files": len(logs),
                "logs_with_traceback": sorted(failures),
                "launcher_done": (root / "LAUNCHER_DONE.json").exists(),
                "gpu": gpu,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
