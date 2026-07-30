#!/usr/bin/env python3
"""Compact progress monitor for the Topic 5 minimal-kernel closeout."""
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
            "results/topic5_minimal_sequence_kernel_closeout/formal_v0_2"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summaries = list(root.glob("seed_*/*/summary.json"))
    logs = list((root / "logs").glob("*.log"))
    failures = []
    for log in logs:
        text = log.read_text(errors="replace")
        if "Traceback (most recent call last)" in text or "RuntimeError:" in text:
            if not any(
                summary.parent.name in log.name for summary in summaries
            ):
                failures.append(log.name)
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
    payload = {
        "root": str(root),
        "expected_cells": 102,
        "complete_cells": len(summaries),
        "log_files": len(logs),
        "suspected_failures": sorted(failures),
        "gpu": gpu,
        "launcher_done": (root / "LAUNCHER_DONE.json").exists(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
