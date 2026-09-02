#!/usr/bin/env python3
"""Evaluate pre-fixed representative free fields for four ECoG network families."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084"))
    parser.add_argument("--training-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_id < args.n_shards:
        raise ValueError("require 0 <= shard-id < n-shards")
    names = (
        "TRUE_GRID__TRUE_GRID__seed{seed}",
        "SUFFIX_SHUFFLED__TRUE_GRID__seed{seed}",
        "WRONG_GRID__WRONG_GRID_00__seed{seed}",
        "DEGREE_RANDOM__DEGREE_RANDOM_00__seed{seed}",
    )
    units = [
        args.training_root / subject / template.format(seed=seed) / "summary.json"
        for subject in args.subjects for seed in range(3) for template in names
    ]
    selected = [path for index, path in enumerate(units) if index % args.n_shards == args.shard_id]
    log_dir = args.training_root / "field_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"shard_{args.shard_id:02d}_of_{args.n_shards:02d}.json"
    status = {"selected": len(selected), "completed": [], "failed": []}
    for summary_path in selected:
        command = [
            sys.executable, str(ROOT / "scripts/evaluate_topic5_ecog_free_fields_v0_1.py"),
            "--summary-path", str(summary_path), "--device", args.device,
        ]
        if args.force:
            command.append("--force")
        started = time.time()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        row = {
            "summary_path": str(summary_path), "returncode": process.returncode,
            "runtime_sec": time.time() - started,
            "stdout_tail": process.stdout[-4000:], "stderr_tail": process.stderr[-4000:],
        }
        status["completed" if process.returncode == 0 else "failed"].append(row)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
