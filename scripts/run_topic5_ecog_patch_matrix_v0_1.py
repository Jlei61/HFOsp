#!/usr/bin/env python3
"""Run all frozen true-grid patch-necessity units resumably."""
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084"))
    parser.add_argument("--patch-sides", nargs="+", type=int, default=(2, 3), choices=(2, 3))
    parser.add_argument(
        "--lesion-mode", default="symmetric_incident",
        choices=("symmetric_incident", "inbound_first_entry"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/patch_necessity"
    ))
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_id < args.n_shards:
        raise ValueError("require 0 <= shard-id < n-shards")
    units = [
        (subject, seed_index, side)
        for subject in args.subjects for seed_index in range(3) for side in args.patch_sides
    ]
    selected = [unit for index, unit in enumerate(units) if index % args.n_shards == args.shard_id]
    log_dir = args.output_root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"shard_{args.shard_id:02d}_of_{args.n_shards:02d}.json"
    status = {"selected": len(selected), "completed": [], "failed": []}
    for subject, seed_index, side in selected:
        command = [
            sys.executable, str(ROOT / "scripts/run_topic5_ecog_patch_necessity_v0_1.py"),
            "--subject", subject, "--seed-index", str(seed_index),
            "--patch-side", str(side), "--device", args.device,
            "--lesion-mode", args.lesion_mode,
            "--output-root", str(args.output_root),
        ]
        if args.smoke:
            command.extend(["--max-patches", "1", "--n-controls", "2"])
        if args.force:
            command.append("--force")
        started = time.time()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        row = {
            "subject": subject, "seed_index": seed_index, "patch_side": side,
            "returncode": process.returncode, "runtime_sec": time.time() - started,
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
