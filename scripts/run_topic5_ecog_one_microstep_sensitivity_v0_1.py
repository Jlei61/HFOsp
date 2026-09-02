#!/usr/bin/env python3
"""Train the pre-specified one-microstep true-vs-wrong-grid sensitivity."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084")
    )
    parser.add_argument("--graph-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training_one_microstep"
    ))
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_id < args.n_shards:
        raise ValueError("require 0 <= shard-id < n-shards")
    units = []
    for seed_index in range(3):
        for graph_index in range(-1, 31):
            for subject in args.subjects:
                graph_id = "TRUE_GRID" if graph_index < 0 else f"WRONG_GRID_{graph_index:02d}"
                units.append({
                    "subject": subject,
                    "family": "TRUE_GRID" if graph_index < 0 else "WRONG_GRID",
                    "graph_index": graph_index,
                    "seed_index": seed_index,
                    "graph_path": str(
                        args.graph_root / subject / "four_neighbour" / f"{graph_id}.npz"
                    ),
                })
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "TRAINING_UNIT_MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "unit_index", "subject", "family", "graph_index", "seed_index", "graph_path",
        ))
        writer.writeheader()
        for index, unit in enumerate(units):
            writer.writerow({"unit_index": index, **unit})
    selected = [unit for index, unit in enumerate(units) if index % args.n_shards == args.shard_id]
    failures = []
    for unit in selected:
        command = [
            sys.executable, str(ROOT / "scripts/train_topic5_ecog_graph_unit_v0_1.py"),
            "--subject", unit["subject"], "--family", unit["family"],
            "--graph-path", unit["graph_path"], "--seed-index", str(unit["seed_index"]),
            "--device", args.device, "--microsteps", "1",
            "--output-root", str(args.output_root),
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            failures.append({"unit": unit, "stderr": result.stderr[-3000:]})
    payload = {"selected": len(selected), "failed": failures, "complete": not failures}
    log_root = args.output_root / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / f"shard_{args.shard_id:02d}_of_{args.n_shards:02d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
