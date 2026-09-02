#!/usr/bin/env python3
"""Run the frozen full-grid ECoG graph-training matrix resumably."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def units_for_subject(subject: str, graph_root: Path) -> list[dict[str, str | int]]:
    root = graph_root / subject / "four_neighbour"
    true_graph = root / "TRUE_GRID.npz"
    units: list[dict[str, str | int]] = []
    for seed_index in range(3):
        units.append({
            "subject": subject, "family": "TRUE_GRID", "graph_path": str(true_graph),
            "graph_index": -1, "seed_index": seed_index,
        })
        units.append({
            "subject": subject, "family": "SUFFIX_SHUFFLED", "graph_path": str(true_graph),
            "graph_index": -1, "seed_index": seed_index,
        })
        for graph_index in range(31):
            units.append({
                "subject": subject, "family": "WRONG_GRID",
                "graph_path": str(root / f"WRONG_GRID_{graph_index:02d}.npz"),
                "graph_index": graph_index, "seed_index": seed_index,
            })
            units.append({
                "subject": subject, "family": "DEGREE_RANDOM",
                "graph_path": str(root / f"DEGREE_RANDOM_{graph_index:02d}.npz"),
                "graph_index": graph_index, "seed_index": seed_index,
            })
    return units


def write_manifest(units: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "unit_index", "subject", "family", "graph_index", "seed_index", "graph_path",
        ))
        writer.writeheader()
        for unit_index, unit in enumerate(units):
            writer.writerow({"unit_index": unit_index, **unit})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084"))
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--graph-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_id < args.n_shards:
        raise ValueError("require 0 <= shard-id < n-shards")

    units: list[dict[str, str | int]] = []
    for subject in args.subjects:
        units.extend(units_for_subject(subject, args.graph_root))
    # Interleave families and patients deterministically across worker shards.
    units.sort(key=lambda unit: (
        int(unit["seed_index"]), int(unit["graph_index"]), str(unit["family"]), str(unit["subject"])
    ))
    manifest = args.output_root / "TRAINING_UNIT_MANIFEST.csv"
    write_manifest(units, manifest)
    if args.prepare_only:
        print(json.dumps({"n_units": len(units), "manifest": str(manifest)}, indent=2))
        return

    selected = [(index, unit) for index, unit in enumerate(units) if index % args.n_shards == args.shard_id]
    log_dir = args.output_root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"shard_{args.shard_id:02d}_of_{args.n_shards:02d}.json"
    status = {
        "schema": "topic5_ecog_training_worker_v0.1",
        "shard_id": args.shard_id,
        "n_shards": args.n_shards,
        "selected_units": len(selected),
        "completed": [],
        "failed": [],
        "started_epoch": time.time(),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    for unit_index, unit in selected:
        command = [
            str(PYTHON), str(ROOT / "scripts/train_topic5_ecog_graph_unit_v0_1.py"),
            "--subject", str(unit["subject"]),
            "--family", str(unit["family"]),
            "--graph-path", str(unit["graph_path"]),
            "--seed-index", str(unit["seed_index"]),
            "--device", args.device,
        ]
        if args.smoke:
            command.append("--smoke")
        if args.force:
            command.append("--force")
        started = time.time()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        record = {
            "unit_index": unit_index,
            **unit,
            "returncode": result.returncode,
            "runtime_sec": time.time() - started,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        if result.returncode == 0:
            status["completed"].append(record)
        else:
            status["failed"].append(record)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    status["complete"] = True
    status["finished_epoch"] = time.time()
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
