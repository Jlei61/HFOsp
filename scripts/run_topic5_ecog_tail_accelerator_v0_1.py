#!/usr/bin/env python3
"""Run a disjoint far-tail slice while the canonical manifest advances from the front."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--minimum-graph-index", type=int, default=15)
    parser.add_argument("--maximum-graph-index", type=int, default=29)
    parser.add_argument("--seed-index", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    units = [
        (family, graph_index)
        for graph_index in range(args.maximum_graph_index, args.minimum_graph_index - 1, -1)
        for family in ("WRONG_GRID", "DEGREE_RANDOM")
    ]
    selected = [unit for index, unit in enumerate(units) if index % args.n_shards == args.shard_id]
    failures = []
    for family, graph_index in selected:
        graph_id = f"{family}_{graph_index:02d}"
        command = [
            sys.executable, str(ROOT / "scripts/train_topic5_ecog_graph_unit_v0_1.py"),
            "--subject", "958", "--family", family,
            "--graph-path", str(Path(
                f"results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs/958/four_neighbour/{graph_id}.npz"
            )),
            "--seed-index", str(args.seed_index), "--device", args.device,
        ]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if process.returncode:
            failures.append({"family": family, "graph_index": graph_index, "stderr": process.stderr[-3000:]})
    payload = {
        "selected": len(selected), "seed_index": args.seed_index,
        "minimum_graph_index": args.minimum_graph_index,
        "maximum_graph_index": args.maximum_graph_index,
        "failed": failures, "complete": not failures,
    }
    log = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1/training/worker_logs")
    (log / f"tail_seed{args.seed_index}_{args.minimum_graph_index}_{args.maximum_graph_index}_{args.shard_id}_of_{args.n_shards}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
