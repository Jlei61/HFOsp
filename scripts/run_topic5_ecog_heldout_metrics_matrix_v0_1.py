#!/usr/bin/env python3
"""Resumable sharded runner for extended ECoG held-out metrics."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--subjects", nargs="+", default=("958", "1084"), choices=("958", "1084"))
    parser.add_argument("--training-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    args = parser.parse_args()
    paths = sorted(
        path for subject in args.subjects
        for path in (args.training_root / subject).glob("*/summary.json")
    )
    paths = [path for path in paths if not json.loads(path.read_text()).get("smoke", False)]
    selected = [path for index, path in enumerate(paths) if index % args.n_shards == args.shard_id]
    failures = []
    for path in selected:
        command = [
            sys.executable, str(ROOT / "scripts/evaluate_topic5_ecog_heldout_metrics_v0_1.py"),
            "--summary-path", str(path), "--device", args.device,
            "--batch-size", str(args.batch_size),
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            failures.append({"path": str(path), "stderr": result.stderr[-3000:]})
    payload = {"selected": len(selected), "failed": failures, "complete": not failures}
    status = args.training_root / "worker_logs" / f"heldout_metrics_{args.shard_id:02d}_of_{args.n_shards:02d}.json"
    status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
