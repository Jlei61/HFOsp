#!/usr/bin/env python3
"""Refresh stale training units without rewriting the frozen 384-unit manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_ecog_graph_training_v0_1 import units_for_subject  # noqa: E402
from scripts.train_topic5_ecog_graph_unit_v0_1 import TOP1_CONTRACT  # noqa: E402


def output_directory(output_root: Path, unit: dict[str, str | int]) -> Path:
    graph_id = Path(str(unit["graph_path"])).stem
    return output_root / str(unit["subject"]) / (
        f"{unit['family']}__{graph_id}__seed{unit['seed_index']}"
    )


def is_stale(summary_path: Path, requested_device: str) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    requested_device_type = requested_device.split(":", 1)[0]
    return not (
        summary.get("training_device_type") == requested_device_type
        and summary.get("top1_contract") == TOP1_CONTRACT
        and summary.get("batch_size") == 512
        and summary.get("microsteps") == 2
        and summary.get("state_dim") == 1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    args = parser.parse_args()
    units = units_for_subject(args.subject, args.graph_root)
    units.sort(key=lambda unit: (int(unit["seed_index"]), int(unit["graph_index"]), str(unit["family"])))
    stale_units = [
        unit for unit in units
        if is_stale(output_directory(args.output_root, unit) / "summary.json", args.device)
    ]
    selected = [
        unit for index, unit in enumerate(stale_units)
        if index % args.n_shards == args.shard_id
    ]
    failures = []
    for unit in selected:
        command = [
            sys.executable, str(ROOT / "scripts/train_topic5_ecog_graph_unit_v0_1.py"),
            "--subject", str(unit["subject"]), "--family", str(unit["family"]),
            "--graph-path", str(unit["graph_path"]), "--seed-index", str(unit["seed_index"]),
            "--device", args.device, "--output-root", str(args.output_root),
            "--force",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            failures.append({"unit": unit, "stderr": result.stderr[-3000:]})
    payload = {
        "subject": args.subject,
        "n_stale_detected": len(stale_units),
        "selected": len(selected),
        "failed": failures,
        "complete": not failures,
    }
    log_dir = args.output_root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"stale_{args.subject}_{args.shard_id:02d}_of_{args.n_shards:02d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
