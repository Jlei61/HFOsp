#!/usr/bin/env python3
"""Run, aggregate and notify the frozen rev18 fit screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--maximum-workers", type=int, default=12)
    args = parser.parse_args()
    config_path = args.config.resolve()
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/launch_topic4_rev12_node_workers.py"),
        "--config", str(config_path), "--seed-pool", "fit",
        "--expected-commit", args.expected_commit,
        "--maximum-workers", str(args.maximum_workers),
        "--unit-prefix", "codex-t4-r18-global",
    ], cwd=ROOT, check=True)
    completed = subprocess.run([
        str(PYTHON),
        str(ROOT / "scripts/aggregate_topic4_rev18_dual_field_global_screen.py"),
        "--config", str(config_path),
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    print(completed.stdout, flush=True)
    payload = json.loads((
        Path("/home/honglab/leijiaxin/HFOsp")
        / json.loads(config_path.read_text())["output_root"]
        / "analysis/global_screen_aggregate.json"
    ).read_text())
    status = str(payload["status"])
    nominees = [row["candidate_id"] for row in payload["nominated_candidates"]]
    subprocess.run([
        "notify-send", "Topic 4 rev18 Node screen complete",
        f"{status}; nominees={len(nominees)}",
    ], check=False)
    if status != "REV18_DUAL_FIELD_GLOBAL_SCREEN_AGGREGATE_COMPLETE":
        raise RuntimeError(f"rev18 aggregate did not complete: {status}")


if __name__ == "__main__":
    main()
