"""Persistent, read-only progress watcher for the v0.4 pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=72.0)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    log_path = out_root / "WATCHER.jsonl"
    deadline = time.time() + args.max_hours * 3600.0
    while time.time() < deadline:
        snapshot = {"timestamp": time.time(), "stages": {}}
        for stage in ("smoke", "core", "dose", "gru"):
            path = out_root / f"STAGE_{stage.upper()}_STATUS.json"
            if path.exists():
                try:
                    snapshot["stages"][stage] = json.loads(path.read_text())
                except json.JSONDecodeError:
                    snapshot["stages"][stage] = {"status": "TRANSIENT_PARTIAL_WRITE"}
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True
        )
        snapshot["gpu"] = gpu.stdout.strip()
        snapshot["done_units_on_disk"] = sum(1 for _ in (out_root / "per_subject").glob("**/DONE.json"))
        with log_path.open("a") as handle:
            handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        if (out_root / "PIPELINE_COMPLETE.json").exists():
            return 0
        time.sleep(max(10, int(args.interval_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
