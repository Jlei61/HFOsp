#!/usr/bin/env python3
"""Status watcher for the Topic 5.2 dynamical motif run.

Rewrites RUN_STATUS.json every interval with per-unit terminal state, live
claims, GPU memory and recent progress.  Read-only with respect to results.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"


def gpu_snapshot() -> list[dict]:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:  # noqa: BLE001 - the watcher must never take the run down
        return []
    rows = []
    for line in raw.strip().splitlines():
        index, used, total, util = [part.strip() for part in line.split(",")]
        rows.append({"gpu": int(index), "memory_used_mib": int(used),
                     "memory_total_mib": int(total), "utilization_percent": int(util)})
    return rows


def alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def scan(out_root: Path, tag: str) -> dict:
    manifest_path = out_root / "FORMAL_UNIT_MANIFEST.csv"
    if not manifest_path.exists():
        return {"status": "no_manifest"}
    manifest = pd.read_csv(manifest_path)
    states, seconds = [], []
    for _, row in manifest.iterrows():
        directory = (out_root / tag / str(row.frame) / str(row.unit_id)
                     / str(row.model_id) / f"seed{int(row.seed_index)}")
        done = directory / "DONE.json"
        if done.exists() and (directory / "checkpoint.pt").exists():
            states.append("DONE")
            try:
                seconds.append(float(json.loads(done.read_text()).get("seconds", float("nan"))))
            except json.JSONDecodeError:
                seconds.append(float("nan"))
        elif (directory / "FAILED.json").exists():
            states.append("FAILED")
            seconds.append(float("nan"))
        elif (directory / "resume.pt").exists():
            states.append("RUNNING")
            seconds.append(float("nan"))
        else:
            states.append("PENDING")
            seconds.append(float("nan"))
    manifest["state"] = states
    manifest["seconds"] = seconds

    claims = []
    lock_root = out_root / "locks" / tag
    if lock_root.exists():
        for lock in sorted(lock_root.iterdir()):
            record_path = lock / "claim.json"
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                continue
            record["age_seconds"] = time.time() - float(record.get("heartbeat", 0))
            record["pid_alive"] = alive(record.get("pid", -1))
            claims.append(record)

    counts = manifest.state.value_counts().to_dict()
    return {
        "contract": "topic5_dynamical_motif_run_status_v0_1",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": tag,
        "n_units": int(len(manifest)),
        "counts": {k: int(v) for k, v in counts.items()},
        "fraction_done": float(counts.get("DONE", 0) / max(1, len(manifest))),
        "median_unit_seconds": float(manifest.seconds.median(skipna=True))
        if manifest.seconds.notna().any() else None,
        "total_gpu_hours": float(manifest.seconds.sum(skipna=True) / 3600.0),
        "per_model": manifest.groupby("model_id").state.value_counts().unstack(fill_value=0)
        .to_dict(orient="index"),
        "failed_units": manifest[manifest.state == "FAILED"][
            ["unit_id", "model_id", "seed_index"]].to_dict(orient="records"),
        "active_claims": [c for c in claims if c["age_seconds"] < 3600],
        "stale_claims": [c for c in claims if c["age_seconds"] >= 3600],
        "gpu": gpu_snapshot(),
        "load_average": os.getloadavg(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        status = scan(args.out_root, args.tag)
        path = args.out_root / "RUN_STATUS.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
        print(f"[watch] {status.get('counts')} done={status.get('fraction_done')}", flush=True)
        if args.once:
            return
        time.sleep(float(args.interval))


if __name__ == "__main__":
    main()
