#!/usr/bin/env python3
"""Wait for the full-rank reference, then launch and monitor low-rank runs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _counts(root: Path, seeds: list[int]) -> dict[str, dict[str, int]]:
    out = {}
    for seed in seeds:
        out[str(seed)] = {
            str(rank): len(
                list(
                    (root / f"seed_{seed}" / f"rank_{rank}").glob(
                        "*/DONE.json"
                    )
                )
            )
            for rank in range(5)
        }
    return out


def _errors(root: Path) -> list[dict[str, str]]:
    patterns = ("traceback", "out of memory", "cuda error", "killed", "no space left")
    hits = []
    for path in root.glob("seed_*/rank_*/logs/*.log"):
        try:
            lines = path.read_text(errors="replace").splitlines()[-200:]
        except OSError:
            continue
        for line in lines:
            if any(pattern in line.lower() for pattern in patterns):
                hits.append({"file": str(path), "line": line[-1000:]})
    return hits[-20:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-launcher-pid", type=int, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--low-rank-tag", required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    reference = args.reference_root.resolve()
    low_rank_root = (
        repo
        / "results/topic5_low_rank_dynamics/runs"
        / args.low_rank_tag
    )
    watcher_status = reference / "low_rank_handoff_status.json"
    while not (reference / "DONE.json").exists():
        if (reference / "MONITOR_ALERT.json").exists():
            _write(
                watcher_status,
                {
                    "status": "blocked_by_reference_alert",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return
        if not _alive(args.reference_launcher_pid):
            _write(
                watcher_status,
                {
                    "status": "reference_launcher_exited_before_done",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return
        _write(
            watcher_status,
            {
                "status": "waiting_for_reference",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(max(10.0, float(args.interval_seconds)))

    launcher_log = reference / "low_rank_launcher.log"
    with launcher_log.open("a") as handle:
        process = subprocess.Popen(
            [
                "bash",
                str(repo / "scripts/run_topic5_low_rank_leaky_multiseed.sh"),
                str(args.selection.resolve()),
                args.low_rank_tag,
            ],
            cwd=repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "status": "low_rank_launched",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": int(process.pid),
        "run_root": str(low_rank_root),
        "ranks": [0, 1, 2, 3, 4],
        "seeds": [20260725, 20260726, 20260727],
    }
    _write(reference / "LOW_RANK_LAUNCHED.json", launch)
    seeds = [20260725, 20260726, 20260727]
    while True:
        counts = _counts(low_rank_root, seeds)
        errors = _errors(low_rank_root)
        complete = (low_rank_root / "DONE.json").exists()
        alive = _alive(process.pid)
        status = {
            **launch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "complete"
                if complete
                else "running"
                if alive
                else "low_rank_launcher_exited_before_done"
            ),
            "launcher_alive": alive,
            "completed_folds": counts,
            "error_count": len(errors),
            "recent_errors": errors,
        }
        low_rank_root.mkdir(parents=True, exist_ok=True)
        _write(low_rank_root / "monitor_status.json", status)
        if errors:
            _write(
                low_rank_root / "MONITOR_ALERT.json",
                {
                    "status": "training_error_detected",
                    "timestamp_utc": status["timestamp_utc"],
                    "recent_errors": errors,
                },
            )
        if complete:
            _write(low_rank_root / "MONITOR_DONE.json", status)
            break
        if not alive:
            _write(low_rank_root / "MONITOR_ALERT.json", status)
            break
        time.sleep(max(10.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
