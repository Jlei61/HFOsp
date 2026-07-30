#!/usr/bin/env python3
"""Monitor formal multi-seed training and write progress/alert sentinels."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
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


def _meminfo() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = float(raw.strip().split()[0]) / (1024.0 * 1024.0)
    return values


def _gpu() -> dict[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
        memory, utilization, temperature = [
            float(item.strip()) for item in output.split(",")
        ]
        return {
            "gpu_memory_used_mb": memory,
            "gpu_utilization_percent": utilization,
            "gpu_temperature_c": temperature,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {
            "gpu_memory_used_mb": float("nan"),
            "gpu_utilization_percent": float("nan"),
            "gpu_temperature_c": float("nan"),
        }


def _error_hits(log_root: Path) -> list[dict[str, str]]:
    patterns = (
        "traceback",
        "out of memory",
        "cuda error",
        "killed",
        "no space left",
    )
    hits = []
    for path in log_root.glob("seed_*/logs/*.log"):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[-200:]:
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                hits.append({"file": str(path), "line": line[-1000:]})
    return hits[-20:]


def _current_subjects(formal_root: Path, seeds: list[int]) -> dict[str, str | None]:
    current = {}
    for seed in seeds:
        seed_root = formal_root / f"seed_{seed}"
        running = []
        for path in seed_root.glob("*/run_state.json"):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "RUNNING":
                running.append(path.parent.name)
        current[str(seed)] = sorted(running)[0] if running else None
    return current


def _counts(formal_root: Path, seeds: list[int]) -> dict[str, int]:
    return {
        str(seed): len(list((formal_root / f"seed_{seed}").glob("*/DONE.json")))
        for seed in seeds
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--min-mem-available-gb", type=float, default=32.0)
    parser.add_argument("--max-swap-delta-gb", type=float, default=1.0)
    parser.add_argument("--resource-breach-samples", type=int, default=3)
    args = parser.parse_args()
    root = args.formal_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "monitor_progress.csv"
    fields = [
        "timestamp_utc",
        *[f"done_seed_{seed}" for seed in args.seeds],
        *[f"current_seed_{seed}" for seed in args.seeds],
        "launcher_alive",
        "error_count",
        "mem_available_gb",
        "swap_used_gb",
        "swap_delta_gb",
        "gpu_memory_used_mb",
        "gpu_utilization_percent",
        "gpu_temperature_c",
    ]
    memory = _meminfo()
    baseline_swap_used = memory.get("SwapTotal", 0.0) - memory.get("SwapFree", 0.0)
    breach_count = 0
    with progress_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if progress_path.stat().st_size == 0:
            writer.writeheader()
        while True:
            timestamp = datetime.now(timezone.utc).isoformat()
            counts = _counts(root, args.seeds)
            current = _current_subjects(root, args.seeds)
            errors = _error_hits(root)
            memory = _meminfo()
            swap_used = memory.get("SwapTotal", 0.0) - memory.get("SwapFree", 0.0)
            swap_delta = swap_used - baseline_swap_used
            gpu = _gpu()
            launcher_alive = _alive(args.launcher_pid)
            complete = all(counts[str(seed)] == 34 for seed in args.seeds)
            status = {
                "timestamp_utc": timestamp,
                "status": (
                    "complete"
                    if complete
                    else "running"
                    if launcher_alive
                    else "launcher_exited_before_complete"
                ),
                "launcher_pid": int(args.launcher_pid),
                "launcher_alive": launcher_alive,
                "completed_folds": counts,
                "current_subjects": current,
                "error_count": len(errors),
                "recent_errors": errors,
                "mem_available_gb": memory.get("MemAvailable", float("nan")),
                "swap_delta_gb": swap_delta,
                **gpu,
            }
            _atomic_json(root / "monitor_status.json", status)
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    **{
                        f"done_seed_{seed}": counts[str(seed)]
                        for seed in args.seeds
                    },
                    **{
                        f"current_seed_{seed}": current[str(seed)]
                        for seed in args.seeds
                    },
                    "launcher_alive": launcher_alive,
                    "error_count": len(errors),
                    "mem_available_gb": memory.get(
                        "MemAvailable", float("nan")
                    ),
                    "swap_used_gb": swap_used,
                    "swap_delta_gb": swap_delta,
                    **gpu,
                }
            )
            handle.flush()
            if errors:
                _atomic_json(
                    root / "MONITOR_ALERT.json",
                    {
                        "status": "training_error_detected",
                        "timestamp_utc": timestamp,
                        "recent_errors": errors,
                    },
                )
            resource_breach = (
                memory.get("MemAvailable", float("inf"))
                < float(args.min_mem_available_gb)
                or swap_delta > float(args.max_swap_delta_gb)
            )
            breach_count = breach_count + 1 if resource_breach else 0
            if breach_count >= int(args.resource_breach_samples):
                alert = {
                    "status": "resource_breach_training_terminated",
                    "timestamp_utc": timestamp,
                    "launcher_pid": int(args.launcher_pid),
                    "mem_available_gb": memory.get("MemAvailable"),
                    "swap_delta_gb": swap_delta,
                }
                _atomic_json(root / "MONITOR_ALERT.json", alert)
                try:
                    os.killpg(int(args.launcher_pid), signal.SIGTERM)
                except OSError:
                    pass
                break
            if complete:
                _atomic_json(
                    root / "MONITOR_DONE.json",
                    {
                        "status": "all_folds_complete",
                        "timestamp_utc": timestamp,
                        "completed_folds": counts,
                    },
                )
                break
            if not launcher_alive:
                _atomic_json(
                    root / "MONITOR_ALERT.json",
                    {
                        "status": "launcher_exited_before_complete",
                        "timestamp_utc": timestamp,
                        "completed_folds": counts,
                        "recent_errors": errors,
                    },
                )
                break
            time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
