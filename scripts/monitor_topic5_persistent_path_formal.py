#!/usr/bin/env python3
"""Monitor the 34-subject persistent path-mode formal run."""
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


SEEDS = (20260726, 20260727, 20260728)
SPECS = (
    (0, "no_history"),
    (1, "merged_path"),
    (2, "intact"),
    (2, "weight_shuffle"),
    (2, "mode_shuffle"),
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _subjects() -> list[str]:
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(
        root
        / "results/topic5_interictal_rank_distribution/"
        "dataset_v0_4/subject_audit.csv"
    )
    values = sorted(frame.loc[frame.status.eq("ok"), "subject"].astype(str))
    if len(values) != 34:
        raise RuntimeError(f"expected 34 subjects, found {len(values)}")
    return values


def _memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = float(raw.strip().split()[0]) / (1024.0 * 1024.0)
    return {
        "mem_available_gb": values.get("MemAvailable", float("nan")),
        "swap_used_gb": (
            values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0)
        ),
    }


def _gpu() -> dict[str, float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
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


def snapshot(root: Path, launcher_pid: int | None) -> dict:
    counts = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "PENDING": 0}
    failed = []
    current = []
    sealed = True
    for seed in SEEDS:
        for subject in _subjects():
            for mode_count, control in SPECS:
                run_dir = (
                    root
                    / f"seed_{seed}"
                    / f"k_{mode_count}"
                    / control
                    / subject
                )
                state_path = run_dir / "run_state.json"
                if not state_path.exists():
                    counts["PENDING"] += 1
                    continue
                try:
                    state = json.loads(state_path.read_text())
                    status = str(state.get("status", "FAILED")).upper()
                except (OSError, json.JSONDecodeError):
                    status = "FAILED"
                    state = {}
                if status not in counts:
                    status = "FAILED"
                counts[status] += 1
                sealed &= not bool(state.get("ictal_target_read", True))
                if status == "RUNNING":
                    current.append(str(run_dir.relative_to(root)))
                elif status == "FAILED":
                    failed.append(str(run_dir.relative_to(root)))
    complete = counts["COMPLETE"] == 510
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete"
            if complete
            else "running"
            if launcher_pid is None or _alive(launcher_pid)
            else "launcher_exited_before_complete"
        ),
        "expected_runs": 510,
        "status_counts": counts,
        "percent_complete": round(100.0 * counts["COMPLETE"] / 510, 2),
        "current_runs": current,
        "failed_runs": failed,
        "launcher_pid": launcher_pid,
        "launcher_alive": _alive(launcher_pid),
        "ictal_target_sealed": bool(sealed),
        **_memory(),
        **_gpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--min-mem-available-gb", type=float, default=64.0)
    parser.add_argument("--max-swap-increase-gb", type=float, default=1.0)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.watch:
        print(json.dumps(snapshot(root, args.launcher_pid), indent=2))
        return

    baseline_swap = _memory()["swap_used_gb"]
    breach_count = 0
    progress = root / "monitor_progress.csv"
    fields = [
        "timestamp_utc",
        "complete",
        "running",
        "failed",
        "pending",
        "percent_complete",
        "mem_available_gb",
        "swap_used_gb",
        "gpu_memory_used_mb",
        "gpu_utilization_percent",
        "gpu_temperature_c",
    ]
    with progress.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if progress.stat().st_size == 0:
            writer.writeheader()
        while True:
            value = snapshot(root, args.launcher_pid)
            _atomic_json(root / "monitor_status.json", value)
            writer.writerow(
                {
                    "timestamp_utc": value["timestamp_utc"],
                    "complete": value["status_counts"]["COMPLETE"],
                    "running": value["status_counts"]["RUNNING"],
                    "failed": value["status_counts"]["FAILED"],
                    "pending": value["status_counts"]["PENDING"],
                    "percent_complete": value["percent_complete"],
                    "mem_available_gb": value["mem_available_gb"],
                    "swap_used_gb": value["swap_used_gb"],
                    "gpu_memory_used_mb": value["gpu_memory_used_mb"],
                    "gpu_utilization_percent": value[
                        "gpu_utilization_percent"
                    ],
                    "gpu_temperature_c": value["gpu_temperature_c"],
                }
            )
            handle.flush()
            if value["failed_runs"]:
                _atomic_json(
                    root / "MONITOR_ALERT.json",
                    {
                        "status": "failed_run_detected",
                        "failed_runs": value["failed_runs"],
                        "timestamp_utc": value["timestamp_utc"],
                    },
                )
            resource_breach = (
                value["mem_available_gb"] < args.min_mem_available_gb
                or value["swap_used_gb"] - baseline_swap
                > args.max_swap_increase_gb
            )
            breach_count = breach_count + 1 if resource_breach else 0
            if breach_count >= 3:
                _atomic_json(
                    root / "MONITOR_ALERT.json",
                    {
                        "status": "resource_breach_training_terminated",
                        "timestamp_utc": value["timestamp_utc"],
                        "mem_available_gb": value["mem_available_gb"],
                        "swap_increase_gb": value["swap_used_gb"]
                        - baseline_swap,
                    },
                )
                if args.launcher_pid is not None:
                    try:
                        os.killpg(int(args.launcher_pid), signal.SIGTERM)
                    except OSError:
                        pass
                break
            if value["status"] == "complete":
                _atomic_json(
                    root / "MONITOR_DONE.json",
                    {
                        "status": "all_runs_complete",
                        "timestamp_utc": value["timestamp_utc"],
                    },
                )
                break
            if value["status"] == "launcher_exited_before_complete":
                _atomic_json(root / "MONITOR_ALERT.json", value)
                break
            time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
