#!/usr/bin/env python
"""Run one Figure 6 command with resource logging and fail-safe sentinels."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def meminfo():
    values = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) / 1024**2
    return {
        "mem_available_gb": values.get("MemAvailable", 0.0),
        "swap_used_gb": values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0),
    }


def process_rss_gb(pid: int):
    """Sum RSS over the launched process tree (conda is only the parent shim)."""
    proc_rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            proc_rows[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {int(pid)}
    changed = True
    while changed:
        changed = False
        for child, parent in proc_rows.items():
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    rss_kib = 0
    for child in descendants:
        try:
            with open(f"/proc/{child}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        rss_kib += int(line.split()[1])
                        break
        except OSError:
            continue
    return rss_kib / 1024**2


def gpu_stats():
    query = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(query, text=True, timeout=5).strip().splitlines()[0]
        used, free, util, temp = [float(x.strip()) for x in out.split(",")]
        return {
            "gpu_mem_used_mib": used,
            "gpu_mem_free_mib": free,
            "gpu_util_percent": util,
            "gpu_temp_c": temp,
        }
    except Exception:
        return {
            "gpu_mem_used_mib": float("nan"),
            "gpu_mem_free_mib": float("nan"),
            "gpu_util_percent": float("nan"),
            "gpu_temp_c": float("nan"),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--interval-sec", type=float, default=15.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("missing command after --")
    cfg = yaml.safe_load(args.config.read_text())
    thresholds = cfg["resources"]
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(ROOT)
    initial = {**meminfo(), **gpu_stats(), "disk_free_gb": disk.free / 1024**3}
    preflight = {
        "command": command,
        "cwd": str(ROOT),
        "started_epoch": time.time(),
        "initial_resources": initial,
        "thresholds": {
            "min_mem_available_gb": thresholds["min_mem_available_gb"],
            "min_disk_available_gb": thresholds["min_disk_available_gb"],
            "max_swap_used_gb": thresholds["max_swap_used_gb"],
        },
    }
    (run_dir / "RUNNING.json").write_text(json.dumps(preflight, indent=2))
    if initial["mem_available_gb"] < float(thresholds["min_mem_available_gb"]):
        raise SystemExit("preflight failed: MemAvailable below threshold")
    if initial["disk_free_gb"] < float(thresholds["min_disk_available_gb"]):
        raise SystemExit("preflight failed: disk free below threshold")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NUMBA_CACHE_DIR": "/tmp/hfosp_fig6_numba",
            "MPLCONFIGDIR": "/tmp/hfosp_fig6_mpl",
            "_MNE_FAKE_HOME_DIR": "/tmp/hfosp_fig6_mne",
            "OMP_NUM_THREADS": str(thresholds["cpu_threads"]),
            "MKL_NUM_THREADS": str(thresholds["cpu_threads"]),
        }
    )
    stdout_path = run_dir / "stdout.log"
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    (run_dir / "PID").write_text(str(proc.pid))

    def copy_output():
        with open(stdout_path, "a", buffering=1) as out:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                out.write(line)

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    fields = [
        "epoch",
        "elapsed_sec",
        "pid_rss_gb",
        "mem_available_gb",
        "swap_used_gb",
        "disk_free_gb",
        "gpu_mem_used_mib",
        "gpu_mem_free_mib",
        "gpu_util_percent",
        "gpu_temp_c",
    ]
    resource_path = run_dir / "resource.csv"
    violations = 0
    started = time.time()
    with open(resource_path, "w", newline="", buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        while proc.poll() is None:
            disk = shutil.disk_usage(ROOT)
            row = {
                "epoch": time.time(),
                "elapsed_sec": time.time() - started,
                "pid_rss_gb": process_rss_gb(proc.pid),
                **meminfo(),
                "disk_free_gb": disk.free / 1024**3,
                **gpu_stats(),
            }
            writer.writerow(row)
            unsafe = (
                row["mem_available_gb"] < float(thresholds["min_mem_available_gb"])
                or row["disk_free_gb"] < float(thresholds["min_disk_available_gb"])
                or row["swap_used_gb"] > float(thresholds["max_swap_used_gb"])
            )
            violations = violations + 1 if unsafe else 0
            if violations >= 3:
                paused = {
                    "reason": "resource threshold violated for three consecutive samples",
                    "last_resource": row,
                    "pid": proc.pid,
                }
                (run_dir / "RESOURCE_PAUSED.json").write_text(json.dumps(paused, indent=2))
                os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(min(max(args.interval_sec, 2.0), 30.0))
    rc = proc.wait()
    reader.join(timeout=10)
    record = {
        **preflight,
        "finished_epoch": time.time(),
        "elapsed_sec": time.time() - started,
        "return_code": rc,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "resource_log": str(resource_path.relative_to(ROOT)),
    }
    sentinel = "DONE_PROCESS.json" if rc == 0 else "ABORTED.json"
    (run_dir / sentinel).write_text(json.dumps(record, indent=2))
    if rc == 0:
        (run_dir / "RUNNING.json").unlink(missing_ok=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
