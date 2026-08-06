#!/usr/bin/env python3
"""Compact progress and resource monitor for the v0.4 formal run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "INPUT_MANIFEST.json").read_text())
    subjects = manifest["cohort"]["primary_subjects"]
    seeds = [11, 29, 47]
    done = []
    failed = []
    running = []
    for seed in seeds:
        for subject in subjects:
            directory = root / "per_subject" / f"seed_{seed}" / subject
            if (directory / "DONE.json").exists():
                payload = json.loads((directory / "DONE.json").read_text())
                done.append(
                    {
                        "seed": seed,
                        "subject": subject,
                        "elapsed_seconds": payload.get("elapsed_seconds"),
                        "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
                    }
                )
            elif (directory / "FAILED.json").exists():
                failed.append({"seed": seed, "subject": subject})
            elif directory.exists():
                progress_log = root / "logs" / f"train_{subject}_seed{seed}.log"
                progress = None
                if progress_log.exists():
                    lines = [line for line in progress_log.read_text().splitlines() if line.strip()]
                    if lines:
                        try:
                            progress = json.loads(lines[-1])
                        except json.JSONDecodeError:
                            progress = {"last_line": lines[-1][-240:]}
                running.append(
                    {
                        "seed": seed,
                        "subject": subject,
                        "stage": progress.get("stage") if progress else None,
                        "epoch": progress.get("epoch") if progress else None,
                        "elapsed_seconds": progress.get("elapsed_seconds") if progress else None,
                    }
                )
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as error:  # pragma: no cover
        gpu = f"unavailable:{error}"
    cache_done = sum(
        (root / "cache" / f"outer_{subject}" / "DONE.json").exists()
        for subject in subjects
    )
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
    available_ram_gb = float(meminfo.get("MemAvailable", "0 kB").split()[0]) / 1024**2
    process_output = subprocess.check_output(
        ["ps", "-eo", "cmd"], text=True
    )
    train_process_count = sum(
        "run_topic5_history_conditioned_field_fold_v0_4.py" in line
        for line in process_output.splitlines()
    )
    eta_seconds = None
    launcher = root / "watchers" / "launcher_state.json"
    if done and launcher.exists():
        launch_time = dt.datetime.fromisoformat(json.loads(launcher.read_text())["launched_at"])
        now = dt.datetime.now(dt.timezone.utc)
        elapsed = max((now - launch_time).total_seconds(), 1.0)
        eta_seconds = elapsed / len(done) * (len(subjects) * len(seeds) - len(done))
    payload = {
        "cache_done": cache_done,
        "cache_total": len(subjects),
        "train_done": len(done),
        "train_failed": len(failed),
        "train_running_or_partial": len(running),
        "train_total": len(subjects) * len(seeds),
        "train_process_count": train_process_count,
        "available_ram_gb": round(available_ram_gb, 2),
        "eta_seconds_from_completed_rate": eta_seconds,
        "gpu_index_used_free_util_percent": gpu,
        "recent_done": done[-5:],
        "failed": failed,
        "running_or_partial": running,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
