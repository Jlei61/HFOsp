#!/usr/bin/env python3
"""Persistent CPU queue for the full v0.3.5 dynamic-rate recipe search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS, OUTPUT_ROOT, V035_SUBJECTS, atomic_json  # noqa: E402

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RECIPES = tuple(sorted((ROOT / "config/group_event_state_v035_rate_search").glob("*.json")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    control = OUTPUT_ROOT / "dynamic_rate_search_supervisor"
    logs = control / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = []
    for recipe in RECIPES:
        for subject in V035_SUBJECTS:
            for seed in LOCKED_SEEDS[:3]:
                out = OUTPUT_ROOT / "dynamic_rate_search" / recipe.stem / subject / f"seed{seed}" / "card.json"
                pending.append({"recipe": recipe.stem, "config": str(recipe), "subject": subject, "seed": seed, "out": str(out)})
    complete, failed, running = [], [], {}
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job = row["job"]
            tail = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-5000:]
            if code == 0 and Path(job["out"]).exists():
                complete.append(job)
            else:
                failed.append({**job, "returncode": code, "log": row["log"], "tail": tail[-2000:]})
            del running[slot]
        for slot in range(args.workers):
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            if Path(job["out"]).exists():
                complete.append(job)
                continue
            out_root = OUTPUT_ROOT / "dynamic_rate_search" / job["recipe"]
            log = logs / f"{job['recipe']}_{job['subject']}_{job['seed']}.log"
            handle = log.open("a", encoding="utf-8")
            cmd = [
                str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"),
                "--subject", job["subject"], "--seed", str(job["seed"]),
                "--device", "cpu", "--config-json", job["config"], "--hold-selection",
                "--out-root", str(out_root),
            ]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            running[slot] = {"process": process, "handle": handle, "job": job, "log": str(log), "started": time.time()}
        atomic_json(control / "queue_state.json", {
            "format": "group_event_state_v0_3_5_dynamic_rate_search_queue_v1",
            "status": "RUNNING", "updated_epoch": time.time(),
            "planned": len(RECIPES) * len(V035_SUBJECTS) * 3,
            "pending": len(pending), "complete": len(complete), "failed": failed,
            "running": {str(k): {"pid": v["process"].pid, "job": v["job"], "elapsed_seconds": time.time() - v["started"]} for k, v in running.items()},
            "device": "cpu", "selection_targets_read": False,
        })
        if pending or running:
            time.sleep(5)
    if failed:
        atomic_json(control / "queue_done.json", {"status": "FAILED", "complete": complete, "failed": failed, "selection_targets_read": False})
        raise SystemExit(1)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/select_group_event_state_v035_rate_recipe.py")], cwd=ROOT, env=env, check=True)
    selection = json.loads((OUTPUT_ROOT / "dynamic_rate_search" / "selected_recipe.json").read_text(encoding="utf-8"))
    selected = selection["selected_recipe"]
    final_pending = []
    for subject in V035_SUBJECTS:
        for seed in LOCKED_SEEDS[:3]:
            out = OUTPUT_ROOT / "dynamic_rate_final" / subject / f"seed{seed}" / "card.json"
            final_pending.append({"recipe": selected, "config": str(ROOT / "config/group_event_state_v035_rate_search" / f"{selected}.json"), "subject": subject, "seed": seed, "out": str(out)})
    # Re-enter the same bounded worker pool, now scoring SELECTION exactly once.
    pending, complete, failed, running = final_pending, [], [], {}
    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close(); job = row["job"]
            tail = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-5000:]
            if code == 0 and Path(job["out"]).exists(): complete.append(job)
            else: failed.append({**job, "returncode": code, "log": row["log"], "tail": tail[-2000:]})
            del running[slot]
        for slot in range(args.workers):
            if slot in running or not pending: continue
            job = pending.pop(0)
            if Path(job["out"]).exists(): complete.append(job); continue
            log = logs / f"final_{job['subject']}_{job['seed']}.log"; handle = log.open("a", encoding="utf-8")
            cmd = [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"), "--subject", job["subject"], "--seed", str(job["seed"]), "--device", "cpu", "--config-json", job["config"], "--out-root", str(OUTPUT_ROOT / "dynamic_rate_final")]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            running[slot] = {"process": process, "handle": handle, "job": job, "log": str(log), "started": time.time()}
        atomic_json(control / "final_queue_state.json", {"format": "group_event_state_v0_3_5_dynamic_rate_final_queue_v1", "status": "RUNNING", "updated_epoch": time.time(), "selected_recipe": selected, "pending": len(pending), "complete": len(complete), "failed": failed, "running": {str(k): {"pid": v["process"].pid, "job": v["job"], "elapsed_seconds": time.time()-v["started"]} for k,v in running.items()}, "selection_targets_read": True})
        if pending or running: time.sleep(5)
    status = "DONE" if not failed else "FAILED"
    if failed: raise SystemExit(1)
    # The background increment is nested above the selected rate model, so it
    # must be refit after the rate recipe is locked rather than inherited from
    # the provisional base recipe.
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v035_background_rate.py"),
        "--workers", str(args.workers), "--three-seeds",
        "--rate-root", str(OUTPUT_ROOT / "dynamic_rate_final"),
        "--out-root", str(OUTPUT_ROOT / "background_rate_final"),
        "--control-root", str(OUTPUT_ROOT / "background_rate_final_supervisor"),
    ], cwd=ROOT, env=env, check=True)
    atomic_json(control / "queue_done.json", {"format": "group_event_state_v0_3_5_dynamic_rate_search_done_v1", "status": status, "search_complete": len(RECIPES)*len(V035_SUBJECTS)*3, "final_complete": len(complete), "background_final_complete": len(V035_SUBJECTS)*3, "failed": failed, "selected_recipe": selected, "selection_targets_read_only_after_lock": True})


if __name__ == "__main__":
    main()
