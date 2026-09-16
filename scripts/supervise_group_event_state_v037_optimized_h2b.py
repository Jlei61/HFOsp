#!/usr/bin/env python3
"""Freeze all three optimized interictal producers, then fit H2b outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
FREEZE = ROOT / "scripts/freeze_group_event_state_v037_h2b.py"
OUTCOME = ROOT / "scripts/run_group_event_state_v037_h2b.py"
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125", "epilepsiae_916")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_optimized"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2b_optimized"))
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=("both", "freeze", "outcomes"), default="both")
    parser.add_argument("--global-freeze-marker", type=Path, default=None)
    args = parser.parse_args()
    upstream = args.h1_root / "supervisor/queue_status.json"
    if not upstream.exists() or json.loads(upstream.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("optimized H1 must be complete before H2b freeze")
    subjects = tuple(dict.fromkeys(args.subjects))
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    jobs = [(subject, seed) for seed in seeds for subject in subjects]
    local_marker = args.out_root / "supervisor/ALL_FEATURES_FROZEN"
    if args.phase == "outcomes":
        if not local_marker.exists():
            raise RuntimeError("local interictal features are not globally frozen")
        if args.global_freeze_marker is None or not args.global_freeze_marker.exists():
            raise RuntimeError("all registered cohorts must be frozen before outcome access")
        phase = "OUTCOMES"
        pending = [job for job in jobs if not (
            args.out_root / "outcomes" / job[0] / f"seed{job[1]}" / "card.json"
        ).exists()]
    else:
        phase = "FREEZE"
        pending = [job for job in jobs if not (
            args.out_root / "features" / job[0] / f"seed{job[1]}" / "freeze_card.json"
        ).exists()]
    running = {}; failures = []; logs = args.out_root / "supervisor/logs"; logs.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in list(running):
            process, job, handle = running[slot]
            code = process.poll()
            if code is None: continue
            handle.close(); del running[slot]
            relative = "features" if phase == "FREEZE" else "outcomes"
            filename = "freeze_card.json" if phase == "FREEZE" else "card.json"
            expected = args.out_root / relative / job[0] / f"seed{job[1]}" / filename
            if code != 0 or not expected.exists(): failures.append({"phase": phase, "job": job, "returncode": code})
        if phase == "FREEZE" and not pending and not running:
            if failures: phase = "FAILED"
            else:
                local_marker.write_text(
                    "All event, grid and dual interictal features frozen before outcome access.\n"
                )
                if args.phase == "freeze":
                    phase = "FEATURES_FROZEN"
                else:
                    phase = "OUTCOMES"
                    pending = [job for job in jobs if not (
                        args.out_root / "outcomes" / job[0] / f"seed{job[1]}" / "card.json"
                    ).exists()]
        for slot in range(int(args.workers)):
            if slot in running or not pending or phase not in {"FREEZE", "OUTCOMES"}: continue
            subject, seed = pending.pop(0)
            handle = (logs / f"{phase.lower()}__{subject}__seed{seed}.log").open("a", encoding="utf-8")
            if phase == "FREEZE":
                cmd = [str(PYTHON), str(FREEZE), "--subject", subject, "--seed", str(seed),
                       "--h1-root", str(args.h1_root / "dual"),
                       "--event-h1-root", str(args.h1_root / "event"),
                       "--grid-h1-root", str(args.h1_root / "grid"),
                       "--out-root", str(args.out_root)]
            else:
                if not local_marker.exists():
                    raise RuntimeError("outcome access before global feature freeze")
                if args.global_freeze_marker is not None and not args.global_freeze_marker.exists():
                    raise RuntimeError("outcome access before all registered cohorts are frozen")
                cmd = [str(PYTHON), str(OUTCOME), "--subject", subject, "--seed", str(seed),
                       "--out-root", str(args.out_root)]
            env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "2")
            running[slot] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT), (subject, seed), handle)
        freeze_n = sum((args.out_root / "features" / s / f"seed{z}" / "freeze_card.json").exists() for s,z in jobs)
        outcome_n = sum((args.out_root / "outcomes" / s / f"seed{z}" / "card.json").exists() for s,z in jobs)
        status = "FAILED" if failures else (
            "FEATURES_FROZEN" if phase == "FEATURES_FROZEN" else
            ("COMPLETE" if phase == "OUTCOMES" and not pending and not running else phase)
        )
        _write(args.out_root / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_7_optimized_h2b_queue_v1",
            "status": status, "total": len(jobs), "features_frozen": freeze_n,
            "outcomes_complete": outcome_n, "pending": len(pending),
            "running": [{"slot": slot, "pid": item[0].pid, "job": item[1]}
                        for slot,item in sorted(running.items())], "failures": failures,
            "subjects": list(subjects), "seeds": list(seeds),
            "all_three_producers_frozen_before_outcome": local_marker.exists(),
            "global_freeze_marker": None if args.global_freeze_marker is None else str(args.global_freeze_marker),
            "development_targets_read": False, "sealed_partition_opened": False,
        })
        if status in {"COMPLETE", "FEATURES_FROZEN", "FAILED"}:
            raise SystemExit(0 if status != "FAILED" else 1)
        time.sleep(15.0)


if __name__ == "__main__": main()
