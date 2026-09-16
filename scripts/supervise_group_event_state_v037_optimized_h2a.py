#!/usr/bin/env python3
"""Frozen-decoder H2a for all three optimized observer families."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RUNNER = ROOT / "scripts/run_group_event_state_v037_h2a.py"
MODELS = ("event", "grid", "dual")
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_optimized"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_optimized"))
    parser.add_argument("--decoder-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/decoder_strict"))
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    upstream = args.h1_root / "supervisor/queue_status.json"
    if not upstream.exists() or json.loads(upstream.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("optimized H1 must be complete before frozen-transfer H2a")
    gpus = tuple(int(v) for v in args.gpus.split(",") if v.strip())
    slots = tuple(gpu for gpu in gpus for _ in range(int(args.workers_per_gpu)))
    subjects = tuple(dict.fromkeys(args.subjects))
    models = tuple(dict.fromkeys(args.models))
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    jobs = [(model, subject, seed) for model in models for seed in seeds for subject in subjects]
    pending = [job for job in jobs if not (
        args.out_root / job[0] / job[1] / f"seed{job[2]}" / "card.json"
    ).exists()]
    running = {}; failures = []; logs = args.out_root / "supervisor/logs"; logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for slot in list(running):
            process, job, handle = running[slot]
            code = process.poll()
            if code is None: continue
            handle.close(); del running[slot]
            expected = args.out_root / job[0] / job[1] / f"seed{job[2]}" / "card.json"
            if code != 0 or not expected.exists(): failures.append({"job": job, "returncode": code})
        for slot, gpu in enumerate(slots):
            if slot in running or not pending: continue
            model, subject, seed = pending.pop(0)
            handle = (logs / f"{model}__{subject}__seed{seed}.log").open("a", encoding="utf-8")
            cmd = [str(PYTHON), str(RUNNER), "--subject", subject, "--seed", str(seed),
                   "--device", f"cuda:{gpu}", "--h1-root", str(args.h1_root / model),
                   "--decoder-root", str(args.decoder_root),
                   "--state-family", model, "--out-root", str(args.out_root / model)]
            env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "2")
            running[slot] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT), (model, subject, seed), handle)
        _write(args.out_root / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_7_optimized_h2a_queue_v1",
            "status": "FAILED" if failures else ("COMPLETE" if not pending and not running else "RUNNING"),
            "total": len(jobs), "complete": len(jobs)-len(pending)-len(running)-len(failures),
            "pending": len(pending),
            "running": [{"slot": slot, "gpu": slots[slot], "pid": item[0].pid, "job": item[1]}
                        for slot, item in sorted(running.items())],
            "failures": failures,
            "subjects": list(subjects), "models": list(models), "seeds": list(seeds),
            "comparisons": ["prefix/static", "B_mark", "correct state", "same-prefix shifted state", "future oracle"],
            "structural_diagnostics": ["empirical-support interpolation", "local decoder Jacobian"],
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        if failures:
            for process, _job, handle in running.values(): process.terminate(); handle.close()
            raise SystemExit(1)
        time.sleep(15.0)


if __name__ == "__main__": main()
