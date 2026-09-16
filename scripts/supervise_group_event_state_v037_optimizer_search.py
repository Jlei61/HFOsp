#!/usr/bin/env python3
"""Two-stage v0.3.7 search: freeze controls, then search learned models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_group_event_state_v037_optimizer_search import BASELINE_RECIPES, RECIPES


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RUNNER = ROOT / "scripts/run_group_event_state_v037_optimizer_search.py"
BASELINE_FINALIZER = ROOT / "scripts/finalize_group_event_state_v037_baseline_search.py"
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904)
MODELS = ("event", "grid", "dual")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_jobs(
    jobs: list[tuple[str, str, str, int]], *, out_root: Path,
    slots: tuple[int, ...], poll_seconds: float, phase: str,
    baseline_recipe: str | None,
) -> list[dict]:
    pending = [job for job in jobs if not (
        out_root / job[0] / job[1] / job[2] / f"seed{job[3]}" / "card.json"
    ).exists()]
    running: dict[int, tuple[subprocess.Popen, tuple, object]] = {}
    failures: list[dict] = []
    logs = out_root / "supervisor/logs"; logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for slot in list(running):
            process, job, handle = running[slot]
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[slot]
            expected = out_root / job[0] / job[1] / job[2] / f"seed{job[3]}" / "card.json"
            if code != 0 or not expected.exists():
                failures.append({"phase": phase, "job": job, "returncode": code})
        for slot, gpu in enumerate(slots):
            if slot in running or not pending:
                continue
            model, recipe, subject, seed = pending.pop(0)
            handle = (logs / f"{model}__{recipe}__{subject}__seed{seed}.log").open("a", encoding="utf-8")
            command = [
                str(PYTHON), str(RUNNER), "--model", model, "--recipe", recipe,
                "--subject", subject, "--seed", str(seed), "--device", f"cuda:{gpu}",
                "--out-root", str(out_root),
            ]
            if baseline_recipe is not None:
                command.extend(("--baseline-recipe", baseline_recipe))
            environment = dict(os.environ); environment.setdefault("OMP_NUM_THREADS", "2")
            running[slot] = (
                subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                 stderr=subprocess.STDOUT),
                (model, recipe, subject, seed), handle,
            )
        completed = len(jobs) - len(pending) - len(running) - len(failures)
        _write(out_root / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_7_optimizer_search_queue_v3",
            "status": "FAILED" if failures else "RUNNING",
            "phase": phase, "phase_total": len(jobs), "phase_complete": completed,
            "phase_pending": len(pending),
            "running": [
                {"slot": slot, "gpu": slots[slot], "pid": item[0].pid, "job": item[1]}
                for slot, item in sorted(running.items())
            ],
            "failures": failures, "frozen_baseline_recipe": baseline_recipe,
            "control_and_model_families_share_trainability_gate": True,
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        if failures:
            for process, _job, handle in running.values():
                process.terminate(); handle.close()
            break
        time.sleep(poll_seconds)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/optimizer_search"))
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    slots = tuple(gpu for gpu in gpus for _ in range(int(args.workers_per_gpu)))
    baseline_jobs = [
        ("bmark", recipe, subject, seed)
        for recipe in BASELINE_RECIPES for subject in SUBJECTS for seed in SEEDS
    ]
    failures = _run_jobs(
        baseline_jobs, out_root=args.out_root, slots=slots,
        poll_seconds=args.poll_seconds, phase="CONTROL_SEARCH", baseline_recipe=None,
    )
    if failures:
        raise SystemExit(1)
    subprocess.run([
        str(PYTHON), str(BASELINE_FINALIZER), "--root", str(args.out_root)
    ], cwd=ROOT, check=True)
    baseline = json.loads((args.out_root / "baseline_summary.json").read_text(encoding="utf-8"))
    selected = baseline.get("selected_recipe")
    if baseline.get("status") != "TRAINABLE_BASELINE_RECIPE_SELECTED" or selected is None:
        _write(args.out_root / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_7_optimizer_search_queue_v3",
            "status": "STOPPED_AT_CONTROL_GATE", "phase": "CONTROL_SEARCH_COMPLETE",
            "frozen_baseline_recipe": None, "failures": [],
            "baseline_summary": str(args.out_root / "baseline_summary.json"),
            "model_search_started": False,
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        return
    model_jobs = [
        (model, recipe, subject, seed)
        for model in MODELS for recipe in RECIPES for subject in SUBJECTS for seed in SEEDS
    ]
    failures = _run_jobs(
        model_jobs, out_root=args.out_root, slots=slots,
        poll_seconds=args.poll_seconds, phase="MODEL_SEARCH_ON_FROZEN_CONTROL",
        baseline_recipe=str(selected),
    )
    if failures:
        raise SystemExit(1)
    _write(args.out_root / "supervisor/queue_status.json", {
        "format": "group_event_state_v0_3_7_optimizer_search_queue_v3",
        "status": "COMPLETE", "phase": "COMPLETE",
        "total": len(baseline_jobs) + len(model_jobs),
        "complete": len(baseline_jobs) + len(model_jobs),
        "pending": 0, "running": [], "failures": [],
        "frozen_baseline_recipe": selected,
        "baseline_frozen_before_model_search": True,
        "control_and_model_families_share_trainability_gate": True,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })


if __name__ == "__main__":
    main()
