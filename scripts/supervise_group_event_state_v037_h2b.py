#!/usr/bin/env python3
"""Two-phase H2b supervisor: freeze every feature first, then open outcomes."""

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
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _complete(root: Path, relative: str) -> bool:
    return all((root / relative / subject / f"seed{seed}").exists()
               for subject in SUBJECTS for seed in SEEDS)


def _upstream_complete(root: Path) -> bool:
    status = root / "supervisor/queue_status.json"
    if not status.exists():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except json.JSONDecodeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete"))
    parser.add_argument("--event-h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"))
    parser.add_argument("--grid-h1-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2b_frozen_transfer"))
    args = parser.parse_args()
    supervisor = args.out_root / "supervisor"; logs = supervisor / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(subject, seed) for seed in SEEDS for subject in SUBJECTS]
    phase = "WAITING_FOR_INTERICTAL_FREEZE_SOURCES"; running = {}; failures = []
    pending = list(jobs)
    while True:
        for slot in list(running):
            process, subject, seed, handle = running[slot]
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[slot]
            expected = (
                args.out_root / "features" / subject / f"seed{seed}" / "freeze_card.json"
                if phase == "FREEZING_INTERICTAL_FEATURES" else
                args.out_root / "outcomes" / subject / f"seed{seed}" / "card.json"
            )
            if code != 0 or not expected.exists():
                failures.append({"phase": phase, "subject": subject, "seed": seed, "returncode": code})
        upstream = _upstream_complete(args.h1_root) and _upstream_complete(args.event_h1_root)
        if phase == "WAITING_FOR_INTERICTAL_FREEZE_SOURCES" and upstream:
            phase = "FREEZING_INTERICTAL_FEATURES"
            pending = [job for job in jobs if not (
                args.out_root / "features" / job[0] / f"seed{job[1]}" / "freeze_card.json"
            ).exists()]
        if phase == "FREEZING_INTERICTAL_FEATURES" and not pending and not running:
            if failures:
                phase = "FAILED"
            else:
                marker = supervisor / "ALL_FEATURES_FROZEN"
                marker.write_text("All interictal features were frozen before seizure outcomes were opened.\n", encoding="utf-8")
                phase = "FITTING_OUTCOMES"
                pending = [job for job in jobs if not (
                    args.out_root / "outcomes" / job[0] / f"seed{job[1]}" / "card.json"
                ).exists()]
        if phase == "FITTING_OUTCOMES" and not (supervisor / "ALL_FEATURES_FROZEN").exists():
            raise RuntimeError("refusing to open seizure outcomes before all feature files are frozen")
        if phase in {"FREEZING_INTERICTAL_FEATURES", "FITTING_OUTCOMES"}:
            for slot in range(args.workers):
                if slot in running or not pending:
                    continue
                subject, seed = pending.pop(0)
                script = FREEZE if phase == "FREEZING_INTERICTAL_FEATURES" else OUTCOME
                cmd = [str(PYTHON), str(script), "--subject", subject, "--seed", str(seed),
                       "--out-root", str(args.out_root)]
                if phase == "FREEZING_INTERICTAL_FEATURES":
                    cmd += ["--h1-root", str(args.h1_root), "--event-h1-root", str(args.event_h1_root)]
                    if args.grid_h1_root is not None:
                        cmd += ["--grid-h1-root", str(args.grid_h1_root)]
                handle = (logs / f"{phase.lower()}__{subject}__seed{seed}.log").open("a", encoding="utf-8")
                env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "2")
                running[slot] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                                   stderr=subprocess.STDOUT), subject, seed, handle)
        if phase == "FITTING_OUTCOMES" and not pending and not running:
            phase = "FAILED" if failures else "COMPLETE"
        complete_freeze = sum((args.out_root / "features" / s / f"seed{z}" / "freeze_card.json").exists()
                              for s, z in jobs)
        complete_outcomes = sum((args.out_root / "outcomes" / s / f"seed{z}" / "card.json").exists()
                                for s, z in jobs)
        _write(supervisor / "queue_status.json", {
            "format": "group_event_state_v0_3_7_h2b_two_phase_queue_v1",
            "status": phase, "total": len(jobs), "features_frozen": complete_freeze,
            "outcomes_complete": complete_outcomes, "pending": len(pending),
            "running": [{"pid": p.pid, "subject": s, "seed": z, "slot": slot}
                        for slot, (p, s, z, _h) in sorted(running.items())],
            "failures": failures,
            "all_features_frozen_before_outcomes": (supervisor / "ALL_FEATURES_FROZEN").exists(),
            "development_targets_read": False, "sealed_partition_opened": False,
            "seizure_outcomes_read": complete_outcomes > 0,
        })
        if phase in {"COMPLETE", "FAILED"}:
            raise SystemExit(0 if phase == "COMPLETE" else 1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
