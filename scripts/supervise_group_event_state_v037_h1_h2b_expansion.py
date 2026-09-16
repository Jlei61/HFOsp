#!/usr/bin/env python3
"""Train the pre-registered H1 producers for E916, the H2b-informative subject."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
OUT = Path("/data/hfosp_group_event_state_v0_3_7/h1_h2b_expansion")
SUBJECT = "epilepsiae_916"
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)
FAMILIES = {
    "event": (ROOT / "scripts/run_group_event_state_v037_h1.py",
              Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon")),
    "dual": (ROOT / "scripts/run_group_event_state_v037_h1_dual.py",
             Path("/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete")),
}


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    logs = OUT / "logs"; logs.mkdir(parents=True, exist_ok=True)
    jobs = [(family, seed) for seed in SEEDS for family in FAMILIES]
    pending = [(family, seed) for family, seed in jobs if not (
        FAMILIES[family][1] / SUBJECT / f"seed{seed}" / "card.json"
    ).exists()]
    running = {}; failures = []
    while pending or running:
        for gpu in list(running):
            process, family, seed, handle = running[gpu]
            code = process.poll()
            if code is None: continue
            handle.close(); del running[gpu]
            if code != 0 or not (FAMILIES[family][1] / SUBJECT / f"seed{seed}" / "card.json").exists():
                failures.append({"family": family, "seed": seed, "returncode": code, "gpu": gpu})
        for gpu in (0, 1):
            if gpu in running or not pending: continue
            family, seed = pending.pop(0); script, root = FAMILIES[family]
            handle = (logs / f"{family}__seed{seed}.log").open("a", encoding="utf-8")
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env.setdefault("OMP_NUM_THREADS", "2")
            process = subprocess.Popen(
                [str(PYTHON), str(script), "--subject", SUBJECT, "--seed", str(seed),
                 "--device", "cuda:0", "--out-root", str(root)],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, family, seed, handle)
        complete = sum((FAMILIES[f][1] / SUBJECT / f"seed{z}" / "card.json").exists() for f, z in jobs)
        write(OUT / "queue_status.json", {
            "format": "group_event_state_v0_3_7_h1_h2b_expansion_queue_v1",
            "status": "RUNNING", "subject": SUBJECT, "total": len(jobs),
            "complete": complete, "pending": len(pending),
            "running": [{"family": f, "seed": z, "pid": p.pid, "gpu": gpu}
                        for gpu, (p, f, z, _h) in sorted(running.items())],
            "failures": failures, "selection_reason": "adequate FIT seizure count for frozen H2b; not selected by H1 effect",
            "development_targets_read": False, "seizure_targets_read_for_model_selection": False,
            "sealed_partition_opened": False,
        })
        time.sleep(20)
    write(OUT / "queue_status.json", {
        "format": "group_event_state_v0_3_7_h1_h2b_expansion_queue_v1",
        "status": "FAILED" if failures else "COMPLETE", "subject": SUBJECT,
        "total": len(jobs), "complete": len(jobs) - len(failures), "pending": 0,
        "running": [], "failures": failures,
        "selection_reason": "adequate FIT seizure count for frozen H2b; not selected by H1 effect",
        "development_targets_read": False, "seizure_targets_read_for_model_selection": False,
        "sealed_partition_opened": False,
    })
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__": main()

