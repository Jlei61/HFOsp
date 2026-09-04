#!/usr/bin/env python3
"""Persistent source-attribution queue for the triggered v0.3.5 mark signal.

Every arm keeps event times, q(t), targets, state bank, frozen decoder and the
INNER-selected optimiser recipe fixed.  Only the information carried by an
observed group-event update changes.  Each state unit is immediately followed
by the same frozen functional future-block readout.
"""

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

from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    LOCKED_SEEDS, OUTPUT_ROOT, atomic_json,
)

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECTS = (
    "epilepsiae_253", "epilepsiae_548", "epilepsiae_583",
    "epilepsiae_1096", "epilepsiae_384", "epilepsiae_1125",
    "epilepsiae_1146",
)
VIEWS = (
    "times_only", "spatial_only", "waveform_only", "multiband_only",
    "mark_shuffle",
)


def _unit(root: Path, job: dict) -> Path:
    tag = f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"
    return root / job["view"] / "full_mark" / job["subject"] / tag


def _output(root: Path, job: dict) -> Path:
    tag = f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"
    if job["stage"] == "state":
        return _unit(root, job) / "card.json"
    return root / job["view"] / "functional_readouts" / job["subject"] / tag / "card.json"


def _command(root: Path, config: Path, job: dict, gpu: str) -> list[str]:
    common = [
        "--subject", job["subject"], "--decoder-seed", str(job["decoder_seed"]),
        "--state-seed", str(job["state_seed"]), "--device", f"cuda:{gpu}",
    ]
    if job["stage"] == "state":
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"),
            *common, "--config-json", str(config), "--input-view", job["view"],
            "--chunk-events", str(job["chunk_events"]),
            "--out-root", str(root / job["view"] / "full_mark"),
        ]
    tag = f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"
    return [
        str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_functional_readouts.py"),
        *common, "--state-unit", str(_unit(root, job)),
        "--out-dir", str(root / job["view"] / "functional_readouts" / job["subject"] / tag),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--root", type=Path,
        default=OUTPUT_ROOT / "source_attribution",
    )
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    slots = [(f"{gpu}:{i}", gpu) for gpu in gpus for i in range(args.workers_per_gpu)]

    budget_path = OUTPUT_ROOT / "full_mark_search_budget_extension" / "budget_audit.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    config = Path(budget["final_config"])
    if budget.get("status") != "ORIGINAL_BUDGET_ADEQUATE" or not config.is_file():
        raise RuntimeError("locked full-mark training recipe/budget is unavailable")

    args.root.mkdir(parents=True, exist_ok=True)
    logs = args.root / "supervisor" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "stage": "state", "view": view, "subject": subject,
            "decoder_seed": decoder_seed, "state_seed": state_seed,
            "chunk_events": 256, "retries": 0,
        }
        for view in VIEWS
        for subject in SUBJECTS
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3])
    ]
    pending, running, complete, failed = jobs, {}, [], []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    atomic_json(args.root / "RUN_CONTRACT.json", {
        "format": "group_event_state_v0_3_5_source_attribution_contract_v1",
        "subjects": list(SUBJECTS), "views": list(VIEWS),
        "seeds": list(LOCKED_SEEDS[:3]), "locked_config": str(config),
        "reference_full_mark_root": str(OUTPUT_ROOT / "full_mark_final"),
        "estimand": "incremental value of observed group-event content beyond exact event times and q(t)",
        "mark_shuffle": "whole payload shifted within coverage-segment and phase; event times and targets fixed",
        "selection_policy": "checkpoint selected on chronological INNER; SELECTION reported once",
        "development_targets_read": False, "sealed_partition_opened": False,
    })

    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job = row["job"]
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-30000:]
            if code == 0 and _output(args.root, job).is_file():
                if job["stage"] == "state":
                    pending.append({**job, "stage": "functional", "retries": 0})
                else:
                    complete.append(job)
            elif "out of memory" in body.lower() and job["stage"] == "state" and job["retries"] < 3:
                job["chunk_events"] = max(24, int(job["chunk_events"]) // 2)
                job["retries"] += 1
                pending.insert(0, job)
            else:
                failed.append({
                    **job, "returncode": code, "log": row["log"], "tail": body[-4000:],
                })
            del running[slot]

        for slot, gpu in slots:
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            if _output(args.root, job).is_file():
                if job["stage"] == "state":
                    pending.append({**job, "stage": "functional", "retries": 0})
                else:
                    complete.append(job)
                continue
            log = logs / (
                f"{job['view']}_{job['stage']}_{job['subject']}_decoder{job['decoder_seed']}"
                f"_state{job['state_seed']}_gpu{gpu}_slot{slot.replace(':', '_')}.log"
            )
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                _command(args.root, config, job, gpu), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            running[slot] = {
                "process": process, "handle": handle, "job": job,
                "log": str(log), "gpu": gpu, "started": time.time(),
            }

        atomic_json(args.root / "supervisor" / "queue_state.json", {
            "format": "group_event_state_v0_3_5_source_attribution_queue_v1",
            "status": "RUNNING", "updated_epoch": time.time(),
            "workers_per_gpu": args.workers_per_gpu, "pending": len(pending),
            "complete": len(complete), "failed": failed,
            "running": {
                slot: {
                    "pid": row["process"].pid, "gpu": row["gpu"],
                    "job": row["job"], "log": row["log"],
                    "elapsed_seconds": time.time() - row["started"],
                }
                for slot, row in running.items()
            },
        })
        if pending or running:
            time.sleep(15)

    atomic_json(args.root / "supervisor" / "queue_done.json", {
        "format": "group_event_state_v0_3_5_source_attribution_done_v1",
        "status": "COMPLETE" if not failed else "PARTIAL",
        "complete": complete, "failed": failed,
    })


if __name__ == "__main__":
    main()
