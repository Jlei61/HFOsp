#!/usr/bin/env python3
"""Expand the complete v0.3.5 chain to the registered second-wave cohort.

The supervisor first waits for all immutable human-input prefixes, builds the
recorded-time decoder caches, and waits for the source-attribution queue to
release the GPUs. It then runs every registered patient/seed through decoder
training, q(t), step-wise adaptation, full event-content state and W4--W6.
Failures are isolated to one unit and CUDA OOM retries reduce only batch/chunk
size; no scientific hyperparameter is changed by the supervisor.
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
    DECODER_ROOT,
    INPUT_ROOT,
    LOCKED_SEEDS,
    OUTPUT_ROOT,
    V035_COHORT_EXPANSION_SUBJECTS,
    V035_DECODER_FITS,
    atomic_json,
)

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
ARM = "L3_LOCAL_PLUS_LEARNED_LR"
STAGES = ("decoder", "dynamic_rate", "stepwise", "full_mark", "downstream")


def output_for(job: dict, stage: str) -> Path:
    subject = job["subject"]
    tag = f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"
    if stage == "decoder":
        return (
            DECODER_ROOT / "formal_units" / job["fit"] / ARM
            / f"seed{job['decoder_seed']}" / "DONE.json"
        )
    if stage == "dynamic_rate":
        return OUTPUT_ROOT / "dynamic_rate" / subject / f"seed{job['state_seed']}" / "card.json"
    if stage == "stepwise":
        return OUTPUT_ROOT / "stepwise_decoder" / subject / tag / "card.json"
    if stage == "full_mark":
        return OUTPUT_ROOT / "full_mark_final" / subject / tag / "card.json"
    return OUTPUT_ROOT / "final_downstream" / subject / tag / "card.json"


def command(job: dict, stage: str, gpu: str, full_config: Path) -> list[str]:
    subject = job["subject"]
    decoder_seed = str(job["decoder_seed"])
    state_seed = str(job["state_seed"])
    common = [
        "--subject", subject, "--decoder-seed", decoder_seed,
        "--state-seed", state_seed,
    ]
    if stage == "decoder":
        return [
            str(PYTHON), str(ROOT / "scripts/train_topic5_lbss_unit_v0_2.py"),
            "--fit-id", job["fit"], "--arm", ARM, "--seed", decoder_seed,
            "--out-root", str(DECODER_ROOT), "--unit-root-name", "formal_units",
            "--contract-label", "group_event_state_v035_recorded_time_decoder",
            "--device", f"cuda:{gpu}",
        ]
    if stage == "dynamic_rate":
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"),
            "--subject", subject, "--seed", state_seed,
            "--config-json", str(ROOT / "config/group_event_state_v035_rate_search/low_lr.json"),
            "--device", f"cuda:{gpu}",
        ]
    if stage == "stepwise":
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_stepwise_decoder.py"),
            *common, "--device", f"cuda:{gpu}",
        ]
    if stage == "full_mark":
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"),
            *common, "--config-json", str(full_config),
            "--chunk-events", str(job["chunk_events"]),
            "--out-root", str(OUTPUT_ROOT / "full_mark_final"),
            "--device", f"cuda:{gpu}",
        ]
    return [
        str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_final_downstream.py"),
        *common, "--batch-events", str(job["batch_events"]),
        "--device", f"cuda:{gpu}",
    ]


def wait_for_inputs(root: Path) -> None:
    required = [INPUT_ROOT / subject / "manifest_v3.json" for subject in V035_COHORT_EXPANSION_SUBJECTS]
    while True:
        missing = [str(path) for path in required if not path.is_file()]
        atomic_json(root / "input_wait.json", {
            "format": "group_event_state_v0_3_5_expansion_input_wait_v1",
            "status": "READY" if not missing else "WAITING_FOR_IMMUTABLE_INPUTS",
            "missing": missing, "updated_epoch": time.time(),
        })
        if not missing:
            return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--wait-for-gpu-owner", type=Path,
        default=OUTPUT_ROOT / "source_attribution" / "supervisor" / "queue_done.json",
    )
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    slots = [(f"{gpu}:{worker}", gpu) for gpu in gpus for worker in range(args.workers_per_gpu)]
    root = OUTPUT_ROOT / "cohort_expansion" / "supervisor"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    wait_for_inputs(root)
    cache_manifest = DECODER_ROOT / "V035_EXPANSION_INPUT_CACHE_MANIFEST.json"
    if not cache_manifest.is_file():
        build_log = root / "decoder_cache_build.log"
        with build_log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(PYTHON), str(ROOT / "scripts/build_group_event_state_v035_decoder_expansion_cache.py")],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0 or not cache_manifest.is_file():
            raise RuntimeError(f"decoder expansion cache build failed; see {build_log}")

    while not args.wait_for_gpu_owner.is_file():
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_cohort_expansion_queue_v1",
            "status": "WAITING_FOR_SOURCE_ATTRIBUTION_TO_RELEASE_GPUS",
            "wait_for": str(args.wait_for_gpu_owner), "updated_epoch": time.time(),
        })
        time.sleep(30)

    budget_path = OUTPUT_ROOT / "full_mark_search_budget_extension" / "budget_audit.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    full_config = Path(budget["final_config"])
    if not full_config.is_file():
        raise FileNotFoundError(full_config)

    jobs = [
        {
            "subject": subject, "fit": V035_DECODER_FITS[subject],
            "decoder_seed": decoder_seed, "state_seed": state_seed,
            "stage_index": 0, "chunk_events": 256, "batch_events": 96,
            "retries": 0,
        }
        for subject in V035_COHORT_EXPANSION_SUBJECTS
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3])
    ]
    pending, running, complete, failed = jobs, {}, [], []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    atomic_json(root / "RUN_CONTRACT.json", {
        "format": "group_event_state_v0_3_5_cohort_expansion_contract_v1",
        "subjects": list(V035_COHORT_EXPANSION_SUBJECTS),
        "seeds": list(LOCKED_SEEDS[:3]), "stages": list(STAGES),
        "decoder_cache_manifest": str(cache_manifest),
        "full_mark_config": str(full_config),
        "selection_policy": "same locked configs and chronological INNER/SELECTION contract as first wave",
        "development_targets_read": False, "sealed_partition_opened": False,
    })

    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job, stage = row["job"], row["stage"]
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-30000:]
            if code == 0 and output_for(job, stage).is_file():
                job["stage_index"] += 1
                job["retries"] = 0
                if job["stage_index"] == len(STAGES):
                    complete.append(job)
                else:
                    pending.append(job)
            elif "out of memory" in body.lower() and job["retries"] < 3:
                if stage == "full_mark":
                    job["chunk_events"] = max(24, int(job["chunk_events"]) // 2)
                elif stage == "downstream":
                    job["batch_events"] = max(12, int(job["batch_events"]) // 2)
                else:
                    failed.append({**job, "failed_stage": stage, "returncode": code,
                                   "log": row["log"], "tail": body[-4000:]})
                    del running[slot]
                    continue
                job["retries"] += 1
                pending.insert(0, job)
            else:
                failed.append({**job, "failed_stage": stage, "returncode": code,
                               "log": row["log"], "tail": body[-4000:]})
            del running[slot]

        for slot, gpu in slots:
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            while job["stage_index"] < len(STAGES) and output_for(job, STAGES[job["stage_index"]]).is_file():
                job["stage_index"] += 1
            if job["stage_index"] == len(STAGES):
                complete.append(job)
                continue
            stage = STAGES[job["stage_index"]]
            log = logs / (
                f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}"
                f"_{stage}_gpu{gpu}_slot{slot.replace(':', '_')}.log"
            )
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command(job, stage, gpu, full_config), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            running[slot] = {
                "process": process, "handle": handle, "job": job,
                "stage": stage, "log": str(log), "started": time.time(), "gpu": gpu,
            }

        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_cohort_expansion_queue_v1",
            "status": "RUNNING", "updated_epoch": time.time(),
            "workers_per_gpu": args.workers_per_gpu,
            "pending": len(pending), "complete": len(complete), "failed": failed,
            "running": {
                slot: {
                    "pid": row["process"].pid, "gpu": row["gpu"],
                    "stage": row["stage"], "job": row["job"],
                    "elapsed_seconds": time.time() - row["started"], "log": row["log"],
                }
                for slot, row in running.items()
            },
        })
        if pending or running:
            time.sleep(15)

    atomic_json(root / "queue_done.json", {
        "format": "group_event_state_v0_3_5_cohort_expansion_done_v1",
        "status": "COMPLETE" if not failed else "PARTIAL",
        "complete": complete, "failed": failed, "all_registered_stages": list(STAGES),
    })


if __name__ == "__main__":
    main()
