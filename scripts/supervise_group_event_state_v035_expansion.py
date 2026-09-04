#!/usr/bin/env python3
"""Expand v0.3.5 through decoder, W2--W6 for the registered second-wave patients.

The queue waits for the first-wave per-step auxiliary jobs to release the two
GPUs, then runs every subject/seed through the same ordered scientific stack.
One unit failing never cancels other patients or seeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS, OUTPUT_ROOT, atomic_json  # noqa: E402

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
DECODER_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
ARM = "L3_LOCAL_PLUS_LEARNED_LR"
FITS = {
    "epilepsiae_1096": "epilepsiae_1096__own_a",
    "epilepsiae_384": "epilepsiae_384__shared",
    "epilepsiae_1125": "epilepsiae_1125__own_a",
}
STAGES = ("decoder", "stepwise", "full_mark", "functional", "auxiliary", "seizure", "feedback")


def output_for(job: dict, stage: str) -> Path:
    subject, decoder_seed, state_seed = job["subject"], job["decoder_seed"], job["state_seed"]
    unit = f"decoder_seed{decoder_seed}_state_seed{state_seed}"
    if stage == "decoder":
        return DECODER_ROOT / "formal_units" / job["fit"] / ARM / f"seed{decoder_seed}" / "DONE.json"
    names = {
        "stepwise": "stepwise_decoder", "full_mark": "full_mark_state",
        "functional": "functional_readouts", "auxiliary": "stepwise_auxiliary",
        "seizure": "seizure_transfer", "feedback": "feedback_models",
    }
    return OUTPUT_ROOT / names[stage] / subject / unit / "card.json"


def command(job: dict, stage: str, gpu: str) -> list[str]:
    subject, decoder_seed, state_seed = job["subject"], job["decoder_seed"], job["state_seed"]
    common = ["--subject", subject, "--decoder-seed", str(decoder_seed), "--state-seed", str(state_seed)]
    if stage == "decoder":
        return [str(PYTHON), str(ROOT / "scripts/train_topic5_lbss_unit_v0_2.py"),
                "--fit-id", job["fit"], "--arm", ARM, "--seed", str(decoder_seed),
                "--out-root", str(DECODER_ROOT), "--unit-root-name", "formal_units",
                "--contract-label", "group_event_state_v035_time_split_decoder",
                "--device", f"cuda:{gpu}"]
    scripts = {
        "stepwise": "run_group_event_state_v035_stepwise_decoder.py",
        "full_mark": "run_group_event_state_v035_full_mark_state.py",
        "functional": "run_group_event_state_v035_functional_readouts.py",
        "auxiliary": "run_group_event_state_v035_stepwise_auxiliary.py",
        "seizure": "run_group_event_state_v035_seizure_transfer.py",
        "feedback": "run_group_event_state_v035_feedback_models.py",
    }
    cmd = [str(PYTHON), str(ROOT / f"scripts/{scripts[stage]}"), *common]
    if stage in {"stepwise", "full_mark", "functional", "auxiliary"}:
        cmd += ["--device", f"cuda:{gpu}"]
    if stage == "full_mark":
        cmd += ["--chunk-events", str(job["chunk_events"])]
    if stage == "auxiliary":
        cmd += ["--batch-events", str(job["batch_events"])]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--wait-for", type=Path,
                    default=OUTPUT_ROOT / "stepwise_auxiliary_supervisor" / "queue_done.json")
    args = ap.parse_args()
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    root = OUTPUT_ROOT / "expansion_supervisor"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    while not args.wait_for.exists():
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_expansion_queue_v1", "status": "WAITING_FOR_GPU_OWNER",
            "wait_for": str(args.wait_for), "updated_epoch": time.time(),
        })
        time.sleep(30)
    pending = []
    for subject, fit in FITS.items():
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            pending.append({"subject": subject, "fit": fit, "decoder_seed": decoder_seed,
                            "state_seed": state_seed, "stage_index": 0, "chunk_events": 256,
                            "batch_events": 96, "retries": 0})
    running: dict[str, dict] = {}
    complete, failed = [], []
    while pending or running:
        for gpu, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job, stage = row["job"], row["stage"]
            out = output_for(job, stage)
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-30000:]
            if code == 0 and out.exists():
                job["stage_index"] += 1
                job["retries"] = 0
                if job["stage_index"] == len(STAGES):
                    complete.append(job)
                else:
                    pending.append(job)
            elif "out of memory" in body.lower() and job["retries"] < 3 and stage in {"full_mark", "auxiliary"}:
                key = "chunk_events" if stage == "full_mark" else "batch_events"
                job[key] = max(12, job[key] // 2)
                job["retries"] += 1
                pending.insert(0, job)
            else:
                failed.append({**job, "failed_stage": stage, "returncode": code,
                               "log": row["log"], "tail": body[-4000:]})
            del running[gpu]
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            # Skip materialised stages from prior safe retries/restarts.
            while job["stage_index"] < len(STAGES) and output_for(job, STAGES[job["stage_index"]]).exists():
                job["stage_index"] += 1
            if job["stage_index"] == len(STAGES):
                complete.append(job)
                continue
            stage = STAGES[job["stage_index"]]
            log = logs / f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_{stage}_gpu{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(command(job, stage, gpu), cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            running[gpu] = {"job": job, "stage": stage, "process": process, "handle": handle,
                            "log": str(log), "started": time.time()}
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_expansion_queue_v1", "status": "RUNNING",
            "updated_epoch": time.time(), "pending": len(pending), "complete": len(complete),
            "failed": failed,
            "running": {gpu: {"pid": row["process"].pid, "stage": row["stage"],
                              "job": row["job"], "log": row["log"],
                              "elapsed_seconds": time.time() - row["started"]}
                        for gpu, row in running.items()},
        })
        if pending or running:
            time.sleep(15)
    atomic_json(root / "queue_done.json", {
        "format": "group_event_state_v0_3_5_expansion_done_v1", "complete": complete,
        "failed": failed, "all_registered_stages": list(STAGES),
    })


if __name__ == "__main__":
    main()
