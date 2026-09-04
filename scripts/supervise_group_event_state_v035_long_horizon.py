#!/usr/bin/env python3
"""Run corrected long-scale H1/H2 exploration without the obsolete block rules."""

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
    DECODER_ROOT, LOCKED_SEEDS, V035_DECODER_FITS, atomic_json,
)

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
LONG_ROOT = Path("/data/hfosp_group_event_state_v0_3_5_long_observed_support")
CAUSAL_ROOT = Path("/data/hfosp_group_event_state_v0_3_5_causal")
EXPANSION_ROOT = Path("/data/hfosp_group_event_state_v0_3_5")
ARM = "L3_LOCAL_PLUS_LEARNED_LR"
HORIZON_SECONDS = {
    "2h": 7200.0,
    "6h": 21600.0,
    "8h": 28800.0,
    "12h": 43200.0,
    "24h": 86400.0,
}
EVENT_OFFSETS = (100, 500, 1000)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return env


def _tag(job: dict) -> str:
    return f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"


def _rate_output(job: dict) -> Path:
    return LONG_ROOT / "physical" / job["scale"] / "dynamic_rate" / job["subject"] / f"seed{job['seed']}" / "card.json"


def _state_output(job: dict) -> Path:
    family = "physical" if job["family"] == "physical" else "event_offset"
    key = job["scale"] if family == "physical" else str(job["offset"])
    return LONG_ROOT / family / key / "full_mark" / job["subject"] / _tag(job) / "card.json"


def _adapter_root(subject: str) -> Path | None:
    for root in (CAUSAL_ROOT / "stepwise_decoder", EXPANSION_ROOT / "stepwise_decoder"):
        if (root / subject).is_dir():
            return root
    return None


def _core_rate_root(subject: str, state_seed: int) -> Path | None:
    for root in (CAUSAL_ROOT / "dynamic_rate", EXPANSION_ROOT / "dynamic_rate"):
        card = root / subject / f"seed{state_seed}" / "card.json"
        if not card.is_file():
            continue
        payload = json.loads(card.read_text(encoding="utf-8"))
        if payload.get("q_names", [None])[-1] != "segment_elapsed_over_8h":
            continue
        return root
    return None


def _run_pool(
    jobs: list[dict], slots: list[str], command, output, queue_dir: Path, kind: str,
) -> tuple[list[dict], list[dict]]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    logs = queue_dir / "logs"; logs.mkdir(exist_ok=True)
    pending, running, complete, failed = list(jobs), {}, [], []
    env = _env()
    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close(); job = row["job"]
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-30000:]
            if code == 0 and output(job).is_file():
                complete.append(job)
            elif "out of memory" in body.lower() and job.get("retries", 0) < 3:
                job["chunk_events"] = max(24, int(job.get("chunk_events", 256)) // 2)
                job["retries"] = int(job.get("retries", 0)) + 1
                pending.insert(0, job)
            else:
                failed.append({**job, "returncode": code, "log": row["log"], "tail": body[-4000:]})
            del running[slot]
        for slot in slots:
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            if output(job).is_file():
                complete.append(job); continue
            log = logs / f"{kind}_{job.get('family','rate')}_{job.get('scale',job.get('offset'))}_{job['subject']}_{job.get('seed',job.get('state_seed'))}_{slot.replace(':','_')}.log"
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command(job, slot), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            running[slot] = {"process": process, "handle": handle, "job": job,
                             "log": str(log), "started": time.time()}
        atomic_json(queue_dir / "queue_state.json", {
            "format": f"group_event_state_v0_3_5_long_{kind}_queue_v2",
            "status": "RUNNING", "pending": len(pending), "complete": len(complete),
            "failed": failed, "updated_epoch": time.time(),
            "running": {slot: {"pid": row["process"].pid, "job": row["job"],
                               "elapsed_seconds": time.time() - row["started"]}
                        for slot, row in running.items()},
        })
        if pending or running:
            time.sleep(10)
    atomic_json(queue_dir / "queue_done.json", {
        "format": f"group_event_state_v0_3_5_long_{kind}_done_v2",
        "status": "COMPLETE" if not failed else "PARTIAL",
        "complete": complete, "failed": failed,
    })
    return complete, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument(
        "--enable-horizon-specific-state", action="store_true",
        help=(
            "Legacy diagnostic only. By default this runner stops after the "
            "horizon-specific L0 rate baselines because a separately trained "
            "state per horizon cannot establish one persistent state."
        ),
    )
    parser.add_argument("--wait-causal", type=Path, default=CAUSAL_ROOT / "causal_supervisor" / "queue_done.json")
    parser.add_argument("--wait-expansion", type=Path, default=EXPANSION_ROOT / "cohort_expansion" / "supervisor" / "queue_done.json")
    args = parser.parse_args()
    supervisor = LONG_ROOT / "supervisor"; supervisor.mkdir(parents=True, exist_ok=True)
    audit = LONG_ROOT / "estimability_v3.json"
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/audit_group_event_state_v035_long_horizon_estimability.py"),
        "--out", str(audit),
    ], cwd=ROOT, check=True)
    estimability = json.loads(audit.read_text(encoding="utf-8"))
    atomic_json(supervisor / "RUN_CONTRACT.json", {
        "format": "group_event_state_v0_3_5_long_observed_support_contract_v3",
        "window": "may cross excluded/unobserved intervals; effective observed seconds enter the count likelihood as an offset",
        "state_carry": "carry across <=10 min non-seizure gaps; never count those missing seconds as observed exposure",
        "split": "each physical horizon gets its own FIT/INNER/final holdout; final holdout >=3H observed exposure",
        "checkpoint": "horizon-specific checkpoints apply to L0 baseline heads only",
        "full_state_contract": "one shared S_N producer and one shared S_G producer across horizons; implemented by a separate runner",
        "physical_scales": HORIZON_SECONDS, "event_offsets": list(EVENT_OFFSETS),
        "seeds": list(LOCKED_SEEDS[:3]),
        "h2b": "right-censored person-period log score is primary; outcome-selected binary risk is descriptive only",
        "development_targets_read": False, "sealed_partition_opened": False,
    })

    rate_jobs = []
    for scale in HORIZON_SECONDS:
        for subject in estimability["summary"][scale]["split_estimable"]:
            for seed in LOCKED_SEEDS[:3]:
                rate_jobs.append({"subject": subject, "scale": scale, "seed": int(seed)})

    def rate_command(job: dict, slot: str) -> list[str]:
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"),
            "--subject", job["subject"], "--seed", str(job["seed"]),
            "--config-json", str(ROOT / f"config/group_event_state_v035_rate_search/observed_support_{job['scale']}.json"),
            "--out-root", str(LONG_ROOT / "physical" / job["scale"] / "dynamic_rate"),
            "--device", "cpu",
        ]

    _run_pool(
        rate_jobs, [f"cpu:{i}" for i in range(args.cpu_workers)], rate_command,
        _rate_output, supervisor / "rate", "rate",
    )

    if not args.enable_horizon_specific_state:
        atomic_json(supervisor / "queue_done.json", {
            "format": "group_event_state_v0_3_5_long_baseline_done_v1",
            "status": "BASELINE_COMPLETE",
            "rate_jobs": len(rate_jobs),
            "state_stage": "DISABLED_BY_SHARED_STATE_PRODUCER_AMENDMENT",
            "reason": (
                "horizon-specific evaluator heads remain valid, but separately "
                "trained state producers cannot test persistence of one state "
                "across 2h/6h/8h/12h horizons"
            ),
            "development_targets_read": False,
            "sealed_partition_opened": False,
        })
        return

    while not (args.wait_causal.is_file() and args.wait_expansion.is_file()):
        atomic_json(supervisor / "queue_state.json", {
            "format": "group_event_state_v0_3_5_long_master_queue_v3",
            "status": "WAITING_FOR_CAUSAL_AND_EXPANSION_ADAPTERS",
            "causal_ready": args.wait_causal.is_file(), "expansion_ready": args.wait_expansion.is_file(),
            "rate_jobs_complete": len(rate_jobs), "updated_epoch": time.time(),
        })
        time.sleep(30)

    state_jobs, unavailable = [], []
    for scale in HORIZON_SECONDS:
        for subject in estimability["summary"][scale]["split_estimable"]:
            if subject not in V035_DECODER_FITS:
                unavailable.append({"family": "physical", "scale": scale, "subject": subject,
                                    "reason": "no mature recorded-time contact decoder"})
                continue
            for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
                state_jobs.append({"family": "physical", "scale": scale, "subject": subject,
                                   "decoder_seed": decoder_seed, "state_seed": int(state_seed),
                                   "chunk_events": 256, "retries": 0})
    by_subject = {row["subject"]: row for row in estimability["rows"]}
    for subject in sorted(V035_DECODER_FITS):
        row = by_subject.get(subject)
        if row is None:
            continue
        for offset in EVENT_OFFSETS:
            support = row["event_offsets"][str(offset)]
            if min(support[p]["same_segment_target_pairs"] for p in ("FIT", "INNER", "SELECTION")) < 1:
                unavailable.append({"family": "event_offset", "offset": offset, "subject": subject,
                                    "reason": "no nonoverlap pair in every chronological phase"})
                continue
            for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
                state_jobs.append({"family": "event_offset", "offset": offset, "subject": subject,
                                   "decoder_seed": decoder_seed, "state_seed": int(state_seed),
                                   "chunk_events": 256, "retries": 0})

    ready_jobs = []
    for job in state_jobs:
        subject, seed = job["subject"], job["state_seed"]
        adapter_root = _adapter_root(subject)
        decoder = DECODER_ROOT / "formal_units" / V035_DECODER_FITS[subject] / ARM / f"seed{job['decoder_seed']}" / "DONE.json"
        if adapter_root is None or not decoder.is_file():
            unavailable.append({**job, "reason": "causal adapter or mature decoder missing"}); continue
        if job["family"] == "event_offset" and _core_rate_root(subject, seed) is None:
            unavailable.append({**job, "reason": "causal core q trajectory missing"}); continue
        job["adapter_root"] = str(adapter_root)
        ready_jobs.append(job)
    atomic_json(supervisor / "state_inventory.json", {
        "format": "group_event_state_v0_3_5_long_state_inventory_v3",
        "ready_jobs": ready_jobs, "unavailable": unavailable,
    })

    def state_command(job: dict, gpu: str) -> list[str]:
        if job["family"] == "physical":
            rate_root = LONG_ROOT / "physical" / job["scale"] / "dynamic_rate"
            config = ROOT / f"config/group_event_state_v035_search/observed_support_{job['scale']}.json"
            out_root = LONG_ROOT / "physical" / job["scale"] / "full_mark"
        else:
            rate_root = _core_rate_root(job["subject"], job["state_seed"])
            config = ROOT / f"config/group_event_state_v035_search/event_offset_{job['offset']}.json"
            out_root = LONG_ROOT / "event_offset" / str(job["offset"]) / "full_mark"
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"),
            "--subject", job["subject"], "--decoder-seed", str(job["decoder_seed"]),
            "--state-seed", str(job["state_seed"]), "--config-json", str(config),
            "--rate-root", str(rate_root), "--adapter-root", job["adapter_root"],
            "--out-root", str(out_root), "--chunk-events", str(job["chunk_events"]),
            "--device", f"cuda:{gpu.split(':', 1)[0]}",
        ]

    gpu_slots = [
        f"{gpu}:{worker}"
        for gpu in [value.strip() for value in args.gpus.split(",") if value.strip()]
        for worker in range(args.workers_per_gpu)
    ]
    complete, failed = _run_pool(
        ready_jobs, gpu_slots, state_command, _state_output,
        supervisor / "state", "state",
    )

    seizure_jobs = [job for job in complete if job["family"] == "physical"]
    def seizure_output(job: dict) -> Path:
        return LONG_ROOT / "physical" / job["scale"] / "seizure" / job["subject"] / _tag(job) / "card.json"
    def seizure_command(job: dict, slot: str) -> list[str]:
        unit = _state_output(job).parent
        return [
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_long_seizure_transfer.py"),
            "--subject", job["subject"], "--trajectory", str(unit / "state_trajectory.npz"),
            "--rate", str(LONG_ROOT / "physical" / job["scale"] / "dynamic_rate" / job["subject"] / f"seed{job['state_seed']}" / "trajectory_and_scores.npz"),
            "--horizon-seconds", str(HORIZON_SECONDS[job["scale"]]),
            "--out-dir", str(seizure_output(job).parent),
        ]
    seizure_complete, seizure_failed = _run_pool(
        seizure_jobs, [f"cpu:{i}" for i in range(args.cpu_workers)],
        seizure_command, seizure_output, supervisor / "seizure", "seizure",
    )
    atomic_json(supervisor / "queue_done.json", {
        "format": "group_event_state_v0_3_5_long_master_done_v3",
        "status": "COMPLETE" if not failed and not seizure_failed else "PARTIAL",
        "rate_jobs": len(rate_jobs), "state_complete": complete, "state_failed": failed,
        "seizure_complete": seizure_complete, "seizure_failed": seizure_failed,
        "unavailable": unavailable,
    })


if __name__ == "__main__":
    main()
