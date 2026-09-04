#!/usr/bin/env python3
"""Re-run the complete v0.3.5 chain with the causal q(t) into a parallel root.

Review 2026-09-04 found that the original q(t) carried a non-causal
``segment_fraction`` feature (it used the coverage-segment END, which coincides
with the next seizure onset for most patients).  Because q(t) feeds W2--W6,
spec §11 stop-condition 2 applies and the chain is re-run from W1 with the
fixed feature.  Nothing in the original root is modified; the frozen decoders
(v0.3.4) and the step-wise future-oracle positive control (independent of q
values) are reused.  One GPU worker per card by default so concurrent jobs of
other users are only mildly slowed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORIGINAL_ROOT = Path("/data/hfosp_group_event_state_v0_3_5")
CAUSAL_ROOT = Path(os.environ.get("HFOSP_GES_V035_OUTPUT_ROOT", "/data/hfosp_group_event_state_v0_3_5_causal"))
os.environ["HFOSP_GES_V035_OUTPUT_ROOT"] = str(CAUSAL_ROOT)

from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    LOCKED_SEEDS, OUTPUT_ROOT, V035_SUBJECTS, atomic_json,
)

assert OUTPUT_ROOT == CAUSAL_ROOT, (OUTPUT_ROOT, CAUSAL_ROOT)
PY = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
STATE_SUBJECTS = ("epilepsiae_253", "epilepsiae_1096", "epilepsiae_548", "epilepsiae_583",
                  "epilepsiae_1146", "epilepsiae_384", "epilepsiae_1125")
FINAL_CONFIG = ROOT / "config/group_event_state_v035_search/compact.json"


def _seeds(subject: str) -> tuple[int, ...]:
    return LOCKED_SEEDS if subject not in {"epilepsiae_384", "epilepsiae_1125"} else LOCKED_SEEDS[:3]


def _jobs() -> list[dict]:
    jobs = []
    for subject in V035_SUBJECTS:
        for seed in _seeds(subject):
            rate_card = CAUSAL_ROOT / "dynamic_rate" / subject / f"seed{seed}" / "card.json"
            jobs.append({"kind": "W1", "device": "cpu", "subject": subject, "seed": seed, "out": str(rate_card), "deps": [],
                         "cmd": [str(PY), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"),
                                 "--subject", subject, "--seed", str(seed), "--device", "cpu"]})
            jobs.append({"kind": "W1b", "device": "cpu", "subject": subject, "seed": seed,
                         "out": str(CAUSAL_ROOT / "background_rate" / subject / f"seed{seed}" / "card.json"),
                         "deps": [str(rate_card)],
                         "cmd": [str(PY), str(ROOT / "scripts/run_group_event_state_v035_background_rate.py"),
                                 "--subject", subject, "--seed", str(seed), "--device", "cpu"]})
    for subject in STATE_SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            tag = f"decoder_seed{decoder_seed}_state_seed{state_seed}"
            rate_traj = CAUSAL_ROOT / "dynamic_rate" / subject / f"seed{state_seed}" / "trajectory_and_scores.npz"
            adapter_card = CAUSAL_ROOT / "stepwise_decoder" / subject / tag / "card.json"
            state_card = CAUSAL_ROOT / "full_mark_final" / subject / tag / "card.json"
            common = ["--subject", subject, "--decoder-seed", str(decoder_seed), "--state-seed", str(state_seed)]
            jobs.append({"kind": "W2", "device": "gpu", "subject": subject, "seed": state_seed, "out": str(adapter_card),
                         "deps": [str(rate_traj)],
                         "cmd": [str(PY), str(ROOT / "scripts/run_group_event_state_v035_stepwise_decoder.py"), *common]})
            jobs.append({"kind": "W3", "device": "gpu", "subject": subject, "seed": state_seed, "out": str(state_card),
                         "deps": [str(adapter_card)], "chunk_events": 256, "retries": 0,
                         "cmd": [str(PY), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"), *common,
                                 "--config-json", str(FINAL_CONFIG), "--out-root", str(CAUSAL_ROOT / "full_mark_final")]})
            jobs.append({"kind": "W456", "device": "gpu", "subject": subject, "seed": state_seed,
                         "out": str(CAUSAL_ROOT / "final_downstream" / subject / tag / "card.json"),
                         "deps": [str(state_card)], "batch_events": 96, "retries": 0,
                         "cmd": [str(PY), str(ROOT / "scripts/run_group_event_state_v035_final_downstream.py"), *common]})
    return jobs


def _extra_args(job: dict) -> list[str]:
    if job["kind"] == "W3":
        return ["--chunk-events", str(job["chunk_events"])]
    if job["kind"] == "W456":
        return ["--batch-events", str(job["batch_events"])]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--skip-finalize", action="store_true")
    a = ap.parse_args()
    gpus = [v.strip() for v in a.gpus.split(",") if v.strip()]
    CAUSAL_ROOT.mkdir(parents=True, exist_ok=True)
    control = CAUSAL_ROOT / "causal_supervisor"; logs = control / "logs"; logs.mkdir(parents=True, exist_ok=True)
    atomic_json(CAUSAL_ROOT / "RUN_CONTRACT.json", {
        "format": "group_event_state_v0_3_5_causal_rerun_contract_v1",
        "reason": "review 2026-09-04: q(t) segment_fraction used the coverage-segment end (next seizure onset); "
                  "re-run W1-W6 with log_time_since_segment_start",
        "original_root": str(ORIGINAL_ROOT), "causal_root": str(CAUSAL_ROOT),
        "reused_from_original": ["stepwise_oracle (future-participation positive control; independent of q values)"],
        "final_config": str(FINAL_CONFIG), "started_epoch": time.time(),
        "development_targets_read": False, "sealed_partition_opened": False,
    })
    oracle_dst = CAUSAL_ROOT / "stepwise_oracle"
    if not oracle_dst.exists() and (ORIGINAL_ROOT / "stepwise_oracle").exists():
        shutil.copytree(ORIGINAL_ROOT / "stepwise_oracle", oracle_dst)
        (oracle_dst / "PROVENANCE.txt").write_text(
            "Copied unchanged from the original root: the future-oracle context replaces q, so q values "
            "do not enter this positive control.\n", encoding="utf-8")
    slots = [(f"gpu{g}:{w}", g) for g in gpus for w in range(a.workers_per_gpu)] + [(f"cpu:{w}", None) for w in range(a.cpu_workers)]
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    pending = _jobs(); running = {}; complete = []; failed = []
    while pending or running:
        for slot, row in list(running.items()):
            code = row["p"].poll()
            if code is None: continue
            row["h"].close(); job = row["job"]
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-30000:]
            if code == 0 and Path(job["out"]).exists():
                complete.append({k: v for k, v in job.items() if k != "cmd"})
            elif "out of memory" in body.lower() and job.get("retries", 0) < 3 and job["kind"] in {"W3", "W456"}:
                key = "chunk_events" if job["kind"] == "W3" else "batch_events"
                job[key] = max(12, job[key] // 2); job["retries"] += 1; pending.insert(0, job)
            else:
                failed.append({**{k: v for k, v in job.items() if k != "cmd"}, "returncode": code, "log": row["log"], "tail": body[-3000:]})
            del running[slot]
        failed_outs = {f["out"] for f in failed}
        for slot, gpu in slots:
            if slot in running: continue
            chosen = None
            for i, job in enumerate(pending):
                if (job["device"] == "gpu") != (gpu is not None): continue
                if Path(job["out"]).exists():
                    complete.append({k: v for k, v in job.items() if k != "cmd"}); pending.pop(i); chosen = (None, None); break
                if any(dep in failed_outs for dep in job["deps"]):
                    failed.append({**{k: v for k, v in job.items() if k != "cmd"}, "returncode": None, "log": None, "tail": "dependency failed"})
                    pending.pop(i); chosen = (None, None); break
                if all(Path(dep).exists() for dep in job["deps"]):
                    chosen = (i, job); break
            if chosen is None or chosen[1] is None: continue
            i, job = chosen; pending.pop(i)
            cmd = job["cmd"] + _extra_args(job) + (["--device", f"cuda:{gpu}"] if gpu is not None else [])
            log = logs / f"{job['kind']}_{job['subject']}_seed{job['seed']}_{slot.replace(':', '_')}.log"
            h = log.open("a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
            running[slot] = {"p": p, "h": h, "job": job, "log": str(log), "started": time.time()}
        atomic_json(control / "queue_state.json", {
            "format": "group_event_state_v0_3_5_causal_queue_v1", "updated_epoch": time.time(),
            "pending": len(pending), "pending_by_kind": {k: sum(j["kind"] == k for j in pending) for k in ("W1", "W1b", "W2", "W3", "W456")},
            "complete": len(complete), "failed": failed,
            "running": {slot: {"pid": row["p"].pid, "kind": row["job"]["kind"], "subject": row["job"]["subject"],
                               "seed": row["job"]["seed"], "elapsed_seconds": time.time() - row["started"]} for slot, row in running.items()},
        })
        if pending or running: time.sleep(a.poll_seconds)
    atomic_json(CAUSAL_ROOT / "full_mark_final" / "epilepsiae_922" / "NOT_ESTIMABLE.json", {
        "format": "group_event_state_v0_3_5_final_not_estimable_v1", "subject": "epilepsiae_922", "status": "NOT_ESTIMABLE",
        "reason": "mature decoder has no scorable event in the registered evaluation window (unchanged from original run)",
        "development_targets_read": False, "sealed_partition_opened": False})
    finalize = None
    if not a.skip_finalize and not failed:
        fenv = {**env, "HFOSP_GES_V035_REPORT_TAG": "causal_rerun"}
        flog = logs / "finalize.log"
        with flog.open("a", encoding="utf-8") as h:
            finalize = subprocess.run([str(PY), str(ROOT / "scripts/finalize_group_event_state_v035.py")], cwd=ROOT, env=fenv,
                                      stdout=h, stderr=subprocess.STDOUT).returncode
    atomic_json(control / "queue_done.json", {"format": "group_event_state_v0_3_5_causal_done_v1", "complete": complete,
                                              "failed": failed, "finalize_returncode": finalize, "finished_epoch": time.time()})


if __name__ == "__main__":
    main()
