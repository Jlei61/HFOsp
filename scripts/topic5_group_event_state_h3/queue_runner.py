#!/usr/bin/env python3
"""Single-owner queue for the H3 grid.

One process owns the queue, writes its own PID/PGID into the lease, and launches
jobs as children.  Nothing else may start, stop or adopt these jobs, and no job
is ever addressed by pattern -- ``pkill -f`` has already cost this project a
neighbouring queue once.

Idempotent by result hash: a run whose ``result.json`` exists, parses, says
``ok`` and carries the current ``config_hash`` is skipped.  A missing file, a
stale hash or a non-finite score all mean *not done*.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.models import ARM_NAMES  # noqa: E402

PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"
LEASE = ROOT / "results/epi_prssm/group_event_state/v0_2/shared/resource_leases/agent_c.json"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def gpu_free_gib(index: int) -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used",
         "--format=csv,noheader,nounits", "-i", str(index)],
        text=True,
    ).strip().split(",")
    return (float(out[0]) - float(out[1])) / 1024.0


def write_lease(payload: dict) -> None:
    LEASE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEASE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, LEASE)


STAGES = {
    # stage -> (script, result subdirectory, whether the arm is swept)
    "models": ("run_h3_models.py", "{tag}", True),
    "perturbation": ("run_perturbation.py", "perturbation_{tag}", False),
    "impulse": ("run_impulse.py", "impulse_{tag}", False),
    "innovation": ("run_innovation.py", "innovation_{tag}", False),
}


def job_done(stage: str, tag: str, subject: str, arm: str, seed: int) -> bool:
    subdir = STAGES[stage][1].format(tag=tag)
    path = OUT_ROOT / "machine" / subdir / f"{subject}__{arm}__seed{seed}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("status") == "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--stage", default="models", choices=sorted(STAGES))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train-seconds", type=float, default=2400.0)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument(
        "--include-secondary", action="store_true",
        help="perturbation stage only: also run the two secondary arms",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    script_name, _subdir, sweeps_arms = STAGES[args.stage]
    if not sweeps_arms and len(args.arms) > 1:
        # The downstream stages replay one frozen arm; sweeping arms there would
        # silently multiply the work and produce files nothing reads.
        args.arms = ["M2_mark_specific_feedback"]
    log_dir = OUT_ROOT / "logs" / f"{args.stage}_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Long patients first: they dominate wall time, and starting them last leaves
    # the whole queue waiting on one straggler.
    jobs = [
        (subject, arm, seed)
        for subject in args.subjects
        for arm in args.arms
        for seed in args.seeds
        if not job_done(args.stage, args.tag, subject, arm, seed)
    ]
    print(f"queue: {len(jobs)} jobs pending "
          f"({len(args.subjects)} subjects x {len(args.arms)} arms x {len(args.seeds)} seeds)")

    running: list[tuple[subprocess.Popen, tuple, int, object]] = []
    started_at = time.time()
    failures: list[dict] = []
    resource_failed: list[dict] = []
    attempts: dict[tuple, int] = {}

    def spawn(job: tuple, gpu: int, batch_scale: int) -> None:
        subject, arm, seed = job
        cmd = [
            PYTHON, str(ROOT / "scripts/topic5_group_event_state_h3" / script_name),
            "--subject", subject, "--arm", arm, "--seed", str(seed),
            "--tag", args.tag, "--device", f"cuda:{gpu}",
        ]
        if args.stage == "models":
            cmd += [
                "--max-epochs", str(args.max_epochs),
                "--max-train-seconds", str(args.max_train_seconds),
            ]
            if args.lr is not None:
                cmd += ["--lr", str(args.lr)]
            if batch_scale > 0:
                # OOM back-off only ever shrinks the *current* job's window; it
                # never drops a patient and never changes an endpoint.
                cmd += ["--window-steps", str(max(512, 4096 >> batch_scale))]
        if args.stage == "perturbation" and args.include_secondary:
            cmd += ["--include-secondary"]
        cmd += list(args.extra)
        env = dict(os.environ, **THREAD_ENV)
        log_path = log_dir / f"{subject}__{arm}__seed{seed}.log"
        handle = log_path.open("a")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        running.append((proc, job, gpu, handle))
        print(f"  start gpu{gpu} {subject} {arm} seed{seed} pid={proc.pid}", flush=True)

    try:
        while jobs or running:
            for entry in list(running):
                proc, job, gpu, handle = entry
                if proc.poll() is None:
                    continue
                running.remove(entry)
                handle.close()
                if proc.returncode != 0:
                    attempts[job] = attempts.get(job, 0) + 1
                    log_text = (log_dir / f"{job[0]}__{job[1]}__seed{job[2]}.log").read_text()[-4000:]
                    oom = "CUDA out of memory" in log_text or "out of memory" in log_text
                    if oom and attempts[job] <= 3:
                        print(f"  OOM retry {attempts[job]} for {job}", flush=True)
                        jobs.insert(0, job)
                    else:
                        record = {"job": list(job), "returncode": proc.returncode,
                                  "attempts": attempts[job],
                                  "kind": "resource_failed" if oom else "failed"}
                        (resource_failed if oom else failures).append(record)
                        print(f"  FAIL {job} rc={proc.returncode} ({record['kind']})", flush=True)

            capacity = []
            for gpu in args.gpus:
                free = gpu_free_gib(gpu)
                used_slots = sum(1 for _p, _j, g, _h in running if g == gpu)
                if used_slots < args.slots_per_gpu and free > args.reserve_gib:
                    capacity.append(gpu)
            while jobs and capacity:
                gpu = capacity.pop(0)
                job = jobs.pop(0)
                spawn(job, gpu, attempts.get(job, 0))

            write_lease(
                {
                    "agent": "agent_c",
                    "role": "H3 event feedback",
                    "pid": os.getpid(),
                    "pgid": os.getpgid(0),
                    "gpu": args.gpus,
                    "slots": len(running),
                    "tag": args.tag,
                    "phase": args.stage,
                    "n_pending": len(jobs),
                    "n_running": len(running),
                    "n_failed": len(failures),
                    "n_resource_failed": len(resource_failed),
                    "heartbeat_epoch": time.time(),
                    "elapsed_seconds": round(time.time() - started_at, 1),
                }
            )
            if jobs or running:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("interrupt: terminating this queue's own children only")
        for proc, _job, _gpu, _handle in running:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise
    finally:
        write_json_atomic(
            {
                "tag": args.tag,
                "n_jobs": len(args.subjects) * len(args.arms) * len(args.seeds),
                "failures": failures,
                "resource_failed": resource_failed,
                "elapsed_seconds": round(time.time() - started_at, 1),
            },
            OUT_ROOT / "machine" / f"queue_state_{args.stage}_{args.tag}.json",
        )
    print(f"done in {round(time.time() - started_at, 1)}s; "
          f"{len(failures)} failed, {len(resource_failed)} resource_failed")


if __name__ == "__main__":
    main()
