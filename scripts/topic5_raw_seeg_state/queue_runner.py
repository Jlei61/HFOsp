#!/usr/bin/env python
"""Patient-level GPU job queue: at most one training job on the card at a time.

The box has a single RTX 3090 and other work shares it, so patient jobs are
strictly serialised behind an advisory file lock on ``jobs/gpu.lock``. Launching
a second runner is a no-op with a clear message and a non-zero exit code, never
a double-run.

Survives terminal hangup (SIGHUP ignored, no tty needed):

    LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH \
    setsid nohup /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
      scripts/topic5_raw_seeg_state/queue_runner.py --jobs jobs/r0_1_jobs.json \
      >> logs/queue_runner.log 2>&1 &

Job list format::

    {"jobs": [{"job_id": "epilepsiae_1073__full__s0",
               "subject": "epilepsiae_1073", "arm": "full", "seed": 0,
               "args": ["--batch-size", "2"]}]}

A job whose ``jobs/<job_id>.status.json`` says DONE with the current package
hash is skipped. A job that exits with the OOM code is re-queued at most once
with a halved batch size.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import contract  # noqa: E402
from src.topic5_raw_seeg_state.train import EXIT_OOM_BUDGET  # noqa: E402

RUN_PATIENT = REPO / "scripts" / "topic5_raw_seeg_state" / "run_patient.py"


class LockBusy(RuntimeError):
    """Another queue runner already holds the GPU lock."""


@contextmanager
def gpu_lock(path: Path):
    """Advisory exclusive lock; raises :class:`LockBusy` instead of blocking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        holder = fh.read().strip()
        fh.close()
        raise LockBusy(holder or "unknown")
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"pid": os.getpid(), "host": os.uname().nodename,
                             "started": time.time()}))
        fh.flush()
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def status_path(job_dir: Path, job_id: str) -> Path:
    return Path(job_dir) / f"{job_id}.status.json"


def read_status(job_dir: Path, job_id: str) -> dict:
    p = status_path(job_dir, job_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def write_status(job_dir: Path, job_id: str, **fields) -> None:
    payload = read_status(job_dir, job_id)
    payload.update(fields)
    payload["job_id"] = job_id
    payload["updated"] = time.time()
    contract.atomic_write_json(status_path(job_dir, job_id), payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help="job list JSON")
    p.add_argument("--job-dir", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--lock", default=None)
    p.add_argument("--python", default=contract.PYTHON_BIN)
    p.add_argument("--runner", default=str(RUN_PATIENT))
    p.add_argument("--force", action="store_true", help="ignore DONE status files")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    job_dir = Path(args.job_dir) if args.job_dir else contract.JOB_DIR
    log_dir = Path(args.log_dir) if args.log_dir else contract.LOG_DIR
    lock_path = Path(args.lock) if args.lock else (job_dir / "gpu.lock")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    package_hash = contract.package_hash(contract.r0_1_source_files())

    try:
        with gpu_lock(lock_path):
            queue = list(jobs)
            done, failed, skipped = 0, 0, 0
            while queue:
                job = queue.pop(0)
                job_id = job.get("job_id") or (
                    f"{job['subject']}__{job.get('arm', 'full')}__s{job.get('seed', 0)}")
                prev = read_status(job_dir, job_id)
                if (not args.force and prev.get("status") == "DONE"
                        and prev.get("package_hash") == package_hash):
                    print(f"[skip] {job_id} already DONE at this package hash", flush=True)
                    skipped += 1
                    continue

                cmd = [args.python, args.runner, "--subject", job["subject"],
                       "--arm", job.get("arm", "full"), "--seed", str(job.get("seed", 0)),
                       "--job-id", job_id, "--resume"]
                cmd += [str(x) for x in job.get("args", [])]
                log_path = log_dir / f"{job_id}.log"
                write_status(job_dir, job_id, status="RUNNING",
                             subject=job["subject"], arm=job.get("arm", "full"),
                             seed=job.get("seed", 0), package_hash=package_hash,
                             command=cmd, log=str(log_path), started=time.time(),
                             requeued_after_oom=bool(job.get("_requeued", False)))
                print(f"[run ] {job_id} -> {log_path}", flush=True)
                if args.dry_run:
                    write_status(job_dir, job_id, status="DONE", returncode=0,
                                 dry_run=True)
                    done += 1
                    continue
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = f"{contract.CONDA_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                with open(log_path, "a") as fh:
                    fh.write(f"\n===== {job_id} start {time.ctime()} =====\n")
                    fh.flush()
                    rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                         cwd=str(REPO), env=env, stdin=subprocess.DEVNULL)
                if rc == 0:
                    write_status(job_dir, job_id, status="DONE", returncode=0,
                                 finished=time.time())
                    done += 1
                elif rc == EXIT_OOM_BUDGET and not job.get("_requeued"):
                    downgraded = _downgrade(job)
                    downgraded["_requeued"] = True
                    write_status(job_dir, job_id, status="FAILED", returncode=rc,
                                 reason="oom_budget_exhausted", requeued=True,
                                 downgraded_args=downgraded.get("args"))
                    queue.append(downgraded)
                    print(f"[oom ] {job_id} re-queued once with a downgraded config",
                          flush=True)
                else:
                    write_status(job_dir, job_id, status="FAILED", returncode=rc,
                                 reason=("oom_budget_exhausted" if rc == EXIT_OOM_BUDGET
                                         else "nonzero_exit"),
                                 finished=time.time())
                    failed += 1
            print(json.dumps({"done": done, "failed": failed, "skipped": skipped}), flush=True)
            return 0 if failed == 0 else 2
    except LockBusy as exc:
        print(f"another queue runner already holds {lock_path} ({exc}); "
              "refusing to double-run", file=sys.stderr, flush=True)
        return 3


def _downgrade(job: dict) -> dict:
    """Halve the batch size and double accumulation for the single re-queue."""
    out = dict(job)
    argv = [str(x) for x in out.get("args", [])]
    batch = 1
    if "--batch-size" in argv:
        i = argv.index("--batch-size")
        batch = max(1, int(argv[i + 1]) // 2)
        argv[i + 1] = str(batch)
    else:
        argv += ["--batch-size", "1"]
    if "--grad-accum" in argv:
        i = argv.index("--grad-accum")
        argv[i + 1] = str(int(argv[i + 1]) * 2)
    else:
        argv += ["--grad-accum", "8"]
    out["args"] = argv
    return out


if __name__ == "__main__":
    raise SystemExit(main())
