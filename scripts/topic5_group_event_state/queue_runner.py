#!/usr/bin/env python3
"""Single-owner GPU job queue for Group-Event State v0.1.

One process owns the queue.  Each job is an independent subprocess pinned to a
GPU, so a CUDA failure in one run cannot poison another.  Completed runs are
detected by their ``result.json``, so re-running the queue resumes instead of
repeating work.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
PYBIN = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--max-train-seconds", type=float, default=2400.0)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--chunk-events", type=int, default=128)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    state_file = args.state_file or (V0_1 / f"queue_state_{args.tag}.json")
    jobs = [
        {"subject": s, "arm": a, "seed": d}
        for s, a, d in itertools.product(args.subjects, args.arms, args.seeds)
    ]
    pending = [
        j
        for j in jobs
        if not (V0_1 / "runs" / args.tag / f"{j['subject']}__{j['arm']}__seed{j['seed']}" / "result.json").exists()
    ]
    print(f"{len(jobs)} jobs, {len(pending)} pending, "
          f"{len(args.gpus)} GPUs x {args.jobs_per_gpu} = {len(args.gpus)*args.jobs_per_gpu} slots",
          flush=True)

    work: queue.Queue = queue.Queue()
    for job in pending:
        work.put(job)
    results: list[dict] = []
    lock = threading.Lock()
    started_at = time.time()

    def _worker(gpu: int, slot: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            run_id = f"{job['subject']}__{job['arm']}__seed{job['seed']}"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["PYTHONPATH"] = str(ROOT)
            cmd = [
                PYBIN, str(ROOT / "scripts/topic5_group_event_state/run_experiment.py"),
                "--subject", job["subject"], "--arm", job["arm"], "--seed", str(job["seed"]),
                "--tag", args.tag, "--max-train-seconds", str(args.max_train_seconds),
                "--max-epochs", str(args.max_epochs), "--chunk-events", str(args.chunk_events),
                *args.extra,
            ]
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            record = {
                "run_id": run_id, "gpu": gpu, "slot": slot,
                "returncode": proc.returncode, "seconds": round(time.time() - t0, 1),
                "stdout_tail": proc.stdout.strip().splitlines()[-3:],
                "stderr_tail": proc.stderr.strip().splitlines()[-6:] if proc.returncode else [],
            }
            with lock:
                results.append(record)
                done = len(results)
                total = len(pending)
                elapsed = time.time() - started_at
                eta = (elapsed / done) * (total - done) / 3600 if done else float("nan")
                print(
                    f"[{done}/{total}] gpu{gpu} rc={proc.returncode} {record['seconds']}s "
                    f"{run_id} | ETA {eta:.2f}h",
                    flush=True,
                )
                tmp = Path(str(state_file) + ".tmp")
                tmp.write_text(json.dumps(
                    {"tag": args.tag, "n_jobs": len(jobs), "n_pending": total,
                     "n_done": done, "elapsed_sec": round(elapsed, 1),
                     "results": results}, indent=2))
                os.replace(tmp, state_file)

    threads = [
        threading.Thread(target=_worker, args=(gpu, slot), daemon=False)
        for gpu in args.gpus
        for slot in range(args.jobs_per_gpu)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bad = [r for r in results if r["returncode"] != 0]
    print(f"QUEUE DONE: {len(results)} run, {len(bad)} nonzero exit, "
          f"{(time.time()-started_at)/3600:.2f}h", flush=True)
    for r in bad[:20]:
        print(f"  FAIL {r['run_id']} rc={r['returncode']}: {r['stderr_tail'][-1] if r['stderr_tail'] else ''}")


if __name__ == "__main__":
    main()
