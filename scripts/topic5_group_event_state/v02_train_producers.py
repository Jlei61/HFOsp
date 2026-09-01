#!/usr/bin/env python3
"""Train ``P_local`` / ``P_slow`` for a cohort, one GPU-pinned subprocess per job.

Two modes:

    # one job (what the queue owner spawns; also usable by hand)
    ... --job --subject epilepsiae_916 --producer P_slow --seed 1 --gpu 0

    # queue owner
    ... --cohort a1 --producers P_local P_slow --seeds 1 2 3 --gpus 0 1 \
        --jobs-per-gpu 2

Queue discipline (EI 4-5): a single owner, one lease, per-job atomic outputs,
idempotent resume by configuration hash, and processes managed by recorded PID --
never by ``pkill -f``, which in this repository once matched the shell that
issued it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers")
SHARED_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/shared"
)
A1_SMOKE = ("epilepsiae_916", "epilepsiae_253", "epilepsiae_1073")
A2_MIDSTAGE = (
    "epilepsiae_1073", "epilepsiae_1077", "epilepsiae_1096", "epilepsiae_1125",
    "epilepsiae_1146", "epilepsiae_253", "epilepsiae_384", "epilepsiae_548",
)
DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")

MAX_OOM_RETRIES = 3


def _cohort(name: str | None, subjects: list[str] | None) -> list[str]:
    if subjects:
        return list(subjects)
    if name == "a1":
        return list(A1_SMOKE)
    if name == "a2":
        return list(A2_MIDSTAGE)
    return sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists())


def _run_job(args: argparse.Namespace) -> int:
    # Pin the job to one card before torch initialises CUDA.  Selecting the
    # device afterwards left ``reset_peak_memory_stats`` on an uninitialised
    # context ("Invalid device argument"), and it also lets a stray allocation
    # land on the wrong card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from src.topic5_group_event_state.dataset import SubjectSequence
    from src.topic5_group_event_state.v02.producers import (
        build_producer_configs,
        train_producer,
    )
    from src.topic5_group_event_state.v02.registry import atomic_write_json, source_commit
    from src.topic5_group_event_state.v02.runtime import config_fingerprint
    from src.topic5_group_event_state.v02.subject import (
        SubjectTimelineConfig,
        load_subject_timeline,
    )

    from dataclasses import replace

    out_dir = Path(args.out_root) / args.tag / "runs" / args.subject / args.producer / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    cfg = build_producer_configs()[args.producer]
    cfg = replace(cfg, max_epochs=args.max_epochs, patience=args.patience,
                  max_train_seconds=args.max_train_seconds,
                  chunk_events=args.chunk_events, batch_segments=args.batch_segments)
    commit = source_commit(ROOT)
    cfg_hash = config_fingerprint(cfg.as_dict(), SubjectTimelineConfig().as_dict(),
                                  args.seed, commit)
    if result_path.exists():
        try:
            if json.loads(result_path.read_text()).get("config_hash") == cfg_hash:
                print(f"{args.subject}/{args.producer}/seed{args.seed}: already done")
                return 0
        except json.JSONDecodeError:
            pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tl = load_subject_timeline(args.subject, config=SubjectTimelineConfig())
    seq = SubjectSequence(DATASET_ROOT / args.subject)

    attempt = 0
    while True:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.time()
            result = train_producer(tl, seq, cfg, args.seed, device, out_dir)
            break
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
            attempt += 1
            torch.cuda.empty_cache()
            if attempt > MAX_OOM_RETRIES:
                atomic_write_json(out_dir / "resource_failed.json", {
                    "subject": args.subject, "producer": args.producer,
                    "seed": args.seed, "reason": "cuda_out_of_memory",
                    "attempts": attempt, "final_chunk_events": cfg.chunk_events,
                    "final_batch_segments": cfg.batch_segments,
                    "note": "this is a resource failure, NOT a scientific negative",
                    "error": str(exc)[:400],
                })
                print(f"RESOURCE_FAILED {args.subject}/{args.producer}/seed{args.seed}")
                return 2
            cfg = replace(cfg, chunk_events=max(16, cfg.chunk_events // 2),
                          batch_segments=max(1, cfg.batch_segments // 2))
            print(f"OOM retry {attempt}: chunk={cfg.chunk_events} "
                  f"batch={cfg.batch_segments}", flush=True)

    result["config_hash"] = cfg_hash
    result["commit"] = commit
    result["gpu"] = int(args.gpu)
    result["oom_retries"] = attempt
    result["wall_seconds"] = round(time.time() - started, 1)
    if device.type == "cuda":
        result["peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        result["peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
    atomic_write_json(result_path, result)

    state_dir = Path(args.out_root) / args.tag / "states" / f"{args.producer}_seed{args.seed}"
    state_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "anchor_state.npz"
    dst = state_dir / f"{args.subject}.npz"
    tmp = dst.with_suffix(".npz.tmp")
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)
    print(f"OK {args.subject}/{args.producer}/seed{args.seed} "
          f"{result['wall_seconds']}s epoch={result['selected_epoch']} "
          f"peak={result.get('peak_memory_reserved_gb', 0):.2f}GB")
    return 0


def _queue(args: argparse.Namespace) -> None:
    from src.topic5_group_event_state.v02.registry import atomic_write_json, source_commit
    from src.topic5_group_event_state.v02.runtime import ResourceLease, write_status

    subjects = _cohort(args.cohort, args.subjects)
    jobs = [
        (s, p, seed)
        for s in subjects for p in args.producers for seed in args.seeds
    ]
    out_root = Path(args.out_root) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "job_logs"
    log_dir.mkdir(exist_ok=True)

    lease = ResourceLease(
        SHARED_ROOT / "resource_leases" / f"agent_a_train_{args.tag}.json", "agent_a"
    )
    lease.acquire(
        gpus=list(args.gpus), slots=len(args.gpus) * args.jobs_per_gpu,
        task="train_producers", n_jobs=len(jobs),
        role="H1/H2a predictive-state producers (owns core + registry entries)",
        branch="codex/topic5-group-event-state-v02-a",
        worktree=str(ROOT), out_root=str(out_root),
        jobs_per_gpu=args.jobs_per_gpu,
        must_not_touch=[
            "/tmp/hfosp_group_event_state_v01 (v0.1 tree, read-only)",
            "agent B h2b/ and agent C h3/ result roots",
            "formal/sealed partitions; paper-ready Fig1-Fig4",
        ],
    )
    status = out_root / "STATUS.json"
    write_status(status, state="running", n_jobs=len(jobs), done=0, pid=os.getpid())

    slots: list[tuple[subprocess.Popen, tuple, int, object]] = []
    pending = list(jobs)
    done = 0
    results: list[dict] = []
    capacity = len(args.gpus) * args.jobs_per_gpu
    started = time.time()
    try:
        while pending or slots:
            while pending and len(slots) < capacity:
                subject, producer, seed = pending.pop(0)
                used = [g for _p, _j, g, _h in slots]
                gpu = min(args.gpus, key=lambda g: used.count(g))
                handle = (log_dir / f"{subject}_{producer}_seed{seed}.log").open("w")
                cmd = [
                    sys.executable, str(Path(__file__).resolve()), "--job",
                    "--subject", subject, "--producer", producer,
                    "--seed", str(seed), "--gpu", str(gpu),
                    "--out-root", str(args.out_root), "--tag", args.tag,
                    "--max-epochs", str(args.max_epochs),
                    "--patience", str(args.patience),
                    "--max-train-seconds", str(args.max_train_seconds),
                    "--chunk-events", str(args.chunk_events),
                    "--batch-segments", str(args.batch_segments),
                ]
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
                slots.append((proc, (subject, producer, seed), gpu, handle))
                print(f"launch {subject}/{producer}/seed{seed} gpu{gpu} pid={proc.pid}",
                      flush=True)
            time.sleep(5.0)
            still: list = []
            for proc, job, gpu, handle in slots:
                if proc.poll() is None:
                    still.append((proc, job, gpu, handle))
                    continue
                handle.close()
                done += 1
                results.append({"subject": job[0], "producer": job[1], "seed": job[2],
                                "gpu": gpu, "returncode": proc.returncode})
                print(f"[{done}/{len(jobs)}] {job} rc={proc.returncode}", flush=True)
                write_status(status, state="running", n_jobs=len(jobs), done=done,
                             pid=os.getpid(), elapsed=round(time.time() - started, 1))
                lease.beat(gpus=list(args.gpus), slots=capacity, task="train_producers",
                           done=done, n_jobs=len(jobs))
            slots = still
    finally:
        lease.release()

    manifest = {
        "tag": args.tag, "commit": source_commit(ROOT), "jobs": len(jobs),
        "n_ok": sum(1 for r in results if r["returncode"] == 0),
        "n_resource_failed": sum(1 for r in results if r["returncode"] == 2),
        "n_failed": sum(1 for r in results if r["returncode"] not in (0, 2)),
        "results": results, "elapsed_seconds": round(time.time() - started, 1),
        "producers": list(args.producers), "seeds": list(args.seeds),
        "subjects": subjects,
    }
    atomic_write_json(out_root / "manifest.json", manifest)
    write_status(status, state="finished", n_jobs=len(jobs), done=done,
                 n_failed=manifest["n_failed"],
                 n_resource_failed=manifest["n_resource_failed"])
    print(json.dumps({k: manifest[k] for k in
                      ("n_ok", "n_resource_failed", "n_failed", "elapsed_seconds")},
                     indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="store_true")
    parser.add_argument("--subject")
    parser.add_argument("--producer")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--cohort", choices=("a1", "a2", "all"), default=None)
    parser.add_argument("--producers", nargs="+", default=["P_local", "P_slow"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-train-seconds", type=float, default=5400.0)
    parser.add_argument("--chunk-events", type=int, default=128)
    parser.add_argument("--batch-segments", type=int, default=8)
    args = parser.parse_args()

    if args.job:
        if not (args.subject and args.producer):
            parser.error("--job needs --subject and --producer")
        sys.exit(_run_job(args))
    _queue(args)


if __name__ == "__main__":
    main()
