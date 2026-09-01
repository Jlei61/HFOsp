#!/usr/bin/env python3
"""Held-out next-event scores, carrying state vs not carrying it.

SP 8's first allowed conclusion ("only the next event improves -> short-range
predictive filtering") needs a next-event comparison, which the future-block
analysis does not provide.  ``P_local`` and ``P_memoryless`` share an encoder, a
capacity and a local objective; the only difference is whether the state survives
an event.  Scoring both on the development-test events therefore isolates exactly
what carrying buys at the event scale.

Each pass replays whole carry segments from their own start and scores only the
events of the target split, so a held-out chain is warmed causally rather than
starting from initialisation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main")
DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")


def _run_job(args) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from src.topic5_group_event_state.dataset import SubjectSequence
    from src.topic5_group_event_state.v02.producers import (
        build_anchor_targets,
        load_producer,
        run_session_pass,
        segments_touching_split,
        split_event_mask,
    )
    from src.topic5_group_event_state.v02.registry import atomic_write_json
    from src.topic5_group_event_state.v02.subject import (
        SubjectTimelineConfig,
        load_subject_timeline,
    )

    out = (Path(args.producer_root) / "next_event" / args.subject)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.producer}_seed{args.seed}.json"
    if dst.exists() and not args.overwrite:
        print(f"{args.subject}/{args.producer}: already done")
        return 0
    ckpt = (Path(args.producer_root) / "runs" / args.subject / args.producer
            / f"seed{args.seed}" / "checkpoint.pt")
    if not ckpt.exists():
        print(f"MISSING {ckpt}")
        return 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tl = load_subject_timeline(args.subject, config=SubjectTimelineConfig())
    seq = SubjectSequence(DATASET_ROOT / args.subject)
    model, cfg = load_producer(ckpt, tl, seq, device)

    payload = {"subject": args.subject, "producer": args.producer, "seed": args.seed}
    for split in ("test", "val"):
        mask = split_event_mask(tl, split)
        with torch.no_grad():
            means, extra = run_session_pass(
                model, tl, seq, segments_touching_split(tl, split),
                build_anchor_targets(tl, split), device, cfg, train=False,
                rng=np.random.default_rng(0), score_mask=mask,
            )
        payload[split] = {k: v for k, v in means.items() if k.startswith("local.")}
        payload[f"n_{split}_events"] = int(mask.sum())
    atomic_write_json(dst, payload)
    print(f"OK {args.subject}/{args.producer}/seed{args.seed}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="store_true")
    parser.add_argument("--subject")
    parser.add_argument("--producer", default="P_local")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--producers", nargs="+", default=["P_local", "P_memoryless"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.job:
        sys.exit(_run_job(args))

    subjects = sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists())
    jobs = [(s, p, seed) for s in subjects for p in args.producers for seed in args.seeds]
    log_dir = Path(args.producer_root) / "next_event_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    slots: list = []
    pending = list(jobs)
    capacity = len(args.gpus) * args.jobs_per_gpu
    done = 0
    started = time.time()
    failures = []
    while pending or slots:
        while pending and len(slots) < capacity:
            subject, producer, seed = pending.pop(0)
            used = [g for _p, _j, g, _h in slots]
            gpu = min(args.gpus, key=lambda g: used.count(g))
            handle = (log_dir / f"{subject}_{producer}_seed{seed}.log").open("w")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--job",
                   "--subject", subject, "--producer", producer, "--seed", str(seed),
                   "--gpu", str(gpu), "--producer-root", str(args.producer_root)]
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            slots.append((proc, (subject, producer, seed), gpu, handle))
        time.sleep(4.0)
        still = []
        for proc, job, gpu, handle in slots:
            if proc.poll() is None:
                still.append((proc, job, gpu, handle))
                continue
            handle.close()
            done += 1
            if proc.returncode != 0:
                failures.append({"job": job, "returncode": proc.returncode})
            print(f"[{done}/{len(jobs)}] {job} rc={proc.returncode}", flush=True)
        slots = still
    print(json.dumps({"n_jobs": len(jobs), "n_failed": len(failures),
                      "failures": failures,
                      "elapsed_seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
