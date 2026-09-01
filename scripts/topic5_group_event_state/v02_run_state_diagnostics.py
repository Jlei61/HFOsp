#!/usr/bin/env python3
"""A4 memory-truncation probes: replay a frozen producer with a shortened memory.

Two reduced grids, exactly as CC 6 specifies:

    event count   1, 100, 1000, full
    physical time 5 min, 30 min, 120 min, full

"full" is the state the training run already saved, so it is not recomputed.
These are diagnostics: CC 6 decides the history scale from the future-block curve
against real physical horizons, never from "which reset first stops being
significant".
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

EVENT_RESETS = (1, 100, 1000)
TIME_RESETS = (300.0, 1800.0, 7200.0)
DEFAULT_PRODUCER_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main")


def _labels() -> list[tuple[str, int, float]]:
    out = [(f"reset_e{k}", k, 0.0) for k in EVENT_RESETS]
    out += [(f"reset_t{int(t)}", 0, t) for t in TIME_RESETS]
    return out


def _run_job(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from src.topic5_group_event_state.dataset import SubjectSequence
    from src.topic5_group_event_state.v02.contract_paths import DATASET_ROOT
    from src.topic5_group_event_state.v02.producers import load_producer, replay_with_reset
    from src.topic5_group_event_state.v02.subject import (
        SubjectTimelineConfig,
        load_subject_timeline,
    )

    run_dir = (Path(args.producer_root) / "runs" / args.subject / args.producer
               / f"seed{args.seed}")
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        print(f"MISSING checkpoint {ckpt}")
        return 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tl = load_subject_timeline(args.subject, config=SubjectTimelineConfig())
    seq = SubjectSequence(DATASET_ROOT / args.subject)
    model, cfg = load_producer(ckpt, tl, seq, device)

    for label, n_events, seconds in _labels():
        out_dir = (Path(args.producer_root) / "states_diag"
                   / f"{args.producer}_seed{args.seed}_{label}")
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{args.subject}.npz"
        if dst.exists() and not args.overwrite:
            continue
        started = time.time()
        states = replay_with_reset(
            model, tl, seq, device, cfg,
            reset_every_events=n_events, reset_every_seconds=seconds,
        )
        tmp = dst.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, state=states.astype(np.float32),
                     t_anchor=tl.grid.t_anchor, split_index=tl.grid.split_index,
                     session_id=tl.grid.session_id)
        os.replace(tmp, dst)
        print(f"  {label}: {time.time() - started:.0f}s", flush=True)
    print(f"OK {args.subject}/{args.producer}/seed{args.seed}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="store_true")
    parser.add_argument("--subject")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--producer", default="P_slow")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_PRODUCER_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.job:
        sys.exit(_run_job(args))

    from src.topic5_group_event_state.v02.contract_paths import DATASET_ROOT
    from src.topic5_group_event_state.v02.registry import atomic_write_json

    subjects = args.subjects or sorted(
        p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists()
    )
    log_dir = Path(args.producer_root) / "diag_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(subjects)
    slots: list = []
    capacity = len(args.gpus) * args.jobs_per_gpu
    results = []
    started = time.time()
    while pending or slots:
        while pending and len(slots) < capacity:
            subject = pending.pop(0)
            used = [g for _p, _s, g, _h in slots]
            gpu = min(args.gpus, key=lambda g: used.count(g))
            handle = (log_dir / f"{subject}_{args.producer}_seed{args.seed}.log").open("w")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--job",
                   "--subject", subject, "--producer", args.producer,
                   "--seed", str(args.seed), "--gpu", str(gpu),
                   "--producer-root", str(args.producer_root)]
            if args.overwrite:
                cmd.append("--overwrite")
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            slots.append((proc, subject, gpu, handle))
            print(f"launch {subject} gpu{gpu} pid={proc.pid}", flush=True)
        time.sleep(5.0)
        still = []
        for proc, subject, gpu, handle in slots:
            if proc.poll() is None:
                still.append((proc, subject, gpu, handle))
                continue
            handle.close()
            results.append({"subject": subject, "returncode": proc.returncode})
            print(f"done {subject} rc={proc.returncode}", flush=True)
        slots = still
    atomic_write_json(Path(args.producer_root) / f"diagnostics_manifest_{args.producer}_seed{args.seed}.json", {
        "producer": args.producer, "seed": args.seed, "subjects": sorted(subjects),
        "labels": [lbl for lbl, _e, _t in _labels()], "results": results,
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({"n_ok": sum(1 for r in results if r["returncode"] == 0),
                      "n_failed": sum(1 for r in results if r["returncode"] != 0)}))


if __name__ == "__main__":
    main()
