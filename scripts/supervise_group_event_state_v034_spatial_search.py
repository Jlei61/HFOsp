#!/usr/bin/env python3
"""Run the bounded v0.3.4 S_P TRAIN/STATE_SELECTION search across GPUs.

The supervisor never opens development, seizure or sealed outcomes.  It skips
completed cards, resumes interrupted cells, limits per-GPU concurrency, and
atomically records queue state so it can itself be restarted safely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v034_spatial_state.contracts import (
    TUNING_SUBJECTS,
    lr_search_cells,
    require_human_release_gate,
)


OUTPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/spatial_state_seedfixed")
ARCHITECTURES = ((32, 1), (64, 2), (128, 2), (64, 4))


@dataclass(frozen=True)
class Job:
    output_root: Path
    subject: str
    width: int
    depth: int
    lr_encoder: float
    lr_state_adapter: float
    lr_auxiliary: float
    seed: int
    rung: int

    @property
    def cell(self) -> str:
        return (
            f"w{self.width}_d{self.depth}_le{self.lr_encoder:g}_"
            f"la{self.lr_state_adapter:g}_lx{self.lr_auxiliary:g}_seed{self.seed}"
        )

    @property
    def output_dir(self) -> Path:
        return self.output_root / "human" / self.subject / f"rung{self.rung}" / self.cell

    def record(self) -> dict:
        return {
            "subject": self.subject,
            "width": self.width,
            "depth": self.depth,
            "lr_encoder": self.lr_encoder,
            "lr_state_adapter": self.lr_state_adapter,
            "lr_auxiliary": self.lr_auxiliary,
            "seed": self.seed,
            "rung": self.rung,
            "cell": self.cell,
            "output_dir": str(self.output_dir),
        }


def _card_passes(job: Job) -> bool:
    path = job.output_dir / "training_card.json"
    if not path.is_file():
        return False
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
        contract = card["contract"]
        return (
            card.get("status") == "PASS"
            and contract.get("subject") == job.subject
            and int(contract["arch"]["width"]) == job.width
            and int(contract["arch"]["depth"]) == job.depth
            and int(contract["train"]["max_steps"]) == job.rung
            and int(contract["train"]["seed"]) == job.seed
            and card.get("development_targets_read") is False
            and card.get("sealed_partition_opened") is False
            and card.get("seizure_outcomes_read") is False
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _jobs(output_root: Path, subjects: tuple[str, ...], rung: int, seed: int) -> list[Job]:
    return [
        Job(output_root, subject, width, depth, float(cell["lr_encoder"]),
            float(cell["lr_state_adapter"]), float(cell["lr_auxiliary"]), seed, rung)
        for subject in subjects
        for width, depth in ARCHITECTURES
        for cell in lr_search_cells()
    ]


def _command(job: Job, gate: Path, python: Path) -> list[str]:
    cmd = [
        str(python), str(ROOT / "scripts/run_group_event_state_v034_spatial_state.py"),
        "human", "--subject", job.subject, "--device", "cuda:0",
        "--rung", str(job.rung), "--width", str(job.width), "--depth", str(job.depth),
        "--lr-encoder", str(job.lr_encoder),
        "--lr-state-adapter", str(job.lr_state_adapter),
        "--lr-auxiliary", str(job.lr_auxiliary),
        "--seed", str(job.seed), "--gate-manifest", str(gate),
        "--output-dir", str(job.output_dir),
    ]
    if (job.output_dir / "resume.pt").is_file():
        cmd.append("--resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=TUNING_SUBJECTS,
                        default=list(TUNING_SUBJECTS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--rung", type=int, choices=(300, 900, 2700), default=300)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--python", type=Path,
                        default=Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--status-json", type=Path,
                        default=OUTPUT_ROOT / "manifests/spatial_search_status.json")
    parser.add_argument("--log-dir", type=Path,
                        default=OUTPUT_ROOT / "logs/spatial_search")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers_per_gpu < 1 or not args.gpus:
        parser.error("workers-per-gpu and gpus must be non-empty and positive")
    for subject in args.subjects:
        require_human_release_gate(args.gate_manifest, subject=subject)

    jobs = _jobs(args.output_root, tuple(args.subjects), args.rung, args.seed)
    completed = [job for job in jobs if _card_passes(job)]
    pending = [job for job in jobs if not _card_passes(job)]
    initial = {
        "format": "group_event_state_v0_3_4_spatial_search_supervisor_v1",
        "status": "DRY_RUN" if args.dry_run else "ACTIVE",
        "total": len(jobs),
        "already_completed": len(completed),
        "pending": len(pending),
        "subjects": list(args.subjects),
        "architectures": [list(x) for x in ARCHITECTURES],
        "lr_cells": lr_search_cells(),
        "rung": args.rung,
        "seed": args.seed,
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "output_root": str(args.output_root),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(args.status_json, initial)
    if args.dry_run:
        print(json.dumps(initial, indent=2))
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.status_json.with_suffix(args.status_json.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another spatial search supervisor owns {lock_path}") from exc

    running: dict[int, list[tuple[Job, subprocess.Popen, object, Path]]] = {
        gpu: [] for gpu in args.gpus
    }
    failures: list[dict] = []
    while pending or any(running.values()):
        for gpu in args.gpus:
            while pending and len(running[gpu]) < args.workers_per_gpu:
                job = pending.pop(0)
                log_path = args.log_dir / f"{job.subject}_{job.cell}_rung{job.rung}_gpu{gpu}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                env = dict(os.environ)
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                })
                proc = subprocess.Popen(
                    _command(job, args.gate_manifest, args.python), cwd=ROOT,
                    stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                    start_new_session=True,
                )
                running[gpu].append((job, proc, log_handle, log_path))

        for gpu in args.gpus:
            survivors = []
            for job, proc, handle, log_path in running[gpu]:
                code = proc.poll()
                if code is None:
                    survivors.append((job, proc, handle, log_path))
                    continue
                handle.close()
                if code == 0 and _card_passes(job):
                    completed.append(job)
                else:
                    failures.append({**job.record(), "returncode": code, "log": str(log_path)})
            running[gpu] = survivors

        payload = {
            **initial,
            "status": "ACTIVE" if pending or any(running.values()) else (
                "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES"
            ),
            "completed": len(completed),
            "pending": len(pending),
            "running": {
                str(gpu): [{**job.record(), "pid": proc.pid} for job, proc, _h, _p in rows]
                for gpu, rows in running.items()
            },
            "failures": failures,
            "updated_unix": time.time(),
        }
        atomic_write_json(args.status_json, payload)
        if pending or any(running.values()):
            time.sleep(args.poll_seconds)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
