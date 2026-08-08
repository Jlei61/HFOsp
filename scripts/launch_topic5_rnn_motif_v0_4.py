"""Bounded, resumable launcher for the locked v0.4 model matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
sys.path.insert(0, str(ROOT))

from src.topic5_rnn_motif_v0_4 import CORE_IDS, DOSE_IDS, GRU_IDS, MODEL_SPECS  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def fit_rows(out_root: Path) -> list[dict[str, Any]]:
    return json.loads((out_root / "INPUT_MANIFEST.json").read_text())["fits"]


def development_fits(out_root: Path) -> list[str]:
    rows = sorted(fit_rows(out_root), key=lambda row: (row["n_contacts"], row["fit_id"]))
    return [rows[0]["fit_id"], rows[len(rows) // 2]["fit_id"], rows[-1]["fit_id"]]


def run_id(model_id: str, cell: str) -> str:
    return f"{model_id}__{cell}"


def unit_directory(out_root: Path, fit_id: str, model_id: str, cell: str, seed: int) -> Path:
    return out_root / "per_subject" / fit_id / run_id(model_id, cell) / f"seed{seed}"


def build_jobs(out_root: Path, stage: str) -> list[dict[str, Any]]:
    all_fits = [row["fit_id"] for row in fit_rows(out_root)]
    jobs: list[dict[str, Any]] = []
    if stage == "smoke":
        for fit_id in development_fits(out_root):
            for model_id in ("M0_NO_REC", "M3_FIXED_LOCAL", "M6_SPATIAL_MID"):
                jobs.append({"fit_id": fit_id, "model_id": f"SMOKE_{model_id}",
                             "spec_id": model_id, "cell": "rnn", "seed": 0,
                             "epochs_freeze": 30})
    elif stage == "core":
        for fit_id in all_fits:
            for model_id in CORE_IDS:
                for seed in MODEL_SPECS[model_id].seeds:
                    jobs.append({"fit_id": fit_id, "model_id": model_id,
                                 "spec_id": model_id, "cell": "rnn", "seed": seed})
    elif stage == "dose":
        for fit_id in all_fits:
            for model_id in DOSE_IDS:
                for seed in MODEL_SPECS[model_id].seeds:
                    jobs.append({"fit_id": fit_id, "model_id": model_id,
                                 "spec_id": model_id, "cell": "rnn", "seed": seed})
    elif stage == "gru":
        for fit_id in all_fits:
            for model_id in GRU_IDS:
                for seed in MODEL_SPECS[model_id].seeds:
                    jobs.append({"fit_id": fit_id, "model_id": model_id,
                                 "spec_id": model_id, "cell": "gru", "seed": seed})
    else:
        raise ValueError(f"unknown stage {stage!r}")
    for job in jobs:
        job["out_dir"] = unit_directory(out_root, job["fit_id"], job["model_id"],
                                         job["cell"], job["seed"])
    return jobs


def command(job: dict[str, Any], out_root: Path, device: str) -> list[str]:
    spec = MODEL_SPECS[job["spec_id"]]
    cmd = [
        PYTHON, str(ROOT / "scripts/train_topic5_we_unit.py"),
        "--fit-id", job["fit_id"], "--arm", spec.arm,
        "--cell", job["cell"], "--seed", str(job["seed"]),
        "--device", device, "--out-root", str(out_root),
        "--model-id", run_id(job["model_id"], job["cell"]),
        "--eta", str(spec.eta),
    ]
    if job["spec_id"] == "C_ORDER_SHUFFLED":
        cmd += ["--shuffle-mode", "keep_first"]
    elif job["spec_id"] == "C_FULL_RANK_SHUFFLED":
        cmd += ["--shuffle-mode", "full"]
    if "epochs_freeze" in job:
        cmd += ["--epochs-freeze", str(job["epochs_freeze"])]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "core", "dose", "gru"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    jobs = build_jobs(out_root, args.stage)
    todo = [job for job in jobs if not (job["out_dir"] / "DONE.json").exists()]
    if args.dry_run:
        print(f"{args.stage}: {len(jobs)} total, {len(todo)} pending")
        for job in todo[:12]:
            print(" ".join(command(job, out_root, args.device)))
        return 0

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if args.device.startswith("cuda") and not gpus:
        raise ValueError("CUDA stage requires at least one GPU id")
    logs = out_root / "logs" / args.stage
    logs.mkdir(parents=True, exist_ok=True)
    status_path = out_root / f"STAGE_{args.stage.upper()}_STATUS.json"
    active: list[tuple[subprocess.Popen, dict[str, Any], Any, float]] = []
    done = failed = oom = nonfinite = 0
    started = time.time()
    launched = 0

    def status() -> None:
        write_json(status_path, {
            "stage": args.stage, "total": len(jobs),
            "already_done": len(jobs) - len(todo), "launched": launched,
            "completed_this_run": done, "failed": failed, "oom": oom,
            "nonfinite": nonfinite, "active": len(active),
            "remaining": max(0, len(todo) - done - failed - len(active)),
            "elapsed_seconds": round(time.time() - started, 1),
            "updated_at_epoch": time.time(),
        })

    def reap() -> None:
        nonlocal done, failed, oom, nonfinite
        for entry in list(active):
            process, job, handle, _ = entry
            if process.poll() is None:
                continue
            handle.close()
            active.remove(entry)
            log_path = Path(job["log_path"])
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            if "out of memory" in log_text.lower():
                oom += 1
            if "nan" in log_text.lower() or "nonfinite" in log_text.lower():
                nonfinite += 1
            if process.returncode == 0 and (job["out_dir"] / "DONE.json").exists():
                done += 1
            else:
                failed += 1
            status()

    status()
    for index, job in enumerate(todo):
        while len(active) >= int(args.max_workers):
            reap()
            time.sleep(1.0)
        gpu = gpus[index % len(gpus)] if gpus else ""
        name = f"{job['fit_id']}__{job['model_id']}__{job['cell']}__seed{job['seed']}"
        log_path = logs / f"{name}.log"
        job["log_path"] = str(log_path)
        handle = log_path.open("w")
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(command(job, out_root, args.device), stdout=handle,
                                   stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        active.append((process, job, handle, time.time()))
        launched += 1
        status()
    while active:
        reap()
        time.sleep(1.0)
    status()
    write_json(out_root / f"STAGE_{args.stage.upper()}_{'COMPLETE' if failed == 0 else 'FAILED'}.json", {
        "stage": args.stage, "total": len(jobs), "failed": failed, "oom": oom,
        "nonfinite": nonfinite, "elapsed_seconds": round(time.time() - started, 1),
    })
    print(f"{args.stage}: {len(jobs)} total; {done} completed now; {failed} failed; "
          f"{oom} OOM; {nonfinite} nonfinite")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
