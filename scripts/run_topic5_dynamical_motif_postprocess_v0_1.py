#!/usr/bin/env python3
"""Post-training queue: model-unseen evaluation, baselines and counterfactuals.

Claims one (patient, seed) at a time with the same atomic-directory protocol as
the trainer, so several workers can share the cohort without repeating a unit.
Every step is skipped when its output already exists, so the queue is safe to
re-run after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic5_dynamical_motif_queue_v0_1 import PYTHON, _alive, chain_units  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import MAIN_MODELS  # noqa: E402

# Low-cost arms scored with the same decoder; they are not formal RNN units.
BASELINE_ARMS = ("LAYOUT_AXIS_ANISOTROPY", "LAYOUT_AXIS_REPLAY", "EVENT_VECTOR_DIRECTIONAL",
                 "GAIN_MATCHED_DM1_FREE_AXIS", "GAIN_MATCHED_DM2_LOCAL_DIRECTIONAL")

OUT_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
STALE_SECONDS = 5400


def claim(lock_root: Path, key: str) -> bool:
    lock = lock_root / key.replace("|", "__")
    try:
        os.mkdir(lock)
    except FileExistsError:
        record_path = lock / "claim.json"
        if not record_path.exists():
            return False
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            return False
        if time.time() - float(record.get("heartbeat", 0)) < STALE_SECONDS:
            return False
        if record.get("host") == socket.gethostname() and _alive(int(record.get("pid", -1))):
            return False
    (lock / "claim.json").write_text(json.dumps({
        "pid": os.getpid(), "host": socket.gethostname(),
        "claimed": time.time(), "heartbeat": time.time(), "key": key}))
    return True


def beat(lock_root: Path, key: str, note: str) -> None:
    path = lock_root / key.replace("|", "__") / "claim.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        record = {"pid": os.getpid(), "host": socket.gethostname()}
    record.update({"heartbeat": time.time(), "note": note})
    path.write_text(json.dumps(record))


def run(command: list[str]) -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = environment.get("CUDA_VISIBLE_DEVICES", "0")
    return int(subprocess.run(command, cwd=str(ROOT), env=environment).returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", default="p0")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--reference-draws", type=int, default=128)
    parser.add_argument("--counterfactual-draws", type=int, default=64)
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--skip-counterfactual", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.out_root / "FORMAL_UNIT_MANIFEST.csv")
    chains = manifest.drop_duplicates(["unit_id", "seed_index"])[["unit_id", "seed_index"]]
    lock_root = args.out_root / "locks" / f"{args.tag}_post"
    lock_root.mkdir(parents=True, exist_ok=True)
    base = args.out_root / args.tag / args.frame
    print(f"[post:{args.worker_id}] {len(chains)} chains", flush=True)

    for _, chain in chains.iterrows():
        unit_id, seed = str(chain.unit_id), int(chain.seed_index)
        key = f"{args.frame}|{unit_id}|seed{seed}"
        models = chain_units(seed)
        if not all((base / unit_id / m / f"seed{seed}" / "checkpoint.pt").exists()
                   for m in models):
            continue
        scorable = list(models) + ([m for m in BASELINE_ARMS
                                    if (base / unit_id / m / f"seed{seed}" / "checkpoint.pt").exists()]
                                   if seed == 0 else [])
        pending = [m for m in scorable
                   if not (base / unit_id / m / f"seed{seed}" / "unseen_evaluation.json").exists()]
        counterfactual_pending = (
            not args.skip_counterfactual and seed == 0
            and any(not (base / unit_id / m / f"seed{seed}"
                         / "counterfactual_summary.json").exists() for m in MAIN_MODELS))
        baseline_pending = seed == 0 and not (
            base / unit_id / f"baselines_seed{seed}.json").exists()
        if not pending and not counterfactual_pending and not baseline_pending:
            continue
        if not claim(lock_root, key):
            continue
        print(f"[post:{args.worker_id}] claimed {key}", flush=True)

        if baseline_pending:
            beat(lock_root, key, "baselines")
            started = time.time()
            code = run([PYTHON, "scripts/run_topic5_dynamical_motif_baselines_v0_1.py",
                        "--frame", args.frame, "--unit-id", unit_id,
                        "--seed-index", str(seed), "--tag", args.tag,
                        "--device", args.device, "--gate-rule", args.gate_rule,
                        "--out-root", str(args.out_root)])
            print(f"[post:{args.worker_id}] {key} baselines rc={code} "
                  f"{time.time() - started:.0f}s", flush=True)

        for model_id in scorable:
            directory = base / unit_id / model_id / f"seed{seed}"
            if not (directory / "checkpoint.pt").exists():
                continue
            if (directory / "unseen_evaluation.json").exists():
                continue
            beat(lock_root, key, f"evaluate:{model_id}")
            started = time.time()
            code = run([PYTHON, "scripts/evaluate_topic5_dynamical_motif_unseen_v0_1.py",
                        "--frame", args.frame, "--unit-id", unit_id, "--model", model_id,
                        "--seed-index", str(seed), "--tag", args.tag,
                        "--device", args.device, "--draws", str(args.draws),
                        "--reference-draws", str(args.reference_draws),
                        "--gate-rule", args.gate_rule, "--out-root", str(args.out_root)])
            print(f"[post:{args.worker_id}] {key}/{model_id} evaluate rc={code} "
                  f"{time.time() - started:.0f}s", flush=True)

        if counterfactual_pending:
            for model_id in MAIN_MODELS:
                directory = base / unit_id / model_id / f"seed{seed}"
                if (directory / "counterfactual_summary.json").exists():
                    continue
                beat(lock_root, key, f"counterfactual:{model_id}")
                started = time.time()
                code = run([PYTHON, "scripts/run_topic5_dynamical_motif_counterfactual_v0_1.py",
                            "--frame", args.frame, "--unit-id", unit_id, "--model", model_id,
                            "--seed-index", str(seed), "--tag", args.tag,
                            "--device", args.device, "--draws", str(args.counterfactual_draws),
                            "--gate-rule", args.gate_rule, "--out-root", str(args.out_root)])
                print(f"[post:{args.worker_id}] {key}/{model_id} counterfactual rc={code} "
                      f"{time.time() - started:.0f}s", flush=True)
        beat(lock_root, key, "chain_finished")
    print(f"[post:{args.worker_id}] pass complete", flush=True)


if __name__ == "__main__":
    main()
