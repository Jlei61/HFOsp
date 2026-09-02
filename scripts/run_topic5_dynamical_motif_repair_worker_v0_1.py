#!/usr/bin/env python3
"""Partitioned, resumable worker for the motif-RNN repair experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
RESULT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
FRAME = "GEOMETRY_ONLY_PCA2"
MODELS = (
    "DM0_ISOTROPIC",
    "DM1_FREE_AXIS",
    "DM2_LOCAL_DIRECTIONAL",
    "DM3_AXIS_FEEDFORWARD_TRANSIENT",
)


def run(command: list[str]) -> None:
    print("[repair-worker]", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"command failed with return code {completed.returncode}")


def contact_selected(unit_id: str, device: str) -> None:
    for model in MODELS:
        directory = RESULT / "contact_selected" / FRAME / unit_id / model / "seed0"
        if (directory / "DONE.json").exists() and (directory / "checkpoint.pt").exists():
            continue
        command = [
            PYTHON, str(ROOT / "scripts/train_topic5_dynamical_motif_unit_v0_1.py"),
            "--frame", FRAME, "--unit-id", unit_id, "--model", model,
            "--seed-index", "0", "--device", device, "--tag", "contact_selected",
            "--selection-metric", "contact_nll", "--out-root", str(RESULT),
        ]
        if model != MODELS[0]:
            parent = MODELS[MODELS.index(model) - 1]
            command += ["--warm-start", str(
                RESULT / "contact_selected" / FRAME / unit_id / parent
                / "seed0/checkpoint.pt")]
        run(command)


def baseline(unit_id: str, device: str, seed_indices: list[int]) -> None:
    for seed_index in seed_indices:
        output = (RESULT / "formal" / FRAME / unit_id
                  / f"capacity_matched_static_seed{seed_index}.json")
        if output.exists():
            continue
        run([
            PYTHON, str(ROOT / "scripts/run_topic5_dynamical_motif_baselines_v0_1.py"),
            "--frame", FRAME, "--unit-id", unit_id,
            "--seed-index", str(seed_index),
            "--device", device, "--tag", "formal", "--out-root", str(RESULT),
            "--static-only", "--matched-only",
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contact", "baseline"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-indices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    census = pd.read_csv(RESULT / "GEOMETRY_ONLY_FIT_CENSUS.csv")
    ordered = census.assign(
        cost=census["n_events"] * census["n_nodes"]).sort_values("cost", ascending=False)
    units = ordered["subject"].astype(str).tolist()[args.worker_index::args.workers]
    for unit_id in units:
        print(f"[repair-worker] mode={args.mode} unit={unit_id}", flush=True)
        if args.mode == "contact":
            contact_selected(unit_id, args.device)
        else:
            baseline(unit_id, args.device, args.seed_indices)
    print(f"[repair-worker] mode={args.mode} worker={args.worker_index} DONE", flush=True)


if __name__ == "__main__":
    main()
