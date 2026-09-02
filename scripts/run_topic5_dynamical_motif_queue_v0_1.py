#!/usr/bin/env python3
"""Idempotent chain queue for the Topic 5.2 dynamical motif training units.

A unit cannot start before the layer it warm-starts from, so the schedulable
object is a *seed chain* rather than a single unit:

    (unit_id, seed) -> DM0 -> DM1 -> DM2 -> DM3            (4 units)
    (unit_id, seed 0) additionally -> the three M3 controls (3 units)

28 patients x 3 seeds x 4 main + 28 patients x 3 controls = 420 units in
84 chains.  Chains are independent, so workers never wait on each other.
Claims are atomic directory creations; a stale claim is only reclaimed after
the recorded pid is confirmed dead.
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

from src.topic5_dynamical_motif_rnn_v0_1 import M3_CONTROLS, MAIN_MODELS  # noqa: E402

OUT_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
TRAINER = ROOT / "scripts/train_topic5_dynamical_motif_unit_v0_1.py"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
STALE_SECONDS = 3600


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def chain_units(seed_index: int) -> list[str]:
    return list(MAIN_MODELS) + (list(M3_CONTROLS) if seed_index == 0 else [])


def build_manifest(out_root: Path, frame: str, seeds: int, subjects: list[str]) -> pd.DataFrame:
    rows = []
    for unit_id in subjects:
        for seed_index in range(seeds):
            for model_id in chain_units(seed_index):
                rows.append({
                    "frame": frame, "unit_id": unit_id, "seed_index": seed_index,
                    "model_id": model_id,
                    "chain_id": f"{frame}|{unit_id}|seed{seed_index}",
                    "warm_start_from": (
                        "DM2_LOCAL_DIRECTIONAL" if model_id in M3_CONTROLS
                        else {"DM0_ISOTROPIC": None, "DM1_FREE_AXIS": "DM0_ISOTROPIC",
                              "DM2_LOCAL_DIRECTIONAL": "DM1_FREE_AXIS",
                              "DM3_AXIS_FEEDFORWARD_TRANSIENT": "DM2_LOCAL_DIRECTIONAL"}[model_id]
                    ),
                    "counts_toward_420": True,
                })
    frame_rows = pd.DataFrame(rows)
    name = ("FORMAL_UNIT_MANIFEST.csv" if frame == "GEOMETRY_ONLY_PCA2"
            else f"UNIT_MANIFEST_{frame}.csv")
    frame_rows.to_csv(out_root / name, index=False)
    return frame_rows


def unit_dir(out_root: Path, tag: str, frame: str, unit_id: str, model_id: str, seed_index: int) -> Path:
    return out_root / tag / frame / unit_id / model_id / f"seed{seed_index}"


def unit_state(directory: Path) -> str:
    if (directory / "DONE.json").exists() and (directory / "checkpoint.pt").exists():
        return "DONE"
    if (directory / "FAILED.json").exists():
        return "FAILED"
    return "PENDING"


def claim(lock_root: Path, chain_id: str) -> bool:
    lock = lock_root / chain_id.replace("|", "__")
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
        print(f"[queue] reclaiming stale chain {chain_id} from pid {record.get('pid')}", flush=True)
    (lock / "claim.json").write_text(json.dumps({
        "pid": os.getpid(), "host": socket.gethostname(),
        "claimed": time.time(), "heartbeat": time.time(), "chain_id": chain_id,
    }))
    return True


def beat(lock_root: Path, chain_id: str, note: str) -> None:
    path = lock_root / chain_id.replace("|", "__") / "claim.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        record = {"pid": os.getpid(), "host": socket.gethostname()}
    record.update({"heartbeat": time.time(), "note": note})
    path.write_text(json.dumps(record))


def run_unit(out_root: Path, tag: str, frame: str, unit_id: str, model_id: str,
             seed_index: int, device: str, gate_rule: str, selection_metric: str,
             extra: list[str]) -> int:
    directory = unit_dir(out_root, tag, frame, unit_id, model_id, seed_index)
    if unit_state(directory) == "DONE":
        return 0
    warm = None
    parent = None
    if model_id in M3_CONTROLS:
        parent = "DM2_LOCAL_DIRECTIONAL"
    elif model_id != "DM0_ISOTROPIC":
        parent = {"DM1_FREE_AXIS": "DM0_ISOTROPIC",
                  "DM2_LOCAL_DIRECTIONAL": "DM1_FREE_AXIS",
                  "DM3_AXIS_FEEDFORWARD_TRANSIENT": "DM2_LOCAL_DIRECTIONAL"}[model_id]
    if parent is not None:
        warm = unit_dir(out_root, tag, frame, unit_id, parent, seed_index) / "checkpoint.pt"
        if not warm.exists():
            raise RuntimeError(f"missing warm start {warm}")
    command = [PYTHON, str(TRAINER), "--frame", frame, "--unit-id", unit_id,
               "--model", model_id, "--seed-index", str(seed_index), "--device", device,
               "--gate-rule", gate_rule, "--selection-metric", selection_metric,
               "--tag", tag, "--out-root", str(out_root)]
    if warm is not None:
        command += ["--warm-start", str(warm)]
    command += extra
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = environment.get("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(command, cwd=str(ROOT), env=environment)
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--selection-metric", choices=("joint", "contact_nll"),
                        default="joint")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--worker-id", default="w0")
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    out_root = args.out_root
    if args.frame == "PARENT_FROZEN_FRAME":
        parent = pd.read_csv(out_root.parent / "topic5_multiscale_effective_scaffold_v0_5"
                             / "FULL_PARENT_FIT_CENSUS.csv")
        # The parent census already has a patient-level ``subject``; the unit
        # here is the fit (own_a / own_b / shared), so drop it before renaming.
        census = parent.drop(columns=["subject"]).rename(
            columns={"fit_id": "subject"})[["subject", "n_events", "n_nodes"]]
    else:
        census = pd.read_csv(out_root / "GEOMETRY_ONLY_FIT_CENSUS.csv")
    subjects = args.subjects or sorted(census.subject.astype(str).tolist())
    # Largest patients first so one worker does not inherit every long chain at
    # the end of the queue.
    cost = census.set_index("subject")[["n_events", "n_nodes"]]
    subjects = sorted(subjects, key=lambda s: -(int(cost.loc[s, "n_events"])
                                                * int(cost.loc[s, "n_nodes"])))
    manifest = build_manifest(out_root, args.frame, args.seeds, subjects)
    if args.build_manifest_only:
        print(f"[queue] manifest rows={len(manifest)}")
        return

    lock_root = out_root / "locks" / args.tag
    lock_root.mkdir(parents=True, exist_ok=True)
    chains = manifest.drop_duplicates("chain_id")[["chain_id", "unit_id", "seed_index"]]
    print(f"[queue:{args.worker_id}] {len(chains)} chains, device={args.device}", flush=True)
    for _, chain in chains.iterrows():
        chain_id = str(chain.chain_id)
        models = chain_units(int(chain.seed_index))
        if all(unit_state(unit_dir(out_root, args.tag, args.frame, str(chain.unit_id), m,
                                   int(chain.seed_index))) == "DONE" for m in models):
            continue
        if not claim(lock_root, chain_id):
            continue
        print(f"[queue:{args.worker_id}] claimed {chain_id}", flush=True)
        for model_id in models:
            beat(lock_root, chain_id, model_id)
            started = time.time()
            try:
                code = run_unit(out_root, args.tag, args.frame, str(chain.unit_id), model_id,
                                int(chain.seed_index), args.device, args.gate_rule,
                                args.selection_metric, list(args.extra))
            except Exception as error:  # noqa: BLE001 - record and move to the next chain
                print(f"[queue:{args.worker_id}] {chain_id}/{model_id} error {error}", flush=True)
                break
            print(f"[queue:{args.worker_id}] {chain_id}/{model_id} rc={code} "
                  f"{time.time() - started:.0f}s", flush=True)
            if code != 0:
                break
        beat(lock_root, chain_id, "chain_finished")
    print(f"[queue:{args.worker_id}] queue pass complete", flush=True)


if __name__ == "__main__":
    main()
