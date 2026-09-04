#!/usr/bin/env python3
"""Extend a locked W3 recipe when its INNER optimum is still budget-bound."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS, OUTPUT_ROOT, atomic_json  # noqa: E402


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECTS = ("epilepsiae_253", "epilepsiae_548", "epilepsiae_583", "epilepsiae_1096")


def _cards(root: Path) -> dict[tuple[str, int], dict]:
    output = {}
    for path in root.glob("*/decoder_seed*_state_seed*/card.json"):
        card = json.loads(path.read_text(encoding="utf-8"))
        if card.get("selection", {}).get("status") != "HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH":
            raise ValueError(f"budget audit found an opened SELECTION card: {path}")
        output[(card["subject"], int(card["seed"]))] = card
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()
    lock_path = OUTPUT_ROOT / "full_mark_search" / "selected_recipe.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    recipe = str(lock["selected_recipe"])
    source_root = OUTPUT_ROOT / "full_mark_search" / recipe
    config_path = ROOT / "config/group_event_state_v035_search" / f"{recipe}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original = _cards(source_root)
    expected = {(subject, seed) for subject in SUBJECTS for seed in LOCKED_SEEDS[:3]}
    if set(original) != expected:
        raise RuntimeError(f"selected recipe is incomplete: missing {sorted(expected - set(original))}")
    max_epochs = int(config.get("max_epochs", 24))
    boundary = [
        {"subject": subject, "seed": seed, "selected_epoch": int(card["selected_epoch"])}
        for (subject, seed), card in original.items()
        if int(card["selected_epoch"]) >= max_epochs - 2
    ]
    audit_root = OUTPUT_ROOT / "full_mark_search_budget_extension"
    audit_root.mkdir(parents=True, exist_ok=True)
    if not boundary:
        payload = {
            "format": "group_event_state_v0_3_5_full_mark_budget_audit_v1",
            "status": "ORIGINAL_BUDGET_ADEQUATE",
            "selected_recipe": recipe,
            "boundary_units": [],
            "original_max_epochs": max_epochs,
            "final_config": str(config_path),
            "final_source_root": str(source_root),
            "selection_targets_read": False,
            "development_targets_read": False,
            "sealed_partition_opened": False,
        }
        atomic_json(audit_root / "budget_audit.json", payload)
        print(json.dumps(payload, indent=2))
        return

    extended = dict(config)
    extended["max_epochs"] = max(max_epochs * 3, max_epochs + 24)
    extended["patience"] = max(int(config.get("patience", 4)), 6)
    extended_config = audit_root / f"{recipe}_extended_config.json"
    atomic_json(extended_config, extended)
    out_root = audit_root / recipe
    control = audit_root / "supervisor"
    logs = control / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = []
    for subject in SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            path = out_root / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "card.json"
            pending.append({
                "subject": subject,
                "decoder_seed": decoder_seed,
                "state_seed": state_seed,
                "out": str(path),
                "chunk_events": 256,
                "retries": 0,
            })
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    slots = [(f"{gpu}:{worker}", gpu) for gpu in gpus for worker in range(args.workers_per_gpu)]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    running, complete, failed = {}, [], []
    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job = row["job"]
            body = Path(row["log"]).read_text(encoding="utf-8", errors="replace")[-20000:]
            if code == 0 and Path(job["out"]).exists():
                complete.append(job)
            elif "out of memory" in body.lower() and job["chunk_events"] > 32 and job["retries"] < 3:
                job["chunk_events"] //= 2
                job["retries"] += 1
                pending.insert(0, job)
            else:
                failed.append({**job, "returncode": code, "log": row["log"], "tail": body[-3000:]})
            del running[slot]
        for slot, gpu in slots:
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            if Path(job["out"]).exists():
                complete.append(job)
                continue
            log = logs / f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            command = [
                str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"),
                "--subject", job["subject"],
                "--decoder-seed", str(job["decoder_seed"]),
                "--state-seed", str(job["state_seed"]),
                "--chunk-events", str(job["chunk_events"]),
                "--config-json", str(extended_config),
                "--hold-selection",
                "--out-root", str(out_root),
                "--device", f"cuda:{gpu}",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[slot] = {
                "process": process, "handle": handle, "job": job,
                "log": str(log), "started": time.time(), "gpu": gpu,
            }
        atomic_json(control / "queue_state.json", {
            "format": "group_event_state_v0_3_5_full_mark_budget_extension_queue_v1",
            "status": "RUNNING", "updated_epoch": time.time(),
            "pending": len(pending), "complete": len(complete), "failed": failed,
            "running": {
                slot: {"pid": row["process"].pid, "gpu": row["gpu"],
                       "job": row["job"], "elapsed_seconds": time.time() - row["started"]}
                for slot, row in running.items()
            },
            "selection_targets_read": False,
        })
        if pending or running:
            time.sleep(15)
    if failed:
        atomic_json(control / "queue_done.json", {"status": "FAILED", "failed": failed})
        raise SystemExit(1)
    new = _cards(out_root)
    if set(new) != expected:
        raise RuntimeError("extended recipe did not produce the full paired unit set")
    paired = []
    for key in sorted(expected):
        paired.append({
            "subject": key[0], "seed": key[1],
            "original_inner_loss": float(original[key]["best_inner_loss"]),
            "extended_inner_loss": float(new[key]["best_inner_loss"]),
            "delta_extended_minus_original": float(new[key]["best_inner_loss"] - original[key]["best_inner_loss"]),
            "extended_selected_epoch": int(new[key]["selected_epoch"]),
        })
    use_extended = float(np.median([row["delta_extended_minus_original"] for row in paired])) <= 0.0
    payload = {
        "format": "group_event_state_v0_3_5_full_mark_budget_audit_v1",
        "status": "EXTENDED_SELECTED" if use_extended else "ORIGINAL_RETAINED_AFTER_EXTENSION",
        "selected_recipe": recipe,
        "boundary_units": boundary,
        "original_max_epochs": max_epochs,
        "extended_max_epochs": int(extended["max_epochs"]),
        "paired_units": paired,
        "median_extended_minus_original": float(np.median([row["delta_extended_minus_original"] for row in paired])),
        "final_config": str(extended_config if use_extended else config_path),
        "final_source_root": str(out_root if use_extended else source_root),
        "selection_targets_read": False,
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(audit_root / "budget_audit.json", payload)
    atomic_json(control / "queue_done.json", {"status": "DONE", "complete": complete, "failed": []})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
