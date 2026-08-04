#!/usr/bin/env python3
"""Target-sealed training sweep for the v0.3 structured model.

Only the three development patients are used and only their interictal
validation20 contact NLL is read.  No early-ictal value and no test20 value
enters the selection, so this cannot tune the model towards the frozen
cross-state readout.

Both the structured model and the ordinary GRU are swept, because the open
question is whether the gap between them is a training-budget artefact
rather than a property of the architectures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_topic5_shared_scaffold_rnn_unit_v0_2.py"


# Micro-batch size is deliberately absent from this grid: the frozen runner
# exposes no override for it, so a batch axis here would be recorded in the
# output without ever reaching the training run.
def tag(cycles: int, updates: int, rate: float) -> str:
    return f"c{cycles}_u{updates}_lr{rate:g}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_source_conditioned_shared_scaffold_rnn_v0_3_final.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=["structured", "ordinary_gru"])
    parser.add_argument("--cycles", nargs="*", type=int, default=[7, 28, 84])
    parser.add_argument("--updates", nargs="*", type=int, default=[32])
    parser.add_argument("--rates", nargs="*", type=float, default=[0.03])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    subjects = list(map(str, config["development_lr_audit"]["subjects"]))
    seed = int(config["development_lr_audit"]["seed"])
    root = args.output_root.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.cycles, args.updates, args.rates))

    def execute(model: str, subject: str, cell: tuple) -> dict:
        cycles, updates, rate = cell
        cell_root = root / tag(cycles, updates, rate)
        done = cell_root / "per_subject" / subject / model / f"seed_{seed}" / "DONE.json"
        if not done.exists():
            command = [
                str(config["resources"]["python"]), str(RUNNER),
                "--config", str(config_path), "--subject", subject,
                "--model", model, "--seed", str(seed),
                "--device", "cuda:0", "--output-root", str(cell_root),
                "--learning-rate", str(rate),
                "--coverage-cycles", str(cycles),
                "--optimizer-updates-per-cycle", str(updates),
                "--resume",
            ]
            environment = os.environ.copy()
            environment.update(
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            log = logs / f"{model}__{subject}__{tag(cycles, updates, rate)}.log"
            with log.open("a") as handle:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=handle,
                    stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{model} {subject} {cell}: rc={result.returncode}")
        payload = json.loads(done.read_text())
        return {
            "model": model,
            "subject": subject,
            "coverage_cycles": cycles,
            "optimizer_updates_per_cycle": updates,
            "learning_rate": rate,
            # Selection reads validation only; test is carried for audit and
            # must never be used to choose a cell.
            "validation_contact_nll": float(
                payload["validation"]["contact_nll_per_continue_decision"]
            ),
            "validation_top1": float(payload["validation"]["top1_next_contact_accuracy"]),
            "best_cycle": int(payload["best_cycle"]),
            "runtime_seconds": float(payload["runtime_seconds"]),
            "target_values_read": False,
        }

    tasks = [
        (model, subject, cell)
        for model in args.models
        for subject in subjects
        for cell in grid
    ]
    rows = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(execute, *task): task for task in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"[{len(rows)}/{len(tasks)}]", flush=True)

    summary = {}
    for model in args.models:
        for cell in grid:
            cycles, updates, rate = cell
            values = [
                row["validation_contact_nll"] for row in rows
                if row["model"] == model
                and (row["coverage_cycles"], row["optimizer_updates_per_cycle"],
                     row["learning_rate"]) == cell
            ]
            summary[f"{model}__{tag(*cell)}"] = float(np.median(values))
    structured = {k: v for k, v in summary.items() if k.startswith("structured__")}
    selected = min(structured, key=lambda key: (structured[key], key)) if structured else None
    payload = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "subjects": subjects,
        "seed": seed,
        "grid": [list(cell) for cell in grid],
        "selection_metric": "median_validation_contact_nll_across_three_patients",
        "median_validation_contact_nll": summary,
        "selected_structured_cell": selected,
        "test_values_used_for_selection": False,
        "ictal_target_values_read": False,
        "rows": sorted(rows, key=lambda row: (row["model"], row["subject"], row["coverage_cycles"])),
        "runtime_seconds": time.time() - started,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    }
    atomic_json(root / "SWEEP_SELECTION.json", payload)
    print(json.dumps({k: payload[k] for k in
                      ("status", "median_validation_contact_nll", "selected_structured_cell")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
