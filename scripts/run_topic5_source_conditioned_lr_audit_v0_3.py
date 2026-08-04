#!/usr/bin/env python3
"""Target-sealed three-patient learning-rate audit for the v0.3 model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_topic5_shared_scaffold_rnn_unit_v0_2.py"


def tag(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "")


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
        default=ROOT / "config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml",
    )
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    audit = config["development_lr_audit"]
    subjects = list(map(str, audit["subjects"]))
    rates = list(map(float, audit["learning_rates"]))
    seed = int(audit["seed"])
    root = ROOT / config["output_root"] / "development_lr_audit"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def execute(subject: str, learning_rate: float) -> dict:
        lr_root = root / f"lr_{tag(learning_rate)}"
        done = lr_root / "per_subject" / subject / "structured" / f"seed_{seed}" / "DONE.json"
        if args.resume and done.exists():
            payload = json.loads(done.read_text())
            if payload.get("status") == "COMPLETE":
                return payload
        command = [
            str(config["resources"]["python"]), str(RUNNER),
            "--config", str(config_path), "--subject", subject,
            "--model", "structured", "--seed", str(seed),
            "--device", "cuda:0", "--output-root", str(lr_root),
            "--learning-rate", str(learning_rate), "--resume",
        ]
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        log = logs / f"{subject}__lr_{tag(learning_rate)}.log"
        with log.open("a") as handle:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{subject} lr={learning_rate}: return code {result.returncode}")
        return json.loads(done.read_text())

    rows = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(execute, subject, rate): (subject, rate)
            for subject in subjects for rate in rates
        }
        for future in as_completed(futures):
            subject, rate = futures[future]
            payload = future.result()
            rows.append(
                {
                    "subject": subject,
                    "learning_rate": rate,
                    "validation_contact_nll": float(
                        payload["validation"]["contact_nll_per_continue_decision"]
                    ),
                    "test_values_used_for_selection": False,
                    "target_values_read": False,
                }
            )
            print(f"[{len(rows)}/{len(subjects) * len(rates)}] {subject} lr={rate:g}", flush=True)
    medians = {
        str(rate): float(
            np.median([row["validation_contact_nll"] for row in rows if row["learning_rate"] == rate])
        )
        for rate in rates
    }
    selected = min(rates, key=lambda rate: (medians[str(rate)], rate))
    selection = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "subjects": subjects,
        "seed": seed,
        "learning_rates": rates,
        "selection_metric": audit["selection_metric"],
        "validation_contact_nll_median": medians,
        "selected_learning_rate": selected,
        "test_values_used_for_selection": False,
        "ictal_target_values_read": False,
        "rows": sorted(rows, key=lambda row: (row["subject"], row["learning_rate"])),
        "runtime_seconds": time.time() - started,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    }
    atomic_json(root / "SELECTION.json", selection)
    print(json.dumps(selection, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
