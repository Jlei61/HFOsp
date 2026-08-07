#!/usr/bin/env python3
"""Parallel-safe patient worker and aggregate-only wrapper for v2.6.

The scientific implementation remains in run_topic5_stateful_event_rnn_v2_6.py;
this file only schedules already-frozen patient jobs without concurrent cohort
aggregation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    DEFAULT_CONFIG,
    aggregate,
    run_subject,
    sha256,
)


def verify_frozen(config_path: Path, output: Path):
    frozen = json.load(
        (output / "validation_screen/FROZEN_VALIDATION_STATE.json").open()
    )
    expected = {
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_6.py"),
        "runner_sha256": sha256(
            ROOT / "scripts/run_topic5_stateful_event_rnn_v2_6.py"
        ),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"v2.6 frozen validation hash mismatch: {key}")
    if frozen.get("status") != "ALL_PATIENT_VALIDATION_PROFILES_FROZEN":
        raise RuntimeError("v2.6 validation state is not fully frozen")


def aggregate_only(config, config_path: Path, output: Path):
    subjects = sorted(
        path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz")
    )
    results = []
    failures = []
    for subject in subjects:
        path = output / "per_subject" / f"{subject}.json"
        if not path.exists():
            failures.append(
                {
                    "subject": subject,
                    "error_type": "MissingArtifact",
                    "reason": str(path),
                }
            )
            continue
        results.append(json.load(path.open()))
    return aggregate(results, failures, config, config_path, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    output = ROOT / config["output_root"]
    verify_frozen(config_path, output)
    torch.set_num_threads(int(config["torch_num_threads"]))
    if args.phase == "aggregate":
        aggregate_only(config, config_path, output)
        return
    if not args.subjects:
        raise ValueError("patients phase requires --subjects")
    for subject in args.subjects:
        print(f"[v2.6 frozen patient test] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()
