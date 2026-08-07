#!/usr/bin/env python3
"""Parallel-safe per-patient worker for the frozen v2.7 formal test."""
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

from scripts.run_topic5_stateful_event_rnn_v2_7_formal import (  # noqa: E402
    DEFAULT_CONFIG,
    aggregate,
    run_subject,
    verify_frozen,
)
from src.topic5_resource_guard import (  # noqa: E402
    configure_torch_threads,
    pin_thread_environment,
)


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
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return aggregate(results, failures, config, config_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()

    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    verify_frozen(config_path, output)
    configure_torch_threads(torch, int(config["torch_num_threads"]))
    if args.phase == "aggregate":
        aggregate_only(config, config_path, output)
        return
    if not args.subjects:
        raise ValueError("patients phase requires --subjects")
    for subject in args.subjects:
        print(f"[v2.7 frozen patient test] {subject}", flush=True)
        run_subject(subject, config, output)


if __name__ == "__main__":
    main()

