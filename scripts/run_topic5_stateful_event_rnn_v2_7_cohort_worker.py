#!/usr/bin/env python3
"""Parallel-safe validation worker for stateful event RNN v2.7.

Each process screens explicitly named patients and never aggregates concurrently.
The aggregate phase is read-only with respect to patient artifacts and fails closed
until all validation-only epoch-boundary audits are present.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_resource_guard import (  # noqa: E402
    configure_torch_threads,
    pin_thread_environment,
)

pin_thread_environment(1)

import torch  # noqa: E402

from scripts.run_topic5_stateful_event_rnn_v2_7 import (  # noqa: E402
    DEFAULT_CONFIG,
    assert_repair_only_config,
    freeze_screen,
    provenance_manifest,
    screen_subject,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = assert_repair_only_config(config_path)
    provenance_manifest(config_path)
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    configure_torch_threads(torch, int(config["torch_num_threads"]))

    if args.phase == "aggregate":
        freeze_screen(config, config_path, output, args.subjects)
        return
    if not args.subjects:
        raise ValueError("patients phase requires --subjects")
    for subject in args.subjects:
        print(f"[v2.7 validation worker] {subject}", flush=True)
        screen_subject(subject, config, output)


if __name__ == "__main__":
    main()
