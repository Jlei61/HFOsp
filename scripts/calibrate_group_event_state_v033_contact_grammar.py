#!/usr/bin/env python3
"""Calibration-prefix legacy-scoring grammar runner for E253/E916."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Grammar jobs are GPU-light but small linear algebra kernels otherwise ask the
# host for hundreds of threads.  Keep one CPU thread per detached worker.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (  # noqa: E402
    LegacyGrammarCalibrationConfig,
    TUNING_SUBJECTS,
    calibrate_legacy_contact_grammar,
)
from src.topic5_group_event_state.v033_training_lab.paths import AGENT_B_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=TUNING_SUBJECTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root", type=Path, default=AGENT_B_ROOT / "contact_grammar"
    )
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--offset-learning-rate", type=float, default=3e-3)
    parser.add_argument("--base-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--smoke-fit-events", type=int)
    parser.add_argument("--smoke-inner-events", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = LegacyGrammarCalibrationConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        offset_learning_rate=args.offset_learning_rate,
        base_learning_rate=args.base_learning_rate,
        seed=args.seed,
        max_fit_events=args.smoke_fit_events,
        max_inner_events=args.smoke_inner_events,
    )
    report = calibrate_legacy_contact_grammar(
        args.subject,
        out_dir=args.output_root / args.subject,
        device=torch.device(args.device),
        cfg=cfg,
        overwrite=args.overwrite,
    )
    print(report)


if __name__ == "__main__":
    main()
