#!/usr/bin/env python3
"""CLI for grammar calibration and one v0.3 pilot state run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v03.pilot import (  # noqa: E402
    PILOT_SUBJECTS,
    PilotConfig,
    calibrate_grammar,
    train_state_model,
)
from src.topic5_group_event_state.v03.evaluate import evaluate_open_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "train", "evaluate"), required=True)
    parser.add_argument("--subject", choices=PILOT_SUBJECTS, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3/pilot"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--grammar-epochs", type=int, default=12)
    parser.add_argument("--chunk-events", type=int, default=1024)
    parser.add_argument("--chunk-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = PilotConfig(
        max_epochs=args.max_epochs,
        grammar_epochs=args.grammar_epochs,
        chunk_events=args.chunk_events,
        chunk_seconds=args.chunk_seconds,
    )
    subject_root = args.output_root / args.subject
    if args.mode == "calibrate":
        result = calibrate_grammar(
            args.subject,
            device=device,
            out_dir=subject_root / "grammar",
            cfg=cfg,
            overwrite=args.overwrite,
        )
    elif args.mode == "train":
        result = train_state_model(
            args.subject,
            args.seed,
            device=device,
            grammar_dir=subject_root / "grammar",
            out_dir=subject_root / f"seed_{args.seed}",
            cfg=cfg,
            overwrite=args.overwrite,
        )
    else:
        run_root = subject_root / f"seed_{args.seed}"
        result = evaluate_open_loop(
            args.subject,
            args.seed,
            checkpoint=run_root / "checkpoint.pt",
            grammar_checkpoint=subject_root / "grammar/grammar_v03.pt",
            out_dir=run_root,
            device=device,
            cfg=cfg,
            overwrite=args.overwrite,
        )
    print(result)


if __name__ == "__main__":
    main()
