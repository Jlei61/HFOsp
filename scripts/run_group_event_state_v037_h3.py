#!/usr/bin/env python3
"""Run one independent M0/M1/M2 H3 generative comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h3_generative import H3Config, build_h3_data, train_h3_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--patience-checks", type=int, default=None)
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h3_independent_generative"),
    )
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        print(f"already complete: {out}")
        return
    config = H3Config(seed=args.seed)
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.learning_rate is not None:
        overrides["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        overrides["weight_decay"] = float(args.weight_decay)
    if args.validate_every is not None:
        overrides["validate_every"] = int(args.validate_every)
    if args.patience_checks is not None:
        overrides["patience_checks"] = int(args.patience_checks)
    if overrides:
        config = replace(config, **overrides)
    data = build_h3_data(args.subject, args.seed, config)
    card = train_h3_subject(data, config, device=torch.device(args.device), out_dir=out)
    print(card["primary_contrasts"])


if __name__ == "__main__":
    main()
