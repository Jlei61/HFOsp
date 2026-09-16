#!/usr/bin/env python3
"""Fit one persistent H3 M0/M1/M2 comparison from a frozen primary H3 run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h3_persistent import train_persistent_h3_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h3_independent_generative"),
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h3_persistent_feedback"),
    )
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        print(f"already complete: {out}")
        return
    card = train_persistent_h3_subject(
        args.subject, args.seed, source_root=args.source_root, out_dir=out,
        device=torch.device(args.device),
    )
    print(card["primary_contrasts"])


if __name__ == "__main__":
    main()
