#!/usr/bin/env python3
"""Fit seizure readouts from an already frozen v0.3.7 interictal feature file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h2b import run_h2b_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h2b_frozen_transfer"),
    )
    args = parser.parse_args()
    freeze_dir = args.out_root / "features" / args.subject / f"seed{args.seed}"
    out = args.out_root / "outcomes" / args.subject / f"seed{args.seed}"
    card = out / "card.json"
    if card.exists():
        print(f"already complete: {card}")
        return
    payload = run_h2b_outcomes(
        args.subject, args.seed, freeze_dir=freeze_dir, out_dir=out,
    )
    print(payload["primary_contrasts"])


if __name__ == "__main__":
    main()

