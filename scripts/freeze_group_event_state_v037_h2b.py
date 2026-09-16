#!/usr/bin/env python3
"""Freeze v0.3.7 interictal features before any seizure outcome is opened."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h2b import freeze_h2b_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--h1-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete"),
    )
    parser.add_argument(
        "--event-h1-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"),
    )
    parser.add_argument("--grid-h1-root", type=Path, default=None)
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h2b_frozen_transfer"),
    )
    args = parser.parse_args()
    out = args.out_root / "features" / args.subject / f"seed{args.seed}"
    card = out / "freeze_card.json"
    if card.exists():
        print(f"already frozen: {card}")
        return
    payload = freeze_h2b_features(
        args.subject, args.seed, h1_root=args.h1_root,
        event_h1_root=args.event_h1_root, grid_h1_root=args.grid_h1_root,
        out_dir=out,
    )
    print(payload["feature_sha256"])


if __name__ == "__main__":
    main()
