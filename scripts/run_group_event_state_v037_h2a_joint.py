#!/usr/bin/env python3
"""Run one interictal-only H2a-to-observer joint sensitivity unit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.h2a import (
    H2AJointTrainConfig,
    train_h2a_joint_subject,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"))
    parser.add_argument("--primary-h2a-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_equal_horizon"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_joint_sensitivity"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        print(f"already complete: {out}")
        return
    config = H2AJointTrainConfig(seed=int(args.seed))
    if args.max_steps is not None:
        config = replace(
            config, max_steps=int(args.max_steps), validate_every=1,
            patience_checks=max(2, int(args.max_steps)),
            fit_events_per_step=min(1024, config.fit_events_per_step),
        )
    card = train_h2a_joint_subject(
        args.subject, int(args.seed), device=torch.device(args.device), out_dir=out,
        h1_root=args.h1_root, primary_h2a_root=args.primary_h2a_root,
        config=config,
    )
    print({"h2a": card["h2a_primary_contrasts"],
           "h1": card["h1_mandatory_reevaluation"]["primary_contrasts"]})


if __name__ == "__main__":
    main()
