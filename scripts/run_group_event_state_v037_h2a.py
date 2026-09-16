#!/usr/bin/env python3
"""Run one frozen-state, frozen-decoder v0.3.7 H2a unit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v037.h2a import H2ATrainConfig, train_h2a_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_equal_horizon"))
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"))
    parser.add_argument("--decoder-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/decoder_strict"))
    parser.add_argument("--state-family", choices=("event", "dual", "grid"), default="event")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists(): print(f"already complete: {out}"); return
    config = H2ATrainConfig(seed=int(args.seed))
    if args.max_epochs is not None:
        config = replace(config, max_epochs_static=args.max_epochs, max_epochs_state=args.max_epochs,
                         max_epochs_oracle=args.max_epochs, patience_epochs=max(2, args.max_epochs))
    card = train_h2a_subject(
        args.subject, int(args.seed), device=torch.device(args.device), out_dir=out,
        config=config, h1_root=args.h1_root, decoder_root=args.decoder_root,
        state_family=args.state_family,
    )
    print(card["primary_contrasts"])


if __name__ == "__main__": main()
