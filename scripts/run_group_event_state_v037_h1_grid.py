#!/usr/bin/env python3
"""Run one five-minute hierarchical v0.3.7 H1 observer unit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037 import CheckpointEntry, update_checkpoint_registry
from src.topic5_group_event_state.v037.contracts import sha256_file
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import H1TrainConfig, train_h1_subject


FAMILY = "S_grid_hierarchical_5min_equal_horizon"


def _register(out_root: Path, subject: str, seed: int, card: dict) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes = "+".join(
        sha256_file(ROOT / path)
        for path in (
            "src/topic5_group_event_state/v037/h1_train.py",
            "src/topic5_group_event_state/v037/ctssm.py",
        )
    )
    update_checkpoint_registry(
        out_root.parent / "registry" / "checkpoint_registry_grid.json",
        CheckpointEntry(
            key=f"{FAMILY}::{subject}::seed{seed}",
            model_family=FAMILY,
            state_semantics="observer",
            subject=subject,
            seed=int(seed),
            checkpoint_path=card["checkpoint_path"],
            maximum_training_time=float(card["maximum_training_time"]),
            input_streams=("five_minute_group_event_burden", "five_minute_conditional_grammar"),
            objectives=("future_burden", "conditional_grammar_multi_horizon"),
            code_commit=f"{commit}+source_sha256:{hashes}",
            normalization_provenance=(
                "event representation fit before INNER; every grid frame contains only events "
                "strictly before its timestamp; horizons equally weighted"
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_grid"))
    parser.add_argument("--grid-seconds", type=float, default=300.0)
    parser.add_argument("--lr-state", type=float, default=None)
    parser.add_argument("--lr-state-head", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--state-init-std", type=float, default=None)
    parser.add_argument("--burden-channels", type=int, default=None)
    parser.add_argument("--grammar-channels", type=int, default=None)
    parser.add_argument("--max-steps-state", type=int, default=None)
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        import json
        card = json.loads((out / "card.json").read_text(encoding="utf-8"))
        _register(args.out_root, args.subject, args.seed, card)
        print(f"already complete: {out}")
        return
    config = H1TrainConfig(seed=int(args.seed))
    override = {
        "lr_state": args.lr_state,
        "lr_state_head": args.lr_state_head,
        "weight_decay": args.weight_decay,
        "warmup_steps_state": args.warmup_steps,
        "state_readout_init_std": args.state_init_std,
        "burden_channels_per_tau": args.burden_channels,
        "grammar_channels_per_tau": args.grammar_channels,
        "max_steps_state": args.max_steps_state,
    }
    config = replace(config, **{key: value for key, value in override.items() if value is not None})
    data = build_h1_subject_data(args.subject, seed=int(args.seed))
    card = train_h1_subject(
        data, config, device=torch.device(args.device), out_dir=out,
        state_mode="grid", grid_seconds=float(args.grid_seconds),
    )
    _register(args.out_root, args.subject, args.seed, card)
    print(card["primary_contrasts"])


if __name__ == "__main__":
    main()
