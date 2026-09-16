#!/usr/bin/env python3
"""Run one v0.3.7 shared multi-horizon H1 observer unit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.topic5_group_event_state.v037 import CheckpointEntry, update_checkpoint_registry
from src.topic5_group_event_state.v037.contracts import sha256_file
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_train import H1TrainConfig, train_h1_subject


MODEL_FAMILY = "S_event_stream_separated_equal_horizon"


def _register_card(args: argparse.Namespace, card: dict) -> None:
    """Register a completed unit, including units recovered after a late registry failure."""

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hash = sha256_file(REPO_ROOT / "src/topic5_group_event_state/v037/h1_train.py")
    update_checkpoint_registry(
        args.out_root.parent / "registry" / "checkpoint_registry.json",
        CheckpointEntry(
            key=f"{MODEL_FAMILY}::{args.subject}::seed{args.seed}",
            model_family=MODEL_FAMILY,
            state_semantics="observer",
            subject=args.subject,
            seed=int(args.seed),
            checkpoint_path=card["checkpoint_path"],
            maximum_training_time=float(card["maximum_training_time"]),
            input_streams=("group_event_burden", "group_event_conditional_grammar"),
            objectives=("future_burden", "conditional_grammar_multi_horizon"),
            code_commit=f"{commit}+h1_train_sha256:{source_hash}",
            normalization_provenance=(
                "all dictionaries, residualisation and scales fit before INNER; "
                "burden and conditional-grammar readouts are structurally separated"
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"))
    parser.add_argument("--max-steps", type=int, default=None, help="debug override applied to every stage")
    args = parser.parse_args()
    out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        import json

        card = json.loads((out / "card.json").read_text(encoding="utf-8"))
        _register_card(args, card)
        print(f"already complete: {out}")
        return
    config = H1TrainConfig(seed=int(args.seed))
    if args.max_steps is not None:
        config = replace(
            config,
            max_steps_q=int(args.max_steps),
            max_steps_bmark=int(args.max_steps),
            max_steps_state=int(args.max_steps),
            max_steps_random=int(args.max_steps),
            validate_every=max(1, min(5, int(args.max_steps))),
            patience_checks=max(2, int(args.max_steps)),
        )
    data = build_h1_subject_data(args.subject, seed=int(args.seed))
    card = train_h1_subject(data, config, device=torch.device(args.device), out_dir=out)
    _register_card(args, card)
    print(card["primary_contrasts"])


if __name__ == "__main__": main()
