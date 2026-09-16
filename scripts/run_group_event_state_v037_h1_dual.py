#!/usr/bin/env python3
"""Run one v0.3.7 background plus marked-event H1 observer unit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src.topic5_group_event_state.v037 import CheckpointEntry, update_checkpoint_registry
from src.topic5_group_event_state.v037.contracts import sha256_file
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.h1_dual_train import H1DualTrainConfig, train_h1_dual_subject


FAMILY = "S_dual_background_event_budget_complete"


def _register(args: argparse.Namespace, card: dict) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes = "+".join(
        sha256_file(REPO / path) for path in (
            "src/topic5_group_event_state/v037/h1_dual_train.py",
            "src/topic5_group_event_state/v037/ctssm.py",
        )
    )
    update_checkpoint_registry(
        args.out_root.parent / "registry" / "checkpoint_registry_dual_formal.json",
        CheckpointEntry(
            key=f"{FAMILY}::{args.subject}::seed{args.seed}", model_family=FAMILY,
            state_semantics="observer", subject=args.subject, seed=int(args.seed),
            checkpoint_path=card["checkpoint_path"], maximum_training_time=float(card["maximum_training_time"]),
            input_streams=("fixed_clock_non_event_background", "group_event_burden", "group_event_conditional_grammar"),
            objectives=("future_burden", "conditional_grammar_multi_horizon"),
            code_commit=f"{commit}+source_sha256:{hashes}",
            normalization_provenance="all transforms fit before INNER; horizons equally weighted",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(); out = args.out_root / args.subject / f"seed{args.seed}"
    if (out / "card.json").exists():
        import json
        card = json.loads((out / "card.json").read_text(encoding="utf-8")); _register(args, card)
        print(f"already complete: {out}"); return
    config = H1DualTrainConfig(seed=int(args.seed))
    if args.max_steps is not None:
        n = int(args.max_steps)
        config = replace(config, max_steps_q=n, max_steps_bmark=n, max_steps_random=n,
                         max_steps_state=n, max_steps_background_current=n,
                         max_steps_background_state=n, validate_every=max(1, min(5, n)),
                         patience_checks=max(2, n))
    data = build_h1_subject_data(args.subject, seed=int(args.seed))
    card = train_h1_dual_subject(data, config, device=torch.device(args.device), out_dir=out)
    _register(args, card); print(card["primary_contrasts"])


if __name__ == "__main__": main()
