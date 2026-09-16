#!/usr/bin/env python3
"""Fit horizon-specific frozen readouts on one shared S_N/S_G trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT, V035_DECODER_FITS  # noqa: E402
from src.topic5_group_event_state.v035.frozen_shared_evaluator import evaluate_frozen_shared  # noqa: E402
from src.topic5_group_event_state.v035.full_mark_state import FullMarkTrainConfig, load_full_mark_data  # noqa: E402
from src.topic5_group_event_state.v035.shared_state import (  # noqa: E402
    SharedProducerConfig, build_shared_grammar_data,
)

ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def _base_config(raw: dict) -> FullMarkTrainConfig:
    value = dict(raw)
    for key in ("state_taus_seconds", "offset_weights", "event_offsets"):
        if key in value:
            value[key] = tuple(value[key])
    return FullMarkTrainConfig(**value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--producer-unit", type=Path, required=True)
    ap.add_argument("--decoder-seed", type=int, required=True)
    ap.add_argument("--rate-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-family", choices=("S_N", "S_G"))
    ap.add_argument("--extra-producer-unit", type=Path)
    args = ap.parse_args()
    checkpoint_path = args.producer_unit / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    subject, family = checkpoint["subject"], checkpoint["family"]
    if subject not in V035_DECODER_FITS:
        raise KeyError(subject)
    base = _base_config(checkpoint["config"]["base"])
    fit = V035_DECODER_FITS[subject]
    bundle = load_frozen_decoder(
        DECODER_ROOT / "formal_units" / fit / ARM / f"seed{args.decoder_seed}",
        DECODER_ROOT / "cache" / fit, device=torch.device("cpu"),
    )
    rate = args.rate_root / subject / f"seed{base.seed}" / "trajectory_and_scores.npz"
    data = load_full_mark_data(subject, bundle, rate, event_offsets=base.event_offsets)
    target_family = args.target_family or family
    if args.extra_producer_unit is not None:
        extra_checkpoint = torch.load(
            args.extra_producer_unit / "checkpoint.pt", map_location="cpu", weights_only=False,
        )
        if extra_checkpoint["subject"] != subject:
            raise ValueError("combined evaluator producer subjects differ")
        if int(extra_checkpoint["config"]["base"]["seed"]) != int(base.seed):
            raise ValueError("combined evaluator producer seeds differ")
        if extra_checkpoint["family"] == family:
            raise ValueError("combined evaluator needs one S_N and one S_G producer")
    grammar = None
    if target_family == "S_G":
        shared = SharedProducerConfig(
            family=family, base=base,
            block_grammar_weight=float(checkpoint["config"]["block_grammar_weight"]),
            local_grammar_weight=float(checkpoint["config"]["local_grammar_weight"]),
            physical_weight=float(checkpoint["config"]["physical_weight"]),
            requested_communities=int(checkpoint["config"]["requested_communities"]),
            requested_repertoires=int(checkpoint["config"]["requested_repertoires"]),
        )
        grammar = build_shared_grammar_data(data, shared).targets
    card = evaluate_frozen_shared(
        data, args.producer_unit / "state_trajectory.npz", family=family,
        out_dir=args.out_dir, grammar_targets=grammar, target_family=target_family,
        extra_trajectory=(None if args.extra_producer_unit is None else
                          args.extra_producer_unit / "state_trajectory.npz"),
    )
    print(json.dumps({"subject": subject, "source_family": card["source_family"],
                      "target_family": target_family, "horizons": card["horizons"]}, indent=2))


if __name__ == "__main__":
    main()
