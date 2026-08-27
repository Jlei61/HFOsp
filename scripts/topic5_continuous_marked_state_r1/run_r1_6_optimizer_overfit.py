#!/usr/bin/env python3
"""Run one fixed short-segment overfit check for R1.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import (
    R1_6_REVISION,
    nested_time_split,
    overfit_target_segment,
)
from src.topic5_continuous_marked_state_r1.optimizer_runtime import (
    load_explicit_target_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True, choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS
    )
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--config-id", default="overfit_warm_lr1e-3")
    parser.add_argument("--prefix-config-id", default="prefix_adamw_lr3e-4_wd1e-3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-anchors", type=int, default=64)
    parser.add_argument("--state-learning-rate", type=float, default=1e-3)
    parser.add_argument("--observer-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--chunk-anchors", type=int, default=8)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--observation-cache-root", type=Path,
        default=contract.RESULT_ROOT / "r1_5" / "cache",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    loaded = load_explicit_target_model(
        subject=args.subject, seed=args.seed, device=args.device,
        r1_2_root=args.r1_2_root,
        observation_cache_root=args.observation_cache_root,
        output_root=args.output_root,
        prefix_config_id=args.prefix_config_id,
    )
    result = overfit_target_segment(
        loaded["model"], loaded["design"], loaded["loader"],
        device=args.device, split=nested_time_split(loaded["design"]),
        epochs=args.epochs, maximum_anchors=args.maximum_anchors,
        state_lr=args.state_learning_rate,
        observer_lr=args.state_learning_rate * args.observer_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip_norm=(
            None if args.grad_clip_norm <= 0 else args.grad_clip_norm
        ),
        warmup_fraction=args.warmup_fraction,
        chunk_anchors=args.chunk_anchors, optimizer_name=args.optimizer,
    )
    result.update({
        "revision": R1_6_REVISION, "stage": "short_segment_overfit",
        "subject": args.subject, "seed": int(args.seed),
        "config_id": args.config_id,
        "prefix_result": str(loaded["prefix_result_path"]),
        "prefix_result_sha256": contract.sha256_file(
            loaded["prefix_result_path"]
        ),
        "development_validation_scored": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })
    output = (
        args.output_root / "overfit" / args.config_id
        / args.subject / f"seed_{args.seed}/result.json"
    )
    contract.atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
