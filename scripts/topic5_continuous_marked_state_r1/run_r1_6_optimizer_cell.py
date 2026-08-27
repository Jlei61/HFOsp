#!/usr/bin/env python3
"""Run one selection-only R1.6 optimizer cell without opening dev validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import (
    R1_6_REVISION,
    nested_time_split,
)
from src.topic5_continuous_marked_state_r1.optimizer_runtime import (
    load_explicit_target_model,
)
from src.topic5_continuous_marked_state_r1.r1_2 import (
    evaluate_full_t1,
)
from src.topic5_continuous_marked_state_r1.r1_3 import (
    fit_target_observer,
    materialize_embedding,
)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True, choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS
    )
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--prefix-config-id", default="prefix_adamw_lr3e-4_wd1e-3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observer-epochs", type=int, default=4)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--state-learning-rate", type=float, default=3e-4)
    parser.add_argument("--observer-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--chunk-anchors", type=int, default=8)
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
    model = loaded["model"]
    design = loaded["design"]
    loader = loaded["loader"]
    manifest = loaded["manifest"]
    manifest_path = loaded["manifest_path"]
    prefix_result = loaded["prefix_result"]
    prefix_result_path = loaded["prefix_result_path"]
    split = nested_time_split(design)
    initial = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    trace = fit_target_observer(
        model, design, loader, device=args.device,
        observer_epochs=args.observer_epochs, joint_epochs=args.joint_epochs,
        state_lr=args.state_learning_rate,
        observer_lr=args.state_learning_rate * args.observer_lr_ratio,
        raw_lr=1e-5, chunk_anchors=args.chunk_anchors,
        optimizer_name=args.optimizer, weight_decay=args.weight_decay,
        grad_clip_norm=(
            None if args.grad_clip_norm <= 0 else args.grad_clip_norm
        ),
        warmup_fraction=args.warmup_fraction,
        selection_min_delta=args.selection_min_delta,
        early_stopping_patience=(
            None if args.early_stopping_patience <= 0
            else args.early_stopping_patience
        ),
        epoch_zero_seen_inner_validation=False,
        refit_mode="selection_best",
    )
    embedding = materialize_embedding(
        model, design, loader, device=args.device,
        batch_size=args.chunk_anchors,
        anchor_limit=int(design.anchor_ids("train")[-1]) + 1,
    )
    alignment = asdict(evaluate_full_t1(
        model, design, embedding, "train", device=args.device,
        time_lower=split.alignment_select_lower,
        time_upper=split.alignment_select_upper,
    ))
    current = model.state_dict()
    update_norm = float(np.sqrt(sum(
        float((value.detach().cpu().float() - initial[key].float()).square().sum())
        for key, value in current.items() if key in initial
    )))
    output = (
        args.output_root / "selection_cells" / args.config_id
        / args.subject / f"seed_{args.seed}"
    )
    checkpoint = output / "model.pt"
    atomic_torch(checkpoint, {
        "revision": R1_6_REVISION,
        "stage": "optimizer_selection",
        "subject": args.subject,
        "seed": int(args.seed),
        "config_id": args.config_id,
        "model": model.state_dict(),
        "fit_trace": asdict(trace),
    })
    result = {
        "status": "COMPLETE",
        "revision": R1_6_REVISION,
        "stage": "optimizer_selection",
        "subject": args.subject,
        "seed": int(args.seed),
        "config_id": args.config_id,
        "prefix_config_id": args.prefix_config_id,
        "prefix_result": str(prefix_result_path),
        "prefix_result_sha256": contract.sha256_file(prefix_result_path),
        "fit_trace": asdict(trace),
        "alignment_selection": alignment,
        "total_parameter_update_norm": update_norm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "observation_cache_manifest": str(manifest_path),
        "observation_cache_manifest_sha256": contract.sha256_file(manifest_path),
        "development_validation_scored": False,
        "epoch_zero_seen_alignment_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
