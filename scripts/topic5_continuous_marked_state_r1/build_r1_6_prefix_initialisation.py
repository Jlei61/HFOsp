#!/usr/bin/env python3
"""Build one selection-safe R1.6 prefix T1 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.optimizer_audit import (
    R1_6_REVISION,
    fit_prefix_safe_core,
    nested_time_split,
)
from src.topic5_continuous_marked_state_r1.r1_2 import (
    FrozenEmbeddingStateModel,
    load_full_admissible_event_stream,
    load_full_anchor_cache,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--chunk-anchors", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--config-id", default="prefix_adamw_lr3e-4_wd1e-3")
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    design, embedding, cache = load_full_anchor_cache(
        args.subject, arm="explicit", output_root=args.r1_2_root
    )
    baseline_path = (
        args.r1_2_root / "baselines" / args.subject / "seed_0/models.pt"
    )
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage_path = args.r1_2_root / "coverage" / f"{args.subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(args.subject, coverage)
    model = FrozenEmbeddingStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, observation_dim=embedding.shape[1], state_dim=8,
    ).to(args.device)
    split = nested_time_split(design)
    model, trace = fit_prefix_safe_core(
        model, design, embedding, device=args.device,
        epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, chunk_anchors=args.chunk_anchors,
        optimizer_name=args.optimizer, split=split,
    )
    output = (
        args.output_root / "prefix_initialisation" / args.config_id
        / args.subject / f"seed_{args.seed}"
    )
    checkpoint = output / "model.pt"
    atomic_torch(checkpoint, {
        "revision": R1_6_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "config_id": args.config_id,
        "model": model.state_dict(),
        "trace": trace,
    })
    result = {
        "status": "COMPLETE",
        "revision": R1_6_REVISION,
        "stage": "prefix_initialisation",
        "subject": args.subject,
        "seed": int(args.seed),
        "config_id": args.config_id,
        "trace": trace,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "cache_manifest": str(
            args.r1_2_root / "cache" / args.subject / "manifest.json"
        ),
        "cache_manifest_sha256": contract.sha256_file(
            args.r1_2_root / "cache" / args.subject / "manifest.json"
        ),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": contract.sha256_file(baseline_path),
        "coverage": str(coverage_path),
        "coverage_sha256": contract.sha256_file(coverage_path),
        "design_sha256": cache["design_sha256"],
        "epoch_zero_seen_alignment_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
