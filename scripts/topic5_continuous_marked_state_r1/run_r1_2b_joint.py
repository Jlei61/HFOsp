#!/usr/bin/env python3
"""Fit one R1.2b limited joint observer-state arm."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import make_paired_models
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    _bridge_scaler, evaluate_full_t1, load_full_admissible_event_stream,
    load_full_design, matched_wrong_time_permutation,
)
from src.topic5_continuous_marked_state_r1.r1_2b import (
    R1_2B_REVISION, R1_2B_SUBJECTS, JointLastLayerStateModel,
    evaluate_joint, fit_joint_t1, horizon_correction_off,
    load_joint_node_cache, materialize_joint_embedding,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=R1_2B_SUBJECTS)
    parser.add_argument("--arm", required=True, choices=("joint_explicit", "joint_explicit_raw"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunk-anchors", type=int, default=256)
    parser.add_argument("--state-learning-rate", type=float, default=3e-4)
    parser.add_argument("--observer-learning-rate", type=float, default=3e-5)
    parser.add_argument("--horizon-starts", type=int, default=64)
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_2b"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base_node, raw_node, contact_mask, cache_manifest = load_joint_node_cache(
        args.subject, output_root=args.output_root
    )
    upstream_manifest = json.loads(
        Path(cache_manifest["upstream_cache_manifest"]).read_text()
    )
    design = load_full_design(Path(upstream_manifest["design"]))
    baseline_path = args.r1_2_root / "baselines" / args.subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage = CoverageTable.load(
        args.r1_2_root / "coverage" / f"{args.subject}.npz"
    )
    stream = load_full_admissible_event_stream(args.subject, coverage)
    bridge_result_path = (
        args.r1_2_root / "bridge_e1" / args.subject / "seed_0/result.json"
    )
    bridge_result = json.loads(bridge_result_path.read_text())
    _, _, sampled, _ = _bridge_scaler(
        args.subject, baseline_path, bridge_result, stream, coverage
    )
    explicit_bridge, raw_bridge = make_paired_models(
        baseline, sampled, stream.adjacency, seed=0, device=args.device
    )
    bridge_checkpoint_path = (
        args.r1_2_root / "bridge_e1" / args.subject / "seed_0/models.pt"
    )
    bridge_checkpoint = torch.load(
        bridge_checkpoint_path, map_location=args.device, weights_only=False
    )
    explicit_bridge.load_state_dict(bridge_checkpoint["explicit"])
    raw_bridge.load_state_dict(bridge_checkpoint["explicit_raw"])
    source_observer = raw_bridge.observer
    raw_enabled = args.arm == "joint_explicit_raw"

    # The state/adapters vary by seed; the paired observer tail has one fixed,
    # audited initialisation for both arms and raw_gain starts exactly at zero.
    torch.manual_seed(args.seed)
    model = JointLastLayerStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, source_observer, raw_enabled=raw_enabled,
        state_dim=8,
    ).to(args.device)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    forbidden_trainable = [
        name for name in trainable
        if name.startswith((
            "last_observer.explicit", "last_observer.coordinate",
            "last_observer.shaft", "last_observer.raw.tokenizer",
            "last_observer.raw.transformer",
        ))
    ]
    if forbidden_trainable:
        raise RuntimeError(f"upstream observer component became trainable: {forbidden_trainable}")

    initial_embedding = materialize_joint_embedding(
        model, base_node, raw_node, contact_mask, device=args.device
    )
    initial_filtered = asdict(evaluate_full_t1(
        model, design, initial_embedding, "validation", device=args.device
    ))
    initial_off = asdict(evaluate_full_t1(
        model, design, initial_embedding, "validation", device=args.device,
        validation_correction_off=True,
    ))
    parity = {
        key: abs(initial_filtered[key] - initial_off[key])
        for key in (
            "joint_nll_per_event", "timing_nll_per_event", "mark_nll_per_event",
            "group_size_nll_per_event", "subset_nll_per_event",
        )
    }
    if max(parity.values()) > 1e-6:
        raise RuntimeError(f"R1.2b zero-effect parity failed: {parity}")

    model = fit_joint_t1(
        model, design, base_node, raw_node, contact_mask, device=args.device,
        epochs=args.epochs, state_learning_rate=args.state_learning_rate,
        observer_learning_rate=args.observer_learning_rate,
        chunk_anchors=args.chunk_anchors,
    )
    embedding = materialize_joint_embedding(
        model, base_node, raw_node, contact_mask, device=args.device
    )
    filtered = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device
    ))
    validation_off = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        validation_correction_off=True,
    ))
    all_off = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        correction_enabled=False,
    ))
    permutation, matched = matched_wrong_time_permutation(design, split="validation")
    wrong_all = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        state_permutation=permutation,
    ))
    if bool(matched.any()):
        matched_filtered = asdict(evaluate_full_t1(
            model, design, embedding, "validation", device=args.device,
            matched_anchor_mask=matched,
        ))
        matched_wrong = asdict(evaluate_full_t1(
            model, design, embedding, "validation", device=args.device,
            state_permutation=permutation, matched_anchor_mask=matched,
        ))
    else:
        matched_filtered = None; matched_wrong = None
    horizon = horizon_correction_off(
        model, design, embedding, stream, coverage, baseline,
        horizons=(5, 10, 20), max_start_anchors=args.horizon_starts,
        device=args.device,
    )

    output = args.output_root / "joint" / args.subject / f"{args.arm}_seed_{args.seed}"
    checkpoint = output / "model.pt"
    _atomic_torch(checkpoint, {
        "contract": contract.REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "subject": args.subject, "arm": args.arm, "seed": args.seed,
        "model": model.state_dict(),
        "selected_epochs": int(model.selected_epochs),
    })
    final = {
        "filtered": filtered,
        "validation_correction_off": validation_off,
        "all_correction_off": all_off,
        "wrong_time_all_validation": wrong_all,
        "matched_filtered": matched_filtered,
        "matched_wrong_time": matched_wrong,
    }
    contrasts = {
        "filtered_minus_no_state_joint_nll": (
            filtered["joint_nll_per_event"] - initial_filtered["joint_nll_per_event"]
        ),
        "filtered_minus_validation_correction_off_joint_nll": (
            filtered["joint_nll_per_event"] - validation_off["joint_nll_per_event"]
        ),
        "filtered_minus_validation_correction_off_timing_nll": (
            filtered["timing_nll_per_event"] - validation_off["timing_nll_per_event"]
        ),
        "filtered_minus_validation_correction_off_mark_nll": (
            filtered["mark_nll_per_event"] - validation_off["mark_nll_per_event"]
        ),
        "filtered_minus_all_correction_off_joint_nll": (
            filtered["joint_nll_per_event"] - all_off["joint_nll_per_event"]
        ),
        "filtered_minus_wrong_time_all_joint_nll": (
            filtered["joint_nll_per_event"] - wrong_all["joint_nll_per_event"]
        ),
        "matched_filtered_minus_wrong_time_joint_nll": (
            matched_filtered["joint_nll_per_event"] - matched_wrong["joint_nll_per_event"]
            if matched_filtered is not None else None
        ),
        "matched_filtered_minus_wrong_time_timing_nll": (
            matched_filtered["timing_nll_per_event"] - matched_wrong["timing_nll_per_event"]
            if matched_filtered is not None else None
        ),
        "matched_filtered_minus_wrong_time_mark_nll": (
            matched_filtered["mark_nll_per_event"] - matched_wrong["mark_nll_per_event"]
            if matched_filtered is not None else None
        ),
    }
    frozen_name = "explicit_raw" if raw_enabled else "explicit"
    frozen_result_path = (
        args.r1_2_root / "t1_full" / args.subject
        / f"{frozen_name}_d8_seed_0/result.json"
    )
    frozen_result = json.loads(frozen_result_path.read_text())
    result = {
        "status": "COMPLETE", "contract": contract.REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "subject": args.subject, "arm": args.arm, "seed": args.seed,
        "state_dim": 8, "epochs_budget": args.epochs,
        "selected_epochs": int(model.selected_epochs),
        "inner_validation_joint_nll": float(model.inner_validation_joint_nll),
        "state_learning_rate": args.state_learning_rate,
        "observer_learning_rate": args.observer_learning_rate,
        "observer_to_state_lr_ratio": args.observer_learning_rate / args.state_learning_rate,
        "trainable_parameter_names": trainable,
        "trainable_observer_component": "last_spatial_aggregation_block",
        "frozen_upstream_raw_temporal_encoder": True,
        "raw_enabled": raw_enabled,
        "initial_raw_gain": 0.0,
        "final_raw_gain": float(model.last_observer.raw_gain.detach().cpu()),
        "initial_validation": {
            "filtered": initial_filtered, "validation_correction_off": initial_off,
        },
        "initial_parity_abs": parity,
        "final_validation": final,
        "contrasts": contrasts,
        "horizon_correction_off": horizon,
        "wrong_time_match": {
            "n_validation_anchors": int(np.sum(design.anchor_split == 1)),
            "n_matched_anchors": int(matched.sum()),
            "matched_support_events": (
                int(matched_filtered["n_events"]) if matched_filtered is not None else 0
            ),
            "same_session": True, "minimum_separation_seconds": 300.0,
        },
        "frozen_r1_2_reference": {
            "path": str(frozen_result_path),
            "sha256": contract.sha256_file(frozen_result_path),
            "selected_epochs": frozen_result["selected_epochs"],
            "filtered_joint_nll": frozen_result["final_validation"]["filtered"]["joint_nll_per_event"],
            "joint_minus_frozen_filtered_nll": (
                filtered["joint_nll_per_event"]
                - frozen_result["final_validation"]["filtered"]["joint_nll_per_event"]
            ),
        },
        "cache_manifest": str(args.output_root / "cache" / args.subject / "manifest.json"),
        "cache_manifest_sha256": contract.sha256_file(
            args.output_root / "cache" / args.subject / "manifest.json"
        ),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": contract.sha256_file(baseline_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "full_recorded_support": True, "sealed_opened": False,
        "claim_boundary": (
            "three-subject development R1.2b limited last-spatial-layer alignment; "
            "upstream raw features remain epoch-zero random features; not cohort, H2b or H3 evidence"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
