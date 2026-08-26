#!/usr/bin/env python3
"""Fit one frozen-embedding, full-recorded-support R1.2 T1 arm."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    R1_2_REVISION,
    FrozenEmbeddingStateModel,
    evaluate_full_t1,
    fit_full_t1,
    load_full_anchor_cache,
    load_full_admissible_event_stream,
    matched_wrong_time_permutation,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True, choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS
    )
    parser.add_argument("--arm", required=True, choices=("explicit", "explicit_raw"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunk-anchors", type=int, default=256)
    parser.add_argument("--state-dim", type=int, choices=(8,), default=8)
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    design, embedding, cache_manifest = load_full_anchor_cache(
        args.subject, arm=args.arm, output_root=args.output_root
    )
    baseline_path = args.output_root / "baselines" / args.subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage = CoverageTable.load(
        args.output_root / "coverage" / f"{args.subject}.npz"
    )
    stream = load_full_admissible_event_stream(args.subject, coverage)
    model = FrozenEmbeddingStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, observation_dim=embedding.shape[1],
        state_dim=args.state_dim,
    ).to(args.device)

    initial_filtered = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device
    ))
    initial_validation_off = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        validation_correction_off=True,
    ))
    parity = {
        key: abs(initial_filtered[key] - initial_validation_off[key])
        for key in (
            "joint_nll_per_event", "timing_nll_per_event", "mark_nll_per_event",
            "group_size_nll_per_event", "subset_nll_per_event",
        )
    }
    if max(parity.values()) > 1e-6:
        raise RuntimeError(f"R1.2 zero-effect initial parity failed: {parity}")

    model = fit_full_t1(
        model, design, embedding, device=args.device,
        epochs=args.epochs, chunk_anchors=args.chunk_anchors,
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
    permutation, matched = matched_wrong_time_permutation(
        design, split="validation"
    )
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
        matched_filtered = None
        matched_wrong = None

    output = (
        args.output_root / "t1_full" / args.subject
        / f"{args.arm}_d{args.state_dim}_seed_{args.seed}"
    )
    checkpoint = output / "model.pt"
    _atomic_torch(checkpoint, {
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "subject": args.subject,
        "arm": args.arm,
        "state_dim": args.state_dim,
        "seed": args.seed,
        "model": model.state_dict(),
        "selected_epochs": int(model.selected_epochs),
        "truncated_bptt_anchors": int(model.truncated_bptt_anchors),
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
            matched_filtered["joint_nll_per_event"]
            - matched_wrong["joint_nll_per_event"]
            if matched_filtered is not None else None
        ),
        "matched_filtered_minus_wrong_time_timing_nll": (
            matched_filtered["timing_nll_per_event"]
            - matched_wrong["timing_nll_per_event"]
            if matched_filtered is not None else None
        ),
        "matched_filtered_minus_wrong_time_mark_nll": (
            matched_filtered["mark_nll_per_event"]
            - matched_wrong["mark_nll_per_event"]
            if matched_filtered is not None else None
        ),
    }
    result = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "subject": args.subject,
        "arm": args.arm,
        "state_dim": args.state_dim,
        "seed": args.seed,
        "epochs_budget": args.epochs,
        "selected_epochs": int(model.selected_epochs),
        "inner_validation_joint_nll": float(model.inner_validation_joint_nll),
        "truncated_bptt_anchors": int(model.truncated_bptt_anchors),
        "observer_frozen": True,
        "full_recorded_support": True,
        "initial_validation": {
            "filtered": initial_filtered,
            "validation_correction_off": initial_validation_off,
        },
        "initial_parity_abs": parity,
        "final_validation": final,
        "contrasts": contrasts,
        "wrong_time_match": {
            "n_validation_anchors": int(np.sum(design.anchor_split == 1)),
            "n_matched_anchors": int(matched.sum()),
            "matched_support_events": (
                int(matched_filtered["n_events"]) if matched_filtered is not None else 0
            ),
            "same_session": True,
            "minimum_separation_seconds": 300.0,
            "swapped_object": "filtered_anchor_state",
        },
        "cache_manifest": str(
            args.output_root / "cache" / args.subject / "manifest.json"
        ),
        "cache_manifest_sha256": contract.sha256_file(
            args.output_root / "cache" / args.subject / "manifest.json"
        ),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": contract.sha256_file(baseline_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "sealed_opened": False,
        "claim_boundary": (
            "development R1.2 arm with a frozen observer and full recorded-support "
            "exact event likelihood; long-H3 subjects were selected by pre-fit "
            "support, and this is not a cohort or H3 result"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
