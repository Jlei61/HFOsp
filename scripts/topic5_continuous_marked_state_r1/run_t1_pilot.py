#!/usr/bin/env python3
"""Run a bounded exact-likelihood T1 persistent-state development pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import build_bridge_e1_design
from src.topic5_continuous_marked_state_r1.data import load_event_stream
from src.topic5_continuous_marked_state_r1.t1_pilot import (
    T1_PILOT_REVISION, PersistentEventModel, evaluate_t1, fit_t1,
    matched_wrong_time_permutation,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--anchor-batch-size", type=int, default=8)
    parser.add_argument("--max-train-anchors", type=int, default=64)
    parser.add_argument("--max-validation-anchors", type=int, default=32)
    parser.add_argument("--state-dim", type=int, choices=(8, 16), default=8)
    parser.add_argument("--raw-enabled", action="store_true")
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "t1_pilot")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    baseline_path = (
        contract.RESULT_ROOT / "baselines" / args.subject
        / f"seed_{args.seed}" / "models.pt"
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline checkpoint missing: {baseline_path}")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    stream = load_event_stream(args.subject)
    design, reader, design_manifest = build_bridge_e1_design(
        args.subject, baseline_path,
        max_train_anchors=args.max_train_anchors,
        max_validation_anchors=args.max_validation_anchors,
        quadrature_order=4,
    )
    model = PersistentEventModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, design.explicit.shape[2],
        raw_enabled=args.raw_enabled, state_dim=args.state_dim,
    ).to(args.device)

    initial = {
        "filtered": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
        "all_correction_off": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
            correction_enabled=False,
        )),
        "validation_correction_off": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
            validation_correction_off=True,
        )),
    }
    parity = {
        key: max(
            abs(initial["filtered"][key] - initial["all_correction_off"][key]),
            abs(initial["filtered"][key] - initial["validation_correction_off"][key]),
        )
        for key in (
            "joint_nll_per_event", "timing_nll_per_event",
            "mark_nll_per_event", "group_size_nll_per_event",
            "subset_nll_per_event",
        )
    }
    if max(parity.values()) > 1e-6:
        raise RuntimeError(f"T1 zero-effect initial parity failed: {parity}")

    model = fit_t1(
        model, design, reader, seed=args.seed, device=args.device,
        epochs=args.epochs, anchor_batch_size=args.anchor_batch_size,
    )
    wrong_permutation, wrong_matched = matched_wrong_time_permutation(
        design, split="validation"
    )
    final = {
        "filtered": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
        "all_correction_off": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
            correction_enabled=False,
        )),
        "validation_correction_off": asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
            validation_correction_off=True,
        )),
    }
    if bool(wrong_matched.any()):
        final["wrong_time"] = asdict(evaluate_t1(
            model, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
            state_permutation=wrong_permutation,
        ))
    arm = "explicit_raw" if args.raw_enabled else "explicit"
    output = args.output_root / args.subject / f"t1_{arm}_d{args.state_dim}_seed_{args.seed}"
    checkpoint = output / "model.pt"
    _atomic_torch(checkpoint, {
        "contract": contract.REVISION,
        "t1_pilot_revision": T1_PILOT_REVISION,
        "subject": args.subject, "seed": args.seed,
        "arm": arm, "state_dim": args.state_dim,
        "model": model.state_dict(),
    })
    result = {
        **design_manifest,
        "status": "COMPLETE",
        "t1_pilot_revision": T1_PILOT_REVISION,
        "subject": args.subject, "seed": args.seed,
        "arm": arm, "state_dim": args.state_dim,
        "epochs_budget": args.epochs,
        "selected_epochs": int(model.selected_epochs),
        "inner_validation_joint_nll": float(model.inner_validation_joint_nll),
        "initial_validation": initial,
        "initial_filtered_minus_correction_off_abs": parity,
        "final_validation": final,
        "wrong_time_match": {
            "n_validation_anchors": int(np.sum(design.anchor_split == 1)),
            "n_matched_anchors": int(wrong_matched.sum()),
            "same_session": True,
            "swapped_object": "filtered_anchor_state",
            "minimum_separation_seconds": 300.0,
            "matching_features": (
                "count traces, load summaries, time of day, session elapsed"
            ),
        },
        "contrasts": {
            "filtered_minus_initial_joint_nll": (
                final["filtered"]["joint_nll_per_event"]
                - initial["filtered"]["joint_nll_per_event"]
            ),
            "filtered_minus_validation_correction_off_joint_nll": (
                final["filtered"]["joint_nll_per_event"]
                - final["validation_correction_off"]["joint_nll_per_event"]
            ),
            "filtered_minus_validation_correction_off_timing_nll": (
                final["filtered"]["timing_nll_per_event"]
                - final["validation_correction_off"]["timing_nll_per_event"]
            ),
            "filtered_minus_validation_correction_off_mark_nll": (
                final["filtered"]["mark_nll_per_event"]
                - final["validation_correction_off"]["mark_nll_per_event"]
            ),
            "filtered_minus_all_correction_off_joint_nll": (
                final["filtered"]["joint_nll_per_event"]
                - final["all_correction_off"]["joint_nll_per_event"]
            ),
            "filtered_minus_wrong_time_joint_nll": (
                final["filtered"]["joint_nll_per_event"]
                - final["wrong_time"]["joint_nll_per_event"]
                if "wrong_time" in final else None
            ),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "sealed_opened": False,
        "claim_boundary": (
            "bounded development T1 filtered-state pilot on sampled 30-s "
            "support; not a cohort or autonomous-state result"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
