#!/usr/bin/env python3
"""Run the paired explicit versus explicit+raw Bridge-E1 experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import (
    BRIDGE_E1_REVISION,
    batch_log_terms,
    build_bridge_e1_design,
    evaluate_bridge_e1,
    fit_bridge_e1,
    make_paired_models,
)
from src.topic5_continuous_marked_state_r1.data import load_event_stream


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--anchor-batch-size", type=int, default=4)
    parser.add_argument("--max-train-anchors", type=int, default=256)
    parser.add_argument("--max-validation-anchors", type=int, default=128)
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "bridge_e1")
    args = parser.parse_args()

    baseline_path = (
        contract.RESULT_ROOT / "baselines" / args.subject
        / f"seed_{args.seed}" / "models.pt"
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline checkpoint missing: {baseline_path}")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    stream = load_event_stream(args.subject)
    design, reader, manifest = build_bridge_e1_design(
        args.subject, baseline_path,
        max_train_anchors=args.max_train_anchors,
        max_validation_anchors=args.max_validation_anchors,
        quadrature_order=4,
    )
    explicit, raw = make_paired_models(
        baseline, design, stream.adjacency, seed=args.seed, device=args.device
    )

    parity_anchor = design.anchor_ids("train")[:min(4, args.anchor_batch_size)]
    with torch.no_grad():
        left = batch_log_terms(explicit, design, reader, parity_anchor, args.device)
        right = batch_log_terms(raw, design, reader, parity_anchor, args.device)
    parity = {
        key: abs(float(left[key]) - float(right[key]))
        for key in ("event_log", "survival", "mark_log", "size_log", "subset_log")
    }
    if max(parity.values(), default=0.0) > 1e-6:
        raise RuntimeError(f"zero-raw initial parity failed: {parity}")

    initial = {
        "explicit": asdict(evaluate_bridge_e1(
            explicit, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
        "explicit_raw": asdict(evaluate_bridge_e1(
            raw, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
    }
    explicit = fit_bridge_e1(
        explicit, design, reader, seed=args.seed, device=args.device,
        epochs=args.epochs, anchor_batch_size=args.anchor_batch_size,
    )
    raw = fit_bridge_e1(
        raw, design, reader, seed=args.seed, device=args.device,
        epochs=args.epochs, anchor_batch_size=args.anchor_batch_size,
    )
    final = {
        "explicit": asdict(evaluate_bridge_e1(
            explicit, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
        "explicit_raw": asdict(evaluate_bridge_e1(
            raw, design, reader, "validation", device=args.device,
            anchor_batch_size=args.anchor_batch_size,
        )),
    }
    output = args.output_root / args.subject / f"seed_{args.seed}"
    checkpoint = output / "models.pt"
    _atomic_torch(checkpoint, {
        "contract": contract.REVISION,
        "bridge_e1_revision": BRIDGE_E1_REVISION,
        "subject": args.subject,
        "seed": args.seed,
        "explicit": explicit.state_dict(),
        "explicit_raw": raw.state_dict(),
        "raw_gain": float(raw.observer.raw_gain.detach().cpu()),
    })
    result = {
        **manifest,
        "status": "COMPLETE",
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "selected_epochs": {
            "explicit": int(explicit.selected_epochs),
            "explicit_raw": int(raw.selected_epochs),
        },
        "inner_validation_joint_nll": {
            "explicit": float(explicit.inner_validation_joint_nll),
            "explicit_raw": float(raw.inner_validation_joint_nll),
        },
        "anchor_batch_size": args.anchor_batch_size,
        "zero_raw_initial_parity_abs": parity,
        "initial_validation": initial,
        "final_validation": final,
        "contrasts_raw_minus_explicit": {
            key: final["explicit_raw"][key] - final["explicit"][key]
            for key in (
                "joint_nll_per_event", "timing_nll_per_event",
                "mark_nll_per_event", "group_size_nll_per_event",
                "subset_nll_per_event",
            )
        },
        "raw_gain": float(raw.observer.raw_gain.detach().cpu()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "claim_boundary": (
            "paired development Bridge-E1 on exact event likelihood; a raw gain "
            "is observation information, not yet a persistent-state result"
        ),
        "sealed_opened": False,
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
