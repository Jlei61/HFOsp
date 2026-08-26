#!/usr/bin/env python3
"""Fit one subject's exact timing and tied-mark history baselines."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.baseline import (
    fit_history_intensity,
    fit_mark_decoder,
    intensity_metrics,
    mark_metrics,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.data import load_event_stream
from src.topic5_continuous_marked_state_r1.design import build_subject_design


BASELINE_REVISION = "r1_exact_history_timing_mark_baseline_v1"


def _cyclic_session_permutation(session: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    output = np.arange(len(session))
    for label in np.unique(session):
        index = np.flatnonzero(session == label)
        if len(index) <= 1:
            continue
        shift = int(rng.integers(1, len(index)))
        output[index] = np.roll(index, shift)
    return output


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mark-epochs", type=int, default=30)
    parser.add_argument("--output-root", type=Path,
                        default=contract.RESULT_ROOT / "baselines")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stream = load_event_stream(args.subject)
    coverage_path = contract.RESULT_ROOT / "coverage" / f"{args.subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    design = build_subject_design(stream, coverage, quadrature_order=4)

    timing_models = {
        "static": fit_history_intensity(
            design.train, history_visible=False, device=args.device
        ),
        "history": fit_history_intensity(
            design.train, history_visible=True, device=args.device
        ),
    }
    timing_result = {
        arm: {
            "train": asdict(intensity_metrics(model, design.train, device=args.device)),
            "validation": asdict(intensity_metrics(
                model, design.validation, device=args.device
            )),
        }
        for arm, model in timing_models.items()
    }

    train_index = design.train.event_index
    validation_index = design.validation.event_index
    train_group = stream.group_ids[train_index]
    train_count = stream.group_count[train_index]
    validation_group = stream.group_ids[validation_index]
    validation_count = stream.group_count[validation_index]
    train_session = stream.session[train_index]
    permutation = _cyclic_session_permutation(train_session, args.seed)
    mark_models = {
        "static": fit_mark_decoder(
            design.train.event_history, train_group, train_count, stream.adjacency,
            history_visible=False, seed=args.seed, epochs=args.mark_epochs,
            device=args.device,
        ),
        "history": fit_mark_decoder(
            design.train.event_history, train_group, train_count, stream.adjacency,
            history_visible=True, seed=args.seed, epochs=args.mark_epochs,
            device=args.device,
        ),
        "shuffled_history": fit_mark_decoder(
            design.train.event_history[permutation], train_group, train_count,
            stream.adjacency, history_visible=True, seed=args.seed,
            epochs=args.mark_epochs, device=args.device,
        ),
    }
    mark_result = {
        arm: {
            "train": asdict(mark_metrics(
                model, design.train.event_history, train_group, train_count,
                device=args.device,
            )),
            "validation": asdict(mark_metrics(
                model, design.validation.event_history,
                validation_group, validation_count, device=args.device,
            )),
        }
        for arm, model in mark_models.items()
    }

    output_dir = args.output_root / args.subject / f"seed_{args.seed}"
    checkpoint = output_dir / "models.pt"
    _atomic_torch(checkpoint, {
        "contract": contract.REVISION,
        "baseline_revision": BASELINE_REVISION,
        "subject": args.subject,
        "seed": args.seed,
        "timing": {name: model.state_dict() for name, model in timing_models.items()},
        "mark": {name: model.state_dict() for name, model in mark_models.items()},
        "history_scaler": {
            "mean": design.scaler.mean,
            "scale": design.scaler.scale,
        },
    })
    result = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "baseline_revision": BASELINE_REVISION,
        "subject": args.subject,
        "seed": args.seed,
        "device": args.device,
        "n_contacts": stream.n_contacts,
        "n_train_events": int(len(train_index)),
        "n_validation_events": int(len(validation_index)),
        "train_recorded_hours": float(design.train.recorded_seconds / 3600.0),
        "validation_recorded_hours": float(design.validation.recorded_seconds / 3600.0),
        "history_dim": int(design.train.event_history.shape[1]),
        "timing": timing_result,
        "mark": mark_result,
        "contrasts": {
            "timing_history_minus_static_validation_nll": (
                timing_result["history"]["validation"]["nll_per_event"]
                - timing_result["static"]["validation"]["nll_per_event"]
            ),
            "mark_history_minus_static_validation_nll": (
                mark_result["history"]["validation"]["event_nll"]
                - mark_result["static"]["validation"]["event_nll"]
            ),
            "mark_history_minus_shuffled_validation_nll": (
                mark_result["history"]["validation"]["event_nll"]
                - mark_result["shuffled_history"]["validation"]["event_nll"]
            ),
        },
        "coverage_manifest_sha256": contract.sha256_file(
            coverage_path.with_suffix(".manifest.json")
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": None,
        "sealed_opened": False,
        "claim_boundary": (
            "event-only R1 instrument smoke; negative values favour correctly "
            "aligned deterministic history and do not establish a latent state"
        ),
    }
    result["checkpoint_sha256"] = contract.sha256_file(checkpoint)
    contract.atomic_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
