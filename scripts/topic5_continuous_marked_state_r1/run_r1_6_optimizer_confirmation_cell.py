#!/usr/bin/env python3
"""Fit and score one frozen-config R1.6 confirmation seed exactly once."""
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
from src.topic5_continuous_marked_state_r1.optimizer_audit import (
    R1_6_REVISION,
    parameter_group_update_norms,
)
from src.topic5_continuous_marked_state_r1.optimizer_runtime import (
    load_explicit_target_model,
)
from src.topic5_continuous_marked_state_r1.r1_2 import evaluate_full_t1
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    evaluate_mark_endpoints,
    median_metric_dict,
    metric_contrast,
    strict_matched_wrong_time_permutations,
)
from src.topic5_continuous_marked_state_r1.r1_3 import (
    fit_target_observer,
    materialize_embedding,
)


CONFIRMATION_REVISION = "r1_6_frozen_optimizer_confirmation_v1"
FIXED_SUBJECTS = (
    "epilepsiae_1096", "epilepsiae_384", "yuquan_zhangkexuan",
    "yuquan_chengshuai", "yuquan_chenziyang", "yuquan_zhangjiaqi",
)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=FIXED_SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
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

    tuning_path = args.output_root / "reports/tuning_summary.json"
    tuning = json.loads(tuning_path.read_text())
    if (tuning.get("status") != "COMPLETE"
            or tuning.get("selection_uses_development_validation") is not False
            or tuning.get("formal_test_partition_opened") is not False):
        raise ValueError("R1.6 tuning summary is not admissible")
    selected_config = str(tuning["selected_config"])
    selected_prefix_config = str(tuning["selected_prefix_config"])
    status = json.loads(
        (args.output_root / "ALIGNMENT_TUNING_STATUS.json").read_text()
    )
    config = status["configs"][selected_config]
    loaded = load_explicit_target_model(
        subject=args.subject, seed=args.seed, device=args.device,
        r1_2_root=args.r1_2_root,
        observation_cache_root=args.observation_cache_root,
        output_root=args.output_root,
        prefix_config_id=selected_prefix_config,
    )
    model = loaded["model"]
    design = loaded["design"]
    loader = loaded["loader"]
    initial = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    trace = fit_target_observer(
        model, design, loader, device=args.device,
        observer_epochs=int(config.get("observer_epochs", 4)),
        joint_epochs=int(config.get("joint_epochs", 4)),
        state_lr=float(config["state_lr"]),
        observer_lr=float(config["state_lr"]) * float(config["observer_ratio"]),
        raw_lr=1e-5, chunk_anchors=int(config["chunk"]),
        optimizer_name=str(config["optimizer"]),
        weight_decay=float(config["weight_decay"]),
        grad_clip_norm=float(config["clip"]),
        warmup_fraction=float(config["warmup"]),
        selection_min_delta=float(config["min_delta"]),
        early_stopping_patience=(
            None if int(config["patience"]) <= 0
            else int(config["patience"])
        ),
        epoch_zero_seen_inner_validation=False,
        refit_mode="full_train",
    )
    embedding = materialize_embedding(
        model, design, loader, device=args.device,
        batch_size=int(config["chunk"]),
    )
    persistent = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="persistent",
    ))
    memoryless = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="memoryless",
    ))
    no_correction = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        correction_enabled=False,
    ))
    validation_off = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        validation_correction_off=True,
    ))
    persistent_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="persistent",
    ))
    memoryless_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="memoryless",
    ))

    coverage = CoverageTable.load(
        args.r1_2_root / "coverage" / f"{args.subject}.npz"
    )
    anchor_segment = np.searchsorted(
        coverage.stop, np.asarray(design.anchor_time, dtype=np.float64),
        side="right",
    )
    if np.any(anchor_segment >= len(coverage.start)):
        raise ValueError("R1.6 confirmation anchor exceeds recorded support")
    inside = (
        (design.anchor_time >= coverage.start[anchor_segment])
        & (design.anchor_time < coverage.stop[anchor_segment])
    )
    if not bool(np.all(inside)):
        raise ValueError("R1.6 confirmation anchor lies in an unrecorded gap")
    observation_coverage = np.asarray(
        loader.cached_contact_mask, dtype=np.float64
    ).mean(1)
    permutations, matched, match_audit = strict_matched_wrong_time_permutations(
        design, observation_coverage, anchor_segment=anchor_segment,
        split="validation", n_donors=5, min_separation_seconds=1800.0,
    )
    matched_correct = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        matched_anchor_mask=matched,
    ))
    matched_correct_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        matched_anchor_mask=matched,
    ))
    wrong = [asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        state_permutation=permutation, matched_anchor_mask=matched,
    )) for permutation in permutations]
    wrong_endpoint = [asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        state_permutation=permutation, matched_anchor_mask=matched,
    )) for permutation in permutations]
    wrong_median = median_metric_dict(wrong)
    wrong_endpoint_median = median_metric_dict(wrong_endpoint)

    output = (
        args.output_root / "confirmation" / selected_prefix_config
        / selected_config
        / args.subject / f"seed_{args.seed}"
    )
    checkpoint = output / "model.pt"
    atomic_torch(checkpoint, {
        "revision": R1_6_REVISION,
        "confirmation_revision": CONFIRMATION_REVISION,
        "subject": args.subject, "seed": int(args.seed),
        "selected_prefix_config": selected_prefix_config,
        "selected_config": selected_config, "config": config,
        "model": model.state_dict(), "fit_trace": asdict(trace),
    })
    persistent_minus_memoryless = metric_contrast(persistent, memoryless)
    correct_minus_wrong = metric_contrast(matched_correct, wrong_median)
    result = {
        "status": "COMPLETE", "revision": R1_6_REVISION,
        "confirmation_revision": CONFIRMATION_REVISION,
        "subject": args.subject, "seed": int(args.seed),
        "selected_prefix_config": selected_prefix_config,
        "selected_config": selected_config, "config": config,
        "fit_trace": asdict(trace),
        "parameter_group_update_norm": parameter_group_update_norms(
            initial, model
        ),
        "validation": {
            "persistent": persistent, "memoryless": memoryless,
            "no_correction": no_correction,
            "validation_correction_off": validation_off,
            "persistent_minus_memoryless": persistent_minus_memoryless,
            "persistent_minus_no_correction": metric_contrast(
                persistent, no_correction
            ),
            "persistent_minus_validation_correction_off": metric_contrast(
                persistent, validation_off
            ),
            "mark_endpoints": {
                "persistent": persistent_endpoint,
                "memoryless": memoryless_endpoint,
                "persistent_minus_memoryless": metric_contrast(
                    persistent_endpoint, memoryless_endpoint
                ),
            },
            "strict_matched_wrong_time": {
                "audit": match_audit, "correct": matched_correct,
                "wrong_median": wrong_median,
                "correct_minus_wrong_median": correct_minus_wrong,
                "endpoint_correct": matched_correct_endpoint,
                "endpoint_wrong_median": wrong_endpoint_median,
                "endpoint_correct_minus_wrong_median": metric_contrast(
                    matched_correct_endpoint, wrong_endpoint_median
                ),
            },
        },
        "stable_checkpoint": bool(
            trace.selected_total_epoch > 0
            and match_audit["n_matched_anchors"] > 0
            and matched_correct["n_events"] > 0
            and persistent_minus_memoryless["joint_nll_per_event"] < 0
            and correct_minus_wrong["joint_nll_per_event"] < 0
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "tuning_summary": str(tuning_path),
        "tuning_summary_sha256": contract.sha256_file(tuning_path),
        "prefix_result": str(loaded["prefix_result_path"]),
        "prefix_result_sha256": contract.sha256_file(
            loaded["prefix_result_path"]
        ),
        "development_validation_scored": True,
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": (
            "development frozen-optimizer confirmation; seed is optimisation "
            "stability, not an independent patient"
        ),
        "seed_role": (
            "independent_optimizer_confirmation" if args.seed in (3, 4)
            else "tuning_seed_rescored_after_config_freeze"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
