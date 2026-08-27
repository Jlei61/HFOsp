#!/usr/bin/env python3
"""Fit one frozen-config R1.7A seed and score only the D_state layer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.optimizer_audit import parameter_group_update_norms
from src.topic5_continuous_marked_state_r1.optimizer_runtime import load_explicit_target_model
from src.topic5_continuous_marked_state_r1.r1_2 import (
    evaluate_full_t1, filtered_anchor_states, memoryless_anchor_states,
)
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    evaluate_mark_endpoints, median_metric_dict, metric_contrast,
    strict_matched_wrong_time_permutations,
)
from src.topic5_continuous_marked_state_r1.r1_3 import fit_target_observer, materialize_embedding
from src.topic5_continuous_marked_state_r1.r1_7 import (
    NONFINITE_GRADIENT_STATUS, R1_7A_REVISION, block_bootstrap_length_seconds,
    is_nonfinite_gradient_failure, split_validation_by_recorded_time,
)


def source_hashes() -> dict[str, str]:
    return {
        "runner": contract.sha256_file(Path(__file__)),
        "r1_7": contract.sha256_file(
            contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/r1_7.py"
        ),
        "r1_2": contract.sha256_file(
            contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/r1_2.py"
        ),
        "diagnostics": contract.sha256_file(
            contract.REPO_ROOT
            / "src/topic5_continuous_marked_state_r1/r1_2b_diagnostics.py"
        ),
        "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.R1_7A_SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--r1-2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--r1-6-root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    inventory_path = args.output_root / "manifests/cohort_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("status") != "FROZEN"
            or inventory.get("selection_uses_model_outcomes") is not False
            or args.subject not in inventory.get("selected_subjects", [])):
        raise ValueError("invalid R1.7A prospective cohort manifest")
    config_path = args.r1_6_root / "reports/recommended_optimizer_config.json"
    frozen = json.loads(config_path.read_text())
    prefix = frozen["prefix_core"]; align = frozen["target_alignment"]
    loaded = load_explicit_target_model(
        subject=args.subject, seed=args.seed, device=args.device,
        r1_2_root=args.r1_2_root,
        observation_cache_root=args.output_root / "cache",
        output_root=args.output_root,
        prefix_config_id=prefix["config_id"],
    )
    model, design, loader = loaded["model"], loaded["design"], loaded["loader"]
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output = args.output_root / "fits" / args.subject / f"seed_{args.seed}"
    try:
        trace = fit_target_observer(
            model, design, loader, device=args.device,
            observer_epochs=int(align["observer_epochs"]),
            joint_epochs=int(align["joint_epochs"]),
            state_lr=float(align["state_lr"]),
            observer_lr=float(align["state_lr"]) * float(align["observer_ratio"]),
            raw_lr=1e-5, chunk_anchors=int(align["chunk"]),
            optimizer_name=str(align["optimizer"]),
            weight_decay=float(align["weight_decay"]),
            grad_clip_norm=float(align["clip"]),
            warmup_fraction=float(align["warmup"]),
            selection_min_delta=float(align["min_delta"]),
            early_stopping_patience=None,
            epoch_zero_seen_inner_validation=False, refit_mode="full_train",
        )
    except RuntimeError as error:
        # The frozen optimiser's own non-finite guard: record the seed as an
        # instrument failure instead of aborting the whole ten-patient cohort.
        # It is never scored and never counts as a stable checkpoint.  Any other
        # RuntimeError is an implementation fault and must still abort.
        if not is_nonfinite_gradient_failure(error):
            raise
        contract.atomic_json(output / "result.json", {
            "status": "COMPLETE",
            "analysis_status": NONFINITE_GRADIENT_STATUS,
            "revision": R1_7A_REVISION,
            "subject": args.subject, "seed": args.seed,
            "stable_checkpoint": False,
            "d_mechanism_scored_here": False,
            "nonfinite_reason": str(error),
            "cohort_inventory": str(inventory_path),
            "cohort_inventory_sha256": contract.sha256_file(inventory_path),
            "frozen_r1_6_config": str(config_path),
            "frozen_r1_6_config_sha256": contract.sha256_file(config_path),
            "source_hashes": source_hashes(),
            "development_validation_used_for_selection": False,
            "formal_test_partition_opened": False, "sealed_opened": False,
        })
        return
    embedding = materialize_embedding(
        model, design, loader, device=args.device, batch_size=int(align["chunk"])
    )
    coverage_path = args.r1_2_root / "coverage" / f"{args.subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    layer = split_validation_by_recorded_time(
        coverage, validation_start=coverage.train_end_epoch,
        validation_stop=coverage.dev_end_epoch,
    )
    bounds = {"time_lower": layer.state_start, "time_upper": layer.state_stop}
    with torch.no_grad():
        persistent_state = filtered_anchor_states(
            model, design, embedding, device=args.device
        )
        memoryless_state = memoryless_anchor_states(
            model, design, embedding, device=args.device
        )
    persistent = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="persistent", anchor_state_override=persistent_state,
        **bounds,
    ))
    memoryless = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="memoryless", anchor_state_override=memoryless_state,
        **bounds,
    ))
    persistent_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="persistent", anchor_state_override=persistent_state,
        **bounds,
    ))
    memoryless_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="memoryless", anchor_state_override=memoryless_state,
        **bounds,
    ))
    anchor_segment = np.searchsorted(
        coverage.stop, np.asarray(design.anchor_time, dtype=np.float64), side="right"
    )
    if np.any(anchor_segment >= len(coverage.start)):
        raise ValueError("anchor exceeds recorded support")
    observation_coverage = np.asarray(loader.cached_contact_mask, dtype=np.float64).mean(1)
    permutations, matched, match_audit = strict_matched_wrong_time_permutations(
        design, observation_coverage, anchor_segment=anchor_segment,
        split="validation", n_donors=5, min_separation_seconds=1800.0,
        **bounds,
    )
    correct = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        matched_anchor_mask=matched, anchor_state_override=persistent_state,
        **bounds,
    ))
    correct_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        matched_anchor_mask=matched, anchor_state_override=persistent_state,
        **bounds,
    ))
    wrong = []; wrong_endpoint = []
    for permutation in permutations:
        wrong.append(asdict(evaluate_full_t1(
            model, design, embedding, "validation", device=args.device,
            state_permutation=permutation, matched_anchor_mask=matched,
            anchor_state_override=persistent_state, **bounds,
        )))
        wrong_endpoint.append(asdict(evaluate_mark_endpoints(
            model, design, embedding, device=args.device,
            state_permutation=permutation, matched_anchor_mask=matched,
            anchor_state_override=persistent_state, **bounds,
        )))
    wrong_median = median_metric_dict(wrong)
    wrong_endpoint_median = median_metric_dict(wrong_endpoint)
    train = design.event_split == 0
    bootstrap_seconds = block_bootstrap_length_seconds(
        design.event_time[train], design.event_session[train]
    )
    # Save continuous, session-respecting block effects.  The patient-first
    # aggregator resamples these blocks after taking the five-seed median;
    # seeds themselves are never treated as scientific replicates.
    block_rows = []
    left = np.maximum(coverage.start, layer.state_start)
    right = np.minimum(coverage.stop, layer.state_stop)
    for segment, (start, stop) in enumerate(zip(left, right)):
        if stop <= start:
            continue
        cursor = float(start)
        while cursor < float(stop):
            block_stop = min(float(stop), cursor + bootstrap_seconds)
            block_bounds = {"time_lower": cursor, "time_upper": block_stop}
            block_p = asdict(evaluate_full_t1(
                model, design, embedding, "validation", device=args.device,
                anchor_state_override=persistent_state, **block_bounds,
            ))
            if block_p["n_events"]:
                block_m = asdict(evaluate_full_t1(
                    model, design, embedding, "validation", device=args.device,
                    anchor_state_override=memoryless_state, **block_bounds,
                ))
                block_pe = asdict(evaluate_mark_endpoints(
                    model, design, embedding, device=args.device,
                    anchor_state_override=persistent_state, **block_bounds,
                ))
                block_me = asdict(evaluate_mark_endpoints(
                    model, design, embedding, device=args.device,
                    anchor_state_override=memoryless_state, **block_bounds,
                ))
                block_correct = asdict(evaluate_full_t1(
                    model, design, embedding, "validation", device=args.device,
                    matched_anchor_mask=matched,
                    anchor_state_override=persistent_state, **block_bounds,
                ))
                block_wrong = [asdict(evaluate_full_t1(
                    model, design, embedding, "validation", device=args.device,
                    state_permutation=permutation, matched_anchor_mask=matched,
                    anchor_state_override=persistent_state, **block_bounds,
                )) for permutation in permutations]
                block_correct_endpoint = asdict(evaluate_mark_endpoints(
                    model, design, embedding, device=args.device,
                    matched_anchor_mask=matched,
                    anchor_state_override=persistent_state, **block_bounds,
                ))
                block_wrong_endpoint = [asdict(evaluate_mark_endpoints(
                    model, design, embedding, device=args.device,
                    state_permutation=permutation,
                    matched_anchor_mask=matched,
                    anchor_state_override=persistent_state, **block_bounds,
                )) for permutation in permutations]
                block_rows.append({
                    "segment": int(segment), "start": cursor,
                    "stop": block_stop, "n_events": int(block_p["n_events"]),
                    "n_matched_events": int(block_correct["n_events"]),
                    "persistent_minus_memoryless": metric_contrast(block_p, block_m),
                    "persistent_minus_memoryless_endpoints": metric_contrast(
                        block_pe, block_me
                    ),
                    "correct_minus_wrong": metric_contrast(
                        block_correct, median_metric_dict(block_wrong)
                    ),
                    "correct_minus_wrong_endpoints": metric_contrast(
                        block_correct_endpoint,
                        median_metric_dict(block_wrong_endpoint),
                    ),
                })
            cursor = block_stop
    checkpoint = output / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({
        "revision": R1_7A_REVISION, "subject": args.subject,
        "seed": args.seed, "model": model.state_dict(), "fit_trace": asdict(trace),
        "d_state": asdict(layer),
    }, temporary)
    temporary.replace(checkpoint)
    pm = metric_contrast(persistent, memoryless)
    cw = metric_contrast(correct, wrong_median)
    payload = {
        "status": "COMPLETE", "revision": R1_7A_REVISION,
        "subject": args.subject, "seed": args.seed,
        "frozen_r1_6_config": str(config_path),
        "frozen_r1_6_config_sha256": contract.sha256_file(config_path),
        "cohort_inventory": str(inventory_path),
        "cohort_inventory_sha256": contract.sha256_file(inventory_path),
        "fit_trace": asdict(trace),
        "parameter_group_update_norm": parameter_group_update_norms(initial, model),
        "d_state": {
            "support": asdict(layer),
            "persistent": persistent, "memoryless": memoryless,
            "persistent_minus_memoryless": pm,
            "mark_endpoints": {
                "persistent": persistent_endpoint,
                "memoryless": memoryless_endpoint,
                "persistent_minus_memoryless": metric_contrast(
                    persistent_endpoint, memoryless_endpoint
                ),
            },
            "strict_matched_wrong_time": {
                "audit": match_audit, "correct": correct,
                "wrong_median": wrong_median,
                "correct_minus_wrong_median": cw,
                "endpoint_correct": correct_endpoint,
                "endpoint_wrong_median": wrong_endpoint_median,
                "endpoint_correct_minus_wrong_median": metric_contrast(
                    correct_endpoint, wrong_endpoint_median
                ),
            },
            "bootstrap_block_seconds_frozen_from_train": bootstrap_seconds,
            "nonoverlap_time_blocks": block_rows,
            "n_nonempty_time_blocks": len(block_rows),
        },
        "stable_checkpoint": bool(
            trace.selected_total_epoch > 0
            and match_audit["n_matched_anchors"] > 0
            and persistent["n_events"] > 0
            and pm["joint_nll_per_event"] < 0
            and cw["joint_nll_per_event"] < 0
        ),
        "d_mechanism_scored_here": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "source_hashes": source_hashes(),
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(output / "result.json", payload)


if __name__ == "__main__":
    main()
