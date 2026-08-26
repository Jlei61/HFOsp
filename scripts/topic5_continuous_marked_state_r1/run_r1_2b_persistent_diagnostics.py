#!/usr/bin/env python3
"""Post-hoc persistent-vs-memoryless and strict-swap diagnostics for R1.2b."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import make_paired_models
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    _bridge_scaler,
    evaluate_full_t1,
    load_full_admissible_event_stream,
    load_full_design,
)
from src.topic5_continuous_marked_state_r1.r1_2b import (
    R1_2B_REVISION,
    R1_2B_SUBJECTS,
    JointLastLayerStateModel,
    load_joint_node_cache,
    materialize_joint_embedding,
)
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    DIAGNOSTIC_REVISION,
    evaluate_mark_endpoints,
    median_metric_dict,
    metric_contrast,
    strict_matched_wrong_time_permutations,
)


def load_fitted_model(args, design, stream, baseline, cache_manifest):
    bridge_result_path = (
        args.r1_2_root / "bridge_e1" / args.subject / "seed_0/result.json"
    )
    bridge_result = json.loads(bridge_result_path.read_text())
    coverage = CoverageTable.load(
        args.r1_2_root / "coverage" / f"{args.subject}.npz"
    )
    _, _, sampled, _ = _bridge_scaler(
        args.subject,
        args.r1_2_root / "baselines" / args.subject / "seed_0/models.pt",
        bridge_result,
        stream,
        coverage,
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
    model = JointLastLayerStateModel(
        baseline,
        design.event_history.shape[1],
        stream.n_contacts,
        stream.adjacency,
        raw_bridge.observer,
        raw_enabled=args.arm == "joint_explicit_raw",
        state_dim=8,
    ).to(args.device)
    fit_root = args.r1_2b_root / "joint" / args.subject / f"{args.arm}_seed_{args.seed}"
    result_path = fit_root / "result.json"
    checkpoint_path = fit_root / "model.pt"
    fit_result = json.loads(result_path.read_text())
    if fit_result.get("status") != "COMPLETE":
        raise ValueError(f"incomplete fitted R1.2b result: {result_path}")
    if fit_result.get("sealed_opened") is not False:
        raise ValueError("R1.2b fit opened the sealed partition")
    if fit_result.get("r1_2b_revision") != R1_2B_REVISION:
        raise ValueError("R1.2b revision mismatch")
    if contract.sha256_file(checkpoint_path) != fit_result["checkpoint_sha256"]:
        raise ValueError("R1.2b checkpoint hash mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.selected_epochs = int(checkpoint["selected_epochs"])
    return model.eval(), result_path, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=R1_2B_SUBJECTS)
    parser.add_argument(
        "--arm", required=True,
        choices=("joint_explicit", "joint_explicit_raw"),
    )
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-donors", type=int, default=5)
    parser.add_argument("--min-separation-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--r1-2b-root", type=Path, default=contract.RESULT_ROOT / "r1_2b"
    )
    args = parser.parse_args()

    base, raw, contact_mask, cache_manifest = load_joint_node_cache(
        args.subject, output_root=args.r1_2b_root
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
    model, fit_result_path, checkpoint_path = load_fitted_model(
        args, design, stream, baseline, cache_manifest
    )
    embedding = materialize_joint_embedding(
        model, base, raw, contact_mask, device=args.device
    )
    persistent = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="persistent",
    ))
    memoryless = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        anchor_state_mode="memoryless",
    ))
    persistent_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="persistent",
    ))
    memoryless_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        anchor_state_mode="memoryless",
    ))

    observation_coverage = np.asarray(contact_mask, dtype=np.float64).mean(1)
    permutations, matched, match_audit = strict_matched_wrong_time_permutations(
        design,
        observation_coverage,
        n_donors=args.n_donors,
        min_separation_seconds=args.min_separation_seconds,
    )
    matched_correct = asdict(evaluate_full_t1(
        model, design, embedding, "validation", device=args.device,
        matched_anchor_mask=matched,
    ))
    matched_correct_endpoint = asdict(evaluate_mark_endpoints(
        model, design, embedding, device=args.device,
        matched_anchor_mask=matched,
    ))
    wrong = []
    wrong_endpoint = []
    for permutation in permutations:
        wrong.append(asdict(evaluate_full_t1(
            model, design, embedding, "validation", device=args.device,
            state_permutation=permutation, matched_anchor_mask=matched,
        )))
        wrong_endpoint.append(asdict(evaluate_mark_endpoints(
            model, design, embedding, device=args.device,
            state_permutation=permutation, matched_anchor_mask=matched,
        )))
    wrong_median = median_metric_dict(wrong)
    wrong_endpoint_median = median_metric_dict(wrong_endpoint)

    fit_result = json.loads(fit_result_path.read_text())
    payload = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "subject": args.subject,
        "arm": args.arm,
        "seed": int(args.seed),
        "selected_epochs": int(model.selected_epochs),
        "persistent": persistent,
        "memoryless": memoryless,
        "persistent_minus_memoryless": metric_contrast(persistent, memoryless),
        "mark_endpoints": {
            "persistent": persistent_endpoint,
            "memoryless": memoryless_endpoint,
            "persistent_minus_memoryless": metric_contrast(
                persistent_endpoint, memoryless_endpoint
            ),
        },
        "strict_matched_wrong_time": {
            "audit": match_audit,
            "correct": matched_correct,
            "wrong_donor_metrics": wrong,
            "wrong_median": wrong_median,
            "correct_minus_wrong_median": metric_contrast(
                matched_correct, wrong_median
            ),
            "endpoint_correct": matched_correct_endpoint,
            "endpoint_wrong_donor_metrics": wrong_endpoint,
            "endpoint_wrong_median": wrong_endpoint_median,
            "endpoint_correct_minus_wrong_median": metric_contrast(
                matched_correct_endpoint, wrong_endpoint_median
            ),
        },
        "fitted_result": str(fit_result_path),
        "fitted_result_sha256": contract.sha256_file(fit_result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": fit_result["checkpoint_sha256"],
        "full_recorded_support": True,
        "sealed_opened": False,
        "claim_boundary": (
            "post-hoc development diagnostic on the frozen R1.2b checkpoint; "
            "memoryless isolates current-window observation from cross-anchor carry; "
            "no raw-backbone training, cohort, seizure or H3 claim"
        ),
    }
    output = (
        args.r1_2b_root / "diagnostics" / args.subject
        / f"{args.arm}_seed_{args.seed}" / "result.json"
    )
    contract.atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
