#!/usr/bin/env python3
"""Run one stable R1.7A seed through four-arm D_mechanism N=100 T2."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_7_t2 import (
    R1_7_T2_REVISION, build_r1_7a_r2_designs, load_fitted_r1_7a_t1,
)
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    T2_R2_REVISION, ExposureEdge, classify_one_shot_persistence,
    edge_estimability_audit, evaluate_horizon_mark, evaluate_r2_edge, fit_r2_edge,
)
from src.topic5_continuous_marked_state_r1.t2_r2_human import SOURCES


FIT_ARMS = ("real_cumulative", "state_matched_placebo", "current_event_only")


def contrast(left: dict, right: dict) -> dict:
    return {key: float(value - right[key]) for key, value in left.items()
            if isinstance(value, (int, float)) and not key.startswith("n_")}


def save_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.R1_7A_SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--r1-7-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output = args.output_root / args.subject / f"{args.source}_seed_{args.seed}_n_100"
    context = load_fitted_r1_7a_t1(
        args.subject, args.seed, device=args.device, root=args.r1_7_root
    )
    try:
        one_step, horizons, design_audit = build_r1_7a_r2_designs(
            context, source=args.source
        )
    except ValueError as error:
        contract.atomic_json(output / "result.json", {
            "status": "COMPLETE", "analysis_status": "NOT_ESTIMABLE",
            "revision": R1_7_T2_REVISION, "t2_revision": T2_R2_REVISION,
            "subject": args.subject, "seed": args.seed, "source": args.source,
            "non_estimable_reason": str(error), "t1": context.audit,
            "formal_test_partition_opened": False, "sealed_opened": False,
        })
        return
    reference = one_step["real_cumulative"]
    n_train = int(np.sum(reference.split == 0)); n_validation = int(np.sum(reference.split == 1))
    if n_train < 100 or n_validation < 20:
        contract.atomic_json(output / "result.json", {
            "status": "COMPLETE", "analysis_status": "NOT_ESTIMABLE",
            "revision": R1_7_T2_REVISION, "t2_revision": T2_R2_REVISION,
            "subject": args.subject, "seed": args.seed, "source": args.source,
            "non_estimable_reason": f"insufficient pairs: {n_train} TRAIN/{n_validation} D_mechanism",
            "t1": context.audit, "design": design_audit,
            "formal_test_partition_opened": False, "sealed_opened": False,
        })
        return
    estimability = {label: edge_estimability_audit(
        context.model, one_step[label], device=args.device, batch_size=args.batch_size
    ) for label in FIT_ARMS}
    exposure_dim = 1 if reference.exposure.ndim == 1 else reference.exposure.shape[1]
    edges = {"no_edge": ExposureEdge(reference.current_state.shape[1], exposure_dim).to(args.device).eval()}
    fits = {"no_edge": {"selected_epoch": 0, "fixed_B_zero": True}}
    for label in FIT_ARMS:
        edges[label], fits[label] = fit_r2_edge(
            context.model, one_step[label], device=args.device, seed=args.seed,
            epochs=args.epochs, learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    next_event = {label: asdict(evaluate_r2_edge(
        context.model, edge, one_step[label], split="validation",
        device=args.device, batch_size=args.batch_size,
    )) for label, edge in edges.items()}
    horizon_metrics = {f"H{horizon}": {label: asdict(evaluate_horizon_mark(
        context.model, edge, horizons[horizon][label], split="validation",
        device=args.device, batch_size=args.batch_size,
    )) for label, edge in edges.items()} for horizon in (5, 10)}
    controls = ("no_edge", "state_matched_placebo", "current_event_only")
    comparisons = {
        "next_event": {f"real_minus_{right}": contrast(
            next_event["real_cumulative"], next_event[right]
        ) for right in controls},
        **{f"H{h}": {f"real_minus_{right}": contrast(
            horizon_metrics[f"H{h}"]["real_cumulative"],
            horizon_metrics[f"H{h}"][right],
        ) for right in controls} for h in (5, 10)},
    }
    audit = estimability["real_cumulative"]; fit = fits["real_cumulative"]
    estimable = bool(
        audit["gradient_finite"] and audit["gradient_at_zero_norm"] > 1e-8
        and audit["exposure_rank"] == audit["exposure_dim"]
        and min(audit["exposure_sd"]) > 1e-8
        and fit["edge_left_zero_initialisation"]
    )
    primary = bool(estimable and all(
        comparisons["next_event"][f"real_minus_{right}"]["joint_nll_per_event"] < 0
        for right in controls
    ))
    persistence = {f"H{h}": classify_one_shot_persistence(
        comparisons[f"H{h}"]["real_minus_state_matched_placebo"],
        horizon_metrics[f"H{h}"]["real_cumulative"],
        real_edge_estimable=estimable,
    ) for h in (5, 10)}
    checkpoint = output / "edges.pt"
    save_torch(checkpoint, {
        "revision": R1_7_T2_REVISION, "subject": args.subject,
        "seed": args.seed, "source": args.source,
        "edges": {label: edge.state_dict() for label, edge in edges.items()},
    })
    d_events = int(np.sum(
        (context.design.event_split == 1)
        & (context.design.event_time >= context.audit["d_mechanism_start"])
    ))
    result = {
        "status": "COMPLETE", "analysis_status": "ESTIMATED",
        "revision": R1_7_T2_REVISION, "t2_revision": T2_R2_REVISION,
        "subject": args.subject, "seed": args.seed, "source": args.source,
        "scale_events": 100, "t1": context.audit, "design": design_audit,
        "d_mechanism_events": d_events,
        "d_mechanism_nonoverlap_100_event_blocks": d_events // 100,
        "inference_class": "COHORT_ELIGIBLE" if d_events >= 500 else "CASE_ONLY",
        "estimability": estimability, "fits": fits,
        "validation": {"next_event": next_event, "horizons": horizon_metrics},
        "comparisons": comparisons, "real_edge_estimable": estimable,
        "primary_next_event_increment": primary,
        "one_shot_persistence": persistence,
        "free_exposure_intercept_present": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": contract.sha256_file(checkpoint),
        "source_hashes": {
            "runner": contract.sha256_file(Path(__file__)),
            "r1_7_t2": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/r1_7_t2.py"
            ),
            "t2_r2": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_r2.py"
            ),
            "t2_r2_human": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_r2_human.py"
            ),
            "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
        },
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": "development D_mechanism N=100 conditional increment; not causal mechanism",
    }
    contract.atomic_json(output / "result.json", result)


if __name__ == "__main__":
    main()
