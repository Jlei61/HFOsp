#!/usr/bin/env python3
"""Run one frozen R1.4 seed through the N=100 T2-R2.0 experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    T2_R2_REVISION,
    ExposureEdge,
    edge_estimability_audit,
    evaluate_horizon_mark,
    evaluate_r2_edge,
    fit_r2_edge,
)
from src.topic5_continuous_marked_state_r1.t2_r2_human import (
    R1_4_REVISION,
    SOURCES,
    build_r2_arm_designs,
    load_fitted_r1_4_explicit_t1,
)


SUBJECTS = (
    "epilepsiae_620", "epilepsiae_958", "yuquan_huanghanwen",
    "epilepsiae_922", "yuquan_pengzihang", "yuquan_hanyuxuan",
)
FIT_ARMS = (
    "real_cumulative", "state_matched_placebo", "current_event_only",
    "fitted_intercept_diagnostic",
)


def contrast(left: dict, right: dict) -> dict:
    return {
        key: float(value - right[key])
        for key, value in left.items()
        if isinstance(value, (int, float)) and not key.startswith("n_")
    }


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--r1-4-root", type=Path, default=contract.RESULT_ROOT / "r1_4"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "t2_r2"
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    context = load_fitted_r1_4_explicit_t1(
        args.subject, args.seed, device=args.device, r1_4_root=args.r1_4_root
    )
    one_step, horizons, design_audit = build_r2_arm_designs(
        context, source=args.source, scale_events=100
    )
    reference = one_step["real_cumulative"]
    n_train = int((reference.split == 0).sum())
    n_validation = int((reference.split == 1).sum())
    if n_train < 100 or n_validation < 100:
        raise ValueError(
            f"{args.subject}/{args.source}: insufficient N=100 support "
            f"({n_train} TRAIN, {n_validation} validation)"
        )

    estimability = {
        label: edge_estimability_audit(
            context.model, one_step[label], device=args.device,
            batch_size=args.batch_size,
        ) for label in FIT_ARMS
    }
    exposure_dim = (
        1 if reference.exposure.ndim == 1 else reference.exposure.shape[1]
    )
    no_edge = ExposureEdge(
        reference.current_state.shape[1], exposure_dim
    ).to(args.device).eval()
    edges = {"no_edge": no_edge}
    fits = {"no_edge": {
        "selected_epoch": 0, "fixed_B_zero": True,
        "edge_left_zero_initialisation": False,
    }}
    for label in FIT_ARMS:
        edge, fit = fit_r2_edge(
            context.model, one_step[label], device=args.device,
            seed=args.seed, epochs=args.epochs,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
        )
        edges[label] = edge
        fits[label] = fit

    next_event = {
        label: asdict(evaluate_r2_edge(
            context.model, edge, one_step[label], split="validation",
            device=args.device, batch_size=args.batch_size,
        )) for label, edge in edges.items()
    }
    horizon_metrics = {
        f"H{horizon}": {
            label: asdict(evaluate_horizon_mark(
                context.model, edge, horizons[horizon][label],
                split="validation", device=args.device,
                batch_size=args.batch_size,
            )) for label, edge in edges.items()
        } for horizon in (5, 10)
    }
    comparisons = {
        "next_event": {
            f"real_minus_{right}": contrast(
                next_event["real_cumulative"], next_event[right]
            ) for right in (
                "no_edge", "state_matched_placebo", "current_event_only",
                "fitted_intercept_diagnostic",
            )
        },
        **{
            f"H{horizon}": {
                f"real_minus_{right}": contrast(
                    horizon_metrics[f"H{horizon}"]["real_cumulative"],
                    horizon_metrics[f"H{horizon}"][right],
                ) for right in (
                    "no_edge", "state_matched_placebo", "current_event_only",
                    "fitted_intercept_diagnostic",
                )
            } for horizon in (5, 10)
        },
    }
    real_audit = estimability["real_cumulative"]
    real_fit = fits["real_cumulative"]
    estimable = bool(
        real_audit["gradient_finite"]
        and real_audit["gradient_at_zero_norm"] > 1e-8
        and real_audit["exposure_rank"] == real_audit["exposure_dim"]
        and min(real_audit["exposure_sd"]) > 1e-8
        and real_fit["edge_left_zero_initialisation"]
    )
    primary_increment = bool(
        estimable
        and comparisons["next_event"][
            "real_minus_state_matched_placebo"
        ]["joint_nll_per_event"] < 0
        and comparisons["next_event"][
            "real_minus_current_event_only"
        ]["joint_nll_per_event"] < 0
    )
    persistence = {}
    for horizon in (5, 10):
        mark_increment = bool(
            comparisons[f"H{horizon}"][
                "real_minus_state_matched_placebo"
            ]["mark_nll_per_event"] < 0
        )
        state_increment = bool(
            comparisons[f"H{horizon}"][
                "real_minus_state_matched_placebo"
            ]["state_mse_to_filtered_target"] < 0
        )
        persistence[f"H{horizon}"] = {
            "mark_prediction_increment": mark_increment,
            "state_prediction_increment": state_increment,
            "state_and_mark_persist": bool(mark_increment and state_increment),
            "nonzero_propagated_displacement": bool(
                horizon_metrics[f"H{horizon}"]["real_cumulative"][
                    "mean_state_displacement_from_no_edge"
                ] > 1e-8
            ),
        }

    output = (
        args.output_root / "human" / args.subject
        / f"{args.source}_seed_{args.seed}_n_100"
    )
    checkpoint_path = output / "edges.pt"
    atomic_torch(checkpoint_path, {
        "revision": T2_R2_REVISION,
        "r1_4_revision": R1_4_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "source": args.source,
        "edges": {label: edge.state_dict() for label, edge in edges.items()},
    })
    result = {
        "status": "COMPLETE",
        "revision": T2_R2_REVISION,
        "r1_4_revision": R1_4_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "source": args.source,
        "scale_events": 100,
        "t1": context.audit,
        "design": design_audit,
        "estimability": estimability,
        "fits": fits,
        "edge_matrices": {
            label: edge.matrix.detach().cpu().numpy().tolist()
            for label, edge in edges.items()
        },
        "validation": {
            "next_event": next_event,
            "horizons": horizon_metrics,
        },
        "comparisons": comparisons,
        "real_edge_estimable": estimable,
        "primary_next_event_increment": primary_increment,
        "one_shot_persistence": persistence,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "source_hashes": {
            "t2_r2": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_r2.py"
            ),
            "t2_r2_human": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_r2_human.py"
            ),
            "runner": contract.sha256_file(Path(__file__)),
            "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "development N=100 residual event-edge screen conditional on a "
            "frozen R1.4 explicit T1 state; H5/H10 use one anchor jump, frozen "
            "generator flow, true fixed history covariates, no later raw "
            "correction and no later T2 jump"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps({
        "status": result["status"], "subject": args.subject,
        "seed": args.seed, "source": args.source,
        "real_edge_estimable": estimable,
        "primary_next_event_increment": primary_increment,
        "one_shot_persistence": persistence,
        "output": str(output / "result.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
