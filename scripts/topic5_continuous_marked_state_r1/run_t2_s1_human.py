#!/usr/bin/env python3
"""Run one formal development T2-S1 subject/seed/event-scale fit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_human import (
    T2_HUMAN_REVISION,
    build_exposure_arm_designs,
    load_fitted_explicit_t1,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import (
    SignedExposureEdge,
    evaluate_edge,
    fit_edge,
)


SUBJECTS = ("epilepsiae_620", "epilepsiae_958")
SCALES = (100, 1000)


def contrast(left: dict, right: dict) -> dict:
    return {
        key: float(value - right[key])
        for key, value in left.items()
        if isinstance(value, (int, float)) and not key.startswith("n_")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--scale-events", required=True, type=int, choices=SCALES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "t2_s1_long_scale",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    context = load_fitted_explicit_t1(
        args.subject, args.seed, device=args.device
    )
    arms, design_audit = build_exposure_arm_designs(
        context, scale_events=args.scale_events
    )
    reference = arms["no_edge"]
    n_train = int((reference.split == 0).sum())
    n_validation = int((reference.split == 1).sum())
    if n_train < 100 or n_validation < 100:
        raise ValueError(
            f"{args.subject} N={args.scale_events}: insufficient one-step support "
            f"({n_train} train, {n_validation} validation)"
        )
    metrics = {}
    fits = {}
    vectors = {}
    null_edge = SignedExposureEdge(reference.current_state.shape[1]).to(args.device)
    null_edge.eval()
    metrics["no_edge"] = asdict(evaluate_edge(
        context.model, null_edge, reference, split="validation",
        device=args.device, batch_size=args.batch_size,
    ))
    fits["no_edge"] = {"selected_epoch": 0, "fixed_zero_edge": True}
    vectors["no_edge"] = null_edge.vector.detach().cpu().tolist()
    for arm in ("real_cumulative", "state_matched_placebo", "current_event_only"):
        edge, fit_audit = fit_edge(
            context.model, arms[arm], device=args.device, seed=args.seed,
            epochs=args.epochs, learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        metrics[arm] = asdict(evaluate_edge(
            context.model, edge, arms[arm], split="validation",
            device=args.device, batch_size=args.batch_size,
        ))
        fits[arm] = fit_audit
        vectors[arm] = edge.vector.detach().cpu().tolist()
    comparisons = {
        "real_minus_no_edge": contrast(metrics["real_cumulative"], metrics["no_edge"]),
        "real_minus_state_matched_placebo": contrast(
            metrics["real_cumulative"], metrics["state_matched_placebo"]
        ),
        "current_event_minus_no_edge": contrast(
            metrics["current_event_only"], metrics["no_edge"]
        ),
        "placebo_minus_no_edge": contrast(
            metrics["state_matched_placebo"], metrics["no_edge"]
        ),
    }
    output = (
        args.output_root / "human" / args.subject
        / f"seed_{args.seed}_n_{args.scale_events}"
    )
    result = {
        "status": "COMPLETE",
        "revision": T2_HUMAN_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "scale_events": int(args.scale_events),
        "t1": context.audit,
        "design": design_audit,
        "fits": fits,
        "edge_vectors": vectors,
        "validation": metrics,
        "comparisons": comparisons,
        "primary_endpoint": "one-step exact joint timing plus full sequential mark NLL",
        "lower_nll_is_better": True,
        "ordinary_null_is_not_a_blocker": True,
        "n_100_role": "short-scale reference" if args.scale_events == 100 else None,
        "n_1000_role": "current multi-patient long-scale primary" if args.scale_events == 1000 else None,
        "n_10000_role": (
            "deferred until a target-trained T1 checkpoint exists in an eligible "
            "high-event patient; no fixed R1.3 pilot patient is eligible"
        ),
        "sealed_opened": False,
        "claim_boundary": (
            "development one-step residual generator-edge screen conditional on "
            "the fitted explicit T1 state; not yet a recursively simulated or "
            "causal physiological mechanism"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
