#!/usr/bin/env python3
"""Run one frozen N=1000 H3 cell from a stable R1.6 T1 checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_SUPPORT_REVISION,
    SOURCES,
    AffineExposureEdge,
    affine_estimability_audit,
    classify_affine_estimability,
    evaluate_affine_edge,
    fit_affine_edge,
)
from src.topic5_continuous_marked_state_r1.h3_long_human import (
    build_long_arm_designs,
    event_innovation,
)
from src.topic5_continuous_marked_state_r1.optimizer_h3 import (
    R1_6_MINIMAL_H3_REVISION,
    load_fitted_r1_6_confirmation_t1,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import _evaluate_rows


SCALE_EVENTS = 1000


def contrast(left: dict, right: dict) -> dict:
    return {
        key: float(value - right[key])
        for key, value in left.items()
        if isinstance(value, (int, float)) and not key.startswith("n_")
    }


def block_contrasts(model, edges: dict, designs: dict,
                    rows: np.ndarray, *, device: str) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    metrics = {}
    for label, edge in edges.items():
        design = (
            designs["real_cumulative"] if label == "no_edge"
            else designs[label]
        )
        metrics[label] = [
            asdict(_evaluate_rows(
                model, edge, design, np.asarray([row]),
                device=device, batch_size=1,
            ))
            for row in rows
        ]
    comparisons = {}
    for right in metrics:
        if right == "real_cumulative":
            continue
        values = np.asarray([
            left["joint_nll_per_event"] - other["joint_nll_per_event"]
            for left, other in zip(metrics["real_cumulative"], metrics[right])
        ], dtype=np.float64)
        leave_one_out = (
            (values.sum() - values) / (len(values) - 1)
            if len(values) > 1 else values.copy()
        )
        comparisons[f"real_minus_{right}"] = {
            "joint_nll_per_event_by_independent_unit": values.tolist(),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "favourable_units": int((values < 0).sum()),
            "n_units": int(len(values)),
            "leave_one_unit_out_mean_min": float(np.min(leave_one_out)),
            "leave_one_unit_out_mean_max": float(np.max(leave_one_out)),
        }
    return {"n_units": int(len(rows)), "comparisons": comparisons}


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--optimizer-root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    parser.add_argument(
        "--support", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long/support/summary.json",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    support = json.loads(args.support.read_text())
    if (
        support.get("status") != "COMPLETE"
        or support.get("revision") != H3_LONG_SUPPORT_REVISION
        or support.get("development_time_contract_verified") is not True
        or support.get("formal_test_partition_opened") is not False
        or support.get("sealed_opened") is not False
    ):
        raise ValueError("invalid frozen H3-long support manifest")
    cells = [
        value for value in support["scheduled_cells"]
        if value["subject"] == args.subject
        and int(value["scale_events"]) == SCALE_EVENTS
        and value["role"] == "full_control"
    ]
    if len(cells) != 1 or cells[0].get("full_causal_control_support") is not True:
        raise ValueError("R1.6 minimal H3 needs one frozen N=1000 full-control cell")
    cell = cells[0]
    context = load_fitted_r1_6_confirmation_t1(
        args.subject, args.seed, device=args.device,
        output_root=args.optimizer_root, require_stable=True,
    )
    innovation, innovation_audit = event_innovation(context, args.source)
    designs, _, design_audit = build_long_arm_designs(
        context,
        innovation,
        source=args.source,
        scale_events=SCALE_EVENTS,
        full_causal_control=True,
        include_horizons=False,
    )
    reference = designs["real_cumulative"]
    n_train = int((reference.split == 0).sum())
    n_validation = int((reference.split == 1).sum())
    if min(n_train, n_validation) < 100:
        raise ValueError(
            f"insufficient minimal H3 support ({n_train} TRAIN, "
            f"{n_validation} validation)"
        )
    if not design_audit["state_matching_estimable"]:
        raise ValueError("state-matched H3 placebo is not estimable")
    required = [
        "state_matched_nonoverlap",
        "current_event_only",
        "chronological_trend",
        "intercept_only",
        "causal_previous_block",
    ]
    if any(label not in designs for label in required):
        raise RuntimeError("minimal H3 omitted a frozen full-control arm")
    estimability = {
        label: affine_estimability_audit(
            context.model, design, device=args.device,
            batch_size=args.batch_size,
        )
        for label, design in designs.items()
    }
    exposure_dim = (
        1 if reference.exposure.ndim == 1 else reference.exposure.shape[1]
    )
    no_edge = AffineExposureEdge(
        reference.current_state.shape[1], exposure_dim
    ).to(args.device).eval()
    edges = {"no_edge": no_edge}
    fits = {"no_edge": {
        "selected_epoch": 0,
        "fixed_intercept_zero": True,
        "fixed_edge_zero": True,
    }}
    for label, design in designs.items():
        edge, fit = fit_affine_edge(
            context.model,
            design,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        edges[label] = edge
        fits[label] = fit
    next_event = {
        label: asdict(evaluate_affine_edge(
            context.model,
            edge,
            reference if label == "no_edge" else designs[label],
            split="validation",
            device=args.device,
            batch_size=args.batch_size,
        ))
        for label, edge in edges.items()
    }
    comparisons = {
        f"real_minus_{right}": contrast(
            next_event["real_cumulative"], next_event[right]
        )
        for right in next_event if right != "real_cumulative"
    }
    independent = block_contrasts(
        context.model,
        edges,
        designs,
        np.asarray(
            design_audit["validation_independent_design_rows"],
            dtype=np.int64,
        ),
        device=args.device,
    )
    classes = {
        label: classify_affine_estimability(estimability[label], fits[label])
        for label in designs
    }
    real_estimable = classes["real_cumulative"] == "ESTIMABLE"
    controls_valid = bool(all(
        estimability[label]["gradient_finite"]
        and estimability[label]["affine_design_rank"]
        == estimability[label]["expected_affine_rank"]
        and min(estimability[label]["exposure_sd"]) > 1e-8
        for label in required if label != "intercept_only"
    ))
    beats = {
        right: comparisons[f"real_minus_{right}"]["joint_nll_per_event"] < 0
        for right in required
    }
    block_beats = {
        right: independent["comparisons"][f"real_minus_{right}"]["median"] < 0
        for right in required
    }
    enough_units = independent["n_units"] >= 3
    primary = bool(
        real_estimable and controls_valid and enough_units
        and all(beats.values()) and all(block_beats.values())
    )
    output = (
        args.optimizer_root / "minimal_h3" / args.subject / args.source
        / f"seed_{args.seed}_n_{SCALE_EVENTS}"
    )
    checkpoint = output / "edges.pt"
    atomic_torch(checkpoint, {
        "revision": R1_6_MINIMAL_H3_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "source": args.source,
        "scale_events": SCALE_EVENTS,
        "edges": {label: edge.state_dict() for label, edge in edges.items()},
    })
    payload = {
        "t1_checkpoint_sha256": context.audit["r1_6_checkpoint_sha256"],
        "real_edge_matrix": edges["real_cumulative"].matrix.detach().cpu().numpy().tolist(),
        "real_edge_intercept": edges["real_cumulative"].intercept.detach().cpu().numpy().tolist(),
        "real_fit_trajectory": fits["real_cumulative"]["trajectory"],
    }
    payload_sha = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    result = {
        "status": "COMPLETE",
        "revision": R1_6_MINIMAL_H3_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "source": args.source,
        "scale_events": SCALE_EVENTS,
        "support_role": "full_control",
        "support_cell": cell,
        "t1": context.audit,
        "innovation": innovation_audit,
        "design": design_audit,
        "estimability": estimability,
        "fits": fits,
        "real_estimability_class": classes["real_cumulative"],
        "arm_estimability_classes": classes,
        "control_numerically_valid": controls_valid,
        "independent_block_analysis": independent,
        "validation": {"next_event": next_event},
        "comparisons": {"next_event": comparisons},
        "real_edge_estimable": real_estimable,
        "enough_independent_validation_units": enough_units,
        "primary_full_control_increment": primary,
        "seed_payload_sha256": payload_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "source_hashes": {
            "optimizer_h3": contract.sha256_file(
                contract.REPO_ROOT
                / "src/topic5_continuous_marked_state_r1/optimizer_h3.py"
            ),
            "runner": contract.sha256_file(Path(__file__)),
            "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
            "support_manifest": contract.sha256_file(args.support),
        },
        "development_time_contract_verified": True,
        "development_validation_used_for_t1_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "development frozen N=1000 next-event H3 diagnostic; no scale "
            "selection, no autonomous rollout, and seed is not a patient"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
