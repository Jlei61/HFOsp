#!/usr/bin/env python3
"""Fit one R1.5 seed/source across its frozen supported H3-long scales."""
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
    H3_LONG_REVISION,
    H3_LONG_SUPPORT_REVISION,
    SOURCES,
    AffineExposureEdge,
    affine_estimability_audit,
    classify_affine_estimability,
    evaluate_affine_edge,
    fit_affine_edge,
)
from src.topic5_continuous_marked_state_r1.h3_long_human import (
    R1_5_REVISION,
    build_long_arm_designs,
    cell_package_fingerprint,
    event_innovation,
    load_fitted_r1_5_explicit_t1,
)
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    classify_one_shot_persistence,
    evaluate_horizon_mark,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import _evaluate_rows


SEEDS = (0, 1, 2, 3, 4)


def contrast(left: dict, right: dict) -> dict:
    return {
        key: float(value - right[key])
        for key, value in left.items()
        if isinstance(value, (int, float)) and not key.startswith("n_")
    }


def complete(path: Path, expected_fingerprint: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == H3_LONG_REVISION
        and value.get("sealed_opened") is False
        and value.get("formal_test_partition_opened") is False
        and (
            expected_fingerprint is None
            or value.get("package_fingerprint") == expected_fingerprint
        )
    )


def block_contrasts(model, edges: dict, designs: dict,
                    rows: np.ndarray, *, device: str) -> dict:
    """Score one non-overlapping endpoint per exposure unit."""
    rows = np.asarray(rows, dtype=np.int64)
    metrics = {}
    for label, edge in edges.items():
        design = designs["real_cumulative"] if label == "no_edge" else designs[label]
        metrics[label] = [
            asdict(_evaluate_rows(
                model, edge, design, np.asarray([row]),
                device=device, batch_size=1,
            )) for row in rows
        ]
    comparisons = {}
    for right in metrics:
        if right == "real_cumulative":
            continue
        differences = np.asarray([
            left["joint_nll_per_event"] - other["joint_nll_per_event"]
            for left, other in zip(metrics["real_cumulative"], metrics[right])
        ], dtype=np.float64)
        if len(differences) > 1:
            leave_one_out = (
                differences.sum() - differences
            ) / (len(differences) - 1)
        else:
            leave_one_out = differences.copy()
        comparisons[f"real_minus_{right}"] = {
            "joint_nll_per_event_by_independent_unit": differences.tolist(),
            "median": float(np.median(differences)),
            "mean": float(np.mean(differences)),
            "favourable_units": int((differences < 0).sum()),
            "n_units": int(len(differences)),
            "leave_one_unit_out_mean_min": float(np.min(leave_one_out)),
            "leave_one_unit_out_mean_max": float(np.max(leave_one_out)),
        }
    return {"n_units": int(len(rows)), "comparisons": comparisons}


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def persist_not_estimable(
    output: Path,
    *,
    subject: str,
    seed: int,
    source: str,
    scale: int,
    role: str,
    reason: str,
    t1: dict,
    package_fingerprint: str,
    package_components: dict,
) -> None:
    contract.atomic_json(output / "result.json", {
        "status": "COMPLETE", "analysis_status": "NOT_ESTIMABLE",
        "non_estimable_reason": reason, "revision": H3_LONG_REVISION,
        "r1_5_revision": R1_5_REVISION, "subject": subject,
        "seed": int(seed), "source": source, "scale_events": int(scale),
        "support_role": role, "t1": t1, "real_edge_estimable": False,
        "real_estimability_class": "SUPPORT_NOT_ESTIMABLE",
        "package_fingerprint": package_fingerprint,
        "package_components": package_components,
        "primary_full_control_increment": False,
        "supportive_boundary_increment": False,
        "development_time_contract_verified": bool(
            t1.get("development_time_contract_verified", False)
        ),
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": "support or estimability limitation; not a negative",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True, choices=contract.R1_5_EXTENSION_SUBJECTS
    )
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--r1-5-root", type=Path, default=contract.RESULT_ROOT / "r1_5"
    )
    parser.add_argument(
        "--support", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long/support/summary.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    support = json.loads(args.support.read_text())
    if (
        support.get("status") != "COMPLETE"
        or support.get("revision") != H3_LONG_SUPPORT_REVISION
        or support.get("sealed_opened") is not False
        or support.get("formal_test_partition_opened") is not False
        or support.get("development_time_contract_verified") is not True
    ):
        raise ValueError("invalid frozen H3-long support manifest")
    cells = [
        value for value in support["scheduled_cells"]
        if value["subject"] == args.subject
    ]
    if not cells:
        raise ValueError(f"{args.subject}: no frozen H3-long support cells")
    r1_summary = json.loads((
        args.r1_5_root / "reports/r1_5_summary.json"
    ).read_text())
    stable_t1_patient = bool(
        r1_summary["by_subject"][args.subject]["stable_explicit_t1_for_h3"]
    )
    context = load_fitted_r1_5_explicit_t1(
        args.subject, args.seed, device=args.device, r1_5_root=args.r1_5_root
    )
    stable_t1_seed = bool(context.audit["seed_stable_t1"])
    innovation, innovation_audit = event_innovation(context, args.source)
    summaries = []
    for cell in cells:
        scale = int(cell["scale_events"])
        role = str(cell["role"])
        output = (
            args.output_root / "human" / args.subject / args.source
            / f"seed_{args.seed}_n_{scale}"
        )
        package_fingerprint, package_components = cell_package_fingerprint(
            args.subject, args.seed, args.source, scale, role,
            support_path=args.support, r1_5_root=args.r1_5_root,
            runner_path=Path(__file__),
        )
        if complete(output / "result.json", package_fingerprint):
            summaries.append({"scale": scale, "skipped": True})
            continue
        try:
            one_step, horizons, design_audit = build_long_arm_designs(
                context, innovation, source=args.source, scale_events=scale,
                full_causal_control=bool(cell["full_causal_control_support"]),
                include_horizons=stable_t1_seed,
            )
        except ValueError as error:
            persist_not_estimable(
                output, subject=args.subject, seed=args.seed,
                source=args.source, scale=scale, role=role,
                reason=str(error), t1=context.audit,
                package_fingerprint=package_fingerprint,
                package_components=package_components,
            )
            summaries.append({"scale": scale, "not_estimable": str(error)})
            continue
        reference = one_step["real_cumulative"]
        n_train = int((reference.split == 0).sum())
        n_validation = int((reference.split == 1).sum())
        if min(n_train, n_validation) < 100:
            reason = f"insufficient common support ({n_train} TRAIN, {n_validation} validation)"
            persist_not_estimable(
                output, subject=args.subject, seed=args.seed,
                source=args.source, scale=scale, role=role,
                reason=reason, t1=context.audit,
                package_fingerprint=package_fingerprint,
                package_components=package_components,
            )
            summaries.append({"scale": scale, "not_estimable": reason})
            continue
        if role == "full_control" and "causal_previous_block" not in one_step:
            raise RuntimeError("full-control manifest cell omitted causal arm")
        if role != "full_control" and "causal_previous_block" in one_step:
            raise RuntimeError("boundary manifest cell unexpectedly has causal arm")
        if not design_audit["state_matching_estimable"]:
            persist_not_estimable(
                output, subject=args.subject, seed=args.seed,
                source=args.source, scale=scale, role=role,
                reason="state-matched placebo collapsed on final common support",
                t1=context.audit, package_fingerprint=package_fingerprint,
                package_components=package_components,
            )
            summaries.append({"scale": scale, "not_estimable": "matching"})
            continue
        trainable = tuple(one_step)
        estimability = {
            label: affine_estimability_audit(
                context.model, design, device=args.device,
                batch_size=args.batch_size,
            ) for label, design in one_step.items()
        }
        exposure_dim = (
            1 if reference.exposure.ndim == 1 else reference.exposure.shape[1]
        )
        no_edge = AffineExposureEdge(
            reference.current_state.shape[1], exposure_dim
        ).to(args.device).eval()
        edges = {"no_edge": no_edge}
        fits = {"no_edge": {
            "selected_epoch": 0, "fixed_intercept_zero": True,
            "fixed_edge_zero": True, "edge_left_zero_initialisation": False,
        }}
        for label in trainable:
            edge, fit = fit_affine_edge(
                context.model, one_step[label], device=args.device,
                seed=args.seed, epochs=args.epochs,
                learning_rate=args.learning_rate, batch_size=args.batch_size,
            )
            edges[label] = edge; fits[label] = fit
        next_event = {
            label: asdict(evaluate_affine_edge(
                context.model, edge,
                reference if label == "no_edge" else one_step[label],
                split="validation", device=args.device,
                batch_size=args.batch_size,
            )) for label, edge in edges.items()
        }
        comparisons = {
            "next_event": {
                f"real_minus_{right}": contrast(
                    next_event["real_cumulative"], next_event[right]
                ) for right in next_event if right != "real_cumulative"
            }
        }
        independent_blocks = block_contrasts(
            context.model, edges, one_step,
            np.asarray(
                design_audit["validation_independent_design_rows"],
                dtype=np.int64,
            ),
            device=args.device,
        )
        horizon_metrics = {}
        if stable_t1_seed:
            for horizon in (5, 10):
                horizon_metrics[f"H{horizon}"] = {
                    label: asdict(evaluate_horizon_mark(
                        context.model, edge,
                        horizons[horizon]["real_cumulative"]
                        if label == "no_edge" else horizons[horizon][label],
                        split="validation", device=args.device,
                        batch_size=args.batch_size,
                    )) for label, edge in edges.items()
                }
                comparisons[f"H{horizon}"] = {
                    f"real_minus_{right}": contrast(
                        horizon_metrics[f"H{horizon}"]["real_cumulative"],
                        horizon_metrics[f"H{horizon}"][right],
                    ) for right in horizon_metrics[f"H{horizon}"]
                    if right != "real_cumulative"
                }
        real_audit = estimability["real_cumulative"]
        real_fit = fits["real_cumulative"]
        estimability_class = classify_affine_estimability(real_audit, real_fit)
        estimable = estimability_class == "ESTIMABLE"
        estimability_classes = {
            label: classify_affine_estimability(
                estimability[label], fits[label]
            ) for label in one_step
        }
        required = [
            "state_matched_nonoverlap", "current_event_only",
            "chronological_trend", "intercept_only",
        ]
        full_required = required + (["causal_previous_block"] if (
            "causal_previous_block" in next_event
        ) else [])
        control_numerically_valid = bool(all(
            estimability[label]["gradient_finite"]
            and estimability[label]["affine_design_rank"]
            == estimability[label]["expected_affine_rank"]
            and min(estimability[label]["exposure_sd"]) > 1e-8
            for label in full_required if label != "intercept_only"
        ))
        beats = {
            right: bool(comparisons["next_event"][
                f"real_minus_{right}"
            ]["joint_nll_per_event"] < 0)
            for right in full_required
        }
        block_beats = {
            right: bool(independent_blocks["comparisons"][
                f"real_minus_{right}"
            ]["median"] < 0)
            for right in full_required
        }
        enough_independent_units = bool(
            independent_blocks["n_units"] >= 3
        )
        primary = bool(
            estimable and role == "full_control" and enough_independent_units
            and control_numerically_valid
            and all(beats.values()) and all(block_beats.values())
        )
        supportive = bool(
            estimable and enough_independent_units
            and control_numerically_valid
            and all(beats[right] for right in required)
            and all(block_beats[right] for right in required)
        )
        persistence = {}
        if stable_t1_seed:
            for horizon in (5, 10):
                right = [
                    "state_matched_nonoverlap", "current_event_only",
                    "chronological_trend", "intercept_only",
                ]
                if "causal_previous_block" in horizon_metrics[f"H{horizon}"]:
                    right.append("causal_previous_block")
                per_comparator = {
                    label: classify_one_shot_persistence(
                        comparisons[f"H{horizon}"][f"real_minus_{label}"],
                        horizon_metrics[f"H{horizon}"]["real_cumulative"],
                        real_edge_estimable=estimable,
                    ) for label in right
                }
                persistence[f"H{horizon}"] = {
                    "state_and_mark_persist": bool(
                        all(value["state_and_mark_persist"]
                            for value in per_comparator.values())
                    ),
                    "comparators": right,
                    "per_comparator": per_comparator,
                    "future_event_history_teacher_forced": True,
                    "autonomous_rollout": False,
                }
        checkpoint = output / "edges.pt"
        atomic_torch(checkpoint, {
            "revision": H3_LONG_REVISION, "subject": args.subject,
            "seed": int(args.seed), "source": args.source,
            "scale_events": scale,
            "edges": {label: edge.state_dict() for label, edge in edges.items()},
        })
        seed_payload = {
            "t1_checkpoint_sha256": context.audit["r1_3_checkpoint_sha256"],
            "real_edge_matrix": edges["real_cumulative"].matrix.detach().cpu().numpy().tolist(),
            "real_edge_intercept": edges["real_cumulative"].intercept.detach().cpu().numpy().tolist(),
            "real_fit_trajectory": real_fit["trajectory"],
        }
        seed_payload_sha256 = hashlib.sha256(
            json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result = {
            "status": "COMPLETE", "analysis_status": "ESTIMATED",
            "revision": H3_LONG_REVISION, "r1_5_revision": R1_5_REVISION,
            "subject": args.subject, "seed": int(args.seed),
            "source": args.source, "scale_events": scale,
            "support_role": role, "support_cell": cell,
            "t1": context.audit, "stable_t1_patient": stable_t1_patient,
            "stable_t1_seed": stable_t1_seed,
            "innovation": innovation_audit, "design": design_audit,
            "estimability": estimability, "fits": fits,
            "real_estimability_class": estimability_class,
            "arm_estimability_classes": estimability_classes,
            "control_numerically_valid": control_numerically_valid,
            "independent_block_analysis": independent_blocks,
            "edge_matrices": {
                label: edge.matrix.detach().cpu().numpy().tolist()
                for label, edge in edges.items()
            },
            "edge_intercepts": {
                label: edge.intercept.detach().cpu().numpy().tolist()
                for label, edge in edges.items()
            },
            "validation": {
                "next_event": next_event, "horizons": horizon_metrics,
            },
            "comparisons": comparisons,
            "real_edge_estimable": estimable,
            "primary_full_control_increment": primary,
            "supportive_boundary_increment": supportive,
            "one_shot_persistence": persistence,
            "seed_payload_sha256": seed_payload_sha256,
            "package_fingerprint": package_fingerprint,
            "package_components": package_components,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": contract.sha256_file(checkpoint),
            "source_hashes": {
                "h3_long": contract.sha256_file(
                    contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long.py"
                ),
                "h3_long_human": contract.sha256_file(
                    contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long_human.py"
                ),
                "runner": contract.sha256_file(Path(__file__)),
                "split_manifest": contract.sha256_file(contract.SPLIT_MANIFEST),
                "support_manifest": contract.sha256_file(args.support),
            },
            "development_time_contract_verified": True,
            "formal_test_partition_opened": False, "sealed_opened": False,
            "claim_boundary": (
                "development exact-N antecedent screen with an exposure-"
                "conditioned latent correction; H5/H10 are teacher-forced "
                "one-shot persistence diagnostics, not an autonomous generator"
            ),
        }
        contract.atomic_json(output / "result.json", result)
        summaries.append({
            "scale": scale, "role": role, "estimable": estimable,
            "primary": primary, "supportive": supportive,
        })
    print(json.dumps({
        "status": "COMPLETE", "subject": args.subject, "seed": args.seed,
        "source": args.source, "stable_t1_patient": stable_t1_patient,
        "stable_t1_seed": stable_t1_seed,
        "cells": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
