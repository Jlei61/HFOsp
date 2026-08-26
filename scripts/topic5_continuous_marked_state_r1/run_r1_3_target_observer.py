#!/usr/bin/env python3
"""Fit one R1.3 fully target-trained explicit or raw observer arm."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
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
from src.topic5_continuous_marked_state_r1.r1_2b import JointLastLayerStateModel
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    evaluate_mark_endpoints,
    median_metric_dict,
    metric_contrast,
    strict_matched_wrong_time_permutations,
)
from src.topic5_continuous_marked_state_r1.r1_3 import (
    R1_3_REVISION,
    FullAnchorObservationLoader,
    FullTargetObserverStateModel,
    classify_raw_gradient_coverage,
    fit_target_observer,
    initialise_raw_from_explicit,
    materialize_embedding,
    transfer_r1_2b_initialisation,
)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fitted_r1_2b_explicit(subject: str, seed: int, baseline: dict,
                          design, stream, source_observer, *,
                          device: str, root: Path) -> JointLastLayerStateModel:
    model = JointLastLayerStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, source_observer, raw_enabled=False, state_dim=8,
    ).to(device)
    fit_root = root / "joint" / subject / f"joint_explicit_seed_{seed}"
    result_path = fit_root / "result.json"
    checkpoint_path = fit_root / "model.pt"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETE" or result.get("sealed_opened") is not False:
        raise ValueError(f"invalid R1.2b explicit fit: {result_path}")
    if contract.sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("R1.2b explicit checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def update_norms(initial: dict[str, torch.Tensor], model) -> dict[str, float]:
    groups = {
        "raw_tokenizer": "observer.raw.tokenizer",
        "raw_temporal_layer_0": "observer.raw.transformer.layers.0",
        "raw_temporal_layer_1": "observer.raw.transformer.layers.1",
        "raw_projection_and_gate": "observer.raw",
        "spatial_fusion": "observer.spatial",
        "explicit_projection": "observer.explicit",
        "observation_correction": "state.correction",
        "state_readout_timing": "state_timing",
        "state_readout_contact": "state_contact",
        "state_readout_size": "state_size",
        "stable_generator": "state.generator",
    }
    current = model.state_dict()
    result = {}
    for label, prefix in groups.items():
        square = 0.0
        for name, value in current.items():
            if name.startswith(prefix) and name in initial:
                square += float(
                    (value.detach().cpu().float() - initial[name].float()).square().sum()
                )
        result[label] = float(np.sqrt(square))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True,
        choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS,
    )
    parser.add_argument("--arm", required=True, choices=("explicit", "explicit_raw"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-label", default="r1_3_formal_pilot")
    parser.add_argument("--observer-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=2)
    parser.add_argument("--chunk-anchors", type=int, default=8)
    parser.add_argument("--state-learning-rate", type=float, default=3e-4)
    parser.add_argument("--observer-learning-rate", type=float, default=3e-5)
    parser.add_argument("--raw-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--initialisation-source",
        choices=("prefer_r1_2b_same_seed", "r1_2_matching_seed"),
        default="prefer_r1_2b_same_seed",
        help=(
            "R1.4 uses r1_2_matching_seed for every subject so the six-patient "
            "replication does not give the original three an extra R1.2b stage."
        ),
    )
    parser.add_argument(
        "--matched-wrong-donors", type=int, choices=(5, 10), default=5,
    )
    parser.add_argument(
        "--r1-2-fallback-seed-mode",
        choices=("common_seed_0", "matching_seed"),
        default="common_seed_0",
        help=(
            "When no same-seed R1.2b checkpoint exists, preserve the original "
            "R1.3 common seed-0 initialisation or use the matching R1.2 seed "
            "for a genuine multi-start development triage."
        ),
    )
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--r1-2b-root", type=Path, default=contract.RESULT_ROOT / "r1_2b"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_3"
    )
    parser.add_argument(
        "--observation-cache-root", type=Path,
        default=contract.RESULT_ROOT / "r1_3" / "cache",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observation_cache_manifest_path = (
        args.observation_cache_root / args.subject / "manifest.json"
    )
    if not observation_cache_manifest_path.exists():
        raise FileNotFoundError(
            f"build the frozen R1.3 observation cache first: "
            f"{observation_cache_manifest_path}"
        )
    observation_cache = json.loads(observation_cache_manifest_path.read_text())
    if observation_cache.get("status") != "COMPLETE":
        raise ValueError("R1.3 observation cache is incomplete")
    if observation_cache.get("sealed_opened") is not False:
        raise ValueError("R1.3 observation cache opened the sealed partition")
    design_path = Path(observation_cache["design"])
    if contract.sha256_file(design_path) != observation_cache["design_sha256"]:
        raise ValueError("R1.3 observation-cache design hash mismatch")
    design = load_full_design(design_path)
    baseline_path = args.r1_2_root / "baselines" / args.subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage = CoverageTable.load(
        args.r1_2_root / "coverage" / f"{args.subject}.npz"
    )
    stream = load_full_admissible_event_stream(args.subject, coverage)
    bridge_result_path = (
        args.r1_2_root / "bridge_e1" / args.subject / "seed_0/result.json"
    )
    bridge_result = json.loads(bridge_result_path.read_text())
    _, _, sampled, _ = _bridge_scaler(
        args.subject, baseline_path, bridge_result, stream, coverage
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
    r1_2b_path = (
        args.r1_2b_root / "joint" / args.subject
        / f"joint_explicit_seed_{args.seed}" / "model.pt"
    )
    use_r1_2b = (
        args.initialisation_source == "prefer_r1_2b_same_seed"
        and r1_2b_path.exists()
    )
    if use_r1_2b:
        r1_2b_explicit = fitted_r1_2b_explicit(
            args.subject, args.seed, baseline, design, stream, raw_bridge.observer,
            device=args.device, root=args.r1_2b_root,
        )
        initialisation = {
            "kind": "r1_2b_joint_explicit_same_seed",
            "checkpoint": str(r1_2b_path),
            "checkpoint_sha256": contract.sha256_file(r1_2b_path),
        }
    else:
        r1_2b_explicit = None
        fallback_seed = args.seed if (
            args.initialisation_source == "r1_2_matching_seed"
            or args.r1_2_fallback_seed_mode == "matching_seed"
        ) else 0
        r1_2_path = (
            args.r1_2_root / "t1_full" / args.subject
            / f"explicit_d8_seed_{fallback_seed}" / "model.pt"
        )
        r1_2_result_path = r1_2_path.with_name("result.json")
        r1_2_result = json.loads(r1_2_result_path.read_text())
        if (r1_2_result.get("status") != "COMPLETE"
                or r1_2_result.get("sealed_opened") is not False):
            raise ValueError(f"invalid R1.2 T1 initialisation: {r1_2_result_path}")
        if contract.sha256_file(r1_2_path) != r1_2_result["checkpoint_sha256"]:
            raise ValueError("R1.2 T1 initialisation hash mismatch")
        initialisation = {
            "kind": (
                "r1_2_explicit_matching_seed_then_target_alignment"
                if fallback_seed == args.seed
                else "r1_2_explicit_seed_0_common_core_then_target_alignment"
            ),
            "checkpoint": str(r1_2_path),
            "checkpoint_sha256": contract.sha256_file(r1_2_path),
            "fallback_seed": int(fallback_seed),
            "shared_initialisation_across_r1_3_seeds": bool(
                fallback_seed == 0 and args.seed != 0
            ),
        }

    def build_model() -> FullTargetObserverStateModel:
        model = FullTargetObserverStateModel(
            baseline, design.event_history.shape[1], stream.n_contacts,
            stream.adjacency, raw_bridge.observer,
            use_raw=args.arm == "explicit_raw", state_dim=8,
        ).to(args.device)
        if r1_2b_explicit is not None:
            transfer_r1_2b_initialisation(model, r1_2b_explicit)
        else:
            checkpoint = torch.load(
                initialisation["checkpoint"], map_location=args.device,
                weights_only=False,
            )
            missing, unexpected = model.load_state_dict(
                checkpoint["model"], strict=False
            )
            invalid_missing = [
                name for name in missing if not name.startswith("observer.")
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "R1.2 common-core transfer mismatch: "
                    f"missing={invalid_missing}, unexpected={unexpected}"
                )
        if args.arm == "explicit_raw":
            paired_path = (
                args.output_root / "fits" / args.subject
                / f"explicit_seed_{args.seed}" / "model.pt"
            )
            paired_result_path = paired_path.with_name("result.json")
            if not paired_path.exists() or not paired_result_path.exists():
                raise FileNotFoundError(
                    f"raw arm requires paired completed explicit arm: {paired_path}"
                )
            paired_result = json.loads(paired_result_path.read_text())
            if paired_result.get("status") != "COMPLETE":
                raise ValueError("paired R1.3 explicit arm is incomplete")
            if contract.sha256_file(paired_path) != paired_result["checkpoint_sha256"]:
                raise ValueError("paired R1.3 explicit checkpoint hash mismatch")
            paired = torch.load(paired_path, map_location=args.device, weights_only=False)
            initialise_raw_from_explicit(model, paired["model"], raw_gain=0.02)
        return model

    explicit_cache_path = Path(observation_cache["explicit"])
    contact_mask_cache_path = Path(observation_cache["contact_mask"])
    if contract.sha256_file(explicit_cache_path) != observation_cache["explicit_sha256"]:
        raise ValueError("R1.3 explicit cache hash mismatch")
    if contract.sha256_file(contact_mask_cache_path) != observation_cache[
        "contact_mask_sha256"
    ]:
        raise ValueError("R1.3 contact-mask cache hash mismatch")
    cached_explicit = np.load(explicit_cache_path, mmap_mode="r")
    cached_contact_mask = np.load(contact_mask_cache_path, mmap_mode="r")
    loader = FullAnchorObservationLoader(
        args.subject, design, stream.event_time,
        sampled.explicit_mean, sampled.explicit_scale,
        cached_explicit=cached_explicit,
        cached_contact_mask=cached_contact_mask,
    )
    chunk = int(args.chunk_anchors)
    oom_retries = []
    while True:
        model = build_model()
        initial_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        try:
            trace = fit_target_observer(
                model, design, loader, device=args.device,
                observer_epochs=args.observer_epochs,
                joint_epochs=args.joint_epochs,
                state_lr=args.state_learning_rate,
                observer_lr=args.observer_learning_rate,
                raw_lr=args.raw_learning_rate,
                chunk_anchors=chunk,
            )
            break
        except torch.OutOfMemoryError as error:
            oom_retries.append({"chunk_anchors": chunk, "error": str(error)})
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()

    embedding = materialize_embedding(
        model, design, loader, device=args.device, batch_size=chunk
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
    observation_coverage = np.asarray(cached_contact_mask, dtype=np.float64).mean(1)
    anchor_segment = None
    if args.experiment_label == "r1_4_six_patient_explicit_primary_raw_residual_v1":
        anchor_segment = np.searchsorted(
            coverage.stop, np.asarray(design.anchor_time, dtype=np.float64), side="right"
        )
        if np.any(anchor_segment >= len(coverage.start)):
            raise ValueError("R1.4 anchor occurs after the final recorded segment")
        anchor_inside = (
            (design.anchor_time >= coverage.start[anchor_segment])
            & (design.anchor_time < coverage.stop[anchor_segment])
        )
        if not bool(np.all(anchor_inside)):
            raise ValueError("R1.4 anchor occurs outside recorded coverage")
    permutations, matched, match_audit = strict_matched_wrong_time_permutations(
        design, observation_coverage, anchor_segment=anchor_segment,
        n_donors=int(args.matched_wrong_donors),
        min_separation_seconds=1800.0,
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

    parameter_updates = update_norms(initial_state, model)
    raw_analysis_status = None
    raw_non_estimable_reason = None
    if args.arm == "explicit_raw":
        raw_gradients = [
            trace.selection_gradient_max.get("raw_tokenizer", 0.0),
            trace.selection_gradient_max.get("raw_temporal_layer_0", 0.0),
            trace.selection_gradient_max.get("raw_temporal_layer_1", 0.0),
        ]
        try:
            raw_analysis_status, raw_non_estimable_reason = (
                classify_raw_gradient_coverage(raw_gradients)
            )
        except ValueError as error:
            raise RuntimeError(
                f"raw target-gradient coverage failed: {raw_gradients}"
            ) from error
        # A paired explicit checkpoint can leave the complete downstream raw
        # path at an exact structural zero (for example when the explicit T1
        # itself selected epoch zero).  That is not evidence against raw
        # waveform information and must not abort H1/H2a.  Persist the paired
        # no-update result, but exclude it from the raw favourable denominator.
        common = {
            key: parameter_updates[key] for key in (
                "spatial_fusion", "observation_correction",
                "state_readout_timing", "state_readout_contact",
                "state_readout_size", "stable_generator",
                "explicit_projection",
            )
        }
        if any(value > 1e-10 for value in common.values()):
            raise RuntimeError(
                f"paired raw arm changed a common explicit/T1 parameter: {common}"
            )

    output = args.output_root / "fits" / args.subject / f"{args.arm}_seed_{args.seed}"
    checkpoint_path = output / "model.pt"
    atomic_torch(checkpoint_path, {
        "contract": contract.REVISION,
        "r1_3_revision": R1_3_REVISION,
        "experiment_label": args.experiment_label,
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "model": model.state_dict(),
        "fit_trace": asdict(trace),
    })
    result = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_3_revision": R1_3_REVISION,
        "experiment_label": args.experiment_label,
        "subject": args.subject,
        "arm": args.arm,
        "seed": int(args.seed),
        "full_raw_temporal_target_trained": (
            args.arm == "explicit_raw" and raw_analysis_status == "ESTIMATED"
        ),
        "raw_analysis_status": raw_analysis_status,
        "raw_non_estimable_reason": raw_non_estimable_reason,
        "raw_patch_tokenizer_target_gradient": (
            trace.selection_gradient_max.get("raw_tokenizer", 0.0)
        ),
        "raw_temporal_layer_target_gradients": [
            trace.selection_gradient_max.get("raw_temporal_layer_0", 0.0),
            trace.selection_gradient_max.get("raw_temporal_layer_1", 0.0),
        ],
        "fit_trace": asdict(trace),
        "effective_chunk_anchors": int(chunk),
        "oom_retries": oom_retries,
        "parameter_update_norm": parameter_updates,
        "paired_raw_common_parameter_update_exact_zero": (
            args.arm != "explicit_raw" or all(
                parameter_updates[key] <= 1e-10 for key in (
                    "spatial_fusion", "observation_correction",
                    "state_readout_timing", "state_readout_contact",
                    "state_readout_size", "stable_generator",
                    "explicit_projection",
                )
            )
        ),
        "validation": {
            "persistent": persistent,
            "memoryless": memoryless,
            "persistent_minus_memoryless": metric_contrast(
                persistent, memoryless
            ),
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
                "wrong_median": wrong_median,
                "correct_minus_wrong_median": metric_contrast(
                    matched_correct, wrong_median
                ),
                "endpoint_correct": matched_correct_endpoint,
                "endpoint_wrong_median": wrong_endpoint_median,
                "endpoint_correct_minus_wrong_median": metric_contrast(
                    matched_correct_endpoint, wrong_endpoint_median
                ),
            },
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "initialisation": initialisation,
        "initialisation_source_policy": args.initialisation_source,
        "matched_wrong_donors": int(args.matched_wrong_donors),
        "observation_cache_manifest": str(observation_cache_manifest_path),
        "observation_cache_manifest_sha256": contract.sha256_file(
            observation_cache_manifest_path
        ),
        "full_recorded_support": True,
        "sealed_opened": False,
        "claim_boundary": (
            "development R1.3 target-trained observer; "
            "not a cohort, seizure, H3 or autonomous-state result"
        ),
    }
    if args.arm == "explicit_raw":
        paired_result_path = (
            args.output_root / "fits" / args.subject
            / f"explicit_seed_{args.seed}" / "result.json"
        )
        paired_result = json.loads(paired_result_path.read_text())
        result["paired_raw_minus_explicit"] = metric_contrast(
            persistent, paired_result["validation"]["persistent"]
        )
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
