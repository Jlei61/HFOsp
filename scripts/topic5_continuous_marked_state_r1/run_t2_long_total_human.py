#!/usr/bin/env python3
"""Run one development-only very-long total-effect arm set from a frozen T1."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_human import (
    load_fitted_explicit_t1,
    load_fitted_r1_2_explicit_t1,
)
from src.topic5_continuous_marked_state_r1.t2_s1 import (
    SignedExposureEdge,
    _evaluate_rows,
    build_one_step_design,
    fit_load_innovation,
    fit_participation_innovation,
)
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    LONG_TOTAL_REVISION,
    boxcar_memory_audit,
    build_long_window_design,
    count_windows_crossing_segment,
    decoder_readout,
    delayed_control_overlap,
    delayed_union_start_index,
    effective_memory_audit,
    endpoint_support_audit,
    estimability_guard,
    fit_decoder_space_edge,
    intercept_operator,
    metric_contrast,
    nonoverlapping_window_audit,
    occurrence_block_variation,
    predict_state,
    state_prediction_metrics,
    target_shift_audit,
)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _next_event_metrics(context, endpoint: np.ndarray, predicted: np.ndarray, *,
                        device: torch.device | str) -> tuple[dict, np.ndarray]:
    eligible = np.zeros(len(context.design.event_time), dtype=bool)
    eligible[np.asarray(endpoint, dtype=np.int64)] = True
    state = np.array(context.pre_event_state, copy=True)
    state[np.asarray(endpoint, dtype=np.int64)] = np.asarray(predicted, dtype=np.float32)
    one_step = build_one_step_design(
        context.design, state, context.event_segment,
        np.zeros(len(state), dtype=np.float32), eligible,
    )
    rows = np.flatnonzero(one_step.split == 1)
    edge = SignedExposureEdge(state.shape[1]).to(device).eval()
    for parameter in edge.parameters():
        parameter.requires_grad_(False)
    metrics = asdict(_evaluate_rows(
        context.model, edge, one_step, rows, device=device,
    ))
    return metrics, one_step.current_index[rows]


def _block_summary(metrics_fn, predicted: np.ndarray, target: np.ndarray,
                   validation_rows: np.ndarray, endpoint_time: np.ndarray,
                   train_end: float, readout) -> dict:
    block = np.floor((endpoint_time - float(train_end)) / 1800.0).astype(np.int64)
    values = []
    sizes = []
    for label in np.unique(block[validation_rows]):
        rows = validation_rows[block[validation_rows] == label]
        if not len(rows):
            continue
        value = metrics_fn(predicted, target, rows, readout)
        values.append(value["decoder_total_equal_block_mse"])
        sizes.append(int(len(rows)))
    return {
        "n_half_hour_blocks": int(len(values)),
        "windows_per_block": sizes,
        "median_decoder_total_equal_block_mse": (
            float(np.median(values)) if values else None
        ),
        "block_scores": [float(value) for value in values],
        "inference_warning": (
            "windows overlap heavily; blocks are a dependence sensitivity, not "
            "independent patient replicates or a p-value"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True,
        choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS,
    )
    parser.add_argument("--seed", type=int, required=True, choices=tuple(range(7)))
    parser.add_argument(
        "--window", required=True,
        choices=(
            "event_count_1000", "event_count_2000", "event_count_3000",
            "event_count_4000", "event_count_5000", "event_count_10000",
            "event_count_15000", "event_count_20000", "physical_6h",
        ),
    )
    parser.add_argument("--t1-source", choices=("r1_3", "r1_2"), default="r1_3")
    parser.add_argument(
        "--exposure-memory", choices=("generator_weighted", "boxcar"),
        default="generator_weighted",
    )
    parser.add_argument(
        "--exposure-source", choices=("load", "repertoire"), default="load",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--t1-root", type=Path, default=None,
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "t2_long_total_effect/human",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scale_events = (
        int(args.window.rsplit("_", 1)[1])
        if args.window.startswith("event_count_") else 10000
    )
    if args.t1_source == "r1_3":
        t1_root = args.t1_root or (
            contract.RESULT_ROOT / "t2_long_total_effect/t1_r1_3"
        )
        context = load_fitted_explicit_t1(
            args.subject, args.seed, device=args.device,
            r1_3_root=t1_root,
            observation_cache_root=t1_root / "cache",
            embedding_batch_size=128,
        )
        t1_result_path = (
            t1_root / "fits" / args.subject / f"explicit_seed_{args.seed}"
            / "result.json"
        )
    else:
        t1_root = args.t1_root or (contract.RESULT_ROOT / "r1_2")
        context = load_fitted_r1_2_explicit_t1(
            args.subject, args.seed, device=args.device, r1_2_root=t1_root
        )
        t1_result_path = (
            t1_root / "t1_full" / args.subject
            / f"explicit_d8_seed_{args.seed}" / "result.json"
        )
    train_event = context.design.event_split == 0
    if args.exposure_source == "load":
        innovation, innovation_audit = fit_load_innovation(
            context.pre_event_state, context.design.event_history,
            np.asarray(context.stream.load, dtype=np.float32), train_event,
        )
        exposure_channels = [
            "IED occurrence", "TRAIN-residualised load innovation",
        ]
        real_name = "real_occurrence_plus_load"
        delayed_name = "causal_delayed_load_1000"
    else:
        innovation, innovation_audit = fit_participation_innovation(
            context.pre_event_state, context.design.event_history,
            np.asarray(context.stream.participation, dtype=np.float32),
            train_event, n_components=2,
        )
        exposure_channels = [
            "IED occurrence",
            *[
                f"TRAIN-residualised participation-composition PC{index + 1}"
                for index in range(innovation.shape[1])
            ],
        ]
        real_name = "real_occurrence_plus_repertoire"
        delayed_name = "causal_delayed_repertoire_1000"
    matrix = context.model.state.generator.matrix().detach().cpu().numpy()
    mu = context.model.state.generator.mu.detach().cpu().numpy()
    windows = build_long_window_design(
        context.design.event_time, context.design.event_split,
        context.event_segment, context.pre_event_state, innovation, matrix, mu,
        window_kind=args.window, scale_events=scale_events, duration_hours=6.0,
        delay_events=1000, coverage_start=context.coverage.start,
        exposure_memory=args.exposure_memory,
    )
    for split_name, code in (("train", 0), ("validation", 1)):
        endpoint_time = context.design.event_time[windows.end_index[windows.split == code]]
        contract.assert_development_times(args.subject, endpoint_time, split_name)
    target_delta = windows.target_state - windows.natural_state
    readout = decoder_readout(
        context.model, target_delta, windows.split == 0
    )
    theta_real, real_fit = fit_decoder_space_edge(
        windows.real_operator, target_delta, windows.split, readout
    )
    theta_delayed, delayed_fit = fit_decoder_space_edge(
        windows.delayed_operator, target_delta, windows.split, readout
    )
    offset = intercept_operator(windows)
    theta_intercept, intercept_fit = fit_decoder_space_edge(
        offset, target_delta, windows.split, readout
    )
    predicted = {
        "no_edge_natural_flow": windows.natural_state,
        "no_edge_plus_fitted_intercept": predict_state(
            windows, offset, theta_intercept
        ),
        real_name: predict_state(
            windows, windows.real_operator, theta_real
        ),
        delayed_name: predict_state(
            windows, windows.delayed_operator, theta_delayed
        ),
    }
    validation = np.flatnonzero(windows.split == 1)
    validation_metrics = {
        name: state_prediction_metrics(
            value, windows.target_state, validation, readout
        ) for name, value in predicted.items()
    }
    train_end, _ = contract.load_split(args.subject)
    endpoint_time = context.design.event_time[windows.end_index]
    block_sensitivity = {
        name: _block_summary(
            state_prediction_metrics, value, windows.target_state,
            validation, endpoint_time, train_end, readout,
        ) for name, value in predicted.items()
    }
    next_event = {}
    next_support = None
    for name, value in predicted.items():
        metric, support = _next_event_metrics(
            context, windows.end_index[validation], value[validation],
            device=args.device,
        )
        if next_support is None:
            next_support = support
        elif not np.array_equal(next_support, support):
            raise RuntimeError("next-event support changed across long arms")
        next_event[name] = metric
    real = validation_metrics[real_name]
    no_edge = validation_metrics["no_edge_natural_flow"]
    intercept = validation_metrics["no_edge_plus_fitted_intercept"]
    delayed = validation_metrics[delayed_name]
    output = args.output_root / args.subject / args.window
    if args.exposure_source != "load":
        output = output / args.exposure_source
    output = output / f"seed_{args.seed}"
    parameter_path = output / "parameters_and_support.npz"
    atomic_npz(
        parameter_path, theta_real=theta_real, theta_delayed=theta_delayed,
        theta_intercept=theta_intercept,
        exposure_feature_names=np.asarray(exposure_channels, dtype="U96"),
        start_index=windows.start_index, end_index=windows.end_index,
        split=windows.split, duration_hours=windows.duration_hours,
        n_events=windows.n_events,
    )
    t1_result = json.loads(t1_result_path.read_text())
    selected_t1_epoch = int(
        t1_result["fit_trace"]["selected_total_epoch"]
        if args.t1_source == "r1_3" else t1_result["selected_epochs"]
    )
    if args.t1_source == "r1_3":
        persistent_increment = t1_result["validation"][
            "persistent_minus_memoryless"
        ]["joint_nll_per_event"]
        time_specific_increment = t1_result["validation"][
            "strict_matched_wrong_time"
        ]["correct_minus_wrong_median"]["joint_nll_per_event"]
        # R1.3 has no filtered-vs-no-state arm at all, so the "current
        # observation is useful" condition is simply not evaluable here.
        # Asserting it from the persistent-minus-memoryless number would make a
        # two-condition gate look like a three-condition one.
        t1_predictive_validation = None
        t1_persistent_validation = bool(persistent_increment < 0.0)
        t1_validation_summary = {
            "persistent_minus_memoryless": t1_result["validation"][
                "persistent_minus_memoryless"
            ],
            "strict_matched_wrong_time": t1_result["validation"][
                "strict_matched_wrong_time"
            ]["correct_minus_wrong_median"],
            "persistent_memory_supported": t1_persistent_validation,
            "time_specific_supported": bool(time_specific_increment < 0.0),
        }
    else:
        t1_predictive_validation = bool(
            t1_result["contrasts"]["filtered_minus_no_state_joint_nll"] < 0.0
        )
        t1_persistent_validation = bool(
            t1_result["contrasts"][
                "filtered_minus_validation_correction_off_joint_nll"
            ] < 0.0
        )
        t1_validation_summary = {
            "persistent_minus_validation_correction_off": t1_result[
                "contrasts"
            ]["filtered_minus_validation_correction_off_joint_nll"],
            "matched_correct_minus_wrong_time": t1_result["contrasts"][
                "matched_filtered_minus_wrong_time_joint_nll"
            ],
            "diagnostic_strength": "R1.2 supportive, not formal R1.3 diagnostic",
        }
    biological_contrasts_admissible = bool(
        not readout.degenerate and selected_t1_epoch > 0
        and t1_persistent_validation
        and (t1_predictive_validation is not False)
    )
    delayed_union_start = delayed_union_start_index(
        windows.start_index, context.event_segment, windows.delay_events,
    )
    real_window_nonoverlap = nonoverlapping_window_audit(
        context.design.event_time, windows.start_index,
        windows.end_index, windows.split,
    )
    full_instrument_nonoverlap = nonoverlapping_window_audit(
        context.design.event_time, delayed_union_start,
        windows.end_index, windows.split,
    )
    result = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "revision": LONG_TOTAL_REVISION,
        "subject": args.subject,
        "seed": int(args.seed),
        "window_kind": args.window,
        "exposure_memory": args.exposure_memory,
        "exposure_source": args.exposure_source,
        "window_definition": (
            f"exactly {scale_events} previous events within one recorded segment"
            if args.window.startswith("event_count_") else
            "approximately 6 h within one continuously recorded segment"
        ),
        "exposure": {
            "channels": exposure_channels,
            "innovation": innovation_audit,
            "load_innovation": (
                innovation_audit if args.exposure_source == "load" else None
            ),
            "participation_innovation": (
                innovation_audit
                if args.exposure_source == "repertoire" else None
            ),
            "scale_events": float(windows.exposure_scale_events),
            "counterfactual": (
                "same occurrence; all exposure innovations delayed 1000 events causally"
            ),
            "delay_events": int(windows.delay_events),
            "occurrence_block_variation": occurrence_block_variation(
                windows.real_operator, validation
            ),
        },
        "effective_exposure_time_scale": (
            boxcar_memory_audit(
                context.design.event_time, windows.start_index, windows.end_index
            )
            if args.exposure_memory == "boxcar" else
            effective_memory_audit(
                context.design.event_time, windows.start_index,
                windows.end_index, matrix,
            )
        ),
        "endpoint_support": endpoint_support_audit(
            context.design.event_time, windows.end_index, windows.split, matrix,
            exposure_memory=args.exposure_memory,
            start_index=windows.start_index,
        ),
        "delayed_control_overlap": delayed_control_overlap(windows),
        "train_validation_target_shift": target_shift_audit(
            target_delta, windows.split, readout,
        ),
        "estimability": {
            name: estimability_guard(
                validation_metrics[name],
                validation_metrics["no_edge_plus_fitted_intercept"],
            )
            for name in (real_name, delayed_name)
        },
        "real_exposure_window_nonoverlap_support": real_window_nonoverlap,
        "whole_instrument_nonoverlap_support": full_instrument_nonoverlap,
        # Backward-facing name now points to the scientifically conservative
        # real-plus-delayed union.  The explicit real-only field above prevents
        # ambiguity when comparing against v2 artifacts.
        "whole_window_nonoverlap_support": full_instrument_nonoverlap,
        "denominators": {
            "train_windows": int((windows.split == 0).sum()),
            "validation_windows": int((windows.split == 1).sum()),
            "validation_next_event_pairs": int(len(next_support)),
            "median_window_hours_validation": float(
                np.median(windows.duration_hours[validation])
            ),
            "median_events_per_window_validation": float(
                np.median(windows.n_events[validation])
            ),
            "windows_cross_unrecorded_gap": count_windows_crossing_segment(
                windows.start_index, windows.end_index, context.event_segment,
            ),
            "full_instrument_support_includes_causal_delay_events": int(
                windows.delay_events
            ),
        },
        "decoder_readout": {
            "rank": int(readout.rank),
            "state_dim": int(windows.start_state.shape[1]),
            "block_scales_train_only": readout.scales,
            "raw_block_scales_train_only": readout.raw_scales,
            "blocks_at_scale_floor": list(readout.blocks_at_scale_floor),
            "degenerate": bool(readout.degenerate),
            "primary_admissible": bool(not readout.degenerate),
        },
        "instrument_admissibility": {
            "t1_selected_epoch_above_zero": bool(selected_t1_epoch > 0),
            "t1_predictive_on_development_validation": t1_predictive_validation,
            "t1_predictive_check_available": bool(
                t1_predictive_validation is not None
            ),
            "t1_persistent_on_development_validation": t1_persistent_validation,
            "decoder_rank_above_zero": bool(readout.rank > 0),
            "decoder_blocks_carry_train_variation": bool(
                not readout.blocks_at_scale_floor
            ),
            "human_biological_contrasts_admissible": biological_contrasts_admissible,
            "structural_zero_if_false": True,
            "interpretation": (
                "if false, zero arm contrasts mean the fitted T1 never formed a "
                "state/readout instrument; they are not evidence against H3"
            ),
        },
        "fits": {
            real_name: real_fit,
            delayed_name: delayed_fit,
            "no_edge_plus_fitted_intercept": intercept_fit,
        },
        "validation_decoder_space": validation_metrics,
        "contrasts": {
            "real_minus_intercept_matched": metric_contrast(real, intercept),
            "real_minus_causal_delayed": metric_contrast(real, delayed),
            "delayed_minus_intercept_matched": metric_contrast(delayed, intercept),
            "real_minus_no_edge": metric_contrast(real, no_edge),
            "intercept_minus_no_edge": metric_contrast(intercept, no_edge),
        },
        "primary_contrasts_estimable": bool(
            estimability_guard(
                validation_metrics[real_name],
                validation_metrics["no_edge_plus_fitted_intercept"],
            )["estimable"]
            and estimability_guard(
                validation_metrics[delayed_name],
                validation_metrics["no_edge_plus_fitted_intercept"],
            )["estimable"]
        ),
        "contrast_roles": {
            "real_minus_intercept_matched": (
                "primary: exposure-driven variation beyond a free state offset"
            ),
            "real_minus_causal_delayed": (
                "primary: correct load timing, parameter- and intercept-matched"
            ),
            "delayed_minus_intercept_matched": (
                "occurrence-like cumulative variation beyond a free state offset"
            ),
            "real_minus_no_edge": (
                "NOT exposure evidence; the exposure arms own a free state-space "
                "intercept that no-edge lacks, so any mean offset between the "
                "frozen natural flow and the observed target wins here"
            ),
            "intercept_minus_no_edge": (
                "size of that intercept artefact, reported so the previous line "
                "cannot be read as cumulative-exposure signal"
            ),
        },
        "validation_half_hour_block_sensitivity": block_sensitivity,
        "next_event_exact_likelihood_secondary": next_event,
        "t1": {
            **context.audit,
            "selected_total_epoch": selected_t1_epoch,
            "source_class": args.t1_source,
            **t1_validation_summary,
        },
        "parameters_and_support": str(parameter_path),
        "parameters_and_support_sha256": contract.sha256_file(parameter_path),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "support-selected development patients; predictive long-exposure "
            "total-effect candidates, not causal network shaping or a cohort claim"
        ),
    }
    contract.atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
