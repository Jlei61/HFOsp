"""Human-data assembly for R1.5 exact-window long H3 experiments."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from . import contract
from .h3_long import (
    H3_LONG_REVISION,
    SOURCES,
    chronological_trend_exposure,
    exact_boxcar_event_exposure,
    exact_previous_block_placebo,
    independent_endpoint_rows,
    state_matching_estimable,
    standardise_exposure_on_train,
)
from .t2_human import FittedT1Context, load_fitted_explicit_t1
from .t2_r2 import (
    build_horizon_mark_design,
    fit_load_innovation_crossfit,
    fit_participation_innovation_crossfit,
    state_matched_nonoverlap_placebo,
)
from .t2_s1 import build_one_step_design


R1_5_REVISION = "r1_5_long_support_explicit_extension_v1"


def cell_package_fingerprint(
    subject: str,
    seed: int,
    source: str,
    scale: int,
    role: str,
    *,
    support_path: Path,
    r1_5_root: Path,
    runner_path: Path,
) -> tuple[str, dict]:
    """Bind one resumable cell to its data, model and exact producer package."""
    result_path = (
        Path(r1_5_root) / "fits" / subject
        / f"explicit_seed_{int(seed)}/result.json"
    )
    result = json.loads(result_path.read_text())
    checkpoint = Path(result["checkpoint"])
    components = {
        "subject": subject, "seed": int(seed), "source": source,
        "scale_events": int(scale), "support_role": role,
        "h3_long_revision": H3_LONG_REVISION,
        "r1_5_revision": R1_5_REVISION,
        "support_manifest_sha256": contract.sha256_file(support_path),
        "split_manifest_sha256": contract.sha256_file(contract.SPLIT_MANIFEST),
        "h3_long_sha256": contract.sha256_file(
            contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/h3_long.py"
        ),
        "h3_long_human_sha256": contract.sha256_file(Path(__file__)),
        "runner_sha256": contract.sha256_file(runner_path),
        "r1_5_result_sha256": contract.sha256_file(result_path),
        "r1_5_checkpoint_sha256": contract.sha256_file(checkpoint),
    }
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, components


def load_fitted_r1_5_explicit_t1(
    subject: str,
    seed: int,
    *,
    device: str = "cuda",
    r1_5_root: Path | None = None,
) -> FittedT1Context:
    root = Path(r1_5_root or contract.RESULT_ROOT / "r1_5")
    result_path = root / "fits" / subject / f"explicit_seed_{seed}/result.json"
    result = json.loads(result_path.read_text())
    if (
        result.get("status") != "COMPLETE"
        or result.get("experiment_label") != R1_5_REVISION
        or result.get("sealed_opened") is not False
    ):
        raise ValueError(f"invalid R1.5 explicit fit: {result_path}")
    checkpoint = Path(result["checkpoint"])
    if contract.sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("R1.5 checkpoint hash mismatch")
    cache_manifest = Path(result["observation_cache_manifest"])
    if contract.sha256_file(cache_manifest) != result[
        "observation_cache_manifest_sha256"
    ]:
        raise ValueError("R1.5 observation manifest hash mismatch")
    context = load_fitted_explicit_t1(
        subject, seed, device=device, r1_3_root=root,
        observation_cache_root=cache_manifest.parent.parent,
    )
    if context.pre_event_observation is None:
        raise RuntimeError("R1.5 loader omitted event observation embeddings")
    event_split = np.asarray(context.design.event_split, dtype=np.int8)
    contract.assert_development_times(
        subject, context.design.event_time[event_split == 0], "train"
    )
    contract.assert_development_times(
        subject, context.design.event_time[event_split == 1], "validation"
    )
    audit = {
        **context.audit,
        "t1_source": "r1_5_explicit_target_trained_observer",
        "r1_5_result": str(result_path),
        "r1_5_result_sha256": contract.sha256_file(result_path),
        "r1_5_experiment_label": result["experiment_label"],
        "persistent_minus_memoryless_joint": result["validation"][
            "persistent_minus_memoryless"
        ]["joint_nll_per_event"],
        "correct_minus_wrong_joint": result["validation"][
            "strict_matched_wrong_time"
        ]["correct_minus_wrong_median"]["joint_nll_per_event"],
        "selected_total_epoch": result["fit_trace"]["selected_total_epoch"],
        "seed_stable_t1": bool(
            result["fit_trace"]["selected_total_epoch"] > 0
            and result["validation"]["persistent_minus_memoryless"][
                "joint_nll_per_event"
            ] < 0
            and result["validation"]["strict_matched_wrong_time"][
                "correct_minus_wrong_median"
            ]["joint_nll_per_event"] < 0
        ),
        "development_time_contract_verified": True,
        "raw_arm_not_run_or_used": True,
    }
    return replace(context, audit=audit)


def event_innovation(context: FittedT1Context, source: str) -> tuple[np.ndarray, dict]:
    if source not in SOURCES:
        raise ValueError(f"unknown H3-long source: {source}")
    design = context.design
    train = np.asarray(design.event_split == 0)
    if source == "load":
        return fit_load_innovation_crossfit(
            context.pre_event_state, design.event_history,
            context.pre_event_observation,
            np.asarray(context.stream.load, dtype=np.float32), train,
        )
    return fit_participation_innovation_crossfit(
        context.pre_event_state, design.event_history,
        context.pre_event_observation,
        np.asarray(context.stream.participation, dtype=np.float32), train,
        components=2,
    )


def build_long_arm_designs(
    context: FittedT1Context,
    innovation: np.ndarray,
    *,
    source: str,
    scale_events: int,
    full_causal_control: bool,
    include_horizons: bool,
) -> tuple[dict, dict[int, dict], dict]:
    design = context.design
    train = np.asarray(design.event_split == 0)
    real_raw, real_eligible, real_audit = exact_boxcar_event_exposure(
        innovation, context.event_segment, scale_events=int(scale_events)
    )
    causal_raw, causal_eligible, causal_audit = exact_previous_block_placebo(
        real_raw, context.event_segment, scale_events=int(scale_events)
    )
    matched_raw, matched, matched_audit = state_matched_nonoverlap_placebo(
        real_raw, context.pre_event_state, design.event_history,
        context.pre_event_observation, train, real_eligible,
        context.event_segment, scale_events=int(scale_events),
        history_multiples=1, neighbours=32,
    )
    matched_audit.update({
        "exact_boxcar_windows": True,
        "real_and_donor_windows_exactly_disjoint": True,
        "history_nonoverlap_residual_weight_upper_bound": 0.0,
    })
    common = real_eligible & matched
    if full_causal_control:
        common &= causal_eligible
    real, real_scaler = standardise_exposure_on_train(
        real_raw, train, common
    )
    state_matched, matched_scaler = standardise_exposure_on_train(
        matched_raw, train, common
    )
    current, current_scaler = standardise_exposure_on_train(
        innovation, train, common
    )
    exposure_dim = 1 if real.ndim == 1 else int(real.shape[1])
    chronological_raw = chronological_trend_exposure(
        design.event_time, context.event_segment, exposure_dim
    )
    if exposure_dim == 1:
        chronological_raw = chronological_raw[:, 0]
    chronological, chronological_scaler = standardise_exposure_on_train(
        chronological_raw, train, common
    )
    arm_exposure = {
        "real_cumulative": real,
        "state_matched_nonoverlap": state_matched,
        "current_event_only": current,
        "chronological_trend": chronological,
        "intercept_only": np.zeros_like(real, dtype=np.float32),
    }
    scalers = {
        "real_cumulative": real_scaler,
        "state_matched_nonoverlap": matched_scaler,
        "current_event_only": current_scaler,
        "chronological_trend": chronological_scaler,
    }
    if full_causal_control:
        causal, causal_scaler = standardise_exposure_on_train(
            causal_raw, train, common
        )
        arm_exposure["causal_previous_block"] = causal
        scalers["causal_previous_block"] = causal_scaler
    one_step = {
        label: build_one_step_design(
            design, context.pre_event_state, context.event_segment,
            value, common,
        ) for label, value in arm_exposure.items()
    }
    reference = one_step["real_cumulative"].current_index
    if any(not np.array_equal(value.current_index, reference)
           for value in one_step.values()):
        raise RuntimeError("H3-long arms changed one-step support")
    horizons = {}
    if include_horizons:
        for horizon in (5, 10):
            horizons[horizon] = {
                label: build_horizon_mark_design(
                    design, context.pre_event_state, context.event_segment,
                    value, common, horizon,
                ) for label, value in arm_exposure.items()
            }
            reference_h = horizons[horizon]["real_cumulative"].current_index
            if any(not np.array_equal(value.current_index, reference_h)
                   for value in horizons[horizon].values()):
                raise RuntimeError(f"H3-long arms changed H{horizon} support")
    validation_reference = reference[
        one_step["real_cumulative"].split == 1
    ]
    unit_width = int(2 * scale_events if full_causal_control else scale_events)
    train_rows = np.flatnonzero(one_step["real_cumulative"].split == 0)
    validation_rows = np.flatnonzero(one_step["real_cumulative"].split == 1)
    independent_train_local = independent_endpoint_rows(
        reference[train_rows], context.event_segment, width_events=unit_width
    )
    independent_validation_local = independent_endpoint_rows(
        reference[validation_rows], context.event_segment, width_events=unit_width
    )
    independent_train_rows = train_rows[independent_train_local]
    independent_validation_rows = validation_rows[independent_validation_local]
    duration = (
        np.asarray(design.event_time[validation_reference], dtype=np.float64)
        - np.asarray(
            design.event_time[
                validation_reference - int(scale_events) + 1
            ],
            dtype=np.float64,
        )
    ) / 3600.0
    audit = {
        "revision": H3_LONG_REVISION, "source": source,
        "scale_events": int(scale_events),
        "full_causal_control": bool(full_causal_control),
        "innovation_is_train_only_cross_fitted": True,
        "real_exposure": real_audit, "causal_control": causal_audit,
        "state_matched_nonoverlap": matched_audit, "scalers": scalers,
        "state_matching_estimable": state_matching_estimable(matched_audit),
        "eligible_events_before_matching": int(real_eligible.sum()),
        "eligible_events_common_support": int(common.sum()),
        "train_next_event_pairs": int(
            (one_step["real_cumulative"].split == 0).sum()
        ),
        "validation_next_event_pairs": int(
            (one_step["real_cumulative"].split == 1).sum()
        ),
        "independent_unit_width_events": unit_width,
        "train_independent_units_on_final_common_support": int(
            len(independent_train_rows)
        ),
        "validation_independent_units_on_final_common_support": int(
            len(independent_validation_rows)
        ),
        "train_independent_design_rows": independent_train_rows.tolist(),
        "validation_independent_design_rows": independent_validation_rows.tolist(),
        "validation_window_hours_median": float(np.median(duration)),
        "validation_window_hours_q25": float(np.quantile(duration, .25)),
        "validation_window_hours_q75": float(np.quantile(duration, .75)),
        "arms_share_exact_support": True,
        "fitted_intercept_in_every_trainable_arm": True,
        "raw_correction_after_anchor": False,
        "later_h3_jumps": False,
        "future_event_history_teacher_forced": bool(include_horizons),
        "autonomous_rollout": False,
        "nuisance_and_edge_inner_selection_are_not_nested": True,
        "sealed_opened": False,
    }
    return one_step, horizons, audit
