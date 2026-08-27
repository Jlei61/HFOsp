"""Human-data assembly for the frozen N=100 T2-R2.0 experiment."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import contract
from .t2_human import FittedT1Context, load_fitted_explicit_t1
from .t2_r2 import (
    T2_R2_REVISION,
    build_horizon_mark_design,
    exponential_event_exposure,
    fit_load_innovation_crossfit,
    fit_participation_innovation_crossfit,
    state_matched_nonoverlap_placebo,
)
from .t2_s1 import build_one_step_design


R1_4_REVISION = "r1_4_six_patient_explicit_primary_raw_residual_v1"
SOURCES = ("load", "participation")


def load_fitted_r1_4_explicit_t1(
    subject: str,
    seed: int,
    *,
    device: str = "cuda",
    r1_4_root: Path | None = None,
) -> FittedT1Context:
    """Reconstruct one R1.4 explicit checkpoint and its event observations."""
    root = Path(r1_4_root or contract.RESULT_ROOT / "r1_4")
    result_path = root / "fits" / subject / f"explicit_seed_{seed}" / "result.json"
    result = json.loads(result_path.read_text())
    if (result.get("status") != "COMPLETE"
            or result.get("experiment_label") != R1_4_REVISION
            or result.get("sealed_opened") is not False):
        raise ValueError(f"invalid R1.4 explicit fit: {result_path}")
    checkpoint = Path(result["checkpoint"])
    if contract.sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("R1.4 checkpoint hash mismatch")
    cache_manifest_path = Path(result["observation_cache_manifest"])
    if contract.sha256_file(cache_manifest_path) != result[
        "observation_cache_manifest_sha256"
    ]:
        raise ValueError("R1.4 observation manifest hash mismatch")
    context = load_fitted_explicit_t1(
        subject, seed, device=device, r1_3_root=root,
        observation_cache_root=cache_manifest_path.parent.parent,
    )
    if context.pre_event_observation is None:
        raise RuntimeError("R1.4 loader did not return event observation embeddings")
    audit = {
        **context.audit,
        "t1_source": "r1_4_explicit_target_trained_observer",
        "r1_4_result": str(result_path),
        "r1_4_result_sha256": contract.sha256_file(result_path),
        "r1_4_experiment_label": result["experiment_label"],
        "persistent_minus_memoryless_joint": result["validation"][
            "persistent_minus_memoryless"
        ]["joint_nll_per_event"],
        "correct_minus_wrong_joint": result["validation"][
            "strict_matched_wrong_time"
        ]["correct_minus_wrong_median"]["joint_nll_per_event"],
        "selected_total_epoch": result["fit_trace"]["selected_total_epoch"],
        "raw_arm_not_used_to_select_or_fit_t2": True,
    }
    return replace(context, audit=audit)


def build_r2_arm_designs(
    context: FittedT1Context,
    *,
    source: str,
    scale_events: int = 100,
    validation_time_lower: float | None = None,
    include_fitted_intercept_diagnostic: bool = True,
) -> tuple[dict, dict[int, dict], dict]:
    """Build identical-support next-event and H5/H10 arm designs."""
    if source not in SOURCES:
        raise ValueError(f"unknown T2-R2.0 source: {source}")
    if int(scale_events) != 100:
        raise ValueError("T2-R2.0 first stage freezes N=100")
    if context.pre_event_observation is None:
        raise ValueError("T2-R2.0 requires the pre-event observation embedding")
    design = context.design
    train = np.asarray(design.event_split == 0)
    if source == "load":
        innovation, innovation_audit = fit_load_innovation_crossfit(
            context.pre_event_state,
            design.event_history,
            context.pre_event_observation,
            np.asarray(context.stream.load, dtype=np.float32),
            train,
        )
    else:
        innovation, innovation_audit = fit_participation_innovation_crossfit(
            context.pre_event_state,
            design.event_history,
            context.pre_event_observation,
            np.asarray(context.stream.participation, dtype=np.float32),
            train,
            components=2,
        )
    real, eligible, exposure_audit = exponential_event_exposure(
        innovation, context.event_segment, scale_events=100
    )
    placebo, matched, placebo_audit = state_matched_nonoverlap_placebo(
        real,
        context.pre_event_state,
        design.event_history,
        context.pre_event_observation,
        train,
        eligible,
        context.event_segment,
        scale_events=100,
        history_multiples=5,
    )
    common = eligible & matched
    if validation_time_lower is not None:
        common &= (
            train | (np.asarray(design.event_time) >= float(validation_time_lower))
        )
    zeros = np.zeros_like(real, dtype=np.float32)
    current = np.asarray(innovation, dtype=np.float32)
    exposures = {
        "no_edge": zeros,
        "real_cumulative": real,
        "state_matched_placebo": placebo,
        "current_event_only": current,
    }
    if include_fitted_intercept_diagnostic:
        exposures["fitted_intercept_diagnostic"] = np.ones(
            (len(real), 1), dtype=np.float32
        )
    one_step = {
        label: build_one_step_design(
            design, context.pre_event_state, context.event_segment,
            value, common,
        ) for label, value in exposures.items()
    }
    reference = one_step["real_cumulative"].current_index
    if any(not np.array_equal(value.current_index, reference)
           for value in one_step.values()):
        raise RuntimeError("T2-R2.0 arms changed next-event support")
    horizons = {}
    for horizon in (5, 10):
        horizons[horizon] = {
            label: build_horizon_mark_design(
                design, context.pre_event_state, context.event_segment,
                value, common, horizon,
            ) for label, value in exposures.items()
        }
        reference_h = horizons[horizon]["real_cumulative"].current_index
        if any(not np.array_equal(value.current_index, reference_h)
               for value in horizons[horizon].values()):
            raise RuntimeError(f"T2-R2.0 arms changed H{horizon} support")
    audit = {
        "revision": T2_R2_REVISION,
        "source": source,
        "scale_events": 100,
        "innovation": innovation_audit,
        "exposure": exposure_audit,
        "placebo": placebo_audit,
        "eligible_events_before_placebo": int(eligible.sum()),
        "eligible_events_after_placebo": int(common.sum()),
        "train_next_event_pairs": int((one_step["real_cumulative"].split == 0).sum()),
        "validation_next_event_pairs": int((one_step["real_cumulative"].split == 1).sum()),
        "horizon_support": {
            f"H{horizon}": {
                "train": int((value["real_cumulative"].split == 0).sum()),
                "validation": int((value["real_cumulative"].split == 1).sum()),
            } for horizon, value in horizons.items()
        },
        "arms_share_exact_support": True,
        "observer_generator_history_and_decoders_frozen": True,
        "raw_correction_after_anchor": False,
        "later_t2_jumps": False,
        "validation_time_lower": validation_time_lower,
        "d_state_validation_events_excluded": validation_time_lower is not None,
        "fitted_intercept_is_diagnostic_not_primary_control": bool(
            include_fitted_intercept_diagnostic
        ),
        "free_exposure_intercept_present": bool(
            include_fitted_intercept_diagnostic
        ),
        "sealed_opened": False,
    }
    return one_step, horizons, audit
