#!/usr/bin/env python3
"""Close the locked R3 actual-entry regional hybrid lifecycle at one center."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
)
from src.topic4_mz_spatial_autonomous_latch import (  # noqa: E402
    Pulse,
    RegionalSlowParameters,
    integrate_autonomous_latch_batch,
    regional_slow_rhs,
)
from src.topic4_mz_spatial_patch import (  # noqa: E402
    LOCAL_FIELDS,
    patch_rhs_fast_and_moments,
    prepare_patch_rhs,
)
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_actual_entry_lifecycle_closure.yaml"
SUPPORTED = "R4_ACTUAL_ENTRY_REGIONAL_HYBRID_LIFECYCLE_CENTER_SUPPORTED"
SOURCE_MISMATCH = "R4_ACTUAL_ENTRY_SOURCE_ARTIFACT_MISMATCH_UNRESOLVED"
CHECKPOINT_INVALID = "R4_ACTUAL_ENTRY_CHECKPOINT_INVALID_UNRESOLVED"
PROTECTED_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_PROTECTED_RECOVERY_LOST_LOW_BRANCH"
RESET_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_LATCH_RESET_FAILED"
SAME_BASIN_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_SAME_BASIN_RETURN_FAILED"
EARLY_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_EARLY_REFRACTORINESS_FAILED"
LATE_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_LATE_LIFECYCLE_FAILED"
DT_NO_GO = "R4_ACTUAL_ENTRY_CLEAN_NO_GO_BASE_HALF_DT_DISAGREEMENT"
NUMERIC_UNRESOLVED = "R4_ACTUAL_ENTRY_NUMERIC_OR_RESOURCE_UNRESOLVED"
PATCH_NAMES = ("core", "annulus", "bath")
PATCH_COLORS = ("#B2182B", "#EF8A62", "#2166AC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_manifest(paths: Sequence[Path], *, root: Path = ROOT) -> dict[str, str]:
    """Hash immutable artifacts using paths relative to one explicit root."""

    base = root.resolve()
    manifest: dict[str, str] = {}
    for raw in paths:
        path = raw.resolve()
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"output artifact escapes manifest root: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"output artifact is missing: {path}")
        manifest[str(relative)] = _sha256(path)
    return manifest


def _verify_hash_manifest(manifest: dict[str, str], *, root: Path = ROOT) -> None:
    """Fail closed before a reporting-only refresh reads canonical outputs."""

    if not manifest:
        raise RuntimeError("canonical output_sha256 manifest is missing or empty")
    base = root.resolve()
    for relative, expected in manifest.items():
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"manifest path escapes root: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"manifest artifact is missing: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"canonical output hash mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _optional(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _validate_config(cfg: dict[str, Any]) -> None:
    if set(cfg["input_sha256"]) != {
        "source_summary_path", "source_trace_path", "source_config_path",
    }:
        raise RuntimeError("R4 must lock exactly the three R3 source artifacts")
    source = cfg["source_phenotype"]
    exact_source = {
        "entry_fold_q": 0.8558315843088748,
        "response_exclusion_ms": 100.0,
        "pairing_max_lag_ms": 20.0,
        "minimum_paired_returns": 4,
        "section_level_khz": 0.020,
        "rearm_level_khz": 0.015,
        "low_threshold_khz": 0.005,
        "sustained_low_ms": 250.0,
        "expected_first_entry_event": 5,
    }
    for key, expected in exact_source.items():
        if not np.isclose(float(source[key]), expected, rtol=0.0, atol=1.0e-14):
            raise RuntimeError(f"source phenotype contract drifted at {key}")
    bridge = cfg["protected_recovery"]
    if (
        float(bridge["q_stop_epsilon"]) != 0.0005
        or [float(x) for x in bridge["sentinel_fractions"]] != [0.25, 0.5, 0.75]
        or float(bridge["sentinel_duration_ms"]) != 500.0
    ):
        raise RuntimeError("protected recovery contract drifted")
    release = cfg["natural_release"]
    if [float(x) for x in release["additive_sentinels_mv"]] != [0.10, 0.02, 0.002]:
        raise RuntimeError("natural-release sentinels drifted")
    challenge = cfg["retrigger"]
    if (
        [float(x) for x in challenge["onsets_ms"]]
        != [1000.0, 3122.0, 5044.0, 6321.0, 7531.0, 10915.0]
        or float(challenge["pulse_duration_ms"]) != 20.0
        or float(challenge["amplitude_mv"]) != 3.0
        or [float(x) for x in challenge["profile_core_annulus_bath"]] != [1.0, 0.0, 0.0]
    ):
        raise RuntimeError("registered challenge drifted")
    integration = cfg["integration"]
    if (
        [float(x) for x in integration["dt_values_ms"]] != [0.125, 0.0625]
        or float(integration["save_dt_ms"]) != 1.0
        or int(integration["max_trace_bytes"]) > 64 * 1024 * 1024
    ):
        raise RuntimeError("R4 integration contract drifted")
    resources = cfg["resource_contract"]
    if (
        int(resources["processes"]) != 1
        or int(resources["blas_threads"]) != 1
        or float(resources["max_memory_gib"]) > 1.5
        or float(resources["max_total_wall_seconds"]) > 1200.0
        or not bool(resources["base_must_pass_before_half_dt"])
    ):
        raise RuntimeError("R4 resource contract drifted")
    scope = cfg["scope"]
    required = (
        "center_only", "actual_entry_alignment", "hybrid_analytic_zero_use_bridge",
        "real_latch_state_machine_reset", "common_early_late_classifier",
        "fixed_bath_resource_mask",
    )
    forbidden = (
        "q_recalibration", "parameter_grid", "continuous_field", "full_snn",
        "ee_weight_change", "ee_kernel_change", "conductance_membrane", "relay_change",
    )
    if not all(bool(scope[key]) for key in required) or any(bool(scope[key]) for key in forbidden):
        raise RuntimeError("R4 scope/non-overlap contract drifted")


def _validate_inputs(
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    observed: dict[str, str] = {}
    for key, expected in cfg["input_sha256"].items():
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = _sha256(path)
        if observed[key] != str(expected):
            raise RuntimeError(f"locked source drift for {key}: {observed[key]}")
    summary = json.loads((ROOT / cfg["source_summary_path"]).read_text(encoding="utf-8"))
    if summary.get("status") != cfg["source_phenotype"]["required_status"]:
        raise RuntimeError("R3 source status no longer matches the locked clean no-go")
    with np.load(ROOT / cfg["source_trace_path"], allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    if (
        arrays.get("final_state", np.empty(0)).shape != (32,)
        or arrays["final_state"].dtype != np.float64
        or arrays.get("final_latch_state", np.empty(0)).shape != (3,)
        or arrays["final_latch_state"].dtype != np.bool_
    ):
        raise RuntimeError("R3 continuation checkpoint is not saved in the locked double/bool form")
    source_cfg = yaml.safe_load((ROOT / cfg["source_config_path"]).read_text(encoding="utf-8"))
    return observed, summary, arrays, source_cfg


def _crossings(
    time_ms: Sequence[float], rate_khz: Sequence[float], level: float, rearm: float,
) -> list[float]:
    time_values = np.asarray(time_ms, dtype=float)
    rates = np.asarray(rate_khz, dtype=float)
    if time_values.ndim != 1 or rates.shape != time_values.shape or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("crossing arrays must be aligned and strictly increasing")
    armed = bool(rates[0] <= rearm)
    output: list[float] = []
    for index in range(time_values.size - 1):
        left, right = float(rates[index]), float(rates[index + 1])
        if left <= rearm:
            armed = True
        if armed and left < level <= right:
            fraction = (level - left) / (right - left)
            output.append(float(time_values[index] + fraction * (time_values[index + 1] - time_values[index])))
            armed = False
    return output


def _downcrossings(time_ms: Sequence[float], rate_khz: Sequence[float], level: float) -> list[float]:
    time_values = np.asarray(time_ms, dtype=float)
    rates = np.asarray(rate_khz, dtype=float)
    output: list[float] = []
    for index in range(time_values.size - 1):
        left, right = float(rates[index]), float(rates[index + 1])
        if left >= level > right:
            fraction = (left - level) / (left - right)
            output.append(float(time_values[index] + fraction * (time_values[index + 1] - time_values[index])))
    return output


def _pair_crossings(
    core: Sequence[float], annulus: Sequence[float], max_lag_ms: float,
) -> list[tuple[float, float]]:
    """Greedy chronological one-to-one matching with an inclusive lag bound."""

    left = [float(x) for x in core]
    right = [float(x) for x in annulus]
    pairs: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= float(max_lag_ms):
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif delta < 0.0:
            i += 1
        else:
            j += 1
    return pairs


def _first_sustained_low(
    time_ms: np.ndarray, rates: np.ndarray, threshold: float, duration_ms: float, start_ms: float,
) -> float | None:
    low = np.all(np.asarray(rates, dtype=float) <= float(threshold), axis=1)
    low &= np.asarray(time_ms, dtype=float) >= float(start_ms)
    starts = np.flatnonzero(low & np.r_[True, ~low[:-1]])
    for index in starts:
        stop = int(np.searchsorted(time_ms, float(time_ms[index]) + duration_ms, side="left"))
        if stop < time_ms.size and bool(np.all(low[index:stop + 1])):
            return float(time_ms[index])
    return None


def _numeric_clean(result: dict[str, Any]) -> bool:
    failures = np.r_[
        np.asarray(result["first_support_failure_ms"], dtype=float),
        np.asarray(result["first_bound_failure_ms"], dtype=float),
        np.asarray(result["first_nonfinite_ms"], dtype=float),
    ]
    return bool(
        bool(np.asarray(result["finite"])[0])
        and bool(np.asarray(result["active_at_end"])[0])
        and not np.any(np.isfinite(failures))
        and int(np.sum(result["support_violation_count"])) == 0
        and int(np.sum(result["state_bound_violation_count"])) == 0
    )


def _source_phenotype(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], cfg: dict[str, Any],
) -> dict[str, Any]:
    contract = cfg["source_phenotype"]
    time_ms = np.asarray(arrays["time_ms"], dtype=float)
    q = np.asarray(arrays["q"], dtype=float)
    rate = np.asarray(arrays["rE_khz"], dtype=float)
    fast = np.asarray(arrays["rE_fast_khz"], dtype=float)
    latch = np.asarray(arrays["latch"], dtype=bool)
    onsets = np.asarray(summary["config"]["background_event_challenge"]["realized_onsets_ms"], dtype=float)
    pulse_duration = float(summary["config"]["background_event_challenge"]["duration_ms"])
    fold = float(contract["entry_fold_q"])
    entry_indices = np.flatnonzero(np.min(q[:, :2], axis=1) < fold)
    entry_time = None if not entry_indices.size else float(time_ms[int(entry_indices[0])])
    if entry_time is None:
        entry_event = None
    elif entry_time < onsets[0]:
        entry_event = 0
    else:
        entry_event = int(np.flatnonzero(onsets <= entry_time)[-1] + 1)
    event_min_q = []
    for onset in onsets:
        mask = (time_ms >= onset) & (time_ms <= onset + pulse_duration + float(contract["response_exclusion_ms"]))
        event_min_q.append(float(np.min(q[mask, :2])))
    returns = [
        _crossings(time_ms, fast[:, patch], float(contract["section_level_khz"]), float(contract["rearm_level_khz"]))
        for patch in range(2)
    ]
    analysis_start = float(onsets[4] + pulse_duration + float(contract["response_exclusion_ms"]))
    event6 = float(onsets[5])
    aligned = [[value for value in values if analysis_start <= value < event6] for values in returns]
    pairs = _pair_crossings(aligned[0], aligned[1], float(contract["pairing_max_lag_ms"]))
    paired_times = [max(pair) for pair in pairs]
    fourth_time = paired_times[3] if len(paired_times) >= 4 else None
    joint_down = None
    if fourth_time is not None:
        downs = [
            [value for value in _downcrossings(time_ms, fast[:, patch], float(contract["section_level_khz"])) if value >= fourth_time]
            for patch in range(2)
        ]
        if all(values for values in downs):
            joint_down = max(downs[0][0], downs[1][0])
    low_onset = None
    if joint_down is not None:
        low_onset = _first_sustained_low(
            time_ms, np.c_[rate[:, :2], fast[:, :2]], float(contract["low_threshold_khz"]),
            float(contract["sustained_low_ms"]), joint_down,
        )
    event6_returns = [[value for value in values if value >= event6] for values in returns]
    transitions = np.flatnonzero(np.any(latch[1:, :2] != latch[:-1, :2], axis=1)) + 1
    set_times = [float(time_ms[index]) for index in transitions if bool(np.all(latch[index, :2]))]
    tolerance = float(contract["diagnostic_time_tolerance_ms"])
    sentinels = {
        "entry_time_matches": entry_time is not None and abs(entry_time - float(contract["expected_first_entry_time_ms"])) <= tolerance,
        "paired_times_match": len(paired_times) >= 4 and bool(np.all(np.abs(np.asarray(paired_times[:4]) - np.asarray(contract["expected_paired_times_ms"])) <= tolerance)),
        "joint_down_matches": joint_down is not None and abs(joint_down - float(contract["expected_joint_last_downcross_ms"])) <= tolerance,
        "low_onset_matches": low_onset is not None and abs(low_onset - float(contract["expected_sustained_all_low_onset_ms"])) <= tolerance,
        "latch_set_matches": bool(set_times) and abs(set_times[0] - float(contract["expected_latch_set_ms"])) <= tolerance,
    }
    gates = {
        "events_1_to_4_above_fold": bool(all(value >= fold for value in event_min_q[:4])),
        "event_5_first_fold_crossing": entry_event == int(contract["expected_first_entry_event"]),
        "four_one_to_one_paired_returns": len(pairs) >= int(contract["minimum_paired_returns"]),
        "fourth_pair_defined_as_later_crossing": fourth_time is not None and fourth_time == max(pairs[3]) if len(pairs) >= 4 else False,
        "post_fourth_downcross_and_250ms_low": joint_down is not None and low_onset is not None,
        "event_6_no_section_crossing": not event6_returns[0] and not event6_returns[1],
        "numeric_clean": bool(
            summary.get("finite") and summary.get("active_at_end")
            and summary.get("first_support_failure_ms") is None
            and summary.get("first_bound_failure_ms") is None
            and summary.get("first_nonfinite_ms") is None
        ),
        "registered_diagnostic_sentinels_match": all(sentinels.values()),
    }
    return {
        "accepted": all(gates.values()), "gates": gates, "diagnostic_sentinels": sentinels,
        "entry_event_index": entry_event, "entry_time_ms": entry_time,
        "event_min_regional_q": event_min_q,
        "core_return_times_ms": aligned[0], "annulus_return_times_ms": aligned[1],
        "paired_return_times_ms": paired_times, "fourth_paired_time_ms": fourth_time,
        "joint_last_downcross_ms": joint_down, "sustained_all_low_onset_ms": low_onset,
        "termination_complete_ms": None if low_onset is None else low_onset + float(contract["sustained_low_ms"]),
        "latch_set_times_ms": set_times, "event6_return_times_ms": event6_returns,
    }


def _analytic_active_latch(
    *, q: float, p: float, m: float, duration_ms: float, q_rest: float,
    tau_p_ms: float, tau_slow_ms: float, tau_fast_ms: float,
) -> tuple[float, float, float]:
    rate = (1.0 - float(m)) / float(tau_slow_ms) + float(m) / float(tau_fast_ms)
    return (
        float(q_rest - (q_rest - q) * np.exp(-rate * duration_ms)),
        float(p * np.exp(-duration_ms / tau_p_ms)),
        float(m),
    )


def _analytic_released_latch(
    *, q: float, p: float, m: float, duration_ms: float, q_rest: float,
    tau_p_ms: float, tau_m_ms: float, tau_slow_ms: float, tau_fast_ms: float,
) -> tuple[float, float, float]:
    exponential = (
        duration_ms / tau_slow_ms
        + (1.0 / tau_fast_ms - 1.0 / tau_slow_ms)
        * m * tau_m_ms * (1.0 - np.exp(-duration_ms / tau_m_ms))
    )
    return (
        float(q_rest - (q_rest - q) * np.exp(-exponential)),
        float(p * np.exp(-duration_ms / tau_p_ms)),
        float(m * np.exp(-duration_ms / tau_m_ms)),
    )


def _solve_released_duration_for_q(
    *, q: float, m: float, target_q: float, q_rest: float,
    tau_m_ms: float, tau_slow_ms: float, tau_fast_ms: float,
) -> float:
    if target_q < q or target_q >= q_rest:
        raise ValueError("released q target must satisfy q<=target<q_rest")
    if np.isclose(target_q, q, rtol=0.0, atol=1.0e-15):
        return 0.0
    low, high = 0.0, float(tau_slow_ms)
    while _analytic_released_latch(
        q=q, p=0.0, m=m, duration_ms=high, q_rest=q_rest,
        tau_p_ms=1.0, tau_m_ms=tau_m_ms, tau_slow_ms=tau_slow_ms,
        tau_fast_ms=tau_fast_ms,
    )[0] < target_q:
        high *= 2.0
        if high > 1.0e9:
            raise RuntimeError("failed to bracket released q target")
    for _ in range(100):
        middle = 0.5 * (low + high)
        value = _analytic_released_latch(
            q=q, p=0.0, m=m, duration_ms=middle, q_rest=q_rest,
            tau_p_ms=1.0, tau_m_ms=tau_m_ms, tau_slow_ms=tau_slow_ms,
            tau_fast_ms=tau_fast_ms,
        )[0]
        if value < target_q:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def _advance_active_state(state: np.ndarray, duration_ms: float, arm: RegionalSlowParameters) -> np.ndarray:
    output = np.asarray(state, dtype=float).copy()
    pcount = 3
    regional_m = float(np.mean(output[9 * pcount:9 * pcount + 2]))
    for patch in range(2):
        q, persistence, _ = _analytic_active_latch(
            q=float(output[7 * pcount + patch]), p=float(output[8 * pcount + patch]),
            m=regional_m, duration_ms=duration_ms, q_rest=arm.z_rest,
            tau_p_ms=arm.tau_p_ms, tau_slow_ms=arm.tau_z_recovery_ms,
            tau_fast_ms=float(arm.tau_z_fast_recovery_ms),
        )
        output[7 * pcount + patch] = q
        output[8 * pcount + patch] = persistence
    output[8 * pcount + 2] *= np.exp(-duration_ms / arm.tau_p_ms)
    return output


def _advance_released_state(state: np.ndarray, duration_ms: float, arm: RegionalSlowParameters) -> np.ndarray:
    output = np.asarray(state, dtype=float).copy()
    pcount = 3
    regional_m = float(np.mean(output[9 * pcount:9 * pcount + 2]))
    for patch in range(2):
        q, persistence, m = _analytic_released_latch(
            q=float(output[7 * pcount + patch]), p=float(output[8 * pcount + patch]),
            m=regional_m, duration_ms=duration_ms, q_rest=arm.z_rest,
            tau_p_ms=arm.tau_p_ms, tau_m_ms=arm.tau_m_down_ms,
            tau_slow_ms=arm.tau_z_recovery_ms,
            tau_fast_ms=float(arm.tau_z_fast_recovery_ms),
        )
        output[7 * pcount + patch] = q
        output[8 * pcount + patch] = persistence
        output[9 * pcount + patch] = m
    output[8 * pcount + 2] *= np.exp(-duration_ms / arm.tau_p_ms)
    output[9 * pcount + 2] *= np.exp(-duration_ms / arm.tau_m_down_ms)
    return output


def _build_runtime(source_cfg: dict[str, Any]) -> dict[str, Any]:
    transfer = _load_transfer(source_cfg)
    parameters, low_parameters = _model(source_cfg)
    geometry = source_cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, low_root = _low_template(transfer, low_parameters)
    low_state = _low_initial(low, float(source_cfg["model"]["z_interictal"]), reduction, parameters)
    baseline = np.asarray(low_state[9:12], dtype=float)
    center = source_cfg["center_canary"]
    common = source_cfg["slow_common"]
    arm = RegionalSlowParameters(
        z_rest=float(common["z_rest"]), tau_z_recovery_ms=1000.0 * float(center["tau_slow_s"]),
        tau_z_depletion_ms=float(center["tau_depletion_ms"]),
        inhibitory_use_threshold_khz=float(common["inhibitory_use_threshold_khz"]),
        inhibitory_use_width_khz=float(common["inhibitory_use_width_khz"]),
        tau_p_ms=float(common["tau_p_ms"]),
        occupancy_threshold_khz=float(common["occupancy_threshold_khz"]),
        occupancy_width_khz=float(common["occupancy_width_khz"]),
        persistence_on=float(center["persistence_on"]),
        persistence_off=float(common["persistence_off"]),
        recruitment_on=float(common["recruitment_on"]),
        low_reset_threshold_khz=float(common["low_reset_threshold_khz"]),
        z_safe=float(common["z_safe"]), tau_m_up_ms=float(center["tau_m_up_ms"]),
        tau_m_down_ms=float(center["tau_m_down_ms"]),
        depletion_mask=tuple(float(x) for x in common["depletion_mask"]),
        pool_core_annulus_resource=bool(common["pool_core_annulus_resource"]),
        pool_core_annulus_effector=bool(common["pool_core_annulus_effector"]),
        enable_z=bool(common["enable_z"]), enable_m=bool(common["enable_m"]),
        q_reserve=float(center["q_reserve"]),
        tau_z_fast_recovery_ms=1000.0 * float(center["tau_fast_s"]),
        enable_m_gated_z_recovery=bool(common["enable_m_gated_z_recovery"]),
    ).validate()
    return {
        "transfer": transfer, "parameters": parameters, "prepared": prepared,
        "reduction": reduction, "low_state": low_state, "low_root": low_root,
        "baseline": baseline, "arm": arm,
    }


def _pulses(cfg: dict[str, Any]) -> list[Pulse]:
    challenge = cfg["retrigger"]
    return [
        Pulse(
            onset_ms=float(onset), duration_ms=float(challenge["pulse_duration_ms"]),
            amplitude_mv=float(challenge["amplitude_mv"]),
            profile=tuple(float(x) for x in challenge["profile_core_annulus_bath"]),
        )
        for onset in challenge["onsets_ms"]
    ]


def _integrate(
    state: np.ndarray, latch: np.ndarray, runtime: dict[str, Any], cfg: dict[str, Any],
    dt_ms: float, duration_ms: float, pulses: Sequence[Pulse] = (),
) -> dict[str, Any]:
    source = cfg["source_phenotype"]
    return integrate_autonomous_latch_batch(
        np.asarray(state, dtype=float)[None, :], runtime["prepared"], runtime["transfer"],
        [runtime["arm"]], list(pulses), inhibitory_baseline_khz=runtime["baseline"],
        dt_ms=float(dt_ms), duration_ms=float(duration_ms),
        save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
        section_level_khz=float(source["section_level_khz"]),
        rearm_level_khz=float(source["rearm_level_khz"]),
        max_trace_bytes=int(cfg["integration"]["max_trace_bytes"]),
        initial_latch_state=np.asarray(latch, dtype=bool)[None, :],
    )


def _pooled(state: np.ndarray) -> dict[str, float]:
    state = np.asarray(state, dtype=float)
    return {
        "q": float(np.mean(state[21:23])), "p": float(np.max(state[24:26])),
        "m": float(np.mean(state[27:29])), "A_mv": 1.6 * float(np.mean(state[27:29])),
        "rE_max_hz": 1000.0 * float(np.max(state[0:2])),
        "rE_fast_max_hz": 1000.0 * float(np.max(state[18:20])),
    }


def _endpoint(rows: list[dict[str, Any]], dt_ms: float, segment: str, kind: str, time_ms: float, state: np.ndarray, latch: np.ndarray) -> None:
    values = _pooled(state)
    rows.append({
        "dt_ms": dt_ms, "segment": segment, "kind": kind, "absolute_time_ms": float(time_ms),
        **values, "latch_on": bool(np.any(np.asarray(latch, dtype=bool)[:2])),
    })


def _sentinel_gates(
    result: dict[str, Any], *, expected_latch: bool, settling_ms: float, zero_tolerance: float,
) -> dict[str, bool]:
    time_ms = np.asarray(result["time_ms"], dtype=float)
    settled = time_ms >= float(settling_ms)
    latch = np.asarray(result["latch"][:, 0, :2], dtype=bool)
    return {
        "numeric_clean": _numeric_clean(result),
        "no_section_return": not result["return_times_ms"][0][0] and not result["return_times_ms"][0][1],
        "latch_state_preserved": bool(np.all(latch == expected_latch)),
        "zero_use_after_settling": float(np.max(np.abs(result["z_use"][settled, 0, :2]))) <= zero_tolerance,
        "zero_occupancy_after_settling": float(np.max(np.abs(result["occupancy"][settled, 0, :2]))) <= zero_tolerance,
        "low_branch_after_settling": bool(
            np.max(result["rE"][settled, 0, :2]) < 0.005
            and np.max(result["rE_fast"][settled, 0, :2]) < 0.005
        ),
    }


def _checkpoint_gates(
    state: np.ndarray, latch: np.ndarray, runtime: dict[str, Any], cfg: dict[str, Any],
) -> dict[str, Any]:
    arm = runtime["arm"]
    _, sensors = regional_slow_rhs(
        np.asarray(state, dtype=float)[None, :], [arm],
        inhibitory_baseline_khz=runtime["baseline"],
        recruitment_kernel=runtime["prepared"].K_EE,
        patch_weights=runtime["prepared"].patch_weights,
        latch_state=np.asarray(latch, dtype=bool)[None, :],
    )
    _, moments = patch_rhs_fast_and_moments(state, runtime["prepared"], runtime["transfer"])
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    support = runtime["transfer"].support_mask(mu_e, sigma_e) & runtime["transfer"].support_mask(mu_i, sigma_i)
    limits = cfg["checkpoint"]
    values = _pooled(state)
    gates = {
        "latch_on": bool(np.all(np.asarray(latch, dtype=bool)[:2])) and not bool(latch[2]),
        "regional_use_zero": float(np.max(np.abs(sensors["z_use"][0, :2]))) <= float(limits["sensor_zero_tolerance"]),
        "regional_occupancy_zero": float(np.max(np.abs(sensors["occupancy"][0, :2]))) <= float(limits["sensor_zero_tolerance"]),
        "regional_rates_low": values["rE_max_hz"] < 5.0 and values["rE_fast_max_hz"] < 5.0,
        "p_below_limit": values["p"] <= float(limits["p_max"]),
        "q_below_reset": values["q"] < float(limits["q_reset_threshold"]),
        "m_positive": values["m"] > 0.0,
        "finite_supported": bool(np.all(np.isfinite(state)) and np.all(support)),
    }
    return {"accepted": all(gates.values()), "gates": gates, "values": values}


def _classify_challenge(
    result: dict[str, Any], cfg: dict[str, Any], *,
    numeric_clean_override: bool | None = None,
) -> dict[str, Any]:
    contract = cfg["source_phenotype"]
    challenge = cfg["retrigger"]
    time_ms = np.asarray(result["time_ms"], dtype=float)
    q = np.asarray(result["z"][:, 0, :], dtype=float)
    rate = np.asarray(result["rE"][:, 0, :], dtype=float)
    fast = np.asarray(result["rE_fast"][:, 0, :], dtype=float)
    onsets = np.asarray(challenge["onsets_ms"], dtype=float)
    duration = float(challenge["pulse_duration_ms"])
    fold = float(contract["entry_fold_q"])
    entry_indices = np.flatnonzero(np.min(q[:, :2], axis=1) < fold)
    entry_time = None if not entry_indices.size else float(time_ms[int(entry_indices[0])])
    if entry_time is None:
        entry_event = None
    elif entry_time < onsets[0]:
        entry_event = 0
    else:
        entry_event = int(np.flatnonzero(onsets <= entry_time)[-1] + 1)
    event_min = []
    for onset in onsets:
        mask = (time_ms >= onset) & (time_ms <= onset + duration + float(contract["response_exclusion_ms"]))
        event_min.append(float(np.min(q[mask, :2])))
    returns = [
        _crossings(time_ms, fast[:, patch], float(contract["section_level_khz"]), float(contract["rearm_level_khz"]))
        for patch in range(2)
    ]
    # Search every response-excluded event interval with the same rule.  This is
    # deliberately independent of the q-fold label: an early fork starts below
    # the fold, but must still fail if its challenge regenerates a burst train.
    window_rows: list[dict[str, Any]] = []
    for event_index, onset in enumerate(onsets, start=1):
        start = float(onset + duration + float(contract["response_exclusion_ms"]))
        stop = float(onsets[event_index]) if event_index < len(onsets) else float(time_ms[-1]) + 1.0
        aligned_window = [[value for value in values if start <= value < stop] for values in returns]
        pairs_window = _pair_crossings(
            aligned_window[0], aligned_window[1], float(contract["pairing_max_lag_ms"]),
        )
        paired_window = [max(pair) for pair in pairs_window]
        joint_down_window = None
        low_onset_window = None
        if len(pairs_window) >= int(contract["minimum_paired_returns"]):
            last_pair = paired_window[-1]
            downs = [
                [value for value in _downcrossings(time_ms, fast[:, patch], float(contract["section_level_khz"])) if value >= last_pair]
                for patch in range(2)
            ]
            if all(values for values in downs):
                joint_down_window = max(downs[0][0], downs[1][0])
                low_onset_window = _first_sustained_low(
                    time_ms, np.c_[rate[:, :2], fast[:, :2]],
                    float(contract["low_threshold_khz"]),
                    float(contract["sustained_low_ms"]), joint_down_window,
                )
        window_rows.append({
            "trigger_event_index": event_index,
            "analysis_start_ms": start, "analysis_stop_ms": stop,
            "core_returns_ms": aligned_window[0],
            "annulus_returns_ms": aligned_window[1],
            "paired_return_times_ms": paired_window,
            "paired_returns": len(pairs_window),
            "joint_last_downcross_ms": joint_down_window,
            "sustained_all_low_onset_ms": low_onset_window,
        })
    qualifying = [
        row for row in window_rows
        if row["paired_returns"] >= int(contract["minimum_paired_returns"])
        and row["joint_last_downcross_ms"] is not None
        and row["sustained_all_low_onset_ms"] is not None
    ]
    selected = qualifying[0] if qualifying else max(window_rows, key=lambda row: row["paired_returns"])
    aligned = [selected["core_returns_ms"], selected["annulus_returns_ms"]]
    paired_times = selected["paired_return_times_ms"]
    joint_down = selected["joint_last_downcross_ms"]
    low_onset = selected["sustained_all_low_onset_ms"]
    event6_returns = [[value for value in values if value >= float(onsets[5])] for values in returns]
    evoked_response_crossings = []
    for onset in onsets:
        stop = float(onset + duration + float(contract["response_exclusion_ms"]))
        evoked_response_crossings.append([
            sum(float(onset) <= value <= stop for value in returns[patch])
            for patch in range(2)
        ])
    numeric_clean = (
        _numeric_clean(result)
        if numeric_clean_override is None else bool(numeric_clean_override)
    )
    tail_start = float(time_ms[-1] - float(contract["sustained_low_ms"]))
    tail = time_ms >= tail_start
    final_sustained_low = bool(
        np.any(tail)
        and np.all(np.c_[rate[tail, :2], fast[tail, :2]] <= float(contract["low_threshold_khz"]))
    )
    candidate = bool(qualifying and numeric_clean)
    return {
        "entry_event_index": entry_event, "entry_time_ms": entry_time,
        "event_min_regional_q": event_min,
        "core_autonomous_returns_ms": aligned[0], "annulus_autonomous_returns_ms": aligned[1],
        "paired_return_times_ms": paired_times, "paired_returns": int(selected["paired_returns"]),
        "joint_last_downcross_ms": joint_down, "sustained_all_low_onset_ms": low_onset,
        "event6_no_section_crossing": not event6_returns[0] and not event6_returns[1],
        "event6_return_times_ms": event6_returns, "numeric_clean": numeric_clean,
        "all_core_section_crossings_ms": returns[0],
        "all_annulus_section_crossings_ms": returns[1],
        "evoked_response_crossing_counts_core_annulus": evoked_response_crossings,
        "evoked_section_crossings_present": bool(
            sum(sum(row) for row in evoked_response_crossings) > 0
        ),
        "actual_entry_lifecycle_candidate": candidate,
        "candidate_trigger_event_index": qualifying[0]["trigger_event_index"] if qualifying else None,
        "event_window_classification": window_rows,
        "final_sustained_low": final_sustained_low,
        "bounded_high_or_runaway_tail": not final_sustained_low,
        "events_1_to_4_above_fold": bool(all(value >= fold for value in event_min[:4])),
        "event_5_first_entry": entry_event == 5,
    }


def _early_gates(classification: dict[str, Any]) -> dict[str, bool]:
    """Fail closed: no-cycle is insufficient without a finite supported low tail."""

    return {
        "numeric_clean": bool(classification["numeric_clean"]),
        "no_lifecycle_candidate": not bool(classification["actual_entry_lifecycle_candidate"]),
        "finite_supported_sustained_low_tail": bool(
            classification["numeric_clean"] and classification["final_sustained_low"]
        ),
        "not_bounded_high_or_runaway": not bool(classification["bounded_high_or_runaway_tail"]),
    }


def _late_gates(classification: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    """Require both a finite exit and a persistently low final challenge tail."""

    return {
        "numeric_clean": bool(classification["numeric_clean"]),
        "events_1_to_4_above_fold": bool(classification["events_1_to_4_above_fold"]),
        "event_5_first_entry": bool(classification["event_5_first_entry"]),
        "at_least_four_paired_returns": classification["paired_returns"] >= int(
            cfg["source_phenotype"]["minimum_paired_returns"]
        ),
        "event_6_no_section_crossing": bool(classification["event6_no_section_crossing"]),
        "finite_low_exit_recurs": bool(classification["actual_entry_lifecycle_candidate"]),
        "final_sustained_low": bool(classification["final_sustained_low"]),
        "not_bounded_high_or_runaway_tail": not bool(
            classification["bounded_high_or_runaway_tail"]
        ),
    }


def _trace_payload(result: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "time_ms": np.asarray(result["time_ms"]),
        "rE_khz": np.asarray(result["rE"][:, 0]),
        "rE_fast_khz": np.asarray(result["rE_fast"][:, 0]),
        "q": np.asarray(result["z"][:, 0]), "p": np.asarray(result["p"][:, 0]),
        "m": np.asarray(result["m"][:, 0]), "z_use": np.asarray(result["z_use"][:, 0]),
        "occupancy": np.asarray(result["occupancy"][:, 0]),
        "latch": np.asarray(result["latch"][:, 0]),
        "final_state": np.asarray(result["final_state"][0]),
        "final_latch_state": np.asarray(result["final_latch_state"][0]),
    }


def _fast_distance(state: np.ndarray, reference: np.ndarray) -> float:
    mask = np.r_[np.arange(0, 21), np.arange(30, 32)]
    return float(np.linalg.norm(np.asarray(state)[mask] - np.asarray(reference)[mask]))


def _run_one_dt(
    *, dt_ms: float, checkpoint_state: np.ndarray, checkpoint_latch: np.ndarray,
    runtime: dict[str, Any], cfg: dict[str, Any], source: dict[str, Any], output: Path,
) -> dict[str, Any]:
    arm: RegionalSlowParameters = runtime["arm"]
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    state = np.asarray(checkpoint_state, dtype=float).copy()
    latch = np.asarray(checkpoint_latch, dtype=bool).copy()
    absolute_time = float(cfg["checkpoint"]["time_ms"])
    _endpoint(rows, dt_ms, "checkpoint", "source_double", absolute_time, state, latch)
    checkpoint = _checkpoint_gates(state, latch, runtime, cfg)
    for key, value in checkpoint["gates"].items():
        gate_rows.append({"dt_ms": dt_ms, "segment": "checkpoint", "gate": key, "pass": value})
    if not checkpoint["accepted"]:
        return {"pass": False, "status": CHECKPOINT_INVALID, "checkpoint": checkpoint, "endpoint_rows": rows, "gate_rows": gate_rows}

    # Segment B: analytic zero-use intervals interrupted by chained full sentinels.
    target_q = float(cfg["checkpoint"]["q_reset_threshold"]) - float(cfg["protected_recovery"]["q_stop_epsilon"])
    pooled = _pooled(state)
    rate = (1.0 - pooled["m"]) / arm.tau_z_recovery_ms + pooled["m"] / float(arm.tau_z_fast_recovery_ms)
    predicted = float(-np.log((arm.z_rest - target_q) / (arm.z_rest - pooled["q"])) / rate)
    protected_elapsed = 0.0
    protected_sentinels: list[dict[str, Any]] = []
    for fraction in cfg["protected_recovery"]["sentinel_fractions"]:
        nominal = float(fraction) * predicted
        gap = nominal - protected_elapsed
        if gap <= 0.0:
            return {"pass": False, "status": PROTECTED_NO_GO, "reason": "nonpositive protected analytic gap", "endpoint_rows": rows, "gate_rows": gate_rows}
        state = _advance_active_state(state, gap, arm)
        absolute_time += gap
        protected_elapsed += gap
        _endpoint(rows, dt_ms, "B_protected", f"analytic_{fraction:.2f}", absolute_time, state, latch)
        sentinel = _integrate(
            state, latch, runtime, cfg, dt_ms,
            float(cfg["protected_recovery"]["sentinel_duration_ms"]), (),
        )
        gates = _sentinel_gates(
            sentinel, expected_latch=True,
            settling_ms=float(cfg["protected_recovery"]["sentinel_settling_ms"]),
            zero_tolerance=float(cfg["checkpoint"]["sensor_zero_tolerance"]),
        )
        protected_sentinels.append({"fraction": float(fraction), "gates": gates})
        for key, value in gates.items():
            gate_rows.append({"dt_ms": dt_ms, "segment": f"B_sentinel_{fraction:.2f}", "gate": key, "pass": value})
        if not all(gates.values()):
            return {"pass": False, "status": PROTECTED_NO_GO, "checkpoint": checkpoint, "protected_sentinels": protected_sentinels, "endpoint_rows": rows, "gate_rows": gate_rows}
        state = np.asarray(sentinel["final_state"][0], dtype=float)
        latch = np.asarray(sentinel["final_latch_state"][0], dtype=bool)
        duration = float(cfg["protected_recovery"]["sentinel_duration_ms"])
        absolute_time += duration
        protected_elapsed += duration
        _endpoint(rows, dt_ms, "B_protected", f"full_{fraction:.2f}", absolute_time, state, latch)
    pooled = _pooled(state)
    if pooled["q"] > target_q + 1.0e-10:
        return {"pass": False, "status": PROTECTED_NO_GO, "reason": "full sentinels overshot protected q stop", "endpoint_rows": rows, "gate_rows": gate_rows}
    remaining = float(-np.log((arm.z_rest - target_q) / (arm.z_rest - pooled["q"])) / (
        (1.0 - pooled["m"]) / arm.tau_z_recovery_ms + pooled["m"] / float(arm.tau_z_fast_recovery_ms)
    ))
    state = _advance_active_state(state, remaining, arm)
    absolute_time += remaining
    protected_elapsed += remaining
    _endpoint(rows, dt_ms, "B_protected", "analytic_q_0p8845", absolute_time, state, latch)
    bridge_gate = bool(abs(_pooled(state)["q"] - target_q) <= 2.0e-12 and np.all(latch[:2]))
    gate_rows.append({"dt_ms": dt_ms, "segment": "B_protected", "gate": "stops_just_below_q_reset", "pass": bridge_gate})
    if not bridge_gate:
        return {"pass": False, "status": PROTECTED_NO_GO, "endpoint_rows": rows, "gate_rows": gate_rows}

    # Segment C: only the real state machine can release the latch.
    reset_times: list[float] = []
    set_times: list[float] = []
    reset_return_count = 0
    reset_trace_parts: list[dict[str, np.ndarray]] = []
    deadline = float(source["termination_complete_ms"]) + float(cfg["latch_reset"]["max_from_segment_a_termination_ms"])
    while absolute_time < deadline and not reset_times:
        duration = min(float(cfg["latch_reset"]["integration_chunk_ms"]), deadline - absolute_time)
        if duration <= dt_ms:
            break
        chunk = _integrate(state, latch, runtime, cfg, dt_ms, duration, ())
        if not _numeric_clean(chunk):
            return {"pass": False, "status": NUMERIC_UNRESOLVED, "reason": "numeric failure during latch reset", "endpoint_rows": rows, "gate_rows": gate_rows}
        local_resets = [absolute_time + float(value) for value in chunk["latch_reset_times_ms"][0]]
        local_sets = [absolute_time + float(value) for value in chunk["latch_set_times_ms"][0]]
        reset_times.extend(local_resets)
        set_times.extend(local_sets)
        reset_return_count += sum(len(chunk["return_times_ms"][0][patch]) for patch in range(2))
        payload = _trace_payload(chunk)
        payload["time_ms"] = payload["time_ms"] + absolute_time
        reset_trace_parts.append(payload)
        state = np.asarray(chunk["final_state"][0], dtype=float)
        latch = np.asarray(chunk["final_latch_state"][0], dtype=bool)
        absolute_time += duration
        _endpoint(rows, dt_ms, "C_reset", "full_chunk", absolute_time, state, latch)
    transition_values: dict[str, Any] = {}
    if reset_times and reset_trace_parts:
        trace = reset_trace_parts[-1]
        reset_index = int(np.flatnonzero(~np.any(trace["latch"][:, :2].astype(bool), axis=1))[0])
        transition_values = {
            "q_min": float(np.min(trace["q"][reset_index, :2])),
            "p_max": float(np.max(trace["p"][reset_index, :2])),
            "rE_fast_max_khz": float(np.max(trace["rE_fast_khz"][reset_index, :2])),
        }
    reset_gates = {
        "exactly_one_true_to_false": len(reset_times) == 1,
        "no_reset_chatter_or_latch_reset": len(set_times) == 0 and not bool(np.any(latch[:2])),
        "reset_before_deadline": len(reset_times) == 1 and reset_times[0] <= deadline,
        "transition_fast_low": bool(transition_values) and transition_values["rE_fast_max_khz"] <= float(cfg["latch_reset"]["regional_fast_max_khz"]),
        "transition_q_safe": bool(transition_values) and transition_values["q_min"] >= float(cfg["latch_reset"]["q_min"]),
        "transition_p_low": bool(transition_values) and transition_values["p_max"] <= float(cfg["latch_reset"]["p_max"]),
        "no_section_return": reset_return_count == 0,
    }
    for key, value in reset_gates.items():
        gate_rows.append({"dt_ms": dt_ms, "segment": "C_reset", "gate": key, "pass": value})
    if not all(reset_gates.values()):
        return {"pass": False, "status": RESET_NO_GO, "reset_times_ms": reset_times, "reset_transition": transition_values, "reset_gates": reset_gates, "endpoint_rows": rows, "gate_rows": gate_rows}

    # Segment D: exact released-latch slow bridge, interrupted at registered A sentinels.
    release_sentinels: list[dict[str, Any]] = []
    monotonic_q = True
    monotonic_m = True
    for additive in cfg["natural_release"]["additive_sentinels_mv"]:
        pooled = _pooled(state)
        target_m = float(additive) / float(runtime["parameters"].additive_max_mv)
        if pooled["m"] <= target_m:
            return {"pass": False, "status": SAME_BASIN_NO_GO, "reason": f"A sentinel {additive} already passed", "endpoint_rows": rows, "gate_rows": gate_rows}
        gap = float(arm.tau_m_down_ms * np.log(pooled["m"] / target_m))
        before = pooled
        state = _advance_released_state(state, gap, arm)
        absolute_time += gap
        after = _pooled(state)
        monotonic_q &= after["q"] >= before["q"] - 1.0e-12
        monotonic_m &= after["m"] <= before["m"] + 1.0e-12
        _endpoint(rows, dt_ms, "D_release", f"analytic_A_{additive}", absolute_time, state, latch)
        sentinel = _integrate(
            state, latch, runtime, cfg, dt_ms,
            float(cfg["natural_release"]["sentinel_duration_ms"]), (),
        )
        gates = _sentinel_gates(
            sentinel, expected_latch=False,
            settling_ms=float(cfg["natural_release"]["sentinel_settling_ms"]),
            zero_tolerance=float(cfg["checkpoint"]["sensor_zero_tolerance"]),
        )
        q_trace = np.asarray(sentinel["z"][:, 0, :2], dtype=float)
        m_trace = np.asarray(sentinel["m"][:, 0, :2], dtype=float)
        gates["q_nondecreasing"] = bool(np.min(np.diff(q_trace, axis=0)) >= -1.0e-9)
        gates["m_nonincreasing"] = bool(np.max(np.diff(m_trace, axis=0)) <= 1.0e-9)
        gates["no_latch_reactivation"] = not sentinel["latch_set_times_ms"][0]
        release_sentinels.append({"additive_mv": float(additive), "gates": gates})
        for key, value in gates.items():
            gate_rows.append({"dt_ms": dt_ms, "segment": f"D_sentinel_A_{additive}", "gate": key, "pass": value})
        if not all(gates.values()):
            return {"pass": False, "status": SAME_BASIN_NO_GO, "release_sentinels": release_sentinels, "endpoint_rows": rows, "gate_rows": gate_rows}
        before = _pooled(state)
        state = np.asarray(sentinel["final_state"][0], dtype=float)
        latch = np.asarray(sentinel["final_latch_state"][0], dtype=bool)
        after = _pooled(state)
        monotonic_q &= after["q"] >= before["q"] - 1.0e-9
        monotonic_m &= after["m"] <= before["m"] + 1.0e-9
        absolute_time += float(cfg["natural_release"]["sentinel_duration_ms"])
        _endpoint(rows, dt_ms, "D_release", f"full_A_{additive}", absolute_time, state, latch)
    pooled = _pooled(state)
    target_q_same = float(cfg["natural_release"]["q_same_basin_min"])
    q_bridge_duration = 0.0
    if pooled["q"] < target_q_same:
        q_bridge_duration = _solve_released_duration_for_q(
            q=pooled["q"], m=pooled["m"], target_q=target_q_same, q_rest=arm.z_rest,
            tau_m_ms=arm.tau_m_down_ms, tau_slow_ms=arm.tau_z_recovery_ms,
            tau_fast_ms=float(arm.tau_z_fast_recovery_ms),
        )
        state = _advance_released_state(state, q_bridge_duration, arm)
        absolute_time += q_bridge_duration
        _endpoint(rows, dt_ms, "D_release", "analytic_q_0p899", absolute_time, state, latch)
    final_start_state = state.copy()
    final = _integrate(
        state, latch, runtime, cfg, dt_ms,
        float(cfg["natural_release"]["final_full_duration_ms"]), (),
    )
    final_state = np.asarray(final["final_state"][0], dtype=float)
    final_latch = np.asarray(final["final_latch_state"][0], dtype=bool)
    absolute_time += float(cfg["natural_release"]["final_full_duration_ms"])
    _endpoint(rows, dt_ms, "D_same_basin", "final_4s_full", absolute_time, final_state, final_latch)
    fast_rhs, _ = patch_rhs_fast_and_moments(final_state, runtime["prepared"], runtime["transfer"])
    fast_rhs_norm = float(np.max(np.abs(fast_rhs)))
    distance_start = _fast_distance(final_start_state, runtime["low_state"])
    distance_end = _fast_distance(final_state, runtime["low_state"])
    final_rates_low = bool(
        np.max(final["rE"][:, 0, :2]) < 0.005 and np.max(final["rE_fast"][:, 0, :2]) < 0.005
    )
    final_values = _pooled(final_state)
    same_basin_gates = {
        "numeric_clean": _numeric_clean(final),
        "latch_off": not bool(np.any(final_latch[:2])),
        "q_at_least_0p899": final_values["q"] >= target_q_same,
        "A_at_most_0p002": final_values["A_mv"] <= float(cfg["natural_release"]["additive_same_basin_max_mv"]) + 1.0e-12,
        "p_at_most_0p001": final_values["p"] <= float(cfg["natural_release"]["p_same_basin_max"]),
        "LLL_low_rates_no_return": final_rates_low and not final["return_times_ms"][0][0] and not final["return_times_ms"][0][1],
        "fast_vector_field_below_1e_8": fast_rhs_norm <= float(cfg["natural_release"]["final_fast_rhs_max_per_ms"]),
        "fast_distance_to_original_root_decreases": distance_end < distance_start,
        "q_nondecreasing_M_nonincreasing": monotonic_q and monotonic_m and bool(
            np.min(np.diff(final["z"][:, 0, :2], axis=0)) >= -1.0e-9
            and np.max(np.diff(final["m"][:, 0, :2], axis=0)) <= 1.0e-9
        ),
        "no_latch_reactivation": not final["latch_set_times_ms"][0],
    }
    for key, value in same_basin_gates.items():
        gate_rows.append({"dt_ms": dt_ms, "segment": "D_same_basin", "gate": key, "pass": value})
    if not all(same_basin_gates.values()):
        return {
            "pass": False, "status": SAME_BASIN_NO_GO, "same_basin_gates": same_basin_gates,
            "fast_rhs_norm_per_ms": fast_rhs_norm, "fast_distance_start": distance_start,
            "fast_distance_end": distance_end, "endpoint_rows": rows, "gate_rows": gate_rows,
        }

    # Common-classifier forks.  Early starts from the immutable 20-s checkpoint;
    # late starts from the naturally recovered state above.
    early = _integrate(checkpoint_state, checkpoint_latch, runtime, cfg, dt_ms, float(cfg["retrigger"]["duration_ms"]), _pulses(cfg))
    early_class = _classify_challenge(early, cfg)
    early_gates = _early_gates(early_class)
    for key, value in early_gates.items():
        gate_rows.append({"dt_ms": dt_ms, "segment": "early_retrigger", "gate": key, "pass": value})
    if not all(early_gates.values()):
        return {"pass": False, "status": EARLY_NO_GO, "early": early_class, "endpoint_rows": rows, "gate_rows": gate_rows}
    late = _integrate(final_state, final_latch, runtime, cfg, dt_ms, float(cfg["retrigger"]["duration_ms"]), _pulses(cfg))
    late_class = _classify_challenge(late, cfg)
    late_gates = _late_gates(late_class, cfg)
    for key, value in late_gates.items():
        gate_rows.append({"dt_ms": dt_ms, "segment": "late_retrigger", "gate": key, "pass": value})
    if not all(late_gates.values()):
        return {"pass": False, "status": LATE_NO_GO, "early": early_class, "late": late_class, "endpoint_rows": rows, "gate_rows": gate_rows}

    trace_path = output / f"representative_traces_dt{str(dt_ms).replace('.', 'p')}.npz"
    early_payload, late_payload, final_payload = _trace_payload(early), _trace_payload(late), _trace_payload(final)
    np.savez_compressed(
        trace_path,
        early_time_ms=early_payload["time_ms"], early_rE_khz=early_payload["rE_khz"],
        early_rE_fast_khz=early_payload["rE_fast_khz"], early_q=early_payload["q"],
        early_m=early_payload["m"], early_latch=early_payload["latch"],
        late_time_ms=late_payload["time_ms"], late_rE_khz=late_payload["rE_khz"],
        late_rE_fast_khz=late_payload["rE_fast_khz"], late_q=late_payload["q"],
        late_m=late_payload["m"], late_latch=late_payload["latch"],
        final_time_ms=final_payload["time_ms"], final_rE_khz=final_payload["rE_khz"],
        final_rE_fast_khz=final_payload["rE_fast_khz"], final_q=final_payload["q"],
        final_m=final_payload["m"], final_latch=final_payload["latch"],
    )
    trace_bytes = int(trace_path.stat().st_size)
    trace_gate = trace_bytes <= int(cfg["integration"]["max_trace_bytes"])
    gate_rows.append({"dt_ms": dt_ms, "segment": "resource", "gate": "saved_trace_below_64_mib", "pass": trace_gate})
    return {
        "pass": trace_gate, "status": SUPPORTED if trace_gate else NUMERIC_UNRESOLVED,
        "dt_ms": dt_ms, "checkpoint": checkpoint, "protected_predicted_duration_ms": predicted,
        "protected_actual_duration_ms": protected_elapsed, "protected_sentinels": protected_sentinels,
        "reset_times_ms": reset_times, "reset_transition": transition_values, "reset_gates": reset_gates,
        "release_sentinels": release_sentinels, "q_release_bridge_duration_ms": q_bridge_duration,
        "same_basin_gates": same_basin_gates, "same_basin_values": final_values,
        "fast_rhs_norm_per_ms": fast_rhs_norm, "fast_distance_start": distance_start,
        "fast_distance_end": distance_end, "same_basin_absolute_time_ms": absolute_time,
        "early": early_class, "early_gates": early_gates, "late": late_class,
        "late_gates": late_gates, "endpoint_rows": rows, "gate_rows": gate_rows,
        "trace_path": str(trace_path.relative_to(ROOT)), "trace_file_bytes": trace_bytes,
        "_plot": {"early": early_payload, "late": late_payload, "final": final_payload},
    }


def _dt_agreement(base: dict[str, Any], half: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    tolerance = float(cfg["source_phenotype"]["pairing_max_lag_ms"])
    base_pairs = np.asarray(base["late"]["paired_return_times_ms"], dtype=float)
    half_pairs = np.asarray(half["late"]["paired_return_times_ms"], dtype=float)
    return {
        "source_phenotype_common_hash_locked": True,
        "both_reset_once": len(base["reset_times_ms"]) == len(half["reset_times_ms"]) == 1,
        "both_same_basin": all(base["same_basin_gates"].values()) and all(half["same_basin_gates"].values()),
        "both_early_no_lifecycle_candidate": not base["early"]["actual_entry_lifecycle_candidate"] and not half["early"]["actual_entry_lifecycle_candidate"],
        "both_late_event5_lifecycle": base["late"]["event_5_first_entry"] and half["late"]["event_5_first_entry"] and base["late"]["actual_entry_lifecycle_candidate"] and half["late"]["actual_entry_lifecycle_candidate"],
        "late_entry_times_within_20ms": abs(float(base["late"]["entry_time_ms"]) - float(half["late"]["entry_time_ms"])) <= tolerance,
        "paired_crossing_times_within_20ms": base_pairs.size == half_pairs.size and bool(np.all(np.abs(base_pairs - half_pairs) <= tolerance)),
        "reset_times_within_20ms": abs(float(base["reset_times_ms"][0]) - float(half["reset_times_ms"][0])) <= tolerance,
    }


def _strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_") and key not in {"endpoint_rows", "gate_rows"}}


def _plot(output: Path, source_arrays: dict[str, np.ndarray], source: dict[str, Any], runs: list[dict[str, Any]], status: str) -> Path:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    base = runs[0]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    time = np.asarray(source_arrays["time_ms"], dtype=float) / 1000.0
    fast = 1000.0 * np.asarray(source_arrays["rE_fast_khz"], dtype=float)
    mask = (time >= 7.45) & (time <= 11.25)
    ax = axes[0, 0]
    ax.plot(time[mask], fast[mask, 0], color=PATCH_COLORS[0], lw=1.0, label="core")
    ax.plot(time[mask], fast[mask, 1], color=PATCH_COLORS[1], lw=1.0, label="annulus")
    for value in source["paired_return_times_ms"][:4]:
        ax.axvline(value / 1000.0, color="0.45", lw=0.55)
    ax.axvline(10.915, color="#762A83", ls="--", lw=0.9, label="event 6")
    ax.set(xlabel="source time (s)", ylabel="rE fast (Hz)", title="A  Event-5 entry and four paired returns")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    endpoint = base["endpoint_rows"]
    et = np.asarray([row["absolute_time_ms"] for row in endpoint], dtype=float) / 1000.0
    eq = np.asarray([row["q"] for row in endpoint], dtype=float)
    ea = np.asarray([row["A_mv"] for row in endpoint], dtype=float)
    ax.plot(et, eq, "o-", ms=3, lw=1.0, color="#2166AC", label="q")
    ax.axhline(0.885, color="#2166AC", ls="--", lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(et, ea, "s-", ms=2.5, lw=0.9, color="#B2182B", label="A")
    ax.set(xlabel="hybrid absolute time (s)", ylabel="q", title="B  Protected reset and natural release")
    ax2.set_ylabel("A (mV)")
    lines = ax.get_lines()[:1] + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7)

    ax = axes[0, 2]
    reset = base["reset_times_ms"][0] / 1000.0
    reset_rows = [
        row for row in endpoint
        if row["segment"] == "C_reset" or row["kind"] == "analytic_q_0p8845"
    ]
    rt = np.asarray([row["absolute_time_ms"] for row in reset_rows], dtype=float) / 1000.0
    rq = np.asarray([row["q"] for row in reset_rows], dtype=float)
    ax.plot(rt, rq, "o-", color="#2166AC", lw=1.2, ms=4, label="q")
    ax.axhline(0.885, color="#2166AC", ls="--", lw=0.8, label="q reset")
    ax.axvline(reset, color="black", ls=":", lw=1.0, label="true->false")
    ax2 = ax.twinx()
    ax2.plot(
        [rt[0], reset, reset, rt[-1]], [1.0, 1.0, 0.0, 0.0],
        color="#B2182B", lw=1.1, label="latch",
    )
    ax.set(xlabel="hybrid absolute time (s)", ylabel="q", title=f"C  Real latch reset at {reset:.3f} s")
    ax2.set(ylim=(-0.05, 1.05), yticks=[0, 1], yticklabels=["off", "on"])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7, loc="lower right")

    ax = axes[1, 0]
    labels = ["reset", "same basin", "early no cycle", "late cycle"]
    for index, run in enumerate(runs):
        values = [
            all(run["reset_gates"].values()), all(run["same_basin_gates"].values()),
            not run["early"]["actual_entry_lifecycle_candidate"], run["late"]["actual_entry_lifecycle_candidate"],
        ]
        ax.scatter(np.arange(4), np.full(4, index), c=["#1B7837" if value else "#B2182B" for value in values], s=90)
    ax.set(xticks=np.arange(4), xticklabels=labels, yticks=np.arange(len(runs)), yticklabels=[f"dt={run['dt_ms']}" for run in runs], xlim=(-0.5, 3.5), title="D  Base/half-dt gate labels")

    ax = axes[1, 1]
    early = base["_plot"]["early"]
    late = base["_plot"]["late"]
    ax.plot(early["time_ms"] / 1000.0, 1000.0 * early["rE_fast_khz"][:, 0], color="#2166AC", lw=0.8, label="protected fork")
    ax.plot(late["time_ms"] / 1000.0, 1000.0 * late["rE_fast_khz"][:, 0], color="#B2182B", lw=0.8, label="recovered fork")
    ax.set(xlabel="challenge time (s)", ylabel="core rE fast (Hz)", title="E  Evoked responses remain; autonomous lifecycle differs")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    verdict = [
        "F  R4 center verdict", "", status,
        f"source entry / paired returns: event {source['entry_event_index']} / {len(source['paired_return_times_ms'])}",
        f"base reset: {base['reset_times_ms'][0]:.3f} ms",
        f"base same-basin q / A: {base['same_basin_values']['q']:.6f} / {base['same_basin_values']['A_mv']:.6f}",
        f"early candidate: {base['early']['actual_entry_lifecycle_candidate']}",
        f"late entry / paired: {base['late']['entry_event_index']} / {base['late']['paired_returns']}",
        "", "Center-point regional hybrid result only.",
        "No q fit, E-E change, conductance, or relay.",
    ]
    ax.text(0.0, 1.0, "\n".join(verdict), va="top", family="monospace", fontsize=8.2)
    fig.suptitle("R4 actual-entry-aligned regional lifecycle closure", fontsize=13, fontweight="bold")
    path = figures / "mz_actual_entry_lifecycle_closure.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run(config_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    hashes, source_summary, source_arrays, source_cfg = _validate_inputs(cfg)
    source = _source_phenotype(source_arrays, source_summary, cfg)
    if not source["accepted"]:
        summary = {"status": SOURCE_MISMATCH, "decision": "stop_before_integration", "source_phenotype": source, "input_sha256": hashes}
        _save_json(output / "actual_entry_lifecycle_closure_summary.json", summary)
        return summary
    runtime = _build_runtime(source_cfg)
    checkpoint_state = np.asarray(source_arrays["final_state"], dtype=float)
    checkpoint_latch = np.asarray(source_arrays["final_latch_state"], dtype=bool)
    runs: list[dict[str, Any]] = []
    for index, dt_ms in enumerate(cfg["integration"]["dt_values_ms"]):
        if index == 1 and (not runs or not runs[0]["pass"]):
            break
        elapsed = time.perf_counter() - start
        if elapsed >= float(cfg["resource_contract"]["max_total_wall_seconds"]):
            break
        result = _run_one_dt(
            dt_ms=float(dt_ms), checkpoint_state=checkpoint_state,
            checkpoint_latch=checkpoint_latch, runtime=runtime, cfg=cfg,
            source=source, output=output,
        )
        result["dt_ms"] = float(dt_ms)
        result["runtime_seconds"] = float(time.perf_counter() - start - elapsed)
        runs.append(result)
        if not result["pass"]:
            break
    agreement: dict[str, bool] = {}
    if len(runs) == 2 and all(run["pass"] for run in runs):
        agreement = _dt_agreement(runs[0], runs[1], cfg)
    total_runtime = time.perf_counter() - start
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    trace_paths = [ROOT / run["trace_path"] for run in runs if run.get("trace_path")]
    resource_gates = {
        "trace_files_below_64_mib": bool(trace_paths) and all(path.stat().st_size <= int(cfg["integration"]["max_trace_bytes"]) for path in trace_paths),
        "peak_rss_below_1p5_gib": peak_rss < float(cfg["resource_contract"]["max_memory_gib"]) * 1024 * 1024,
        "total_wall_below_20min": total_runtime < float(cfg["resource_contract"]["max_total_wall_seconds"]),
        "single_process_single_blas": True,
    }
    if not runs:
        status, decision = NUMERIC_UNRESOLVED, "stop_before_base_dt_for_wall_contract"
    elif not runs[0]["pass"]:
        status, decision = runs[0]["status"], "stop_after_base_dt_fail_closed"
    elif len(runs) < 2:
        status, decision = NUMERIC_UNRESOLVED, "base_passed_but_half_dt_not_completed"
    elif not runs[1]["pass"]:
        status, decision = runs[1]["status"], "stop_after_half_dt_fail_closed"
    elif not all(agreement.values()):
        status, decision = DT_NO_GO, "stop_for_base_half_dt_label_or_crossing_disagreement"
    elif not all(resource_gates.values()):
        status, decision = NUMERIC_UNRESOLVED, "stop_for_resource_contract_violation"
    else:
        status, decision = SUPPORTED, "unlock_fully_coupled_q_map_recalibration_and_local_robustness_only"
    all_endpoint_rows = [row for run in runs for row in run.get("endpoint_rows", [])]
    all_gate_rows = [row for run in runs for row in run.get("gate_rows", [])]
    if all_endpoint_rows:
        _save_csv(output / "hybrid_endpoint_table.csv", all_endpoint_rows)
    if all_gate_rows:
        _save_csv(output / "closure_gate_table.csv", all_gate_rows)
    figure = None
    if runs and runs[0].get("pass") and runs[0].get("_plot"):
        figure = _plot(output, source_arrays, source, runs, status)
        (output / "figures/README.md").write_text(
            "### mz_actual_entry_lifecycle_closure.png\n\n"
            "这张 2x3 图按 R4 锁定合同展示 actual-entry-aligned center closure。A 从 hash-lock 的 R3 trace 重算 event-5 entry、四次 core/annulus 配对回返和 event-6 suppression；B–C 展示从 20 s double checkpoint 出发的 protected q recovery、真实 latch reset、M 自然衰减和同一低态回归；D 对照 base/half-dt gate label；E 用完全相同的六事件 classifier 比较 early 与 late fork；F 给出边界化 verdict。\n\n"
            "图中的长时间空档是经过 500 ms full sentinel 认证的 zero-use analytic bridge，不是逐点快变量积分，也没有人工把 q/M 赋回根。protected fork 来自原始 20 s checkpoint，recovered fork 来自自然恢复 checkpoint。protected fork 仍保留六次事件诱发的 section crossings；它通过的是 response-excluded window 内没有四次自主配对回返、且末段回到持续低态，而不是 electrical silence。\n\n"
            "**关注点**：真实 state machine 是否只 reset 一次、最终 q/A/p 与 fast vector field 是否回到 LLL basin、protected challenge 是否仅有 evoked responses 而没有 autonomous lifecycle、recovered challenge 是否重新出现 event-5 entry 和至少四次配对 burst。该结果只属于 fixed-bath regional hybrid center，不代表零输入自发 onset、连续空间 wavefront 或 full SNN。\n",
            encoding="utf-8",
        )
    serial_runs = [_strip_private(run) for run in runs]
    canonical_output_paths = [
        *trace_paths,
        *[
            path for path in (
                output / "closure_gate_table.csv",
                output / "hybrid_endpoint_table.csv",
            ) if path.is_file()
        ],
    ]
    output_hashes = (
        _hash_manifest(canonical_output_paths) if canonical_output_paths else {}
    )
    summary = {
        "status": status, "decision": decision,
        "scientific_layer": "actual_entry_aligned_regional_hybrid_center_closure_fixed_bath",
        "source_phenotype": source, "dt_runs": serial_runs,
        "base_half_dt_agreement": agreement, "runtime_seconds": float(total_runtime),
        "peak_rss_kib": peak_rss, "resource_gates": resource_gates,
        "input_sha256": hashes,
        "output_sha256": output_hashes,
        "claim_boundary": [
            "the fixed six-event sequence triggers entry; zero-input spontaneous onset was not tested",
            "the result is one center point under an imposed fixed bath mask, not a robust q corridor or continuous spatial wavefront",
            "analytic gaps were used only after full sentinels certified zero use and occupancy; q, p, M, and latch were never manually reset",
            "no E-E weight, kernel, delay, recurrent saturation, conductance membrane, or relay variable was changed",
            "support does not imply Hopf, SNIC, torus, or a smooth full-system limit cycle",
        ],
        "artifacts": {
            "summary": str((output / "actual_entry_lifecycle_closure_summary.json").relative_to(ROOT)),
            "gate_csv": str((output / "closure_gate_table.csv").relative_to(ROOT)) if all_gate_rows else None,
            "endpoint_csv": str((output / "hybrid_endpoint_table.csv").relative_to(ROOT)) if all_endpoint_rows else None,
            "figure_png": None if figure is None else str(figure.relative_to(ROOT)),
            "figure_pdf": None if figure is None else str(figure.with_suffix(".pdf").relative_to(ROOT)),
            "trace_npz": [str(path.relative_to(ROOT)) for path in trace_paths],
        },
        "config": cfg,
    }
    _save_json(output / "actual_entry_lifecycle_closure_summary.json", summary)
    return summary


def refresh_existing_reporting(config_path: Path) -> dict[str, Any]:
    """Reapply the common classifier/figure to immutable canonical traces."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    _, source_summary, source_arrays, _ = _validate_inputs(cfg)
    source = _source_phenotype(source_arrays, source_summary, cfg)
    output = ROOT / cfg["result_root"]
    summary_path = output / "actual_entry_lifecycle_closure_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != SUPPORTED or not source["accepted"]:
        raise RuntimeError("reporting refresh requires the supported immutable canonical closure")
    manifest = summary.get("output_sha256", {})
    expected_manifest_paths = {
        *(str(run["trace_path"]) for run in summary["dt_runs"]),
        str(summary["artifacts"]["gate_csv"]),
        str(summary["artifacts"]["endpoint_csv"]),
    }
    if set(manifest) != expected_manifest_paths:
        raise RuntimeError(
            "canonical output_sha256 manifest must lock exactly both traces "
            "plus the gate and endpoint tables"
        )
    _verify_hash_manifest(manifest)
    challenge_gate_lookup: dict[tuple[float, str], dict[str, bool]] = {}
    with (ROOT / summary["artifacts"]["gate_csv"]).open(
        newline="", encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["segment"] in {"early_retrigger", "late_retrigger"}:
                if row["pass"] not in {"True", "False"}:
                    raise RuntimeError(f"invalid canonical gate value: {row['pass']}")
                key = (float(row["dt_ms"]), str(row["segment"]))
                gates = challenge_gate_lookup.setdefault(key, {})
                if row["gate"] in gates:
                    raise RuntimeError(
                        f"duplicate canonical challenge gate: {key} / {row['gate']}"
                    )
                gates[str(row["gate"])] = row["pass"] == "True"
    endpoint_rows: list[dict[str, Any]] = []
    with (output / "hybrid_endpoint_table.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in ("dt_ms", "absolute_time_ms", "q", "p", "m", "A_mv", "rE_max_hz", "rE_fast_max_hz"):
                row[key] = float(row[key])
            row["latch_on"] = row["latch_on"] == "True"
            endpoint_rows.append(row)
    runs = summary["dt_runs"]
    for run in runs:
        path = ROOT / run["trace_path"]
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}

        def saved_result(prefix: str) -> dict[str, Any]:
            return {
                "time_ms": arrays[f"{prefix}_time_ms"],
                "z": arrays[f"{prefix}_q"][:, None, :],
                "rE": arrays[f"{prefix}_rE_khz"][:, None, :],
                "rE_fast": arrays[f"{prefix}_rE_fast_khz"][:, None, :],
            }

        dt_key = float(run["dt_ms"])
        try:
            canonical_early_gates = challenge_gate_lookup[(dt_key, "early_retrigger")]
            canonical_late_gates = challenge_gate_lookup[(dt_key, "late_retrigger")]
            canonical_early_clean = canonical_early_gates["numeric_clean"]
            canonical_late_clean = canonical_late_gates["numeric_clean"]
        except KeyError as exc:
            raise RuntimeError(
                f"hash-locked gate CSV lacks canonical numeric metadata for dt={dt_key}"
            ) from exc
        run["early"] = _classify_challenge(
            saved_result("early"), cfg,
            numeric_clean_override=canonical_early_clean,
        )
        run["late"] = _classify_challenge(
            saved_result("late"), cfg,
            numeric_clean_override=canonical_late_clean,
        )
        run["early_gates"] = _early_gates(run["early"])
        run["late_gates"] = _late_gates(run["late"], cfg)
        if (
            run["early_gates"] != canonical_early_gates
            or run["late_gates"] != canonical_late_gates
        ):
            raise RuntimeError(
                "reporting refresh changed the hash-locked challenge gate contract; "
                "a new canonical run is required"
            )
        run["endpoint_rows"] = [row for row in endpoint_rows if row["dt_ms"] == float(run["dt_ms"])]
        run["_plot"] = {
            prefix: {
                key[len(prefix) + 1:]: value
                for key, value in arrays.items() if key.startswith(f"{prefix}_")
            }
            for prefix in ("early", "late", "final")
        }
    agreement = _dt_agreement(runs[0], runs[1], cfg)
    if not all(agreement.values()) or not all(all(run[name].values()) for run in runs for name in ("early_gates", "late_gates")):
        raise RuntimeError("refreshed classifier no longer supports the registered verdict")
    summary["source_phenotype"] = source
    summary["base_half_dt_agreement"] = agreement
    summary["dt_runs"] = [_strip_private(run) for run in runs]
    summary["reporting_refresh"] = {
        "canonical_output_sha256_verified": True,
        "numeric_clean_inherited_from_hash_locked_gate_csv": True,
        "classifier_recomputed_from_hash_locked_trace_arrays": True,
        "challenge_gate_contract_matches_hash_locked_gate_csv": True,
    }
    _save_json(summary_path, summary)
    _plot(output, source_arrays, source, runs, summary["status"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh-existing-reporting", action="store_true")
    args = parser.parse_args()
    summary = (
        refresh_existing_reporting(args.config.resolve())
        if args.refresh_existing_reporting else run(args.config.resolve())
    )
    print(json.dumps({
        "status": summary["status"], "decision": summary["decision"],
        "dt_runs": len(summary.get("dt_runs", [])),
        "runtime_seconds": summary.get("runtime_seconds"),
        "peak_rss_kib": summary.get("peak_rss_kib"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
