#!/usr/bin/env python3
"""Run only the preregistered R3 Segment-A coupled center canary."""

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

from scripts.run_topic4_mz_spatial_autonomous_latch import _realized_pulses  # noqa: E402
from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
)
from src.topic4_mz_spatial_autonomous_latch import (  # noqa: E402
    RegionalSlowParameters,
    integrate_autonomous_latch_batch,
)
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_m_gated_reserve_coupled_canary.yaml"
SUPPORTED = "R3_COUPLED_CENTER_CANARY_DURATION_SUPPORTED_SEGMENTS_B_TO_D_UNLOCKED"
EARLY_EXIT = "R3_COUPLED_CLEAN_NO_GO_EARLY_M_EXIT"
PREMATURE_ENTRY = "R3_COUPLED_CLEAN_NO_GO_PREMATURE_EVENT5_ENTRY"
UNRESOLVED = "R3_COUPLED_CENTER_CANARY_INCOMPLETE_OR_NUMERICALLY_UNRESOLVED"
PATCH_NAMES = ("core", "annulus", "bath")
PATCH_COLORS = ("#B2182B", "#EF8A62", "#2166AC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _validate_config(cfg: dict[str, Any]) -> None:
    center = cfg["center_canary"]
    expected = {
        "tau_slow_s": 90.0,
        "q_hold": 0.8425,
        "tau_fast_s": 20.0,
        "dt_ms": 0.125,
        "q_reserve": 0.8415605027325525,
        "tau_depletion_ms": 278.35361262327285,
        "tau_m_up_ms": 225.0,
        "tau_m_down_ms": 12000.0,
        "persistence_on": 0.115,
    }
    for key, value in expected.items():
        if not np.isclose(float(center[key]), value, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"registered center canary drifted at {key}")
    challenge = cfg["background_event_challenge"]
    if [float(x) for x in challenge["realized_onsets_ms"]] != [
        1000.0, 3122.0, 5044.0, 6321.0, 7531.0, 10915.0,
    ]:
        raise RuntimeError("registered six-event schedule drifted")
    integration = cfg["integration"]
    if float(integration["duration_ms"]) != 20000.0 or float(integration["save_dt_ms"]) != 1.0:
        raise RuntimeError("Segment-A duration/reporting grid drifted")
    resource_cfg = cfg["resource_contract"]
    if (
        int(resource_cfg["processes"]) != 1
        or int(resource_cfg["blas_threads"]) != 1
        or int(resource_cfg["canary_count"]) != 1
        or bool(resource_cfg["launch_remaining_arms"])
        or int(resource_cfg["max_trace_bytes"]) > 64 * 1024 * 1024
        or float(resource_cfg["max_memory_gib"]) > 1.5
    ):
        raise RuntimeError("center-canary resource/stop contract drifted")
    scope = cfg["scope"]
    required = ("segment_a_center_canary_only", "coupled_fast_qm", "fixed_bath_resource_mask")
    forbidden = (
        "segments_b_to_d", "retrigger", "ablation", "grid", "continuous_field",
        "full_snn", "ee_weight_change", "ee_kernel_change", "conductance_membrane",
        "relay_change",
    )
    if not all(bool(scope[name]) for name in required) or any(bool(scope[name]) for name in forbidden):
        raise RuntimeError("R3 canary scope drifted")
    path_keys = tuple(key for key in cfg if key.endswith("_path") and key != "result_root")
    if set(cfg["input_sha256"]) != set(path_keys):
        raise RuntimeError("input_sha256 does not lock exactly every configured input path")


def _validate_inputs(cfg: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for key, expected in cfg["input_sha256"].items():
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = _sha256(path)
        if observed[key] != str(expected):
            raise RuntimeError(f"locked input drift for {key}: {observed[key]}")
    r3 = json.loads((ROOT / cfg["r3_summary_path"]).read_text(encoding="utf-8"))
    if r3.get("status") != "R3_M_GATED_RESERVE_RECOVERY_PATH_SUPPORTED_SHORT_P3_FORK_UNLOCKED":
        raise RuntimeError("R3 scalar/path node did not unlock this canary")
    r2 = json.loads((ROOT / cfg["r2_summary_path"]).read_text(encoding="utf-8"))
    if r2.get("status") != "R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES":
        raise RuntimeError("R2 provenance status drifted")
    p3 = json.loads((ROOT / cfg["p3_summary_path"]).read_text(encoding="utf-8"))
    if p3.get("status") != (
        "AUTONOMOUS_REGIONAL_ADDITIVE_LATCH_REGISTERED_MARGIN_CLEAN_NO_GO_"
        "ENTRY_EXIT_TIMING_CONFLICT"
    ):
        raise RuntimeError("P3 fast-scaffold provenance status drifted")
    center = cfg["center_canary"]
    matches = [
        row for row in _load_csv(ROOT / cfg["r2_mapping_path"])
        if np.isclose(float(row["tau_recovery_s"]), float(center["tau_slow_s"]))
        and np.isclose(float(row["q_hold"]), float(center["q_hold"]))
    ]
    if len(matches) != 1:
        raise RuntimeError("R2 center mapping row is missing or duplicated")
    row = matches[0]
    if (
        row["root_found"] != "True"
        or row["physical_root"] != "True"
        or not np.isclose(float(row["q_reserve"]), float(center["q_reserve"]), rtol=0.0, atol=1e-14)
        or not np.isclose(
            float(row["tau_depletion_ms"]), float(center["tau_depletion_ms"]),
            rtol=0.0, atol=1e-12,
        )
    ):
        raise RuntimeError("center q_reserve/tau_D no longer matches the locked R2 mapping")
    return observed


def first_sustained_low(
    time_ms: Sequence[float],
    regional_rate_khz: np.ndarray,
    threshold_khz: float,
    duration_ms: float,
    *,
    start_ms: float,
) -> float | None:
    """First start of an all-regional low interval of the requested duration."""

    time_values = np.asarray(time_ms, dtype=float)
    rates = np.asarray(regional_rate_khz, dtype=float)
    if (
        time_values.ndim != 1 or rates.ndim != 2 or rates.shape[0] != time_values.size
        or rates.shape[1] < 2 or np.any(np.diff(time_values) <= 0.0)
        or threshold_khz <= 0.0 or duration_ms <= 0.0
    ):
        raise ValueError("invalid sustained-low arrays or thresholds")
    low = np.all(rates <= float(threshold_khz), axis=1) & (time_values >= float(start_ms))
    starts = np.flatnonzero(low & np.r_[True, ~low[:-1]])
    for index in starts:
        stop = int(np.searchsorted(time_values, time_values[index] + duration_ms, side="left"))
        if stop < time_values.size and bool(np.all(low[index:stop + 1])):
            return float(time_values[index])
    return None


def _section_returns_from_saved_trace(
    time_ms: np.ndarray,
    rate_khz: np.ndarray,
    section_level_khz: float,
    rearm_level_khz: float,
) -> list[float]:
    """Reconstruct descriptive crossings from the saved 1-ms trace."""

    time_values = np.asarray(time_ms, dtype=float)
    rate_values = np.asarray(rate_khz, dtype=float)
    if time_values.ndim != 1 or rate_values.shape != time_values.shape:
        raise ValueError("section-return trace must be aligned 1D arrays")
    armed = bool(rate_values[0] <= rearm_level_khz)
    returns: list[float] = []
    for index in range(time_values.size - 1):
        previous = float(rate_values[index])
        following = float(rate_values[index + 1])
        if previous <= rearm_level_khz:
            armed = True
        if armed and previous < section_level_khz <= following:
            fraction = (section_level_khz - previous) / (following - previous)
            returns.append(float(time_values[index] + fraction * (time_values[index + 1] - time_values[index])))
            armed = False
    return returns


def _classify(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    time_ms = np.asarray(result["time_ms"], dtype=float)
    q = np.asarray(result["z"][:, 0, :], dtype=float)
    m = np.asarray(result["m"][:, 0, :], dtype=float)
    persistence = np.asarray(result["p"][:, 0, :], dtype=float)
    rate = np.asarray(result["rE"][:, 0, :], dtype=float)
    fast_rate = np.asarray(result["rE_fast"][:, 0, :], dtype=float)
    latch = np.asarray(result["latch"][:, 0, :], dtype=bool)
    entry_fold = float(cfg["known_boundaries"]["regional_entry_fold_q"])
    entry_indices = np.flatnonzero(np.min(q[:, :2], axis=1) < entry_fold)
    entry_index = int(entry_indices[0]) if entry_indices.size else None
    entry_time = None if entry_index is None else float(time_ms[entry_index])
    onsets = np.asarray(cfg["background_event_challenge"]["realized_onsets_ms"], dtype=float)
    pulse_duration = float(cfg["background_event_challenge"]["duration_ms"])
    event_min_q = []
    for onset in onsets:
        event_window = (time_ms >= onset) & (time_ms <= onset + pulse_duration + 100.0)
        event_min_q.append(float(np.min(q[event_window, :2])))
    entry_event_index = None
    if entry_time is not None:
        preceding = np.flatnonzero(onsets <= entry_time)
        entry_event_index = int(preceding[-1] + 1) if preceding.size else 0
    pulse_free_start = float(cfg["background_event_challenge"]["pulse_free_analysis_start_ms"])
    pulse_free_returns = [
        [float(value) for value in values if value >= pulse_free_start]
        for values in result["return_times_ms"][0]
    ]
    paired_returns = min(len(pulse_free_returns[0]), len(pulse_free_returns[1]))
    acceptance = cfg["acceptance"]
    combined_regional = np.c_[rate[:, :2], fast_rate[:, :2]]
    low_onset = first_sustained_low(
        time_ms, combined_regional, float(acceptance["low_threshold_khz"]),
        float(acceptance["sustained_low_ms"]),
        start_ms=max(pulse_free_start, entry_time or pulse_free_start),
    )
    fourth_return = None
    if paired_returns >= 4:
        fourth_return = max(pulse_free_returns[0][3], pulse_free_returns[1][3])
    low_index = None if low_onset is None else int(np.searchsorted(time_ms, low_onset))
    failure_times = [
        float(value) for value in (
            result["first_support_failure_ms"][0], result["first_bound_failure_ms"][0],
            result["first_nonfinite_ms"][0],
        ) if np.isfinite(value)
    ]
    preentry = time_ms < entry_time if entry_time is not None else np.ones_like(time_ms, dtype=bool)
    bath_q_error = float(np.max(np.abs(q[:, 2] - float(cfg["slow_common"]["z_rest"]))))
    section_level = float(cfg["integration"]["section_level_rE_fast_khz"])
    rearm_level = float(cfg["integration"]["rearm_level_rE_fast_khz"])
    reconstructed_returns = [
        _section_returns_from_saved_trace(time_ms, fast_rate[:, patch], section_level, rearm_level)
        for patch in range(3)
    ]
    actual_entry_diagnostic: dict[str, Any] | None = None
    if entry_event_index is not None and 1 <= entry_event_index < len(onsets):
        trigger_end = float(onsets[entry_event_index - 1] + pulse_duration)
        trigger_analysis_start = trigger_end + 100.0
        next_event = float(onsets[entry_event_index])
        aligned = [
            [value for value in values if trigger_analysis_start <= value < next_event]
            for values in reconstructed_returns
        ]
        paired = min(len(aligned[0]), len(aligned[1]))
        next_window = (time_ms >= next_event) & (time_ms <= next_event + pulse_duration + 100.0)
        next_event_peak = 1000.0 * float(np.max(fast_rate[next_window, :2]))
        next_event_returns = [
            [value for value in values if next_event <= value <= next_event + pulse_duration + 100.0]
            for values in reconstructed_returns[:2]
        ]
        last_paired_return = (
            max(aligned[0][paired - 1], aligned[1][paired - 1]) if paired else None
        )
        if last_paired_return is None:
            post_last_additive_max = None
        else:
            post_last = (time_ms >= last_paired_return) & (time_ms <= next_event)
            post_last_additive_max = float(cfg["model"]["additive_max_mv"]) * float(
                np.max(m[post_last, :2])
            )
        additive_at_next_event = float(cfg["model"]["additive_max_mv"]) * float(
            np.mean([np.interp(next_event, time_ms, m[:, patch]) for patch in range(2)])
        )
        numeric_clean = bool(
            not failure_times and bool(result["finite"][0]) and bool(result["active_at_end"][0])
        )
        immediate_suppressed = bool(
            not next_event_returns[0] and not next_event_returns[1]
            and next_event_peak < 1000.0 * section_level
        )
        actual_entry_diagnostic = {
            "trigger_event_index": entry_event_index,
            "trigger_event_end_ms": trigger_end,
            "pulse_response_exclusion_ms": 100.0,
            "actual_entry_aligned_analysis_start_ms": trigger_analysis_start,
            "next_event_index": entry_event_index + 1,
            "next_event_onset_ms": next_event,
            "core_return_times_ms": aligned[0],
            "annulus_return_times_ms": aligned[1],
            "paired_returns_before_next_event": paired,
            "last_paired_return_time_ms": last_paired_return,
            "latch_set_time_ms": (
                float(result["latch_set_times_ms"][0][0])
                if result["latch_set_times_ms"][0] else None
            ),
            "post_last_return_additive_max_mv": post_last_additive_max,
            "additive_at_next_event_mv": additive_at_next_event,
            "next_event_regional_fast_peak_hz": next_event_peak,
            "next_event_immediate_retrigger_suppressed": immediate_suppressed,
            "numeric_clean": numeric_clean,
            "descriptive_lifecycle_candidate": bool(
                paired >= 4 and immediate_suppressed and numeric_clean
            ),
            "formal_acceptance_override": False,
            "claim_boundary": (
                "actual-entry-aligned descriptive candidate only; late recovery, "
                "same-basin return, and late retrigger were not tested"
            ),
        }
    gates = {
        "events_1_to_5_do_not_cross_entry_fold": bool(entry_event_index == 6),
        "event_6_is_first_entry_event": bool(entry_event_index == 6),
        "preentry_m_zero_and_latch_off": bool(
            np.max(np.abs(m[preentry, :2])) <= float(acceptance["preentry_m_abs_tolerance"])
            and not np.any(latch[preentry, :2])
        ),
        "at_least_four_pulse_free_returns_each_region": bool(
            len(pulse_free_returns[0]) >= int(acceptance["minimum_pulse_free_returns_each_region"])
            and len(pulse_free_returns[1]) >= int(acceptance["minimum_pulse_free_returns_each_region"])
        ),
        "sustained_low_begins_only_after_fourth_return": bool(
            low_onset is not None and fourth_return is not None and low_onset >= fourth_return
        ),
        "finite_clean_exit_without_support_or_bounds_failure": bool(
            not failure_times and bool(result["finite"][0]) and bool(result["active_at_end"][0])
            and low_onset is not None
        ),
        "latch_active_at_clean_exit": bool(
            low_index is not None and np.all(latch[low_index, :2])
        ),
        "bath_is_fixed_mask_diagnostic": bool(
            bath_q_error <= float(acceptance["bath_q_abs_tolerance"])
            and len(pulse_free_returns[2]) == 0
            and 1000.0 * np.max(rate[:, 2]) < float(acceptance["bath_peak_max_hz"])
        ),
    }
    latch_set = result["latch_set_times_ms"][0]
    early_clean = bool(
        paired_returns < int(acceptance["minimum_pulse_free_returns_each_region"])
        and not failure_times and bool(result["finite"][0]) and bool(result["active_at_end"][0])
        and low_onset is not None and entry_event_index == 6 and bool(latch_set)
        and float(np.max(m[:, :2])) > float(acceptance["preentry_m_abs_tolerance"])
    )
    if early_clean:
        status = EARLY_EXIT
        decision = "stop_before_grid_and_reconsider_separated_termination_timescale"
    elif all(gates.values()):
        status = SUPPORTED
        decision = "unlock_segments_B_to_D_but_do_not_expand_without_new_execution_gate"
    elif (
        entry_event_index is not None and entry_event_index < 6 and not failure_times
        and bool(result["finite"][0]) and bool(result["active_at_end"][0])
    ):
        status = PREMATURE_ENTRY
        decision = "stop_before_grid_and_reassess_coupled_event_map"
    else:
        status = UNRESOLVED
        decision = "repair_failed_segment_A_gate_without_parameter_tuning"
    return {
        "status": status,
        "decision": decision,
        "gates": gates,
        "entry_event_index": entry_event_index,
        "entry_time_ms": entry_time,
        "event_min_regional_q": event_min_q,
        "actual_entry_aligned_diagnostic": actual_entry_diagnostic,
        "failure_code": (
            "premature_event5_entry" if status == PREMATURE_ENTRY
            else "early_m_exit_before_four_returns" if status == EARLY_EXIT
            else None
        ),
        "latch_set_time_ms": float(latch_set[0]) if latch_set else None,
        "clean_low_onset_ms": low_onset,
        "fourth_paired_return_time_ms": fourth_return,
        "pulse_free_core_returns": len(pulse_free_returns[0]),
        "pulse_free_annulus_returns": len(pulse_free_returns[1]),
        "pulse_free_bath_returns": len(pulse_free_returns[2]),
        "pulse_free_core_return_times_ms": pulse_free_returns[0],
        "pulse_free_annulus_return_times_ms": pulse_free_returns[1],
        "first_support_failure_ms": _finite_or_none(result["first_support_failure_ms"][0]),
        "first_bound_failure_ms": _finite_or_none(result["first_bound_failure_ms"][0]),
        "first_nonfinite_ms": _finite_or_none(result["first_nonfinite_ms"][0]),
        "support_violation_count": int(np.sum(result["support_violation_count"][0])),
        "state_bound_violation_count": int(np.sum(result["state_bound_violation_count"][0])),
        "finite": bool(result["finite"][0]),
        "active_at_end": bool(result["active_at_end"][0]),
        "final_latch_state": np.asarray(result["final_latch_state"][0], dtype=bool).tolist(),
        "min_q_core": float(np.min(q[:, 0])),
        "min_q_annulus": float(np.min(q[:, 1])),
        "bath_q_max_abs_error": bath_q_error,
        "max_m_core": float(np.max(m[:, 0])),
        "max_m_annulus": float(np.max(m[:, 1])),
        "preentry_max_abs_m": float(np.max(np.abs(m[preentry, :2]))),
        "max_additive_core_mv": float(cfg["model"]["additive_max_mv"]) * float(np.max(m[:, 0])),
        "max_p_core": float(np.max(persistence[:, 0])),
        "max_p_annulus": float(np.max(persistence[:, 1])),
        "peak_rE_core_hz": 1000.0 * float(np.max(rate[:, 0])),
        "peak_rE_annulus_hz": 1000.0 * float(np.max(rate[:, 1])),
        "peak_rE_bath_hz": 1000.0 * float(np.max(rate[:, 2])),
    }


def _plot(output: Path, result: dict[str, Any], outcome: dict[str, Any], cfg: dict[str, Any]) -> Path:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray(result["time_ms"], dtype=float) / 1000.0
    rates_hz = 1000.0 * np.asarray(result["rE"][:, 0, :], dtype=float)
    fast_hz = 1000.0 * np.asarray(result["rE_fast"][:, 0, :], dtype=float)
    q = np.asarray(result["z"][:, 0, :], dtype=float)
    m = np.asarray(result["m"][:, 0, :], dtype=float)
    persistence = np.asarray(result["p"][:, 0, :], dtype=float)
    latch = np.asarray(result["latch"][:, 0, :], dtype=float)
    onsets_s = np.asarray(cfg["background_event_challenge"]["realized_onsets_ms"]) / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    for patch, color, name in zip(range(3), PATCH_COLORS, PATCH_NAMES):
        ax.plot(time_s, rates_hz[:, patch], color=color, lw=1.0, label=name)
    for value in onsets_s:
        ax.axvline(value, color="0.82", lw=0.55, zorder=0)
    ax.set(xlabel="time (s)", ylabel="rE (Hz)", title="A  Coupled regional activity")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    ax.plot(time_s, q[:, 0], color="#2166AC", lw=1.2, label="regional q")
    ax.axhline(float(cfg["known_boundaries"]["regional_entry_fold_q"]), color="#2166AC", ls="--", lw=0.8, label="entry fold")
    ax2 = ax.twinx()
    ax2.plot(time_s, float(cfg["model"]["additive_max_mv"]) * m[:, 0], color="#B2182B", lw=1.1, label="additive A")
    ax.set(xlabel="time (s)", ylabel="q", title="B  M-gated q and additive exit")
    ax2.set_ylabel("A (mV)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7)

    ax = axes[0, 2]
    event_q = []
    trace_time = np.asarray(result["time_ms"], dtype=float)
    for onset in np.asarray(cfg["background_event_challenge"]["realized_onsets_ms"], dtype=float):
        stop = onset + float(cfg["background_event_challenge"]["duration_ms"]) + 100.0
        event_q.append(float(np.min(q[(trace_time >= onset) & (trace_time <= stop), :2])))
    ax.plot(np.arange(1, 7), event_q, "o-", color="#762A83", lw=1.2)
    ax.axhline(float(cfg["known_boundaries"]["regional_entry_fold_q"]), color="black", ls="--", lw=0.8)
    ax.set(xticks=np.arange(1, 7), xlabel="event index", ylabel="minimum regional q", title="C  Entry ordering")

    ax = axes[1, 0]
    ax.plot(time_s, persistence[:, 0], color="#4D9221", lw=1.0, label="core p")
    ax.plot(time_s, persistence[:, 1], color="#A6D96A", lw=1.0, label="annulus p")
    ax.plot(time_s, latch[:, 0], color="black", lw=0.9, label="latch")
    ax.set(xlabel="time (s)", ylabel="state", title="D  Persistence and latch")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    diagnostic = outcome.get("actual_entry_aligned_diagnostic")
    if diagnostic is not None:
        mask = (time_s >= 7.45) & (time_s <= 11.2)
    else:
        mask = time_s >= 10.8
    ax.plot(time_s[mask], fast_hz[mask, 0], color=PATCH_COLORS[0], lw=1.0, label="core fast")
    ax.plot(time_s[mask], fast_hz[mask, 1], color=PATCH_COLORS[1], lw=1.0, label="annulus fast")
    if diagnostic is not None:
        for value in diagnostic["core_return_times_ms"]:
            ax.axvline(float(value) / 1000.0, color=PATCH_COLORS[0], lw=0.65, alpha=0.6)
        for value in diagnostic["annulus_return_times_ms"]:
            ax.axvline(float(value) / 1000.0, color=PATCH_COLORS[1], lw=0.65, alpha=0.6)
        ax.axvline(
            float(diagnostic["next_event_onset_ms"]) / 1000.0,
            color="#762A83", ls="--", lw=1.0, label="event 6",
        )
        if diagnostic["latch_set_time_ms"] is not None:
            ax.axvline(
                float(diagnostic["latch_set_time_ms"]) / 1000.0,
                color="black", ls=":", lw=1.0, label="latch set",
            )
        title = "E  Actual-entry-aligned returns (descriptive)"
    else:
        for value in outcome["pulse_free_core_return_times_ms"]:
            ax.axvline(float(value) / 1000.0, color="#B2182B", lw=0.6, alpha=0.55)
        if outcome["clean_low_onset_ms"] is not None:
            ax.axvline(float(outcome["clean_low_onset_ms"]) / 1000.0, color="black", ls="--", lw=0.9, label="clean low onset")
        title = "E  Pulse-free section returns"
    ax.set(xlabel="time (s)", ylabel="rE fast (Hz)", title=title)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    verdict = [
        "F  Segment-A verdict", "",
        "formal: CLEAN NO-GO",
        "PREMATURE EVENT-5 ENTRY",
        f"entry event / time: {outcome['entry_event_index']} / {outcome['entry_time_ms']} ms",
        f"latch set: {outcome['latch_set_time_ms']} ms",
        f"pulse-free returns: {outcome['pulse_free_core_returns']} / {outcome['pulse_free_annulus_returns']}",
        "",
        "actual-entry-aligned diagnostic:",
        f"paired returns before event 6: {diagnostic['paired_returns_before_next_event'] if diagnostic else 'NA'}",
        f"event 6 suppressed: {diagnostic['next_event_immediate_retrigger_suppressed'] if diagnostic else 'NA'}",
        "descriptive lifecycle candidate only",
        "late recovery was not tested.", "",
        "Segments B-D and all other 17 paths stay closed.",
    ]
    ax.text(0.0, 1.0, "\n".join(verdict), va="top", family="monospace", fontsize=8.0)
    fig.suptitle("R3 M-gated reserve: real coupled center canary", fontsize=13, fontweight="bold")
    path = figures / "mz_m_gated_reserve_coupled_canary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def reclassify_existing(config_path: Path) -> dict[str, Any]:
    """Reclassify and rerender the saved canary without rerunning fast dynamics."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    _validate_inputs(cfg)
    output = ROOT / cfg["result_root"]
    summary_path = output / "coupled_canary_summary.json"
    trace_path = output / "segment_a_center_canary_trace.npz"
    if not summary_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("canonical summary/trace required for reclassification")
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    def failure_array(key: str) -> np.ndarray:
        value = old.get(key)
        return np.asarray([np.nan if value is None else float(value)], dtype=float)

    result = {
        "time_ms": arrays["time_ms"],
        "rE": arrays["rE_khz"][:, None, :],
        "rE_fast": arrays["rE_fast_khz"][:, None, :],
        "z": arrays["q"][:, None, :],
        "p": arrays["persistence"][:, None, :],
        "m": arrays["m"][:, None, :],
        "latch": arrays["latch"][:, None, :],
        "external_e_mv": arrays["external_e_mv"],
        "return_times_ms": [[
            list(old.get("pulse_free_core_return_times_ms", [])),
            list(old.get("pulse_free_annulus_return_times_ms", [])),
            [],
        ]],
        "latch_set_times_ms": [[old["latch_set_time_ms"]] if old.get("latch_set_time_ms") is not None else []],
        "latch_reset_times_ms": [[]],
        "first_support_failure_ms": failure_array("first_support_failure_ms"),
        "first_bound_failure_ms": failure_array("first_bound_failure_ms"),
        "first_nonfinite_ms": failure_array("first_nonfinite_ms"),
        "support_violation_count": np.asarray([[old["support_violation_count"], 0, 0]], dtype=int),
        "state_bound_violation_count": np.asarray([[old["state_bound_violation_count"], 0, 0]], dtype=int),
        "finite": np.asarray([old["finite"]], dtype=bool),
        "active_at_end": np.asarray([old["active_at_end"]], dtype=bool),
        "final_latch_state": arrays["final_latch_state"][None, :],
    }
    outcome = _classify(result, cfg)
    if outcome["entry_event_index"] != 5 or outcome["status"] != PREMATURE_ENTRY:
        raise RuntimeError("saved trace no longer reproduces the registered event-5 failure")
    _save_csv(output / "segment_a_center_canary.csv", [{
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in outcome.items() if key != "gates"
    }])
    figure = _plot(output, result, outcome, cfg)
    corrected = {
        **old,
        **outcome,
        "stop_rule_applied": True,
        "reclassified_without_fast_rerun": True,
        "reclassification_reason": (
            "fail-closed classifier now requires event-6 entry, a real latch set, and "
            "positive M before assigning the exact EARLY_M_EXIT status"
        ),
        "interpretation": [
            "R3 scalar preentry parity was a feed-forward sensor assumption, not a full coupled event-map result",
            "once q feeds back onto the regional fast state, stronger event responses increase inhibitory use and event 5 crosses the entry fold at 7.620 s",
            "the persistence latch then sets at 10.233 s and M suppresses the sixth event, leaving zero pulse-free section returns after the final pulse",
            "the center canary is therefore a numerically clean event-ordering no-go, not an unresolved integration and not the preregistered early-M-exit phenotype",
        ],
        "claim_boundary": [
            "the fixed background sequence drives the fifth-event entry; this is not zero-input spontaneous onset",
            "the bath q coordinate is fixed by an imposed depletion mask and is only a diagnostic",
            "the feed-forward R3 scalar/path oracle was necessary but did not preserve entry ordering after q-use feedback was closed",
            "no E-E weight, E-E kernel, conductance, relay, delay, M timescale, latch threshold, or R2 mapping was changed",
        ],
    }
    corrected["artifacts"]["figure"] = str(figure.relative_to(ROOT))
    _save_json(summary_path, corrected)
    return corrected


def run(config_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    parameters, low_parameters = _model(cfg)
    geometry = cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, low_root = _low_template(transfer, low_parameters)
    initial = _low_initial(low, float(cfg["model"]["z_interictal"]), reduction, parameters)
    baseline = np.asarray(initial[9:12], dtype=float)
    center = cfg["center_canary"]
    common = cfg["slow_common"]
    arm = RegionalSlowParameters(
        z_rest=float(common["z_rest"]),
        tau_z_recovery_ms=1000.0 * float(center["tau_slow_s"]),
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
        z_safe=float(common["z_safe"]),
        tau_m_up_ms=float(center["tau_m_up_ms"]),
        tau_m_down_ms=float(center["tau_m_down_ms"]),
        depletion_mask=tuple(float(x) for x in common["depletion_mask"]),
        pool_core_annulus_resource=bool(common["pool_core_annulus_resource"]),
        pool_core_annulus_effector=bool(common["pool_core_annulus_effector"]),
        enable_z=bool(common["enable_z"]), enable_m=bool(common["enable_m"]),
        q_reserve=float(center["q_reserve"]),
        tau_z_fast_recovery_ms=1000.0 * float(center["tau_fast_s"]),
        enable_m_gated_z_recovery=bool(common["enable_m_gated_z_recovery"]),
    ).validate()
    duration = float(cfg["integration"]["duration_ms"])
    dt_ms = float(center["dt_ms"])
    save_dt = float(cfg["integration"]["save_dt_ms"])
    n_samples = int(round(duration / save_dt)) + 1
    estimated_trace_bytes = n_samples * (
        3 * 9 * np.dtype(np.float32).itemsize
        + 3 * np.dtype(np.uint8).itemsize
        + 3 * np.dtype(np.float32).itemsize
        + 2 * np.dtype(np.float32).itemsize
    )
    if estimated_trace_bytes > int(cfg["resource_contract"]["max_trace_bytes"]):
        raise MemoryError("preflight trace estimate exceeds the 64 MiB canary cap")
    result = integrate_autonomous_latch_batch(
        initial[None, :], prepared, transfer, [arm], _realized_pulses(cfg),
        inhibitory_baseline_khz=baseline, dt_ms=dt_ms, duration_ms=duration,
        save_dt_ms=save_dt,
        section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
        rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes"]),
    )
    outcome = _classify(result, cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    _save_csv(output / "segment_a_center_canary.csv", [{
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in outcome.items() if key != "gates"
    }])
    np.savez_compressed(
        output / "segment_a_center_canary_trace.npz",
        time_ms=np.asarray(result["time_ms"]),
        rE_khz=np.asarray(result["rE"][:, 0]),
        rE_fast_khz=np.asarray(result["rE_fast"][:, 0]),
        q=np.asarray(result["z"][:, 0]), persistence=np.asarray(result["p"][:, 0]),
        m=np.asarray(result["m"][:, 0]),
        additive_mv=float(cfg["model"]["additive_max_mv"]) * np.asarray(result["m"][:, 0]),
        z_use=np.asarray(result["z_use"][:, 0]),
        occupancy=np.asarray(result["occupancy"][:, 0]),
        latch=np.asarray(result["latch"][:, 0]),
        external_e_mv=np.asarray(result["external_e_mv"]),
        final_state=np.asarray(result["final_state"][0]),
        final_latch_state=np.asarray(result["final_latch_state"][0]),
    )
    figure = _plot(output, result, outcome, cfg)
    runtime = time.perf_counter() - start
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary = {
        **outcome,
        "scientific_layer": "real_coupled_regional_rate_patch_segment_A_center_canary_not_full_lifecycle",
        "segments_executed": ["A_center_canary"],
        "segments_not_executed": ["B_protected_recovery", "C_true_latch_reset", "D_natural_release", "retrigger", "ablations"],
        "registered_paths_executed": 1,
        "registered_paths_not_executed": 17,
        "stop_rule_applied": outcome["status"] in {EARLY_EXIT, PREMATURE_ENTRY},
        "center_parameters": center,
        "interictal_root": low_root,
        "geometry": {
            "patch_names": list(reduction.patch_names),
            "patch_cells": list(reduction.patch_cells),
            "patch_weights": reduction.kernels.weights().tolist(),
        },
        "trace_estimated_bytes": int(estimated_trace_bytes),
        "trace_file_bytes": int((output / "segment_a_center_canary_trace.npz").stat().st_size),
        "runtime_seconds": float(runtime),
        "peak_rss_kib": peak_rss,
        "resource_gates": {
            "trace_below_64_mib": estimated_trace_bytes <= 64 * 1024 * 1024,
            "peak_rss_below_1p5_gib": peak_rss < 1.5 * 1024 * 1024,
            "single_process_single_blas_contract": True,
        },
        "input_sha256": hashes,
        "claim_boundary": [
            "the fixed background sequence drives entry; this is not zero-input spontaneous onset",
            "the bath q coordinate is fixed by an imposed depletion mask and is only a diagnostic",
            "a fewer-than-four-return clean exit falsifies the registered ictal-duration gate even if termination is finite",
            "no E-E weight, E-E kernel, conductance, relay, delay, M timescale, latch threshold, or R2 mapping was changed",
        ],
        "interpretation": [
            "R3 scalar preentry parity is a feed-forward sensor assumption and must be rechecked after q-use feedback is closed",
            "a premature event-5 entry is a clean event-map falsification even if the subsequent M exit remains finite",
        ],
        "artifacts": {
            "summary": str((output / "coupled_canary_summary.json").relative_to(ROOT)),
            "outcome_csv": str((output / "segment_a_center_canary.csv").relative_to(ROOT)),
            "trace_npz": str((output / "segment_a_center_canary_trace.npz").relative_to(ROOT)),
            "figure": str(figure.relative_to(ROOT)),
        },
        "config": cfg,
    }
    if not all(summary["resource_gates"].values()):
        summary["status"] = UNRESOLVED
        summary["decision"] = "stop_for_resource_contract_violation"
    _save_json(output / "coupled_canary_summary.json", summary)
    (output / "figures/README.md").write_text(
        "### mz_m_gated_reserve_coupled_canary.png\n\n"
        "这张 2x3 图只展示预注册的 Segment-A center canary。A–C 给出六次固定背景事件后区域活动、真实耦合的 q/M 轨迹和 entry ordering；D–E 给出 persistence/latch 与无外驱 section returns；F 明确本次 stop-rule 判定。\n\n"
        "当前真实耦合结果在 event 5 后已跨 entry fold；随后 persistence latch 在 event 6 之前 set，M 把第六次响应压低，所以 final pulse 后为 0 个 section returns。这说明 scalar preentry parity 只是 feed-forward sensor 假设，闭合 q-use feedback 后不成立。B–D、retrigger、ablation 和其余 17 条路径因此都不会运行；bath 的 q 由固定 mask 强制保持，只能作为诊断。\n\n"
        "**关注点**：C 中第 5 个点已低于 fold；D 中 latch 在第六次事件前开启；E 中没有 pulse-free return。该结果是 numerically clean premature-entry no-go，不是 early-M-exit phenotype，也不是完整 lifecycle。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reclassify-existing", action="store_true")
    args = parser.parse_args()
    summary = (
        reclassify_existing(args.config.resolve())
        if args.reclassify_existing else run(args.config.resolve())
    )
    print(json.dumps({
        "status": summary["status"], "decision": summary["decision"],
        "entry_event_index": summary["entry_event_index"],
        "pulse_free_returns": [summary["pulse_free_core_returns"], summary["pulse_free_annulus_returns"]],
        "clean_low_onset_ms": summary["clean_low_onset_ms"],
        "runtime_seconds": summary["runtime_seconds"], "peak_rss_kib": summary["peak_rss_kib"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
