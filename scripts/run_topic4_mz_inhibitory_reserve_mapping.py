#!/usr/bin/env python3
"""Map a confirmed fixed-q corridor to q_res and depletion time constants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_topic4_mz_inhibitory_reserve_corridor_r0b import (  # noqa: E402
    _all_nonempty,
    _cartesian_complete,
    _frozen_view,
)
from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _cycle_initial,
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
    _pattern_summary,
    _validate_inputs,
)
from src.topic4_mz_inhibitory_reserve import (  # noqa: E402
    InhibitoryReserveParameters,
    reserve_floor_for_hold,
)
from src.topic4_mz_spatial_autonomous_latch import (  # noqa: E402
    Pulse,
    RegionalSlowParameters,
    integrate_autonomous_latch_batch,
)
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_mapping.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_r0_provenance(cfg: dict) -> tuple[dict[str, str], dict, dict]:
    keys = ("r0b_summary_path", "r0b_sentinel_path")
    if set(cfg.get("r0_provenance_sha256", {})) != set(keys):
        raise ValueError(f"r0_provenance_sha256 must lock exactly {keys}")
    observed = {}
    payloads = []
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = _sha256(path)
        if observed[key] != str(cfg["r0_provenance_sha256"][key]):
            raise RuntimeError(f"locked R0 provenance drift for {key}: {observed[key]}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    r0b, sentinel = payloads
    if not str(r0b.get("status", "")).startswith("R0B_RESERVE_COMPATIBLE"):
        raise RuntimeError("R0b provenance does not support the fixed-q corridor")
    if not r0b.get("gates") or not all(bool(value) for value in r0b["gates"].values()):
        raise RuntimeError("R0b provenance contains a failed gate")
    if sentinel.get("status") != "R0B_LOWER_RAMP_CONFIRMED_ANCHOR_BRACKET":
        raise RuntimeError("R0b sentinel is not the locked confirmed-anchor artifact")

    q_axis = [float(value) for value in cfg["mapping"]["q_hold_axis"]]
    intervals = [
        [float(value) for value in interval]
        for interval in r0b["gate_diagnostics"]["safe_q_intervals"]
    ]
    if not all(any(q in interval for interval in intervals) for q in q_axis):
        raise RuntimeError("mapping q_hold axis is not contained in an accepted R0b interval")
    lower = float(sentinel["lowest_confirmed_safe_q"])
    if lower != float(cfg["mapping"]["confirmed_lower_anchor_q"]):
        raise RuntimeError("confirmed lower anchor drifted from the R0b sentinel")
    midpoint = 0.5 * (lower + float(cfg["mapping"]["entry_fold_q"]))
    preferred = min(q_axis, key=lambda q: abs(q - midpoint))
    if preferred != float(cfg["mapping"]["preferred_q_hold"]):
        raise RuntimeError("preferred q_hold is not the geometry-locked maximin node")
    return observed, r0b, sentinel


def period_average(
    time_ms: np.ndarray,
    values: np.ndarray,
    start_ms: float,
    stop_ms: float,
) -> tuple[float, float]:
    """Integrate a saved sensor between exact Poincare-return endpoints."""

    time = np.asarray(time_ms, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 2:
        raise ValueError("time and values must be aligned non-empty vectors")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")
    if not np.all(np.isfinite(signal)):
        raise ValueError("period signal must be finite")
    if not time[0] <= start_ms < stop_ms <= time[-1]:
        raise ValueError("period endpoints must lie inside the saved trace")
    inside = (time > start_ms) & (time < stop_ms)
    window_time = np.r_[start_ms, time[inside], stop_ms]
    window_signal = np.r_[
        np.interp(start_ms, time, signal), signal[inside],
        np.interp(stop_ms, time, signal),
    ]
    dose = float(np.trapz(window_signal, window_time))
    return dose / (stop_ms - start_ms), dose


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sensor_arm(cfg: dict) -> RegionalSlowParameters:
    sensor = cfg["sensor"]
    return RegionalSlowParameters(
        z_rest=float(cfg["mapping"]["q_rest"]),
        tau_z_recovery_ms=float(cfg["mapping"]["tau_recovery_ms"]),
        tau_z_depletion_ms=1000.0,
        inhibitory_use_threshold_khz=float(sensor["inhibitory_use_threshold_khz"]),
        inhibitory_use_width_khz=float(sensor["inhibitory_use_width_khz"]),
        tau_p_ms=float(sensor["tau_p_ms"]),
        occupancy_threshold_khz=float(sensor["occupancy_threshold_khz"]),
        occupancy_width_khz=float(sensor["occupancy_width_khz"]),
        persistence_on=0.99,
        persistence_off=0.03,
        recruitment_on=0.60,
        low_reset_threshold_khz=0.005,
        z_safe=0.885,
        tau_m_up_ms=225.0,
        tau_m_down_ms=12000.0,
        depletion_mask=(1.0, 1.0, 0.0),
        pool_core_annulus_resource=True,
        pool_core_annulus_effector=True,
        enable_z=False,
        enable_m=False,
    ).validate()


def _event_pulses(cfg: dict) -> list[Pulse]:
    challenge = cfg["background_event_challenge"]
    profile = tuple(float(value) for value in challenge["profile_core_annulus_bath"])
    return [
        Pulse(
            float(onset), float(challenge["duration_ms"]),
            float(challenge["amplitude_mv"]), profile,
        ).validate()
        for onset in challenge["realized_onsets_ms"]
    ]


def integrate_event_q(
    time_ms: np.ndarray,
    use: np.ndarray,
    *,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
) -> np.ndarray:
    """Replay a recorded state-independent use trace through the q equation."""

    time = np.asarray(time_ms, dtype=float)
    sensor = np.asarray(use, dtype=float)
    if time.ndim != 1 or sensor.shape != time.shape or time.size < 2:
        raise ValueError("event time and use must be aligned vectors")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("event time must be finite and strictly increasing")
    if not np.all(np.isfinite(sensor)) or np.any(sensor < 0.0):
        raise ValueError("event use must be finite and non-negative")
    params = InhibitoryReserveParameters(
        q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms,
        tau_depletion_ms=tau_depletion_ms,
    ).validate()
    q = np.empty_like(time)
    q[0] = q_rest
    for index, dt in enumerate(np.diff(time)):
        q[index + 1] = q[index] + dt * float(params.q_rhs(q[index], sensor[index]))
    return q


def _mapping_for_hold(
    q_hold: float,
    mean_use: float,
    event_time: np.ndarray,
    event_use: np.ndarray,
    use_slope_per_q: float,
    cfg: dict,
) -> tuple[dict[str, Any], np.ndarray]:
    mapping = cfg["mapping"]
    q_rest = float(mapping["q_rest"])
    tau_recovery = float(mapping["tau_recovery_ms"])
    target = float(mapping["event_target_q"])
    lower, upper = map(float, mapping["tau_depletion_search_ms"])
    axis = np.geomspace(lower, upper, int(mapping["tau_depletion_scan_points"]))

    def evaluate(tau_depletion: float) -> tuple[float, float, np.ndarray] | None:
        q_reserve = float(reserve_floor_for_hold(
            q_hold, mean_use, q_rest=q_rest,
            tau_recovery_ms=tau_recovery, tau_depletion_ms=tau_depletion,
        ))
        if not 0.0 < q_reserve < q_hold:
            return None
        trace = integrate_event_q(
            event_time, event_use, q_rest=q_rest, q_reserve=q_reserve,
            tau_recovery_ms=tau_recovery, tau_depletion_ms=tau_depletion,
        )
        return float(trace[-1] - target), q_reserve, trace

    evaluations = [(float(tau), evaluate(float(tau))) for tau in axis]
    valid = [(tau, value) for tau, value in evaluations if value is not None]
    residuals = np.asarray([value[0] for _, value in valid], dtype=float)
    residual_diff = np.diff(residuals)
    scan_monotone = bool(
        residual_diff.size > 0
        and (np.all(residual_diff >= 0.0) or np.all(residual_diff <= 0.0))
    )
    brackets: list[tuple[float, float]] = []
    for (left_tau, left), (right_tau, right) in zip(evaluations[:-1], evaluations[1:]):
        if left is None or right is None:
            continue
        if left[0] == 0.0 or left[0] * right[0] <= 0.0:
            brackets.append((float(left_tau), float(right_tau)))
    if not brackets:
        return ({
            "mapping_status": "no_physical_tau_root",
            "q_hold": q_hold,
            "mean_cycle_use": mean_use,
            "root_bracket_count": 0,
            "root_scan_monotone": scan_monotone,
        }, np.full_like(event_time, np.nan, dtype=float))
    left, right = brackets[0]
    for _ in range(80):
        middle = 0.5 * (left + right)
        left_eval = evaluate(left)
        middle_eval = evaluate(middle)
        if left_eval is None or middle_eval is None:
            raise RuntimeError("mapping bracket left the physical reserve domain")
        if abs(middle_eval[0]) <= float(mapping["root_tolerance_q"]):
            left = right = middle
            break
        if left_eval[0] * middle_eval[0] <= 0.0:
            right = middle
        else:
            left = middle
    tau = 0.5 * (left + right)
    final = evaluate(tau)
    if final is None:
        raise RuntimeError("final reserve mapping is nonphysical")
    residual, q_reserve, trace = final
    epsilon = max(1.0e-3, tau * 1.0e-4)
    left_condition = evaluate(max(lower, tau - epsilon))
    right_condition = evaluate(min(upper, tau + epsilon))
    root_slope = np.nan
    if left_condition is not None and right_condition is not None:
        denominator = min(upper, tau + epsilon) - max(lower, tau - epsilon)
        if denominator > 0.0:
            root_slope = float(
                (right_condition[0] - left_condition[0]) / denominator
            )
    params = InhibitoryReserveParameters(
        q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery, tau_depletion_ms=tau,
    ).validate()
    nullcline = float(params.q_nullcline(mean_use))
    derivative = (
        -1.0 / tau_recovery
        - (mean_use + (q_hold - q_reserve) * use_slope_per_q) / tau
    )
    fold = float(mapping["entry_fold_q"])
    last_pulse = float(cfg["background_event_challenge"]["last_pulse_onset_ms"])
    before_last = event_time < last_pulse
    crossing = np.flatnonzero(trace < fold)
    return ({
        "mapping_status": "root_found",
        "q_hold": q_hold,
        "mean_cycle_use": mean_use,
        "tau_depletion_ms": tau,
        "q_reserve": q_reserve,
        "event_final_q": float(trace[-1]),
        "event_target_residual": residual,
        "minimum_q_before_last_pulse": float(np.min(trace[before_last])),
        "minimum_q_full_replay": float(np.min(trace)),
        "first_entry_fold_crossing_ms": None if crossing.size == 0 else float(event_time[int(crossing[0])]),
        "averaged_nullcline_q": nullcline,
        "nullcline_residual": nullcline - q_hold,
        "slow_q_derivative_per_ms": derivative,
        "root_bracket_count": len(brackets),
        "root_scan_monotone": scan_monotone,
        "event_endpoint_root_slope_per_ms": root_slope,
        "q_reserve_is_physical": 0.0 < q_reserve < q_hold,
        "q_reserve_above_confirmed_lower_anchor_diagnostic": (
            q_reserve >= float(mapping["confirmed_lower_anchor_q"])
        ),
        "pre_last_event_stays_above_entry_fold": float(np.min(trace[before_last])) > fold,
        "full_replay_crosses_entry_fold": float(np.min(trace)) < fold,
    }, trace)


def _plot(
    figures: Path,
    cycle_rows: list[dict[str, Any]],
    event_time: np.ndarray,
    event_use: np.ndarray,
    mappings: list[dict[str, Any]],
    q_traces: dict[float, np.ndarray],
    selected: dict[str, Any] | None,
    gates: dict[str, bool],
    cfg: dict,
) -> Path:
    plt.rcParams.update({"font.size": 8.1, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    q_axis = [float(value) for value in cfg["mapping"]["q_hold_axis"]]
    valid_mappings = [
        row for row in mappings if row.get("mapping_status") == "root_found"
    ]

    ax = axes[0, 0]
    for dt, marker in zip(map(float, cfg["integration"]["dt_ms"]), ("o", "s")):
        rows = [row for row in cycle_rows if row["dt_ms"] == dt]
        ax.scatter([row["q_hold"] for row in rows], [row["mean_cycle_use"] for row in rows], marker=marker, s=25, alpha=0.75, label=f"dt={dt}")
    means = [np.mean([row["mean_cycle_use"] for row in cycle_rows if row["q_hold"] == q]) for q in q_axis]
    ax.plot(q_axis, means, color="black", lw=1.0, label="phase/dt mean")
    ax.set(xlabel="fixed q hold", ylabel="cycle-averaged inhibitory use", title="A  Ubar measured on the bounded CCO")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    time_s = event_time * 1.0e-3
    ax.plot(time_s, event_use, color="#B2182B", lw=0.8, label="regional U(t)")
    ax2 = ax.twinx()
    cumulative = np.r_[0.0, np.cumsum(0.5 * (event_use[:-1] + event_use[1:]) * np.diff(event_time))]
    ax2.plot(time_s, cumulative, color="#2166AC", lw=1.0, label="cumulative use")
    ax.set(xlabel="time (s)", ylabel="U(t)", title="B  Fixed returning-event sensor replay")
    ax2.set_ylabel("cumulative use (ms)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7)

    ax = axes[0, 2]
    ax.plot([row["q_hold"] for row in valid_mappings], [row["q_reserve"] for row in valid_mappings], "o-", color="#1B7837", label="q_res")
    ax.axhline(float(cfg["mapping"]["confirmed_lower_anchor_q"]), color="#B2182B", ls="--", lw=0.8, label="confirmed q anchor (diagnostic)")
    ax.set(xlabel="target q_hold", ylabel="mapped q_res", title="C  q_res is a parameter, not a safety boundary")
    ax2 = ax.twinx()
    ax2.plot([row["q_hold"] for row in valid_mappings], [row["tau_depletion_ms"] for row in valid_mappings], "s--", color="#2166AC", label="tau_D,d")
    ax2.set_ylabel("tau depletion (ms)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=6.7)

    ax = axes[1, 0]
    for row in valid_mappings:
        ax.plot(time_s, q_traces[row["q_hold"]], lw=1.0, label=f"hold={row['q_hold']:.4f}")
        crossing = row.get("first_entry_fold_crossing_ms")
        if crossing is not None:
            ax.scatter([float(crossing) * 1.0e-3], [float(cfg["mapping"]["entry_fold_q"])], s=16)
    ax.axhline(float(cfg["mapping"]["entry_fold_q"]), color="#762A83", ls="--", lw=0.8, label="entry fold")
    ax.axvline(float(cfg["background_event_challenge"]["last_pulse_onset_ms"]) * 1.0e-3, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="replayed q", title="D  Locked schedule tests last-only entry timing")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 1]
    for row in valid_mappings:
        q_min = max(row["q_reserve"] + 1.0e-9, row["q_hold"] - 0.01)
        q_max = min(float(cfg["mapping"]["q_rest"]), row["q_hold"] + 0.01)
        q_values = np.linspace(q_min, q_max, 100)
        use_slope = row["use_slope_per_q"]
        use_values = row["mean_cycle_use"] + use_slope * (q_values - row["q_hold"])
        params = InhibitoryReserveParameters(
            q_rest=float(cfg["mapping"]["q_rest"]), q_reserve=row["q_reserve"],
            tau_recovery_ms=float(cfg["mapping"]["tau_recovery_ms"]),
            tau_depletion_ms=row["tau_depletion_ms"],
        )
        ax.plot(q_values, 1000.0 * params.q_rhs(q_values, use_values), label=f"hold={row['q_hold']:.4f}")
        ax.scatter([row["q_hold"]], [0.0], s=18)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set(xlabel="q", ylabel="dq/dt (q/s)", title="E  Averaged q nullclines are locally attracting")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["F  Mapping verdict", ""] + [
        f"{name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()
    ] + ["", f"accepted candidate: {None if selected is None else selected['q_hold']}",
         "Calibration diagnostic only; autonomous R1 is locked."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=6.35)
    fig.suptitle("Cycle-informed reserve mapping exposes an event-ordering conflict", fontsize=12.5, fontweight="bold")
    stem = figures / "mz_inhibitory_reserve_mapping"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    r0_hashes, r0b, sentinel = _validate_r0_provenance(cfg)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
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
    low_initial = _low_initial(low, float(cfg["mapping"]["q_rest"]), reduction, parameters)
    inhibitory_baseline = np.asarray(low_initial[9:12], dtype=float)
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{cfg['mapping']['cycle_trace_key']}_state"], dtype=float)
    arm = _sensor_arm(cfg)
    q_axis = [float(value) for value in cfg["mapping"]["q_hold_axis"]]
    phases = [float(value) for value in cfg["mapping"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    cycle_rows: list[dict[str, Any]] = []
    cycle_use_blocks: list[np.ndarray] = []
    cycle_metadata: list[tuple[float, float, float]] = []
    cycle_time: np.ndarray | None = None

    for dt in dts:
        metadata = [(q, phase) for q in q_axis for phase in phases]
        result = integrate_autonomous_latch_batch(
            np.asarray([
                _cycle_initial(low, cycle, phase, q, reduction, parameters)
                for q, phase in metadata
            ]),
            prepared, transfer, [arm] * len(metadata), [],
            inhibitory_baseline_khz=inhibitory_baseline,
            dt_ms=dt, duration_ms=float(cfg["integration"]["cycle_measurement_ms"]),
            save_dt_ms=float(cfg["integration"]["cycle_save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes"]),
        )
        view = _frozen_view(result)
        saved_time = np.asarray(result["time_ms"], dtype=float)
        if cycle_time is None:
            cycle_time = saved_time
        elif not np.array_equal(cycle_time, saved_time):
            raise RuntimeError("cycle sensor save grids differ across dt arms")
        cycle_use_blocks.append(
            np.asarray(result["z_use"][:, :, 0], dtype=np.float32).T
        )
        cycle_metadata.extend((q, phase, dt) for q, phase in metadata)
        for index, (q, phase) in enumerate(metadata):
            pattern = _pattern_summary(view, index, cfg, prepared, transfer)
            returns = np.asarray(result["return_times_ms"][index][0], dtype=float)
            discard = int(cfg["mapping"]["cycle_discard_returns"])
            integral_returns = int(cfg["mapping"]["cycle_integral_returns"])
            if returns.size <= discard + integral_returns:
                raise RuntimeError(
                    f"insufficient full cycles for q={q}, phase={phase}, dt={dt}"
                )
            start_ms = float(returns[discard])
            stop_ms = float(returns[discard + integral_returns])
            mean_use, use_dose = period_average(
                saved_time,
                np.asarray(result["z_use"][:, index, 0], dtype=float),
                start_ms,
                stop_ms,
            )
            mean_occupancy, _ = period_average(
                saved_time,
                np.prod(result["occupancy"][:, index, :2], axis=1),
                start_ms,
                stop_ms,
            )
            window = (saved_time >= start_ms) & (saved_time <= stop_ms)
            cycle_rows.append({
                "q_hold": q, "phase": phase, "dt_ms": dt,
                "cycle_window_start_ms": start_ms,
                "cycle_window_stop_ms": stop_ms,
                "integrated_returns": integral_returns,
                "mean_cycle_use": mean_use,
                "cycle_use_dose": use_dose,
                "mean_cycle_period_ms": (stop_ms - start_ms) / integral_returns,
                "mean_joint_occupancy": mean_occupancy,
                "max_core_annulus_use_difference": float(np.max(np.abs(result["z_use"][window, index, 0] - result["z_use"][window, index, 1]))),
                **pattern,
            })

    if cycle_time is None:
        raise RuntimeError("cycle sensor measurement produced no trace")
    _save_csv(output / "cycle_use_measurements.csv", cycle_rows)
    np.savez_compressed(
        output / "cycle_sensor_replay.npz",
        time_ms=cycle_time.astype(np.float32),
        use=np.concatenate(cycle_use_blocks, axis=0),
        q_hold=np.asarray([row[0] for row in cycle_metadata]),
        phase=np.asarray([row[1] for row in cycle_metadata]),
        dt_ms=np.asarray([row[2] for row in cycle_metadata]),
    )

    event_dt = float(cfg["integration"]["event_replay_dt_ms"])
    event = integrate_autonomous_latch_batch(
        low_initial[None, :], prepared, transfer, [arm], _event_pulses(cfg),
        inhibitory_baseline_khz=inhibitory_baseline,
        dt_ms=event_dt,
        duration_ms=float(cfg["background_event_challenge"]["replay_stop_ms"]),
        save_dt_ms=float(cfg["integration"]["event_save_dt_ms"]),
        section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
        rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes"]),
    )
    event_time = np.asarray(event["time_ms"], dtype=float)
    event_use = np.asarray(event["z_use"][:, 0, 0], dtype=float)
    event_dose = float(np.trapz(event_use, event_time))
    event_return_counts = np.asarray([
        len(event["return_times_ms"][0][patch]) for patch in range(3)
    ], dtype=int)
    np.savez_compressed(
        output / "fixed_event_sensor_raw.npz",
        time_ms=event_time.astype(np.float32),
        use=event_use.astype(np.float32),
        return_counts=event_return_counts,
        support_violation_count=np.asarray(event["support_violation_count"]),
        state_bound_violation_count=np.asarray(event["state_bound_violation_count"]),
        finite=np.asarray(event["finite"]),
    )

    mean_use_by_q = {
        q: float(np.mean([row["mean_cycle_use"] for row in cycle_rows if row["q_hold"] == q]))
        for q in q_axis
    }
    use_slope, _ = np.polyfit(q_axis, [mean_use_by_q[q] for q in q_axis], 1)
    mappings = []
    q_traces = {}
    for q in q_axis:
        row, trace = _mapping_for_hold(
            q, mean_use_by_q[q], event_time, event_use, float(use_slope), cfg
        )
        row["use_slope_per_q"] = float(use_slope)
        mappings.append(row)
        q_traces[q] = trace

    preferred = float(cfg["mapping"]["preferred_q_hold"])
    valid_mappings = [
        row for row in mappings if row.get("mapping_status") == "root_found"
    ]
    geometry_preferred = next(
        (row for row in valid_mappings if row["q_hold"] == preferred), None
    )
    spreads = {
        q: float(np.ptp([row["mean_cycle_use"] for row in cycle_rows if row["q_hold"] == q]))
        for q in q_axis
    }
    expected = int(cfg["background_event_challenge"]["expected_returning_events"])
    cycle_complete = _cartesian_complete(
        cycle_rows,
        ("q_hold", "phase", "dt_ms"),
        (q_axis, phases, dts),
    )
    event_failclosed = bool(
        np.all(np.asarray(event["finite"], dtype=bool))
        and int(np.sum(event["support_violation_count"])) == 0
        and int(np.sum(event["state_bound_violation_count"])) == 0
    )
    gates = {
        "locked_R0b_provenance_accepts_mapping_axis": bool(
            r0b["status"].startswith("R0B_RESERVE_COMPATIBLE")
            and sentinel["status"] == "R0B_LOWER_RAMP_CONFIRMED_ANCHOR_BRACKET"
        ),
        "cycle_sensor_cartesian_product_complete": cycle_complete,
        "cycle_sensor_all_q_phase_dt_bounded_CCO": _all_nonempty(
            cycle_rows, lambda row: row["outcome"] == "bounded_CCO"
        ),
        "cycle_use_phase_dt_spread_within_gate": max(spreads.values()) <= float(cfg["mapping"]["maximum_phase_dt_use_spread"]),
        "fixed_event_replay_preserves_six_regional_returns": bool(
            np.array_equal(event_return_counts, [expected, expected, 0])
        ),
        "fixed_event_sensor_is_failclosed_clean": event_failclosed,
        "event_sensor_dose_is_finite_positive": np.isfinite(event_dose) and event_dose > 0.0,
        "all_q_nodes_have_one_monotone_physical_root": bool(
            len(valid_mappings) == len(q_axis)
            and _all_nonempty(
                valid_mappings,
                lambda row: row["root_bracket_count"] == 1
                and row["root_scan_monotone"]
                and row["q_reserve_is_physical"],
            )
        ),
        "all_mapped_nullclines_are_locally_attracting": _all_nonempty(
            valid_mappings, lambda row: row["slow_q_derivative_per_ms"] < 0.0
        ),
        "all_event_replays_cross_only_on_last_event": _all_nonempty(
            valid_mappings,
            lambda row: row["pre_last_event_stays_above_entry_fold"]
            and row["full_replay_crosses_entry_fold"],
        ),
        "geometry_preferred_mapping_root_exists": geometry_preferred is not None,
    }
    timing_gate = gates["all_event_replays_cross_only_on_last_event"]
    non_timing_gates = {
        key: value for key, value in gates.items()
        if key != "all_event_replays_cross_only_on_last_event"
    }
    if all(non_timing_gates.values()) and not timing_gate:
        status = "RESERVE_MAPPING_CLEAN_NO_GO_LOCKED_EVENT_ORDERING_CONFLICT"
        decision = "preserve_no_go_and_require_noncyclic_R1a_periodic_oracle"
    elif all(gates.values()):
        status = "RESERVE_PRELIMINARY_MAPPING_SUPPORTED_R1A_PERIODIC_ORACLE_REQUIRED"
        decision = "run_R1a_periodic_q_oracle_not_autonomous_lifecycle"
    else:
        status = "RESERVE_NULLCLINE_MAPPING_NO_GO_OR_NUMERICALLY_UNRESOLVED"
        decision = "repair_non_timing_mapping_gate_before_any_coupled_test"
    selected = geometry_preferred if all(gates.values()) else None

    _save_csv(output / "reserve_parameter_mappings.csv", mappings)
    np.savez_compressed(
        output / "fixed_event_sensor_replay.npz",
        time_ms=event_time.astype(np.float32), use=event_use.astype(np.float32),
        q_hold=np.asarray(q_axis),
        q_trace=np.asarray([q_traces[q] for q in q_axis], dtype=np.float32),
    )
    expected_figure = figures / "mz_inhibitory_reserve_mapping.png"
    summary = {
        "status": status,
        "scientific_layer": "complete_cycle_informed_scalar_mapping_not_coupled_lifecycle",
        "decision": decision,
        "gates": gates,
        "event_sensor_dose_ms": event_dose,
        "mean_use_by_q": {str(q): mean_use_by_q[q] for q in q_axis},
        "use_spread_by_q": {str(q): spreads[q] for q in q_axis},
        "use_slope_per_q": float(use_slope),
        "mappings": mappings,
        "geometry_preferred_mapping": geometry_preferred,
        "selected_mapping": selected,
        "input_sha256": hashes,
        "r0_provenance_sha256": r0_hashes,
        "interictal_root": low_root,
        "claim_boundary": [
            "the event sensor trace is generated with q and M frozen; q trajectories are scalar offline replays",
            "q_res and tau_D,d are a preliminary two-equation calibration, not an independently validated identification",
            "q_res is a parameter floor and is not used as the dynamical safety boundary",
            "the locked event schedule calibrates the endpoint and therefore cannot serve as held-out validation",
            "even a supported mapping would require periodic and closed-loop q(t) oracles before autonomous testing",
            "no E-E, conductance, relay, or dynamic-threshold mechanism was changed",
        ],
        "config": cfg,
        "artifacts": {
            "figure": str(expected_figure.relative_to(ROOT)),
            "cycle_csv": str((output / "cycle_use_measurements.csv").relative_to(ROOT)),
            "cycle_sensor_trace": str((output / "cycle_sensor_replay.npz").relative_to(ROOT)),
            "mapping_csv": str((output / "reserve_parameter_mappings.csv").relative_to(ROOT)),
            "raw_event_trace": str((output / "fixed_event_sensor_raw.npz").relative_to(ROOT)),
            "event_trace": str((output / "fixed_event_sensor_replay.npz").relative_to(ROOT)),
        },
        "plot_status": "pending",
    }
    (output / "reserve_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure = _plot(
        figures, cycle_rows, event_time, event_use, mappings, q_traces,
        selected, gates, cfg,
    )
    summary["artifacts"]["figure"] = str(figure.relative_to(ROOT))
    summary["plot_status"] = "complete"
    (output / "reserve_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_inhibitory_reserve_mapping.png\n\n"
        "这张图检查 R0b 的 fixed-q corridor 能否映射到 reserve 参数。A 用完整 return-to-return 窗口测量 bounded CCO 的 inhibitory-use；B 显示锁定背景事件的 sensor dose；C 将 q_res 明确标成参数而非安全边界；D 直接检验是否真的只有最后一次事件越过 entry fold；E 只检查平均标量 q-nullcline；F 给出 fail-closed gate。\n\n"
        "本节点的 q trajectory 仍是离线 scalar replay，尚未与 fast state 自洽耦合。锁定 event schedule 同时参与 endpoint calibration，因此只用于暴露 timing conflict，不能作为 autonomous entry 证据。\n\n"
        "**关注点**：如果前五次事件已经越 fold，结果必须登记为 clean no-go；不能换 seed、改 target 或把 q_res 当作 q(t) 的安全下界来救。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps({
        "status": summary["status"],
        "gates": summary["gates"],
        "decision": summary["decision"],
        "selected_mapping": summary["selected_mapping"],
        "mappings": summary["mappings"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
