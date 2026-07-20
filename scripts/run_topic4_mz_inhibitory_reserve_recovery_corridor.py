#!/usr/bin/env python3
"""R2 scalar continuation of the inhibitory-reserve recovery timescale.

This producer uses hash-locked frozen CCO/event sensors and an analytic
post-exit handoff predictor.  It does not run or modify the fast spatial model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
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

from scripts.run_topic4_mz_inhibitory_reserve_periodic_oracle import (  # noqa: E402
    extract_piecewise_constant_window,
)


DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_recovery_corridor.yaml"

ROOT_FOUND = "root_found"
NO_ROOT_IN_DOMAIN = "no_root_in_registered_physical_domain"
NONMONOTONE_SCAN = "nonmonotone_registered_scan"
MULTIPLE_ROOTS = "multiple_roots_in_registered_domain"
NUMERIC_ERROR = "numeric_error"


class NoPhysicalPeriodicReserveRoot(RuntimeError):
    """No q_res in the registered physical interval yields the target hold."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _key(q_hold: float, phase: float, dt_ms: float) -> tuple[float, float, float]:
    return (round(float(q_hold), 10), round(float(phase), 10), round(float(dt_ms), 10))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    if [float(value) for value in model["tau_recovery_s_axis"]] != [
        20.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 160.0
    ]:
        raise RuntimeError("registered tau_r axis drifted")
    if [float(value) for value in model["q_hold_axis"]] != [0.84, 0.8425, 0.845]:
        raise RuntimeError("registered q_hold axis drifted")
    if float(model["preferred_tau_recovery_s"]) != 80.0 or float(model["preferred_q_hold"]) != 0.8425:
        raise RuntimeError("preregistered representative drifted")
    if [int(value) for value in cfg["integration"]["event_scalar_substeps"]] != [1, 2]:
        raise RuntimeError("base/half event replay contract drifted")
    if int(cfg["acceptance"]["minimum_consecutive_tau_nodes"]) != 3:
        raise RuntimeError("registered consecutive-node gate drifted")
    if [float(value) for value in cfg["acceptance"]["short_coupled_tau_recovery_s"]] != [60.0, 70.0, 80.0]:
        raise RuntimeError("short coupled unlock axis drifted")
    if [float(value) for value in cfg["sensitivity"]["fixed_parameter_tau_recovery_s"]] != [72.0, 88.0]:
        raise RuntimeError("fixed-parameter sensitivity axis drifted")
    handoff = cfg["handoff"]
    inherited = (
        float(handoff["q_reset_safe"]), float(handoff["persistence_tau_ms"]),
        float(handoff["persistence_start"]), float(handoff["persistence_off"]),
        float(handoff["reset_horizon_ms"]), float(handoff["tau_m_down_ms"]),
        float(handoff["additive_margin_mv"]), float(handoff["additive_release_threshold_mv"]),
    )
    if inherited != (0.885, 750.0, 1.0, 0.03, 120000.0, 12000.0, 0.025, 0.020):
        raise RuntimeError("inherited hybrid handoff constants drifted")
    scope = cfg["scope"]
    if not scope.get("scalar_sensor_replay_only") or not scope.get("analytic_postictal_handoff_only"):
        raise RuntimeError("R2 must remain a scalar replay plus analytic handoff")
    forbidden = (
        "thresholded_eligibility_used", "coupled_fast_qm", "autonomous_lifecycle",
        "continuous_field", "full_snn", "ee_weight_change", "ee_kernel_change",
        "conductance_membrane",
    )
    if any(bool(scope.get(name)) for name in forbidden):
        raise RuntimeError("R2 config attempts to unlock a forbidden scope")


def _load_and_validate_inputs(
    cfg: dict[str, Any],
) -> tuple[
    dict[str, str], dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]],
    dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], dict[float, float],
    np.ndarray, np.ndarray,
]:
    path_keys = (
        "mapping_summary_path", "cycle_sensor_path", "cycle_measurements_path",
        "event_sensor_path", "r0b_summary_path", "r0b_ramp_path",
    )
    locks = cfg.get("input_sha256", {})
    if set(locks) != set(path_keys):
        raise ValueError(f"input_sha256 must lock exactly {path_keys}")
    hashes: dict[str, str] = {}
    for name in path_keys:
        path = ROOT / str(cfg[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = _sha256(path)
        if hashes[name] != str(locks[name]):
            raise RuntimeError(f"locked R2 input drift for {name}: {hashes[name]}")

    mapping = json.loads((ROOT / str(cfg["mapping_summary_path"])).read_text(encoding="utf-8"))
    if mapping.get("status") != "RESERVE_MAPPING_CLEAN_NO_GO_LOCKED_EVENT_ORDERING_CONFLICT":
        raise RuntimeError("R1 provenance no longer carries the locked entry-ordering no-go")
    if mapping.get("gates", {}).get("all_event_replays_cross_only_on_last_event") is not False:
        raise RuntimeError("R1 provenance no longer exposes its unique failed timing gate")
    model = cfg["model"]
    mapping_cfg = mapping["config"]
    if not np.isclose(float(mapping_cfg["mapping"]["entry_fold_q"]), float(model["entry_fold_q"])):
        raise RuntimeError("entry fold drifted from R1")
    if not np.isclose(float(mapping_cfg["mapping"]["event_target_q"]), float(model["event_final_target_q"])):
        raise RuntimeError("event final target drifted from R1")
    if [float(value) for value in mapping_cfg["background_event_challenge"]["realized_onsets_ms"]] != [
        float(value) for value in cfg["locked_schedule"]["onsets_ms"]
    ]:
        raise RuntimeError("locked event schedule drifted from R1")

    with np.load(ROOT / str(cfg["cycle_sensor_path"]), allow_pickle=False) as payload:
        if set(payload.files) != {"time_ms", "use", "q_hold", "phase", "dt_ms"}:
            raise ValueError("cycle sensor schema drifted")
        cycle = {name: np.asarray(payload[name]) for name in payload.files}
    n_records = int(cycle["q_hold"].size)
    if (
        cycle["time_ms"].ndim != 1
        or cycle["use"].shape != (n_records, cycle["time_ms"].size)
        or cycle["phase"].shape != (n_records,)
        or cycle["dt_ms"].shape != (n_records,)
        or np.any(np.diff(cycle["time_ms"].astype(float)) <= 0.0)
        or any(not np.all(np.isfinite(cycle[name])) for name in cycle)
        or np.any(cycle["use"] < 0.0)
    ):
        raise ValueError("cycle sensor arrays are not finite and aligned")

    raw_cycle_rows = _load_csv(ROOT / str(cfg["cycle_measurements_path"]))
    required_cycle = {
        "q_hold", "phase", "dt_ms", "cycle_window_start_ms", "cycle_window_stop_ms",
        "integrated_returns", "outcome",
    }
    if not raw_cycle_rows or not required_cycle.issubset(raw_cycle_rows[0]):
        raise ValueError("cycle measurement table is missing registered fields")
    cycle_rows: list[dict[str, Any]] = []
    for raw in raw_cycle_rows:
        cycle_rows.append({
            "q_hold": float(raw["q_hold"]),
            "phase": float(raw["phase"]),
            "dt_ms": float(raw["dt_ms"]),
            "cycle_window_start_ms": float(raw["cycle_window_start_ms"]),
            "cycle_window_stop_ms": float(raw["cycle_window_stop_ms"]),
            "integrated_returns": int(raw["integrated_returns"]),
            "outcome": str(raw["outcome"]),
        })
    q_axis = [float(value) for value in model["q_hold_axis"]]
    phases = [float(value) for value in cfg["periodic_gate"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["periodic_gate"]["source_dt_ms"]]
    expected_keys = {_key(q, phase, dt) for q in q_axis for phase in phases for dt in dts}
    observed_keys = [_key(row["q_hold"], row["phase"], row["dt_ms"]) for row in cycle_rows]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
        raise RuntimeError("cycle measurement table is not the complete unique 24-cell product")
    if len(cycle_rows) != n_records:
        raise RuntimeError("cycle CSV and NPZ row counts differ")
    for index, row in enumerate(cycle_rows):
        if _key(cycle["q_hold"][index], cycle["phase"][index], cycle["dt_ms"][index]) != _key(
            row["q_hold"], row["phase"], row["dt_ms"]
        ):
            raise RuntimeError("cycle CSV and NPZ row ordering drifted")
        if row["outcome"] != "bounded_CCO" or row["integrated_returns"] != int(cfg["periodic_gate"]["integrated_returns_per_window"]):
            raise RuntimeError("cycle provenance is not an exact bounded eight-return window")

    with np.load(ROOT / str(cfg["event_sensor_path"]), allow_pickle=False) as payload:
        required_event = {
            "time_ms", "use", "return_counts", "support_violation_count",
            "state_bound_violation_count", "finite",
        }
        if set(payload.files) != required_event:
            raise ValueError("event sensor schema drifted")
        event = {name: np.asarray(payload[name]) for name in payload.files}
    if (
        event["time_ms"].ndim != 1
        or event["use"].shape != event["time_ms"].shape
        or np.any(np.diff(event["time_ms"].astype(float)) <= 0.0)
        or not np.all(np.isfinite(event["time_ms"]))
        or not np.all(np.isfinite(event["use"]))
        or np.any(event["use"] < 0.0)
        or event["return_counts"].tolist() != [6, 6, 0]
        or np.any(event["support_violation_count"])
        or np.any(event["state_bound_violation_count"])
        or not bool(np.all(event["finite"]))
        or not np.isclose(float(event["time_ms"][-1]), float(cfg["locked_schedule"]["stop_ms"]))
    ):
        raise RuntimeError("event sensor is not the locked clean 6/6/0 replay")

    r0b = json.loads((ROOT / str(cfg["r0b_summary_path"])).read_text(encoding="utf-8"))
    if not str(r0b.get("status", "")).startswith("R0B_RESERVE_COMPATIBLE"):
        raise RuntimeError("R0b provenance no longer supports the frozen-q corridor")
    if not r0b.get("gates") or not all(bool(value) for value in r0b["gates"].values()):
        raise RuntimeError("R0b provenance contains a failed gate")
    safe_intervals = [[float(value) for value in row] for row in r0b["gate_diagnostics"]["safe_q_intervals"]]
    if not all(any(q in interval for interval in safe_intervals) for q in q_axis):
        raise RuntimeError("R2 q_hold axis left the accepted R0b safe strip")

    raw_ramp_rows = _load_csv(ROOT / str(cfg["r0b_ramp_path"]))
    required_ramp = {"q", "phase", "dt_ms", "outcome", "final_additive_mv"}
    if not raw_ramp_rows or not required_ramp.issubset(raw_ramp_rows[0]):
        raise ValueError("R0b ramp table is missing handoff fields")
    ramp_rows = [{
        "q": float(raw["q"]), "phase": float(raw["phase"]), "dt_ms": float(raw["dt_ms"]),
        "outcome": str(raw["outcome"]), "final_additive_mv": float(raw["final_additive_mv"]),
    } for raw in raw_ramp_rows if float(raw["q"]) in q_axis]
    expected_ramp = {_key(q, phase, dt) for q in q_axis for phase in phases for dt in dts}
    observed_ramp = [_key(row["q"], row["phase"], row["dt_ms"]) for row in ramp_rows]
    if len(observed_ramp) != len(set(observed_ramp)) or set(observed_ramp) != expected_ramp:
        raise RuntimeError("R0b ramp table is not the complete R2 q/phase/dt product")
    if any(row["outcome"] != "LLL" or not np.isfinite(row["final_additive_mv"]) for row in ramp_rows):
        raise RuntimeError("R0b handoff source includes an unaccepted ramp")
    a0_by_q = {
        q: min(row["final_additive_mv"] for row in ramp_rows if row["q"] == q)
        for q in q_axis
    }

    fold_items = sorted(
        (float(q), float(payload["additive_mv"]))
        for q, payload in r0b["low_root_folds"].items()
        if bool(payload.get("support_all"))
    )
    fold_q = np.asarray([item[0] for item in fold_items] + [float(model["entry_fold_q"])], dtype=float)
    fold_a = np.asarray([item[1] for item in fold_items] + [0.0], dtype=float)
    order = np.argsort(fold_q)
    fold_q, fold_a = fold_q[order], fold_a[order]
    if (
        np.any(np.diff(fold_q) <= 0.0)
        or np.any(np.diff(fold_a) >= 0.0)
        or fold_q[0] > float(cfg["periodic_gate"]["minimum_q"])
        or not np.isclose(fold_q[-1], float(model["entry_fold_q"]))
        or not np.isclose(fold_a[-1], 0.0)
    ):
        raise RuntimeError("augmented R0b A_fold interpolation is invalid")

    return hashes, mapping, cycle, cycle_rows, event, r0b, ramp_rows, a0_by_q, fold_q, fold_a


def _validate_intervals(
    durations_ms: Sequence[float], use: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations_ms, dtype=float)
    sensor = np.asarray(use, dtype=float)
    if durations.ndim != 1 or sensor.shape != durations.shape or durations.size == 0:
        raise ValueError("durations and use must be aligned non-empty vectors")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("durations must be finite and positive")
    if not np.all(np.isfinite(sensor)) or np.any(sensor < 0.0):
        raise ValueError("use must be finite and non-negative")
    return durations, sensor


def _subdivide_intervals(
    durations_ms: Sequence[float], use: Sequence[float], substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    durations, sensor = _validate_intervals(durations_ms, use)
    if int(substeps) != substeps or int(substeps) < 1:
        raise ValueError("substeps must be a positive integer")
    count = int(substeps)
    return np.repeat(durations / count, count), np.repeat(sensor, count)


def _affine_trace(alpha: np.ndarray, beta: np.ndarray, initial: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or a.size == 0:
        raise ValueError("affine coefficients must be aligned vectors")
    prefix = np.r_[1.0, np.cumprod(a)]
    if np.any(prefix <= 0.0) or not np.all(np.isfinite(prefix)):
        raise RuntimeError("affine prefix product left the finite positive domain")
    accumulated = np.r_[0.0, np.cumsum(b / prefix[1:])]
    trace = prefix * (float(initial) + accumulated)
    if not np.all(np.isfinite(trace)):
        raise RuntimeError("affine trace became non-finite")
    return trace


def exact_q_trace(
    durations_ms: Sequence[float],
    use: Sequence[float],
    *,
    q_initial: float,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    substeps: int = 1,
) -> dict[str, Any]:
    durations, sensor = _subdivide_intervals(durations_ms, use, substeps)
    if not 0.0 < q_reserve < q_rest <= 1.0:
        raise ValueError("q bounds must satisfy 0<q_reserve<q_rest<=1")
    if tau_recovery_ms <= 0.0 or tau_depletion_ms <= 0.0:
        raise ValueError("q time constants must be positive")
    decay = 1.0 / float(tau_recovery_ms) + sensor / float(tau_depletion_ms)
    drive = float(q_rest) / float(tau_recovery_ms) + q_reserve * sensor / float(tau_depletion_ms)
    equilibrium = drive / decay
    alpha = np.exp(-decay * durations)
    beta = equilibrium * (1.0 - alpha)
    trace = _affine_trace(alpha, beta, float(q_initial))
    integrals = equilibrium * durations + (trace[:-1] - equilibrium) * (1.0 - alpha) / decay
    return {
        "durations_ms": durations,
        "use": sensor,
        "time_ms": np.r_[0.0, np.cumsum(durations)],
        "q": trace,
        "integral": float(np.sum(integrals)),
        "map_alpha": float(np.prod(alpha)),
    }


def exact_periodic_q_orbit(
    durations_ms: Sequence[float],
    use: Sequence[float],
    *,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    integrated_returns: int,
) -> dict[str, Any]:
    durations, sensor = _validate_intervals(durations_ms, use)
    if integrated_returns <= 0:
        raise ValueError("integrated_returns must be positive")
    zero = exact_q_trace(
        durations, sensor, q_initial=0.0, q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
    )
    alpha = float(zero["map_alpha"])
    beta = float(np.asarray(zero["q"])[-1])
    if not 0.0 <= alpha < 1.0:
        raise RuntimeError("periodic q map is not contractive")
    q_strobe = beta / (1.0 - alpha)
    orbit = exact_q_trace(
        durations, sensor, q_initial=q_strobe, q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
    )
    trace = np.asarray(orbit["q"], dtype=float)
    duration = float(np.sum(durations))
    return {
        "q_min": float(np.min(trace)),
        "q_max": float(np.max(trace)),
        "q_mean": float(orbit["integral"]) / duration,
        "q_stroboscopic": float(q_strobe),
        "window_multiplier": alpha,
        "per_return_multiplier": float(alpha ** (1.0 / float(integrated_returns))),
        "closure_error": float(abs(trace[-1] - trace[0])),
        "time_ms": np.asarray(orbit["time_ms"], dtype=float),
        "q": trace,
    }


def solve_periodic_q_reserve(
    durations_ms: Sequence[float],
    use: Sequence[float],
    *,
    q_hold: float,
    q_rest: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    integrated_returns: int,
    q_reserve_lower: float,
    q_reserve_upper_margin: float,
    tolerance: float,
) -> dict[str, Any]:
    lower = float(q_reserve_lower)
    upper = float(q_hold) - float(q_reserve_upper_margin)
    if not 0.0 < lower < upper < q_hold < q_rest:
        raise ValueError("registered q_reserve interval is invalid")
    endpoints = [
        exact_periodic_q_orbit(
            durations_ms, use, q_rest=q_rest, q_reserve=value,
            tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
            integrated_returns=integrated_returns,
        )
        for value in (lower, upper)
    ]
    means = np.asarray([row["q_mean"] for row in endpoints], dtype=float)
    slope = float((means[1] - means[0]) / (upper - lower))
    if not np.isfinite(slope) or slope <= 0.0:
        raise RuntimeError("periodic q_res map is not increasing")
    if not means[0] <= q_hold <= means[1]:
        raise NoPhysicalPeriodicReserveRoot(
            "periodic q_res root is not bracketed in the registered physical domain"
        )
    q_reserve = float(lower + (q_hold - means[0]) / slope)
    solved = exact_periodic_q_orbit(
        durations_ms, use, q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
        integrated_returns=integrated_returns,
    )
    residual = float(solved["q_mean"] - q_hold)
    if abs(residual) > tolerance:
        raise RuntimeError("periodic q_res solve missed its registered tolerance")
    return {
        **solved,
        "q_reserve": q_reserve,
        "q_mean_residual": residual,
        "q_reserve_mean_slope": slope,
    }


def classify_event_replay(
    trace: dict[str, Any],
    onsets_ms: Sequence[float],
    *,
    entry_fold_q: float,
    final_target_q: float,
    pre_last_margin_q: float,
) -> dict[str, Any]:
    onsets = np.asarray(onsets_ms, dtype=float)
    time = np.asarray(trace["time_ms"], dtype=float)
    q = np.asarray(trace["q"], dtype=float)
    if onsets.ndim != 1 or onsets.size == 0 or np.any(np.diff(onsets) <= 0.0):
        raise ValueError("event onsets must be strictly increasing")
    if q.shape != time.shape or time[-1] < onsets[-1]:
        raise ValueError("event trace does not cover the locked schedule")
    crossing = np.flatnonzero(q < float(entry_fold_q))
    crossing_time = None if crossing.size == 0 else float(time[int(crossing[0])])
    entry_index = None if crossing_time is None else int(np.searchsorted(onsets, crossing_time, side="right"))
    before_last = time < float(onsets[-1])
    after_last = time >= float(onsets[-1])
    minimum_before_last = float(np.min(q[before_last]))
    minimum_after_last = float(np.min(q[after_last]))
    margin_pass = bool(minimum_before_last >= entry_fold_q + pre_last_margin_q)
    target_reached = bool(minimum_after_last <= final_target_q)
    last_event_crossing = entry_index == int(onsets.size)
    if entry_index is None:
        outcome = "no_entry"
    elif entry_index < int(onsets.size):
        outcome = f"premature_entry_event_{entry_index}"
    elif last_event_crossing and margin_pass and target_reached:
        outcome = f"target_entry_event_{entry_index}"
    else:
        outcome = f"entry_event_{entry_index}"
    return {
        "outcome": outcome,
        "entered": entry_index is not None,
        "entry_event_index": entry_index,
        "first_crossing_ms": crossing_time,
        "minimum_q_before_last": minimum_before_last,
        "minimum_q_after_last": minimum_after_last,
        "minimum_q_full": float(np.min(q)),
        "final_q": float(q[-1]),
        "pre_last_margin_pass": margin_pass,
        "last_event_crosses_fold": last_event_crossing,
        "last_event_target_reached": target_reached,
        "entry_pass": bool(last_event_crossing and margin_pass and target_reached),
    }


def _cycle_intervals(
    cycle: dict[str, np.ndarray], row: dict[str, Any], index: int,
) -> tuple[np.ndarray, np.ndarray]:
    durations, use, _ = extract_piecewise_constant_window(
        np.asarray(cycle["time_ms"], dtype=float),
        np.asarray(cycle["use"][index], dtype=float),
        float(row["cycle_window_start_ms"]),
        float(row["cycle_window_stop_ms"]),
    )
    return durations, use


def _event_intervals(event: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(event["time_ms"], dtype=float)
    return np.diff(time), np.asarray(event["use"], dtype=float)[:-1]


def _calibrate_cell(
    tau_recovery_s: float,
    q_hold: float,
    primary_cycle: tuple[np.ndarray, np.ndarray, int],
    event_intervals: tuple[np.ndarray, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration = cfg["calibration"]
    model = cfg["model"]
    cycle_durations, cycle_use, integrated_returns = primary_cycle
    event_durations, event_use = event_intervals
    tau_recovery_ms = float(tau_recovery_s) * 1000.0
    tau_axis = np.geomspace(
        float(calibration["tau_depletion_search_ms"][0]),
        float(calibration["tau_depletion_search_ms"][1]),
        int(calibration["tau_depletion_scan_points"]),
    )
    scan_rows: list[dict[str, Any]] = []

    def failed(
        status: str,
        diagnostic: str,
        *,
        monotone: bool | None,
        bracket_count: int,
        residual_min: float | None,
        residual_max: float | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return ({
            "calibration_status": status,
            "root_found": False,
            "numeric_error": status == NUMERIC_ERROR,
            "calibration_diagnostic": diagnostic,
            "tau_recovery_s": float(tau_recovery_s),
            "q_hold": float(q_hold),
            "tau_depletion_ms": None,
            "q_reserve": None,
            "endpoint_residual_q": None,
            "root_scan_monotone": monotone,
            "root_bracket_count": int(bracket_count),
            "scan_residual_min_q": residual_min,
            "scan_residual_max_q": residual_max,
        }, scan_rows)

    def evaluate(tau_depletion_ms: float) -> tuple[float, dict[str, Any], dict[str, Any]]:
        periodic = solve_periodic_q_reserve(
            cycle_durations, cycle_use, q_hold=q_hold,
            q_rest=float(model["q_rest"]), tau_recovery_ms=tau_recovery_ms,
            tau_depletion_ms=float(tau_depletion_ms),
            integrated_returns=integrated_returns,
            q_reserve_lower=float(calibration["q_reserve_search_lower"]),
            q_reserve_upper_margin=float(calibration["q_reserve_upper_margin"]),
            tolerance=float(calibration["root_tolerance_q"]),
        )
        event = exact_q_trace(
            event_durations, event_use, q_initial=float(model["q_rest"]),
            q_rest=float(model["q_rest"]), q_reserve=float(periodic["q_reserve"]),
            tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=float(tau_depletion_ms),
        )
        return float(np.asarray(event["q"])[-1] - float(model["event_final_target_q"])), periodic, event

    evaluations: list[tuple[float, dict[str, Any], dict[str, Any]] | None] = []
    for tau in tau_axis:
        try:
            evaluated = evaluate(float(tau))
            evaluations.append(evaluated)
            scan_rows.append({
                "tau_recovery_s": tau_recovery_s, "q_hold": q_hold,
                "tau_depletion_ms": float(tau), "evaluation_status": "physical",
                "endpoint_residual_q": float(evaluated[0]),
                "q_reserve": float(evaluated[1]["q_reserve"]),
                "evaluation_error": None,
            })
        except NoPhysicalPeriodicReserveRoot as exc:
            evaluations.append(None)
            scan_rows.append({
                "tau_recovery_s": tau_recovery_s, "q_hold": q_hold,
                "tau_depletion_ms": float(tau), "evaluation_status": "no_physical_q_reserve",
                "endpoint_residual_q": None, "q_reserve": None,
                "evaluation_error": str(exc),
            })
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            evaluations.append(None)
            scan_rows.append({
                "tau_recovery_s": tau_recovery_s, "q_hold": q_hold,
                "tau_depletion_ms": float(tau), "evaluation_status": NUMERIC_ERROR,
                "endpoint_residual_q": None, "q_reserve": None,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
            })
            return failed(
                NUMERIC_ERROR,
                f"scan evaluation failed at tau_D={float(tau):.12g} ms: {type(exc).__name__}: {exc}",
                monotone=None, bracket_count=0, residual_min=None, residual_max=None,
            )

    finite_indices = [index for index, value in enumerate(evaluations) if value is not None]
    if len(finite_indices) < 2:
        return failed(
            NO_ROOT_IN_DOMAIN, "fewer than two physical evaluations in the registered tau_D domain",
            monotone=None, bracket_count=0, residual_min=None, residual_max=None,
        )
    residuals = np.asarray([float(evaluations[index][0]) for index in finite_indices], dtype=float)
    differences = np.diff(residuals)
    monotone = bool(np.all(differences > 0.0) or np.all(differences < 0.0))
    residual_min = float(np.min(residuals))
    residual_max = float(np.max(residuals))
    exact_indices = [index for index in finite_indices if float(evaluations[index][0]) == 0.0]
    if exact_indices:
        brackets = [(index, index) for index in exact_indices]
    else:
        brackets = []
        for left, right in zip(finite_indices[:-1], finite_indices[1:]):
            if right == left + 1 and float(evaluations[left][0]) * float(evaluations[right][0]) < 0.0:
                brackets.append((left, right))
    if not monotone:
        return failed(
            NONMONOTONE_SCAN, "registered endpoint residual scan is non-monotone",
            monotone=False, bracket_count=len(brackets),
            residual_min=residual_min, residual_max=residual_max,
        )
    if not brackets:
        return failed(
            NO_ROOT_IN_DOMAIN, "monotone endpoint residual does not cross zero in the registered domain",
            monotone=True, bracket_count=0,
            residual_min=residual_min, residual_max=residual_max,
        )
    if len(brackets) > 1:
        return failed(
            MULTIPLE_ROOTS, f"registered endpoint scan contains {len(brackets)} root brackets",
            monotone=True, bracket_count=len(brackets),
            residual_min=residual_min, residual_max=residual_max,
        )

    left_index, right_index = brackets[0]
    tolerance = float(calibration["root_tolerance_q"])
    if left_index == right_index:
        tau_depletion_ms = float(tau_axis[left_index])
        final = evaluations[left_index]
    else:
        left = float(tau_axis[left_index])
        right = float(tau_axis[right_index])
        left_eval = evaluations[left_index]
        final = None
        try:
            for _ in range(int(calibration["maximum_bisection_iterations"])):
                middle = 0.5 * (left + right)
                middle_eval = evaluate(middle)
                if abs(float(middle_eval[0])) <= tolerance:
                    left = right = middle
                    final = middle_eval
                    break
                if float(left_eval[0]) * float(middle_eval[0]) <= 0.0:
                    right = middle
                else:
                    left = middle
                    left_eval = middle_eval
            tau_depletion_ms = 0.5 * (left + right)
            if final is None:
                final = evaluate(tau_depletion_ms)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return failed(
                NUMERIC_ERROR, f"root refinement failed: {type(exc).__name__}: {exc}",
                monotone=monotone, bracket_count=1,
                residual_min=residual_min, residual_max=residual_max,
            )
    if final is None or abs(float(final[0])) > tolerance:
        return failed(
            NUMERIC_ERROR, "tau_D root missed its endpoint tolerance",
            monotone=monotone, bracket_count=1,
            residual_min=residual_min, residual_max=residual_max,
        )
    return ({
        "calibration_status": ROOT_FOUND,
        "root_found": True,
        "numeric_error": False,
        "calibration_diagnostic": None,
        "tau_recovery_s": float(tau_recovery_s),
        "q_hold": float(q_hold),
        "tau_depletion_ms": float(tau_depletion_ms),
        "q_reserve": float(final[1]["q_reserve"]),
        "endpoint_residual_q": float(final[0]),
        "root_scan_monotone": monotone,
        "root_bracket_count": 1,
        "scan_residual_min_q": residual_min,
        "scan_residual_max_q": residual_max,
    }, scan_rows)


def _periodic_safety(result: dict[str, Any], q_hold: float, cfg: dict[str, Any]) -> bool:
    gate = cfg["periodic_gate"]
    return bool(
        float(result["q_min"]) >= float(gate["minimum_q"])
        and float(result["q_max"]) <= float(gate["maximum_q"])
        and abs(float(result["q_mean"]) - float(q_hold)) <= float(gate["maximum_abs_mean_minus_hold"])
        and float(result["per_return_multiplier"]) < float(gate["maximum_per_return_multiplier"])
        and float(result["closure_error"]) <= float(gate["maximum_closure_error"])
    )


def _event_template(event: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
    probe = cfg["schedule_probes"]
    onset = float(probe["template_source_onset_ms"])
    stop = onset + float(probe["template_stop_after_onset_ms"])
    time = np.asarray(event["time_ms"], dtype=float)
    use = np.asarray(event["use"], dtype=float)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("event template source must have a uniform sample grid")
    left = int(round((onset - time[0]) / dt))
    right = int(round((stop - time[0]) / dt))
    if left < 0 or right >= time.size or not np.isclose(time[left], onset) or not np.isclose(time[right], stop):
        raise RuntimeError("event template endpoints do not align with the sensor")
    return use[left:right + 1].copy(), dt


def synthesize_schedule(
    onsets_ms: Sequence[float],
    template: Sequence[float],
    *,
    dt_ms: float,
    stop_after_last_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    onsets = np.asarray(onsets_ms, dtype=float)
    wave = np.asarray(template, dtype=float)
    if onsets.ndim != 1 or onsets.size == 0 or np.any(np.diff(onsets) <= 0.0):
        raise ValueError("schedule onsets must be strictly increasing")
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)) or np.any(wave < 0.0):
        raise ValueError("event template must be finite and non-negative")
    if dt_ms <= 0.0 or stop_after_last_ms <= 0.0:
        raise ValueError("schedule grid parameters must be positive")
    if not np.allclose(onsets / dt_ms, np.round(onsets / dt_ms), rtol=0.0, atol=1.0e-10):
        raise ValueError("schedule onsets do not align with the sensor grid")
    stop = float(onsets[-1] + stop_after_last_ms)
    n_steps = int(round(stop / dt_ms))
    if not np.isclose(n_steps * dt_ms, stop):
        raise ValueError("schedule stop does not align with the sensor grid")
    time = np.arange(n_steps + 1, dtype=float) * dt_ms
    use = np.zeros(time.size, dtype=float)
    for onset in onsets:
        start = int(round(onset / dt_ms))
        stop_index = min(use.size, start + wave.size)
        use[start:stop_index] = np.maximum(use[start:stop_index], wave[:stop_index - start])
    return time, use


def _schedule_contract(rows: list[dict[str, Any]]) -> bool:
    schedules = {str(row["schedule"]) for row in rows}
    if not schedules:
        return False
    complete_substeps = all(
        sorted(int(row["substeps"]) for row in rows if row["schedule"] == name) == [1, 2]
        for name in schedules
    )
    labels_match = all(
        len({row["outcome"] for row in rows if row["schedule"] == name}) == 1
        for name in schedules
    )
    base = {str(row["schedule"]): row for row in rows if int(row["substeps"]) == 1}
    required = {"isolated", "dense_1200ms", "sparse_3400ms"}
    if not required.issubset(base):
        return False
    qualitative = bool(
        not base["isolated"]["entered"]
        and base["dense_1200ms"]["entered"]
        and not base["sparse_3400ms"]["entered"]
    )
    heldout = [row for name, row in base.items() if name.startswith("heldout_seed_")]
    heldout_no_early = bool(heldout) and all(
        row["entry_event_index"] is None or int(row["entry_event_index"]) >= 4
        for row in heldout
    )
    return bool(complete_substeps and labels_match and qualitative and heldout_no_early)


def _fixed_parameter_sensitivity_contract(
    event_rows: list[dict[str, Any]],
    periodic_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> bool:
    """Fail closed unless both registered sensitivity products are complete."""

    tau_axis = [float(value) for value in cfg["sensitivity"]["fixed_parameter_tau_recovery_s"]]
    q_hold = float(cfg["model"]["preferred_q_hold"])
    substeps = [int(value) for value in cfg["integration"]["event_scalar_substeps"]]
    phases = [float(value) for value in cfg["periodic_gate"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["periodic_gate"]["source_dt_ms"]]

    expected_event = {
        (round(tau, 10), round(q_hold, 10), step)
        for tau in tau_axis for step in substeps
    }
    observed_event = [
        (
            round(float(row["tau_recovery_s"]), 10),
            round(float(row["q_hold"]), 10),
            int(row["substeps"]),
        )
        for row in event_rows
    ]
    expected_periodic = {
        (
            round(tau, 10), round(q_hold, 10),
            round(phase, 10), round(dt, 10),
        )
        for tau in tau_axis for phase in phases for dt in dts
    }
    observed_periodic = [
        (
            round(float(row["tau_recovery_s"]), 10),
            round(float(row["q_hold"]), 10),
            round(float(row["phase"]), 10),
            round(float(row["source_dt_ms"]), 10),
        )
        for row in periodic_rows
    ]
    if (
        len(observed_event) != len(set(observed_event))
        or set(observed_event) != expected_event
        or len(observed_periodic) != len(set(observed_periodic))
        or set(observed_periodic) != expected_periodic
    ):
        return False
    for tau in tau_axis:
        local_events = [
            row for row in event_rows
            if np.isclose(float(row["tau_recovery_s"]), tau)
        ]
        if (
            len({str(row["outcome"]) for row in local_events}) != 1
            or not all(bool(row["post_event_fold_margin_pass"]) for row in local_events)
        ):
            return False
        local_periodic = [
            row for row in periodic_rows
            if np.isclose(float(row["tau_recovery_s"]), tau)
        ]
        if not all(bool(row["sensitivity_periodic_pass"]) for row in local_periodic):
            return False
    return True


def _recovery_time_ms(
    q_start: float, q_target: float, *, q_rest: float, tau_recovery_ms: float,
) -> float:
    if not q_start < q_rest or not q_target < q_rest:
        raise ValueError("recovery states must lie below q_rest")
    if q_start >= q_target:
        return 0.0
    ratio = (q_rest - q_target) / (q_rest - q_start)
    if not 0.0 < ratio < 1.0:
        raise RuntimeError("recovery target does not define a positive crossing time")
    return float(-tau_recovery_ms * np.log(ratio))


def hybrid_handoff_predictor(
    *,
    q_start: float,
    q_rest: float,
    tau_recovery_ms: float,
    additive_start_mv: float,
    fold_q: np.ndarray,
    fold_a: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Implemented hybrid latch: frozen A before reset, exponential A after reset."""

    handoff = cfg["handoff"]
    q_entry = float(cfg["model"]["entry_fold_q"])
    q_safe = float(handoff["q_reset_safe"])
    q_report = float(handoff["q_recovery_report"])
    if not float(fold_q[0]) <= q_start < q_entry < q_safe < q_report < q_rest:
        raise ValueError("handoff q geometry is outside the registered domain")
    t_entry = _recovery_time_ms(q_start, q_entry, q_rest=q_rest, tau_recovery_ms=tau_recovery_ms)
    t_qsafe = _recovery_time_ms(q_start, q_safe, q_rest=q_rest, tau_recovery_ms=tau_recovery_ms)
    t_qreport = _recovery_time_ms(q_start, q_report, q_rest=q_rest, tau_recovery_ms=tau_recovery_ms)
    t_poff = float(handoff["persistence_tau_ms"]) * np.log(
        float(handoff["persistence_start"]) / float(handoff["persistence_off"])
    )
    t_reset = max(t_qsafe, t_poff)

    reporting_dt = float(handoff["reporting_dt_ms"])
    pre_time = np.arange(0.0, np.floor(t_qsafe / reporting_dt) * reporting_dt + reporting_dt, reporting_dt)
    pre_time = pre_time[pre_time <= t_qsafe]
    if pre_time.size == 0 or not np.isclose(pre_time[-1], t_qsafe):
        pre_time = np.r_[pre_time, t_qsafe]
    pre_q = q_rest - (q_rest - q_start) * np.exp(-pre_time / tau_recovery_ms)
    pre_a = np.full_like(pre_q, float(additive_start_mv))
    pre_fold = np.interp(pre_q, fold_q, fold_a, left=np.nan, right=0.0)
    if not np.all(np.isfinite(pre_fold)):
        raise RuntimeError("handoff fold interpolation left its registered domain")
    zero_margin = pre_a - pre_fold
    registered_margin = zero_margin - float(handoff["additive_margin_mv"])

    release_threshold = float(handoff["additive_release_threshold_mv"])
    tau_m_down = float(handoff["tau_m_down_ms"])
    if additive_start_mv <= 0.0 or release_threshold <= 0.0:
        raise ValueError("additive handoff values must be positive")
    release_after_reset = max(0.0, tau_m_down * np.log(additive_start_mv / release_threshold))
    t_release = t_reset + release_after_reset
    q_at_reset = q_rest - (q_rest - q_start) * np.exp(-t_reset / tau_recovery_ms)
    q_at_release = q_rest - (q_rest - q_start) * np.exp(-t_release / tau_recovery_ms)
    monotone_q = bool(q_start <= q_at_reset <= q_at_release <= q_rest)
    protected_margin_pass = bool(float(np.min(registered_margin)) >= -1.0e-12)
    zero_margin_pass = bool(float(np.min(zero_margin)) >= -1.0e-12)
    reset_horizon_pass = bool(t_reset <= float(handoff["reset_horizon_ms"]))
    stage2_pass = bool(q_at_reset >= q_safe - 1.0e-12 and monotone_q)
    return {
        "q_start": float(q_start),
        "additive_start_mv": float(additive_start_mv),
        "time_to_entry_ms": t_entry,
        "time_to_qsafe_ms": t_qsafe,
        "time_to_qreport_ms": t_qreport,
        "persistence_off_bound_ms": float(t_poff),
        "reset_time_ms": float(t_reset),
        "reset_horizon_pass": reset_horizon_pass,
        "minimum_zero_margin_mv": float(np.min(zero_margin)),
        "minimum_registered_margin_mv": float(np.min(registered_margin)),
        "zero_margin_pass": zero_margin_pass,
        "protected_margin_pass": protected_margin_pass,
        "release_after_reset_ms": float(release_after_reset),
        "time_to_additive_release_ms": float(t_release),
        "q_at_reset": float(q_at_reset),
        "q_at_additive_release": float(q_at_release),
        "q_never_decreases": monotone_q,
        "post_reset_release_pass": stage2_pass,
        "handoff_pass": bool(protected_margin_pass and reset_horizon_pass and stage2_pass),
        "trace_time_ms": pre_time,
        "trace_q": pre_q,
        "trace_additive_mv": pre_a,
        "trace_fold_mv": pre_fold,
    }


def _consecutive_components(values: Sequence[bool]) -> list[list[int]]:
    components: list[list[int]] = []
    current: list[int] = []
    for index, accepted in enumerate(values):
        if bool(accepted):
            current.append(index)
        elif current:
            components.append(current)
            current = []
    if current:
        components.append(current)
    return components


def _status(supported: bool, resolved: bool) -> str:
    if supported:
        return "R2_RECOVERY_TIMESCALE_CORRIDOR_SUPPORTED_SHORT_P3_STATE_FORK_ONLY"
    if resolved:
        return "R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES"
    return "R2_RECOVERY_TIMESCALE_CORRIDOR_NUMERICALLY_UNRESOLVED"


def _plot(
    figures: Path,
    cell_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    periodic_rows: list[dict[str, Any]],
    tau_rows: list[dict[str, Any]],
    representative: dict[str, Any],
    gates: dict[str, bool],
    status: str,
    cfg: dict[str, Any],
) -> Path:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    tau_axis = [float(value) for value in cfg["model"]["tau_recovery_s_axis"]]
    q_axis = [float(value) for value in cfg["model"]["q_hold_axis"]]
    colors = {q_axis[0]: "#2166AC", q_axis[1]: "#B2182B", q_axis[2]: "#1B7837"}

    ax = axes[0, 0]
    twin = ax.twinx()
    for q_hold in q_axis:
        rows = [row for row in cell_rows if row["q_hold"] == q_hold and row["root_found"]]
        ax.plot(
            [row["tau_recovery_s"] for row in rows], [row["q_reserve"] for row in rows],
            marker="o", ms=3, color=colors[q_hold], label=f"q_hold={q_hold:.4f}",
        )
        twin.plot(
            [row["tau_recovery_s"] for row in rows], [row["tau_depletion_ms"] for row in rows],
            ls="--", lw=0.9, color=colors[q_hold], alpha=0.8,
        )
    ax.set(xlabel="Recovery timescale tau_r (s)", ylabel="Mapped q_res", title="A  Parameter remapping on the same nullcline")
    twin.set_ylabel("Mapped tau_d (ms)")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[0, 1]
    for q_hold in q_axis:
        base = [
            row for row in event_rows
            if row["q_hold"] == q_hold and int(row["substeps"]) == 1
        ]
        ax.plot(
            [row["tau_recovery_s"] for row in base],
            [row["minimum_q_before_last"] for row in base],
            marker="o", ms=3, color=colors[q_hold], label=f"pre-last {q_hold:.4f}",
        )
        ax.plot(
            [row["tau_recovery_s"] for row in base],
            [row["minimum_q_after_last"] for row in base],
            ls=":", color=colors[q_hold], lw=1.0,
        )
    fold = float(cfg["model"]["entry_fold_q"])
    ax.axhline(fold, color="#762A83", ls="--", lw=0.8, label="entry fold")
    ax.axhline(fold + float(cfg["entry_gate"]["pre_last_margin_q"]), color="0.45", ls="-.", lw=0.7, label="pre-last margin")
    ax.set(xlabel="tau_r (s)", ylabel="Minimum q", title="B  Event-6-first entry gate")
    ax.legend(frameon=False, fontsize=5.8, ncol=2)

    ax = axes[0, 2]
    matrix = np.zeros((len(q_axis), len(tau_axis)), dtype=float)
    annotations = np.empty(matrix.shape, dtype=object)
    for i, q_hold in enumerate(q_axis):
        for j, tau in enumerate(tau_axis):
            row = next(item for item in cell_rows if item["q_hold"] == q_hold and item["tau_recovery_s"] == tau)
            matrix[i, j] = 1.0 if row["cell_pass"] else 0.0
            annotations[i, j] = "PASS" if row["cell_pass"] else str(row["failure_code"])
    ax.imshow(matrix, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#F4A582", "#92C5DE"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(tau_axis)), [f"{value:g}" for value in tau_axis], rotation=45)
    ax.set_yticks(range(len(q_axis)), [f"{value:.4f}" for value in q_axis])
    for i in range(len(q_axis)):
        for j in range(len(tau_axis)):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=5.5)
    ax.set(
        xlabel="tau_r (s)", ylabel="q_hold",
        title="C  Cell-wise entry + hold + handoff\nE = entry gate; H = handoff gate",
    )

    ax = axes[1, 0]
    preferred_q = float(cfg["model"]["preferred_q_hold"])
    for tau in tau_axis:
        rows = [
            row for row in periodic_rows
            if row["q_hold"] == preferred_q and row["tau_recovery_s"] == tau
        ]
        ax.vlines(tau, min(row["q_min"] for row in rows), max(row["q_max"] for row in rows), color="#2166AC", lw=1.2)
        ax.scatter([tau], [np.mean([row["q_mean"] for row in rows])], color="#B2182B", s=14)
    ax.axhline(float(cfg["periodic_gate"]["minimum_q"]), color="0.5", ls="--", lw=0.7)
    ax.axhline(float(cfg["periodic_gate"]["maximum_q"]), color="0.5", ls="--", lw=0.7)
    ax.axhline(preferred_q, color="0.35", ls=":", lw=0.8)
    ax.set(xlabel="tau_r (s)", ylabel="Periodic q range", title="D  Complete phase x source-dt hold oracle")

    ax = axes[1, 1]
    handoff = representative["handoff"]
    time_s = np.asarray(handoff["trace_time_ms"], dtype=float) * 1.0e-3
    ax.plot(time_s, handoff["trace_q"], color="#2166AC", lw=1.1, label="q recovery")
    ax.axhline(float(cfg["handoff"]["q_reset_safe"]), color="#2166AC", ls="--", lw=0.7, label="q_safe=.885")
    ax2 = ax.twinx()
    ax2.plot(time_s, handoff["trace_additive_mv"], color="#B2182B", lw=1.0, label="A frozen")
    ax2.plot(
        time_s,
        np.asarray(handoff["trace_fold_mv"]) + float(cfg["handoff"]["additive_margin_mv"]),
        color="#1B7837", ls="--", lw=0.9, label="A_fold+.025",
    )
    reset_s = float(handoff["reset_time_ms"]) * 1.0e-3
    release_s = float(handoff["time_to_additive_release_ms"]) * 1.0e-3
    post_time_s = np.linspace(reset_s, release_s, 300)
    post_a = float(handoff["additive_start_mv"]) * np.exp(
        -(post_time_s - reset_s) * 1000.0 / float(cfg["handoff"]["tau_m_down_ms"])
    )
    ax2.plot(post_time_s, post_a, color="#B2182B", lw=1.0, alpha=0.75, label="A decay after reset")
    ax.axvline(reset_s, color="0.3", ls=":", lw=0.8)
    ax.set(xlabel="Time after exit (s)", ylabel="q", title="E  Two-stage latch-reset handoff (80 s, q=.8425)")
    ax2.set_ylabel("additive A (mV)")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=5.7, loc="center right")

    ax = axes[1, 2]
    ax.axis("off")
    passing_tau = [row["tau_recovery_s"] for row in tau_rows if row["tau_node_pass"]]
    lines = ["F  R2 scalar acceptance", "", f"Status: {status}", ""]
    lines.extend(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    lines.extend(["", f"All-q_hold passing tau_r: {passing_tau}", "", "A pass unlocks only the [60,70,80] s", "short P3 regional state-fork.", "No autonomous / field / SNN claim."])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="DejaVu Sans", fontsize=5.45)

    fig.suptitle("R2 inhibitory-reserve recovery timescale: entry-hold-hybrid handoff corridor", fontsize=13.0, fontweight="bold")
    stem = figures / "mz_inhibitory_reserve_recovery_corridor"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    (
        hashes, mapping, cycle, cycle_rows, event, r0b, ramp_rows,
        a0_by_q, fold_q, fold_a,
    ) = _load_and_validate_inputs(cfg)
    output = ROOT / str(cfg["result_root"])
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    model = cfg["model"]
    tau_axis = [float(value) for value in model["tau_recovery_s_axis"]]
    q_axis = [float(value) for value in model["q_hold_axis"]]
    substeps_axis = [int(value) for value in cfg["integration"]["event_scalar_substeps"]]
    cycle_intervals: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray, int]] = {}
    for index, row in enumerate(cycle_rows):
        durations, sensor = _cycle_intervals(cycle, row, index)
        cycle_intervals[_key(row["q_hold"], row["phase"], row["dt_ms"])] = (
            durations, sensor, int(row["integrated_returns"]),
        )
    event_durations, event_use = _event_intervals(event)
    onsets = [float(value) for value in cfg["locked_schedule"]["onsets_ms"]]

    template, template_dt = _event_template(event, cfg)
    schedules: dict[str, tuple[np.ndarray, np.ndarray, list[float]]] = {}
    for name, raw_onsets in cfg["schedule_probes"].items():
        if name in {"template_source_onset_ms", "template_stop_after_onset_ms", "stop_after_last_onset_ms"}:
            continue
        probe_onsets = [float(value) for value in raw_onsets]
        probe_time, probe_use = synthesize_schedule(
            probe_onsets, template, dt_ms=template_dt,
            stop_after_last_ms=float(cfg["schedule_probes"]["stop_after_last_onset_ms"]),
        )
        schedules[name] = (np.diff(probe_time), probe_use[:-1], probe_onsets)

    mapping_rows: list[dict[str, Any]] = []
    root_scan_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    periodic_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}

    for tau_recovery_s in tau_axis:
        for q_hold in q_axis:
            primary_key = _key(
                q_hold, float(cfg["calibration"]["primary_cycle_phase"]),
                float(cfg["calibration"]["primary_cycle_dt_ms"]),
            )
            calibrated, scan = _calibrate_cell(
                tau_recovery_s, q_hold, cycle_intervals[primary_key],
                (event_durations, event_use), cfg,
            )
            root_scan_rows.extend(scan)
            root_found = bool(calibrated["root_found"])
            physical = bool(
                root_found
                and calibrated["tau_depletion_ms"] is not None
                and calibrated["q_reserve"] is not None
                and float(calibrated["tau_depletion_ms"]) > 0.0
                and 0.0 < float(calibrated["q_reserve"]) < q_hold
            )
            entry_pass = False
            base_half_match = False
            periodic_pass = False
            handoff_pass = False
            periodic_q_min = None
            periodic_q_max = None
            failure_codes: list[str] = []

            if root_found:
                q_reserve = float(calibrated["q_reserve"])
                tau_depletion_ms = float(calibrated["tau_depletion_ms"])
                local_event_rows = []
                for substeps in substeps_axis:
                    trace = exact_q_trace(
                        event_durations, event_use, q_initial=float(model["q_rest"]),
                        q_rest=float(model["q_rest"]), q_reserve=q_reserve,
                        tau_recovery_ms=tau_recovery_s * 1000.0,
                        tau_depletion_ms=tau_depletion_ms, substeps=substeps,
                    )
                    classification = classify_event_replay(
                        trace, onsets, entry_fold_q=float(model["entry_fold_q"]),
                        final_target_q=float(model["event_final_target_q"]),
                        pre_last_margin_q=float(cfg["entry_gate"]["pre_last_margin_q"]),
                    )
                    row = {
                        "tau_recovery_s": tau_recovery_s, "q_hold": q_hold,
                        "substeps": substeps, **classification,
                    }
                    event_rows.append(row)
                    local_event_rows.append(row)
                    if tau_recovery_s == float(model["preferred_tau_recovery_s"]) and q_hold == float(model["preferred_q_hold"]):
                        representative[f"event_substeps_{substeps}"] = trace
                base_half_match = len({row["outcome"] for row in local_event_rows}) == 1
                entry_pass = bool(base_half_match and all(row["entry_pass"] for row in local_event_rows))

                local_periodic_rows = []
                for key, (durations, sensor, returns) in cycle_intervals.items():
                    if key[0] != round(q_hold, 10):
                        continue
                    periodic = exact_periodic_q_orbit(
                        durations, sensor, q_rest=float(model["q_rest"]),
                        q_reserve=q_reserve, tau_recovery_ms=tau_recovery_s * 1000.0,
                        tau_depletion_ms=tau_depletion_ms, integrated_returns=returns,
                    )
                    safe = _periodic_safety(periodic, q_hold, cfg)
                    row = {
                        "tau_recovery_s": tau_recovery_s, "q_hold": q_hold,
                        "phase": key[1], "source_dt_ms": key[2],
                        "q_min": periodic["q_min"], "q_max": periodic["q_max"],
                        "q_mean": periodic["q_mean"],
                        "q_mean_minus_hold": float(periodic["q_mean"] - q_hold),
                        "per_return_multiplier": periodic["per_return_multiplier"],
                        "closure_error": periodic["closure_error"],
                        "periodic_pass": safe,
                    }
                    periodic_rows.append(row)
                    local_periodic_rows.append(row)
                periodic_pass = bool(len(local_periodic_rows) == 8 and all(row["periodic_pass"] for row in local_periodic_rows))
                periodic_q_min = min(row["q_min"] for row in local_periodic_rows)
                periodic_q_max = max(row["q_max"] for row in local_periodic_rows)

                handoff = hybrid_handoff_predictor(
                    q_start=float(periodic_q_min), q_rest=float(model["q_rest"]),
                    tau_recovery_ms=tau_recovery_s * 1000.0,
                    additive_start_mv=float(a0_by_q[q_hold]),
                    fold_q=fold_q, fold_a=fold_a, cfg=cfg,
                )
                trace_fields = {
                    name: handoff.pop(name)
                    for name in ("trace_time_ms", "trace_q", "trace_additive_mv", "trace_fold_mv")
                }
                handoff_row = {"tau_recovery_s": tau_recovery_s, "q_hold": q_hold, **handoff}
                handoff_rows.append(handoff_row)
                handoff_pass = bool(handoff["handoff_pass"])
                if tau_recovery_s == float(model["preferred_tau_recovery_s"]) and q_hold == float(model["preferred_q_hold"]):
                    representative["handoff"] = {**handoff, **trace_fields}

            if not root_found:
                failure_codes.append("R")
            else:
                if not physical:
                    failure_codes.append("P")
                if not entry_pass:
                    failure_codes.append("E")
                if not periodic_pass:
                    failure_codes.append("O")
                if not handoff_pass:
                    failure_codes.append("H")
            cell_pass = bool(physical and entry_pass and base_half_match and periodic_pass and handoff_pass)
            mapping_rows.append({
                **calibrated,
                "physical_root": physical,
                "base_half_event_labels_match": base_half_match,
                "entry_pass": entry_pass,
                "periodic_pass": periodic_pass,
                "periodic_q_min": periodic_q_min,
                "periodic_q_max": periodic_q_max,
                "handoff_pass": handoff_pass,
                "cell_pass": cell_pass,
                "failure_code": "PASS" if cell_pass else "".join(failure_codes),
            })

    schedule_rows: list[dict[str, Any]] = []
    schedule_pass_by_tau: dict[float, bool] = {}
    preferred_q = float(model["preferred_q_hold"])
    for tau_recovery_s in tau_axis:
        cell = next(row for row in mapping_rows if row["tau_recovery_s"] == tau_recovery_s and row["q_hold"] == preferred_q)
        local_rows: list[dict[str, Any]] = []
        if cell["root_found"]:
            for name, (durations, sensor, probe_onsets) in schedules.items():
                for substeps in substeps_axis:
                    trace = exact_q_trace(
                        durations, sensor, q_initial=float(model["q_rest"]),
                        q_rest=float(model["q_rest"]), q_reserve=float(cell["q_reserve"]),
                        tau_recovery_ms=tau_recovery_s * 1000.0,
                        tau_depletion_ms=float(cell["tau_depletion_ms"]), substeps=substeps,
                    )
                    classification = classify_event_replay(
                        trace, probe_onsets, entry_fold_q=float(model["entry_fold_q"]),
                        final_target_q=float(model["event_final_target_q"]),
                        pre_last_margin_q=float(cfg["entry_gate"]["pre_last_margin_q"]),
                    )
                    row = {
                        "tau_recovery_s": tau_recovery_s, "q_hold": preferred_q,
                        "schedule": name, "event_count": len(probe_onsets),
                        "substeps": substeps, **classification,
                    }
                    schedule_rows.append(row)
                    local_rows.append(row)
        schedule_pass_by_tau[tau_recovery_s] = _schedule_contract(local_rows)

    primary_cell = next(
        row for row in mapping_rows
        if row["tau_recovery_s"] == float(model["preferred_tau_recovery_s"])
        and row["q_hold"] == preferred_q
    )
    sensitivity_event_rows: list[dict[str, Any]] = []
    sensitivity_periodic_rows: list[dict[str, Any]] = []
    sensitivity_pass = False
    if primary_cell["root_found"]:
        for tau_sensitivity in map(float, cfg["sensitivity"]["fixed_parameter_tau_recovery_s"]):
            local_events = []
            for substeps in substeps_axis:
                trace = exact_q_trace(
                    event_durations, event_use, q_initial=float(model["q_rest"]),
                    q_rest=float(model["q_rest"]), q_reserve=float(primary_cell["q_reserve"]),
                    tau_recovery_ms=tau_sensitivity * 1000.0,
                    tau_depletion_ms=float(primary_cell["tau_depletion_ms"]), substeps=substeps,
                )
                classification = classify_event_replay(
                    trace, onsets, entry_fold_q=float(model["entry_fold_q"]),
                    final_target_q=float(model["event_final_target_q"]),
                    pre_last_margin_q=float(cfg["entry_gate"]["pre_last_margin_q"]),
                )
                robust = bool(
                    classification["entry_event_index"] == len(onsets)
                    and classification["pre_last_margin_pass"]
                    and classification["minimum_q_after_last"]
                    <= float(model["entry_fold_q"]) - float(cfg["sensitivity"]["post_event_below_fold_margin_q"])
                )
                row = {
                    "tau_recovery_s": tau_sensitivity, "q_hold": preferred_q,
                    "substeps": substeps, **classification,
                    "post_event_fold_margin_pass": robust,
                }
                sensitivity_event_rows.append(row)
                local_events.append(row)
            local_periodic = []
            for key, (durations, sensor, returns) in cycle_intervals.items():
                if key[0] != round(preferred_q, 10):
                    continue
                periodic = exact_periodic_q_orbit(
                    durations, sensor, q_rest=float(model["q_rest"]),
                    q_reserve=float(primary_cell["q_reserve"]),
                    tau_recovery_ms=tau_sensitivity * 1000.0,
                    tau_depletion_ms=float(primary_cell["tau_depletion_ms"]),
                    integrated_returns=returns,
                )
                safe = bool(
                    periodic["q_min"] >= float(cfg["periodic_gate"]["minimum_q"])
                    and periodic["q_max"] <= float(cfg["periodic_gate"]["maximum_q"])
                    and abs(periodic["q_mean"] - preferred_q)
                    <= float(cfg["periodic_gate"]["maximum_abs_mean_minus_hold"])
                )
                row = {
                    "tau_recovery_s": tau_sensitivity, "q_hold": preferred_q,
                    "phase": key[1], "source_dt_ms": key[2],
                    "q_min": periodic["q_min"], "q_max": periodic["q_max"],
                    "q_mean": periodic["q_mean"],
                    "q_mean_minus_hold": float(periodic["q_mean"] - preferred_q),
                    "per_return_multiplier": periodic["per_return_multiplier"],
                    "closure_error": periodic["closure_error"],
                    "sensitivity_periodic_pass": safe,
                }
                sensitivity_periodic_rows.append(row)
                local_periodic.append(row)
        sensitivity_pass = _fixed_parameter_sensitivity_contract(
            sensitivity_event_rows, sensitivity_periodic_rows, cfg,
        )

    tau_rows: list[dict[str, Any]] = []
    for tau_recovery_s in tau_axis:
        cells = [row for row in mapping_rows if row["tau_recovery_s"] == tau_recovery_s]
        all_cells = bool(len(cells) == len(q_axis) and all(row["cell_pass"] for row in cells))
        schedule_pass = bool(schedule_pass_by_tau.get(tau_recovery_s, False))
        tau_rows.append({
            "tau_index": tau_axis.index(tau_recovery_s),
            "tau_recovery_s": tau_recovery_s,
            "all_qhold_cells_pass": all_cells,
            "primary_qhold_schedule_pass": schedule_pass,
            "tau_node_pass": bool(all_cells and schedule_pass),
        })
    components = _consecutive_components([row["tau_node_pass"] for row in tau_rows])
    preferred_index = tau_axis.index(float(model["preferred_tau_recovery_s"]))
    preferred_component = next((component for component in components if preferred_index in component), [])
    minimum_component = int(cfg["acceptance"]["minimum_consecutive_tau_nodes"])

    expected_scan_rows = len(tau_axis) * len(q_axis) * int(cfg["calibration"]["tau_depletion_scan_points"])
    resolution_gates = {
        "complete_30_cell_mapping": len(mapping_rows) == len(tau_axis) * len(q_axis),
        "complete_registered_root_scan": len(root_scan_rows) == expected_scan_rows,
        "no_numeric_calibration_errors": not any(bool(row["numeric_error"]) for row in mapping_rows),
        "all_found_roots_unique_monotone_physical": all(
            not row["root_found"]
            or (
                row["root_scan_monotone"]
                and row["root_bracket_count"] == 1
                and row["physical_root"]
            )
            for row in mapping_rows
        ),
    }
    gates = {
        "hash_locked_inputs_and_R0b_R1_provenance_valid": True,
        **resolution_gates,
        "three_node_component_contains_preregistered_80s": len(preferred_component) >= minimum_component,
        "accepted_nodes_pass_all_qhold_entry_periodic_handoff": all(
            row["all_qhold_cells_pass"] for row in tau_rows if row["tau_node_pass"]
        ) and bool(preferred_component),
        "accepted_nodes_pass_primary_schedule_contract": all(
            row["primary_qhold_schedule_pass"] for row in tau_rows if row["tau_node_pass"]
        ) and bool(preferred_component),
        "fixed_parameter_72_88s_sensitivity_pass": sensitivity_pass,
        "thresholded_eligibility_not_used": not bool(cfg["scope"]["thresholded_eligibility_used"]),
        "passing_scope_is_short_P3_state_fork_only": True,
    }
    resolved_names = (
        "hash_locked_inputs_and_R0b_R1_provenance_valid", "complete_30_cell_mapping",
        "complete_registered_root_scan", "no_numeric_calibration_errors",
        "all_found_roots_unique_monotone_physical", "thresholded_eligibility_not_used",
        "passing_scope_is_short_P3_state_fork_only",
    )
    resolved = all(bool(gates[name]) for name in resolved_names)
    supported = all(bool(value) for value in gates.values())
    status = _status(supported, resolved)
    decision = (
        "run_short_P3_regional_state_fork_at_tau_r_60_70_80s_only"
        if supported else "do_not_run_coupled_R2_and_proceed_to_two_pool_resource_design"
    )

    artifacts = {
        "summary": str((output / "recovery_corridor_summary.json").relative_to(ROOT)),
        "mapping_csv": str((output / "recovery_corridor_mapping.csv").relative_to(ROOT)),
        "root_scan_csv": str((output / "recovery_corridor_root_scan.csv").relative_to(ROOT)),
        "event_csv": str((output / "recovery_corridor_event_entry.csv").relative_to(ROOT)),
        "periodic_csv": str((output / "recovery_corridor_periodic_oracle.csv").relative_to(ROOT)),
        "handoff_csv": str((output / "recovery_corridor_hybrid_handoff.csv").relative_to(ROOT)),
        "schedule_csv": str((output / "recovery_corridor_schedule_probes.csv").relative_to(ROOT)),
        "sensitivity_event_csv": str((output / "recovery_corridor_sensitivity_event.csv").relative_to(ROOT)),
        "sensitivity_periodic_csv": str((output / "recovery_corridor_sensitivity_periodic.csv").relative_to(ROOT)),
        "tau_acceptance_csv": str((output / "recovery_corridor_tau_acceptance.csv").relative_to(ROOT)),
        "representative_npz": str((output / "recovery_corridor_representative_traces.npz").relative_to(ROOT)),
        "figure": str((figures / "mz_inhibitory_reserve_recovery_corridor.png").relative_to(ROOT)),
    }
    summary = {
        "status": status,
        "scientific_layer": "pilot_informed_scalar_recovery_timescale_continuation_with_hybrid_handoff_predictor",
        "decision": decision,
        "gates": gates,
        "registered_cell_count": len(mapping_rows),
        "root_found_cell_count": sum(bool(row["root_found"]) for row in mapping_rows),
        "numeric_error_cell_count": sum(bool(row["numeric_error"]) for row in mapping_rows),
        "root_scan_row_count": len(root_scan_rows),
        "expected_root_scan_row_count": expected_scan_rows,
        "entry_pass_cell_count": sum(bool(row["entry_pass"]) for row in mapping_rows),
        "periodic_pass_cell_count": sum(bool(row["periodic_pass"]) for row in mapping_rows),
        "handoff_pass_cell_count": sum(bool(row["handoff_pass"]) for row in mapping_rows),
        "cell_pass_count": sum(bool(row["cell_pass"]) for row in mapping_rows),
        "passing_tau_nodes": [row["tau_recovery_s"] for row in tau_rows if row["tau_node_pass"]],
        "preferred_component_tau_nodes": [tau_axis[index] for index in preferred_component],
        "fixed_parameter_sensitivity_pass": sensitivity_pass,
        "fixed_parameter_sensitivity_event_row_count": len(sensitivity_event_rows),
        "expected_fixed_parameter_sensitivity_event_row_count": (
            len(cfg["sensitivity"]["fixed_parameter_tau_recovery_s"])
            * len(cfg["integration"]["event_scalar_substeps"])
        ),
        "fixed_parameter_sensitivity_periodic_row_count": len(sensitivity_periodic_rows),
        "expected_fixed_parameter_sensitivity_periodic_row_count": (
            len(cfg["sensitivity"]["fixed_parameter_tau_recovery_s"])
            * len(cfg["periodic_gate"]["relative_phase_fractions"])
            * len(cfg["periodic_gate"]["source_dt_ms"])
        ),
        "parameter_mapping": mapping_rows,
        "tau_acceptance": tau_rows,
        "a0_by_q_hold": {str(key): value for key, value in a0_by_q.items()},
        "fold_interpolation": {"q": fold_q.tolist(), "additive_mv": fold_a.tolist()},
        "input_sha256": hashes,
        "mapping_provenance_status": mapping["status"],
        "r0b_provenance_status": r0b["status"],
        "claim_boundary": [
            "q is replayed on hash-locked frozen sensors and is not coupled back to the fast regional model",
            "the postictal predictor preserves the implemented frozen-A latch stage before reset and only then releases A with tau_m_down=12 s",
            "passing licenses only the registered short P3 regional state-fork at tau_r=60/70/80 s",
            "the fixed bath mask remains non-emergent and no continuous field, full SNN, autonomous reset/retrigger, E-E, conductance, relay, or thresholded eligibility run is unlocked",
        ],
        "config": cfg,
        "artifacts": artifacts,
        "plot_status": "pending",
    }

    # All scientific tables and representative traces are persisted before plotting.
    _save_csv(output / "recovery_corridor_mapping.csv", mapping_rows)
    _save_csv(output / "recovery_corridor_root_scan.csv", root_scan_rows)
    _save_csv(output / "recovery_corridor_event_entry.csv", event_rows)
    _save_csv(output / "recovery_corridor_periodic_oracle.csv", periodic_rows)
    _save_csv(output / "recovery_corridor_hybrid_handoff.csv", handoff_rows)
    _save_csv(output / "recovery_corridor_schedule_probes.csv", schedule_rows)
    _save_csv(output / "recovery_corridor_sensitivity_event.csv", sensitivity_event_rows)
    _save_csv(output / "recovery_corridor_sensitivity_periodic.csv", sensitivity_periodic_rows)
    _save_csv(output / "recovery_corridor_tau_acceptance.csv", tau_rows)
    np.savez_compressed(
        output / "recovery_corridor_representative_traces.npz",
        event_time_base_ms=np.asarray(representative["event_substeps_1"]["time_ms"]),
        event_q_base=np.asarray(representative["event_substeps_1"]["q"]),
        event_time_half_ms=np.asarray(representative["event_substeps_2"]["time_ms"]),
        event_q_half=np.asarray(representative["event_substeps_2"]["q"]),
        handoff_time_ms=np.asarray(representative["handoff"]["trace_time_ms"]),
        handoff_q=np.asarray(representative["handoff"]["trace_q"]),
        handoff_additive_mv=np.asarray(representative["handoff"]["trace_additive_mv"]),
        handoff_fold_mv=np.asarray(representative["handoff"]["trace_fold_mv"]),
    )
    _save_json(output / "recovery_corridor_summary.json", summary)

    figures.mkdir(parents=True, exist_ok=True)
    figure = _plot(
        figures, mapping_rows, event_rows, periodic_rows, tau_rows,
        representative, gates, status, cfg,
    )
    summary["artifacts"]["figure"] = str(figure.relative_to(ROOT))
    summary["plot_status"] = "complete"
    _save_json(output / "recovery_corridor_summary.json", summary)
    (figures / "README.md").write_text(
        "### mz_inhibitory_reserve_recovery_corridor.png\n\n"
        "这张 2×3 图检验不改方程、只延长 inhibitory-reserve recovery timescale，能否同时修复第六事件首次 entry、维持 bounded CCO，并通过真实 latch 语义下的两阶段 postictal handoff。A 显示每个 q_hold 的 q_res/tau_D 重映射；B 显示 pre-last 与 post-last q；C 给出 30 个 cell 的联合验收；D 汇总完整 phase×source-dt 周期范围；E 展示 A 在 reset 前冻结、reset 后才衰减；F 锁定机制级结论。\n\n"
        f"当前状态为 `{status}`，全 q_hold 联合通过的 tau_r 节点为 {[row['tau_recovery_s'] for row in tau_rows if row['tau_node_pass']]}，80 s 所在连续 component 为 {[tau_axis[index] for index in preferred_component]}。\n\n"
        "即使通过，本节点也只允许 [60,70,80] s 的短 P3 regional state-fork；fixed bath mask 仍是非涌现的，不能写成 autonomous lifecycle、continuous spatial containment 或 full SNN seizure。\n\n"
        "**关注点**：必须同时查看 entry ordering、完整 periodic oracle、locked schedule probes、72/88 s fixed-parameter sensitivity，以及 reset 前 A 冻结的 hybrid handoff，而不能只看某个 tau_r 点。\n",
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
        "decision": summary["decision"],
        "gates": summary["gates"],
        "passing_tau_nodes": summary["passing_tau_nodes"],
        "preferred_component_tau_nodes": summary["preferred_component_tau_nodes"],
        "fixed_parameter_sensitivity_pass": summary["fixed_parameter_sensitivity_pass"],
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
