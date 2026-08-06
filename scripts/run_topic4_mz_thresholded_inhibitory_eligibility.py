#!/usr/bin/env python3
"""Cheap scalar screen for thresholded inhibitory-use eligibility.

This producer consumes hash-locked frozen sensor traces.  It does not run the
spatial fast model, change E-to-E coupling, or claim an autonomous lifecycle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

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

from scripts.run_topic4_mz_inhibitory_reserve_periodic_oracle import (
    extract_piecewise_constant_window,
)


DEFAULT_CONFIG = ROOT / "config/topic4_mz_thresholded_inhibitory_eligibility.yaml"

ROOT_FOUND = "root_found"
NO_ROOT_IN_DOMAIN = "no_root_in_registered_physical_domain"
NONMONOTONE_SCAN = "nonmonotone_registered_scan"
MULTIPLE_ROOTS = "multiple_roots_in_registered_domain"
NUMERIC_ERROR = "numeric_error"


class NoPhysicalPeriodicReserveRoot(RuntimeError):
    """The registered q_res search does not bracket a physical periodic hold."""


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
    """Write strict RFC-8259 JSON and reject accidental NaN/Infinity evidence."""

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _load_cycle_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "q_hold", "phase", "dt_ms", "cycle_window_start_ms",
        "cycle_window_stop_ms", "integrated_returns", "outcome",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("cycle measurement table is missing locked window fields")
        rows = []
        for raw in reader:
            rows.append({
                "q_hold": float(raw["q_hold"]),
                "phase": float(raw["phase"]),
                "dt_ms": float(raw["dt_ms"]),
                "cycle_window_start_ms": float(raw["cycle_window_start_ms"]),
                "cycle_window_stop_ms": float(raw["cycle_window_stop_ms"]),
                "integrated_returns": int(raw["integrated_returns"]),
                "outcome": str(raw["outcome"]),
            })
    return rows


def eligibility_gate(
    h: float | np.ndarray,
    theta_h: float,
    width: float,
) -> np.ndarray:
    """Registered smooth eligibility gate."""

    values = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("eligibility state must be finite")
    if not np.isfinite(theta_h) or not np.isfinite(width) or theta_h < 0.0 or width <= 0.0:
        raise ValueError("eligibility threshold and width are invalid")
    return 0.5 * (1.0 + np.tanh((values - float(theta_h)) / float(width)))


def _validate_intervals(
    durations_ms: Sequence[float],
    use: Sequence[float],
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
    durations_ms: Sequence[float],
    use: Sequence[float],
    substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    durations, sensor = _validate_intervals(durations_ms, use)
    if int(substeps) != substeps or int(substeps) < 1:
        raise ValueError("scalar substeps must be a positive integer")
    count = int(substeps)
    return np.repeat(durations / count, count), np.repeat(sensor, count)


def _affine_trace(
    alpha: np.ndarray,
    beta: np.ndarray,
    initial: float,
) -> np.ndarray:
    """Vectorized trace for x[n+1] = alpha[n] x[n] + beta[n]."""

    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or a.size == 0:
        raise ValueError("affine coefficients must be aligned vectors")
    prefix = np.r_[1.0, np.cumprod(a)]
    if np.any(prefix <= 0.0) or not np.all(np.isfinite(prefix)):
        raise RuntimeError("affine prefix product left the finite positive domain")
    accumulated = np.r_[0.0, np.cumsum(b / prefix[1:])]
    return prefix * (float(initial) + accumulated)


def eligibility_trace(
    durations_ms: Sequence[float],
    use: Sequence[float],
    *,
    tau_h_ms: float,
    theta_h: float,
    gate_width: float,
    substeps: int,
    periodic: bool,
    initial_h: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Integrate H exactly under ZOH use and evaluate g(H) at midsteps."""

    if not np.isfinite(tau_h_ms) or tau_h_ms <= 0.0:
        raise ValueError("tau_h_ms must be finite and positive")
    durations, sensor = _subdivide_intervals(durations_ms, use, substeps)
    alpha = np.exp(-durations / float(tau_h_ms))
    beta = sensor * (1.0 - alpha)
    map_alpha = float(np.prod(alpha))
    zero_end = float(_affine_trace(alpha, beta, 0.0)[-1])
    if periodic:
        denominator = 1.0 - map_alpha
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError("periodic H map is not contractive")
        h0 = zero_end / denominator
    else:
        if not np.isfinite(initial_h) or initial_h < 0.0:
            raise ValueError("initial_h must be finite and non-negative")
        h0 = float(initial_h)
    trace_h = _affine_trace(alpha, beta, h0)
    half_alpha = np.exp(-0.5 * durations / float(tau_h_ms))
    h_mid = sensor + (trace_h[:-1] - sensor) * half_alpha
    gate = eligibility_gate(h_mid, theta_h, gate_width)
    return {
        "durations_ms": durations,
        "use": sensor,
        "h": trace_h,
        "h_mid": h_mid,
        "gate": gate,
        "weighted_use": sensor * gate,
        "map_alpha": map_alpha,
        "time_ms": np.r_[0.0, np.cumsum(durations)],
    }


def q_trace(
    durations_ms: Sequence[float],
    weighted_use: Sequence[float],
    *,
    q_initial: float,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
) -> dict[str, np.ndarray | float]:
    """Exact ZOH q trajectory and segment integrals."""

    durations, weight = _validate_intervals(durations_ms, weighted_use)
    values = (q_initial, q_rest, q_reserve, tau_recovery_ms, tau_depletion_ms)
    if not all(np.isfinite(values)):
        raise ValueError("q integration inputs must be finite")
    if not 0.0 < q_reserve < q_rest <= 1.0:
        raise ValueError("q bounds must satisfy 0<q_reserve<q_rest<=1")
    if tau_recovery_ms <= 0.0 or tau_depletion_ms <= 0.0:
        raise ValueError("q time constants must be positive")
    decay = 1.0 / float(tau_recovery_ms) + weight / float(tau_depletion_ms)
    drive = float(q_rest) / float(tau_recovery_ms) + q_reserve * weight / float(tau_depletion_ms)
    equilibrium = drive / decay
    alpha = np.exp(-decay * durations)
    beta = equilibrium * (1.0 - alpha)
    trace = _affine_trace(alpha, beta, float(q_initial))
    integrals = (
        equilibrium * durations
        + (trace[:-1] - equilibrium) * (1.0 - alpha) / decay
    )
    return {
        "q": trace,
        "integral": float(np.sum(integrals)),
        "map_alpha": float(np.prod(alpha)),
        "time_ms": np.r_[0.0, np.cumsum(durations)],
    }


def periodic_q_orbit(
    durations_ms: Sequence[float],
    weighted_use: Sequence[float],
    *,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    integrated_returns: int,
) -> dict[str, Any]:
    """Exact periodic q orbit for one registered sensor window."""

    durations, weight = _validate_intervals(durations_ms, weighted_use)
    if integrated_returns <= 0:
        raise ValueError("integrated_returns must be positive")
    low = q_trace(
        durations, weight, q_initial=0.0, q_rest=q_rest,
        q_reserve=q_reserve, tau_recovery_ms=tau_recovery_ms,
        tau_depletion_ms=tau_depletion_ms,
    )
    alpha = float(low["map_alpha"])
    beta = float(np.asarray(low["q"])[-1])
    if not 0.0 <= alpha < 1.0:
        raise RuntimeError("periodic q map is not contractive")
    q_strobe = beta / (1.0 - alpha)
    orbit = q_trace(
        durations, weight, q_initial=q_strobe, q_rest=q_rest,
        q_reserve=q_reserve, tau_recovery_ms=tau_recovery_ms,
        tau_depletion_ms=tau_depletion_ms,
    )
    trace = np.asarray(orbit["q"], dtype=float)
    duration = float(np.sum(durations))
    return {
        "q_min": float(np.min(trace)),
        "q_max": float(np.max(trace)),
        "q_mean": float(orbit["integral"]) / duration,
        "q_stroboscopic": float(q_strobe),
        "window_rho": alpha,
        "per_cycle_multiplier": float(alpha ** (1.0 / float(integrated_returns))),
        "closure_error": float(abs(trace[-1] - trace[0])),
        "time_ms": np.asarray(orbit["time_ms"]),
        "q": trace,
    }


def solve_periodic_q_reserve(
    durations_ms: Sequence[float],
    weighted_use: Sequence[float],
    *,
    q_hold: float,
    q_rest: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    integrated_returns: int,
    q_reserve_search: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    """Solve the unique affine q_res value giving the target periodic mean."""

    bounds = np.asarray(q_reserve_search, dtype=float)
    if bounds.shape != (2,) or not 0.0 < bounds[0] < bounds[1] < q_hold < q_rest:
        raise ValueError("q_reserve_search must bracket a physical floor below q_hold")
    if tolerance <= 0.0:
        raise ValueError("periodic solve tolerance must be positive")
    endpoint = [
        periodic_q_orbit(
            durations_ms, weighted_use, q_rest=q_rest, q_reserve=float(value),
            tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
            integrated_returns=integrated_returns,
        )
        for value in bounds
    ]
    means = np.asarray([row["q_mean"] for row in endpoint], dtype=float)
    slope = (means[1] - means[0]) / (bounds[1] - bounds[0])
    if not np.isfinite(slope) or slope <= 0.0:
        raise RuntimeError("periodic q_res map is non-monotone")
    if not means[0] <= q_hold <= means[1]:
        raise NoPhysicalPeriodicReserveRoot(
            "periodic q_res root is not bracketed in the registered physical domain"
        )
    q_reserve = float(bounds[0] + (q_hold - means[0]) / slope)
    result = periodic_q_orbit(
        durations_ms, weighted_use, q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
        integrated_returns=integrated_returns,
    )
    residual = float(result["q_mean"] - q_hold)
    if abs(residual) > tolerance:
        raise RuntimeError("periodic q_res solve missed its tolerance")
    return {
        **result,
        "q_reserve": q_reserve,
        "q_mean_residual": residual,
        "q_reserve_mean_slope": float(slope),
    }


def _simulate_event_intervals(
    durations_ms: np.ndarray,
    use: np.ndarray,
    *,
    tau_h_ms: float,
    theta_h: float,
    gate_width: float,
    substeps: int,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
) -> dict[str, Any]:
    eligibility = eligibility_trace(
        durations_ms, use, tau_h_ms=tau_h_ms, theta_h=theta_h,
        gate_width=gate_width, substeps=substeps, periodic=False,
    )
    trajectory = q_trace(
        eligibility["durations_ms"], eligibility["weighted_use"],
        q_initial=q_rest, q_rest=q_rest, q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery_ms, tau_depletion_ms=tau_depletion_ms,
    )
    return {
        **eligibility,
        "q": np.asarray(trajectory["q"], dtype=float),
        "q_final": float(np.asarray(trajectory["q"])[-1]),
        "q_min": float(np.min(np.asarray(trajectory["q"]))),
    }


def classify_schedule(
    result: dict[str, Any],
    onsets_ms: Sequence[float],
    *,
    entry_fold_q: float,
    final_target_q: float,
    pre_last_margin_q: float,
) -> dict[str, Any]:
    """Classify entry timing without forcing a no-entry trace into an event index."""

    onsets = np.asarray(onsets_ms, dtype=float)
    time = np.asarray(result["time_ms"], dtype=float)
    q = np.asarray(result["q"], dtype=float)
    if onsets.ndim != 1 or onsets.size == 0 or np.any(np.diff(onsets) <= 0.0):
        raise ValueError("schedule onsets must be strictly increasing")
    if time.shape != q.shape or time[-1] < onsets[-1]:
        raise ValueError("schedule trace does not cover all onsets")
    crossing = np.flatnonzero(q < float(entry_fold_q))
    crossing_time = None if crossing.size == 0 else float(time[int(crossing[0])])
    entry_index = (
        None if crossing_time is None
        else int(np.searchsorted(onsets, crossing_time, side="right"))
    )
    before_last = time < float(onsets[-1])
    minimum_before_last = float(np.min(q[before_last])) if np.any(before_last) else float(q[0])
    pre_last_pass = bool(minimum_before_last >= entry_fold_q + pre_last_margin_q)
    target_reached = bool(float(np.min(q[time >= onsets[-1]])) <= final_target_q)
    if entry_index is None:
        label = "no_entry"
    elif entry_index < int(onsets.size):
        label = f"premature_entry_event_{entry_index}"
    elif target_reached and pre_last_pass:
        label = f"target_entry_event_{entry_index}"
    else:
        label = f"entry_event_{entry_index}"
    return {
        "outcome": label,
        "entered": entry_index is not None,
        "entry_event_index": entry_index,
        "first_crossing_ms": crossing_time,
        "minimum_q_before_last": minimum_before_last,
        "minimum_q_full": float(np.min(q)),
        "final_q": float(q[-1]),
        "pre_last_margin_pass": pre_last_pass,
        "last_event_target_reached": target_reached,
        "locked_last_event_pass": bool(
            entry_index == int(onsets.size) and pre_last_pass and target_reached
        ),
    }


def root_brackets(residuals: Sequence[float]) -> list[tuple[int, int]]:
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("root residuals must be a finite vector")
    exact = np.flatnonzero(values == 0.0)
    if exact.size:
        return [(int(index), int(index)) for index in exact]
    brackets = []
    for index, (left, right) in enumerate(zip(values[:-1], values[1:])):
        if left * right < 0.0:
            brackets.append((index, index + 1))
    return brackets


def largest_edge_component(
    accepted: Iterable[tuple[int, int]],
) -> int:
    """Size of the largest four-neighbour component in the registered grid."""

    nodes = {tuple(map(int, node)) for node in accepted}
    largest = 0
    while nodes:
        frontier = [nodes.pop()]
        size = 0
        while frontier:
            i, j = frontier.pop()
            size += 1
            for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbor in nodes:
                    nodes.remove(neighbor)
                    frontier.append(neighbor)
        largest = max(largest, size)
    return largest


def _status(supported: bool, resolved: bool) -> str:
    if supported:
        return "THRESHOLDED_INHIBITORY_ELIGIBILITY_SCALAR_SUPPORTED_SHORT_COUPLED_ARM_ONLY"
    if resolved:
        return "THRESHOLDED_INHIBITORY_ELIGIBILITY_SCALAR_CLEAN_NO_GO_REGISTERED_ROBUSTNESS_GATES"
    return "THRESHOLDED_INHIBITORY_ELIGIBILITY_SCALAR_NUMERICALLY_UNRESOLVED"


def root_resolution_gates(
    cell_rows: Sequence[dict[str, Any]],
    *,
    observed_scan_rows: int,
    expected_scan_rows: int,
) -> dict[str, bool]:
    """Separate resolved failed cells from genuine numerical uncertainty."""

    if not cell_rows or expected_scan_rows <= 0 or observed_scan_rows < 0:
        raise ValueError("root-resolution gate inputs are invalid")
    return {
        "complete_registered_root_scan": observed_scan_rows == expected_scan_rows,
        "all_found_roots_are_unique_monotone": all(
            not bool(row["root_found"])
            or (bool(row["root_scan_monotone"]) and int(row["root_bracket_count"]) == 1)
            for row in cell_rows
        ),
        "no_numeric_calibration_errors": not any(
            bool(row["numeric_error"]) for row in cell_rows
        ),
    }


def _validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    tau_axis = [float(value) for value in model["tau_h_ms_axis"]]
    theta_axis = [float(value) for value in model["theta_h_axis"]]
    if tau_axis != [5000.0, 10000.0, 15000.0] or theta_axis != [0.015, 0.020, 0.025]:
        raise RuntimeError("registered 3x3 eligibility grid drifted")
    if float(model["q_hold"]) != 0.8425 or float(model["gate_width"]) != 0.002:
        raise RuntimeError("registered q_hold or gate width drifted")
    if (
        float(model["primary_tau_h_ms"]) not in tau_axis
        or float(model["primary_theta_h"]) not in theta_axis
    ):
        raise RuntimeError("primary eligibility cell must belong to the registered grid")
    if [int(value) for value in cfg["integration"]["scalar_substeps"]] != [1, 2]:
        raise RuntimeError("base/half scalar contract drifted")
    scope = cfg["scope"]
    if not scope.get("scalar_sensor_replay_only") or not scope.get("pilot_informed_mechanism_discovery"):
        raise RuntimeError("eligibility scope must remain scalar mechanism discovery")
    forbidden = (
        "coupled_fast_q", "autonomous_lifecycle", "ee_weight_change",
        "ee_kernel_change", "conductance_membrane",
    )
    if any(bool(scope.get(name)) for name in forbidden):
        raise RuntimeError("eligibility config attempts to unlock a forbidden scope")


def _validate_inputs(
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], dict[str, np.ndarray]]:
    path_keys = (
        "mapping_summary_path", "cycle_sensor_path",
        "cycle_measurements_path", "event_sensor_path",
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
            raise RuntimeError(f"locked eligibility input drift for {name}: {hashes[name]}")

    mapping = json.loads((ROOT / str(cfg["mapping_summary_path"])).read_text(encoding="utf-8"))
    if mapping.get("status") != "RESERVE_MAPPING_CLEAN_NO_GO_LOCKED_EVENT_ORDERING_CONFLICT":
        raise RuntimeError("mapping provenance no longer carries the locked event-ordering no-go")
    if mapping.get("gates", {}).get("all_event_replays_cross_only_on_last_event") is not False:
        raise RuntimeError("mapping provenance no longer exposes the locked failed gate")

    with np.load(ROOT / str(cfg["cycle_sensor_path"]), allow_pickle=False) as payload:
        cycle = {name: np.asarray(payload[name]) for name in payload.files}
    required_cycle = {"time_ms", "use", "q_hold", "phase", "dt_ms"}
    if set(cycle) != required_cycle:
        raise ValueError("cycle sensor NPZ schema drifted")
    n_records = int(cycle["q_hold"].size)
    if (
        cycle["time_ms"].ndim != 1
        or cycle["use"].shape != (n_records, cycle["time_ms"].size)
        or cycle["phase"].shape != (n_records,)
        or cycle["dt_ms"].shape != (n_records,)
        or not all(np.all(np.isfinite(cycle[name])) for name in required_cycle)
        or np.any(cycle["use"] < 0.0)
        or np.any(np.diff(cycle["time_ms"]) <= 0.0)
    ):
        raise ValueError("cycle sensor arrays are not finite and aligned")

    cycle_rows = _load_cycle_rows(ROOT / str(cfg["cycle_measurements_path"]))
    if len(cycle_rows) != n_records or any(row["outcome"] != "bounded_CCO" for row in cycle_rows):
        raise RuntimeError("cycle measurement provenance is incomplete or not bounded CCO")
    for index, row in enumerate(cycle_rows):
        observed = (cycle["q_hold"][index], cycle["phase"][index], cycle["dt_ms"][index])
        expected = (row["q_hold"], row["phase"], row["dt_ms"])
        if not np.allclose(observed, expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("cycle NPZ and CSV row ordering drifted")
    target_q = float(cfg["model"]["q_hold"])
    target_rows = [row for row in cycle_rows if row["q_hold"] == target_q]
    if len(target_rows) != 8:
        raise RuntimeError("q_hold=.8425 must have the full four-phase dual-dt sensor product")

    with np.load(ROOT / str(cfg["event_sensor_path"]), allow_pickle=False) as payload:
        event = {name: np.asarray(payload[name]) for name in payload.files}
    required_event = {
        "time_ms", "use", "return_counts", "support_violation_count",
        "state_bound_violation_count", "finite",
    }
    if set(event) != required_event:
        raise ValueError("event sensor NPZ schema drifted")
    if (
        event["time_ms"].ndim != 1
        or event["use"].shape != event["time_ms"].shape
        or np.any(np.diff(event["time_ms"]) <= 0.0)
        or not np.all(np.isfinite(event["time_ms"]))
        or not np.all(np.isfinite(event["use"]))
        or np.any(event["use"] < 0.0)
        or event["return_counts"].tolist() != [6, 6, 0]
        or np.any(event["support_violation_count"])
        or np.any(event["state_bound_violation_count"])
        or not bool(np.all(event["finite"]))
    ):
        raise RuntimeError("fixed event sensor is not the locked clean 6/6/0 replay")
    if not np.isclose(event["time_ms"][-1], float(cfg["locked_schedule"]["stop_ms"])):
        raise RuntimeError("fixed event sensor stop time drifted")
    return hashes, mapping, cycle, cycle_rows, event


def _cycle_intervals(
    cycle: dict[str, np.ndarray],
    row: dict[str, Any],
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    durations, use, _ = extract_piecewise_constant_window(
        np.asarray(cycle["time_ms"], dtype=float),
        np.asarray(cycle["use"][index], dtype=float),
        float(row["cycle_window_start_ms"]),
        float(row["cycle_window_stop_ms"]),
    )
    return durations, use


def _raw_event_intervals(event: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(event["time_ms"], dtype=float)
    use = np.asarray(event["use"], dtype=float)
    return np.diff(time), use[:-1]


def _event_template(
    event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, float]:
    locked = cfg["locked_schedule"]
    onset = float(locked["template_source_onset_ms"])
    stop = onset + float(locked["template_stop_after_onset_ms"])
    time = np.asarray(event["time_ms"], dtype=float)
    use = np.asarray(event["use"], dtype=float)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("event template source must have a uniform sample grid")
    left = int(round((onset - time[0]) / dt))
    right = int(round((stop - time[0]) / dt))
    if left < 0 or right >= time.size or not np.isclose(time[left], onset) or not np.isclose(time[right], stop):
        raise RuntimeError("event template endpoints do not align with the raw sensor")
    return use[left:right + 1].copy(), dt


def synthesize_schedule(
    onsets_ms: Sequence[float],
    template: Sequence[float],
    *,
    dt_ms: float,
    stop_after_last_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the locked isolated-event sensor template at fixed onset arrays."""

    onsets = np.asarray(onsets_ms, dtype=float)
    wave = np.asarray(template, dtype=float)
    if onsets.ndim != 1 or onsets.size == 0 or np.any(np.diff(onsets) <= 0.0):
        raise ValueError("schedule onsets must be strictly increasing")
    if wave.ndim != 1 or wave.size < 2 or np.any(wave < 0.0) or not np.all(np.isfinite(wave)):
        raise ValueError("event template must be a finite non-negative vector")
    if dt_ms <= 0.0 or stop_after_last_ms <= 0.0:
        raise ValueError("schedule sample and stop intervals must be positive")
    if not np.allclose(onsets / dt_ms, np.round(onsets / dt_ms), rtol=0.0, atol=1.0e-10):
        raise ValueError("schedule onsets must align with the sensor sample grid")
    stop = float(onsets[-1] + stop_after_last_ms)
    n_steps = int(round(stop / dt_ms))
    if not np.isclose(n_steps * dt_ms, stop):
        raise ValueError("schedule stop must align with the sensor sample grid")
    time = np.arange(n_steps + 1, dtype=float) * dt_ms
    use = np.zeros(time.size, dtype=float)
    for onset in onsets:
        start = int(round(onset / dt_ms))
        stop_index = min(use.size, start + wave.size)
        use[start:stop_index] = np.maximum(use[start:stop_index], wave[:stop_index - start])
    return time, use


def _find_primary_cycle_index(
    cfg: dict[str, Any],
    cycle_rows: list[dict[str, Any]],
) -> int:
    target = (
        float(cfg["model"]["q_hold"]),
        float(cfg["calibration"]["primary_cycle_phase"]),
        float(cfg["calibration"]["primary_cycle_dt_ms"]),
    )
    matches = [
        index for index, row in enumerate(cycle_rows)
        if (row["q_hold"], row["phase"], row["dt_ms"]) == target
    ]
    if len(matches) != 1:
        raise RuntimeError("primary cycle sensor is missing or duplicated")
    return matches[0]


def _periodic_for_cell(
    durations: np.ndarray,
    use: np.ndarray,
    *,
    tau_h_ms: float,
    theta_h: float,
    tau_depletion_ms: float,
    substeps: int,
    integrated_returns: int,
    cfg: dict[str, Any],
    fixed_q_reserve: float | None = None,
) -> dict[str, Any]:
    model = cfg["model"]
    calibration = cfg["calibration"]
    eligibility = eligibility_trace(
        durations, use, tau_h_ms=tau_h_ms, theta_h=theta_h,
        gate_width=float(model["gate_width"]), substeps=substeps, periodic=True,
    )
    if fixed_q_reserve is None:
        solved = solve_periodic_q_reserve(
            eligibility["durations_ms"], eligibility["weighted_use"],
            q_hold=float(model["q_hold"]), q_rest=float(model["q_rest"]),
            tau_recovery_ms=float(model["tau_recovery_ms"]),
            tau_depletion_ms=tau_depletion_ms,
            integrated_returns=integrated_returns,
            q_reserve_search=calibration["q_reserve_search"],
            tolerance=float(calibration["root_tolerance"]),
        )
    else:
        solved = periodic_q_orbit(
            eligibility["durations_ms"], eligibility["weighted_use"],
            q_rest=float(model["q_rest"]), q_reserve=float(fixed_q_reserve),
            tau_recovery_ms=float(model["tau_recovery_ms"]),
            tau_depletion_ms=tau_depletion_ms,
            integrated_returns=integrated_returns,
        )
        solved["q_reserve"] = float(fixed_q_reserve)
        solved["q_mean_residual"] = float(solved["q_mean"] - float(model["q_hold"]))
    solved["h"] = np.asarray(eligibility["h"])
    solved["gate"] = np.asarray(eligibility["gate"])
    solved["use"] = np.asarray(eligibility["use"])
    solved["eligibility_time_ms"] = np.asarray(eligibility["time_ms"])
    solved["h_window_rho"] = float(eligibility["map_alpha"])
    solved["h_per_cycle_multiplier"] = float(
        float(eligibility["map_alpha"]) ** (1.0 / float(integrated_returns))
    )
    return solved


def _event_for_cell(
    durations: np.ndarray,
    use: np.ndarray,
    onsets: Sequence[float],
    *,
    tau_h_ms: float,
    theta_h: float,
    tau_depletion_ms: float,
    q_reserve: float,
    substeps: int,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = cfg["model"]
    result = _simulate_event_intervals(
        durations, use, tau_h_ms=tau_h_ms, theta_h=theta_h,
        gate_width=float(model["gate_width"]), substeps=substeps,
        q_rest=float(model["q_rest"]), q_reserve=q_reserve,
        tau_recovery_ms=float(model["tau_recovery_ms"]),
        tau_depletion_ms=tau_depletion_ms,
    )
    classification = classify_schedule(
        result, onsets,
        entry_fold_q=float(model["entry_fold_q"]),
        final_target_q=float(model["event_final_target_q"]),
        pre_last_margin_q=float(cfg["gates"]["pre_last_entry_margin_q"]),
    )
    return result, classification


def _calibrate_cell(
    tau_h_ms: float,
    theta_h: float,
    primary_cycle: tuple[np.ndarray, np.ndarray, int],
    locked_event: tuple[np.ndarray, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Nested primary solve: q_res from CCO, tau_d from locked endpoint."""

    cycle_durations, cycle_use, integrated_returns = primary_cycle
    event_durations, event_use = locked_event
    calibration = cfg["calibration"]
    model = cfg["model"]
    onsets = [float(value) for value in cfg["locked_schedule"]["onsets_ms"]]
    tau_axis = np.geomspace(
        float(calibration["tau_depletion_search_ms"][0]),
        float(calibration["tau_depletion_search_ms"][1]),
        int(calibration["scan_points"]),
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
            "tau_h_ms": float(tau_h_ms),
            "theta_h": float(theta_h),
            "tau_depletion_ms": None,
            "q_reserve": None,
            "endpoint_residual_q": None,
            "root_scan_monotone": monotone,
            "root_bracket_count": int(bracket_count),
            "scan_residual_min_q": residual_min,
            "scan_residual_max_q": residual_max,
        }, scan_rows)

    def evaluate(tau_depletion: float) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]:
        periodic = _periodic_for_cell(
            cycle_durations, cycle_use, tau_h_ms=tau_h_ms, theta_h=theta_h,
            tau_depletion_ms=float(tau_depletion), substeps=1,
            integrated_returns=integrated_returns, cfg=cfg,
        )
        event_result, event_class = _event_for_cell(
            event_durations, event_use, onsets,
            tau_h_ms=tau_h_ms, theta_h=theta_h,
            tau_depletion_ms=float(tau_depletion),
            q_reserve=float(periodic["q_reserve"]), substeps=1, cfg=cfg,
        )
        residual = float(event_result["q_final"] - float(model["event_final_target_q"]))
        return residual, periodic, event_result, event_class

    evaluations: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]] | None] = []
    for tau in tau_axis:
        try:
            evaluated = evaluate(float(tau))
            evaluations.append(evaluated)
            scan_rows.append({
                "tau_h_ms": tau_h_ms, "theta_h": theta_h,
                "tau_depletion_ms": float(tau),
                "endpoint_residual_q": float(evaluated[0]),
                "q_reserve": float(evaluated[1]["q_reserve"]),
                "physical": True,
                "evaluation_status": "physical",
                "evaluation_error": None,
            })
        except NoPhysicalPeriodicReserveRoot as exc:
            evaluations.append(None)
            scan_rows.append({
                "tau_h_ms": tau_h_ms, "theta_h": theta_h,
                "tau_depletion_ms": float(tau),
                "endpoint_residual_q": None,
                "q_reserve": None,
                "physical": False,
                "evaluation_status": "no_physical_q_reserve",
                "evaluation_error": str(exc),
            })
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            evaluations.append(None)
            scan_rows.append({
                "tau_h_ms": tau_h_ms, "theta_h": theta_h,
                "tau_depletion_ms": float(tau),
                "endpoint_residual_q": None,
                "q_reserve": None,
                "physical": False,
                "evaluation_status": NUMERIC_ERROR,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
            })
            return failed(
                NUMERIC_ERROR,
                f"registered scan evaluation failed at tau_D={float(tau):.12g} ms: {type(exc).__name__}: {exc}",
                monotone=None,
                bracket_count=0,
                residual_min=None,
                residual_max=None,
            )
    finite_indices = [index for index, item in enumerate(evaluations) if item is not None]
    if len(finite_indices) < 2:
        return failed(
            NO_ROOT_IN_DOMAIN,
            "fewer than two physical evaluations in the registered tau_D domain",
            monotone=None,
            bracket_count=0,
            residual_min=None,
            residual_max=None,
        )
    finite_residuals = np.asarray([evaluations[index][0] for index in finite_indices], dtype=float)
    monotone = bool(np.all(np.diff(finite_residuals) > 0.0))
    residual_min = float(np.min(finite_residuals))
    residual_max = float(np.max(finite_residuals))
    exact_indices = [
        index for index in finite_indices
        if float(evaluations[index][0]) == 0.0
    ]
    if exact_indices:
        bracket_pairs = [(index, index) for index in exact_indices]
    else:
        bracket_pairs = []
        for left, right in zip(finite_indices[:-1], finite_indices[1:]):
            if right != left + 1:
                continue
            a = float(evaluations[left][0])
            b = float(evaluations[right][0])
            if a * b < 0.0:
                bracket_pairs.append((left, right))
    if not monotone:
        return failed(
            NONMONOTONE_SCAN,
            f"registered endpoint scan is non-monotone with {len(bracket_pairs)} root brackets",
            monotone=False,
            bracket_count=len(bracket_pairs),
            residual_min=residual_min,
            residual_max=residual_max,
        )
    if not bracket_pairs:
        return failed(
            NO_ROOT_IN_DOMAIN,
            "monotone endpoint residual does not cross zero in the registered physical tau_D domain",
            monotone=True,
            bracket_count=0,
            residual_min=residual_min,
            residual_max=residual_max,
        )
    if len(bracket_pairs) > 1:
        return failed(
            MULTIPLE_ROOTS,
            f"registered endpoint scan contains {len(bracket_pairs)} root brackets",
            monotone=True,
            bracket_count=len(bracket_pairs),
            residual_min=residual_min,
            residual_max=residual_max,
        )
    left_index, right_index = bracket_pairs[0]
    tolerance = float(calibration["root_tolerance"])
    if left_index == right_index:
        tau = float(tau_axis[left_index])
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
            tau = 0.5 * (left + right)
            if final is None:
                final = evaluate(tau)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            return failed(
                NUMERIC_ERROR,
                f"registered root refinement failed: {type(exc).__name__}: {exc}",
                monotone=monotone,
                bracket_count=1,
                residual_min=residual_min,
                residual_max=residual_max,
            )
    if final is None:
        return failed(
            NUMERIC_ERROR,
            "registered root refinement returned no evaluation",
            monotone=monotone,
            bracket_count=1,
            residual_min=residual_min,
            residual_max=residual_max,
        )
    if abs(float(final[0])) > tolerance:
        return failed(
            NUMERIC_ERROR,
            "tau_depletion root missed its locked endpoint tolerance",
            monotone=monotone,
            bracket_count=1,
            residual_min=residual_min,
            residual_max=residual_max,
        )
    periodic, event_result, event_class = final[1], final[2], final[3]
    return ({
        "calibration_status": ROOT_FOUND,
        "root_found": True,
        "numeric_error": False,
        "calibration_diagnostic": None,
        "tau_h_ms": float(tau_h_ms),
        "theta_h": float(theta_h),
        "tau_depletion_ms": float(tau),
        "q_reserve": float(periodic["q_reserve"]),
        "endpoint_residual_q": float(final[0]),
        "root_scan_monotone": monotone,
        "root_bracket_count": len(bracket_pairs),
        "scan_residual_min_q": residual_min,
        "scan_residual_max_q": residual_max,
        "primary_periodic": periodic,
        "primary_locked_result": event_result,
        "primary_locked_classification": event_class,
    }, scan_rows)


def _cycle_safety(
    result: dict[str, Any],
    cfg: dict[str, Any],
) -> bool:
    gates = cfg["gates"]
    return bool(
        float(result["q_min"]) >= float(gates["minimum_periodic_q"])
        and float(result["q_max"]) <= float(gates["maximum_periodic_q"])
        and abs(float(result["q_mean_residual"])) <= float(gates["maximum_abs_periodic_mean_error_q"])
        and float(result["per_cycle_multiplier"]) < float(gates["maximum_q_per_cycle_multiplier"])
        and float(result["closure_error"]) <= 1.0e-10
    )


def _probe_contract(
    rows: list[dict[str, Any]],
    heldout_minimum_entry_event_index: int,
) -> bool:
    labels_match = all(
        len({row["outcome"] for row in rows if row["schedule"] == name}) == 1
        for name in {row["schedule"] for row in rows}
    )
    base = {row["schedule"]: row for row in rows if row["substeps"] == 1}
    required = {"isolated", "dense_1200ms", "sparse_3400ms"}
    if not required.issubset(base):
        return False
    qualitative = bool(
        not base["isolated"]["entered"]
        and base["dense_1200ms"]["entered"]
        and not base["sparse_3400ms"]["entered"]
    )
    heldout = [row for name, row in base.items() if name.startswith("heldout_seed_")]
    no_early = all(
        row["entry_event_index"] is None
        or int(row["entry_event_index"]) >= int(heldout_minimum_entry_event_index)
        for row in heldout
    )
    mixed = bool(any(row["entered"] for row in heldout) and any(not row["entered"] for row in heldout))
    return bool(labels_match and qualitative and no_early and mixed)


def _cell_failure_reasons(
    calibration_status: str,
    *,
    physical: bool,
    all_cycles_safe: bool,
    locked_pass: bool,
    base_half_locked_match: bool,
    sensitivity_pass: bool,
    probe_pass: bool,
    recovery_pass: bool,
) -> list[str]:
    if calibration_status != ROOT_FOUND:
        return [{
            NO_ROOT_IN_DOMAIN: "no-root",
            NONMONOTONE_SCAN: "nonmonotone",
            MULTIPLE_ROOTS: "multiple-roots",
            NUMERIC_ERROR: "numeric-error",
        }.get(calibration_status, "calibration-error")]
    reasons = []
    if not physical:
        reasons.append("slow-parameter")
    if not all_cycles_safe:
        reasons.append("cycle-safety")
    if not locked_pass:
        reasons.append("entry-order")
    if not base_half_locked_match:
        reasons.append("base-half")
    if not sensitivity_pass:
        reasons.append("theta-sensitivity")
    if not probe_pass:
        reasons.append("schedule")
    if not recovery_pass:
        reasons.append("recovery")
    return reasons


def _plot(
    figures: Path,
    cell_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    representative: dict[str, Any],
    gates: dict[str, bool],
    cfg: dict[str, Any],
) -> Path:
    plt.rcParams.update({"font.size": 8.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), constrained_layout=True)
    model = cfg["model"]
    tau_axis = [float(value) for value in model["tau_h_ms_axis"]]
    theta_axis = [float(value) for value in model["theta_h_axis"]]

    locked = representative["locked_base"]
    ax = axes[0, 0]
    time_s = np.asarray(locked["time_ms"]) * 1.0e-3
    interval_time_s = time_s[:-1]
    ax.plot(interval_time_s, locked["use"], color="#B2182B", lw=0.8, label="U")
    ax.plot(time_s, locked["h"], color="#2166AC", lw=1.0, label="H")
    ax.plot(interval_time_s, locked["gate"], color="#1B7837", lw=0.9, label="g(H)")
    ax.axhline(float(model["primary_theta_h"]), color="0.45", ls="--", lw=0.7)
    ax.set(xlabel="time (s)", ylabel="sensor / eligibility", title="A  Locked event-use eligibility")
    ax.legend(frameon=False, fontsize=7, ncol=3)

    ax = axes[0, 1]
    for key, color in (("locked_base", "#2166AC"), ("locked_half", "#B2182B")):
        trace = representative[key]
        ax.plot(np.asarray(trace["time_ms"]) * 1.0e-3, trace["q"], color=color, lw=1.0, label=key.split("_")[-1])
    for onset in cfg["locked_schedule"]["onsets_ms"]:
        ax.axvline(float(onset) * 1.0e-3, color="0.83", lw=0.45)
    ax.axhline(float(model["entry_fold_q"]), color="#762A83", ls="--", lw=0.8, label="entry fold")
    ax.axhline(float(model["event_final_target_q"]), color="0.4", ls=":", lw=0.8, label="target")
    ax.set(xlabel="time (s)", ylabel="q", title="B  Base/half locked ordering")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    matrix = np.zeros((len(tau_axis), len(theta_axis)), dtype=float)
    annotation = np.empty(matrix.shape, dtype=object)
    compact_reason = {
        "slow-parameter": "tauD<100",
        "cycle-safety": "cycle",
        "entry-order": "entry",
        "base-half": "base/half",
        "theta-sensitivity": "theta-sens",
        "schedule": "schedule",
        "recovery": "recovery",
        "no-root": "no-root",
        "nonmonotone": "nonmono",
        "multiple-roots": "multi-root",
        "numeric-error": "NUMERIC",
    }
    for i, tau in enumerate(tau_axis):
        for j, theta in enumerate(theta_axis):
            row = next(item for item in cell_rows if item["tau_h_ms"] == tau and item["theta_h"] == theta)
            matrix[i, j] = 1.0 if row["discovery_safe"] else 0.0
            reasons = [compact_reason.get(value, value) for value in str(row["failure_reasons"]).split(";") if value]
            if row["root_found"]:
                shown = ", ".join(reasons[:2])
                if len(reasons) > 2:
                    shown += f" +{len(reasons) - 2}"
                annotation[i, j] = f"{row['tau_depletion_ms']:.0f} ms\n{shown or 'SAFE'}"
            else:
                annotation[i, j] = reasons[0] if reasons else "calibration fail"
    ax.imshow(matrix, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#F4A582", "#92C5DE"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(theta_axis)), [f"{value:.3f}" for value in theta_axis])
    ax.set_yticks(range(len(tau_axis)), [f"{value/1000:.0f}" for value in tau_axis])
    for i in range(len(tau_axis)):
        for j in range(len(theta_axis)):
            ax.text(j, i, annotation[i, j], ha="center", va="center", fontsize=7)
    ax.legend(
        handles=[
            matplotlib.patches.Patch(color="#92C5DE", label="discovery-safe"),
            matplotlib.patches.Patch(color="#F4A582", label="failed cell"),
        ],
        frameon=False, fontsize=6.5, loc="upper right",
    )
    ax.set(xlabel="theta_H", ylabel="tau_H (s)", title="C  Cell verdict (color), mapped tau_D and failure reason")

    ax = axes[1, 0]
    selected_cycles = [
        row for row in cycle_rows
        if row["tau_h_ms"] == float(model["primary_tau_h_ms"])
        and row["theta_h"] == float(model["primary_theta_h"])
    ]
    for substeps, marker, color in ((1, "o", "#2166AC"), (2, "s", "#B2182B")):
        rows = [row for row in selected_cycles if row["substeps"] == substeps and row["theta_delta"] == 0.0]
        x = np.arange(len(rows))
        ax.scatter(x, [row["q_min"] for row in rows], marker="v", color=color, s=22, label=f"min, sub={substeps}")
        ax.scatter(x, [row["q_max"] for row in rows], marker="^", facecolors="none", edgecolors=color, s=22, label=f"max, sub={substeps}")
    ax.axhline(float(cfg["gates"]["minimum_periodic_q"]), color="0.5", ls="--", lw=0.7)
    ax.axhline(float(cfg["gates"]["maximum_periodic_q"]), color="0.5", ls="--", lw=0.7)
    ax.set(xlabel="phase/dt cycle record", ylabel="periodic q", title="D  All q_hold=.8425 CCO records")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[1, 1]
    palette = {
        "isolated": "#7B3294", "dense_1200ms": "#D73027",
        "sparse_3400ms": "#4575B4",
    }
    for name, trace in representative["probe_traces"].items():
        if name in palette:
            ax.plot(np.asarray(trace["time_ms"]) * 1.0e-3, trace["q"], color=palette[name], lw=1.0, label=name)
    ax.axhline(float(model["entry_fold_q"]), color="0.4", ls="--", lw=0.7)
    ax.set(xlabel="time (s)", ylabel="q", title="E  Equal-dose schedule probes")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 2]
    ax.axis("off")
    root_count = sum(bool(row["root_found"]) for row in cell_rows)
    theta_count = sum(bool(row["theta_sensitivity_pass"]) for row in cell_rows if row["root_found"])
    schedule_count = sum(bool(row["schedule_probe_contract_pass"]) for row in cell_rows if row["root_found"])
    recovery_count = sum(bool(row["u0_recovery_pass"]) for row in cell_rows if row["root_found"])
    lines = ["F  Scalar verdict", ""] + [
        f"{name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()
    ] + [
        "",
        f"root cells: {root_count}/{len(cell_rows)}",
        f"theta sensitivity: {theta_count}/{root_count}",
        f"schedule probes: {schedule_count}/{root_count}",
        f"U=0 recovery: {recovery_count}/{root_count}",
        "",
        "Passing status unlocks SHORT_COUPLED_ARM_ONLY.",
        "No autonomous lifecycle run.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=5.45)

    fig.suptitle("Thresholded inhibitory-use eligibility: scalar mechanism screen", fontsize=12.5, fontweight="bold")
    stem = figures / "mz_thresholded_inhibitory_eligibility"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    hashes, mapping, cycle, measurement_rows, event = _validate_inputs(cfg)
    output = ROOT / str(cfg["result_root"])
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    primary_index = _find_primary_cycle_index(cfg, measurement_rows)
    primary_row = measurement_rows[primary_index]
    primary_durations, primary_use = _cycle_intervals(cycle, primary_row, primary_index)
    primary_cycle = (primary_durations, primary_use, int(primary_row["integrated_returns"]))
    locked_durations, locked_use = _raw_event_intervals(event)
    locked_event = (locked_durations, locked_use)
    template, template_dt = _event_template(event, cfg)

    model = cfg["model"]
    substeps_axis = [int(value) for value in cfg["integration"]["scalar_substeps"]]
    q_hold = float(model["q_hold"])
    target_cycle_indices = [
        index for index, row in enumerate(measurement_rows)
        if row["q_hold"] == q_hold
    ]
    schedules: dict[str, tuple[np.ndarray, np.ndarray, list[float]]] = {}
    probe_config = cfg["schedule_probes"]
    stop_after_last_ms = float(probe_config["stop_after_last_onset_ms"])
    for name, raw_onsets in probe_config.items():
        if name == "stop_after_last_onset_ms":
            continue
        onsets = [float(value) for value in raw_onsets]
        time, sensor = synthesize_schedule(
            onsets, template, dt_ms=template_dt,
            stop_after_last_ms=stop_after_last_ms,
        )
        schedules[name] = (np.diff(time), sensor[:-1], onsets)

    cell_rows: list[dict[str, Any]] = []
    root_scan_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {"probe_traces": {}}

    for tau_index, tau_h in enumerate(map(float, model["tau_h_ms_axis"])):
        for theta_index, theta_h in enumerate(map(float, model["theta_h_axis"])):
            calibrated, scan = _calibrate_cell(
                tau_h, theta_h, primary_cycle, locked_event, cfg
            )
            root_scan_rows.extend(scan)
            calibration_status = str(calibrated["calibration_status"])
            root_found = bool(calibrated["root_found"])
            tau_depletion = (
                None if calibrated["tau_depletion_ms"] is None
                else float(calibrated["tau_depletion_ms"])
            )
            q_reserve = (
                None if calibrated["q_reserve"] is None
                else float(calibrated["q_reserve"])
            )
            physical = bool(
                root_found
                and tau_depletion is not None
                and q_reserve is not None
                and tau_depletion >= float(cfg["gates"]["minimum_tau_depletion_ms"])
                and 0.0 < q_reserve < q_hold
            )
            all_cycles_safe = False
            locked_pass = False
            base_half_locked_match = False
            sensitivity_pass = False
            probe_pass = False
            recovery_pass = False

            if root_found:
                assert tau_depletion is not None and q_reserve is not None
                sensitivity_flags: list[bool] = []
                for theta_delta in (0.0, -float(model["theta_sensitivity_delta"]), float(model["theta_sensitivity_delta"])):
                    theta_eval = theta_h + theta_delta
                    cycle_flags = []
                    for substeps in substeps_axis:
                        for cycle_index in target_cycle_indices:
                            row = measurement_rows[cycle_index]
                            durations, sensor = _cycle_intervals(cycle, row, cycle_index)
                            periodic = _periodic_for_cell(
                                durations, sensor, tau_h_ms=tau_h, theta_h=theta_eval,
                                tau_depletion_ms=tau_depletion, substeps=substeps,
                                integrated_returns=int(row["integrated_returns"]), cfg=cfg,
                                fixed_q_reserve=q_reserve,
                            )
                            safe = _cycle_safety(periodic, cfg)
                            cycle_flags.append(safe)
                            cycle_rows.append({
                                "tau_h_ms": tau_h, "theta_h": theta_h,
                                "theta_delta": theta_delta, "theta_evaluated": theta_eval,
                                "substeps": substeps, "source_phase": row["phase"],
                                "source_dt_ms": row["dt_ms"],
                                "q_min": periodic["q_min"], "q_max": periodic["q_max"],
                                "q_mean": periodic["q_mean"],
                                "q_mean_residual": periodic["q_mean_residual"],
                                "per_cycle_multiplier": periodic["per_cycle_multiplier"],
                                "h_per_cycle_multiplier": periodic["h_per_cycle_multiplier"],
                                "closure_error": periodic["closure_error"],
                                "cycle_safe": safe,
                            })

                    locked_labels = []
                    locked_flags = []
                    for substeps in substeps_axis:
                        locked_result, locked_class = _event_for_cell(
                            locked_durations, locked_use,
                            cfg["locked_schedule"]["onsets_ms"],
                            tau_h_ms=tau_h, theta_h=theta_eval,
                            tau_depletion_ms=tau_depletion, q_reserve=q_reserve,
                            substeps=substeps, cfg=cfg,
                        )
                        locked_labels.append(str(locked_class["outcome"]))
                        locked_flags.append(bool(locked_class["locked_last_event_pass"]))
                        sensitivity_rows.append({
                            "tau_h_ms": tau_h, "theta_h": theta_h,
                            "theta_delta": theta_delta, "theta_evaluated": theta_eval,
                            "substeps": substeps, **locked_class,
                            "all_cycle_records_safe": all(cycle_flags),
                        })
                        if tau_h == float(model["primary_tau_h_ms"]) and theta_h == float(model["primary_theta_h"]) and theta_delta == 0.0:
                            key = "locked_base" if substeps == 1 else "locked_half"
                            representative[key] = locked_result
                    theta_label_pass = bool(
                        len(set(locked_labels)) == 1
                        and all(locked_flags)
                        and all(cycle_flags)
                    )
                    if theta_delta == 0.0:
                        all_cycles_safe = all(cycle_flags)
                        locked_pass = all(locked_flags)
                        base_half_locked_match = len(set(locked_labels)) == 1
                    else:
                        sensitivity_flags.append(theta_label_pass)

                sensitivity_pass = bool(
                    len(sensitivity_flags) == 2 and all(sensitivity_flags)
                )

                local_probe_rows = []
                for name, (durations, sensor, onsets) in schedules.items():
                    for substeps in substeps_axis:
                        result, classified = _event_for_cell(
                            durations, sensor, onsets,
                            tau_h_ms=tau_h, theta_h=theta_h,
                            tau_depletion_ms=tau_depletion, q_reserve=q_reserve,
                            substeps=substeps, cfg=cfg,
                        )
                        row = {
                            "tau_h_ms": tau_h, "theta_h": theta_h,
                            "substeps": substeps, "schedule": name,
                            "event_count": len(onsets), **classified,
                        }
                        probe_rows.append(row)
                        local_probe_rows.append(row)
                        if tau_h == float(model["primary_tau_h_ms"]) and theta_h == float(model["primary_theta_h"]) and substeps == 1:
                            representative["probe_traces"][name] = result
                probe_pass = _probe_contract(
                    local_probe_rows,
                    int(cfg["gates"]["heldout_minimum_entry_event_index"]),
                )

                for substeps in substeps_axis:
                    periodic = _periodic_for_cell(
                        primary_durations, primary_use,
                        tau_h_ms=tau_h, theta_h=theta_h,
                        tau_depletion_ms=tau_depletion, substeps=substeps,
                        integrated_returns=int(primary_row["integrated_returns"]), cfg=cfg,
                        fixed_q_reserve=q_reserve,
                    )
                    duration = float(cfg["integration"]["recovery_duration_ms"])
                    q_initial = float(periodic["q_stroboscopic"])
                    h_initial = float(np.asarray(periodic["h"])[0])
                    q_final = float(model["q_rest"]) + (q_initial - float(model["q_rest"])) * np.exp(
                        -duration / float(model["tau_recovery_ms"])
                    )
                    h_final = h_initial * np.exp(-duration / tau_h)
                    passed = bool(
                        q_final >= float(cfg["gates"]["recovery_minimum_q"])
                        and h_final <= float(cfg["gates"]["recovery_maximum_h"])
                    )
                    recovery_rows.append({
                        "tau_h_ms": tau_h, "theta_h": theta_h,
                        "substeps": substeps, "q_initial": q_initial,
                        "h_initial": h_initial, "q_final": q_final,
                        "h_final": h_final, "recovery_pass": passed,
                    })
                recovery_pass = all(
                    row["recovery_pass"] for row in recovery_rows
                    if row["tau_h_ms"] == tau_h and row["theta_h"] == theta_h
                )

            discovery_safe = bool(
                physical and all_cycles_safe and locked_pass
                and base_half_locked_match and sensitivity_pass
                and probe_pass and recovery_pass
            )
            failure_reasons = _cell_failure_reasons(
                calibration_status,
                physical=physical,
                all_cycles_safe=all_cycles_safe,
                locked_pass=locked_pass,
                base_half_locked_match=base_half_locked_match,
                sensitivity_pass=sensitivity_pass,
                probe_pass=probe_pass,
                recovery_pass=recovery_pass,
            )
            cell_rows.append({
                "tau_index": tau_index, "theta_index": theta_index,
                "tau_h_ms": tau_h, "theta_h": theta_h,
                "calibration_status": calibration_status,
                "root_found": root_found,
                "numeric_error": bool(calibrated["numeric_error"]),
                "calibration_diagnostic": calibrated["calibration_diagnostic"],
                "tau_depletion_ms": tau_depletion,
                "q_reserve": q_reserve,
                "endpoint_residual_q": calibrated["endpoint_residual_q"],
                "root_scan_monotone": calibrated["root_scan_monotone"],
                "root_bracket_count": calibrated["root_bracket_count"],
                "scan_residual_min_q": calibrated["scan_residual_min_q"],
                "scan_residual_max_q": calibrated["scan_residual_max_q"],
                "physical_slow_parameter_gate": physical,
                "all_qhold_cycle_records_safe": all_cycles_safe,
                "locked_last_event_pass": locked_pass,
                "base_half_locked_labels_match": base_half_locked_match,
                "theta_sensitivity_pass": sensitivity_pass,
                "schedule_probe_contract_pass": probe_pass,
                "u0_recovery_pass": recovery_pass,
                "discovery_safe": discovery_safe,
                "failure_reasons": ";".join(failure_reasons),
            })

    accepted = [
        (int(row["tau_index"]), int(row["theta_index"]))
        for row in cell_rows if row["discovery_safe"]
    ]
    component_size = largest_edge_component(accepted)
    primary = next(
        row for row in cell_rows
        if row["tau_h_ms"] == float(model["primary_tau_h_ms"])
        and row["theta_h"] == float(model["primary_theta_h"])
    )
    expected_scan_rows = len(cell_rows) * int(cfg["calibration"]["scan_points"])
    root_gates = root_resolution_gates(
        cell_rows,
        observed_scan_rows=len(root_scan_rows),
        expected_scan_rows=expected_scan_rows,
    )
    gates = {
        "hash_locked_no_go_provenance_valid": True,
        "registered_3x3_grid_complete": len(cell_rows) == 9,
        **root_gates,
        "primary_center_theta_sensitivity_pass": bool(primary["theta_sensitivity_pass"]),
        "primary_center_schedule_probe_contract_pass": bool(primary["schedule_probe_contract_pass"]),
        "primary_center_u0_recovery_pass": bool(primary["u0_recovery_pass"]),
        "primary_center_is_discovery_safe": bool(primary["discovery_safe"]),
        "edge_adjacent_safe_component_meets_gate": component_size >= int(cfg["gates"]["minimum_adjacent_cells"]),
        "passing_scope_is_short_coupled_arm_only": True,
    }
    supported = all(gates.values())
    resolution_gate_names = (
        "hash_locked_no_go_provenance_valid",
        "registered_3x3_grid_complete",
        "complete_registered_root_scan",
        "all_found_roots_are_unique_monotone",
        "no_numeric_calibration_errors",
        "passing_scope_is_short_coupled_arm_only",
    )
    resolved = all(bool(gates[name]) for name in resolution_gate_names)
    status = _status(supported, resolved)
    decision = (
        "run_registered_center_and_theta_neighbors_short_coupled_arm_only"
        if supported else "do_not_run_coupled_or_autonomous_eligibility_arm"
    )

    artifacts = {
        "summary": str((output / "thresholded_eligibility_summary.json").relative_to(ROOT)),
        "grid_csv": str((output / "thresholded_eligibility_grid.csv").relative_to(ROOT)),
        "root_scan_csv": str((output / "thresholded_eligibility_root_scan.csv").relative_to(ROOT)),
        "cycle_csv": str((output / "thresholded_eligibility_cycle_safety.csv").relative_to(ROOT)),
        "sensitivity_csv": str((output / "thresholded_eligibility_sensitivity.csv").relative_to(ROOT)),
        "schedule_csv": str((output / "thresholded_eligibility_schedule_probes.csv").relative_to(ROOT)),
        "recovery_csv": str((output / "thresholded_eligibility_recovery.csv").relative_to(ROOT)),
        "figure": str((figures / "mz_thresholded_inhibitory_eligibility.png").relative_to(ROOT)),
    }
    summary = {
        "status": status,
        "scientific_layer": "pilot_informed_thresholded_eligibility_scalar_mechanism_screen",
        "decision": decision,
        "gates": gates,
        "safe_cell_count": len(accepted),
        "largest_edge_adjacent_safe_component": component_size,
        "root_found_cell_count": sum(bool(row["root_found"]) for row in cell_rows),
        "no_root_cell_count": sum(row["calibration_status"] == NO_ROOT_IN_DOMAIN for row in cell_rows),
        "numeric_error_cell_count": sum(bool(row["numeric_error"]) for row in cell_rows),
        "theta_sensitivity_pass_cell_count": sum(bool(row["theta_sensitivity_pass"]) for row in cell_rows),
        "schedule_probe_pass_cell_count": sum(bool(row["schedule_probe_contract_pass"]) for row in cell_rows),
        "recovery_pass_cell_count": sum(bool(row["u0_recovery_pass"]) for row in cell_rows),
        "root_scan_row_count": len(root_scan_rows),
        "expected_root_scan_row_count": expected_scan_rows,
        "primary_cell": primary,
        "grid_cells": cell_rows,
        "input_sha256": hashes,
        "mapping_provenance_status": mapping["status"],
        "claim_boundary": [
            "this node replays hash-locked scalar sensors and does not couple q back to the fast spatial model",
            "the 3x3 grid is pilot-informed mechanism discovery and does not identify biological parameters",
            "a passing result unlocks only the registered center and theta neighbors as a short coupled arm",
            "no autonomous lifecycle, M retuning, retrigger, field containment, SNN migration, E-E, or conductance run is unlocked",
        ],
        "config": cfg,
        "artifacts": artifacts,
    }

    # Evidence is written before plotting so a plotting failure cannot erase the scientific verdict.
    _save_csv(output / "thresholded_eligibility_grid.csv", cell_rows)
    _save_csv(output / "thresholded_eligibility_root_scan.csv", root_scan_rows)
    _save_csv(output / "thresholded_eligibility_cycle_safety.csv", cycle_rows)
    _save_csv(output / "thresholded_eligibility_sensitivity.csv", sensitivity_rows)
    _save_csv(output / "thresholded_eligibility_schedule_probes.csv", probe_rows)
    _save_csv(output / "thresholded_eligibility_recovery.csv", recovery_rows)
    _save_json(output / "thresholded_eligibility_summary.json", summary)

    figures.mkdir(parents=True, exist_ok=True)
    _plot(figures, cell_rows, cycle_rows, probe_rows, representative, gates, cfg)
    (figures / "README.md").write_text(
        "### mz_thresholded_inhibitory_eligibility.png\n\n"
        "这张图检验慢 eligibility trace 是否能在不改变 E→E 或膜电导的前提下，把区域 q 的耗竭限制到近期 inhibitory-use 足够密集的事件。A–B 展示注册中心点的 U、H、门函数与 locked 六事件 q 顺序；C 同时显示 3×3 cell verdict、mapped tau_D 与逐格失败原因；D 检查全部 q_hold=.8425 周期 sensor 的安全范围；E 对比 isolated、dense 与 sparse equal-dose probes；F 给出 theta sensitivity、schedule、recovery 与 root-resolution 合同。\n\n"
        f"当前结果为 `{status}`：{sum(bool(row['root_found']) for row in cell_rows)}/9 cells 有唯一单调 root，注册域内 no-root 是 resolved failed cell；有 root 的 cells 中 theta sensitivity 通过 {sum(bool(row['theta_sensitivity_pass']) for row in cell_rows)}/{sum(bool(row['root_found']) for row in cell_rows)}，因此没有 discovery-safe cell。\n\n"
        "即使全部门通过，本节点仍只是 pilot-informed scalar mechanism discovery，只允许中心点及两个 theta 邻点进入短 coupled arm；不允许写成 autonomous lifecycle、空间 containment 或 seizure mechanism proof。\n\n"
        "**关注点**：承重结果是 edge-adjacent safe cells、held-out schedule 混合结果、theta sensitivity 与 U=0 recovery 同时成立，而不是某个单独参数点成功。\n",
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
        "safe_cell_count": summary["safe_cell_count"],
        "largest_edge_adjacent_safe_component": summary["largest_edge_adjacent_safe_component"],
        "primary_cell": summary["primary_cell"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
