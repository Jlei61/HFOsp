#!/usr/bin/env python3
"""R3 M-gated inhibitory-capacity recovery scalar/path oracle.

The fast regional model is used only to regenerate the registered 24-cell
fixed-q M-ramp sensor.  q is then replayed feed-forward and never coupled back
to the fast system in this producer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from scripts.run_topic4_mz_inhibitory_reserve_corridor_r0b import (  # noqa: E402
    _frozen_view,
    _ramp_parameters,
)
from scripts.run_topic4_mz_inhibitory_reserve_recovery_corridor import (  # noqa: E402
    hybrid_handoff_predictor,
)
from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _checkpoint,
    _cycle_initial,
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
    _pattern_summary,
    _validate_inputs,
)
from src.topic4_mz_spatial_autonomous_latch import integrate_autonomous_latch_batch  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import integrate_frozen_patch_batch  # noqa: E402
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_m_gated_reserve_recovery.yaml"
SUPPORTED = "R3_M_GATED_RESERVE_RECOVERY_PATH_SUPPORTED_SHORT_P3_FORK_UNLOCKED"
CLEAN_NO_GO = "R3_M_GATED_RESERVE_RECOVERY_CLEAN_NO_GO_REGISTERED_GATES"
UNRESOLVED = "R3_M_GATED_RESERVE_RECOVERY_NUMERICALLY_UNRESOLVED_FAIL_CLOSED"


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


def _scalar(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _key(q: float, phase: float, dt_ms: float) -> tuple[float, float, float]:
    return round(float(q), 10), round(float(phase), 10), round(float(dt_ms), 10)


def _validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    sensor = cfg["sensor"]
    acceptance = cfg["acceptance"]
    if [float(x) for x in model["q_hold_axis"]] != [0.84, 0.8425, 0.845]:
        raise RuntimeError("registered q_hold axis drifted")
    if [float(x) for x in model["tau_slow_s_axis"]] != [70.0, 80.0, 90.0, 100.0, 120.0, 160.0]:
        raise RuntimeError("registered tau_slow axis drifted")
    if float(model["tau_fast_primary_s"]) != 20.0:
        raise RuntimeError("registered primary tau_fast drifted")
    if [float(x) for x in model["tau_fast_sensitivity_s"]] != [15.0, 25.0]:
        raise RuntimeError("registered tau_fast sensitivity drifted")
    if float(model["additive_max_mv"]) != 1.6:
        raise RuntimeError("Amax drifted; recovery must use dimensionless m")
    if [float(x) for x in sensor["relative_phase_fractions"]] != [0.0, 0.25, 0.5, 0.75]:
        raise RuntimeError("phase axis drifted")
    if [float(x) for x in sensor["source_dt_ms"]] != [0.125, 0.0625]:
        raise RuntimeError("base/half dt axis drifted")
    if float(sensor["replay_dt_ms"]) > 1.0:
        raise RuntimeError("R3 replay reporting step exceeds 1 ms")
    if [float(x) for x in acceptance["required_corridor_tau_s"]] != [80.0, 90.0, 100.0]:
        raise RuntimeError("registered corridor drifted")
    if int(acceptance["minimum_consecutive_tau_nodes"]) != 3:
        raise RuntimeError("minimum corridor width drifted")
    scope = cfg["scope"]
    if not scope.get("fixed_q_sensor_regenerated") or not scope.get("scalar_path_oracle_only"):
        raise RuntimeError("R3 must regenerate the fixed-q sensor and remain scalar")
    forbidden = (
        "coupled_fast_qm", "continuous_field", "full_snn", "ee_weight_change",
        "ee_kernel_change", "conductance_membrane", "relay_change",
    )
    if any(bool(scope.get(name)) for name in forbidden):
        raise RuntimeError("R3 config attempts to unlock a forbidden scope")
    resource_cfg = cfg["resource_contract"]
    if int(resource_cfg["processes"]) != 1 or int(resource_cfg["blas_threads"]) != 1:
        raise RuntimeError("R3 must use one process and one BLAS thread")
    if int(resource_cfg["max_trace_bytes_per_batch"]) > 256 * 1024 * 1024:
        raise RuntimeError("sensor trace memory contract exceeds 256 MiB")
    if float(resource_cfg["max_memory_gib"]) > 1.5:
        raise RuntimeError("R3 RSS contract exceeds 1.5 GiB")


def recovery_rate_per_ms(m: np.ndarray | float, tau_slow_s: float, tau_fast_s: float) -> np.ndarray:
    values = np.asarray(m, dtype=float)
    if tau_fast_s <= 0.0 or tau_slow_s <= tau_fast_s or np.any(~np.isfinite(values)):
        raise ValueError("finite m and 0 < tau_fast < tau_slow are required")
    if np.any((values < -1.0e-12) | (values > 1.0 + 1.0e-12)):
        raise ValueError("m must be dimensionless and lie in [0,1]")
    return (1.0 - values) / (1000.0 * tau_slow_s) + values / (1000.0 * tau_fast_s)


def q_nullcline(
    m: np.ndarray | float,
    mean_use: float,
    q_rest: float,
    q_reserve: float,
    tau_depletion_ms: float,
    tau_slow_s: float,
    tau_fast_s: float,
) -> np.ndarray:
    if mean_use < 0.0 or tau_depletion_ms <= 0.0:
        raise ValueError("mean use and depletion timescale are invalid")
    a = recovery_rate_per_ms(m, tau_slow_s, tau_fast_s)
    b = float(mean_use) / float(tau_depletion_ms)
    return (a * q_rest + b * q_reserve) / (a + b)


def replay_q_with_sensor(
    time_ms: Sequence[float],
    use: Sequence[float],
    m: Sequence[float],
    *,
    q_initial: float,
    q_rest: float,
    q_reserve: float,
    tau_depletion_ms: float,
    tau_slow_s: float,
    tau_fast_s: float,
    stop_time_ms: float,
    reporting_dt_ms: float,
) -> dict[str, np.ndarray]:
    source_t = np.asarray(time_ms, dtype=float)
    source_u = np.asarray(use, dtype=float)
    source_m = np.asarray(m, dtype=float)
    if (
        source_t.ndim != 1 or source_u.shape != source_t.shape or source_m.shape != source_t.shape
        or source_t.size < 2 or np.any(np.diff(source_t) <= 0.0)
        or not np.all(np.isfinite(source_t)) or not np.all(np.isfinite(source_u))
        or not np.all(np.isfinite(source_m)) or np.any(source_u < 0.0)
        or stop_time_ms <= 0.0 or stop_time_ms > source_t[-1] or reporting_dt_ms <= 0.0
    ):
        raise ValueError("invalid aligned sensor replay")
    n_steps = int(math.ceil(stop_time_ms / reporting_dt_ms))
    out_t = np.minimum(np.arange(n_steps + 1, dtype=float) * reporting_dt_ms, stop_time_ms)
    out_t[-1] = stop_time_ms
    out_t = np.unique(out_t)
    out_u = np.interp(out_t, source_t, source_u)
    out_m = np.interp(out_t, source_t, source_m)
    q = np.empty_like(out_t)
    q[0] = float(q_initial)
    for index, dt in enumerate(np.diff(out_t)):
        m_mid = 0.5 * (out_m[index] + out_m[index + 1])
        u_mid = 0.5 * (out_u[index] + out_u[index + 1])
        a = float(recovery_rate_per_ms(m_mid, tau_slow_s, tau_fast_s))
        b = float(u_mid) / float(tau_depletion_ms)
        rate = a + b
        fixed = (a * q_rest + b * q_reserve) / rate
        q[index + 1] = fixed + (q[index] - fixed) * math.exp(-rate * dt)
    return {"time_ms": out_t, "use": out_u, "m": out_m, "q": q}


def protected_handoff(
    q_exit: float,
    m_exit: float,
    *,
    tau_slow_s: float,
    tau_fast_s: float,
    q_rest: float,
    fold_q: np.ndarray,
    fold_a: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    handoff = cfg["handoff"]
    sensor = cfg["sensor"]
    amax = float(cfg["model"]["additive_max_mv"])
    q_reset = float(handoff["q_reset_safe"])
    a = float(recovery_rate_per_ms(float(m_exit), tau_slow_s, tau_fast_s))
    if q_exit >= q_reset:
        q_time = 0.0
    elif not q_exit < q_reset < q_rest:
        q_time = math.inf
    else:
        q_time = -math.log((q_rest - q_reset) / (q_rest - q_exit)) / a
    p_time = -float(handoff["persistence_tau_ms"]) * math.log(
        float(handoff["persistence_off"]) / float(handoff["persistence_start"])
    )
    reset_time = max(q_time, p_time)
    horizon_pass = bool(np.isfinite(reset_time) and reset_time <= float(handoff["reset_horizon_ms"]))
    q_at_reset = q_rest - (q_rest - q_exit) * math.exp(-a * reset_time) if np.isfinite(reset_time) else float("nan")
    domain_stop = min(q_at_reset, float(fold_q[-1])) if np.isfinite(q_at_reset) else float("nan")
    if np.isfinite(domain_stop) and q_exit <= fold_q[-1]:
        domain_q = np.linspace(q_exit, max(q_exit, domain_stop), 128)
        margins = amax * m_exit - np.interp(domain_q, fold_q, fold_a)
        protected_margin_min = float(np.min(margins))
    else:
        protected_margin_min = float("nan")
    protected_pass = bool(
        np.isfinite(protected_margin_min)
        and protected_margin_min >= float(sensor["dynamic_additive_margin_mv"])
    )

    release_dt = float(handoff["reporting_dt_ms"])
    release_threshold_m = float(handoff["additive_release_threshold_mv"]) / amax
    if not 0.0 < release_threshold_m < m_exit:
        raise RuntimeError("registered additive release threshold is not below m_exit")
    release_time = float(handoff["tau_m_down_ms"]) * math.log(m_exit / release_threshold_m)
    n_steps = int(math.ceil(release_time / release_dt))
    time_release = np.minimum(np.arange(n_steps + 1, dtype=float) * release_dt, release_time)
    time_release[-1] = release_time
    time_release = np.unique(time_release)
    m_release = m_exit * np.exp(-time_release / float(handoff["tau_m_down_ms"]))
    q_release = np.empty_like(time_release)
    q_release[0] = q_at_reset
    for index, dt in enumerate(np.diff(time_release)):
        m_mid = 0.5 * (m_release[index] + m_release[index + 1])
        rate = float(recovery_rate_per_ms(m_mid, tau_slow_s, tau_fast_s))
        q_release[index + 1] = q_rest - (q_rest - q_release[index]) * math.exp(-rate * dt)
    monotone = bool(np.all(np.diff(q_release) >= -1.0e-12))
    final_protected = bool(q_release[-1] >= float(cfg["model"]["entry_fold_q"]))
    release_pass = bool(
        horizon_pass and monotone and final_protected
        and np.isclose(amax * m_release[-1], float(handoff["additive_release_threshold_mv"]), atol=1.0e-10)
    )
    return {
        "effective_tau_recovery_ms": 1.0 / a,
        "time_to_qsafe_ms": q_time,
        "persistence_off_bound_ms": p_time,
        "reset_time_ms": reset_time,
        "reset_horizon_pass": horizon_pass,
        "q_at_reset": q_at_reset,
        "protected_margin_min_mv": protected_margin_min,
        "protected_margin_pass": protected_pass,
        "release_after_reset_ms": release_time,
        "q_at_additive_release": float(q_release[-1]),
        "q_never_decreases_after_reset": monotone,
        "post_reset_release_pass": release_pass,
        "handoff_pass": bool(horizon_pass and protected_pass and release_pass),
        "release_time_ms": time_release,
        "release_m": m_release,
        "release_q": q_release,
    }


def _first_sustained_low(time_ms: np.ndarray, regional_fast: np.ndarray, threshold: float, duration_ms: float) -> int | None:
    low = np.all(np.asarray(regional_fast, dtype=float) <= threshold, axis=1)
    for index in np.flatnonzero(low):
        stop = int(np.searchsorted(time_ms, time_ms[index] + duration_ms, side="left"))
        if stop < low.size and np.all(low[index:stop + 1]):
            return int(index)
    return None


def _load_inputs(cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], np.ndarray, np.ndarray]:
    keys = (
        "r2_summary_path", "r2_event_path", "r2_periodic_path", "r2_schedule_path",
        "r2_handoff_path", "r0b_config_path", "r0b_summary_path", "r0b_source_path",
        "r0b_ramp_path", "cycle_sensor_path",
    )
    if set(cfg.get("input_sha256", {})) != set(keys):
        raise RuntimeError("input hash locks are incomplete")
    hashes: dict[str, str] = {}
    for key in keys:
        path = ROOT / str(cfg[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[key] = _sha256(path)
        if hashes[key] != str(cfg["input_sha256"][key]):
            raise RuntimeError(f"locked input drift for {key}: {hashes[key]}")
    r2 = json.loads((ROOT / str(cfg["r2_summary_path"])).read_text(encoding="utf-8"))
    r0b = json.loads((ROOT / str(cfg["r0b_summary_path"])).read_text(encoding="utf-8"))
    r0cfg = yaml.safe_load((ROOT / str(cfg["r0b_config_path"])).read_text(encoding="utf-8"))
    if r2.get("status") != "R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES":
        raise RuntimeError("R2 provenance status drifted")
    if r2.get("passing_tau_nodes") != [80.0]:
        raise RuntimeError("R2 no-go boundary drifted")
    if not all(bool(v) for k, v in r2.get("gates", {}).items() if k != "three_node_component_contains_preregistered_80s"):
        raise RuntimeError("R2 carries an unregistered failed gate")
    if r0b.get("status") != "R0B_RESERVE_COMPATIBLE_2D_CORRIDOR_SUPPORTED_R1_MAPPING_UNLOCKED" or not all(r0b.get("gates", {}).values()):
        raise RuntimeError("R0b provenance is not accepted")
    tables = {
        "event": _load_csv(ROOT / str(cfg["r2_event_path"])),
        "periodic": _load_csv(ROOT / str(cfg["r2_periodic_path"])),
        "schedule": _load_csv(ROOT / str(cfg["r2_schedule_path"])),
        "handoff": _load_csv(ROOT / str(cfg["r2_handoff_path"])),
        "r0_source": _load_csv(ROOT / str(cfg["r0b_source_path"])),
        "r0_ramp": _load_csv(ROOT / str(cfg["r0b_ramp_path"])),
    }
    fold_items = sorted(
        (float(q), float(row["additive_mv"]))
        for q, row in r0b["low_root_folds"].items() if bool(row.get("support_all"))
    )
    fold_q = np.asarray([x[0] for x in fold_items] + [float(cfg["model"]["entry_fold_q"])])
    fold_a = np.asarray([x[1] for x in fold_items] + [0.0])
    order = np.argsort(fold_q)
    fold_q, fold_a = fold_q[order], fold_a[order]
    if np.any(np.diff(fold_q) <= 0.0) or np.any(np.diff(fold_a) >= 0.0):
        raise RuntimeError("R0b fold interpolation is invalid")
    return hashes, r2, r0b, r0cfg, tables, fold_q, fold_a


def _regenerate_sensor(
    cfg: dict[str, Any], r0cfg: dict[str, Any], tables: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    _validate_inputs(r0cfg)
    transfer = _load_transfer(r0cfg)
    parameters, low_parameters = _model(r0cfg)
    geometry = r0cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, _ = _low_template(transfer, low_parameters)
    low_initial = _low_initial(low, float(r0cfg["model"]["z_interictal"]), reduction, parameters)
    inhibitory_baseline = np.asarray(low_initial[9:12], dtype=float)
    with np.load(ROOT / str(r0cfg["orbit_cycle_path"]), allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{r0cfg['r0b']['cycle_trace_key']}_state"], dtype=float)
    q_axis = [float(x) for x in cfg["model"]["q_hold_axis"]]
    phases = [float(x) for x in cfg["sensor"]["relative_phase_fractions"]]
    dts = [float(x) for x in cfg["sensor"]["source_dt_ms"]]
    expected = {_key(q, phase, dt) for q in q_axis for phase in phases for dt in dts}
    original_source = {
        _key(row["q"], row["phase"], row["dt_ms"]): row
        for row in tables["r0_source"] if float(row["q"]) in q_axis
    }
    original_ramp = {
        _key(row["q"], row["phase"], row["dt_ms"]): row
        for row in tables["r0_ramp"] if float(row["q"]) in q_axis
    }
    if set(original_source) != expected or set(original_ramp) != expected:
        raise RuntimeError("R0b source/ramp parity tables are incomplete")

    source_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("time_ms", "m", "use", "occupancy", "rE_khz", "rE_fast_khz")
    }
    record_q: list[float] = []
    record_phase: list[float] = []
    record_dt: list[float] = []
    ramp_arm = _ramp_parameters(r0cfg)
    for dt in dts:  # base and half dt are intentionally sequential.
        print(f"R3 sensor: starting source+ramp dt={dt:g} ms", flush=True)
        meta = [(q, phase) for q in q_axis for phase in phases]
        initial = np.asarray([
            _cycle_initial(low, cycle, phase, q, reduction, parameters) for q, phase in meta
        ])
        source = integrate_frozen_patch_batch(
            initial, prepared, transfer, dt_ms=dt,
            duration_ms=float(r0cfg["integration"]["source_prelude_ms"]),
            save_dt_ms=float(r0cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(r0cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(r0cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        checkpoints: list[np.ndarray] = []
        for index, (q, phase) in enumerate(meta):
            row = {"q": q, "phase": phase, "dt_ms": dt, **_pattern_summary(source, index, r0cfg, prepared, transfer)}
            parity = row["outcome"] == original_source[_key(q, phase, dt)]["outcome"] == "bounded_CCO"
            row["r0b_source_label_match"] = bool(parity)
            source_rows.append(row)
            if not parity:
                raise RuntimeError(f"source CCO parity failed at {(q, phase, dt)}")
            checkpoint, checkpoint_time = _checkpoint(source, index, int(r0cfg["r0b"]["source_min_returns_each_region"]))
            row["checkpoint_time_ms"] = checkpoint_time
            checkpoints.append(checkpoint)
        result = integrate_autonomous_latch_batch(
            np.asarray(checkpoints), prepared, transfer, [ramp_arm] * len(checkpoints), [],
            inhibitory_baseline_khz=inhibitory_baseline,
            dt_ms=dt, duration_ms=float(r0cfg["integration"]["ramp_post_ms"]),
            save_dt_ms=float(cfg["sensor"]["save_dt_ms"]),
            section_level_khz=float(r0cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(r0cfg["integration"]["rearm_level_rE_fast_khz"]),
            max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes_per_batch"]),
            initial_latch_state=np.tile(np.asarray([[True, True, False]], dtype=bool), (len(checkpoints), 1)),
        )
        print(f"R3 sensor: completed integration dt={dt:g} ms", flush=True)
        view = _frozen_view(result)
        for index, (q, phase) in enumerate(meta):
            pattern = _pattern_summary(view, index, r0cfg, prepared, transfer)
            key = _key(q, phase, dt)
            time_values = np.asarray(result["time_ms"], dtype=float)
            m_values = np.mean(np.asarray(result["m"][:, index, :2], dtype=float), axis=1)
            use_values = np.mean(np.asarray(result["z_use"][:, index, :2], dtype=float), axis=1)
            occupancy = np.asarray(result["occupancy"][:, index, :2], dtype=float)
            r_e = np.asarray(result["rE"][:, index, :], dtype=float)
            r_fast = np.asarray(result["rE_fast"][:, index, :], dtype=float)
            low_index = _first_sustained_low(
                time_values, r_fast[:, :2], float(cfg["sensor"]["regional_low_threshold_khz"]),
                float(cfg["sensor"]["sustained_low_ms"]),
            )
            z_values = np.asarray(result["z"][:, index, :2], dtype=float)
            additive = float(cfg["model"]["additive_max_mv"]) * m_values
            original = original_ramp[key]
            final_error = abs(float(additive[-1]) - float(original["final_additive_mv"]))
            row = {
                "record_index": len(sensor_rows), "q_hold": q, "phase": phase, "source_dt_ms": dt,
                "source_outcome": source_rows[-len(meta) + index]["outcome"],
                "ramp_outcome": pattern["outcome"],
                "source_label_match": source_rows[-len(meta) + index]["r0b_source_label_match"],
                "ramp_label_match": pattern["outcome"] == original["outcome"],
                "m_initial": float(m_values[0]), "m_final": float(m_values[-1]),
                "m_nondecreasing": bool(np.all(np.diff(m_values) >= -2.0e-7)),
                "low_state_index": low_index, "low_state_entry_ms": None if low_index is None else float(time_values[low_index]),
                "m_at_low_entry": None if low_index is None else float(m_values[low_index]),
                "use_at_low_entry": None if low_index is None else float(use_values[low_index]),
                "final_additive_mv": float(additive[-1]),
                "accepted_r0b_final_additive_mv": float(original["final_additive_mv"]),
                "final_additive_abs_error_mv": final_error,
                "max_abs_fixed_q_error": float(np.max(np.abs(z_values - q))),
                "finite": bool(result["finite"][index]),
                "support_violation_count": int(np.sum(result["support_violation_count"][index])),
                "bound_violation_count": int(np.sum(result["state_bound_violation_count"][index])),
                "core_return_count": len(result["return_times_ms"][index][0]),
                "annulus_return_count": len(result["return_times_ms"][index][1]),
            }
            row["sensor_pass"] = bool(
                row["source_label_match"] and row["ramp_label_match"]
                and row["ramp_outcome"] == "LLL" and row["finite"]
                and row["support_violation_count"] == 0 and row["bound_violation_count"] == 0
                and abs(row["m_initial"]) <= 1.0e-8 and row["m_nondecreasing"]
                and low_index is not None and row["max_abs_fixed_q_error"] < 1.0e-7
                and final_error <= float(cfg["sensor"]["final_additive_tolerance_mv"])
            )
            sensor_rows.append(row)
            record_q.append(q); record_phase.append(phase); record_dt.append(dt)
            arrays["time_ms"].append(time_values.astype(np.float32))
            arrays["m"].append(m_values.astype(np.float32))
            arrays["use"].append(use_values.astype(np.float32))
            arrays["occupancy"].append(occupancy.astype(np.float32))
            arrays["rE_khz"].append(r_e.astype(np.float32))
            arrays["rE_fast_khz"].append(r_fast.astype(np.float32))
    observed = [_key(row["q_hold"], row["phase"], row["source_dt_ms"]) for row in sensor_rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("regenerated sensor is not the exact unique 24-cell product")
    stacked = {name: np.stack(values) for name, values in arrays.items()}
    stacked.update({
        "q_hold": np.asarray(record_q), "phase": np.asarray(record_phase),
        "source_dt_ms": np.asarray(record_dt),
    })
    print("R3 sensor: complete 24-cell product", flush=True)
    return source_rows, sensor_rows, stacked


def _r2_parity(cfg: dict[str, Any], r2: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[float, bool], dict[tuple[float, float], dict[str, Any]]]:
    tau_axis = [float(x) for x in cfg["model"]["tau_slow_s_axis"]]
    q_axis = [float(x) for x in cfg["model"]["q_hold_axis"]]
    event = [row for row in tables["event"] if float(row["tau_recovery_s"]) in tau_axis]
    periodic = [row for row in tables["periodic"] if float(row["tau_recovery_s"]) in tau_axis]
    schedule = [row for row in tables["schedule"] if float(row["tau_recovery_s"]) in tau_axis]
    if len(event) != len(tau_axis) * len(q_axis) * 2:
        raise RuntimeError("R2 event parity rows incomplete")
    if len(periodic) != len(tau_axis) * len(q_axis) * 8:
        raise RuntimeError("R2 periodic parity rows incomplete")
    if len(schedule) == 0:
        raise RuntimeError("R2 schedule parity rows missing")
    mapping = {
        (float(row["tau_recovery_s"]), float(row["q_hold"])): row
        for row in r2["parameter_mapping"] if float(row["tau_recovery_s"]) in tau_axis
    }
    if len(mapping) != len(tau_axis) * len(q_axis):
        raise RuntimeError("R2 no-refit parameter mapping incomplete")
    schedule_pass = {
        float(row["tau_recovery_s"]): bool(row["primary_qhold_schedule_pass"])
        for row in r2["tau_acceptance"] if float(row["tau_recovery_s"]) in tau_axis
    }
    parity_rows: list[dict[str, Any]] = []
    for tau in tau_axis:
        for q in q_axis:
            local_event = [row for row in event if float(row["tau_recovery_s"]) == tau and float(row["q_hold"]) == q]
            local_periodic = [row for row in periodic if float(row["tau_recovery_s"]) == tau and float(row["q_hold"]) == q]
            source = mapping[(tau, q)]
            parity_rows.append({
                "tau_slow_s": tau, "q_hold": q,
                "event_row_count": len(local_event), "periodic_row_count": len(local_periodic),
                "entry_pass": bool(len(local_event) == 2 and all(bool(row["entry_pass"]) for row in local_event)),
                "periodic_pass": bool(len(local_periodic) == 8 and all(bool(row["periodic_pass"]) for row in local_periodic)),
                "schedule_pass": bool(schedule_pass[tau]),
                "r2_entry_label_match": bool(source["entry_pass"]) == bool(len(local_event) == 2 and all(bool(row["entry_pass"]) for row in local_event)),
                "r2_periodic_label_match": bool(source["periodic_pass"]) == bool(len(local_periodic) == 8 and all(bool(row["periodic_pass"]) for row in local_periodic)),
                "q_reserve": float(source["q_reserve"]),
                "tau_depletion_ms": float(source["tau_depletion_ms"]),
                "periodic_q_min": float(source["periodic_q_min"]),
            })
    return parity_rows, schedule_pass, mapping


def _plot(
    figures: Path, nullcline_rows: list[dict[str, Any]], parity_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]], path_rows: list[dict[str, Any]], tau_rows: list[dict[str, Any]],
    representative: dict[str, np.ndarray], gates: dict[str, bool], status: str, cfg: dict[str, Any],
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.8), constrained_layout=True)
    ax = axes[0, 0]
    for tau_fast in sorted({float(row["tau_fast_s"]) for row in nullcline_rows}):
        rows = [row for row in nullcline_rows if row["tau_fast_s"] == tau_fast and row["representative"]]
        ax.plot([row["m"] for row in rows], [row["q_star"] for row in rows], label=f"tau_fast={tau_fast:g} s")
    ax.axhline(float(cfg["model"]["entry_fold_q"]), color="k", ls="--", lw=1, label="entry fold")
    ax.set(xlabel="dimensionless M", ylabel="frozen q nullcline", title="A  M raises the unique stable q nullcline")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    tau_axis = [float(x) for x in cfg["model"]["tau_slow_s_axis"]]
    q_axis = [float(x) for x in cfg["model"]["q_hold_axis"]]
    matrix = np.zeros((len(q_axis), len(tau_axis)))
    labels = np.empty_like(matrix, dtype=object)
    for i, q in enumerate(q_axis):
        for j, tau in enumerate(tau_axis):
            row = next(x for x in parity_rows if x["tau_slow_s"] == tau and x["q_hold"] == q)
            matrix[i, j] = int(row["entry_pass"]) + int(row["schedule_pass"])
            labels[i, j] = f"E{int(row['entry_pass'])}/S{int(row['schedule_pass'])}"
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=2)
    ax.set_xticks(range(len(tau_axis)), [f"{x:g}" for x in tau_axis])
    ax.set_yticks(range(len(q_axis)), [f"{x:.4f}" for x in q_axis])
    for i in range(len(q_axis)):
        for j in range(len(tau_axis)):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8)
    ax.set(xlabel="tau_slow (s)", ylabel="q_hold", title="B  Inherited gates (E/S: 1=pass, 0=fail)")

    ax = axes[0, 2]
    t = representative["ramp_time_ms"] / 1000.0
    ax.plot(t, representative["ramp_m"], color="#d95f02", label="M")
    ax.plot(t, representative["ramp_use"], color="#7570b3", alpha=.7, label="use")
    ax.set(xlabel="time after CCO fork (s)", ylabel="sensor", title="C  Measured ramp and feed-forward q")
    ax2 = ax.twinx()
    ax2.plot(representative["replay_time_ms"] / 1000.0, representative["replay_q"], color="#1b9e77", label="q")
    ax2.set_ylabel("q")
    ax.set_xlim(0.0, 0.25)
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    ax = axes[1, 0]
    primary = float(cfg["model"]["tau_fast_primary_s"])
    matrix = np.zeros((len(q_axis), len(tau_axis)))
    labels = np.empty_like(matrix, dtype=object)
    for i, q in enumerate(q_axis):
        for j, tau in enumerate(tau_axis):
            rows = [row for row in path_rows if row["tau_fast_s"] == primary and row["tau_slow_s"] == tau and row["q_hold"] == q]
            path_ok = bool(rows and all(row["path_pass"] for row in rows))
            parity = next(row for row in parity_rows if row["tau_slow_s"] == tau and row["q_hold"] == q)
            full = path_ok and parity["entry_pass"] and parity["periodic_pass"] and parity["schedule_pass"]
            matrix[i, j] = 1 if full else 0
            codes = ""
            if not parity["entry_pass"]: codes += "E"
            if not parity["schedule_pass"]: codes += "S"
            if not path_ok: codes += "P"
            labels[i, j] = "PASS" if full else codes
    ax.imshow(matrix, aspect="auto", cmap=plt.get_cmap("RdYlGn", 2), vmin=0, vmax=1)
    ax.set_xticks(range(len(tau_axis)), [f"{x:g}" for x in tau_axis])
    ax.set_yticks(range(len(q_axis)), [f"{x:.4f}" for x in q_axis])
    for i in range(len(q_axis)):
        for j in range(len(tau_axis)):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8)
    ax.set(xlabel="tau_slow (s)", ylabel="q_hold", title="D  Full path (E=entry, S=schedule, P=path)")

    ax = axes[1, 1]
    ax.plot(representative["handoff_time_ms"] / 1000.0, representative["handoff_q"], color="#1b9e77", label="q")
    ax.axhline(float(cfg["handoff"]["q_reset_safe"]), color="#1b9e77", ls="--", lw=1)
    ax.set(xlabel="time after fast exit (s)", ylabel="q", title="E  Protected reset then M release")
    ax2 = ax.twinx()
    ax2.plot(representative["handoff_time_ms"] / 1000.0, representative["handoff_additive_mv"], color="#d95f02", label="A")
    ax2.set_ylabel("additive A (mV)")
    lines = ax.lines[:1] + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    color = "#1b7837" if status == SUPPORTED else "#b2182b"
    ax.text(0.02, .94, "F  Registered verdict", transform=ax.transAxes, fontsize=12, weight="bold", va="top")
    ax.text(0.02, .82, status, transform=ax.transAxes, fontsize=9, color=color, weight="bold", va="top", wrap=True)
    y = .67
    for name, value in gates.items():
        ax.text(.03, y, f"{'PASS' if value else 'FAIL'}  {name.replace('_', ' ')}", transform=ax.transAxes, fontsize=7.6, color="#1b7837" if value else "#b2182b", va="top")
        y -= .065
    fig.suptitle("M-gated inhibitory-capacity recovery: registered scalar/path oracle", fontsize=14)
    path = figures / "mz_m_gated_reserve_recovery.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    print("R3: config validated; loading locked upstream artifacts", flush=True)
    hashes, r2, r0b, r0cfg, tables, fold_q, fold_a = _load_inputs(cfg)
    parity_rows, schedule_pass, mapping = _r2_parity(cfg, r2, tables)
    print("R3: upstream parity validated; regenerating fixed-q M sensors", flush=True)
    source_rows, sensor_rows, sensor_npz = _regenerate_sensor(cfg, r0cfg, tables)
    q_axis = [float(x) for x in cfg["model"]["q_hold_axis"]]
    tau_axis = [float(x) for x in cfg["model"]["tau_slow_s_axis"]]
    fast_axis = [float(cfg["model"]["tau_fast_primary_s"])] + [float(x) for x in cfg["model"]["tau_fast_sensitivity_s"]]
    amax = float(cfg["model"]["additive_max_mv"])

    path_rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] = {}
    for tau_fast in fast_axis:
        for tau_slow in tau_axis:
            for q_hold in q_axis:
                cell = mapping[(tau_slow, q_hold)]
                for sensor_row in [row for row in sensor_rows if row["q_hold"] == q_hold]:
                    index = int(sensor_row["record_index"])
                    low_index = int(sensor_row["low_state_index"])
                    replay = replay_q_with_sensor(
                        sensor_npz["time_ms"][index], sensor_npz["use"][index], sensor_npz["m"][index],
                        q_initial=q_hold, q_rest=float(cfg["model"]["q_rest"]),
                        q_reserve=float(cell["q_reserve"]), tau_depletion_ms=float(cell["tau_depletion_ms"]),
                        tau_slow_s=tau_slow, tau_fast_s=tau_fast,
                        stop_time_ms=float(sensor_npz["time_ms"][index][low_index]),
                        reporting_dt_ms=float(cfg["sensor"]["replay_dt_ms"]),
                    )
                    q_exit = float(replay["q"][-1])
                    m_exit = float(replay["m"][-1])
                    additive_exit = amax * m_exit
                    in_domain = bool(np.all((replay["q"] >= fold_q[0]) & (replay["q"] <= fold_q[-1])))
                    margin_exit = additive_exit - float(np.interp(q_exit, fold_q, fold_a)) if in_domain else float("nan")
                    excursion = float(np.max(np.abs(replay["q"] - q_hold)))
                    replay_pass = bool(
                        np.all(np.isfinite(replay["q"])) and in_domain
                        and excursion <= float(cfg["sensor"]["maximum_q_excursion"])
                        and margin_exit >= float(cfg["sensor"]["dynamic_additive_margin_mv"])
                        and float(np.max(replay["q"])) < float(cfg["model"]["entry_fold_q"])
                    )
                    handoff = protected_handoff(
                        q_exit, m_exit, tau_slow_s=tau_slow, tau_fast_s=tau_fast,
                        q_rest=float(cfg["model"]["q_rest"]), fold_q=fold_q, fold_a=fold_a, cfg=cfg,
                    )
                    trace_fields = {name: handoff.pop(name) for name in ("release_time_ms", "release_m", "release_q")}
                    row = {
                        "tau_fast_s": tau_fast, "tau_slow_s": tau_slow, "q_hold": q_hold,
                        "phase": sensor_row["phase"], "source_dt_ms": sensor_row["source_dt_ms"],
                        "q_reserve": cell["q_reserve"], "tau_depletion_ms": cell["tau_depletion_ms"],
                        "low_state_entry_ms": sensor_row["low_state_entry_ms"],
                        "q_exit": q_exit, "m_exit": m_exit, "additive_exit_mv": additive_exit,
                        "max_abs_q_excursion": excursion, "fold_margin_at_exit_mv": margin_exit,
                        "replay_finite_domain_pass": bool(np.all(np.isfinite(replay["q"])) and in_domain),
                        "small_q_excursion_pass": excursion <= float(cfg["sensor"]["maximum_q_excursion"]),
                        "dynamic_fold_margin_pass": margin_exit >= float(cfg["sensor"]["dynamic_additive_margin_mv"]),
                        "active_q_below_entry_fold": float(np.max(replay["q"])) < float(cfg["model"]["entry_fold_q"]),
                        **handoff,
                    }
                    row["path_pass"] = bool(sensor_row["sensor_pass"] and replay_pass and handoff["handoff_pass"])
                    path_rows.append(row)
                    if tau_fast == 20.0 and tau_slow == 80.0 and q_hold == 0.8425 and sensor_row["phase"] == 0.0 and sensor_row["source_dt_ms"] == 0.125:
                        reset_time = float(handoff["reset_time_ms"])
                        protected_t = np.arange(0.0, reset_time + 1.0, 1.0)
                        protected_q = float(cfg["model"]["q_rest"]) - (float(cfg["model"]["q_rest"]) - q_exit) * np.exp(-protected_t / float(handoff["effective_tau_recovery_ms"]))
                        full_t = np.r_[protected_t, reset_time + trace_fields["release_time_ms"][1:]]
                        full_q = np.r_[protected_q, trace_fields["release_q"][1:]]
                        full_a = np.r_[np.full(protected_t.size, additive_exit), amax * trace_fields["release_m"][1:]]
                        representative = {
                            "ramp_time_ms": sensor_npz["time_ms"][index], "ramp_m": sensor_npz["m"][index],
                            "ramp_use": sensor_npz["use"][index], "replay_time_ms": replay["time_ms"],
                            "replay_q": replay["q"], "handoff_time_ms": full_t,
                            "handoff_q": full_q, "handoff_additive_mv": full_a,
                        }

    # Causal control 1: gate-off exactly reproduces the locked R2 analytic predictor.
    gate_off_rows: list[dict[str, Any]] = []
    r2_handoff = {(float(row["tau_recovery_s"]), float(row["q_hold"])): row for row in tables["handoff"]}
    a0 = {float(q): float(value) for q, value in r2["a0_by_q_hold"].items()}
    r2cfg = r2["config"]
    for tau_slow in tau_axis:
        for q_hold in q_axis:
            cell = mapping[(tau_slow, q_hold)]
            predicted = hybrid_handoff_predictor(
                q_start=float(cell["periodic_q_min"]), q_rest=float(cfg["model"]["q_rest"]),
                tau_recovery_ms=tau_slow * 1000.0, additive_start_mv=a0[q_hold],
                fold_q=fold_q, fold_a=fold_a, cfg=r2cfg,
            )
            locked = r2_handoff[(tau_slow, q_hold)]
            numeric_error = max(
                abs(float(predicted[name]) - float(locked[name]))
                for name in ("reset_time_ms", "q_at_additive_release", "minimum_registered_margin_mv")
            )
            gate_off_rows.append({
                "tau_slow_s": tau_slow, "q_hold": q_hold,
                "predicted_handoff_pass": bool(predicted["handoff_pass"]),
                "locked_r2_handoff_pass": bool(locked["handoff_pass"]),
                "maximum_numeric_error": numeric_error,
                "gate_off_reproduces_r2": bool(bool(predicted["handoff_pass"]) == bool(locked["handoff_pass"]) and numeric_error <= 1.0e-9),
            })

    # Causal control 2: with additive A off, even m=1 cannot move the CCO q-nullcline across entry.
    with np.load(ROOT / str(cfg["cycle_sensor_path"]), allow_pickle=False) as payload:
        cycle_q = np.asarray(payload["q_hold"], dtype=float)
        cycle_use = np.asarray(payload["use"], dtype=float)
    use_lower_by_q = {q: min(float(np.mean(cycle_use[i])) for i in np.flatnonzero(np.isclose(cycle_q, q))) for q in q_axis}
    gate_only_rows: list[dict[str, Any]] = []
    nullcline_rows: list[dict[str, Any]] = []
    m_axis = np.linspace(0.0, 1.0, 101)
    for tau_fast in fast_axis:
        for tau_slow in tau_axis:
            for q_hold in q_axis:
                cell = mapping[(tau_slow, q_hold)]
                qstar = q_nullcline(
                    m_axis, use_lower_by_q[q_hold], float(cfg["model"]["q_rest"]),
                    float(cell["q_reserve"]), float(cell["tau_depletion_ms"]), tau_slow, tau_fast,
                )
                gate_only_rows.append({
                    "tau_fast_s": tau_fast, "tau_slow_s": tau_slow, "q_hold": q_hold,
                    "mean_use_lower_bound": use_lower_by_q[q_hold], "q_star_m0": float(qstar[0]),
                    "q_star_m1_upper_bound": float(qstar[-1]),
                    "entry_fold_q": float(cfg["model"]["entry_fold_q"]),
                    "gate_only_below_entry_fold": bool(float(qstar[-1]) < float(cfg["model"]["entry_fold_q"])),
                })
                if tau_slow == 80.0 and q_hold == 0.8425:
                    nullcline_rows.extend({
                        "tau_fast_s": tau_fast, "tau_slow_s": tau_slow, "q_hold": q_hold,
                        "m": float(m_value), "q_star": float(q_value), "representative": True,
                    } for m_value, q_value in zip(m_axis, qstar))

    tau_rows: list[dict[str, Any]] = []
    for tau_fast in fast_axis:
        for tau_slow in tau_axis:
            cells = []
            for q_hold in q_axis:
                parity = next(row for row in parity_rows if row["tau_slow_s"] == tau_slow and row["q_hold"] == q_hold)
                paths = [row for row in path_rows if row["tau_fast_s"] == tau_fast and row["tau_slow_s"] == tau_slow and row["q_hold"] == q_hold]
                cells.append(bool(
                    len(paths) == 8 and all(row["path_pass"] for row in paths)
                    and parity["entry_pass"] and parity["periodic_pass"] and parity["schedule_pass"]
                ))
            tau_rows.append({
                "tau_fast_s": tau_fast, "tau_slow_s": tau_slow,
                "all_qhold_phase_dt_paths_pass": bool(all(cells)),
                "tau_node_pass": bool(all(cells)),
            })
    accepted_by_fast = {
        tau_fast: [row["tau_slow_s"] for row in tau_rows if row["tau_fast_s"] == tau_fast and row["tau_node_pass"]]
        for tau_fast in fast_axis
    }
    required = [float(x) for x in cfg["acceptance"]["required_corridor_tau_s"]]
    gates = {
        "hash_locked_upstream_and_r2_provenance": True,
        "complete_24_cell_sensor_product": len(sensor_rows) == 24 and len({_key(row["q_hold"], row["phase"], row["source_dt_ms"]) for row in sensor_rows}) == 24,
        "all_sensor_generation_gates_pass": all(row["sensor_pass"] for row in sensor_rows),
        "preentry_event_periodic_schedule_parity_exact": all(row["r2_entry_label_match"] and row["r2_periodic_label_match"] for row in parity_rows),
        "primary_exact_80_90_100_corridor": accepted_by_fast[20.0] == required,
        "fixed_15_25s_no_refit_corridor": all(accepted_by_fast[tau_fast] == required for tau_fast in (15.0, 25.0)),
        "registered_70_entry_and_120_160_schedule_rejections_preserved": bool(
            all(not next(row for row in parity_rows if row["tau_slow_s"] == 70.0 and row["q_hold"] == q)["entry_pass"] for q in [0.845])
            and not schedule_pass[120.0] and not schedule_pass[160.0]
        ),
        "gate_off_reproduces_r2": all(row["gate_off_reproduces_r2"] for row in gate_off_rows),
        "gate_only_cannot_cross_entry_fold": all(row["gate_only_below_entry_fold"] for row in gate_only_rows),
        "all_frozen_sensor_q_excursions_within_gate": all(row["small_q_excursion_pass"] for row in path_rows),
        "slow_off_returning_event_label_inherited_not_reinterpreted": True,
        "scope_remains_short_regional_state_fork_only": True,
    }
    resolved_names = (
        "hash_locked_upstream_and_r2_provenance", "complete_24_cell_sensor_product",
        "preentry_event_periodic_schedule_parity_exact", "gate_off_reproduces_r2",
        "gate_only_cannot_cross_entry_fold", "slow_off_returning_event_label_inherited_not_reinterpreted",
        "scope_remains_short_regional_state_fork_only",
    )
    resolved = all(gates[name] for name in resolved_names)
    supported = all(gates.values())
    status = SUPPORTED if supported else CLEAN_NO_GO if resolved else UNRESOLVED
    decision = (
        "run_short_P3_state_fork_tau_slow_80_90_100_all_qhold_base_half_dt"
        if supported else "stop_R3_and_proceed_to_biologically_separated_two_pool_resource_design"
    )
    output = ROOT / str(cfg["result_root"])
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary": str((output / "m_gated_reserve_recovery_summary.json").relative_to(ROOT)),
        "source_csv": str((output / "m_ramp_source_parity.csv").relative_to(ROOT)),
        "sensor_csv": str((output / "m_ramp_sensor.csv").relative_to(ROOT)),
        "sensor_npz": str((output / "m_ramp_sensor_traces.npz").relative_to(ROOT)),
        "preentry_csv": str((output / "preentry_r2_parity.csv").relative_to(ROOT)),
        "path_csv": str((output / "m_gated_path_oracle.csv").relative_to(ROOT)),
        "tau_csv": str((output / "m_gated_tau_acceptance.csv").relative_to(ROOT)),
        "gate_off_csv": str((output / "control_gate_off_r2_parity.csv").relative_to(ROOT)),
        "gate_only_csv": str((output / "control_gate_only_nullcline.csv").relative_to(ROOT)),
        "nullcline_csv": str((output / "m_gated_nullcline.csv").relative_to(ROOT)),
        "representative_npz": str((output / "m_gated_representative_traces.npz").relative_to(ROOT)),
        "figure": str((figures / "mz_m_gated_reserve_recovery.png").relative_to(ROOT)),
    }
    summary = {
        "status": status,
        "scientific_layer": "fixed_q_fast_sensor_plus_feedforward_qM_scalar_path_oracle_not_coupled_lifecycle",
        "decision": decision,
        "gates": gates,
        "registered_sensor_cell_count": len(sensor_rows),
        "sensor_pass_count": sum(bool(row["sensor_pass"]) for row in sensor_rows),
        "path_row_count": len(path_rows),
        "expected_path_row_count": len(fast_axis) * len(tau_axis) * len(q_axis) * 8,
        "accepted_tau_slow_by_tau_fast": {str(key): value for key, value in accepted_by_fast.items()},
        "maximum_sensor_q_excursion": max(float(row["max_abs_q_excursion"]) for row in path_rows),
        "maximum_gate_only_q_star": max(float(row["q_star_m1_upper_bound"]) for row in gate_only_rows),
        "input_sha256": hashes,
        "r2_provenance_status": r2["status"],
        "r0b_provenance_status": r0b["status"],
        "claim_boundary": [
            "additive M supplies the fast exit whereas M-gated q recovery supplies timely reset; the two causal effects are not interchangeable",
            "q is replayed feed-forward on a regenerated frozen-q M sensor and is not coupled back to the fast regional model",
            "the q-M slow block remains locally stable and triangular; R3 does not establish a Hopf, torus, second q fixed point, or smooth autonomous limit cycle",
            "passing unlocks only the registered short regional P3 state-fork at tau_slow=80/90/100 s",
            "the fixed bath mask remains imposed and no continuous field, full SNN, spatial containment, wavefront annihilation, E-E, conductance, or relay claim is unlocked",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "config": cfg,
        "artifacts": artifacts,
        "plot_status": "pending",
    }

    # Persist all scientific artifacts before plotting.
    _save_csv(output / "m_ramp_source_parity.csv", source_rows)
    _save_csv(output / "m_ramp_sensor.csv", sensor_rows)
    np.savez_compressed(output / "m_ramp_sensor_traces.npz", **sensor_npz)
    _save_csv(output / "preentry_r2_parity.csv", parity_rows)
    _save_csv(output / "m_gated_path_oracle.csv", path_rows)
    _save_csv(output / "m_gated_tau_acceptance.csv", tau_rows)
    _save_csv(output / "control_gate_off_r2_parity.csv", gate_off_rows)
    _save_csv(output / "control_gate_only_nullcline.csv", gate_only_rows)
    _save_csv(output / "m_gated_nullcline.csv", nullcline_rows)
    np.savez_compressed(output / "m_gated_representative_traces.npz", **representative)
    _save_json(output / "m_gated_reserve_recovery_summary.json", summary)

    figures.mkdir(parents=True, exist_ok=True)
    figure = _plot(figures, nullcline_rows, parity_rows, sensor_rows, path_rows, tau_rows, representative, gates, status, cfg)
    summary["artifacts"]["figure"] = str(figure.relative_to(ROOT))
    summary["plot_status"] = "complete"
    summary["runtime_seconds"] = time.perf_counter() - started
    summary["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if summary["peak_rss_kib"] > float(cfg["resource_contract"]["max_memory_gib"]) * 1024.0 * 1024.0:
        summary["status"] = UNRESOLVED
        summary["decision"] = "stop_R3_resource_contract_exceeded"
        summary["gates"]["peak_rss_below_1p5_gib"] = False
    else:
        summary["gates"]["peak_rss_below_1p5_gib"] = True
    _save_json(output / "m_gated_reserve_recovery_summary.json", summary)
    (figures / "README.md").write_text(
        "### mz_m_gated_reserve_recovery.png\n\n"
        "这张 2×3 图检验已有的无量纲 M 状态能否只通过切换 q 的恢复速率，化解 R2 中“发作前需要慢恢复、退出后需要快恢复”的冲突。A 显示 M 只上移唯一稳定的 q-nullcline；B 保留 R2 的 entry/schedule 边界；C 使用重新生成的 24-cell fixed-q M-ramp，而不是把 M 跳到终点；D 给出 primary arm 的完整 path gate；E 显示 latch reset 前 M 冻结、reset 后才以 12 s 释放；F 列出 fail-closed verdict。\n\n"
        f"当前状态为 `{summary['status']}`，tau_fast=20/15/25 s 对应的通过节点分别为 {accepted_by_fast[20.0]}、{accepted_by_fast[15.0]}、{accepted_by_fast[25.0]}。\n\n"
        "这仍不是 coupled lifecycle：q 没有反馈到 fast regional dynamics，fixed bath mask 也仍是 imposed boundary。即使通过，也只解锁 tau_slow=80/90/100 s 的短 P3 state-fork。\n\n"
        "**关注点**：必须同时看 frozen-sensor q excursion、动态 fold margin、120 s reset、gate-off R2 parity 和 gate-only nullcline control；不能把 recovery gate 单独写成 termination。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps({
        "status": summary["status"], "decision": summary["decision"],
        "accepted_tau_slow_by_tau_fast": summary["accepted_tau_slow_by_tau_fast"],
        "runtime_seconds": summary["runtime_seconds"], "peak_rss_kib": summary["peak_rss_kib"],
        "gates": summary["gates"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
