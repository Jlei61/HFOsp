#!/usr/bin/env python3
"""Run the cheap dual-sensor timing/leverage screen for additive M recovery.

This runner does not integrate the full nine-state fast system.  It combines a
locked fixed-point exit-current surface with two independently sourced causal
persistence histories:

1. the recorded spatial SNN ``U_TG`` history; and
2. the alpha_G=15 homogeneous fast-cycle trace from additive continuation.

The same persistence threshold and M kinetics must pass both brackets.  A
single-bracket success is treated as sensor-transfer mismatch, not lifecycle
evidence.
"""

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
SNN_ENGINE = ROOT / "src" / "snn_engine"
if str(SNN_ENGINE) not in os.sys.path:
    os.sys.path.insert(0, str(SNN_ENGINE))

from mz_divisive_pool import slow_gate_drive  # noqa: E402
from src.topic4_mz_persistence_feasibility import (  # noqa: E402
    causal_sustained_onset_ms,
    classify_leverage_race,
    integrate_bounded_effector,
    integrate_lowpass,
    required_additive_from_fold,
    unopposed_z,
)
from src.topic4_spatial_slowfast_stage0c import recruitment_sensor  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_persistence_feasibility.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    keys = (
        "current_capture_path",
        "current_capture_json",
        "entry_exit_summary_path",
        "orbit_summary_path",
        "orbit_cycle_path",
    )
    expected = cfg["input_sha256"]
    if set(expected) != set(keys):
        raise ValueError(f"input_sha256 must lock exactly {keys}")
    observed: dict[str, str] = {}
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"missing locked input: {path}")
        observed[key] = _sha256(path)
        if observed[key] != str(expected[key]):
            raise RuntimeError(
                f"locked input drift for {key}: expected {expected[key]}, observed {observed[key]}"
            )
    return observed


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _threshold_for_tau(mapping: dict, tau_ms: float) -> float:
    for key, value in mapping.items():
        if np.isclose(float(key), float(tau_ms)):
            return float(value)
    raise KeyError(f"no locked persistence threshold for tau={tau_ms}")


def _load_capture(cfg: dict) -> tuple[dict[str, np.ndarray], dict, float, int, np.ndarray]:
    meta = json.loads((ROOT / cfg["current_capture_json"]).read_text(encoding="utf-8"))
    dt_ms = float(meta["simulation"]["dt_ms"])
    with np.load(ROOT / cfg["current_capture_path"], allow_pickle=False) as payload:
        capture = {
            key: np.asarray(payload[key], dtype=float)
            for key in ("times_ms", "rate_E_hz", "slow_UTG", "slow_TG")
        }
    lengths = {array.size for array in capture.values()}
    if lengths != {capture["times_ms"].size}:
        raise RuntimeError("capture arrays are not aligned")
    onset_ms, causal_envelope = causal_sustained_onset_ms(
        capture["rate_E_hz"],
        dt_ms=dt_ms,
        envelope_ms=float(cfg["causal_onset"]["envelope_ms"]),
        threshold_hz=float(cfg["causal_onset"]["threshold_hz"]),
        minimum_duration_ms=float(cfg["causal_onset"]["minimum_duration_ms"]),
    )
    onset_index = int(round(onset_ms / dt_ms))
    if not np.isclose(onset_ms, float(cfg["causal_onset"]["expected_onset_ms"]), atol=dt_ms):
        raise RuntimeError(f"causal onset drifted: {onset_ms}")
    return capture, meta, dt_ms, onset_index, causal_envelope


def _load_fold_surface(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    summary = json.loads((ROOT / cfg["entry_exit_summary_path"]).read_text(encoding="utf-8"))
    rows = summary["fixed_point_fold_surface"]
    fold_z = np.asarray([row["z"] for row in rows], dtype=float)
    fold_a = np.asarray([row["additive_mv"] for row in rows], dtype=float)
    order = np.argsort(fold_z)
    if np.any(np.diff(fold_z[order]) <= 0.0) or np.any(np.diff(fold_a[order]) >= 0.0):
        raise RuntimeError("fixed-point fold surface lost its monotone inverse")
    return fold_z, fold_a, summary


def _load_phase_cycle(cfg: dict) -> tuple[np.ndarray, float, dict]:
    summary = json.loads((ROOT / cfg["orbit_summary_path"]).read_text(encoding="utf-8"))
    contract = summary["model_contract"]
    if "Stage0C" not in contract["fast_system"]:
        raise RuntimeError("orbit source is not the locked Stage0C system")
    key = str(cfg["phase_sensor"]["orbit_trace_key"])
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        time = np.asarray(payload[f"{key}_time_ms"], dtype=float)
        state = np.asarray(payload[f"{key}_state"], dtype=float)
    if time.ndim != 1 or state.shape != (time.size, 9) or time.size < 3:
        raise RuntimeError("invalid orbit trace shape")
    if not np.isclose(time[0], 0.0, atol=1e-9) or np.any(np.diff(time) <= 0.0):
        raise RuntimeError("orbit trace is not a directed time series")
    period_ms = float(time[-1])
    dt_ms = float(cfg["phase_sensor"]["dt_ms"])
    phase_time = np.arange(0.0, period_ms, dt_ms)
    r_e_fast = np.interp(phase_time, time, state[:, 6])
    raw_a_g = recruitment_sensor(r_e_fast)
    gate_cfg = cfg["persistence_gate_drive"]
    raw_drive = np.asarray([
        slow_gate_drive(
            value,
            A0=float(gate_cfg["A0"]),
            A50=float(gate_cfg["A50"]),
            exponent=float(gate_cfg["exponent"]),
        )
        for value in raw_a_g
    ])
    if raw_drive.size < 2 or not np.all(np.isfinite(raw_drive)):
        raise RuntimeError("failed to reconstruct raw phase-cycle persistence drive")
    source = {
        "trace_key": key,
        "period_ms": period_ms,
        "resampled_dt_ms": dt_ms,
        "cycle_samples": int(raw_drive.size),
        "raw_drive_mean": float(np.mean(raw_drive)),
        "raw_drive_max": float(np.max(raw_drive)),
        "drive_definition": "slow_gate_drive(recruitment_sensor(rE_fast))",
        "alpha_G": float(cfg["phase_sensor"]["alpha_G"]),
    }
    return raw_drive, period_ms, source


def _required_current(
    time_ms: np.ndarray,
    fold_z: np.ndarray,
    fold_a: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    oracle = cfg["z_oracle"]
    z = unopposed_z(
        time_ms,
        z_start=float(oracle["z_start"]),
        depletion_occupancy=float(oracle["depletion_occupancy"]),
        tau_z_ms=float(oracle["tau_z_ms"]),
    )
    required = required_additive_from_fold(z, fold_z=fold_z, fold_additive_mv=fold_a)
    return z, required


def _race_row(
    *,
    sensor: str,
    phase_fraction: float | None,
    tau_p_ms: float,
    threshold: float,
    arm: dict,
    time_ms: np.ndarray,
    persistence: np.ndarray,
    required: np.ndarray,
    period_ms: float,
    cfg: dict,
    gate_mode: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if gate_mode not in {"memoryless", "latched_after_first_activation"}:
        raise ValueError(f"unknown diagnostic gate mode {gate_mode!r}")
    effector, gate = integrate_bounded_effector(
        persistence,
        dt_ms=float(time_ms[1] - time_ms[0]),
        gate_low=threshold,
        gate_high=threshold,
        tau_up_ms=float(arm["tau_up_ms"]),
        tau_down_ms=float(cfg["effector"]["tau_down_ms"]),
        unsafe_decay_fraction=float(cfg["effector"]["unsafe_decay_fraction"]),
        initial=0.0,
        latch_after_first_activation=gate_mode == "latched_after_first_activation",
    )
    available = float(arm["Amax_mv"]) * effector
    # Index zero is the entry boundary A_required=0; it is not an exit crossing.
    race = classify_leverage_race(
        time_ms[1:], available[1:], required[1:],
        minimum_cycles=float(cfg["acceptance"]["minimum_cycles"]),
        maximum_cycles=float(cfg["acceptance"]["maximum_cycles"]),
        cycle_period_ms=period_ms,
    )
    row = {
        "sensor": sensor,
        "gate_mode": gate_mode,
        "phase_fraction": phase_fraction,
        "tau_p_ms": tau_p_ms,
        "p_threshold": threshold,
        "arm_id": str(arm["id"]),
        "Amax_mv": float(arm["Amax_mv"]),
        "tau_up_ms": float(arm["tau_up_ms"]),
        "tau_down_ms": float(cfg["effector"]["tau_down_ms"]),
        "unsafe_decay_fraction": float(cfg["effector"]["unsafe_decay_fraction"]),
        "cycle_count_semantics": "A0_baseline_period_equivalents_not_dynamic_returns",
        "gate_first_on_ms": (
            float(time_ms[np.flatnonzero(gate > 0.0)[0]]) if np.any(gate > 0.0) else None
        ),
        **race,
    }
    return row, effector, available


def _plot(
    figures: Path,
    period_ms: float,
    primary_tau: float,
    threshold: float,
    primary_arm: dict,
    snn_trace: dict,
    latched_snn_trace: dict,
    phase_traces: dict[float, dict],
    rows: list[dict],
) -> Path:
    colors = {0.0: "#762a83", 0.25: "#1b7837", 0.5: "#2166ac", 0.75: "#d6604d"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    t_snn_cycles = snn_trace["time_ms"] / period_ms
    ax_a.plot(t_snn_cycles, snn_trace["persistence"], color="black", lw=1.6,
              label="recorded spatial SNN replay")
    for phase, trace in phase_traces.items():
        ax_a.plot(trace["time_ms"] / period_ms, trace["persistence"],
                  color=colors.get(phase, "0.5"), lw=1.0, alpha=0.9,
                  label=f"0D phase {phase:.2f}")
    ax_a.axhline(threshold, color="#b2182b", ls="--", lw=1.2, label="locked gate")
    ax_a.axvspan(3.0, 5.0, color="#f7f7d5", alpha=0.7, lw=0)
    ax_a.set(xlabel="baseline-cycle equivalents after entry", ylabel="persistence p",
             title=f"A  Same persistence gate (tau={primary_tau:g} ms)")
    ax_a.legend(frameon=False, fontsize=7.2, ncol=2)

    ax_b.plot(t_snn_cycles, snn_trace["required"], color="#2166ac", lw=1.5,
              label="required A from fold")
    ax_b.plot(t_snn_cycles, snn_trace["available"], color="#b2182b", lw=1.5,
              label=f"memoryless A ({primary_arm['id']})")
    ax_b.plot(t_snn_cycles, latched_snn_trace["available"], color="#e08214", lw=1.35,
              ls="--", label="latched-after-detection diagnostic")
    ax_b.axvspan(3.0, 5.0, color="#f7f7d5", alpha=0.7, lw=0)
    ax_b.set(xlabel="baseline-cycle equivalents after entry", ylabel="additive current (mV)",
             title="B  Recorded SNN timing/leverage race")
    ax_b.legend(frameon=False, fontsize=7.5)

    ax_c.plot(
        next(iter(phase_traces.values()))["time_ms"] / period_ms,
        next(iter(phase_traces.values()))["required"],
        color="#2166ac", lw=1.5, label="required A from fold",
    )
    for phase, trace in phase_traces.items():
        ax_c.plot(trace["time_ms"] / period_ms, trace["available"],
                  color=colors.get(phase, "0.5"), lw=1.2, label=f"phase {phase:.2f}")
    ax_c.axvspan(3.0, 5.0, color="#f7f7d5", alpha=0.7, lw=0)
    ax_c.set(xlabel="baseline-cycle equivalents after entry", ylabel="additive current (mV)",
             title="C  Mature-cycle sensor-transfer stress test")
    ax_c.legend(frameon=False, fontsize=7.3, ncol=2)

    status_code = {
        "too_early_or_prevention_risk": 0,
        "timing_leverage_feasible": 1,
        "too_late_for_registered_window": 2,
        "insufficient_leverage": 3,
    }
    labels = list(status_code)
    primary_rows = [
        row for row in rows
        if np.isclose(row["tau_p_ms"], primary_tau) and row["gate_mode"] == "memoryless"
    ]
    x_labels = sorted({row["arm_id"] for row in primary_rows})
    sensors = ["recorded_SNN_UTG_replay", "endogenous_phase_all_offsets"]
    matrix = np.full((2, len(x_labels)), np.nan)
    for column, arm_id in enumerate(x_labels):
        snn = next(row for row in primary_rows if row["arm_id"] == arm_id and row["sensor"] == sensors[0])
        phase = [row for row in primary_rows if row["arm_id"] == arm_id and row["sensor"] == "endogenous_phase_sensor"]
        matrix[0, column] = status_code[snn["status"]]
        if all(row["status"] == "timing_leverage_feasible" for row in phase):
            matrix[1, column] = status_code["timing_leverage_feasible"]
        elif any(row["status"] == "too_early_or_prevention_risk" for row in phase):
            matrix[1, column] = status_code["too_early_or_prevention_risk"]
        elif any(row["status"] == "too_late_for_registered_window" for row in phase):
            matrix[1, column] = status_code["too_late_for_registered_window"]
        else:
            matrix[1, column] = status_code["insufficient_leverage"]
    cmap = matplotlib.colors.ListedColormap(["#d73027", "#1a9850", "#fee08b", "#8073ac"])
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
    ax_d.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax_d.set_xticks(range(len(x_labels)), x_labels, rotation=25, ha="right")
    ax_d.set_yticks(range(2), ["SNN replay", "all 0D phases"])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax_d.text(column, row, labels[int(matrix[row, column])].replace("_", "\n"),
                      ha="center", va="center", fontsize=6.3,
                      color="white" if matrix[row, column] in (0, 3) else "black")
    ax_d.set_title("D  Primary dual-sensor verdict")

    fig.suptitle("Persistence-gated additive recovery: necessary-condition screen",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.012,
        "A common arm must be feasible in the recorded spatial history and every registered 0D phase; single-bracket success is not a lifecycle pass.",
        ha="center", fontsize=8.0, color="#7f0000",
    )
    stem = figures / "mz_persistence_dual_sensor_feasibility"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["acceptance"].get("cycle_count_semantics") != (
        "A0_baseline_period_equivalents_not_dynamic_returns"
    ):
        raise ValueError("Stage-B cycle-count semantics must remain explicit")
    observed_hashes = _validate_inputs(cfg)
    capture, capture_meta, capture_dt, onset_index, causal_envelope = _load_capture(cfg)
    fold_z, fold_a, entry_summary = _load_fold_surface(cfg)
    cycle_drive, period_ms, phase_source = _load_phase_cycle(cfg)

    duration_ms = float(cfg["acceptance"]["maximum_cycles"]) * period_ms
    phase_dt = float(cfg["phase_sensor"]["dt_ms"])
    phase_time = np.arange(0.0, duration_ms + 0.5 * phase_dt, phase_dt)
    phase_required_z, phase_required = _required_current(phase_time, fold_z, fold_a, cfg)
    snn_n = int(np.floor(duration_ms / capture_dt)) + 1
    snn_time = np.arange(snn_n, dtype=float) * capture_dt
    if onset_index + snn_n > capture["slow_UTG"].size:
        raise RuntimeError("recorded SNN replay does not cover the registered cycle window")
    snn_required_z, snn_required = _required_current(snn_time, fold_z, fold_a, cfg)

    rows: list[dict] = []
    threshold_rows: list[dict] = []
    traces: dict[str, np.ndarray] = {
        "snn_time_ms": snn_time.astype(np.float32),
        "snn_z_oracle": snn_required_z.astype(np.float32),
        "snn_required_additive_mv": snn_required.astype(np.float32),
        "phase_time_ms": phase_time.astype(np.float32),
        "phase_z_oracle": phase_required_z.astype(np.float32),
        "phase_required_additive_mv": phase_required.astype(np.float32),
        "causal_rate_envelope_hz": causal_envelope.astype(np.float32),
    }
    primary_tau = float(cfg["persistence"]["primary_tau_p_ms"])
    primary_arm_id = str(cfg["plot"]["primary_arm_id"])
    primary_arm = next(arm for arm in cfg["arms"] if str(arm["id"]) == primary_arm_id)
    primary_snn_trace: dict | None = None
    latched_primary_snn_trace: dict | None = None
    primary_phase_traces: dict[float, dict] = {}
    phase_fractions = [float(value) for value in cfg["phase_sensor"]["phase_fractions"]]
    gate_modes = [str(value) for value in cfg["effector"]["diagnostic_gate_modes"]]
    primary_gate_mode = str(cfg["effector"]["primary_gate_mode"])
    if primary_gate_mode != "memoryless" or primary_gate_mode not in gate_modes:
        raise ValueError("the registered primary gate must be memoryless")

    for tau_p_ms in map(float, cfg["persistence"]["tau_p_ms"]):
        threshold = _threshold_for_tau(cfg["persistence"]["threshold_by_tau"], tau_p_ms)
        full_p = integrate_lowpass(
            capture["slow_UTG"], dt_ms=capture_dt, tau_ms=tau_p_ms, initial=0.0
        )
        pre = np.arange(full_p.size) < onset_index
        post = np.arange(full_p.size) * capture_dt >= onset_index * capture_dt + float(
            cfg["persistence"]["post_quantile_delay_ms"]
        )
        pre_max = float(np.max(full_p[pre]))
        post_q = float(np.quantile(full_p[post], float(cfg["persistence"]["post_quantile"])))
        derived_midpoint = 0.5 * (pre_max + post_q)
        if not np.isclose(
            threshold, derived_midpoint, atol=float(cfg["persistence"]["threshold_validation_atol"])
        ):
            raise RuntimeError(
                f"locked threshold drift for tau={tau_p_ms}: locked {threshold}, derived {derived_midpoint}"
            )
        p_onset = float(full_p[onset_index])
        snn_p = full_p[onset_index:onset_index + snn_n]
        crossing = np.flatnonzero(snn_p >= threshold)
        threshold_rows.append({
            "tau_p_ms": tau_p_ms,
            "p_onset": p_onset,
            "pre_onset_max": pre_max,
            "post_onset_q25": post_q,
            "locked_threshold": threshold,
            "derived_midpoint": derived_midpoint,
            "first_crossing_after_onset_ms": (
                float(crossing[0] * capture_dt) if crossing.size else None
            ),
        })
        traces[f"snn_p_tau_{tau_p_ms:g}"] = snn_p.astype(np.float32)

        phase_p_by_fraction: dict[float, np.ndarray] = {}
        for phase_fraction in phase_fractions:
            shift = int(round(phase_fraction * cycle_drive.size)) % cycle_drive.size
            shifted = np.roll(cycle_drive, -shift)
            repeated = np.resize(shifted, phase_time.size)
            persistence = integrate_lowpass(
                repeated, dt_ms=phase_dt, tau_ms=tau_p_ms, initial=p_onset
            )
            phase_p_by_fraction[phase_fraction] = persistence
            traces[
                f"phase_p_tau_{tau_p_ms:g}_fraction_{phase_fraction:.2f}".replace(".", "p")
            ] = persistence.astype(np.float32)

        for arm in cfg["arms"]:
            for gate_mode in gate_modes:
                snn_row, snn_m, snn_available = _race_row(
                    sensor="recorded_SNN_UTG_replay", phase_fraction=None,
                    tau_p_ms=tau_p_ms, threshold=threshold, arm=arm,
                    time_ms=snn_time, persistence=snn_p, required=snn_required,
                    period_ms=period_ms, cfg=cfg, gate_mode=gate_mode,
                )
                rows.append(snn_row)
                for phase_fraction, phase_p in phase_p_by_fraction.items():
                    phase_row, phase_m, phase_available = _race_row(
                        sensor="endogenous_phase_sensor", phase_fraction=phase_fraction,
                        tau_p_ms=tau_p_ms, threshold=threshold, arm=arm,
                        time_ms=phase_time, persistence=phase_p, required=phase_required,
                        period_ms=period_ms, cfg=cfg, gate_mode=gate_mode,
                    )
                    rows.append(phase_row)
                    if (
                        gate_mode == primary_gate_mode
                        and np.isclose(tau_p_ms, primary_tau)
                        and str(arm["id"]) == primary_arm_id
                    ):
                        primary_phase_traces[phase_fraction] = {
                            "time_ms": phase_time,
                            "persistence": phase_p,
                            "effector": phase_m,
                            "available": phase_available,
                            "required": phase_required,
                        }
                if (
                    gate_mode == primary_gate_mode
                    and np.isclose(tau_p_ms, primary_tau)
                    and str(arm["id"]) == primary_arm_id
                ):
                    primary_snn_trace = {
                        "time_ms": snn_time,
                        "persistence": snn_p,
                        "effector": snn_m,
                        "available": snn_available,
                        "required": snn_required,
                    }
                elif (
                    gate_mode == "latched_after_first_activation"
                    and np.isclose(tau_p_ms, primary_tau)
                    and str(arm["id"]) == primary_arm_id
                ):
                    latched_primary_snn_trace = {
                        "time_ms": snn_time,
                        "persistence": snn_p,
                        "effector": snn_m,
                        "available": snn_available,
                        "required": snn_required,
                    }

    if (
        primary_snn_trace is None
        or latched_primary_snn_trace is None
        or set(primary_phase_traces) != set(phase_fractions)
    ):
        raise RuntimeError("primary plot traces were not generated")

    arm_verdicts: list[dict] = []
    for tau_p_ms in map(float, cfg["persistence"]["tau_p_ms"]):
        for arm in cfg["arms"]:
            for gate_mode in gate_modes:
                selected = [
                    row for row in rows
                    if np.isclose(row["tau_p_ms"], tau_p_ms)
                    and row["arm_id"] == str(arm["id"])
                    and row["gate_mode"] == gate_mode
                ]
                snn = next(row for row in selected if row["sensor"] == "recorded_SNN_UTG_replay")
                phase = [row for row in selected if row["sensor"] == "endogenous_phase_sensor"]
                snn_pass = snn["status"] == "timing_leverage_feasible"
                phase_all_pass = bool(phase) and all(
                    row["status"] == "timing_leverage_feasible" for row in phase
                )
                phase_any_early = any(
                    row["status"] == "too_early_or_prevention_risk" for row in phase
                )
                arm_verdicts.append({
                    "tau_p_ms": tau_p_ms,
                    "arm_id": str(arm["id"]),
                    "gate_mode": gate_mode,
                    "snn_replay_feasible": snn_pass,
                    "all_phase_offsets_feasible": phase_all_pass,
                    "any_phase_offset_too_early": phase_any_early,
                    "common_dual_sensor_feasible": snn_pass and phase_all_pass,
                    "interpretation": (
                        "common_candidate" if snn_pass and phase_all_pass
                        else "sensor_transfer_mismatch" if snn_pass != phase_all_pass or phase_any_early
                        else "common_leverage_failure"
                    ),
                })

    primary_common = [
        row for row in arm_verdicts
        if np.isclose(row["tau_p_ms"], primary_tau)
        and row["gate_mode"] == primary_gate_mode
        and row["common_dual_sensor_feasible"]
    ]
    any_common = [
        row for row in arm_verdicts
        if row["gate_mode"] == primary_gate_mode and row["common_dual_sensor_feasible"]
    ]
    latched_snn_candidates = [
        row for row in arm_verdicts
        if row["gate_mode"] == "latched_after_first_activation" and row["snn_replay_feasible"]
    ]
    if primary_common:
        status = "primary_dual_sensor_candidate_ready_for_full_0d"
    elif any_common:
        status = "sensitivity_only_dual_sensor_candidate_primary_not_ready"
    else:
        status = "sensor_transfer_mismatch_no_common_0d_feasible_arm"

    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "persistence_threshold_audit.csv", threshold_rows)
    _write_csv(output / "dual_sensor_races.csv", rows)
    _write_csv(output / "dual_sensor_arm_verdicts.csv", arm_verdicts)
    np.savez_compressed(output / "dual_sensor_traces.npz", **traces)
    figure = _plot(
        figures, period_ms, primary_tau,
        _threshold_for_tau(cfg["persistence"]["threshold_by_tau"], primary_tau),
        primary_arm, primary_snn_trace, latched_primary_snn_trace, primary_phase_traces, rows,
    )
    summary = {
        "status": status,
        "scientific_layer": "necessary_condition_timing_leverage_screen_not_full_lifecycle",
        "causal_onset_ms": float(onset_index * capture_dt),
        "phase_source": phase_source,
        "persistence_threshold_audit": threshold_rows,
        "arm_verdicts": arm_verdicts,
        "primary_common_candidates": primary_common,
        "any_registered_common_candidates": any_common,
        "post_no_go_latched_gate_diagnostic": {
            "can_regrade_primary": False,
            "snn_replay_candidates": latched_snn_candidates,
            "interpretation": (
                "tests whether post-detection gate retention restores SNN leverage; "
                "it does not repair premature homogeneous phase sensing"
            ),
        },
        "stop_rule": (
            "stop_before_full_0d_and_spatial_SNN" if not primary_common
            else "proceed_only_to_full_0d_Z_p_m"
        ),
        "key_contracts": [
            "same tau_p, threshold, Amax, and M kinetics in both sensor brackets",
            "phase drive reconstructed as slow_gate_drive(recruitment_sensor(rE_fast))",
            "recorded SNN U_TG replay retains its old divisive-feedback history and is a stress bracket",
            "M starts at zero but p inherits the complete causal prehistory",
            "first entry-boundary equality is excluded from exit detection",
            "required additive current is interpolated only within the locked fold surface",
            "latched gate is a post-no-go diagnostic and cannot re-grade the memoryless primary",
            "cycle counts are A=0 baseline-period equivalents, not returns of a dynamic slow trajectory",
        ],
        "claim_boundary": [
            "the fold surface is a fixed-point exit-current oracle until cycle-boundary continuation is interpreted",
            "this screen does not integrate feedback from M to the fast cycle or from state exit to Z recovery",
            "no low-state return, refractoriness, spatial containment, or SNN lifecycle is claimed",
            "seed-1 is the only spatial U_TG history currently available",
        ],
        "input_sha256": observed_hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "threshold_csv": str((output / "persistence_threshold_audit.csv").relative_to(ROOT)),
            "races_csv": str((output / "dual_sensor_races.csv").relative_to(ROOT)),
            "verdicts_csv": str((output / "dual_sensor_arm_verdicts.csv").relative_to(ROOT)),
            "traces_npz": str((output / "dual_sensor_traces.npz").relative_to(ROOT)),
        },
        "upstream_status": {
            "entry_exit": entry_summary["status"],
            "orbit": json.loads((ROOT / cfg["orbit_summary_path"]).read_text(encoding="utf-8"))["status"],
        },
        "config": str(config_path.relative_to(ROOT)),
    }
    (output / "persistence_feasibility_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_persistence_dual_sensor_feasibility.png / .pdf\n\n"
        "这张四面板图检验 persistence-gated additive M 是否在时间和幅度上具备退出杠杆。"
        "A 对比真实空间 SNN 历史和同参数 0D 成熟周期的 p 累积；B、C 分别显示两种输入下"
        "可用 additive current 能否在第 3–5 个周期追上 frozen fold 所需电流；D 汇总同一组参数"
        "是否同时通过两个 bracket。B 中橙色虚线是主门失败后追加的 latched-gate 机制诊断，"
        "只用于定位缺失的维持记忆，不能改判 primary。\n\n"
        "**关注点**：单独在 SNN replay 或 0D phase sensor 上成功都不算 lifecycle 成功。"
        "若成熟周期过早开门而空间 SNN 恰好在 3–5 周期退出，说明缺失的是 spatial recruitment"
        "到 homogeneous sensor 的映射，不应继续调 Amax 来掩盖。横轴是 A=0 baseline-period"
        " equivalents，不是 slow trajectory 的真实 Poincaré return count。\n",
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
        "stop_rule": summary["stop_rule"],
        "primary_common_candidates": summary["primary_common_candidates"],
        "figure": summary["artifacts"]["figure"],
    }, indent=2))


if __name__ == "__main__":
    main()
