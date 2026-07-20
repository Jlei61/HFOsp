#!/usr/bin/env python3
"""Run the locked Stage-0E Poincare shooting and Floquet audit."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0e")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _sha256,
    _write_csv,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer  # noqa: E402
from src.topic4_spatial_slowfast_stage0e import (  # noqa: E402
    STATE_NAMES,
    audit_parameter_point,
    floquet_row_report,
)


DEFAULT_CONFIG = ROOT / "config/topic4_spatial_slowfast_stage0e.yaml"
SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0e-poincare-floquet-design.md"
MODULE = ROOT / "src/topic4_spatial_slowfast_stage0e.py"
RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0e.py"
TESTS = ROOT / "tests/test_topic4_spatial_slowfast_stage0e.py"
STAGE0C_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c.py"
TRANSFER_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c_transfer.py"
LOCKED_POINTS = ((0.85, 15.0), (0.85, 16.0))
PRE_BATTERY_ARCHIVE = (
    ROOT
    / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0e_poincare_floquet_audit_pre_battery_incomplete_2026-07-20"
)


def _validate_config(cfg: Mapping[str, Any]) -> None:
    """Reject any scientific or resource drift from the locked Stage0E spec."""

    points = tuple((float(row["z"]), float(row["alpha_G"])) for row in cfg["points"])
    if points != LOCKED_POINTS:
        raise ValueError("Stage0E fixed points drifted")
    if cfg["model"] != {"w_ee_mult": 1.1, "ratio": 1.0}:
        raise ValueError("Stage0E model constants drifted")
    if cfg["seed"] != {
        "source": "stage0d_phase_source",
        "phase_id": "phase_050",
        "phase_index": 2,
    }:
        raise ValueError("Stage0E seed drifted")
    expected_section = {
        "state_index": 8,
        "state_name": "S_G",
        "level": 0.15,
        "direction": "upward",
        "min_return_ms": 300.0,
        "max_return_ms": 1200.0,
    }
    if cfg["section"] != expected_section:
        raise ValueError("Stage0E Poincare section drifted")
    if cfg["scout"] != {
        "dt_ms": 0.125,
        "duration_ms": 12000.0,
        "save_stride": 1,
        "minimum_returns": 12,
        "waveform_phase_bins": 256,
    }:
        raise ValueError("Stage0E scout contract drifted")
    if cfg["shooting"] != {
        "max_iterations": 20,
        "residual_tolerance": 1.0e-6,
        "period_cv_tolerance": 1.0e-3,
        "aligned_cycle_residual_tolerance": 2.0e-3,
        "scale_floor": 1.0e-3,
        "final_monotone_count": 3,
    }:
        raise ValueError("Stage0E shooting contract drifted")
    if cfg["floquet"] != {
        "epsilon_relative": [1.0e-3, 3.0e-4, 1.0e-4],
        "epsilon_rho_range_max": 0.03,
        "jacobian_relative_difference_max": 0.10,
        "gradient_ratio_max": 1.25,
        "gradient_additive_tolerance": 1.0e-3,
        "dt_rho_difference_max": 0.05,
        "minimum_unit_circle_margin": 0.05,
        "uncertainty_multiplier": 3.0,
    }:
        raise ValueError("Stage0E Floquet contract drifted")
    if cfg["dt_half"] != {
        "dt_ms": 0.0625,
        "period_abs_ms": 1.0,
        "period_relative": 0.005,
        "aligned_waveform_residual": 0.03,
    }:
        raise ValueError("Stage0E dt/2 contract drifted")
    expected_battery = {
        "phases": [0.0, 0.25, 0.5, 0.75],
        "perturbation_fraction": 0.03,
        "n_returns": 8,
        "anchor_final_distance_max": 5.0e-4,
        "family_median_ratio_max": 0.70,
        "family_max_final_distance": 0.05,
        "log_slope_max": 0.0,
        "fast_directions": [[1, -1, 1, -1, -1, 1], [1, 1, -1, -1, 1, -1]],
        "pool_directions": [[1, -1, 0], [-1, 0, 1]],
    }
    if cfg["return_battery"] != expected_battery:
        raise ValueError("Stage0E return battery drifted")
    if cfg["physical_acceptance"] != {
        "finite_high_max_hz": 100.0,
        "report_high_threshold_hz": 80.0,
        "max_refractory_occupancy": 0.05,
    }:
        raise ValueError("Stage0E physical acceptance drifted")
    if cfg["resource_contract"] != {"blas_threads": 1, "max_memory_gib": 4.0}:
        raise ValueError("Stage0E resource contract drifted")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("Stage0E scope expansion is forbidden")
    if Path(str(cfg["result_root"])).name != "stage0e_poincare_floquet_audit":
        raise ValueError("Stage0E result root drifted")


def _source_paths(config_path: Path) -> dict[str, Path]:
    return {
        "spec": SPEC,
        "config": config_path,
        "module": MODULE,
        "runner": RUNNER,
        "tests": TESTS,
    }


def _source_hashes(config_path: Path) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in _source_paths(config_path).items()
    }


def _locked_input_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    phase_root = ROOT / str(cfg["phase_source_root"])
    transfer_root = ROOT / str(cfg["transfer_root"])
    return {
        "phase_source_traces": phase_root / "phase_source_traces.npz",
        "phase_source_json": phase_root / "phase_source.json",
        "extra_fine_transfer": transfer_root / "extended_transfer_extra_fine.npz",
        "stage0c_module": STAGE0C_MODULE,
        "transfer_module": TRANSFER_MODULE,
    }


def _verify_locked_inputs(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, path in _locked_input_paths(cfg).items():
        observed = _sha256(path)
        expected = str(cfg["locked_hashes"][name])
        rows[name] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "pass": observed == expected,
        }
    if not all(bool(row["pass"]) for row in rows.values()):
        raise RuntimeError("locked Stage0E upstream provenance mismatch")
    return rows


def _write_execution_lock(
    output: Path,
    config_path: Path,
    locked_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    lock = {
        "schema_version": "topic4_stage0e_execution_lock.v1",
        "locked_before_numerical_execution": True,
        "upstream_inputs_pre_execution": dict(locked_inputs),
        "stage0e_sources_pre_execution": _source_hashes(config_path),
        "upstream_inputs_post_execution": None,
        "stage0e_sources_post_execution": None,
        "all_inputs_and_sources_unchanged_during_execution": None,
    }
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return lock


def _finalize_execution_lock(
    output: Path,
    config_path: Path,
    cfg: Mapping[str, Any],
    lock: dict[str, Any],
) -> bool:
    post_inputs = _verify_locked_inputs(cfg)
    post_sources = _source_hashes(config_path)
    unchanged = bool(
        post_inputs == lock["upstream_inputs_pre_execution"]
        and post_sources == lock["stage0e_sources_pre_execution"]
    )
    lock["upstream_inputs_post_execution"] = post_inputs
    lock["stage0e_sources_post_execution"] = post_sources
    lock["all_inputs_and_sources_unchanged_during_execution"] = unchanged
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return unchanged


def _load_transfer(path: Path) -> ExtendedSiegertTransfer:
    with np.load(path, allow_pickle=False) as payload:
        if "no_clip" not in payload.files or not bool(payload["no_clip"]):
            raise RuntimeError("Stage0E transfer does not assert no clipping")
        return ExtendedSiegertTransfer(
            payload["mu_axis"],
            payload["sigma_axis"],
            payload["log_integral_table"],
            name="extra_fine",
        )


def _load_seed(cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    phase_root = ROOT / str(cfg["phase_source_root"])
    with np.load(phase_root / "phase_source_traces.npz", allow_pickle=False) as payload:
        phase_states = np.asarray(payload["phase_states"], dtype=float)
        source_state = np.asarray(payload["state"], dtype=float)
    metadata = json.loads((phase_root / "phase_source.json").read_text(encoding="utf-8"))
    phase_index = int(cfg["seed"]["phase_index"])
    rows = list(metadata.get("phase_selection", []))
    if phase_states.shape != (4, 9) or len(rows) != 4:
        raise RuntimeError("Stage0D phase-source shape drifted")
    row = rows[phase_index]
    if str(row.get("phase_id")) != str(cfg["seed"]["phase_id"]):
        raise RuntimeError("Stage0E phase seed identity mismatch")
    state_index = int(row["state_index"])
    seed = phase_states[phase_index]
    parity = float(np.max(np.abs(seed - source_state[state_index])))
    if parity > 1e-7 or not np.all(np.isfinite(seed)):
        raise RuntimeError("Stage0E phase seed failed source-trace parity")
    return seed, {
        "phase_id": str(row["phase_id"]),
        "phase_index": phase_index,
        "source_state_index": state_index,
        "source_time_ms": float(row["time_ms"]),
        "phase_state_vs_source_max_abs": parity,
        "seed_state": seed,
    }


def _point_slug(row: Mapping[str, Any]) -> str:
    alpha = int(round(float(row["alpha_G"])))
    return f"z_0p85_alpha_{alpha:02d}"


def _save_trace(path: Path, trace: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        time_ms=np.asarray(trace["time_ms"], dtype=float),
        state=np.asarray(trace["state"], dtype=np.float32),
        moments=np.asarray(trace["moments"], dtype=np.float32),
        crossing_time_ms=np.asarray(trace["crossing_time_ms"], dtype=float),
        crossing_state=np.asarray(trace["crossing_state"], dtype=np.float32),
        crossing_transversality_per_ms=np.asarray(
            trace["crossing_transversality_per_ms"], dtype=float
        ),
        period_ms=np.asarray(trace["period_ms"], dtype=float),
    )


def _floquet_npz_payload(artifacts: Mapping[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for label in ("base", "half"):
        for index, row in enumerate(artifacts.get(f"{label}_floquet", [])):
            prefix = f"{label}_epsilon_{index}"
            payload[f"{prefix}_epsilon_relative"] = np.asarray(
                [row.get("epsilon_relative", np.nan)], dtype=float
            )
            if bool(row.get("valid", False)):
                multipliers = np.asarray(row["multipliers"], dtype=complex)
                payload[f"{prefix}_jacobian"] = np.asarray(row["jacobian"], dtype=float)
                payload[f"{prefix}_multiplier_real"] = multipliers.real
                payload[f"{prefix}_multiplier_imag"] = multipliers.imag
                payload[f"{prefix}_multiplier_modulus"] = np.abs(multipliers)
    return payload


def _save_point_artifacts(
    output: Path,
    result: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    point_dir = output / "per_point" / _point_slug(result)
    point_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(point_dir / "point_outcome.json", result)
    crossing_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    for trace_name in ("scout",):
        if trace_name not in artifacts:
            continue
        trace = artifacts[trace_name]
        _save_trace(point_dir / f"{trace_name}_trace.npz", trace)
        for index, crossing_time in enumerate(np.asarray(trace["crossing_time_ms"], dtype=float)):
            crossing_audit = trace["crossing_audit"][index]
            crossing_rows.append(
                {
                    "z": float(result["z"]),
                    "alpha_G": float(result["alpha_G"]),
                    "trace": trace_name,
                    "crossing_index": index,
                    "crossing_time_ms": float(crossing_time),
                    "period_ms": float(trace["period_ms"][index - 1]) if index > 0 else None,
                    "transversality_per_ms": float(
                        trace["crossing_transversality_per_ms"][index]
                    ),
                    "crossing_clean": crossing_audit["clean"],
                    "finite_full_rhs": crossing_audit["finite_full_rhs"],
                    "transfer_support": crossing_audit["transfer_support"],
                    "natural_bounds": crossing_audit["natural_bounds"],
                    "below_100hz": crossing_audit["below_100hz"],
                    **{
                        f"section_{state_name}": float(trace["crossing_state"][index, state_index])
                        for state_index, state_name in enumerate(STATE_NAMES)
                    },
                }
            )
    for step_label, cycle_key in (("base", "base_cycle"), ("half", "half_cycle")):
        if cycle_key not in artifacts:
            continue
        cycle = artifacts[cycle_key]
        _save_trace(point_dir / f"{step_label}_cycle_trace.npz", cycle["trace"])
        cycle_rows.append(
            {
                "z": float(result["z"]),
                "alpha_G": float(result["alpha_G"]),
                "time_step": step_label,
                "period_1_ms": float(cycle["period_ms"][0]),
                "period_2_ms": float(cycle["period_ms"][1]),
                "closure_residual": float(cycle["closure_residual"]),
                "second_closure_residual": float(cycle["second_closure_residual"]),
                "aligned_cycle_residual": float(cycle["aligned_cycle_residual"]),
                "all_crossing_audits_clean": all(
                    row["clean"] for row in cycle["trace"]["crossing_audit"]
                ),
            }
        )
    if "base_shooting" in artifacts:
        np.savez_compressed(
            point_dir / "shooting_iterates.npz",
            base_state=np.asarray(artifacts["base_shooting"]["iterate_state"], dtype=float),
            base_residual=np.asarray(artifacts["base_shooting"]["residual"], dtype=float),
            base_period_ms=np.asarray(artifacts["base_shooting"]["period_ms"], dtype=float),
            half_state=np.asarray(
                artifacts.get("half_shooting", {}).get("iterate_state", np.empty((0, 9))),
                dtype=float,
            ),
            half_residual=np.asarray(
                artifacts.get("half_shooting", {}).get("residual", []), dtype=float
            ),
            half_period_ms=np.asarray(
                artifacts.get("half_shooting", {}).get("period_ms", []), dtype=float
            ),
        )
    floquet_payload = _floquet_npz_payload(artifacts)
    if floquet_payload:
        np.savez_compressed(point_dir / "poincare_jacobians_and_multipliers.npz", **floquet_payload)
        floquet_report = {
            "z": float(result["z"]),
            "alpha_G": float(result["alpha_G"]),
            "base_dt": [floquet_row_report(row) for row in artifacts["base_floquet"]],
            "half_dt": [floquet_row_report(row) for row in artifacts["half_floquet"]],
            "summary": result.get("floquet"),
        }
        _atomic_json(point_dir / "poincare_floquet_report.json", floquet_report)
    if "return_battery_rows" in artifacts:
        _atomic_json(point_dir / "perturbation_returns.json", artifacts["return_battery_rows"])
        batch = artifacts["return_battery"]
        for history_index, row in enumerate(artifacts["return_battery_rows"]):
            for return_index, distance in enumerate(row["return_distance"]):
                return_rows.append(
                    {
                        "z": float(result["z"]),
                        "alpha_G": float(result["alpha_G"]),
                        "phase_id": row["phase_id"],
                        "phase_fraction": row["phase_fraction"],
                        "history": row["history"],
                        "family": row["family"],
                        "history_valid": row["valid"],
                        "return_index": return_index + 1,
                        "return_time_ms": float(batch["return_time_ms"][return_index, history_index]),
                        "return_distance": float(distance),
                        "transversality_per_ms": float(
                            batch["transversality_per_ms"][return_index, history_index]
                        ),
                    }
                )
        _write_csv(point_dir / "perturbation_returns.csv", return_rows)
    return crossing_rows, cycle_rows, return_rows


def _point_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    scout = row.get("scout", {})
    base = row.get("base_dt_shooting", {})
    half = row.get("half_dt_shooting", {})
    floquet = row.get("floquet", {}).get("stability", {})
    battery = row.get("return_battery", {})
    physical = row.get("base_orbit_physical", {})
    return {
        "z": row["z"],
        "alpha_G": row["alpha_G"],
        "outcome": row["outcome"],
        "stable_periodic_orbit": row["stable_periodic_orbit"],
        "failed_gates": ";".join(row.get("failed_gates", [])),
        "scout_crossing_count": scout.get("crossing_count"),
        "scout_period_mean_ms": scout.get("period_mean_ms"),
        "scout_period_cv": scout.get("period_cv"),
        "minimum_transversality_per_ms": scout.get("minimum_transversality_per_ms"),
        "base_final_residual": (base.get("residual_series") or [None])[-1]
        if isinstance(base.get("residual_series"), list)
        else (
            float(np.asarray(base.get("residual_series"))[-1])
            if np.asarray(base.get("residual_series", [])).size
            else None
        ),
        "base_last_four_period_cv": base.get("last_four_period_cv"),
        "base_aligned_cycle_residual": base.get("aligned_cycle_residual"),
        "half_final_residual": float(np.asarray(half.get("residual_series"))[-1])
        if np.asarray(half.get("residual_series", [])).size
        else None,
        "dt_period_difference_ms": row.get("dt_cycle_consistency", {}).get(
            "period_difference_ms"
        ),
        "dt_aligned_waveform_residual": row.get("dt_cycle_consistency", {}).get(
            "aligned_waveform_residual"
        ),
        "rho_max": floquet.get("rho_max"),
        "unit_circle_margin": floquet.get("unit_circle_margin"),
        "required_margin": floquet.get("required_margin"),
        "return_battery_pass": battery.get("pass"),
        "fast_family_pass": battery.get("families", {}).get("fast", {}).get("pass"),
        "pool_family_pass": battery.get("families", {}).get("pool", {}).get("pass"),
        "period_ms": physical.get("period_ms"),
        "peak_rE_hz": physical.get("peak_rE_hz"),
        "above_80hz_occupancy": physical.get("above_80hz_occupancy"),
        "above_100hz_occupancy": physical.get("above_100hz_occupancy"),
    }


def _overall_verdict(point_rows: list[Mapping[str, Any]], engineering_pass: bool) -> str:
    if not engineering_pass:
        return "STAGE0E_ENGINEERING_OR_PROVENANCE_FAIL"
    stable = {
        float(row["alpha_G"])
        for row in point_rows
        if str(row["outcome"]) == "stable_periodic_orbit"
    }
    if stable == {15.0, 16.0}:
        return "STAGE0E_STABLE_PERIODIC_ORBITS_ALPHA15_AND_ALPHA16"
    if stable == {15.0}:
        return "STAGE0E_STABLE_PERIODIC_ORBIT_ALPHA15_ONLY"
    if stable == {16.0}:
        return "STAGE0E_STABLE_PERIODIC_ORBIT_ALPHA16_ONLY"
    unresolved = {"periodic_orbit_numerically_unresolved", "numerical_failure"}
    if any(str(row["outcome"]) in unresolved for row in point_rows):
        return "STAGE0E_NUMERICAL_UNRESOLVED"
    return "STAGE0E_NO_STABLE_PERIODIC_ORBIT_AT_LOCKED_POINTS"


def _plot(output: Path, point_rows: list[Mapping[str, Any]], all_artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    colors = {15.0: "#2a9d8f", 16.0: "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    for row, artifacts in zip(point_rows, all_artifacts):
        trace = artifacts["scout"]
        time = np.asarray(trace["time_ms"], dtype=float)
        mask = time >= max(0.0, time[-1] - 3000.0)
        alpha = float(row["alpha_G"])
        ax.plot(
            time[mask] / 1000.0,
            1000.0 * np.asarray(trace["state"])[mask, 0],
            lw=0.9,
            color=colors[alpha],
            label=rf"$\alpha_G={alpha:.0f}$",
        )
    ax.axhline(80.0, color="#777777", lw=0.7, ls="--")
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="A  High-resolution scout")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for row, artifacts in zip(point_rows, all_artifacts):
        alpha = float(row["alpha_G"])
        shooting = artifacts.get("base_shooting")
        if shooting is not None and len(shooting["residual"]):
            ax.semilogy(
                np.arange(1, len(shooting["residual"]) + 1),
                shooting["residual"],
                marker="o",
                ms=3,
                color=colors[alpha],
                label=rf"$\alpha_G={alpha:.0f}$",
            )
    ax.axhline(1e-6, color="#555555", lw=0.8, ls="--")
    ax.set(xlabel="Poincare iteration", ylabel="scaled section residual", title="B  Shooting")
    if ax.lines:
        ax.legend(frameon=False)

    ax = axes[1, 0]
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="#777777", lw=0.8, ls="--")
    multiplier_count = 0
    for row, artifacts in zip(point_rows, all_artifacts):
        alpha = float(row["alpha_G"])
        for time_step, marker in (("base_floquet", "o"), ("half_floquet", "x")):
            for floquet_row in artifacts.get(time_step, []):
                if not bool(floquet_row.get("valid", False)):
                    continue
                values = np.asarray(floquet_row["multipliers"], dtype=complex)
                ax.scatter(values.real, values.imag, s=19, marker=marker, color=colors[alpha], alpha=0.65)
                multiplier_count += values.size
    ax.axhline(0.0, color="#bbbbbb", lw=0.5)
    ax.axvline(0.0, color="#bbbbbb", lw=0.5)
    ax.set(aspect="equal", xlabel="Re(lambda)", ylabel="Im(lambda)", title="C  All 8 transverse multipliers")
    if multiplier_count == 0:
        ax.text(0.5, 0.5, "Floquet gate not reached", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    return_count = 0
    for row, artifacts in zip(point_rows, all_artifacts):
        alpha = float(row["alpha_G"])
        for battery_row in artifacts.get("return_battery_rows", []):
            if battery_row["family"] == "anchor":
                continue
            ax.semilogy(
                np.arange(1, len(battery_row["return_distance"]) + 1),
                battery_row["return_distance"],
                color=colors[alpha],
                alpha=0.35,
                lw=0.8,
            )
            return_count += 1
    ax.axhline(0.05, color="#777777", lw=0.7, ls="--")
    ax.set(xlabel="return index", ylabel="distance to fixed section state", title="D  Fast/pool return battery")
    if return_count == 0:
        ax.text(0.5, 0.5, "Return battery not reached", ha="center", va="center", transform=ax.transAxes)

    png = figures / "stage0e_poincare_floquet_audit.png"
    pdf = figures / "stage0e_poincare_floquet_audit.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    readme = """### stage0e_poincare_floquet_audit.png

这张诊断图按同一锁定尺度展示两个固定参数点的高分辨率轨迹、Poincaré shooting 残差、全部八个横向 Floquet 乘子，以及 fast/pool 非共线扰动的逐圈回归距离。若某一后续 panel 留空，表示该参数点在 cheap-first 的更早数值门已经停止，不能用频谱结果补救。

**关注点**：只有 shooting 闭合、三档 epsilon 与 dt/2 均稳健、全部横向乘子远离单位圆边界，并且两类扰动都逐圈回归时，才能称为稳定周期轨道。

### stage0e_poincare_floquet_audit.pdf

与 PNG 内容相同的矢量版本，供文档审阅和局部放大检查复乘子位置。

**关注点**：单位圆内并不自动等于通过；还必须满足预注册的数值不确定性裕量。
"""
    _atomic_text(figures / "README.md", readme)
    metadata = {
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "panels": {
            "A": "last 3 s of every-Euler-state scouts",
            "B": "scaled Poincare fixed-point residuals",
            "C": "all eight transverse multipliers for every valid epsilon and dt",
            "D": "fast and pool perturbation return distances",
        },
        "multiplier_count_rendered": multiplier_count,
        "perturbation_histories_rendered": return_count,
    }
    _atomic_json(output / "figure_metadata.json", metadata)
    return metadata


def _status_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Topic 4 Stage 0E 状态",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Engineering/provenance pass: `{str(summary['engineering_pass']).lower()}`",
        f"- Peak RSS: `{summary['resource_usage']['peak_rss_gib']:.3f} GiB`",
        f"- Wall time: `{summary['resource_usage']['wall_seconds']:.2f} s`",
        "- Stage 1: `CLOSED`",
        "- Spatial simulation: `CLOSED`",
        "",
        "## 固定点结果",
        "",
    ]
    for row in summary["parameter_points"]:
        lines.append(
            f"- `z={row['z']:.2f}, alpha_G={row['alpha_G']:.0f}`: "
            f"`{row['outcome']}`; failed gates = `{','.join(row.get('failed_gates', [])) or 'none'}`"
        )
    lines.extend(
        [
            "",
            "本阶段只审计不变的九维 frozen fast system。无论结果为何，都不自动开放 slow lifecycle、空间耦合或 Stage 1。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("Refusing authoritative Stage0E execution without --confirm-run")
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    output = ROOT / str(cfg["result_root"])
    if output.exists():
        raise RuntimeError(f"Stage0E output already exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    locked_inputs = _verify_locked_inputs(cfg)
    lock = _write_execution_lock(output, config_path, locked_inputs)
    seed_state, seed_provenance = _load_seed(cfg)
    transfer = _load_transfer(_locked_input_paths(cfg)["extra_fine_transfer"])

    point_rows: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    for point in cfg["points"]:
        params = PoolParameters(
            float(point["z"]),
            float(point["alpha_G"]),
            float(cfg["model"]["w_ee_mult"]),
            float(cfg["model"]["ratio"]),
        )
        result, artifacts = audit_parameter_point(seed_state, params, transfer, cfg)
        point_rows.append(result)
        all_artifacts.append(artifacts)
        point_crossings, point_cycles, point_returns = _save_point_artifacts(
            output, result, artifacts
        )
        crossing_rows.extend(point_crossings)
        cycle_rows.extend(point_cycles)
        perturbation_rows.extend(point_returns)

    _atomic_json(output / "parameter_point_outcomes.json", point_rows)
    _write_csv(output / "parameter_point_outcomes.csv", [_point_csv_row(row) for row in point_rows])
    _write_csv(output / "poincare_crossings.csv", crossing_rows)
    _write_csv(output / "shooting_cycle_summary.csv", cycle_rows)
    _write_csv(output / "perturbation_returns.csv", perturbation_rows)
    figure_metadata = _plot(output, point_rows, all_artifacts)
    unchanged = _finalize_execution_lock(output, config_path, cfg, lock)
    wall_seconds = float(time.monotonic() - started)
    peak_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2)
    resource_pass = bool(peak_rss_gib < float(cfg["resource_contract"]["max_memory_gib"]))
    engineering_pass = bool(unchanged and resource_pass)
    summary = {
        "schema_version": "topic4_stage0e_poincare_floquet_audit.v1",
        "verdict": _overall_verdict(point_rows, engineering_pass),
        "engineering_pass": engineering_pass,
        "sources_and_inputs_unchanged": unchanged,
        "seed_provenance": seed_provenance,
        "superseded_incomplete_run": {
            "path": str(PRE_BATTERY_ARCHIVE.resolve()),
            "exists": PRE_BATTERY_ARCHIVE.is_dir(),
            "reason": "Floquet failure incorrectly short-circuited the sibling return-battery and physical-report audits",
            "authoritative": False,
            "execution_lock_sha256": _sha256(PRE_BATTERY_ARCHIVE / "EXECUTION_LOCK.json")
            if (PRE_BATTERY_ARCHIVE / "EXECUTION_LOCK.json").is_file()
            else None,
            "summary_sha256": _sha256(
                PRE_BATTERY_ARCHIVE / "stage0e_poincare_floquet_summary.json"
            )
            if (PRE_BATTERY_ARCHIVE / "stage0e_poincare_floquet_summary.json").is_file()
            else None,
        },
        "parameter_points": point_rows,
        "resource_usage": {
            "blas_threads": 1,
            "peak_rss_gib": peak_rss_gib,
            "rss_limit_gib": float(cfg["resource_contract"]["max_memory_gib"]),
            "rss_pass": resource_pass,
            "wall_seconds": wall_seconds,
        },
        "figure_metadata": figure_metadata,
        "scope": {
            "stage1_open": False,
            "spatial_simulation_open": False,
            "parameter_search_performed": False,
            "fft_substitute_used": False,
        },
    }
    _atomic_json(output / "stage0e_poincare_floquet_summary.json", summary)
    _atomic_text(output / "STATUS.md", _status_text(summary))
    print(json.dumps({"verdict": summary["verdict"], "output": str(output), "peak_rss_gib": peak_rss_gib}, indent=2))


if __name__ == "__main__":
    main()
