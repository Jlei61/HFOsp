#!/usr/bin/env python3
"""Run the locked Stage-0F v1.1 variational-certificate repair."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0f-v1-1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import (  # noqa: E402
    DERIVATIVE_LABELS_V11,
    SmoothSiegertTransferV11,
    run_point_certificate_v1_1,
)


DEFAULT_CONFIG = ROOT / "config/topic4_spatial_slowfast_stage0f_v1_1.yaml"
SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0f-v1_1-engineering-repair-design.md"
MODULE = ROOT / "src/topic4_spatial_slowfast_stage0f_v1_1.py"
RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0f_v1_1.py"
TESTS = ROOT / "tests/test_topic4_spatial_slowfast_stage0f_v1_1.py"
STAGE0F_V1_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0f.py"
STAGE0E_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0e.py"
STAGE0C_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c.py"
TRANSFER_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c_transfer.py"
SEF_HFO_LIF_MODULE = ROOT / "src/sef_hfo_lif.py"
ATOMIC_HELPER_RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0c_transfer.py"
LOCKED_POINTS = ((0.85, 15.0), (0.85, 16.0))


def _validate_config(cfg: Mapping[str, Any]) -> None:
    if tuple((float(row["z"]), float(row["alpha_G"])) for row in cfg["points"]) != LOCKED_POINTS:
        raise ValueError("Stage0F v1.1 fixed points drifted")
    if cfg["model"] != {"w_ee_mult": 1.1, "ratio": 1.0}:
        raise ValueError("Stage0F v1.1 model constants drifted")
    if cfg["section"] != {
        "state_index": 8,
        "state_name": "S_G",
        "level": 0.15,
        "direction": "upward",
        "min_return_ms": 300.0,
        "max_return_ms": 1200.0,
    }:
        raise ValueError("Stage0F v1.1 section drifted")
    if cfg["time_steps_ms"] != [0.125, 0.0625] or int(cfg["phase_bins"]) != 256:
        raise ValueError("Stage0F v1.1 integration schedule drifted")
    if int(cfg["transfer_parity_variational_samples"]) != 512:
        raise ValueError("Stage0F v1.1 transfer parity schedule drifted")
    if cfg["smooth_transfer"] != {
        "mu_min_mv": -160.0,
        "mu_max_mv": 80.0,
        "sigma_min_mv": 3.0,
        "sigma_max_mv": 20.0,
        "spline_degree_mu": 3,
        "spline_degree_sigma": 3,
        "smoothing": 0.0,
    }:
        raise ValueError("Stage0F v1.1 smooth transfer drifted")
    if cfg["transfer_parity"] != {
        "rate_absolute_khz": 5.0e-5,
        "rate_relative": 5.0e-3,
        "rate_relative_floor_khz": 1.0e-4,
        "derivative_absolute_khz_per_mv": 5.0e-5,
        "derivative_relative": 5.0e-2,
        "derivative_relative_floor_khz_per_mv": 1.0e-7,
    }:
        raise ValueError("Stage0F v1.1 transfer parity drifted")
    if cfg["shooting"] != {
        "max_iterations": 20,
        "residual_tolerance": 1.0e-8,
        "period_cv_tolerance": 1.0e-3,
        "aligned_cycle_residual_tolerance": 2.0e-4,
        "minimum_iterations": 4,
        "event_restarted_p_closure_tolerance": 1.0e-8,
        "event_restarted_p2_closure_tolerance": 1.0e-8,
    }:
        raise ValueError("Stage0F v1.1 shooting contract drifted")
    if cfg["orbit_parity"] != {
        "period_abs_ms": 1.0,
        "aligned_waveform_residual": 3.0e-2,
        "dt_period_abs_ms": 1.0,
        "dt_period_relative": 5.0e-3,
        "dt_aligned_waveform_residual": 3.0e-2,
    }:
        raise ValueError("Stage0F v1.1 orbit parity drifted")
    if cfg["variational"] != {
        "finite_difference_relative_steps": [1.0e-5, 3.0e-6],
        "finite_difference_absolute_floor": 1.0e-9,
        "matrix_relative_difference_max": 5.0e-2,
        "matrix_norm_floor": 1.0e-8,
        "spectral_radius_range_max": 2.0e-2,
        "section_row_abs_max": 1.0e-10,
        "minimum_continuous_transversality_per_ms": 1.0e-4,
        "minimum_discrete_transversality_per_ms": 1.0e-4,
        "nominal_period_identity_abs_ms": 1.0e-9,
        "nominal_crossing_identity_scaled": 1.0e-10,
    }:
        raise ValueError("Stage0F v1.1 variational contract drifted")
    if cfg["whole_return_jv"] != {
        "epsilon_relative": [1.0e-3, 3.0e-4],
        "matrix_relative_difference_max": 5.0e-2,
        "spectral_radius_range_max": 2.0e-2,
    }:
        raise ValueError("Stage0F v1.1 whole-return Jv contract drifted")
    if cfg["stability"] != {
        "minimum_unit_circle_margin": 5.0e-2,
        "uncertainty_multiplier": 3.0,
    }:
        raise ValueError("Stage0F v1.1 stability margin drifted")
    if cfg["artifact_contract"] != {
        "complete_parity_rows": 4096,
        "complete_derivative_summary_rows": 4,
        "complete_cycle_traces": 4,
        "complete_certificate_matrices": 12,
        "complete_certificate_multipliers": 96,
        "complete_whole_return_jv_arrays": 8,
        "figure_panels": 4,
    }:
        raise ValueError("Stage0F v1.1 artifact contract drifted")
    if cfg["physical_acceptance"] != {
        "finite_high_max_hz": 100.0,
        "max_refractory_occupancy": 0.05,
    }:
        raise ValueError("Stage0F v1.1 physical contract drifted")
    if cfg["resource_contract"] != {"blas_threads": 1, "max_memory_gib": 4.0}:
        raise ValueError("Stage0F v1.1 resource contract drifted")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("Stage0F v1.1 scope expansion is forbidden")
    if Path(str(cfg["result_root"])).name != "stage0f_smooth_transfer_variational_certificate_v1_1":
        raise ValueError("Stage0F v1.1 result root drifted")


def _source_paths(config_path: Path) -> dict[str, Path]:
    return {"spec": SPEC, "config": config_path, "module": MODULE, "runner": RUNNER, "tests": TESTS}


def _source_hashes(config_path: Path) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in _source_paths(config_path).items()
    }


def _locked_input_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    stage0e = ROOT / str(cfg["stage0e_root"])
    v1 = ROOT / str(cfg["stage0f_v1_archive"])
    mapping = {
        "stage0e_summary": stage0e / "stage0e_poincare_floquet_summary.json",
        "extra_fine_transfer": ROOT / str(cfg["transfer_path"]),
        "stage0e_module": STAGE0E_MODULE,
        "stage0c_module": STAGE0C_MODULE,
        "transfer_module": TRANSFER_MODULE,
        "sef_hfo_lif_module": SEF_HFO_LIF_MODULE,
        "atomic_helper_runner": ATOMIC_HELPER_RUNNER,
        "stage0f_v1_module": STAGE0F_V1_MODULE,
        "stage0f_v1_execution_lock": v1 / "EXECUTION_LOCK.json",
        "stage0f_v1_summary": v1 / "stage0f_smooth_transfer_variational_summary.json",
        "stage0f_v1_alpha15_outcome": v1 / "per_point/z_0p85_alpha_15/point_outcome.json",
        "stage0f_v1_alpha16_outcome": v1 / "per_point/z_0p85_alpha_16/point_outcome.json",
    }
    for alpha in (15, 16):
        point = stage0e / "per_point" / f"z_0p85_alpha_{alpha}"
        prefix = f"alpha{alpha}"
        mapping.update(
            {
                f"{prefix}_shooting": point / "shooting_iterates.npz",
                f"{prefix}_base_trace": point / "base_cycle_trace.npz",
                f"{prefix}_half_trace": point / "half_cycle_trace.npz",
                f"{prefix}_outcome": point / "point_outcome.json",
            }
        )
    return mapping


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
    if not all(row["pass"] for row in rows.values()):
        raise RuntimeError("locked Stage0F v1.1 upstream provenance mismatch")
    return rows


def _write_execution_lock(output: Path, config_path: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    environment = _environment_versions()
    lock = {
        "schema_version": "topic4_stage0f_v1_1_execution_lock.v1",
        "locked_before_numerical_execution": True,
        "upstream_inputs_pre_execution": dict(inputs),
        "stage0f_v1_1_sources_pre_execution": _source_hashes(config_path),
        "numerical_environment_pre_execution": environment,
        "upstream_inputs_post_execution": None,
        "stage0f_v1_1_sources_post_execution": None,
        "numerical_environment_post_execution": None,
        "all_inputs_and_sources_unchanged_during_execution": None,
    }
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return lock


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _finalize_execution_lock(
    output: Path, config_path: Path, cfg: Mapping[str, Any], lock: dict[str, Any]
) -> bool:
    post_inputs = _verify_locked_inputs(cfg)
    post_sources = _source_hashes(config_path)
    post_environment = _environment_versions()
    unchanged = bool(
        post_inputs == lock["upstream_inputs_pre_execution"]
        and post_sources == lock["stage0f_v1_1_sources_pre_execution"]
        and post_environment == lock["numerical_environment_pre_execution"]
    )
    lock["upstream_inputs_post_execution"] = post_inputs
    lock["stage0f_v1_1_sources_post_execution"] = post_sources
    lock["numerical_environment_post_execution"] = post_environment
    lock["all_inputs_and_sources_unchanged_during_execution"] = unchanged
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return unchanged


def _load_transfer(cfg: Mapping[str, Any]) -> SmoothSiegertTransferV11:
    with np.load(ROOT / str(cfg["transfer_path"]), allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("Stage0F v1.1 source transfer did not assert no clipping")
        original = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"], name="extra_fine"
        )
    smooth = cfg["smooth_transfer"]
    return SmoothSiegertTransferV11.from_extended(
        original,
        domain=SmoothDomain(
            float(smooth["mu_min_mv"]),
            float(smooth["mu_max_mv"]),
            float(smooth["sigma_min_mv"]),
            float(smooth["sigma_max_mv"]),
        ),
        kx=int(smooth["spline_degree_mu"]),
        ky=int(smooth["spline_degree_sigma"]),
        smoothing=float(smooth["smoothing"]),
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_stage0e_point(cfg: Mapping[str, Any], alpha: int) -> dict[str, Any]:
    point = ROOT / str(cfg["stage0e_root"]) / "per_point" / f"z_0p85_alpha_{alpha}"
    outcome = json.loads((point / "point_outcome.json").read_text(encoding="utf-8"))
    shooting = _load_npz(point / "shooting_iterates.npz")
    if outcome.get("outcome") != "periodic_orbit_numerically_unresolved" or outcome.get("failed_gates") != ["floquet_epsilon_dt_or_margin"]:
        raise RuntimeError("Stage0E failure boundary drifted before Stage0F v1.1")
    return {
        "scales": np.asarray(outcome["state_scales"], dtype=float),
        "base_shooting_seed": np.asarray(shooting["base_state"][-1], dtype=float),
        "half_shooting_seed": np.asarray(shooting["half_state"][-1], dtype=float),
        "base_lut_trace": _load_npz(point / "base_cycle_trace.npz"),
        "half_lut_trace": _load_npz(point / "half_cycle_trace.npz"),
    }


def _preflight(
    config_path: Path, cfg: Mapping[str, Any], output: Path
) -> tuple[
    dict[str, dict[str, Any]],
    SmoothSiegertTransferV11,
    dict[int, dict[str, Any]],
]:
    """Resolve and verify every dependency without creating the canonical root."""

    if output.exists():
        raise RuntimeError(f"Stage0F v1.1 output already exists; refusing overwrite: {output}")
    inputs = _verify_locked_inputs(cfg)
    transfer = _load_transfer(cfg)
    stage0e_by_alpha = {
        alpha: _load_stage0e_point(cfg, alpha) for alpha in (15, 16)
    }
    # Force readability and hashing of all Stage0F v1.1 sources before output creation.
    _source_hashes(config_path)
    if output.exists():
        raise RuntimeError("Stage0F v1.1 output appeared during preflight")
    return inputs, transfer, stage0e_by_alpha


def _save_cycle(path: Path, cycle: Mapping[str, Any]) -> None:
    trace = cycle["trace"]
    np.savez_compressed(
        path,
        time_ms=np.asarray(trace["time_ms"], dtype=float),
        state=np.asarray(trace["state"], dtype=np.float32),
        moments=np.asarray(trace["moments"], dtype=np.float32),
        crossing_time_ms=np.asarray(trace["crossing_time_ms"], dtype=float),
        crossing_state=np.asarray(trace["crossing_state"], dtype=np.float32),
        period_ms=np.asarray(cycle["period_ms"], dtype=float),
        waveform_first=np.asarray(cycle["waveform_first"], dtype=np.float32),
        waveform_second=np.asarray(cycle["waveform_second"], dtype=np.float32),
    )


def _save_point(output: Path, result: Mapping[str, Any], artifacts: Mapping[str, Any]) -> dict[str, int]:
    alpha = int(round(float(result["alpha_G"])))
    point = output / "per_point" / f"z_0p85_alpha_{alpha}"
    point.mkdir(parents=True, exist_ok=False)
    _atomic_json(point / "point_outcome.json", result)
    counts = {"cycle_traces": 0, "certificate_matrices": 0, "certificate_multipliers": 0, "whole_return_jv_arrays": 0}
    for label, cycle in artifacts.get("cycles", {}).items():
        _save_cycle(point / f"smooth_{label}_cycle_trace.npz", cycle)
        counts["cycle_traces"] += 1
    parity_rows = list(artifacts.get("transfer_parity_rows", []))
    if parity_rows:
        _write_csv(point / "exact_transfer_parity.csv", parity_rows)
    if artifacts.get("variational"):
        matrix_payload: dict[str, np.ndarray] = {}
        variational_report: dict[str, Any] = {}
        for dt_label, item in artifacts["variational"].items():
            variational_report[dt_label] = {
                key: value
                for key, value in item.items()
                if key not in {
                    "poincare_matrices",
                    "full_event_tangents_normalized",
                    "multipliers",
                    "nominal_time_ms",
                    "nominal_state_trace",
                    "nominal_moment_trace",
                }
            }
            variational_report[dt_label]["multipliers"] = {}
            if not bool(item.get("valid", False)):
                continue
            required = {
                "nominal_time_ms",
                "nominal_state_trace",
                "nominal_moment_trace",
                "poincare_matrices",
                "full_event_tangents_normalized",
                "multipliers",
            }
            if not required.issubset(item):
                variational_report[dt_label]["artifact_serialization_error"] = (
                    "valid variational result lacks required arrays"
                )
                continue
            matrix_payload[f"{dt_label}_nominal_time_ms"] = np.asarray(item["nominal_time_ms"], dtype=float)
            matrix_payload[f"{dt_label}_nominal_state_trace"] = np.asarray(item["nominal_state_trace"], dtype=np.float32)
            matrix_payload[f"{dt_label}_nominal_moment_trace"] = np.asarray(item["nominal_moment_trace"], dtype=np.float32)
            for method in DERIVATIVE_LABELS_V11:
                prefix = f"{dt_label}_{method}"
                matrix_payload[f"{prefix}_poincare_matrix"] = np.asarray(item["poincare_matrices"][method], dtype=float)
                matrix_payload[f"{prefix}_full_event_tangent_normalized"] = np.asarray(item["full_event_tangents_normalized"][method], dtype=float)
                multipliers = np.asarray(item["multipliers"][method], dtype=complex)
                matrix_payload[f"{prefix}_multiplier_real"] = multipliers.real
                matrix_payload[f"{prefix}_multiplier_imag"] = multipliers.imag
                variational_report[dt_label]["multipliers"][method] = [
                    {"real": float(value.real), "imag": float(value.imag), "modulus": float(abs(value))}
                    for value in multipliers
                ]
                counts["certificate_matrices"] += 1
                counts["certificate_multipliers"] += int(multipliers.size)
        if matrix_payload:
            np.savez_compressed(point / "discrete_variational_certificate.npz", **matrix_payload)
        _atomic_json(point / "discrete_variational_report.json", variational_report)
    if artifacts.get("whole_return_jv"):
        jv_payload: dict[str, np.ndarray] = {}
        jv_report: dict[str, Any] = {}
        for dt_label, rows in artifacts["whole_return_jv"].items():
            jv_report[dt_label] = []
            for index, row in enumerate(rows):
                prefix = f"{dt_label}_epsilon_{index}"
                if row.get("valid", False):
                    jv_payload[f"{prefix}_jv_matrix"] = np.asarray(row["jv_matrix"], dtype=float)
                    counts["whole_return_jv_arrays"] += 1
                jv_report[dt_label].append(
                    {key: value for key, value in row.items() if key != "jv_matrix"}
                )
        np.savez_compressed(point / "whole_return_jv_audit.npz", **jv_payload)
        _atomic_json(point / "whole_return_jv_report.json", jv_report)
    partial_report = {
        "schema_version": "topic4_stage0f_v1_1_point_artifact_report.v1",
        "z": float(result["z"]),
        "alpha_G": float(result["alpha_G"]),
        "outcome": str(result["outcome"]),
        "failed_gates": list(result.get("failed_gates", [])),
        "cycles_present": sorted(artifacts.get("cycles", {}).keys()),
        "transfer_parity_rows_present": len(parity_rows),
        "variational_levels_present": sorted(artifacts.get("variational", {}).keys()),
        "variational_levels_valid": {
            str(label): bool(item.get("valid", False))
            for label, item in artifacts.get("variational", {}).items()
        },
        "whole_return_jv_levels_present": sorted(
            artifacts.get("whole_return_jv", {}).keys()
        ),
        "whole_return_jv_valid_rows": {
            str(label): int(sum(bool(row.get("valid", False)) for row in rows))
            for label, rows in artifacts.get("whole_return_jv", {}).items()
        },
        "serialized_counts": counts,
    }
    _atomic_json(point / "partial_artifact_report.json", partial_report)
    return counts


def _empty_saved_counts() -> dict[str, int]:
    return {
        "cycle_traces": 0,
        "certificate_matrices": 0,
        "certificate_multipliers": 0,
        "whole_return_jv_arrays": 0,
    }


def _engineering_point_result(
    point: Mapping[str, Any],
    *,
    outcome: str,
    failed_gate: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "z": float(point["z"]),
        "alpha_G": float(point["alpha_G"]),
        "outcome": outcome,
        "derivative_certified": False,
        "failed_gates": [failed_gate],
        "stage1_open": False,
        "space_open": False,
        "engineering_exception": error is not None,
    }
    if error is not None:
        result["execution_exception"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return result


def _save_failure_point_fallback(
    output: Path,
    result: Mapping[str, Any],
    *,
    serialization_error: BaseException,
) -> None:
    """Best-effort minimal point bundle after normal serialization failed."""

    alpha = int(round(float(result["alpha_G"])))
    point = output / "per_point" / f"z_0p85_alpha_{alpha}"
    point.mkdir(parents=True, exist_ok=True)
    _atomic_json(point / "point_outcome.json", result)
    _atomic_json(
        point / "partial_artifact_report.json",
        {
            "schema_version": "topic4_stage0f_v1_1_point_artifact_report.v1",
            "z": float(result["z"]),
            "alpha_G": float(result["alpha_G"]),
            "outcome": str(result["outcome"]),
            "failed_gates": list(result.get("failed_gates", [])),
            "artifact_serialization_exception": {
                "type": type(serialization_error).__name__,
                "message": str(serialization_error),
            },
            "serialized_counts": _empty_saved_counts(),
        },
    )


def _derivative_rows_for_point(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "pass",
        "matrix_relative_differences",
        "spectral_radii",
        "spectral_radius_range",
        "continuous_rhs_transversality_per_ms",
        "discrete_bracket_transversality_per_ms",
        "stencil_totals",
    }
    for dt_label in ("base", "half"):
        summary = result.get(f"{dt_label}_variational_consistency")
        if not isinstance(summary, Mapping) or not required.issubset(summary):
            continue
        rows.append(
            {
                "z": result["z"],
                "alpha_G": result["alpha_G"],
                "dt_label": dt_label,
                "pass": summary["pass"],
                **summary["matrix_relative_differences"],
                **{
                    f"rho_{name}": value
                    for name, value in summary["spectral_radii"].items()
                },
                "spectral_radius_range": summary["spectral_radius_range"],
                "continuous_transversality_per_ms": summary[
                    "continuous_rhs_transversality_per_ms"
                ],
                "discrete_transversality_per_ms": summary[
                    "discrete_bracket_transversality_per_ms"
                ],
                "stencil_totals": json.dumps(summary["stencil_totals"], sort_keys=True),
            }
        )
    return rows


def _captured_exception_records(
    payload: Any, *, path: str = "point"
) -> list[dict[str, str]]:
    """Find exceptions converted to data by lower numerical layers."""

    records: list[dict[str, str]] = []
    if isinstance(payload, Mapping):
        exception_type = payload.get("exception_type")
        exception_message = payload.get("exception_message")
        if exception_type is not None or exception_message is not None:
            records.append(
                {
                    "path": path,
                    "type": str(exception_type or "UnknownException"),
                    "message": str(exception_message or ""),
                }
            )
        for key, value in payload.items():
            records.extend(
                _captured_exception_records(value, path=f"{path}.{key}")
            )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            records.extend(
                _captured_exception_records(value, path=f"{path}[{index}]")
            )
    return records


def _run_locked_points_fail_closed(
    output: Path,
    cfg: Mapping[str, Any],
    transfer: SmoothSiegertTransferV11,
    stage0e_by_alpha: Mapping[int, Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, int]],
    list[dict[str, Any]],
]:
    """Execute both locked points; any unhandled exception blocks later points."""

    rows: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    saved_counts: list[dict[str, int]] = []
    exceptions: list[dict[str, Any]] = []
    blocked = False
    for point in cfg["points"]:
        alpha = int(round(float(point["alpha_G"])))
        artifacts: dict[str, Any] = {}
        if blocked:
            result = _engineering_point_result(
                point,
                outcome="not_executed_due_to_engineering_failure",
                failed_gate="prior_point_engineering_failure",
            )
        else:
            try:
                params = PoolParameters(
                    float(point["z"]),
                    float(point["alpha_G"]),
                    float(cfg["model"]["w_ee_mult"]),
                    float(cfg["model"]["ratio"]),
                )
                result, artifacts = run_point_certificate_v1_1(
                    params, transfer, stage0e_by_alpha[alpha], cfg
                )
                captured = _captured_exception_records(
                    {"result": result, "artifacts": artifacts}
                )
                if captured:
                    prior_outcome = str(result.get("outcome", "unknown"))
                    result = dict(result)
                    result["outcome"] = "numerical_failure"
                    result["derivative_certified"] = False
                    result["engineering_exception"] = True
                    result["outcome_before_captured_exception"] = prior_outcome
                    result.setdefault("failed_gates", []).append(
                        "captured_numerical_exception"
                    )
                    result["captured_exception_records"] = captured
                    exceptions.extend(
                        {
                            "phase": "captured_point_numerical_exception",
                            "alpha_G": float(point["alpha_G"]),
                            **record,
                        }
                        for record in captured
                    )
                    blocked = True
            except Exception as error:  # fail closed at the point boundary
                result = _engineering_point_result(
                    point,
                    outcome="numerical_failure",
                    failed_gate="unhandled_point_exception",
                    error=error,
                )
                exceptions.append(
                    {
                        "phase": "point_numerical_execution",
                        "alpha_G": float(point["alpha_G"]),
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                blocked = True
        try:
            counts = _save_point(output, result, artifacts)
        except Exception as error:  # serialization is an engineering failure
            prior_outcome = str(result.get("outcome", "unknown"))
            result = _engineering_point_result(
                point,
                outcome="numerical_failure",
                failed_gate="point_artifact_serialization_exception",
                error=error,
            )
            result["outcome_before_serialization_exception"] = prior_outcome
            exceptions.append(
                {
                    "phase": "point_artifact_serialization",
                    "alpha_G": float(point["alpha_G"]),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            blocked = True
            counts = _empty_saved_counts()
            _save_failure_point_fallback(
                output, result, serialization_error=error
            )
        rows.append(result)
        all_artifacts.append(artifacts)
        saved_counts.append(counts)
        parity_rows.extend(list(artifacts.get("transfer_parity_rows", [])))
        derivative_rows.extend(_derivative_rows_for_point(result))
    return (
        rows,
        all_artifacts,
        parity_rows,
        derivative_rows,
        saved_counts,
        exceptions,
    )


def _point_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    stability = row.get("stability_certificate", {})
    return {
        "z": row["z"],
        "alpha_G": row["alpha_G"],
        "outcome": row["outcome"],
        "derivative_certified": row["derivative_certified"],
        "failed_gates": ";".join(row.get("failed_gates", [])),
        "base_period_ms": row.get("base_lut_orbit_parity", {}).get("smooth_period_ms"),
        "half_period_ms": row.get("half_lut_orbit_parity", {}).get("smooth_period_ms"),
        "base_p_closure": row.get("base_smooth_shooting", {}).get("event_restarted_p_closure"),
        "base_p2_closure": row.get("base_smooth_shooting", {}).get("event_restarted_p2_closure"),
        "half_p_closure": row.get("half_smooth_shooting", {}).get("event_restarted_p_closure"),
        "half_p2_closure": row.get("half_smooth_shooting", {}).get("event_restarted_p2_closure"),
        "rho_max": stability.get("rho_max"),
        "unit_circle_margin": stability.get("unit_circle_margin"),
        "required_margin": stability.get("required_margin"),
        "transfer_parity_pass": row.get("transfer_parity", {}).get("pass"),
    }


def _overall_verdict(rows: list[Mapping[str, Any]], engineering_pass: bool) -> str:
    if not engineering_pass:
        return "STAGE0F_ENGINEERING_OR_PROVENANCE_FAIL"
    certified = {
        float(row["alpha_G"])
        for row in rows
        if row.get("outcome") == "stable_periodic_orbit_derivative_certified"
    }
    if certified == {15.0, 16.0}:
        return "STAGE0F_DERIVATIVE_CERTIFIED_ALPHA15_AND_ALPHA16"
    if certified == {15.0}:
        return "STAGE0F_DERIVATIVE_CERTIFIED_ALPHA15_ONLY"
    if certified == {16.0}:
        return "STAGE0F_DERIVATIVE_CERTIFIED_ALPHA16_ONLY"
    if any(row.get("outcome") in {"periodic_orbit_derivative_unresolved", "numerical_failure"} for row in rows):
        return "STAGE0F_NUMERICAL_UNRESOLVED"
    return "STAGE0F_NO_DERIVATIVE_CERTIFICATE_AT_LOCKED_POINTS"


def _failure_reason(row: Mapping[str, Any]) -> str:
    return ", ".join(row.get("failed_gates", [])) or str(row.get("outcome", "unavailable"))


def _plot(output: Path, rows: list[Mapping[str, Any]], artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    colors = {15.0: "#2a9d8f", 16.0: "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    panel_content = {"A": False, "B": False, "C": False, "D": False}
    failure_annotations = 0
    partial_points = [
        f"a{int(float(row['alpha_G']))}: {_failure_reason(row)}"
        for row in rows
        if row.get("outcome") != "stable_periodic_orbit_derivative_certified"
    ]

    def annotate_partial(ax: plt.Axes) -> int:
        if not partial_points:
            return 0
        ax.text(
            0.01,
            0.01,
            "partial/failure: " + " | ".join(partial_points),
            ha="left",
            va="bottom",
            fontsize=6,
            color="#8b1a1a",
            wrap=True,
            transform=ax.transAxes,
        )
        return 1

    ax = axes[0, 0]
    for row, item in zip(rows, artifacts):
        cycle = item.get("cycles", {}).get("base")
        if cycle is None:
            continue
        trace = cycle["trace"]
        period = float(cycle["period_ms"][0])
        mask = np.asarray(trace["time_ms"]) <= period
        ax.plot(np.asarray(trace["time_ms"])[mask], 1000.0 * np.asarray(trace["state"])[mask, 0], color=colors[float(row["alpha_G"])], lw=1.0, label=rf"$\alpha_G={row['alpha_G']:.0f}$")
        panel_content["A"] = True
    if panel_content["A"]:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "orbit unavailable\n" + " | ".join(_failure_reason(row) for row in rows), ha="center", va="center", transform=ax.transAxes)
        panel_content["A"] = True
        failure_annotations += 1
    ax.set(xlabel="cycle time (ms)", ylabel="E rate (Hz)", title="A  Smooth exact-table orbit")
    failure_annotations += annotate_partial(ax)

    ax = axes[0, 1]
    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    for row in rows:
        for dt_label in ("base", "half"):
            summary = row.get(f"{dt_label}_variational_consistency", {})
            for method in DERIVATIVE_LABELS_V11:
                value = summary.get("spectral_radii", {}).get(method)
                if value is None:
                    continue
                labels.append(f"a{int(row['alpha_G'])}\n{dt_label[0]}\n{method.replace('finite_difference_', 'fd')}")
                values.append(float(value))
                bar_colors.append(colors[float(row["alpha_G"])])
    if values:
        ax.bar(np.arange(len(values)), values, color=bar_colors, alpha=0.8)
        ax.set_xticks(np.arange(len(values)), labels, fontsize=6)
        panel_content["B"] = True
    else:
        ax.text(0.5, 0.5, "derivative unavailable\n" + " | ".join(_failure_reason(row) for row in rows), ha="center", va="center", transform=ax.transAxes)
        panel_content["B"] = True
        failure_annotations += 1
    ax.axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax.set(ylabel="spectral radius", title="B  Discrete variational agreement")
    failure_annotations += annotate_partial(ax)

    ax = axes[1, 0]
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="#777777", lw=0.8, ls="--")
    multiplier_count = 0
    for row, item in zip(rows, artifacts):
        for dt_label, marker in (("base", "o"), ("half", "x")):
            variation = item.get("variational", {}).get(dt_label)
            if variation is None or not bool(variation.get("valid", False)) or "multipliers" not in variation:
                continue
            for method in DERIVATIVE_LABELS_V11:
                multipliers = np.asarray(variation["multipliers"][method], dtype=complex)
                ax.scatter(multipliers.real, multipliers.imag, s=18, marker=marker, color=colors[float(row["alpha_G"])], alpha=0.55)
                multiplier_count += multipliers.size
    if multiplier_count:
        panel_content["C"] = True
    else:
        ax.text(0.5, 0.5, "multipliers unavailable\n" + " | ".join(_failure_reason(row) for row in rows), ha="center", va="center", transform=ax.transAxes)
        panel_content["C"] = True
        failure_annotations += 1
    ax.axhline(0.0, color="#bbbbbb", lw=0.5)
    ax.axvline(0.0, color="#bbbbbb", lw=0.5)
    ax.set(aspect="equal", xlabel="Re(lambda)", ylabel="Im(lambda)", title="C  Transverse multipliers")
    failure_annotations += annotate_partial(ax)

    ax = axes[1, 1]
    metric_names = ("rate_khz", "d_rate_d_mu_khz_per_mv", "d_rate_d_sigma_khz_per_mv")
    bars = 0
    for row_index, row in enumerate(rows):
        for pop_index, pop in enumerate(("E", "I")):
            metrics = row.get("transfer_parity", {}).get("populations", {}).get(pop, {}).get("metrics", {})
            errors = [metrics.get(name, {}).get("maximum_relative_error_with_floor") for name in metric_names]
            if any(value is None for value in errors):
                continue
            offset = (-0.25 if row_index == 0 else 0.25) + (-0.09 if pop_index == 0 else 0.09)
            ax.bar(np.arange(3) + offset, errors, width=0.17, color=colors[float(row["alpha_G"])], alpha=0.55 if pop == "E" else 0.9, label=f"a{int(row['alpha_G'])} {pop}")
            bars += 1
    if bars:
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        panel_content["D"] = True
    else:
        ax.text(0.5, 0.5, "exact parity unavailable\n" + " | ".join(_failure_reason(row) for row in rows), ha="center", va="center", transform=ax.transAxes)
        panel_content["D"] = True
        failure_annotations += 1
    ax.axhline(0.05, color="#777777", ls="--", lw=0.7)
    ax.set_xticks(np.arange(3), ("rate", "d/dmu", "d/dsigma"))
    ax.set(ylabel="max relative error (floored)", title="D  Variational-path exact parity")
    failure_annotations += annotate_partial(ax)

    png = figures / "stage0f_v1_1_variational_certificate.png"
    pdf = figures / "stage0f_v1_1_variational_certificate.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    readme = """### stage0f_v1_1_variational_certificate.png

这张诊断图展示平滑 exact-table 闭轨、chain-rule 与两档 boundary-aware finite-difference 离散变分结果、全部横向乘子，以及实际 variational Euler 轨道上的 direct exact Siegert parity。若某个前置门提前停止，对应 panel 会直接写出失败原因，不会留空。

**关注点**：P/P² 必须来自同一个 event-restarted Poincaré map；nominal map identity、两种 transversality、whole-return Jv、两档 stencil 和 exact transfer parity 必须同时通过，才可形成导数证书。

### stage0f_v1_1_variational_certificate.pdf

与 PNG 内容相同的矢量版本，便于放大检查靠近原点的乘子和不同导数构造之间的差异。

**关注点**：即使证书通过，Stage 1 和空间模拟仍关闭；该结果不代表完整发作生命周期。
"""
    _atomic_text(figures / "README.md", readme)
    metadata = {
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "panel_content": panel_content,
        "panels_with_content": int(sum(panel_content.values())),
        "failure_annotations": failure_annotations,
        "partial_point_annotations": partial_points,
        "multiplier_count_rendered": int(multiplier_count),
    }
    _atomic_json(output / "figure_metadata.json", metadata)
    return metadata


def _fallback_failure_plot(
    output: Path, rows: list[Mapping[str, Any]], error: BaseException
) -> dict[str, Any]:
    """Always leave a four-panel, self-explanatory figure after plotting failure."""

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    reason = f"plot fallback: {type(error).__name__}: {error}"
    point_reason = " | ".join(_failure_reason(row) for row in rows) or "no point outcome"
    for label, ax in zip(("A", "B", "C", "D"), axes.ravel()):
        ax.text(
            0.5,
            0.5,
            f"Panel {label} unavailable\n{reason}\n{point_reason}",
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    png = figures / "stage0f_v1_1_variational_certificate.png"
    pdf = figures / "stage0f_v1_1_variational_certificate.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    _atomic_text(
        figures / "README.md",
        """### stage0f_v1_1_variational_certificate.png

权威诊断图生成过程中发生工程异常，因此四个 panel 均明确显示异常类型和 point-level failure reason；没有空白 panel，也不能据此形成科学证书。

**关注点**：该图只证明 failure path 完整落盘，结果必须判 engineering/provenance failure。

### stage0f_v1_1_variational_certificate.pdf

与 PNG 相同的失败诊断矢量版本。

**关注点**：任何 plotting fallback 都会关闭证书 verdict。
""",
    )
    metadata = {
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "panel_content": {"A": True, "B": True, "C": True, "D": True},
        "panels_with_content": 4,
        "failure_annotations": 4,
        "multiplier_count_rendered": 0,
        "plotting_exception": {"type": type(error).__name__, "message": str(error)},
    }
    _atomic_json(output / "figure_metadata.json", metadata)
    return metadata


def _npz_exact_keys(path: Path, expected: set[str]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return set(payload.files) == expected
    except (OSError, ValueError):
        return False


def _artifact_structure_summary(
    output: Path,
    rows: list[Mapping[str, Any]],
    all_artifacts: list[Mapping[str, Any]],
    parity_rows: list[Mapping[str, Any]],
    derivative_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit group identity and exact array schemas, not only global counts."""

    expected_parity_groups = {
        (alpha, dt_ms, pop)
        for alpha in (15.0, 16.0)
        for dt_ms in (0.125, 0.0625)
        for pop in ("E", "I")
    }
    parity_groups: dict[tuple[float, float, str], list[Mapping[str, Any]]] = {}
    malformed_parity_rows = 0
    for row in parity_rows:
        try:
            key = (
                float(row["alpha_G"]),
                float(row["dt_ms"]),
                str(row["population"]),
            )
        except (KeyError, TypeError, ValueError):
            malformed_parity_rows += 1
            continue
        parity_groups.setdefault(key, []).append(row)
    parity_group_checks: dict[str, Any] = {}
    for key in sorted(expected_parity_groups):
        group = parity_groups.get(key, [])
        sample_indices = [row.get("sample_index") for row in group]
        source_indices = [row.get("source_state_index") for row in group]
        label = f"alpha_{int(key[0])}_dt_{key[1]:g}_{key[2]}"
        parity_group_checks[label] = {
            "rows": len(group),
            "exact_512_rows": len(group) == 512,
            "sample_index_exact_0_to_511": set(sample_indices) == set(range(512)),
            "source_state_indices_unique": len(source_indices) == 512
            and len(set(source_indices)) == 512,
            "source_exact": {row.get("source") for row in group}
            == {"variational_nominal_euler_state"},
            "z_exact": {float(row.get("z", np.nan)) for row in group} == {0.85},
        }
    parity_structure_pass = bool(
        malformed_parity_rows == 0
        and set(parity_groups) == expected_parity_groups
        and all(
            all(value for name, value in audit.items() if name != "rows")
            for audit in parity_group_checks.values()
        )
    )

    expected_derivative_groups = {
        (alpha, dt_label)
        for alpha in (15.0, 16.0)
        for dt_label in ("base", "half")
    }
    derivative_groups: dict[tuple[float, str], int] = {}
    malformed_derivative_rows = 0
    for row in derivative_rows:
        try:
            key = (float(row["alpha_G"]), str(row["dt_label"]))
        except (KeyError, TypeError, ValueError):
            malformed_derivative_rows += 1
            continue
        derivative_groups[key] = derivative_groups.get(key, 0) + 1
    derivative_structure_pass = bool(
        malformed_derivative_rows == 0
        and set(derivative_groups) == expected_derivative_groups
        and all(count == 1 for count in derivative_groups.values())
    )

    expected_certificate_keys = {
        f"{dt_label}_nominal_{suffix}"
        for dt_label in ("base", "half")
        for suffix in ("time_ms", "state_trace", "moment_trace")
    }
    expected_certificate_keys.update(
        f"{dt_label}_{method}_{suffix}"
        for dt_label in ("base", "half")
        for method in DERIVATIVE_LABELS_V11
        for suffix in (
            "poincare_matrix",
            "full_event_tangent_normalized",
            "multiplier_real",
            "multiplier_imag",
        )
    )
    expected_jv_keys = {
        f"{dt_label}_epsilon_{index}_jv_matrix"
        for dt_label in ("base", "half")
        for index in range(2)
    }
    point_checks: dict[str, Any] = {}
    seen_alphas: set[float] = set()
    for row, artifacts in zip(rows, all_artifacts):
        alpha = float(row.get("alpha_G", np.nan))
        seen_alphas.add(alpha)
        label = f"alpha_{int(alpha)}" if np.isfinite(alpha) else "alpha_invalid"
        variational = artifacts.get("variational", {})
        jv = artifacts.get("whole_return_jv", {})
        variational_schema = True
        jv_schema = True
        for dt_label in ("base", "half"):
            item = variational.get(dt_label, {})
            variational_schema = bool(
                variational_schema
                and item.get("valid", False)
                and set(item.get("poincare_matrices", {})) == set(DERIVATIVE_LABELS_V11)
                and set(item.get("full_event_tangents_normalized", {}))
                == set(DERIVATIVE_LABELS_V11)
                and set(item.get("multipliers", {})) == set(DERIVATIVE_LABELS_V11)
                and all(
                    np.asarray(item["poincare_matrices"][method]).shape == (8, 8)
                    and np.asarray(item["full_event_tangents_normalized"][method]).shape
                    == (9, 8)
                    and np.asarray(item["multipliers"][method]).shape == (8,)
                    for method in DERIVATIVE_LABELS_V11
                )
            )
            jv_rows = list(jv.get(dt_label, []))
            jv_schema = bool(
                jv_schema
                and len(jv_rows) == 2
                and [float(item.get("epsilon_relative", np.nan)) for item in jv_rows]
                == [1.0e-3, 3.0e-4]
                and all(
                    bool(item.get("valid", False))
                    and np.asarray(item.get("jv_matrix", [])).shape == (8, 8)
                    for item in jv_rows
                )
            )
        point = (
            output / "per_point" / f"z_0p85_alpha_{int(alpha)}"
            if np.isfinite(alpha)
            else output / "per_point" / "alpha_invalid"
        )
        point_checks[label] = {
            "cycles_exact": set(artifacts.get("cycles", {})) == {"base", "half"},
            "variational_exact": set(variational) == {"base", "half"}
            and variational_schema,
            "whole_return_jv_exact": set(jv) == {"base", "half"} and jv_schema,
            "certificate_npz_keys_exact": _npz_exact_keys(
                point / "discrete_variational_certificate.npz",
                expected_certificate_keys,
            ),
            "whole_return_jv_npz_keys_exact": _npz_exact_keys(
                point / "whole_return_jv_audit.npz", expected_jv_keys
            ),
        }
    point_structure_pass = bool(
        len(rows) == 2
        and len(all_artifacts) == 2
        and seen_alphas == {15.0, 16.0}
        and set(point_checks) == {"alpha_15", "alpha_16"}
        and all(all(checks.values()) for checks in point_checks.values())
    )
    return {
        "pass": bool(
            parity_structure_pass
            and derivative_structure_pass
            and point_structure_pass
        ),
        "parity_structure_pass": parity_structure_pass,
        "malformed_parity_rows": malformed_parity_rows,
        "parity_group_checks": parity_group_checks,
        "unexpected_parity_groups": sorted(
            [list(key) for key in set(parity_groups) - expected_parity_groups]
        ),
        "derivative_structure_pass": derivative_structure_pass,
        "malformed_derivative_rows": malformed_derivative_rows,
        "derivative_group_counts": {
            f"alpha_{int(key[0])}_{key[1]}": count
            for key, count in sorted(derivative_groups.items())
        },
        "point_structure_pass": point_structure_pass,
        "point_checks": point_checks,
        "expected_certificate_npz_keys": sorted(expected_certificate_keys),
        "expected_whole_return_jv_npz_keys": sorted(expected_jv_keys),
    }


def artifact_completeness_summary(
    output: Path,
    rows: list[Mapping[str, Any]],
    all_artifacts: list[Mapping[str, Any]],
    parity_rows: list[Mapping[str, Any]],
    derivative_rows: list[Mapping[str, Any]],
    saved_counts: list[Mapping[str, int]],
    figure_metadata: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    complete_path = bool(
        len(rows) == 2
        and len(all_artifacts) == 2
        and len(saved_counts) == 2
        and all(
            set(item.get("cycles", {})) == {"base", "half"}
            and set(item.get("variational", {})) == {"base", "half"}
            and set(item.get("whole_return_jv", {})) == {"base", "half"}
            for item in all_artifacts
        )
    )
    observed = {
        "parity_rows": len(parity_rows),
        "derivative_summary_rows": len(derivative_rows),
        "cycle_traces": sum(int(row["cycle_traces"]) for row in saved_counts),
        "certificate_matrices": sum(int(row["certificate_matrices"]) for row in saved_counts),
        "certificate_multipliers": sum(int(row["certificate_multipliers"]) for row in saved_counts),
        "whole_return_jv_arrays": sum(int(row["whole_return_jv_arrays"]) for row in saved_counts),
        "figure_panels_with_content": int(figure_metadata["panels_with_content"]),
    }
    contract = cfg["artifact_contract"]
    exact_checks = {
        "parity_rows": observed["parity_rows"] == int(contract["complete_parity_rows"]),
        "derivative_summary_rows": observed["derivative_summary_rows"] == int(contract["complete_derivative_summary_rows"]),
        "cycle_traces": observed["cycle_traces"] == int(contract["complete_cycle_traces"]),
        "certificate_matrices": observed["certificate_matrices"] == int(contract["complete_certificate_matrices"]),
        "certificate_multipliers": observed["certificate_multipliers"] == int(contract["complete_certificate_multipliers"]),
        "whole_return_jv_arrays": observed["whole_return_jv_arrays"] == int(contract["complete_whole_return_jv_arrays"]),
    }
    structure = _artifact_structure_summary(
        output, rows, all_artifacts, parity_rows, derivative_rows
    )
    required = [
        output / "EXECUTION_LOCK.json",
        output / "parameter_point_outcomes.json",
        output / "parameter_point_outcomes.csv",
        output / "figure_metadata.json",
        output / "figures/stage0f_v1_1_variational_certificate.png",
        output / "figures/stage0f_v1_1_variational_certificate.pdf",
        output / "figures/README.md",
        output / "stage0f_v1_1_variational_summary.json",
        output / "STATUS.md",
    ]
    for alpha in (15, 16):
        point = output / "per_point" / f"z_0p85_alpha_{alpha}"
        required.extend(
            [point / "point_outcome.json", point / "partial_artifact_report.json"]
        )
    if complete_path:
        required.extend([output / "exact_transfer_parity.csv", output / "derivative_consistency.csv"])
        for alpha in (15, 16):
            point = output / "per_point" / f"z_0p85_alpha_{alpha}"
            required.extend(
                [
                    point / "exact_transfer_parity.csv",
                    point / "smooth_base_cycle_trace.npz",
                    point / "smooth_half_cycle_trace.npz",
                    point / "discrete_variational_certificate.npz",
                    point / "discrete_variational_report.json",
                    point / "whole_return_jv_audit.npz",
                    point / "whole_return_jv_report.json",
                ]
            )
    files = {str(path.relative_to(output)): bool(path.is_file() and path.stat().st_size > 0) for path in required}
    figure_pass = bool(
        observed["figure_panels_with_content"] == int(contract["figure_panels"])
        and all(figure_metadata["panel_content"].values())
        and "plotting_exception" not in figure_metadata
    )
    if complete_path:
        count_pass = bool(all(exact_checks.values()) and structure["pass"])
    else:
        count_pass = True
    return {
        "pass": bool(count_pass and all(files.values()) and figure_pass),
        "complete_certificate_path_reached": complete_path,
        "observed_counts": observed,
        "exact_complete_path_checks": exact_checks,
        "exact_complete_path_structure": structure,
        "required_nonempty_files": files,
        "figure_pass": figure_pass,
        "failure_figure_annotations": int(figure_metadata["failure_annotations"]),
    }


def _status_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Topic 4 Stage 0F v1.1 状态",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Engineering/provenance pass: `{str(summary['engineering_pass']).lower()}`",
        f"- Execution exception: `{str(summary.get('execution_exception', False)).lower()}`",
        f"- Artifact completeness pass: `{str(summary['artifact_completeness']['pass']).lower()}`",
        f"- Peak RSS: `{summary['resource_usage']['peak_rss_gib']:.3f} GiB`",
        f"- Wall time: `{summary['resource_usage']['wall_seconds']:.2f} s`",
        "- Stage 1: `CLOSED`",
        "- Spatial simulation: `CLOSED`",
        "",
        "## 固定点结果",
        "",
    ]
    for row in summary["parameter_points"]:
        lines.append(f"- `z={row['z']:.2f}, alpha_G={row['alpha_G']:.0f}`: `{row['outcome']}`; failed gates = `{','.join(row.get('failed_gates', [])) or 'none'}`")
    lines.extend(["", "本阶段只修复 homogeneous frozen fast orbit 的导数证书；无论结果如何，都不自动开放 slow lifecycle、空间耦合或 Stage 1。", ""])
    return "\n".join(lines)


def _recover_two_point_failure_bundle(
    output: Path,
    cfg: Mapping[str, Any],
    error: BaseException,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    """Ensure both locked points have readable outcomes after an outer exception."""

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    counts: list[dict[str, int]] = []
    for point_cfg in cfg["points"]:
        alpha = int(round(float(point_cfg["alpha_G"])))
        point = output / "per_point" / f"z_0p85_alpha_{alpha}"
        outcome_path = point / "point_outcome.json"
        if outcome_path.is_file():
            try:
                result = json.loads(outcome_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = _engineering_point_result(
                    point_cfg,
                    outcome="numerical_failure",
                    failed_gate="outer_execution_exception",
                    error=error,
                )
        else:
            result = _engineering_point_result(
                point_cfg,
                outcome="not_executed_due_to_engineering_failure",
                failed_gate="outer_execution_exception",
                error=error,
            )
        point.mkdir(parents=True, exist_ok=True)
        _atomic_json(outcome_path, result)
        partial = point / "partial_artifact_report.json"
        if not partial.is_file() or partial.stat().st_size == 0:
            _atomic_json(
                partial,
                {
                    "schema_version": "topic4_stage0f_v1_1_point_artifact_report.v1",
                    "z": float(result["z"]),
                    "alpha_G": float(result["alpha_G"]),
                    "outcome": str(result["outcome"]),
                    "failed_gates": list(result.get("failed_gates", [])),
                    "outer_execution_exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "serialized_counts": _empty_saved_counts(),
                },
            )
        rows.append(result)
        artifacts.append({})
        counts.append(_empty_saved_counts())
    return rows, artifacts, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("Refusing authoritative Stage0F v1.1 execution without --confirm-run")
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    output = ROOT / str(cfg["result_root"])
    # Preflight intentionally occurs before mkdir: provenance/transfer/input failure
    # must not consume the non-overwrite canonical result root.
    inputs, transfer, stage0e_by_alpha = _preflight(config_path, cfg, output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    lock = _write_execution_lock(output, config_path, inputs)
    rows: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    all_parity_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    saved_counts: list[dict[str, int]] = []
    execution_exceptions: list[dict[str, Any]] = []
    try:
        try:
            (
                rows,
                all_artifacts,
                all_parity_rows,
                derivative_rows,
                saved_counts,
                point_exceptions,
            ) = _run_locked_points_fail_closed(
                output, cfg, transfer, stage0e_by_alpha
            )
            execution_exceptions.extend(point_exceptions)
        except Exception as error:  # final outer containment around the point runner
            execution_exceptions.append(
                {
                    "phase": "outer_point_execution",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            rows, all_artifacts, saved_counts = _recover_two_point_failure_bundle(
                output, cfg, error
            )
            all_parity_rows = []
            derivative_rows = []
    finally:
        # Everything below is best-effort finalization and is deliberately kept
        # outside the numerical success path.
        if len(rows) != 2:
            error = RuntimeError("locked point runner did not return exactly two outcomes")
            execution_exceptions.append(
                {
                    "phase": "point_cardinality",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            rows, all_artifacts, saved_counts = _recover_two_point_failure_bundle(
                output, cfg, error
            )
            all_parity_rows = []
            derivative_rows = []

        try:
            _atomic_json(output / "parameter_point_outcomes.json", rows)
            _write_csv(
                output / "parameter_point_outcomes.csv",
                [_point_csv_row(row) for row in rows],
            )
            if all_parity_rows:
                _write_csv(output / "exact_transfer_parity.csv", all_parity_rows)
            if derivative_rows:
                _write_csv(output / "derivative_consistency.csv", derivative_rows)
        except Exception as error:
            execution_exceptions.append(
                {
                    "phase": "aggregate_artifact_serialization",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            # JSON and the compact CSV are the minimum aggregate outcome bundle.
            _atomic_json(output / "parameter_point_outcomes.json", rows)
            _write_csv(
                output / "parameter_point_outcomes.csv",
                [_point_csv_row(row) for row in rows],
            )

        try:
            figure_metadata = _plot(output, rows, all_artifacts)
        except Exception as error:
            execution_exceptions.append(
                {
                    "phase": "plotting",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            figure_metadata = _fallback_failure_plot(output, rows, error)

        try:
            unchanged = _finalize_execution_lock(output, config_path, cfg, lock)
        except Exception as error:
            execution_exceptions.append(
                {
                    "phase": "execution_lock_finalization",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            unchanged = False
            lock["all_inputs_and_sources_unchanged_during_execution"] = False
            lock["finalization_exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _atomic_json(output / "EXECUTION_LOCK.json", lock)

        wall_seconds = float(time.monotonic() - started)
        peak_rss_gib = float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2
        )
        rss_pass = peak_rss_gib < float(
            cfg["resource_contract"]["max_memory_gib"]
        )
        execution_exception = bool(execution_exceptions)
        provisional = {
            "schema_version": "topic4_stage0f_v1_1_variational_certificate.v1",
            "verdict": "PENDING_ARTIFACT_COMPLETENESS",
            "engineering_pass": False,
            "execution_exception": execution_exception,
            "execution_exception_records": execution_exceptions,
            "sources_and_inputs_unchanged": unchanged,
            "artifact_completeness": {"pass": False},
            "parameter_points": rows,
            "resource_usage": {
                "blas_threads": 1,
                "peak_rss_gib": peak_rss_gib,
                "rss_limit_gib": float(cfg["resource_contract"]["max_memory_gib"]),
                "rss_pass": rss_pass,
                "wall_seconds": wall_seconds,
            },
            "figure_metadata": figure_metadata,
            "scope": {
                "stage1_open": False,
                "spatial_simulation_open": False,
                "parameter_search_performed": False,
                "equations_changed": False,
            },
        }
        _atomic_json(output / "stage0f_v1_1_variational_summary.json", provisional)
        _atomic_text(output / "STATUS.md", _status_text(provisional))
        try:
            completeness = artifact_completeness_summary(
                output,
                rows,
                all_artifacts,
                all_parity_rows,
                derivative_rows,
                saved_counts,
                figure_metadata,
                cfg,
            )
        except Exception as error:
            execution_exceptions.append(
                {
                    "phase": "artifact_completeness_audit",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            execution_exception = True
            completeness = {
                "pass": False,
                "audit_exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        engineering_pass = bool(
            unchanged
            and rss_pass
            and completeness["pass"]
            and not execution_exception
            and "plotting_exception" not in figure_metadata
        )
        summary = dict(provisional)
        summary["execution_exception"] = execution_exception
        summary["execution_exception_records"] = execution_exceptions
        summary["artifact_completeness"] = completeness
        summary["engineering_pass"] = engineering_pass
        summary["verdict"] = _overall_verdict(rows, engineering_pass)
        summary["stage0f_v1_archive"] = {
            "path": str((ROOT / str(cfg["stage0f_v1_archive"])).resolve()),
            "authoritative": False,
            "reason": "hidden non-event-restarted closure gate and boundary-incompatible centered probes",
        }
        _atomic_json(output / "stage0f_v1_1_variational_summary.json", summary)
        _atomic_text(output / "STATUS.md", _status_text(summary))
        print(
            json.dumps(
                {
                    "verdict": summary["verdict"],
                    "output": str(output),
                    "peak_rss_gib": peak_rss_gib,
                    "artifact_completeness": completeness["pass"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
