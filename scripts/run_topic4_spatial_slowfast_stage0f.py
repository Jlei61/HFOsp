#!/usr/bin/env python3
"""Run the locked Stage-0F smooth-transfer variational certificate."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0f")

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
from src.topic4_spatial_slowfast_stage0f import (  # noqa: E402
    DERIVATIVE_LABELS,
    SmoothDomain,
    SmoothSiegertTransfer,
    run_point_certificate,
)


DEFAULT_CONFIG = ROOT / "config/topic4_spatial_slowfast_stage0f.yaml"
SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0f-smooth-transfer-variational-certificate-design.md"
MODULE = ROOT / "src/topic4_spatial_slowfast_stage0f.py"
RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0f.py"
TESTS = ROOT / "tests/test_topic4_spatial_slowfast_stage0f.py"
STAGE0E_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0e.py"
STAGE0C_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c.py"
TRANSFER_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c_transfer.py"
LOCKED_POINTS = ((0.85, 15.0), (0.85, 16.0))


def _validate_config(cfg: Mapping[str, Any]) -> None:
    points = tuple((float(row["z"]), float(row["alpha_G"])) for row in cfg["points"])
    if points != LOCKED_POINTS:
        raise ValueError("Stage0F fixed points drifted")
    if cfg["model"] != {"w_ee_mult": 1.1, "ratio": 1.0}:
        raise ValueError("Stage0F model constants drifted")
    if cfg["section"] != {
        "state_index": 8,
        "state_name": "S_G",
        "level": 0.15,
        "direction": "upward",
        "min_return_ms": 300.0,
        "max_return_ms": 1200.0,
    }:
        raise ValueError("Stage0F section drifted")
    if cfg["time_steps_ms"] != [0.125, 0.0625] or int(cfg["phase_bins"]) != 256:
        raise ValueError("Stage0F integration schedule drifted")
    if int(cfg["transfer_parity_phase_samples"]) != 512:
        raise ValueError("Stage0F transfer parity schedule drifted")
    if cfg["smooth_transfer"] != {
        "mu_min_mv": -160.0,
        "mu_max_mv": 80.0,
        "sigma_min_mv": 3.0,
        "sigma_max_mv": 20.0,
        "spline_degree_mu": 3,
        "spline_degree_sigma": 3,
        "smoothing": 0.0,
    }:
        raise ValueError("Stage0F smooth transfer drifted")
    if cfg["transfer_parity"] != {
        "rate_absolute_khz": 5.0e-5,
        "rate_relative": 5.0e-3,
        "rate_relative_floor_khz": 1.0e-4,
        "derivative_absolute_khz_per_mv": 5.0e-5,
        "derivative_relative": 5.0e-2,
        "derivative_relative_floor_khz_per_mv": 1.0e-7,
    }:
        raise ValueError("Stage0F exact transfer parity drifted")
    if cfg["shooting"] != {
        "max_iterations": 20,
        "residual_tolerance": 1.0e-8,
        "period_cv_tolerance": 1.0e-3,
        "aligned_cycle_residual_tolerance": 2.0e-4,
        "minimum_iterations": 4,
    }:
        raise ValueError("Stage0F shooting contract drifted")
    if cfg["orbit_parity"] != {
        "period_abs_ms": 1.0,
        "aligned_waveform_residual": 3.0e-2,
        "dt_period_abs_ms": 1.0,
        "dt_period_relative": 5.0e-3,
        "dt_aligned_waveform_residual": 3.0e-2,
    }:
        raise ValueError("Stage0F orbit parity drifted")
    if cfg["variational"] != {
        "centered_relative_steps": [1.0e-5, 3.0e-6],
        "centered_absolute_floor": 1.0e-9,
        "matrix_relative_difference_max": 5.0e-2,
        "matrix_norm_floor": 1.0e-8,
        "spectral_radius_range_max": 2.0e-2,
        "section_row_abs_max": 1.0e-10,
        "minimum_transversality_per_ms": 1.0e-4,
    }:
        raise ValueError("Stage0F variational contract drifted")
    if cfg["stability"] != {
        "minimum_unit_circle_margin": 5.0e-2,
        "uncertainty_multiplier": 3.0,
    }:
        raise ValueError("Stage0F stability margin drifted")
    if cfg["physical_acceptance"] != {
        "finite_high_max_hz": 100.0,
        "max_refractory_occupancy": 0.05,
    }:
        raise ValueError("Stage0F physical contract drifted")
    if cfg["resource_contract"] != {"blas_threads": 1, "max_memory_gib": 4.0}:
        raise ValueError("Stage0F resource contract drifted")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("Stage0F scope expansion is forbidden")
    if Path(str(cfg["result_root"])).name != "stage0f_smooth_transfer_variational_certificate":
        raise ValueError("Stage0F result root drifted")


def _source_paths(config_path: Path) -> dict[str, Path]:
    return {"spec": SPEC, "config": config_path, "module": MODULE, "runner": RUNNER, "tests": TESTS}


def _source_hashes(config_path: Path) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in _source_paths(config_path).items()
    }


def _locked_input_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    stage0e = ROOT / str(cfg["stage0e_root"])
    mapping = {
        "stage0e_summary": stage0e / "stage0e_poincare_floquet_summary.json",
        "extra_fine_transfer": ROOT / str(cfg["transfer_path"]),
        "stage0e_module": STAGE0E_MODULE,
        "stage0c_module": STAGE0C_MODULE,
        "transfer_module": TRANSFER_MODULE,
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
        raise RuntimeError("locked Stage0F upstream provenance mismatch")
    return rows


def _write_execution_lock(output: Path, config_path: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "schema_version": "topic4_stage0f_execution_lock.v1",
        "locked_before_numerical_execution": True,
        "upstream_inputs_pre_execution": dict(inputs),
        "stage0f_sources_pre_execution": _source_hashes(config_path),
        "upstream_inputs_post_execution": None,
        "stage0f_sources_post_execution": None,
        "all_inputs_and_sources_unchanged_during_execution": None,
    }
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return lock


def _finalize_execution_lock(
    output: Path, config_path: Path, cfg: Mapping[str, Any], lock: dict[str, Any]
) -> bool:
    post_inputs = _verify_locked_inputs(cfg)
    post_sources = _source_hashes(config_path)
    unchanged = bool(
        post_inputs == lock["upstream_inputs_pre_execution"]
        and post_sources == lock["stage0f_sources_pre_execution"]
    )
    lock["upstream_inputs_post_execution"] = post_inputs
    lock["stage0f_sources_post_execution"] = post_sources
    lock["all_inputs_and_sources_unchanged_during_execution"] = unchanged
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return unchanged


def _load_transfer(cfg: Mapping[str, Any]) -> SmoothSiegertTransfer:
    path = ROOT / str(cfg["transfer_path"])
    with np.load(path, allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("Stage0F source transfer did not assert no clipping")
        original = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"], name="extra_fine"
        )
    smooth_cfg = cfg["smooth_transfer"]
    return SmoothSiegertTransfer.from_extended(
        original,
        domain=SmoothDomain(
            float(smooth_cfg["mu_min_mv"]),
            float(smooth_cfg["mu_max_mv"]),
            float(smooth_cfg["sigma_min_mv"]),
            float(smooth_cfg["sigma_max_mv"]),
        ),
        kx=int(smooth_cfg["spline_degree_mu"]),
        ky=int(smooth_cfg["spline_degree_sigma"]),
        smoothing=float(smooth_cfg["smoothing"]),
    )


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_stage0e_point(cfg: Mapping[str, Any], alpha: int) -> dict[str, Any]:
    point = ROOT / str(cfg["stage0e_root"]) / "per_point" / f"z_0p85_alpha_{alpha}"
    outcome = json.loads((point / "point_outcome.json").read_text(encoding="utf-8"))
    shooting = _load_npz_dict(point / "shooting_iterates.npz")
    if outcome.get("outcome") != "periodic_orbit_numerically_unresolved":
        raise RuntimeError("Stage0F expected the locked Stage0E derivative-only unresolved outcome")
    if outcome.get("failed_gates") != ["floquet_epsilon_dt_or_margin"]:
        raise RuntimeError("Stage0E failure boundary drifted before Stage0F")
    return {
        "scales": np.asarray(outcome["state_scales"], dtype=float),
        "base_shooting_seed": np.asarray(shooting["base_state"][-1], dtype=float),
        "half_shooting_seed": np.asarray(shooting["half_state"][-1], dtype=float),
        "base_lut_trace": _load_npz_dict(point / "base_cycle_trace.npz"),
        "half_lut_trace": _load_npz_dict(point / "half_cycle_trace.npz"),
    }


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


def _save_point(output: Path, result: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    alpha = int(round(float(result["alpha_G"])))
    point = output / "per_point" / f"z_0p85_alpha_{alpha}"
    point.mkdir(parents=True, exist_ok=False)
    _atomic_json(point / "point_outcome.json", result)
    for label, cycle in artifacts.get("cycles", {}).items():
        _save_cycle(point / f"smooth_{label}_cycle_trace.npz", cycle)
    rows = list(artifacts.get("transfer_parity_rows", []))
    _write_csv(point / "exact_transfer_parity.csv", rows)
    if "variational" in artifacts:
        payload: dict[str, np.ndarray] = {}
        report: dict[str, Any] = {}
        for dt_label, item in artifacts["variational"].items():
            report[dt_label] = {
                key: value
                for key, value in item.items()
                if key not in {"poincare_matrices", "full_event_tangents_normalized", "multipliers"}
            }
            for method in DERIVATIVE_LABELS:
                prefix = f"{dt_label}_{method}"
                payload[f"{prefix}_poincare_matrix"] = np.asarray(item["poincare_matrices"][method], dtype=float)
                payload[f"{prefix}_full_event_tangent_normalized"] = np.asarray(
                    item["full_event_tangents_normalized"][method], dtype=float
                )
                multipliers = np.asarray(item["multipliers"][method], dtype=complex)
                payload[f"{prefix}_multiplier_real"] = multipliers.real
                payload[f"{prefix}_multiplier_imag"] = multipliers.imag
                report[dt_label].setdefault("multipliers", {})[method] = [
                    {"real": float(value.real), "imag": float(value.imag), "modulus": float(abs(value))}
                    for value in multipliers
                ]
        np.savez_compressed(point / "discrete_variational_matrices_and_multipliers.npz", **payload)
        _atomic_json(point / "discrete_variational_report.json", report)


def _point_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    stability = row.get("stability_certificate", {})
    base = row.get("base_variational_consistency", {})
    half = row.get("half_variational_consistency", {})
    return {
        "z": row["z"],
        "alpha_G": row["alpha_G"],
        "outcome": row["outcome"],
        "derivative_certified": row["derivative_certified"],
        "failed_gates": ";".join(row.get("failed_gates", [])),
        "base_smooth_period_ms": row.get("base_lut_orbit_parity", {}).get("smooth_period_ms"),
        "half_smooth_period_ms": row.get("half_lut_orbit_parity", {}).get("smooth_period_ms"),
        "base_lut_waveform_residual": row.get("base_lut_orbit_parity", {}).get("aligned_waveform_residual"),
        "half_lut_waveform_residual": row.get("half_lut_orbit_parity", {}).get("aligned_waveform_residual"),
        "base_matrix_max_relative_difference": max(base.get("matrix_relative_differences", {"none": np.nan}).values()),
        "half_matrix_max_relative_difference": max(half.get("matrix_relative_differences", {"none": np.nan}).values()),
        "base_spectral_radius_range": base.get("spectral_radius_range"),
        "half_spectral_radius_range": half.get("spectral_radius_range"),
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


def _plot(output: Path, rows: list[Mapping[str, Any]], artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    colors = {15.0: "#2a9d8f", 16.0: "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for row, item in zip(rows, artifacts):
        if "cycles" not in item:
            continue
        cycle = item["cycles"]["base"]
        trace = cycle["trace"]
        period = float(cycle["period_ms"][0])
        mask = np.asarray(trace["time_ms"]) <= period
        ax.plot(
            np.asarray(trace["time_ms"])[mask],
            1000.0 * np.asarray(trace["state"])[mask, 0],
            color=colors[float(row["alpha_G"])],
            lw=1.0,
            label=rf"$\alpha_G={row['alpha_G']:.0f}$",
        )
    ax.set(xlabel="cycle time (ms)", ylabel="E rate (Hz)", title="A  Smooth exact-table orbit")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    labels = []
    values = []
    bar_colors = []
    for row in rows:
        for dt_label in ("base", "half"):
            summary = row.get(f"{dt_label}_variational_consistency", {})
            for method in DERIVATIVE_LABELS:
                labels.append(f"a{int(row['alpha_G'])}\n{dt_label[0]}\n{method.replace('centered_', 'fd')}")
                values.append(float(summary.get("spectral_radii", {}).get(method, np.nan)))
                bar_colors.append(colors[float(row["alpha_G"])])
    ax.bar(np.arange(len(values)), values, color=bar_colors, alpha=0.8)
    ax.axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(values)), labels, fontsize=6)
    ax.set(ylabel="spectral radius", title="B  Two derivative constructions")

    ax = axes[1, 0]
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="#777777", lw=0.8, ls="--")
    multiplier_count = 0
    for row, item in zip(rows, artifacts):
        for dt_label, marker in (("base", "o"), ("half", "x")):
            variation = item.get("variational", {}).get(dt_label)
            if variation is None:
                continue
            for method in DERIVATIVE_LABELS:
                values_complex = np.asarray(variation["multipliers"][method], dtype=complex)
                ax.scatter(
                    values_complex.real,
                    values_complex.imag,
                    s=18,
                    marker=marker,
                    color=colors[float(row["alpha_G"])],
                    alpha=0.55,
                )
                multiplier_count += values_complex.size
    ax.axhline(0.0, color="#bbbbbb", lw=0.5)
    ax.axvline(0.0, color="#bbbbbb", lw=0.5)
    ax.set(aspect="equal", xlabel="Re(lambda)", ylabel="Im(lambda)", title="C  Transverse multipliers")

    ax = axes[1, 1]
    metric_names = ("rate_khz", "d_rate_d_mu_khz_per_mv", "d_rate_d_sigma_khz_per_mv")
    x = np.arange(len(metric_names))
    width = 0.18
    for index, (row, offset) in enumerate(zip(rows, (-0.20, 0.20))):
        for pop_index, pop in enumerate(("E", "I")):
            metrics = row.get("transfer_parity", {}).get("populations", {}).get(pop, {}).get("metrics", {})
            errors = [metrics.get(name, {}).get("maximum_relative_error_with_floor", np.nan) for name in metric_names]
            ax.bar(
                x + offset + (pop_index - 0.5) * width,
                errors,
                width=width,
                color=colors[float(row["alpha_G"])],
                alpha=0.55 if pop == "E" else 0.9,
                label=f"a{int(row['alpha_G'])} {pop}",
            )
    ax.axhline(0.05, color="#777777", ls="--", lw=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x, ("rate", "d/dmu", "d/dsigma"))
    ax.set(ylabel="max relative error (floored)", title="D  Direct exact-Siegert parity")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    png = figures / "stage0f_smooth_transfer_variational_certificate.png"
    pdf = figures / "stage0f_smooth_transfer_variational_certificate.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    readme = """### stage0f_smooth_transfer_variational_certificate.png

这张诊断图展示锁定的两个参数点在平滑 exact-table transfer 下重建的单周期轨道、chain-rule 与 centered-RHS 两套离散变分导数得到的谱半径、全部横向乘子，以及 spline 对 direct exact Siegert 值和导数的误差。它只修复 Stage 0E 的导数证书缺口，不是新的参数扫描，也不是空间发作图。

**关注点**：先看 exact transfer parity 和 smooth-vs-LUT orbit parity，再看两套导数及 dt/2 是否给出一致且远离单位圆的乘子；任一前置门失败都不能称为稳定轨道证书。

### stage0f_smooth_transfer_variational_certificate.pdf

与 PNG 内容相同的矢量版本，便于放大检查靠近原点的乘子和两种导数之间的细小差异。

**关注点**：即使证书通过，Stage 1 和空间模拟仍保持关闭；该结果只说明 frozen homogeneous fast system 的局部横向稳定性。
"""
    _atomic_text(figures / "README.md", readme)
    metadata = {
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "multiplier_count_rendered": multiplier_count,
        "panels": {
            "A": "smooth exact-table periodic orbit",
            "B": "spectral radius for chain-rule and two centered-RHS maps",
            "C": "all transverse multipliers",
            "D": "direct exact-Siegert value and derivative parity",
        },
    }
    _atomic_json(output / "figure_metadata.json", metadata)
    return metadata


def _status_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Topic 4 Stage 0F 状态",
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
            f"- `z={row['z']:.2f}, alpha_G={row['alpha_G']:.0f}`: `{row['outcome']}`; "
            f"failed gates = `{','.join(row.get('failed_gates', [])) or 'none'}`"
        )
    lines.extend(
        [
            "",
            "本阶段只解决 Stage 0E 的 transfer derivative / Floquet 数值证书。即使通过，也不证明 slow entry/exit、空间 recruitment 或完整 SNN lifecycle。",
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
        raise SystemExit("Refusing authoritative Stage0F execution without --confirm-run")
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    output = ROOT / str(cfg["result_root"])
    if output.exists():
        raise RuntimeError(f"Stage0F output already exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = _verify_locked_inputs(cfg)
    lock = _write_execution_lock(output, config_path, inputs)
    transfer = _load_transfer(cfg)

    rows: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    all_transfer_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    for point in cfg["points"]:
        alpha = int(round(float(point["alpha_G"])))
        params = PoolParameters(
            float(point["z"]),
            float(point["alpha_G"]),
            float(cfg["model"]["w_ee_mult"]),
            float(cfg["model"]["ratio"]),
        )
        result, artifacts = run_point_certificate(
            params, transfer, _load_stage0e_point(cfg, alpha), cfg
        )
        rows.append(result)
        all_artifacts.append(artifacts)
        _save_point(output, result, artifacts)
        all_transfer_rows.extend(artifacts.get("transfer_parity_rows", []))
        for dt_label in ("base", "half"):
            summary = result.get(f"{dt_label}_variational_consistency", {})
            if not summary:
                continue
            derivative_rows.append(
                {
                    "z": result["z"],
                    "alpha_G": result["alpha_G"],
                    "dt_label": dt_label,
                    "pass": summary.get("pass"),
                    **summary.get("matrix_relative_differences", {}),
                    **{f"rho_{name}": value for name, value in summary.get("spectral_radii", {}).items()},
                    "spectral_radius_range": summary.get("spectral_radius_range"),
                    "transversality_per_ms": summary.get("transversality_per_ms"),
                }
            )

    _atomic_json(output / "parameter_point_outcomes.json", rows)
    _write_csv(output / "parameter_point_outcomes.csv", [_point_csv_row(row) for row in rows])
    _write_csv(output / "exact_transfer_parity.csv", all_transfer_rows)
    _write_csv(output / "derivative_consistency.csv", derivative_rows)
    figure_metadata = _plot(output, rows, all_artifacts)
    unchanged = _finalize_execution_lock(output, config_path, cfg, lock)
    wall_seconds = float(time.monotonic() - started)
    peak_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2)
    rss_pass = peak_rss_gib < float(cfg["resource_contract"]["max_memory_gib"])
    engineering_pass = bool(unchanged and rss_pass)
    summary = {
        "schema_version": "topic4_stage0f_smooth_transfer_variational_certificate.v1",
        "verdict": _overall_verdict(rows, engineering_pass),
        "engineering_pass": engineering_pass,
        "sources_and_inputs_unchanged": unchanged,
        "parameter_points": rows,
        "resource_usage": {
            "blas_threads": 1,
            "peak_rss_gib": peak_rss_gib,
            "rss_limit_gib": float(cfg["resource_contract"]["max_memory_gib"]),
            "rss_pass": rss_pass,
            "wall_seconds": wall_seconds,
        },
        "smooth_transfer": {
            "name": transfer.name,
            "domain": {
                "mu_min_mv": transfer.domain.mu_min_mv,
                "mu_max_mv": transfer.domain.mu_max_mv,
                "sigma_min_mv": transfer.domain.sigma_min_mv,
                "sigma_max_mv": transfer.domain.sigma_max_mv,
            },
            "mu_nodes": int(transfer.mu_axis.size),
            "sigma_nodes": int(transfer.sigma_axis.size),
            "fit_or_regularization": False,
            "extrapolation": False,
        },
        "figure_metadata": figure_metadata,
        "scope": {
            "stage1_open": False,
            "spatial_simulation_open": False,
            "parameter_search_performed": False,
            "equations_changed": False,
        },
    }
    _atomic_json(output / "stage0f_smooth_transfer_variational_summary.json", summary)
    _atomic_text(output / "STATUS.md", _status_text(summary))
    print(json.dumps({"verdict": summary["verdict"], "output": str(output), "peak_rss_gib": peak_rss_gib}, indent=2))


if __name__ == "__main__":
    main()
