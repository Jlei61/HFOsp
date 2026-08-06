#!/usr/bin/env python3
"""Run the cheap additive-current Poincare continuation for the MZ fast cycle."""

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

from src.topic4_mz_additive_orbit_continuation import (  # noqa: E402
    finite_difference_poincare,
    integrate_additive_return,
    predict_section_state,
    shoot_additive_cycle,
)
from src.topic4_spatial_slowfast_stage0c import (  # noqa: E402
    PoolParameters,
    equilibrium_state,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    ExtendedSiegertTransfer,
)
from src.topic4_spatial_slowfast_stage0e import SectionDefinition  # noqa: E402
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import (  # noqa: E402
    SmoothSiegertTransferV11,
)


DEFAULT_CONFIG = ROOT / "config/topic4_mz_additive_orbit_continuation.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    expected = cfg["input_sha256"]
    if set(expected) != {
        "transfer_path", "shooting_seed_path", "stage0f_summary_path",
        "entry_exit_summary_path",
    }:
        raise ValueError("input hash contract drifted")
    observed = {}
    for key, digest in expected.items():
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"missing locked input: {path}")
        observed[key] = _sha256(path)
        if observed[key] != digest:
            raise RuntimeError(
                f"locked input drift for {key}: expected {digest}, observed {observed[key]}"
            )
    return observed


def _load_inputs(cfg: dict) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(ROOT / cfg["transfer_path"], allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("source transfer did not assert no clipping")
        exact = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"],
            name=str(payload["transfer_name"]),
        )
    smooth_cfg = cfg["smooth_transfer"]
    smooth = SmoothSiegertTransferV11.from_extended(
        exact,
        domain=SmoothDomain(
            float(smooth_cfg["mu_min_mv"]), float(smooth_cfg["mu_max_mv"]),
            float(smooth_cfg["sigma_min_mv"]), float(smooth_cfg["sigma_max_mv"]),
        ),
        kx=int(smooth_cfg["spline_degree_mu"]),
        ky=int(smooth_cfg["spline_degree_sigma"]),
        smoothing=float(smooth_cfg["smoothing"]),
    )
    with np.load(ROOT / cfg["shooting_seed_path"], allow_pickle=False) as payload:
        seed = np.asarray(payload["base_state"][-1], dtype=float)
    stage0f = json.loads((ROOT / cfg["stage0f_summary_path"]).read_text(encoding="utf-8"))
    model = cfg["model"]
    point = next(
        row for row in stage0f["parameter_points"]
        if np.isclose(row["z"], model["z"]) and np.isclose(row["alpha_G"], model["alpha_G"])
    )
    scales = np.asarray(point["state_scales"], dtype=float)
    entry = json.loads((ROOT / cfg["entry_exit_summary_path"]).read_text(encoding="utf-8"))
    fold_a = float(entry["persistence_sensor_audit"]["A_fold_at_reference_z_mv"])
    fold_row = min(
        entry["fixed_point_fold_surface"], key=lambda row: abs(float(row["z"]) - float(model["z"]))
    )
    fold_state = equilibrium_state(
        (1e-3 * float(fold_row["rE_hz"]), 1e-3 * float(fold_row["rI_hz"]))
    )
    return exact, smooth, seed, scales, fold_state, fold_a


def _section(cfg: dict) -> SectionDefinition:
    section = cfg["section"]
    return SectionDefinition(
        index=int(section["state_index"]), level=float(section["level"]),
        direction=str(section["direction"]), min_return_ms=float(section["min_return_ms"]),
        max_return_ms=float(section["max_return_ms"]),
    ).validate()


def _model_params(cfg: dict) -> PoolParameters:
    model = cfg["model"]
    return PoolParameters(
        float(model["z"]), float(model["alpha_G"]),
        float(model["w_ee_mult"]), float(model["ratio"]),
    )


def _shoot(
    seed: np.ndarray,
    transfer: Any,
    additive: float,
    dt_ms: float,
    cfg: dict,
    scales: np.ndarray,
    section: SectionDefinition,
) -> dict[str, Any]:
    shooting = cfg["shooting"]
    return shoot_additive_cycle(
        seed,
        _model_params(cfg),
        transfer,
        additive,
        dt_ms=dt_ms,
        section=section,
        scales=scales,
        max_iterations=int(shooting["max_iterations"]),
        minimum_iterations=int(shooting["minimum_iterations"]),
        residual_tolerance=float(shooting["residual_tolerance"]),
        period_cv_tolerance=float(shooting["period_cv_tolerance"]),
    )


def _smooth_branch(
    smooth: Any,
    seed: np.ndarray,
    scales: np.ndarray,
    fold_state: np.ndarray,
    cfg: dict,
    section: SectionDefinition,
) -> tuple[list[dict], dict[float, np.ndarray], dict[str, np.ndarray]]:
    dt_ms = float(cfg["shooting"]["primary_dt_ms"])
    rows: list[dict] = []
    accepted: dict[float, np.ndarray] = {}
    traces: dict[str, np.ndarray] = {}
    previous_state = seed.copy()
    older_state = None
    previous_a = 0.0
    older_a = None
    consecutive_failures = 0
    for additive in map(float, cfg["additive_grid_mv"]):
        candidate = predict_section_state(
            previous_state, older_state, additive, previous_a, older_a, section
        )
        try:
            result = _shoot(candidate, smooth, additive, dt_ms, cfg, scales, section)
        except ValueError:
            result = {"accepted": False, "reason": "secant_predictor_outside_natural_bounds"}
        used_fallback = False
        if not result.get("accepted", False) and accepted:
            used_fallback = True
            result = _shoot(previous_state, smooth, additive, dt_ms, cfg, scales, section)
        row = {
            "transfer": "smooth",
            "dt_ms": dt_ms,
            "additive_mv": additive,
            "accepted": bool(result.get("accepted", False)),
            "reason": result.get("reason"),
            "used_previous_state_fallback": used_fallback,
        }
        if result.get("accepted", False):
            fixed = np.asarray(result["fixed_state"], dtype=float)
            cycle = integrate_additive_return(
                fixed,
                _model_params(cfg),
                smooth,
                additive,
                dt_ms=dt_ms,
                section=section,
                record_trace=True,
                distance_target=fold_state,
                distance_scales=scales,
            )
            if not cycle.valid or cycle.trace is None:
                raise RuntimeError("accepted shooting point failed independent cycle trace")
            wave = np.asarray(cycle.trace["state"], dtype=float)
            local_scales = np.maximum(np.ptp(wave, axis=0), 1e-6)
            row.update(
                period_ms=float(cycle.return_time_ms),
                p_closure=float(result["p_closure"]),
                p2_closure=float(result["p2_closure"]),
                final_shooting_residual=float(np.asarray(result["residual"])[-1]),
                iterations=int(np.asarray(result["residual"]).size),
                peak_rE_hz=float(cycle.peak_r_e_hz),
                over_100hz_count=int(cycle.over_100hz_count),
                rE_amplitude_hz=float(1000.0 * np.ptp(wave[:, 0])),
                minimum_fold_state_distance=float(cycle.minimum_target_distance),
                minimum_transversality_per_ms=float(result["transversality_per_ms"][-1]),
            )
            accepted[additive] = fixed
            label = f"A_{additive:.5f}".replace(".", "p")
            traces[f"{label}_time_ms"] = np.asarray(cycle.trace["time_ms"], dtype=np.float32)
            traces[f"{label}_state"] = np.asarray(wave, dtype=np.float32)
            traces[f"{label}_local_scales"] = np.asarray(local_scales, dtype=np.float32)
            older_state, older_a = previous_state, previous_a
            previous_state, previous_a = fixed, additive
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        rows.append(row)
        if consecutive_failures >= int(cfg["shooting"]["max_consecutive_failures"]):
            break
    return rows, accepted, traces


def _exact_checkpoints(
    exact: Any,
    accepted: dict[float, np.ndarray],
    scales: np.ndarray,
    cfg: dict,
    section: SectionDefinition,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    states: dict[str, np.ndarray] = {}
    for additive in map(float, cfg["exact_transfer_checkpoints_mv"]):
        if additive not in accepted:
            rows.append({
                "transfer": "exact", "additive_mv": additive, "dt_ms": None,
                "accepted": False, "reason": "smooth_branch_point_unavailable",
            })
            continue
        base = _shoot(
            accepted[additive], exact, additive,
            float(cfg["shooting"]["primary_dt_ms"]), cfg, scales, section,
        )
        seed_half = np.asarray(base.get("fixed_state", accepted[additive]), dtype=float)
        half = _shoot(
            seed_half, exact, additive,
            float(cfg["shooting"]["half_dt_ms"]), cfg, scales, section,
        )
        for label, result, dt_ms in (
            ("base", base, float(cfg["shooting"]["primary_dt_ms"])),
            ("half", half, float(cfg["shooting"]["half_dt_ms"])),
        ):
            row = {
                "transfer": "exact", "resolution": label, "additive_mv": additive,
                "dt_ms": dt_ms, "accepted": bool(result.get("accepted", False)),
                "reason": result.get("reason"),
            }
            if result.get("accepted", False):
                row.update(
                    period_ms=float(result["validated_period_ms"]),
                    p_closure=float(result["p_closure"]),
                    p2_closure=float(result["p2_closure"]),
                    peak_rE_hz=float(result["peak_r_e_hz"]),
                    over_100hz_count=int(result["over_100hz_count"]),
                )
                states[f"A_{additive:.5f}_{label}".replace(".", "p")] = np.asarray(
                    result["fixed_state"], dtype=np.float32
                )
            rows.append(row)
    return rows, states


def _derivative_rows(
    smooth: Any,
    accepted: dict[float, np.ndarray],
    scales: np.ndarray,
    cfg: dict,
    section: SectionDefinition,
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    rows: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    summaries: list[dict] = []
    derivative = cfg["poincare_derivative"]
    for additive in map(float, derivative["points_mv"]):
        if additive not in accepted:
            summaries.append({
                "additive_mv": additive, "status": "cycle_point_unavailable",
            })
            continue
        point_results = []
        for epsilon in map(float, derivative["epsilon_relative"]):
            result = finite_difference_poincare(
                accepted[additive], _model_params(cfg), smooth, additive,
                dt_ms=float(cfg["shooting"]["primary_dt_ms"]),
                section=section, scales=scales, epsilon_relative=epsilon,
            )
            row = {
                "additive_mv": additive, "epsilon_relative": epsilon,
                "valid": bool(result.get("valid", False)), "reason": result.get("reason"),
            }
            if result.get("valid", False):
                row.update(
                    spectral_radius=float(result["spectral_radius"]),
                    nearest_plus_one_distance=float(result["nearest_plus_one_distance"]),
                    minimum_probe_transversality_per_ms=float(
                        result["minimum_probe_transversality_per_ms"]
                    ),
                )
                key = f"A_{additive:.5f}_eps_{epsilon:.0e}".replace(".", "p").replace("-", "m")
                matrices[f"{key}_matrix"] = np.asarray(result["matrix"])
                matrices[f"{key}_multipliers"] = np.asarray(result["multipliers"])
                point_results.append(result)
            rows.append(row)
        if len(point_results) >= 2:
            differences = [
                float(
                    np.linalg.norm(left["matrix"] - right["matrix"])
                    / max(np.linalg.norm(right["matrix"]), 1e-12)
                )
                for left, right in zip(point_results[:-1], point_results[1:])
            ]
            radii = [float(result["spectral_radius"]) for result in point_results]
            stable = all(radius < 1.0 for radius in radii)
            platform = bool(
                max(differences) <= float(derivative["matrix_relative_difference_max"])
                and max(radii) - min(radii) <= float(derivative["spectral_radius_range_max"])
            )
            summaries.append({
                "additive_mv": additive,
                "status": "derivative_platform" if platform else "derivative_unresolved",
                "stable_at_all_eps": stable,
                "matrix_relative_differences": differences,
                "spectral_radii": radii,
                "spectral_radius_range": max(radii) - min(radii),
                "minimum_nearest_plus_one_distance": min(
                    float(result["nearest_plus_one_distance"]) for result in point_results
                ),
            })
        else:
            summaries.append({
                "additive_mv": additive, "status": "insufficient_valid_derivative_rows",
            })
    return rows, matrices, summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _plot(
    figures: Path,
    smooth_rows: list[dict],
    exact_rows: list[dict],
    derivative_summary: list[dict],
    fold_a: float,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    accepted = [row for row in smooth_rows if row["accepted"]]
    failed = [row for row in smooth_rows if not row["accepted"]]
    a = np.asarray([row["additive_mv"] for row in accepted])
    period = np.asarray([row["period_ms"] for row in accepted])
    ax_a.plot(a, period, "o-", color="#7b3294", lw=1.5, ms=4, label="smooth stable branch")
    for resolution, marker, color in (("base", "s", "#d6604d"), ("half", "x", "#2166ac")):
        rows = [row for row in exact_rows if row.get("resolution") == resolution and row["accepted"]]
        if rows:
            ax_a.scatter(
                [row["additive_mv"] for row in rows], [row["period_ms"] for row in rows],
                marker=marker, s=28, color=color, label=f"exact {resolution}", zorder=4,
            )
    ax_a.axvline(fold_a, color="#d8902f", ls=":", lw=1.4, label="fixed-point fold")
    ax_a.set_yscale("log")
    ax_a.set(xlabel="additive E current A (mV)", ylabel="cycle period (ms)",
             title="A  Stable-cycle continuation")
    ax_a.legend(frameon=False, fontsize=7.5)

    ax_b.plot(a, [row["rE_amplitude_hz"] for row in accepted], "o-", color="#b2182b",
              lw=1.4, ms=4, label="rE amplitude")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(a, [row["minimum_fold_state_distance"] for row in accepted], "o-",
               color="#2166ac", lw=1.3, ms=3.5, label="distance to fold state")
    ax_b.axvline(fold_a, color="#d8902f", ls=":", lw=1.2)
    ax_b.set(xlabel="additive E current A (mV)", ylabel="rE peak-to-peak (Hz)",
             title="B  Orbit approaches the low-state bottleneck")
    ax_b2.set_ylabel("minimum scaled distance")
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax_b2.legend(frameon=False, fontsize=7.5, loc="upper right")

    valid_derivative = [row for row in derivative_summary if "spectral_radii" in row]
    derivative_labels: set[str] = set()
    for row in valid_derivative:
        x = float(row["additive_mv"])
        resolved = row.get("status") == "derivative_platform"
        label = "FD platform" if resolved else "FD unresolved"
        ax_c.scatter([x] * len(row["spectral_radii"]), row["spectral_radii"],
                     color="#1b7837" if resolved else "#e08214",
                     marker="o" if resolved else "D", s=28,
                     label=label if label not in derivative_labels else None)
        derivative_labels.add(label)
    ax_c.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax_c.axvline(fold_a, color="#d8902f", ls=":", lw=1.2)
    ax_c.set_yscale("log")
    ax_c.set_ylim(3e-5, 2.0)
    ax_c.set(xlabel="additive E current A (mV)", ylabel="transverse spectral radius",
             title="C  No multiplier approach to +1")
    ax_c.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax_d.plot(a, [row["peak_rE_hz"] for row in accepted], "o-", color="#c74343",
              lw=1.4, ms=4, label="smooth peak")
    ax_d.axhline(100.0, color="0.35", ls="--", lw=1.0, label="100-Hz envelope")
    if failed:
        ax_d.scatter([row["additive_mv"] for row in failed], [2.0] * len(failed), marker="v",
                     color="black", s=32, label="no accepted return")
    ax_d.axvline(fold_a, color="#d8902f", ls=":", lw=1.2)
    ax_d.set(xlabel="additive E current A (mV)", ylabel="cycle peak rE (Hz)",
             title="D  Physical ceiling and return loss")
    ax_d.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Additive-current Poincaré continuation of the MZ fast cycle",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.01,
        "Stable directed-return branch only; failure does not exclude an unstable or disconnected cycle.",
        ha="center", fontsize=8.2, color="#7f0000",
    )
    stem = figures / "mz_additive_orbit_continuation"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_sha256 = _validate_inputs(cfg)
    exact, smooth, seed, scales, fold_state, fold_a = _load_inputs(cfg)
    section = _section(cfg)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    smooth_rows, accepted, trace_payload = _smooth_branch(
        smooth, seed, scales, fold_state, cfg, section
    )
    exact_rows, exact_states = _exact_checkpoints(
        exact, accepted, scales, cfg, section
    )
    derivative_rows, derivative_payload, derivative_summary = _derivative_rows(
        smooth, accepted, scales, cfg, section
    )
    _write_csv(output / "stable_cycle_branch.csv", smooth_rows)
    _write_csv(output / "exact_transfer_checkpoints.csv", exact_rows)
    _write_csv(output / "poincare_derivative_ladder.csv", derivative_rows)
    np.savez_compressed(output / "cycle_states_and_traces.npz", **trace_payload, **exact_states)
    np.savez_compressed(output / "poincare_matrices.npz", **derivative_payload)
    figure = _plot(figures, smooth_rows, exact_rows, derivative_summary, fold_a)

    accepted_a = [float(row["additive_mv"]) for row in smooth_rows if row["accepted"]]
    failed_a = [float(row["additive_mv"]) for row in smooth_rows if not row["accepted"]]
    last_accepted = max(accepted_a) if accepted_a else None
    first_failed = min((value for value in failed_a if last_accepted is None or value > last_accepted),
                       default=None)
    derivative_pass = all(
        row.get("status") == "derivative_platform" for row in derivative_summary
    )
    exact_pairs = {}
    for additive in map(float, cfg["exact_transfer_checkpoints_mv"]):
        rows = [row for row in exact_rows if row["additive_mv"] == additive and row["accepted"]]
        exact_pairs[str(additive)] = {
            "base_and_half_accepted": len(rows) == 2,
            "period_difference_ms": (
                abs(float(rows[0]["period_ms"]) - float(rows[1]["period_ms"]))
                if len(rows) == 2 else None
            ),
            "peak_max_hz": max((float(row["peak_rE_hz"]) for row in rows), default=None),
        }
    near_rows = [
        row for row in smooth_rows
        if row["accepted"] and float(row["additive_mv"]) >= 0.28
        and float(row["additive_mv"]) < fold_a
    ]
    if len(near_rows) >= 4:
        inverse_sqrt_distance = np.asarray([
            1.0 / np.sqrt(fold_a - float(row["additive_mv"])) for row in near_rows
        ])
        periods = np.asarray([float(row["period_ms"]) for row in near_rows])
        design = np.column_stack([np.ones(inverse_sqrt_distance.size), inverse_sqrt_distance])
        coefficients = np.linalg.lstsq(design, periods, rcond=None)[0]
        fitted = design @ coefficients
        residual = float(np.sum((periods - fitted) ** 2))
        total = float(np.sum((periods - periods.mean()) ** 2))
        inverse_sqrt_r2 = float(1.0 - residual / total) if total > 0.0 else None
    else:
        coefficients = np.asarray([np.nan, np.nan])
        inverse_sqrt_r2 = None
    accepted_rows = [row for row in smooth_rows if row["accepted"]]
    boundary_evidence = {
        "interpretation": "strong_SNIC_like_candidate_formal_label_open",
        "inverse_sqrt_period_fit_A_ge_0p28": {
            "n_points": len(near_rows),
            "intercept_ms": float(coefficients[0]),
            "coefficient_ms_sqrt_mv": float(coefficients[1]),
            "r_squared": inverse_sqrt_r2,
        },
        "fold_minus_last_accepted_A_mv": (
            float(fold_a - last_accepted) if last_accepted is not None else None
        ),
        "fold_minus_first_failed_A_mv": (
            float(fold_a - first_failed) if first_failed is not None else None
        ),
        "period_growth_last_vs_A0": (
            float(accepted_rows[-1]["period_ms"] / accepted_rows[0]["period_ms"])
            if accepted_rows else None
        ),
        "fold_state_distance_contraction_A0_vs_last": (
            float(
                accepted_rows[0]["minimum_fold_state_distance"]
                / accepted_rows[-1]["minimum_fold_state_distance"]
            ) if accepted_rows else None
        ),
        "reason_label_remains_open": (
            "the connected attracting branch is localized, but unstable/disconnected cycles "
            "and exact-transfer returns inside the final strip are not excluded"
        ),
    }
    summary = {
        "status": "stable_cycle_boundary_localized_formal_label_open",
        "model_contract": {
            "fast_system": "locked Stage0C nine-state system",
            "continuation_parameter": "frozen additive E current A",
            "section": "upward S_G=0.15 event-restarted map",
            "parallel_line_exclusions": [
                "no E-E weight/kernel/delay changes", "no E-E relay",
                "no conductance membrane", "no second recurrent-E divisor",
            ],
        },
        "fixed_point_fold_A_mv": fold_a,
        "stable_cycle_strip": {
            "last_accepted_A_mv": last_accepted,
            "first_failed_A_mv": first_failed,
            "interpretation": "stable directed-return branch, not unstable-cycle exclusion",
        },
        "boundary_evidence": boundary_evidence,
        "derivative_summary": derivative_summary,
        "derivative_platform_all_registered_points": derivative_pass,
        "exact_base_half_checkpoints": exact_pairs,
        "branch_rows": smooth_rows,
        "claim_boundary": [
            "return-map continuation follows only the attracting connected branch",
            "failure does not exclude an unstable or disconnected cycle",
            "pseudo-arclength remains conditional on a finite-period +1 multiplier approach",
            "SNIC requires joint period divergence, fold alignment, and orbit-to-fold-state approach",
            "no slow Z-p-m lifecycle, spatial front, or SNN recovery is claimed",
        ],
        "resource_contract": cfg["resource_contract"],
        "input_sha256": input_sha256,
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "branch_csv": str((output / "stable_cycle_branch.csv").relative_to(ROOT)),
            "exact_csv": str((output / "exact_transfer_checkpoints.csv").relative_to(ROOT)),
            "derivative_csv": str((output / "poincare_derivative_ladder.csv").relative_to(ROOT)),
            "cycle_npz": str((output / "cycle_states_and_traces.npz").relative_to(ROOT)),
            "poincare_npz": str((output / "poincare_matrices.npz").relative_to(ROOT)),
        },
        "config": str(config_path.relative_to(ROOT)),
    }
    (output / "orbit_continuation_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_additive_orbit_continuation.png / .pdf\n\n"
        "这张四面板图沿冻结 additive E-current A 追踪当前 MZ 九维快系统的稳定周期："
        "A 显示周期是否在 fixed-point fold 附近发散，B 检查轨道是否接近 fold state，"
        "C 显示三档有限差分的横向 Poincaré spectral radius，D 单独报告 peak-rate ceiling 和 return loss。"
        "smooth branch 用于可微 continuation，选定点用 exact transfer 与 base/half dt 复核。\n\n"
        "**关注点**：本图只追踪与 A=0 周期连通且能由 directed return-map iteration 吸引的稳定分支；"
        "没有 accepted return 不能自动排除 unstable/disconnected cycle。只有 period、fold alignment、"
        "orbit-to-fold distance 和 multiplier 共同支持时，才可进一步命名具体分岔。\n",
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
        "stable_cycle_strip": summary["stable_cycle_strip"],
        "derivative_platform": summary["derivative_platform_all_registered_points"],
        "figure": summary["artifacts"]["figure"],
    }, indent=2))


if __name__ == "__main__":
    main()
