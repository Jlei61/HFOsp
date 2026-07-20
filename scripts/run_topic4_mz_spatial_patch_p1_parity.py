#!/usr/bin/env python3
"""Prove P=1/uniform-P parity before enabling spatial slow dynamics."""

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

from src.topic4_mz_additive_orbit_continuation import integrate_additive_return  # noqa: E402
from src.topic4_mz_entry_exit_nullclines import additive_rhs  # noqa: E402
from src.topic4_mz_spatial_patch import (  # noqa: E402
    LOCAL_FIELDS,
    PatchKernels,
    PatchParameters,
    patch_rhs,
    patch_rhs_to_stage0c,
    patch_to_stage0c_state,
    stage0c_to_patch_state,
    uniform_patch_state,
    unpack_patch_state,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer  # noqa: E402
from src.topic4_spatial_slowfast_stage0e import SectionDefinition  # noqa: E402
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_patch_p1_parity.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    keys = ("transfer_path", "orbit_summary_path", "orbit_cycle_path", "entry_exit_summary_path")
    if set(cfg["input_sha256"]) != set(keys):
        raise ValueError(f"input_sha256 must lock exactly {keys}")
    observed: dict[str, str] = {}
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"missing locked input: {path}")
        observed[key] = _sha256(path)
        if observed[key] != str(cfg["input_sha256"][key]):
            raise RuntimeError(f"locked input drift for {key}: {observed[key]}")
    return observed


def _load_transfer(cfg: dict) -> Any:
    with np.load(ROOT / cfg["transfer_path"], allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("source transfer did not assert no clipping")
        exact = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"],
            name=str(payload["transfer_name"]),
        )
    smooth = cfg["smooth_transfer"]
    return SmoothSiegertTransferV11.from_extended(
        exact,
        domain=SmoothDomain(
            float(smooth["mu_min_mv"]), float(smooth["mu_max_mv"]),
            float(smooth["sigma_min_mv"]), float(smooth["sigma_max_mv"]),
        ),
        kx=int(smooth["spline_degree_mu"]),
        ky=int(smooth["spline_degree_sigma"]),
        smoothing=float(smooth["smoothing"]),
    )


def _model(cfg: dict) -> tuple[PoolParameters, PatchParameters]:
    model = cfg["model"]
    pool = PoolParameters(
        float(model["z"]), float(model["alpha_G"]),
        float(model["w_ee_mult"]), float(model["ratio"]),
    )
    patch = PatchParameters(
        alpha_g=float(model["alpha_G"]),
        w_ee_mult=float(model["w_ee_mult"]),
        ratio=float(model["ratio"]),
        additive_max_mv=float(model["additive_max_mv"]),
        pool_p=float(model["pool_p"]),
    )
    return pool, patch


def _uniform_kernels(n_patches: int) -> PatchKernels:
    """One fixed constant-preserving test operator; not a P=2 calibration."""

    if n_patches == 1:
        return PatchKernels.identity(1)
    k_ee = np.zeros((n_patches, n_patches), dtype=float)
    for index in range(n_patches):
        k_ee[index, index] = 0.4
        k_ee[index, (index - 1) % n_patches] += 0.3
        k_ee[index, (index + 1) % n_patches] += 0.3
    k_i = np.full((n_patches, n_patches), 1.0 / n_patches)
    return PatchKernels(k_ee, k_i).validate()


def _trace_key(additive_mv: float) -> str:
    return f"A_{float(additive_mv):.5f}".replace(".", "p")


def _rhs_parity(
    cfg: dict,
    transfer: Any,
    orbit_payload: Any,
    pool_params: PoolParameters,
    patch_params: PatchParameters,
) -> list[dict]:
    rows: list[dict] = []
    n_uniform = int(cfg["pointwise_rhs"]["uniform_patch_count"])
    uniform_kernels = _uniform_kernels(n_uniform)
    off_manifold = np.asarray(
        [0.080, 0.150, 0.003, 0.008, 0.011, 0.019, 0.071, 0.42, 0.73],
        dtype=float,
    )
    for additive in map(float, cfg["pointwise_rhs"]["additive_mv"]):
        key = _trace_key(additive)
        if f"{key}_state" not in orbit_payload.files:
            raise KeyError(f"missing locked orbit trace {key}")
        probes = {
            "cycle_section": np.asarray(orbit_payload[f"{key}_state"][0], dtype=float),
            "off_manifold": off_manifold,
        }
        for probe, stage in probes.items():
            expected = additive_rhs(stage, pool_params, transfer, additive)
            p1 = stage0c_to_patch_state(
                stage,
                z=pool_params.z,
                additive_mv=additive,
                parameters=patch_params,
            )
            observed = patch_rhs_to_stage0c(
                patch_rhs(p1, PatchKernels.identity(1), patch_params, transfer)
            )
            uniform = uniform_patch_state(
                stage,
                n_patches=n_uniform,
                z=pool_params.z,
                additive_mv=additive,
                parameters=patch_params,
            )
            uniform_rhs = patch_rhs(uniform, uniform_kernels, patch_params, transfer)
            local, d_mu_g, d_s_g = unpack_patch_state(uniform_rhs, n_uniform)
            uniform_error = max(
                *(float(np.max(np.abs(local[name] - expected[index])))
                  for index, name in enumerate(LOCAL_FIELDS[:7])),
                abs(float(d_mu_g) - float(expected[7])),
                abs(float(d_s_g) - float(expected[8])),
            )
            slow_zero = max(
                float(np.max(np.abs(local["z"]))),
                float(np.max(np.abs(local["p"]))),
                float(np.max(np.abs(local["m"]))),
            )
            rows.append({
                "additive_mv": additive,
                "probe": probe,
                "p1_rhs_max_abs_error": float(np.max(np.abs(observed - expected))),
                "uniform_patch_count": n_uniform,
                "uniform_rhs_max_abs_error": uniform_error,
                "frozen_slow_rhs_max_abs": slow_zero,
            })
    return rows


def _integrate_patch_return(
    initial_patch: np.ndarray,
    *,
    kernels: PatchKernels,
    patch_params: PatchParameters,
    transfer: Any,
    dt_ms: float,
    min_return_ms: float,
    max_return_ms: float,
    section_level: float,
) -> dict[str, Any]:
    state = np.asarray(initial_patch, dtype=float).copy()
    if state.shape != (12,) or not np.isclose(state[-1], section_level, atol=1.0e-8):
        raise ValueError("P=1 return seed must lie on the shared S_G section")
    n_steps = int(np.ceil(max_return_ms / dt_ms)) + 1
    below_seen = False
    time_trace: list[float] = []
    stage_trace: list[np.ndarray] = []
    for step in range(n_steps):
        time = float(step) * dt_ms
        time_trace.append(time)
        stage_trace.append(patch_to_stage0c_state(state))
        rhs = patch_rhs(state, kernels, patch_params, transfer)
        next_state = state + dt_ms * rhs
        h0 = float(state[-1] - section_level)
        h1 = float(next_state[-1] - section_level)
        if h0 < 0.0:
            below_seen = True
        if below_seen and h0 < 0.0 <= h1:
            fraction = -h0 / (h1 - h0)
            crossing_time = time + fraction * dt_ms
            crossing = state + fraction * (next_state - state)
            time_trace.append(crossing_time)
            stage_trace.append(patch_to_stage0c_state(crossing))
            clean = min_return_ms <= crossing_time <= max_return_ms
            return {
                "status": "clean_return" if clean else "unclean_crossing",
                "return_time_ms": crossing_time,
                "crossing_patch_state": crossing,
                "time_ms": np.asarray(time_trace),
                "stage_state": np.asarray(stage_trace),
            }
        state = next_state
    return {
        "status": "no_return_within_locked_window",
        "return_time_ms": None,
        "crossing_patch_state": None,
        "time_ms": np.asarray(time_trace),
        "stage_state": np.asarray(stage_trace),
    }


def _return_parity(
    cfg: dict,
    transfer: Any,
    orbit_payload: Any,
    pool_params: PoolParameters,
    patch_params: PatchParameters,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    contract = cfg["return_map"]
    section = SectionDefinition(
        index=8,
        level=float(contract["section_level_SG"]),
        direction="upward",
        min_return_ms=float(contract["min_return_ms"]),
        max_return_ms=float(contract["max_return_ms"]),
    ).validate()
    rows: list[dict] = []
    traces: dict[str, np.ndarray] = {}
    for checkpoint in contract["checkpoints"]:
        additive = float(checkpoint["additive_mv"])
        key = str(checkpoint["trace_key"])
        registered_status = str(checkpoint.get("expected_status", "clean_return"))
        initial_stage = np.asarray(orbit_payload[f"{key}_state"][0], dtype=float)
        for dt_ms in map(float, contract["dt_ms"]):
            expected = integrate_additive_return(
                initial_stage,
                pool_params,
                transfer,
                additive,
                dt_ms=dt_ms,
                section=section,
                record_trace=True,
            )
            initial_patch = stage0c_to_patch_state(
                initial_stage,
                z=pool_params.z,
                additive_mv=additive,
                parameters=patch_params,
            )
            observed = _integrate_patch_return(
                initial_patch,
                kernels=PatchKernels.identity(1),
                patch_params=patch_params,
                transfer=transfer,
                dt_ms=dt_ms,
                min_return_ms=section.min_return_ms,
                max_return_ms=section.max_return_ms,
                section_level=section.level,
            )
            if expected.trace is None:
                raise RuntimeError(f"Stage-0C return did not record a trace at A={additive}, dt={dt_ms}")
            if expected.status != registered_status:
                raise RuntimeError(
                    f"locked Stage-0C status drift at A={additive}, dt={dt_ms}: "
                    f"expected {registered_status}, observed {expected.status}"
                )
            expected_time = np.asarray(expected.trace["time_ms"], dtype=float)
            expected_state = np.asarray(expected.trace["state"], dtype=float)
            observed_time = np.asarray(observed["time_ms"], dtype=float)
            observed_state = np.asarray(observed["stage_state"], dtype=float)
            same_shape = expected_time.shape == observed_time.shape and expected_state.shape == observed_state.shape
            trace_time_error = (
                float(np.max(np.abs(expected_time - observed_time))) if same_shape else np.inf
            )
            trace_state_error = (
                float(np.max(np.abs(expected_state - observed_state))) if same_shape else np.inf
            )
            if expected.valid:
                if observed["crossing_patch_state"] is None or expected.crossing_state is None:
                    raise RuntimeError("clean return is missing a crossing state")
                crossing_state = patch_to_stage0c_state(observed["crossing_patch_state"])
                crossing_error = float(np.max(np.abs(crossing_state - expected.crossing_state)))
                period_error = abs(float(observed["return_time_ms"]) - float(expected.return_time_ms))
            else:
                crossing_error = None
                period_error = None
            label = f"A_{additive:.3f}_dt_{dt_ms:.4f}".replace(".", "p")
            traces[f"{label}_time_ms"] = observed_time.astype(np.float32)
            traces[f"{label}_expected_state"] = expected_state.astype(np.float32)
            traces[f"{label}_patch_state_projected"] = observed_state.astype(np.float32)
            rows.append({
                "additive_mv": additive,
                "dt_ms": dt_ms,
                "registered_status": registered_status,
                "stage0c_status": expected.status,
                "patch_status": observed["status"],
                "status_match": observed["status"] == expected.status,
                "expected_period_ms": (
                    float(expected.return_time_ms) if expected.return_time_ms is not None else None
                ),
                "patch_period_ms": (
                    float(observed["return_time_ms"]) if observed["return_time_ms"] is not None else None
                ),
                "period_abs_error_ms": period_error,
                "crossing_state_max_abs_error": crossing_error,
                "trace_time_max_abs_error_ms": trace_time_error,
                "trace_state_max_abs_error": trace_state_error,
                "trace_shapes_equal": same_shape,
                "n_trace_states": int(observed_time.size),
            })
    return rows, traces


def _fold_parity(
    cfg: dict,
    transfer: Any,
    entry_summary: dict,
    patch_params: PatchParameters,
) -> list[dict]:
    rows: list[dict] = []
    maximum_a = float(cfg["fold_surface"]["maximum_additive_mv"])
    model = cfg["model"]
    for fold in entry_summary["fixed_point_fold_surface"]:
        additive = float(fold["additive_mv"])
        if additive > maximum_a:
            continue
        z = float(fold["z"])
        stage = equilibrium_state((1e-3 * float(fold["rE_hz"]), 1e-3 * float(fold["rI_hz"])))
        pool = PoolParameters(z, float(model["alpha_G"]), float(model["w_ee_mult"]), float(model["ratio"]))
        expected = additive_rhs(stage, pool, transfer, additive)
        patch = stage0c_to_patch_state(
            stage, z=z, additive_mv=additive, parameters=patch_params
        )
        observed = patch_rhs_to_stage0c(
            patch_rhs(patch, PatchKernels.identity(1), patch_params, transfer)
        )
        rows.append({
            "additive_mv": additive,
            "z": z,
            "source_fold_residual_inf": float(fold["residual_inf"]),
            "smooth_rate_residual_inf": float(max(abs(expected[0]), abs(expected[1]))),
            "p1_fold_rhs_max_abs_error": float(np.max(np.abs(observed - expected))),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    figures: Path,
    rhs_rows: list[dict],
    return_rows: list[dict],
    fold_rows: list[dict],
    traces: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    labels = [f"A={row['additive_mv']:.3f}\n{row['probe']}" for row in rhs_rows]
    x = np.arange(len(rhs_rows))
    ax_a.semilogy(x, np.maximum([row["p1_rhs_max_abs_error"] for row in rhs_rows], 1e-20),
                  "o-", color="#2166ac", label="P=1")
    ax_a.semilogy(x, np.maximum([row["uniform_rhs_max_abs_error"] for row in rhs_rows], 1e-20),
                  "s--", color="#1b7837", label="uniform P=4")
    ax_a.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax_a.set(ylabel="max |RHS error|", title="A  Pointwise vector-field parity")
    ax_a.text(0.02, 0.05, "all plotted values are exact zero\n(display floor = 1e-20)",
              transform=ax_a.transAxes, fontsize=7.5, color="0.35")
    ax_a.legend(frameon=False, fontsize=8)

    clean_rows = [row for row in return_rows if row["registered_status"] == "clean_return"]
    expected = np.asarray([row["expected_period_ms"] for row in clean_rows])
    observed = np.asarray([row["patch_period_ms"] for row in clean_rows])
    colors = np.asarray([row["additive_mv"] for row in clean_rows])
    scatter = ax_b.scatter(expected, observed, c=colors, cmap="viridis", s=48, edgecolor="black", lw=0.4)
    limits = [0.98 * min(expected.min(), observed.min()), 1.02 * max(expected.max(), observed.max())]
    ax_b.plot(limits, limits, color="0.4", ls="--", lw=1)
    ax_b.set(xlabel="Stage-0C return period (ms)", ylabel="P=1 return period (ms)",
             title="B  Directed-return period parity", xlim=limits, ylim=limits)
    ax_b.text(0.03, 0.94, "A=.31648: both no return within 12 s",
              transform=ax_b.transAxes, va="top", fontsize=8, color="#7f0000")
    fig.colorbar(scatter, ax=ax_b, label="additive A (mV)")

    representative = next(row for row in return_rows if np.isclose(row["additive_mv"], 0.3) and np.isclose(row["dt_ms"], 0.125))
    label = f"A_{representative['additive_mv']:.3f}_dt_{representative['dt_ms']:.4f}".replace(".", "p")
    time = traces[f"{label}_time_ms"]
    expected_state = traces[f"{label}_expected_state"]
    patch_state = traces[f"{label}_patch_state_projected"]
    ax_c.plot(time, 1000.0 * expected_state[:, 0], color="#762a83", lw=1.2, label="Stage 0C rE")
    ax_c.plot(time, 1000.0 * patch_state[:, 0], color="#e08214", lw=0.8, ls="--", label="P=1 rE")
    ax_c.set(xlabel="time (ms)", ylabel="rE (Hz)", title="C  A=0.30 cycle overlay")
    ax_c.legend(frameon=False, fontsize=8)

    fold_a = np.asarray([row["additive_mv"] for row in fold_rows])
    fold_error = np.maximum([row["p1_fold_rhs_max_abs_error"] for row in fold_rows], 1e-20)
    ax_d.semilogy(fold_a, fold_error, "o-", color="#2166ac", label="P=1 vs Stage 0C")
    ax_d.set(xlabel="fold-surface additive A (mV)", ylabel="max |RHS error|",
             title="D  Fold-surface vector-field identity")
    ax_d.text(0.02, 0.05, "exact zero (display floor = 1e-20)",
              transform=ax_d.transAxes, fontsize=7.5, color="0.35")
    ax_d.legend(frameon=False, fontsize=8)

    fig.suptitle("P=1 gate: the spatial scaffold is the same fast system",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.012,
        "Local z/p/m derivatives are frozen at this gate; one shared muG/SG pair is retained. No P=2 recruitment result is claimed.",
        ha="center", fontsize=8.0, color="#7f0000",
    )
    stem = figures / "mz_spatial_patch_p1_parity"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed_hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    pool_params, patch_params = _model(cfg)
    orbit_summary = json.loads((ROOT / cfg["orbit_summary_path"]).read_text(encoding="utf-8"))
    if orbit_summary["model_contract"]["fast_system"] != "locked Stage0C nine-state system":
        raise RuntimeError("upstream orbit is not the locked Stage-0C system")
    entry_summary = json.loads((ROOT / cfg["entry_exit_summary_path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as orbit_payload:
        rhs_rows = _rhs_parity(cfg, transfer, orbit_payload, pool_params, patch_params)
        return_rows, traces = _return_parity(cfg, transfer, orbit_payload, pool_params, patch_params)
    fold_rows = _fold_parity(cfg, transfer, entry_summary, patch_params)

    rhs_tol = float(cfg["pointwise_rhs"]["absolute_tolerance"])
    period_tol = float(cfg["return_map"]["period_tolerance_ms"])
    crossing_tol = float(cfg["return_map"]["crossing_state_tolerance"])
    fold_tol = float(cfg["fold_surface"]["rhs_parity_tolerance"])
    gates = {
        "pointwise_p1_rhs": all(row["p1_rhs_max_abs_error"] <= rhs_tol for row in rhs_rows),
        "uniform_patch_rhs": all(row["uniform_rhs_max_abs_error"] <= rhs_tol for row in rhs_rows),
        "local_slow_derivatives_frozen": all(row["frozen_slow_rhs_max_abs"] == 0.0 for row in rhs_rows),
        "directed_return_status": all(row["status_match"] for row in return_rows),
        "directed_return_period": all(
            row["period_abs_error_ms"] <= period_tol
            for row in return_rows if row["period_abs_error_ms"] is not None
        ),
        "directed_return_crossing_state": all(
            row["crossing_state_max_abs_error"] <= crossing_tol
            for row in return_rows if row["crossing_state_max_abs_error"] is not None
        ),
        "full_euler_trace": all(
            row["trace_shapes_equal"] and row["trace_state_max_abs_error"] <= crossing_tol
            for row in return_rows
        ),
        "fold_surface_rhs": all(row["p1_fold_rhs_max_abs_error"] <= fold_tol for row in fold_rows),
    }
    status = "P1_UNIFORM_PARITY_PASS_READY_FOR_P2" if all(gates.values()) else "P1_PARITY_FAIL_STOP_BEFORE_P2"

    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "p1_rhs_parity.csv", rhs_rows)
    _write_csv(output / "p1_return_parity.csv", return_rows)
    _write_csv(output / "p1_fold_surface_parity.csv", fold_rows)
    np.savez_compressed(output / "p1_parity_traces.npz", **traces)
    figure = _plot(figures, rhs_rows, return_rows, fold_rows, traces)
    summary = {
        "status": status,
        "scientific_layer": "engineering_scientific_identity_gate_not_spatial_dynamics",
        "state_contract": {
            "continuous_shape": "10P+2",
            "local_fields": list(LOCAL_FIELDS),
            "shared_fields": ["mu_G", "S_G"],
            "p1_state_size": 12,
            "uniform_patch_count": int(cfg["pointwise_rhs"]["uniform_patch_count"]),
            "local_slow_status": "z_p_m_frozen_exact_zero_rhs",
        },
        "gates": gates,
        "maximum_errors": {
            "p1_rhs": max(row["p1_rhs_max_abs_error"] for row in rhs_rows),
            "uniform_rhs": max(row["uniform_rhs_max_abs_error"] for row in rhs_rows),
            "return_period_ms": max(
                row["period_abs_error_ms"] for row in return_rows
                if row["period_abs_error_ms"] is not None
            ),
            "return_crossing_state": max(
                row["crossing_state_max_abs_error"] for row in return_rows
                if row["crossing_state_max_abs_error"] is not None
            ),
            "full_trace_state": max(row["trace_state_max_abs_error"] for row in return_rows),
            "fold_surface_rhs": max(row["p1_fold_rhs_max_abs_error"] for row in fold_rows),
        },
        "return_rows": return_rows,
        "claim_boundary": [
            "P=1 and constant-field P=4 are the locked Stage-0C additive vector field",
            "there is exactly one shared mu_G/S_G pair; Stage-0C batch rows were not reinterpreted as patches",
            "the P=4 ring/uniform operators are parity probes only and do not calibrate P=2 coupling",
            "z/p/m are frozen; no onset, recruitment, latch, exit, recovery, or spatial pattern is claimed",
        ],
        "next_step": "P2_core_surround_frozen_sheets_then_minimal_full_ODE" if all(gates.values()) else "stop_and_repair_patch_RHS",
        "input_sha256": observed_hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "rhs_csv": str((output / "p1_rhs_parity.csv").relative_to(ROOT)),
            "return_csv": str((output / "p1_return_parity.csv").relative_to(ROOT)),
            "fold_csv": str((output / "p1_fold_surface_parity.csv").relative_to(ROOT)),
            "traces_npz": str((output / "p1_parity_traces.npz").relative_to(ROOT)),
        },
        "config": str(config_path.relative_to(ROOT)),
    }
    (output / "p1_parity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_patch_p1_parity.png / .pdf\n\n"
        "这张四面板图检验新的通用 P-patch scaffold 在 P=1 与 uniform-P 极限下是否仍是同一个 Stage 0C 快系统。"
        "A 比较多个 additive current 与 off-manifold/cycle probe 的逐项 RHS；B 比较 A=0/.30/.316、base/half dt 的 directed-return period；"
        "C 叠加 A=.30 的完整 rE 周期；D 在上游 fold surface 上再次比较 vector field。\n\n"
        "**关注点**：这一步只验方程身份，不是空间动力学结果。local z/p/m 在本 Gate 中严格冻结，"
        "全域只有一个 shared mu_G/S_G；P=4 operator 仅用于 constant-preserving parity，不是 P=2 coupling 调参。\n",
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
        "maximum_errors": summary["maximum_errors"],
        "next_step": summary["next_step"],
        "figure": summary["artifacts"]["figure"],
    }, indent=2))


if __name__ == "__main__":
    main()
