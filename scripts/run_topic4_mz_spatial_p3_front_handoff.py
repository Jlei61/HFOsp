#!/usr/bin/env python3
"""Test local core-to-annulus hand-off before enabling additive slow dynamics."""

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

from src.topic4_mz_entry_exit_nullclines import find_equilibria  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import (  # noqa: E402
    integrate_frozen_patch_batch,
    lift_product_history,
    summarize_local_state,
)
from src.topic4_mz_spatial_patch import PatchKernels, PatchParameters, prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer  # noqa: E402
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_p3_front_handoff.yaml"
PATCH_NAMES = ("core", "annulus", "bath")
COLORS = ("#B2182B", "#EF8A62", "#2166AC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    keys = ("transfer_path", "orbit_cycle_path", "orbit_summary_path")
    if set(cfg["input_sha256"]) != set(keys):
        raise ValueError(f"input_sha256 must lock exactly {keys}")
    observed = {}
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(path)
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
    low = PoolParameters(
        float(model["z_low"]), float(model["alpha_G"]),
        float(model["w_ee_mult"]), float(model["ratio"]),
    )
    patch = PatchParameters(
        alpha_g=float(model["alpha_G"]),
        w_ee_mult=float(model["w_ee_mult"]),
        ratio=float(model["ratio"]),
        additive_max_mv=float(model["additive_max_mv"]),
        pool_p=float(model["pool_p"]),
    )
    return low, patch


def _low_template(transfer: Any, params: PoolParameters) -> tuple[np.ndarray, dict]:
    roots = find_equilibria(params, transfer, 0.0)
    stable = [row for row in roots if row["stability"] == "stable" and row["rE_hz"] < 5.0]
    if not stable:
        raise RuntimeError("no stable low root at registered z_low")
    root = min(stable, key=lambda row: row["rE_hz"])
    return equilibrium_state((root["rE_hz"] * 1e-3, root["rI_hz"] * 1e-3)), root


def _phase_state(cycle: np.ndarray, phase: float) -> np.ndarray:
    return np.asarray(cycle[int(round(float(phase) * (cycle.shape[0] - 1)))], dtype=float)


def _cases(phases: list[float]) -> list[dict]:
    rows = [
        {"case": "LLL_base", "initial_sheet": "LLL", "phases": (None, None, None)},
        {"case": "LLL_antisymmetric", "initial_sheet": "LLL", "phases": (None, None, None),
         "antisymmetric": True},
    ]
    rows.extend(
        {"case": f"CLL_phase_{phase:.2f}", "initial_sheet": "CLL", "phases": (phase, None, None)}
        for phase in phases
    )
    rows.extend(
        {"case": f"LCL_phase_{phase:.2f}", "initial_sheet": "LCL", "phases": (None, phase, None)}
        for phase in phases
    )
    rows.extend(
        {"case": f"CCL_relative_{phase:.2f}", "initial_sheet": "CCL", "phases": (0.0, phase, None)}
        for phase in phases
    )
    rows.extend([
        {"case": "CCC_sync", "initial_sheet": "CCC", "phases": (0.0, 0.0, 0.0)},
        {"case": "CCC_antisymmetric", "initial_sheet": "CCC", "phases": (0.0, 0.0, 0.0),
         "antisymmetric": True},
    ])
    return rows


def _initial_batch(
    cases: list[dict], low: np.ndarray, cycle: np.ndarray, kernels: PatchKernels,
    parameters: PatchParameters, cfg: dict,
) -> np.ndarray:
    states = []
    z_low = float(cfg["model"]["z_low"])
    z_cycle = float(cfg["model"]["z_cycle"])
    perturb = float(cfg["initial_sheets"]["antisymmetric_perturbation_khz"])
    weights = kernels.weights()
    for case in cases:
        templates = [low if phase is None else _phase_state(cycle, phase) for phase in case["phases"]]
        z = [z_low if phase is None else z_cycle for phase in case["phases"]]
        state = lift_product_history(np.asarray(templates), kernels, z=z, parameters=parameters)
        if case.get("antisymmetric", False):
            state[0] += perturb
            state[1] -= perturb * weights[0] / weights[1]
        states.append(state)
    return np.asarray(states)


def _established_onset(return_times: list[float], minimum_returns: int) -> float | None:
    values = np.asarray(return_times, dtype=float)
    if values.size < minimum_returns:
        return None
    for start in range(values.size - minimum_returns + 1):
        segment = values[start:start + minimum_returns]
        intervals = np.diff(segment)
        if np.all((intervals >= 300.0) & (intervals <= 12000.0)):
            return float(segment[0])
    return None


def _multi_label(statuses: list[str]) -> str:
    if all(value in {"L", "C"} for value in statuses):
        return "".join(statuses)
    if "physical_or_numerical_failure" in statuses:
        return "physical_or_numerical_failure"
    if "unbounded_or_saturation" in statuses:
        return "unbounded_or_saturation"
    if "ceiling_or_nonclosed" in statuses:
        return "ceiling_or_nonclosed"
    return "O_unresolved"


def _poincare_closure(return_states: list[np.ndarray], n_patches: int) -> float | None:
    """Scaled max residual between the last two crossings of one section."""

    if len(return_states) < 2:
        return None
    previous = np.asarray(return_states[-2], dtype=float)
    current = np.asarray(return_states[-1], dtype=float)
    floor = np.full(previous.size, 1.0e-3, dtype=float)
    floor[7 * n_patches:10 * n_patches] = 1.0
    scale = np.maximum.reduce([np.abs(previous), np.abs(current), floor])
    return float(np.max(np.abs(current - previous) / scale))


def _recent_peak_drift(
    time_ms: np.ndarray,
    rate_khz: np.ndarray,
    return_times_ms: list[float],
    *,
    cycles: int = 4,
) -> float | None:
    returns = np.asarray(return_times_ms, dtype=float)
    if returns.size < cycles + 1:
        return None
    peaks = []
    for left, right in zip(returns[-cycles - 1:-1], returns[-cycles:]):
        mask = (time_ms >= left) & (time_ms < right)
        if not np.any(mask):
            return None
        peaks.append(float(np.max(rate_khz[mask])))
    values = np.asarray(peaks, dtype=float)
    return float(np.max(np.abs(np.diff(values)) / np.maximum(values[:-1], 1.0e-9)))


def _summaries(
    arm: str, dt_ms: float, cases: list[dict], result: dict[str, Any], cfg: dict,
) -> list[dict]:
    rows = []
    minimum_returns = int(cfg["integration"]["established_min_returns"])
    discard = int(cfg["integration"]["discard_returns"])
    for case_index, case in enumerate(cases):
        local = []
        for patch_index in range(3):
            local.append(
                summarize_local_state(
                    result["time_ms"], result["rE_khz"][:, case_index, patch_index],
                    result["rE_fast_khz"][:, case_index, patch_index],
                    result["return_times_ms"][case_index][patch_index],
                    support_violation_count=int(result["support_violation_count"][case_index, patch_index]),
                    state_bound_violation_count=int(result["state_bound_violation_count"][case_index, patch_index]),
                    finite=bool(result["finite"][case_index]), discard_returns=discard,
                )
            )
        statuses = [entry["status"] for entry in local]
        onsets = [_established_onset(entry["return_times_ms"], minimum_returns) for entry in local]
        core_last = local[0]["return_times_ms"][-1] if local[0]["return_times_ms"] else None
        front_before_core_exit = bool(
            onsets[0] is not None
            and onsets[1] is not None
            and core_last is not None
            and float(onsets[1]) <= float(core_last)
        )
        core_closure = _poincare_closure(
            result["return_states"][case_index][0], len(PATCH_NAMES)
        )
        core_peak_drift = _recent_peak_drift(
            np.asarray(result["time_ms"], dtype=float),
            np.asarray(result["rE_khz"][:, case_index, 0], dtype=float),
            local[0]["return_times_ms"],
        )
        contained_cycle = bool(
            statuses[0] == "C"
            and statuses[1] == "C"
            and local[0]["n_returns"] >= 6
            and local[1]["n_returns"] >= 6
            and local[0]["recent_period_cv"] is not None
            and local[1]["recent_period_cv"] is not None
            and local[0]["recent_period_cv"] <= 0.01
            and local[1]["recent_period_cv"] <= 0.01
            and core_closure is not None
            # Linear section interpolation on the registered Euler solver has
            # a measured 0.3e-6--15e-6 closure floor across base/half dt.
            and core_closure <= 2.0e-5
            and core_peak_drift is not None
            and core_peak_drift <= 0.01
            and local[2]["n_returns"] == 0
            and local[2]["peak_rE_hz"] < 20.0
        )
        ceiling_cycle = bool(
            contained_cycle
            and any(entry["sustained_ceiling_120hz_80of100ms"] for entry in local[:2])
        )
        final_sheet = (
            "bounded_ceiling_CCO" if ceiling_cycle
            else "bounded_CCO" if contained_cycle
            else _multi_label(statuses)
        )
        row = {
            "arm": arm,
            "dt_ms": float(dt_ms),
            "case": case["case"],
            "initial_sheet": case["initial_sheet"],
            "phase_core": case["phases"][0],
            "phase_annulus": case["phases"][1],
            "phase_bath": case["phases"][2],
            "final_sheet": final_sheet,
            "front_established_before_core_exit": front_before_core_exit,
            "contained_local_cycle": contained_cycle,
            "bounded_ceiling_cycle": ceiling_cycle,
            "core_poincare_closure_scaled": core_closure,
            "core_recent_peak_drift": core_peak_drift,
        }
        for patch_name, entry, onset in zip(PATCH_NAMES, local, onsets):
            row.update({
                f"{patch_name}_status": entry["status"],
                f"{patch_name}_returns": entry["n_returns"],
                f"{patch_name}_established_onset_ms": onset,
                f"{patch_name}_period_ms": entry["recent_period_ms"],
                f"{patch_name}_period_cv": entry["recent_period_cv"],
                f"{patch_name}_peak_hz": entry["peak_rE_hz"],
                f"{patch_name}_tail_mean_hz": entry["tail_mean_rE_hz"],
                f"{patch_name}_sustained_ceiling": entry["sustained_ceiling_120hz_80of100ms"],
                f"{patch_name}_support_violations": entry["support_violation_count"],
                f"{patch_name}_bound_violations": entry["state_bound_violation_count"],
                f"{patch_name}_return_times_ms": json.dumps(entry["return_times_ms"]),
            })
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_trace(path: Path, cases: list[dict], result: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        time_ms=result["time_ms"].astype(np.float32),
        case=np.asarray([case["case"] for case in cases]),
        rE_khz=result["rE_khz"], rI_khz=result["rI_khz"],
        rE_fast_khz=result["rE_fast_khz"], shared_state=result["shared_state"],
        final_state=result["final_state"].astype(np.float32), finite=result["finite"],
        support_violation_count=result["support_violation_count"],
        state_bound_violation_count=result["state_bound_violation_count"],
    )


def _gates(rows: list[dict], dts: list[float]) -> dict[str, bool]:
    official = "fixed_m3b_coupling"
    off = "cross_zone_synaptic_coupling_off"

    def chosen(arm: str, case: str) -> list[dict]:
        return [row for row in rows if row["arm"] == arm and row["case"] == case]

    cll = sorted({row["case"] for row in rows if row["initial_sheet"] == "CLL"})
    ccl = sorted({row["case"] for row in rows if row["initial_sheet"] == "CCL"})
    fixed_handoff = {
        case for case in cll
        if all(row["front_established_before_core_exit"] for row in chosen(official, case))
    }
    off_handoff = {
        case for case in cll
        if all(row["front_established_before_core_exit"] for row in chosen(off, case))
    }
    keys = {(row["arm"], row["case"]) for row in rows}
    dt_stable = all(
        len({row["final_sheet"] for row in rows if row["arm"] == arm and row["case"] == case}) == 1
        for arm, case in keys
    )
    fixed_ccl = [
        row for row in rows if row["arm"] == official and row["initial_sheet"] == "CCL"
    ]
    closure_converges = all(
        max(
            row["core_poincare_closure_scaled"] for row in fixed_ccl
            if row["case"] == case and row["dt_ms"] == min(dts)
        )
        <= max(
            row["core_poincare_closure_scaled"] for row in fixed_ccl
            if row["case"] == case and row["dt_ms"] == max(dts)
        )
        for case in ccl
    )
    return {
        "lll_low_basin_preserved": all(
            row["final_sheet"] == "LLL"
            for case in ("LLL_base", "LLL_antisymmetric") for row in chosen(official, case)
        ),
        "ccc_cycle_preserved": all(
            row["final_sheet"] == "CCC"
            for case in ("CCC_sync", "CCC_antisymmetric") for row in chosen(official, case)
        ),
        "base_half_dt_labels_match": dt_stable,
        "half_dt_poincare_closure_converges": closure_converges,
        "core_cycle_survives_cll_all_phases": all(
            row["core_status"] == "C" for case in cll for row in chosen(official, case)
        ),
        "annulus_handoff_any_phase": bool(fixed_handoff),
        "annulus_handoff_all_phases": fixed_handoff == set(cll),
        "coupling_specific_annulus_handoff": bool(fixed_handoff - off_handoff),
        "bounded_ccl_sheet_any_phase": any(
            row["final_sheet"] == "bounded_CCO" for case in ccl for row in chosen(official, case)
        ),
        "bounded_ccl_sheet_all_phases": all(
            row["final_sheet"] == "bounded_CCO" for case in ccl for row in chosen(official, case)
        ),
        "cross_zone_off_has_no_false_handoff": not bool(off_handoff),
    }


def _counts(rows: list[dict], arm: str, dt: float) -> dict[str, dict[str, int]]:
    output = {}
    for initial in ("LLL", "CLL", "LCL", "CCL", "CCC"):
        selected = [
            row for row in rows if row["arm"] == arm and row["dt_ms"] == dt
            and row["initial_sheet"] == initial and "antisymmetric" not in row["case"]
        ]
        output[initial] = {
            label: sum(row["final_sheet"] == label for row in selected)
            for label in sorted({row["final_sheet"] for row in selected})
        }
    return output


def _plot(figures: Path, reduction: Any, cases: list[dict], result: dict[str, Any], rows: list[dict], dt: float) -> Path:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    weights = reduction.kernels.weights()
    x = np.arange(3)
    ax.bar(x, weights, color=COLORS, alpha=0.9)
    ax.set_xticks(x, PATCH_NAMES)
    ax.set(ylabel="tissue area fraction", title="A  Core–front–bath partition")
    ax.text(0.02, 0.97, f"annulus←core E→E = {reduction.kernels.K_EE[1,0]:.3f}\n"
            f"annulus←core I = {reduction.kernels.K_I[1,0]:.3f}\n"
            f"bath retained = {100*weights[2]:.1f}%", transform=ax.transAxes, va="top")

    index = {case["case"]: position for position, case in enumerate(cases)}
    panels = [
        (axes[0, 1], "LLL_base", "B  LLL seed"),
        (axes[0, 2], "CLL_phase_0.00", "C  CLL seed"),
        (axes[1, 0], "CCL_relative_0.00", "D  CCL seed"),
        (axes[1, 1], "LCL_phase_0.00", "E  LCL seed"),
    ]
    time = result["time_ms"] * 1e-3
    for panel, case_name, title in panels:
        ci = index[case_name]
        for pi, (name, color) in enumerate(zip(PATCH_NAMES, COLORS)):
            panel.plot(time, 1000.0 * result["rE_khz"][:, ci, pi], color=color, lw=0.8, label=name)
        panel.axhline(20.0, color="0.72", ls="--", lw=0.7)
        panel.set(xlabel="time (s)", ylabel="rE (Hz)", title=title, xlim=(time[0], time[-1]))
        panel.margins(x=0)
    axes[0, 1].legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    fixed = _counts(rows, "fixed_m3b_coupling", dt)
    off = _counts(rows, "cross_zone_synaptic_coupling_off", dt)
    lines = ["F  Frozen local hand-off", "", "initial   fixed K          K=I (shared pool kept)"]
    for initial in ("LLL", "CLL", "LCL", "CCL", "CCC"):
        display = {
            "physical_or_numerical_failure": "support_fail",
            "ceiling_or_nonclosed": "ceiling/open",
            "bounded_ceiling_CCO": "ceiling_CCO",
        }
        left = ", ".join(f"{display.get(key, key)}:{value}" for key, value in fixed[initial].items()) or "—"
        right = ", ".join(f"{display.get(key, key)}:{value}" for key, value in off[initial].items()) or "—"
        lines.append(f"{initial:<8}{left:<17}{right}")
    lines.extend(["", "Success requires annulus returns before core exit;",
                  "no slow variable or E→E retuning is present."])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.0)
    fig.suptitle("Can a focal cycle hand activity to the first spatial shell?", fontsize=13, fontweight="bold")
    stem = figures / "mz_spatial_p3_front_handoff"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    low_params, patch_params = _model(cfg)
    orbit_summary = json.loads((ROOT / cfg["orbit_summary_path"]).read_text(encoding="utf-8"))
    if orbit_summary["model_contract"]["fast_system"] != "locked Stage0C nine-state system":
        raise RuntimeError("orbit input is not the locked Stage-0C system")
    low, low_root = _low_template(transfer, low_params)
    key = str(cfg["initial_sheets"]["cycle_trace_key"])
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{key}_state"], dtype=float)

    geometry = cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    weights = reduction.kernels.weights()
    arms = {
        "fixed_m3b_coupling": reduction.kernels,
        "cross_zone_synaptic_coupling_off": PatchKernels(np.eye(3), np.eye(3), weights).validate(),
    }
    phases = [float(value) for value in cfg["initial_sheets"]["phase_fractions"]]
    cases = _cases(phases)
    if len(cases) != int(cfg["resource_contract"]["vectorized_forks"]):
        raise RuntimeError("case count drifted from resource contract")
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_rows = []
    base_result = None
    base_dt = float(cfg["integration"]["dt_ms"][0])
    for arm, kernels in arms.items():
        initial = _initial_batch(cases, low, cycle, kernels, patch_params, cfg)
        prepared = prepare_patch_rhs(kernels, patch_params)
        for dt_value in cfg["integration"]["dt_ms"]:
            dt = float(dt_value)
            result = integrate_frozen_patch_batch(
                initial, prepared, transfer, dt_ms=dt,
                duration_ms=float(cfg["integration"]["duration_ms"]),
                save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
                section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
                rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            )
            all_rows.extend(_summaries(arm, dt, cases, result, cfg))
            suffix = str(dt).replace(".", "p")
            _save_trace(output / f"p3_{arm}_dt{suffix}_traces.npz", cases, result)
            if arm == "fixed_m3b_coupling" and dt == base_dt:
                base_result = result
    if base_result is None:
        raise RuntimeError("missing primary result")
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    gates = _gates(all_rows, dts)
    if not gates["lll_low_basin_preserved"] or not gates["ccc_cycle_preserved"]:
        status = "P3_FAST_SCAFFOLD_INVALID"
    elif gates["coupling_specific_annulus_handoff"] and gates["bounded_ccl_sheet_any_phase"]:
        status = "P3_LOCAL_HANDOFF_AND_BOUNDED_CCL_SUPPORTED_READY_FOR_SLOW_LATCH"
    elif gates["coupling_specific_annulus_handoff"]:
        status = "P3_LOCAL_HANDOFF_WITHOUT_BOUNDED_CCL"
    elif gates["bounded_ccl_sheet_any_phase"]:
        status = "P3_BOUNDED_LOCAL_CYCLE_WITHOUT_FAST_CLL_HANDOFF_TEST_REGIONAL_Z_ENTRY"
    else:
        status = "P3_NO_LOCAL_HANDOFF_CURRENT_ADDITIVE_ARCHITECTURE_STOP"

    _write_csv(output / "p3_front_handoff_outcomes.csv", all_rows)
    figure = _plot(figures, reduction, cases, base_result, all_rows, base_dt)
    mass_errors = {
        "K_EE_row_sum": float(np.max(np.abs(reduction.kernels.K_EE.sum(axis=1) - 1.0))),
        "K_I_row_sum": float(np.max(np.abs(reduction.kernels.K_I.sum(axis=1) - 1.0))),
        "K_EE_stationarity": float(np.max(np.abs(weights @ reduction.kernels.K_EE - weights))),
        "K_I_stationarity": float(np.max(np.abs(weights @ reduction.kernels.K_I - weights))),
    }
    summary = {
        "status": status,
        "scientific_layer": "mass_balanced_local_front_frozen_fast_gate_not_lifecycle",
        "geometry": {
            "patch_names": list(reduction.patch_names),
            "patch_cells": list(reduction.patch_cells),
            "patch_weights": weights.tolist(),
            "core_radius_mm": reduction.core_radius_mm,
            "outer_annulus_radius_mm": reduction.outer_annulus_radius_mm,
            "K_EE": reduction.kernels.K_EE.tolist(),
            "K_I": reduction.kernels.K_I.tolist(),
            "mass_balance_errors": mass_errors,
        },
        "anchors": {
            "low_z": float(cfg["model"]["z_low"]), "low_root": low_root,
            "cycle_z": float(cfg["model"]["z_cycle"]), "cycle_trace_key": key,
            "cycle_phase_fractions": phases,
        },
        "gates": gates,
        "outcome_counts_base_dt": {arm: _counts(all_rows, arm, base_dt) for arm in arms},
        "claim_boundary": [
            "the equal-area first annulus is resolved without deleting or renormalizing the far bath",
            "a successful hand-off requires established annulus returns before focal-core exit",
            "K=I keeps the same area-weighted shared pool and removes only cross-zone synaptic coupling",
            "z/p/m are frozen; no termination, recovery, latch, or full lifecycle is claimed",
            "no E-E weight, kernel, delay, relay, or conductance parameter was changed",
        ],
        "decision": (
            "allow_minimal_persistence_AND_recruitment_latch"
            if status == "P3_LOCAL_HANDOFF_AND_BOUNDED_CCL_SUPPORTED_READY_FOR_SLOW_LATCH"
            else "run_regional_z_entry_exit_oracle_only"
            if status == "P3_BOUNDED_LOCAL_CYCLE_WITHOUT_FAST_CLL_HANDOFF_TEST_REGIONAL_Z_ENTRY"
            else "stop_additive_spatial_lifecycle_before_slow_parameter_search"
        ),
        "input_sha256": hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "outcome_csv": str((output / "p3_front_handoff_outcomes.csv").relative_to(ROOT)),
            "trace_glob": str((output / "p3_*_traces.npz").relative_to(ROOT)),
        },
        "config": cfg,
    }
    (output / "p3_front_handoff_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_p3_front_handoff.png\n\n"
        "这张图检验 additive MZ 线在不加慢变量时，局部 core cycle 能否把活动交给第一个等面积邻环。A 显示 core/annulus/far-bath 的真实面积；B–E 分别给出 LLL、CLL、CCL、LCL 初态；F 对比 fixed M3B coupling 与 K=I。\n\n"
        "far bath 没有被删除或重新归一化。只有 annulus 在 core 退出前建立连续 local returns，并且 K=I 不出现同样结果，才算 endogenous local hand-off。\n\n"
        "**关注点**：CLL 中橙色 annulus 是否在红色 core 熄灭前建立 bounded returns；以及 CCL 能否作为 bath 仍低的局部持续 sheet。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps({"status": summary["status"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
