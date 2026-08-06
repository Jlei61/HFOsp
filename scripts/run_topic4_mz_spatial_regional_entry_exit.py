#!/usr/bin/env python3
"""Run the final frozen regional-Z entry and delayed regional-A exit oracle."""

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
from src.topic4_mz_spatial_entry_exit import (  # noqa: E402
    find_regional_equilibria,
    solve_regional_additive_fold,
    solve_regional_fold,
)
from src.topic4_mz_spatial_frozen_sheets import (  # noqa: E402
    integrate_frozen_patch_batch,
    lift_product_history,
    summarize_local_state,
)
from src.topic4_mz_spatial_patch import (  # noqa: E402
    PatchParameters,
    patch_rhs_fast,
    prepare_patch_rhs,
)
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    ExtendedSiegertTransfer,
)
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_regional_entry_exit.yaml"
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


def _model(cfg: dict) -> tuple[PatchParameters, PoolParameters]:
    model = cfg["model"]
    patch = PatchParameters(
        alpha_g=float(model["alpha_G"]),
        w_ee_mult=float(model["w_ee_mult"]),
        ratio=float(model["ratio"]),
        additive_max_mv=float(model["additive_max_mv"]),
        pool_p=float(model["pool_p"]),
    )
    low = PoolParameters(
        float(model["z_interictal"]), float(model["alpha_G"]),
        float(model["w_ee_mult"]), float(model["ratio"]),
    )
    return patch, low


def _low_template(transfer: Any, params: PoolParameters) -> tuple[np.ndarray, dict]:
    roots = find_equilibria(params, transfer, 0.0)
    stable = [row for row in roots if row["stability"] == "stable" and row["rE_hz"] < 5.0]
    if not stable:
        raise RuntimeError("no registered interictal root")
    root = min(stable, key=lambda row: row["rE_hz"])
    return equilibrium_state((1.0e-3 * root["rE_hz"], 1.0e-3 * root["rI_hz"])), root


def _phase_state(cycle: np.ndarray, phase: float) -> np.ndarray:
    return np.asarray(cycle[int(round(float(phase) * (cycle.shape[0] - 1)))], dtype=float)


def _cycle_initial(
    low: np.ndarray,
    cycle: np.ndarray,
    phase: float,
    z_regional: float,
    reduction: Any,
    parameters: PatchParameters,
) -> np.ndarray:
    return lift_product_history(
        np.asarray([_phase_state(cycle, 0.0), _phase_state(cycle, phase), low]),
        reduction.kernels,
        z=[z_regional, z_regional, 0.90],
        parameters=parameters,
    )


def _low_initial(
    low: np.ndarray,
    z_regional: float,
    reduction: Any,
    parameters: PatchParameters,
) -> np.ndarray:
    return lift_product_history(
        np.asarray([low, low, low]), reduction.kernels,
        z=[z_regional, z_regional, 0.90], parameters=parameters,
    )


def _poincare_closure(return_states: list[np.ndarray], state_size: int) -> float | None:
    if len(return_states) < 2:
        return None
    previous = np.asarray(return_states[-2], dtype=float)
    current = np.asarray(return_states[-1], dtype=float)
    if previous.shape != (state_size,) or current.shape != (state_size,):
        raise ValueError("return state shape drift")
    p = (state_size - 2) // 10
    floor = np.full(state_size, 1.0e-3, dtype=float)
    floor[7 * p:10 * p] = 1.0
    scale = np.maximum.reduce([np.abs(previous), np.abs(current), floor])
    return float(np.max(np.abs(current - previous) / scale))


def _peak_drift(time: np.ndarray, rate: np.ndarray, returns: list[float]) -> float | None:
    crossing = np.asarray(returns, dtype=float)
    if crossing.size < 5:
        return None
    peaks = []
    for left, right in zip(crossing[-5:-1], crossing[-4:]):
        mask = (time >= left) & (time < right)
        if not np.any(mask):
            return None
        peaks.append(float(np.max(rate[mask])))
    values = np.asarray(peaks, dtype=float)
    return float(np.max(np.abs(np.diff(values)) / np.maximum(values[:-1], 1.0e-9)))


def _pattern_summary(
    result: dict[str, Any], index: int, cfg: dict, prepared: Any, transfer: Any,
) -> dict[str, Any]:
    contract = cfg["classification"]
    local = []
    for patch_index in range(3):
        local.append(summarize_local_state(
            result["time_ms"], result["rE_khz"][:, index, patch_index],
            result["rE_fast_khz"][:, index, patch_index],
            result["return_times_ms"][index][patch_index],
            support_violation_count=int(result["support_violation_count"][index, patch_index]),
            state_bound_violation_count=int(result["state_bound_violation_count"][index, patch_index]),
            finite=bool(result["finite"][index]),
            discard_returns=int(contract["discard_returns"]),
        ))
    closure = _poincare_closure(
        result["return_states"][index][0], result["final_state"].shape[1]
    )
    peak_drift = _peak_drift(
        np.asarray(result["time_ms"], dtype=float),
        np.asarray(result["rE_khz"][:, index, 0], dtype=float),
        local[0]["return_times_ms"],
    )
    contained = bool(
        local[0]["status"] == "C" and local[1]["status"] == "C"
        and local[0]["n_returns"] >= int(contract["accepted_min_returns"])
        and local[1]["n_returns"] >= int(contract["accepted_min_returns"])
        and local[0]["recent_period_cv"] is not None
        and local[1]["recent_period_cv"] is not None
        and local[0]["recent_period_cv"] <= float(contract["period_cv_max"])
        and local[1]["recent_period_cv"] <= float(contract["period_cv_max"])
        and closure is not None and closure <= float(contract["poincare_closure_max"])
        and peak_drift is not None and peak_drift <= float(contract["peak_drift_max"])
        and local[2]["n_returns"] == 0
        and local[2]["peak_rE_hz"] < float(contract["bath_peak_max_hz"])
    )
    ceiling = bool(
        contained and any(row["sustained_ceiling_120hz_80of100ms"] for row in local[:2])
    )
    time = np.asarray(result["time_ms"], dtype=float)
    tail = time >= max(float(time[-1]) - float(contract["low_tail_ms"]), float(time[0]))
    rates = 1000.0 * np.asarray(result["rE_khz"][:, index, :], dtype=float)
    fast = 1000.0 * np.asarray(result["rE_fast_khz"][:, index, :], dtype=float)
    last_tail_start = float(time[tail][0])
    final_fast_rhs_max = float(np.max(np.abs(
        patch_rhs_fast(np.asarray(result["final_state"][index], dtype=float), prepared, transfer)
    )))
    low = bool(
        np.all(rates[tail] < float(contract["low_tail_max_hz"]))
        and np.all(fast[tail] < float(contract["low_tail_max_hz"]))
        and all(
            not any(value >= last_tail_start for value in row["return_times_ms"])
            for row in local
        )
        and all(row["support_violation_count"] == 0 for row in local)
        and all(row["state_bound_violation_count"] == 0 for row in local)
        and bool(result["finite"][index])
        and final_fast_rhs_max <= float(contract["low_final_fast_rhs_max_per_ms"])
    )
    if contained:
        label = "bounded_ceiling_CCO" if ceiling else "bounded_CCO"
    elif low:
        label = "LLL"
    elif any(row["status"] == "physical_or_numerical_failure" for row in local):
        label = "physical_or_numerical_failure"
    elif any(row["status"] == "ceiling_or_nonclosed" for row in local):
        label = "ceiling_or_nonclosed"
    else:
        label = "O_unresolved"
    output: dict[str, Any] = {
        "outcome": label,
        "contained_local_cycle": contained,
        "bounded_ceiling_cycle": ceiling,
        "core_poincare_closure_scaled": closure,
        "core_recent_peak_drift": peak_drift,
        "final_fast_rhs_max_per_ms": final_fast_rhs_max,
    }
    for name, row in zip(PATCH_NAMES, local):
        output.update({
            f"{name}_status": row["status"],
            f"{name}_returns": row["n_returns"],
            f"{name}_period_ms": row["recent_period_ms"],
            f"{name}_period_cv": row["recent_period_cv"],
            f"{name}_peak_hz": row["peak_rE_hz"],
            f"{name}_tail_mean_hz": row["tail_mean_rE_hz"],
            f"{name}_sustained_ceiling": row["sustained_ceiling_120hz_80of100ms"],
            f"{name}_support_violations": row["support_violation_count"],
            f"{name}_bound_violations": row["state_bound_violation_count"],
            f"{name}_return_times_ms": json.dumps(row["return_times_ms"]),
        })
    return output


def _set_additive(state: np.ndarray, additive_mv: float, parameters: PatchParameters) -> np.ndarray:
    output = np.asarray(state, dtype=float).copy()
    p = 3
    output[9 * p:10 * p] = np.asarray([
        additive_mv / parameters.additive_max_mv,
        additive_mv / parameters.additive_max_mv,
        0.0,
    ])
    return output


def _set_recovered_parameters(state: np.ndarray) -> np.ndarray:
    output = np.asarray(state, dtype=float).copy()
    output[7 * 3:8 * 3] = 0.90
    output[9 * 3:10 * 3] = 0.0
    return output


def _checkpoint(result: dict[str, Any], case_index: int, minimum_returns: int) -> tuple[np.ndarray, float]:
    core_times = np.asarray(result["return_times_ms"][case_index][0], dtype=float)
    annulus_times = np.asarray(result["return_times_ms"][case_index][1], dtype=float)
    if core_times.size < minimum_returns or annulus_times.size < minimum_returns:
        raise RuntimeError("source CCL did not establish enough regional returns")
    target = float(annulus_times[minimum_returns - 1])
    candidates = np.where(
        (core_times >= target) & (np.arange(core_times.size) >= minimum_returns - 1)
    )[0]
    if candidates.size == 0:
        raise RuntimeError("no core section after annulus establishment")
    selected = int(candidates[0])
    return (
        np.asarray(result["return_states"][case_index][0][selected], dtype=float),
        float(core_times[selected]),
    )


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_trace(path: Path, labels: list[str], result: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        time_ms=np.asarray(result["time_ms"], dtype=np.float32),
        case=np.asarray(labels),
        rE_khz=np.asarray(result["rE_khz"], dtype=np.float32),
        rI_khz=np.asarray(result["rI_khz"], dtype=np.float32),
        rE_fast_khz=np.asarray(result["rE_fast_khz"], dtype=np.float32),
        shared_state=np.asarray(result["shared_state"], dtype=np.float32),
        final_state=np.asarray(result["final_state"], dtype=np.float32),
    )


def _entry_runs(
    cfg: dict, low: np.ndarray, cycle: np.ndarray, reduction: Any,
    parameters: PatchParameters, transfer: Any, prepared: Any, output: Path,
) -> tuple[list[dict], dict[tuple[float, float, str, float | None], tuple[dict, int]]]:
    phases = [float(value) for value in cfg["entry_oracle"]["relative_phase_fractions"]]
    cases = []
    states = []
    for z in map(float, cfg["entry_oracle"]["low_relabel_z_axis"]):
        cases.append({"seed": "L90_relabel", "z_regional": z, "phase": None})
        states.append(_low_initial(low, z, reduction, parameters))
    for z in map(float, cfg["entry_oracle"]["cycle_seed_z_axis"]):
        for phase in phases:
            cases.append({"seed": "CCL_warm_cycle", "z_regional": z, "phase": phase})
            states.append(_cycle_initial(low, cycle, phase, z, reduction, parameters))
    if len(cases) > int(cfg["resource_contract"]["maximum_vectorized_forks"]):
        raise RuntimeError("entry fork count exceeds resource contract")
    rows = []
    lookup = {}
    for dt_value in cfg["integration"]["dt_ms"]:
        dt = float(dt_value)
        result = integrate_frozen_patch_batch(
            np.asarray(states), prepared, transfer,
            dt_ms=dt, duration_ms=float(cfg["integration"]["entry_duration_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        labels = []
        for index, case in enumerate(cases):
            row = {"dt_ms": dt, **case, **_pattern_summary(result, index, cfg, prepared, transfer)}
            rows.append(row)
            lookup[(dt, case["z_regional"], case["seed"], case["phase"])] = (result, index)
            labels.append(f"{case['seed']}_z{case['z_regional']:.5f}_p{case['phase']}")
        _save_trace(output / f"entry_dt{str(dt).replace('.', 'p')}_traces.npz", labels, result)
    return rows, lookup


def _exit_runs(
    cfg: dict, low: np.ndarray, cycle: np.ndarray, reduction: Any,
    parameters: PatchParameters, transfer: Any, prepared: Any, output: Path,
) -> tuple[list[dict], list[dict], dict[tuple[float, float, str, float, float], tuple[dict, int]]]:
    phases = [float(value) for value in cfg["entry_oracle"]["relative_phase_fractions"]]
    rows = []
    source_rows = []
    state_lookup = {}
    for z in map(float, cfg["exit_oracle"]["source_z_axis"]):
        additive_axis = [float(value) for value in cfg["exit_oracle"]["additive_axis_mv"][f"{z:.4f}"]]
        for dt_value in cfg["integration"]["dt_ms"]:
            dt = float(dt_value)
            source_initial = np.asarray([
                _cycle_initial(low, cycle, phase, z, reduction, parameters) for phase in phases
            ])
            prelude = integrate_frozen_patch_batch(
                source_initial, prepared, transfer,
                dt_ms=dt, duration_ms=float(cfg["integration"]["exit_prelude_ms"]),
                save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
                section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
                rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            )
            checkpoints = []
            checkpoint_times = []
            for phase_index, phase in enumerate(phases):
                source = _pattern_summary(prelude, phase_index, cfg, prepared, transfer)
                checkpoint, checkpoint_time = _checkpoint(
                    prelude, phase_index,
                    int(cfg["exit_oracle"]["source_min_returns_each_region"]),
                )
                checkpoints.append(checkpoint)
                checkpoint_times.append(checkpoint_time)
                source_rows.append({
                    "z_regional": z, "dt_ms": dt, "phase": phase,
                    "checkpoint_time_ms": checkpoint_time, **source,
                })

            metadata = []
            states = []
            for mode in ("delayed_exit", "from_t0_prevention"):
                base_states = checkpoints if mode == "delayed_exit" else list(source_initial)
                for phase_index, phase in enumerate(phases):
                    for additive in additive_axis:
                        metadata.append({
                            "mode": mode, "phase": phase, "additive_mv": additive,
                            "checkpoint_time_ms": (
                                checkpoint_times[phase_index] if mode == "delayed_exit" else 0.0
                            ),
                        })
                        states.append(_set_additive(base_states[phase_index], additive, parameters))
            low_state = _low_initial(low, z, reduction, parameters)
            for additive in additive_axis:
                metadata.append({
                    "mode": "L90_relabel_control", "phase": None,
                    "additive_mv": additive, "checkpoint_time_ms": 0.0,
                })
                states.append(_set_additive(low_state, additive, parameters))
            if len(states) > int(cfg["resource_contract"]["maximum_vectorized_forks"]):
                raise RuntimeError("exit fork count exceeds resource contract")
            result = integrate_frozen_patch_batch(
                np.asarray(states), prepared, transfer,
                dt_ms=dt, duration_ms=float(cfg["integration"]["exit_post_ms"]),
                save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
                section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
                rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            )
            labels = []
            for index, meta in enumerate(metadata):
                row = {
                    "z_regional": z, "dt_ms": dt, **meta,
                    **_pattern_summary(result, index, cfg, prepared, transfer),
                }
                rows.append(row)
                key = (z, dt, meta["mode"], meta["phase"] if meta["phase"] is not None else -1.0,
                       meta["additive_mv"])
                state_lookup[key] = (result, index)
                labels.append(
                    f"{meta['mode']}_z{z:.4f}_p{meta['phase']}_A{meta['additive_mv']:.5f}"
                )
            _save_trace(
                output / f"exit_z{str(z).replace('.', 'p')}_dt{str(dt).replace('.', 'p')}_traces.npz",
                labels, result,
            )
    return rows, source_rows, state_lookup


def _confirmed_exit_thresholds(cfg: dict, rows: list[dict]) -> dict[float, float | None]:
    phases = [float(value) for value in cfg["entry_oracle"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    output = {}
    for z in map(float, cfg["exit_oracle"]["source_z_axis"]):
        axis = [float(value) for value in cfg["exit_oracle"]["additive_axis_mv"][f"{z:.4f}"]]
        accepted = []
        for additive in axis:
            if additive <= 0.0:
                continue
            selected = [
                row for row in rows
                if row["z_regional"] == z and row["mode"] == "delayed_exit"
                and row["additive_mv"] == additive
            ]
            if len(selected) == len(phases) * len(dts) and all(row["outcome"] == "LLL" for row in selected):
                accepted.append(additive)
        output[z] = min(accepted) if accepted else None
    return output


def _recovery_runs(
    cfg: dict, thresholds: dict[float, float | None], state_lookup: dict,
    prepared: Any, transfer: Any,
) -> list[dict]:
    rows = []
    phases = [float(value) for value in cfg["entry_oracle"]["relative_phase_fractions"]]
    for z, additive in thresholds.items():
        if additive is None:
            continue
        for dt_value in cfg["integration"]["dt_ms"]:
            dt = float(dt_value)
            states = []
            for phase in phases:
                result, index = state_lookup[(z, dt, "delayed_exit", phase, additive)]
                states.append(_set_recovered_parameters(result["final_state"][index]))
            recovered = integrate_frozen_patch_batch(
                np.asarray(states), prepared, transfer,
                dt_ms=dt, duration_ms=float(cfg["integration"]["recovery_ms"]),
                save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
                section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
                rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            )
            for index, phase in enumerate(phases):
                rows.append({
                    "source_z_regional": z, "source_exit_additive_mv": additive,
                    "dt_ms": dt, "phase": phase,
                    "recovered_z_regional": 0.90, "recovered_additive_mv": 0.0,
                    **_pattern_summary(recovered, index, cfg, prepared, transfer),
                })
    return rows


def _dt_labels_match(rows: list[dict], keys: tuple[str, ...]) -> bool:
    groups: dict[tuple, set[str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, set()).add(str(row["outcome"]))
    return all(len(values) == 1 for values in groups.values())


def _plot(
    figures: Path, root_rows: list[dict], fold: dict, entry_rows: list[dict],
    entry_lookup: dict, exit_rows: list[dict], exit_lookup: dict,
    additive_folds: dict[float, dict], thresholds: dict[float, float | None], cfg: dict,
) -> Path:
    plt.rcParams.update({"font.size": 8.3, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    for stability, color, marker in (("stable", "#2166AC", "o"), ("unstable", "#D6604D", "x"),
                                      ("marginal", "#762A83", "s")):
        selected = [row for row in root_rows if row["stability"] == stability]
        ax.scatter([row["z_regional"] for row in selected], [row["rE_hz"][0] for row in selected],
                   s=18, color=color, marker=marker, label=stability)
    ax.scatter([fold["z_regional"]], [fold["rE_hz"][0]], marker="*", s=90,
               color="#1B7837", zorder=5, label="augmented fold")
    ax.axvline(fold["z_regional"], color="#1B7837", ls=":", lw=1.0)
    ax.set(xlabel=r"regional resource $z_R$", ylabel="core equilibrium rE (Hz)",
           title="A  Localized real-mode entry fold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    base_dt = float(cfg["integration"]["dt_ms"][0])
    for z, ls in ((0.8560, "-"), (0.8550, "--")):
        result, index = entry_lookup[(base_dt, z, "L90_relabel", None)]
        time = result["time_ms"] * 1.0e-3
        ax.plot(time, 1000.0 * result["rE_khz"][:, index, 0], color=COLORS[0], ls=ls,
                lw=0.9, label=f"core z={z:.3f}")
        ax.plot(time, 1000.0 * result["rE_khz"][:, index, 1], color=COLORS[1], ls=ls,
                lw=0.8, label=f"annulus z={z:.3f}")
    ax.set(xlabel="time (s)", ylabel="rE (Hz)", title="B  Same low history across the fold")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[0, 2]
    palette = {"LLL": "#2166AC", "bounded_CCO": "#1B7837", "bounded_ceiling_CCO": "#E08214",
               "O_unresolved": "0.5", "ceiling_or_nonclosed": "#B2182B",
               "physical_or_numerical_failure": "black"}
    base_rows = [row for row in entry_rows if row["dt_ms"] == base_dt]
    y_map = {"L90_relabel": 0.0, "CCL_warm_cycle": 1.0}
    for row in base_rows:
        offset = 0.0 if row["phase"] is None else 0.12 * float(row["phase"])
        ax.scatter(row["z_regional"], y_map[row["seed"]] + offset, s=30,
                   color=palette.get(row["outcome"], "0.5"), edgecolor="white", linewidth=0.35)
    for label in ("bounded_CCO", "LLL", "O_unresolved"):
        ax.scatter([], [], s=28, color=palette[label], label=label)
    ax.axvline(fold["z_regional"], color="#1B7837", ls=":", lw=1.0)
    ax.set(yticks=[0, 1], yticklabels=["low relabel", "cycle phases"], xlabel=r"$z_R$",
           title="C  Frozen basin map")
    ax.legend(frameon=False, fontsize=6.7, loc="lower right")

    ax = axes[1, 0]
    zs = sorted(additive_folds)
    ax.plot(zs, [additive_folds[z]["additive_mv"] for z in zs], "o-", color="#762A83",
            label=r"low-root fold $A_{SN}$")
    ax.scatter(zs, [thresholds[z] for z in zs], marker="s", color="#1B7837",
               label=r"confirmed delayed exit $A_{exit}$")
    for z in zs:
        ax.plot([z, z], [additive_folds[z]["additive_mv"], thresholds[z]], color="0.7", lw=0.8)
    ax.set(xlabel=r"source $z_R$", ylabel="regional additive A (mV)",
           title="D  Fixed-point and established-cycle exit")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    z = 0.85
    selected_a = thresholds[z]
    for additive, color, label in ((0.0, "#B2182B", "matched A=0"),
                                   (selected_a, "#1B7837", f"delayed A={selected_a:.2f}")):
        result, index = exit_lookup[(z, base_dt, "delayed_exit", 0.0, additive)]
        time = result["time_ms"] * 1.0e-3
        ax.plot(time, 1000.0 * result["rE_khz"][:, index, 0], color=color, lw=0.9, label=label)
        if additive == selected_a:
            ax.plot(time, 1000.0 * result["rE_khz"][:, index, 1], color="#EF8A62", lw=0.75,
                    label="annulus after exit current")
    ax.set(xlabel="time after event-locked fork (s)", ylabel="rE (Hz)",
           title="E  Exit acts after CCO is established")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    below = float(cfg["entry_oracle"]["gate_below_fold_z"])
    lines = [
        "F  Slow–fast interpretation", "",
        f"entry fold zR = {fold['z_regional']:.9f}",
        f"first tested post-fold zR = {below:.5f}",
        f"A_SN / A_exit at .855 = {additive_folds[.855]['additive_mv']:.5f} / {thresholds[.855]:.3f} mV",
        f"A_SN / A_exit at .850 = {additive_folds[.85]['additive_mv']:.5f} / {thresholds[.85]:.3f} mV",
        "",
        "LLL --regional Z fold--> bounded CCO",
        "bounded CCO --delayed regional A--> LLL",
        "",
        "Frozen oracle only: parameter-restoration",
        "fork passed; autonomous Z/M is not yet run.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.0)
    fig.suptitle("Regional resource creates a spatial entry fold; local additive current supplies exit",
                 fontsize=12.5, fontweight="bold")
    stem = figures / "mz_spatial_regional_entry_exit"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    parameters, low_params = _model(cfg)
    prepared_reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(cfg["geometry"]["grid_n"]),
        grid_L_mm=float(cfg["geometry"]["grid_L_mm"]),
        core_radius_mm=float(cfg["geometry"]["core_radius_mm"]),
        theta_rad=np.deg2rad(float(cfg["geometry"]["theta_deg"])),
    )
    prepared = prepare_patch_rhs(prepared_reduction.kernels, parameters)
    low, low_root = _low_template(transfer, low_params)
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{cfg['entry_oracle']['cycle_trace_key']}_state"], dtype=float)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fold = solve_regional_fold(prepared, parameters, transfer)
    fold_dict = fold.as_dict()
    root_rows = []
    for z in map(float, cfg["entry_oracle"]["equilibrium_z_axis"]):
        root_rows.extend(find_regional_equilibria(z, prepared, parameters, transfer))
    for row in root_rows:
        row["rates_khz"] = json.dumps(row["rates_khz"])

    additive_folds = {
        0.855: solve_regional_additive_fold(
            0.855, prepared, parameters, transfer, initial_additive_mv=0.015
        ).as_dict(),
        0.85: solve_regional_additive_fold(
            0.85, prepared, parameters, transfer, initial_additive_mv=0.09
        ).as_dict(),
    }
    for row in additive_folds.values():
        row["left_FA"] = row.pop("left_Fz")

    entry_rows, entry_lookup = _entry_runs(
        cfg, low, cycle, prepared_reduction, parameters, transfer, prepared, output
    )
    exit_rows, source_rows, exit_lookup = _exit_runs(
        cfg, low, cycle, prepared_reduction, parameters, transfer, prepared, output
    )
    thresholds = _confirmed_exit_thresholds(cfg, exit_rows)
    if any(value is None for value in thresholds.values()):
        raise RuntimeError(f"registered additive axis did not close exit bracket: {thresholds}")
    recovery_rows = _recovery_runs(cfg, thresholds, exit_lookup, prepared, transfer)

    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    phases = [float(value) for value in cfg["entry_oracle"]["relative_phase_fractions"]]
    fold_z = float(fold.z_regional)
    below = float(cfg["entry_oracle"]["gate_below_fold_z"])
    above = float(cfg["entry_oracle"]["gate_above_fold_z"])

    def entry_outcome(dt: float, z: float, seed: str, phase: float | None) -> str:
        return next(
            row["outcome"] for row in entry_rows
            if row["dt_ms"] == dt and row["z_regional"] == z
            and row["seed"] == seed and row["phase"] == phase
        )

    a0_persists = all(
        row["outcome"] == "bounded_CCO"
        for row in exit_rows
        if row["mode"] == "delayed_exit" and row["additive_mv"] == 0.0
    )
    gates = {
        "augmented_real_fold_supported": bool(
            fold.residual_inf < 2.0e-8
            and fold.rate_sigma_min < 1.0e-8
            and abs(fold.fast_leading_real_per_ms) < 1.0e-6
            and fold.fast_leading_imag_per_ms < 1.0e-8
            and abs(fold.left_fz) > 1.0e-3
            and abs(fold.left_d2f_vv) > 1.0
            and fold.support_all
        ),
        "low_history_crosses_LLL_to_bounded_CCO": all(
            entry_outcome(dt, above, "L90_relabel", None) == "LLL"
            and entry_outcome(dt, below, "L90_relabel", None) == "bounded_CCO"
            for dt in dts
        ),
        "bounded_CCO_four_phase_anchor": all(
            entry_outcome(dt, 0.85, "CCL_warm_cycle", phase) == "bounded_CCO"
            for dt in dts for phase in phases
        ),
        "matched_A0_cycle_persists": a0_persists,
        "delayed_regional_A_exit_both_z": all(value is not None for value in thresholds.values()),
        "same_interictal_basin_after_parameter_recovery": bool(
            recovery_rows and all(row["outcome"] == "LLL" for row in recovery_rows)
        ),
        "entry_base_half_dt_labels_match": _dt_labels_match(
            entry_rows, ("seed", "z_regional", "phase")
        ),
        "exit_base_half_dt_labels_match": _dt_labels_match(
            exit_rows, ("mode", "z_regional", "phase", "additive_mv")
        ),
    }
    critical = np.asarray(fold.critical_fast_mode, dtype=float)
    critical_localization = {
        "rE_normalized_core_annulus_bath": critical[:3].tolist(),
        "rI_normalized_core_annulus_bath": critical[3:6].tolist(),
        "bath_to_max_rE_abs": float(abs(critical[2]) / np.max(np.abs(critical[:3]))),
    }
    if all(gates.values()):
        status = "FROZEN_REGIONAL_ENTRY_EXIT_GEOMETRY_SUPPORTED_SLOW_LATCH_NOT_YET_RUN"
        decision = "allow_one_minimal_autonomous_regional_ZM_latch_without_EE_changes"
    else:
        status = "FROZEN_REGIONAL_ENTRY_EXIT_GEOMETRY_INCOMPLETE"
        decision = "stop_before_autonomous_slow_latch_and_explain_failed_gate"

    figure = _plot(
        figures, root_rows, fold_dict, entry_rows, entry_lookup, exit_rows, exit_lookup,
        additive_folds, thresholds, cfg,
    )
    _save_csv(output / "regional_equilibrium_roots.csv", root_rows)
    _save_csv(output / "regional_entry_outcomes.csv", entry_rows)
    _save_csv(output / "regional_exit_outcomes.csv", exit_rows)
    _save_csv(output / "regional_exit_source_checkpoints.csv", source_rows)
    _save_csv(output / "regional_recovery_outcomes.csv", recovery_rows)
    summary = {
        "status": status,
        "scientific_layer": "frozen_regional_entry_exit_oracle_not_autonomous_lifecycle",
        "entry_fold": fold_dict,
        "entry_fold_critical_localization": critical_localization,
        "additive_low_root_folds": {f"{z:.4f}": row for z, row in additive_folds.items()},
        "confirmed_delayed_exit_thresholds_mv": {
            f"{z:.4f}": value for z, value in thresholds.items()
        },
        "gates": gates,
        "decision": decision,
        "geometry": {
            "patch_names": list(prepared_reduction.patch_names),
            "patch_cells": list(prepared_reduction.patch_cells),
            "patch_weights": prepared_reduction.kernels.weights().tolist(),
            "K_EE": prepared_reduction.kernels.K_EE.tolist(),
            "K_I": prepared_reduction.kernels.K_I.tolist(),
        },
        "anchors": {"interictal_root": low_root, "cycle_trace_key": cfg["entry_oracle"]["cycle_trace_key"]},
        "interpretation": [
            "regional Z depletion moves the low-state boundary to a localized real-mode fold",
            "the immediate post-fold attractor is a mathematically bounded core-annulus cycle with an unrecruited bath",
            "a delayed additive current confined to the recruited core and annulus terminates the established cycle",
            "the fixed-point A fold and registered dynamic exit bracket align but are reported separately",
            "after exit, restoring z=.90 and A=0 remains in the same low basin",
        ],
        "claim_boundary": [
            "z and additive A are frozen control coordinates; no autonomous Z/M lifecycle has been simulated",
            "the low-history relabel is a frozen basin oracle, not spontaneous slow drift",
            "P3 CLL-to-CCL fast hand-off remains negative; entry here requires regional core-plus-annulus depletion",
            "far bath is retained with its true area and receives no additive current",
            "no E-E weight, kernel, delay, relay, conductance, or external stimulus was changed",
            "bounded_CCO separates mathematical closure from sustained physiological ceiling occupation",
        ],
        "input_sha256": hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "equilibrium_csv": str((output / "regional_equilibrium_roots.csv").relative_to(ROOT)),
            "entry_csv": str((output / "regional_entry_outcomes.csv").relative_to(ROOT)),
            "exit_csv": str((output / "regional_exit_outcomes.csv").relative_to(ROOT)),
            "source_csv": str((output / "regional_exit_source_checkpoints.csv").relative_to(ROOT)),
            "recovery_csv": str((output / "regional_recovery_outcomes.csv").relative_to(ROOT)),
            "trace_glob": str((output / "*_traces.npz").relative_to(ROOT)),
        },
        "config": cfg,
    }
    (output / "regional_entry_exit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_regional_entry_exit.png\n\n"
        "这张图把 P3 的高峰值轨迹按动态闭合性重新判定，并检验区域性 Z 耗竭能否提供入口、recruited-region additive A 能否在事件建立后提供退出。A–C 展示真实的 regional fold、同一低态历史跨 fold 后的轨迹和 basin map；D–E 对比 fixed-point A fold 与 event-locked delayed exit；F 给出当前 slow–fast 解释。\n\n"
        "`bounded_CCO` 要求 core/annulus 周期闭合、周期与峰值稳定、bath 无 return，且不能有持续 120 Hz ceiling 占用。图中 Z/A 仍是 frozen control coordinates，因此它证明的是可供下一步慢变量穿越的入口–退出几何，不是完整自发发作。\n\n"
        "**关注点**：zR 是否在实模 fold 两侧从 LLL 切到 bounded CCO，以及 A 只在 cycle 已建立后打开时能否回到同一个 LLL basin。\n",
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
        "entry_fold_z": summary["entry_fold"]["z_regional"],
        "exit_thresholds_mv": summary["confirmed_delayed_exit_thresholds_mv"],
    }, indent=2))


if __name__ == "__main__":
    main()
