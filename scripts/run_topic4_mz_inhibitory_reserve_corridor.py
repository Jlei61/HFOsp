#!/usr/bin/env python3
"""Run the cheap-first two-dimensional inhibitory-reserve corridor screen."""

from __future__ import annotations

import argparse
import csv
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

from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _checkpoint,
    _cycle_initial,
    _load_transfer,
    _low_template,
    _model,
    _pattern_summary,
    _set_additive,
    _validate_inputs,
)
from src.topic4_mz_inhibitory_reserve import safe_q_intervals  # noqa: E402
from src.topic4_mz_spatial_entry_exit import solve_regional_additive_fold  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import integrate_frozen_patch_batch  # noqa: E402
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_corridor.yaml"
PATCH_COLORS = ("#B2182B", "#EF8A62", "#2166AC")


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trace_bytes(n_samples: int, n_forks: int, n_patches: int = 3) -> int:
    return int(n_samples * n_forks * (3 * n_patches + 2) * np.dtype(np.float32).itemsize)


def _folds(cfg: dict, prepared: Any, parameters: Any, transfer: Any) -> dict[float, dict]:
    screen = cfg["r0a"]
    output: dict[float, dict] = {}
    for q in map(float, screen["q_axis"]):
        guess = max(
            1.0e-4,
            float(screen["low_fold_linear_guess_slope_mv_per_q"])
            * (float(screen["low_fold_linear_guess_zero_q"]) - q),
        )
        solved = solve_regional_additive_fold(
            q, prepared, parameters, transfer, initial_additive_mv=guess
        ).as_dict()
        solved["left_FA"] = solved.pop("left_Fz")
        output[q] = solved
    return output


def _aggregate_outcome(rows: list[dict[str, Any]]) -> str:
    outcomes = [str(row["outcome"]) for row in rows]
    if not outcomes:
        return "missing"
    if any(value == "physical_or_numerical_failure" for value in outcomes):
        return "any_failure"
    if all(value == "LLL" for value in outcomes):
        return "all_LLL"
    if all(value == "bounded_CCO" for value in outcomes):
        return "all_bounded_CCO"
    if any(value == "bounded_ceiling_CCO" for value in outcomes):
        return "any_ceiling"
    return "phase_mixed_or_unresolved"


def _fiber_summary(
    q: float,
    fold: dict,
    source_rows: list[dict[str, Any]],
    exit_rows: list[dict[str, Any]],
    additive_margin_mv: float,
) -> dict[str, Any]:
    source = [row for row in source_rows if row["q"] == q]
    source_outcome = _aggregate_outcome(source)
    additive_values = sorted({float(row["additive_mv"]) for row in exit_rows if row["q"] == q})
    aggregates = []
    for additive in additive_values:
        selected = [
            row for row in exit_rows
            if row["q"] == q and row["additive_mv"] == additive
        ]
        aggregates.append({
            "q": q,
            "additive_mv": additive,
            "offset_from_low_fold_mv": additive - float(fold["additive_mv"]),
            "aggregate_outcome": _aggregate_outcome(selected),
        })
    exits = [row["additive_mv"] for row in aggregates if row["additive_mv"] > 0.0 and row["aggregate_outcome"] == "all_LLL"]
    failures = [row["additive_mv"] for row in aggregates if row["aggregate_outcome"] == "any_failure"]
    a_exit = min(exits) if exits else None
    a_fail = min(failures) if failures else None
    safe_exit_values = [
        row["additive_mv"] for row in aggregates
        if row["aggregate_outcome"] == "all_LLL"
        and (a_fail is None or row["additive_mv"] < a_fail)
    ]
    tested_margin = None if a_exit is None or not safe_exit_values else max(safe_exit_values) - a_exit
    margin_pass = bool(
        a_exit is not None
        and tested_margin is not None
        and tested_margin >= additive_margin_mv - 1.0e-12
        and (a_fail is None or a_exit + additive_margin_mv < a_fail)
    )
    a0 = next((row for row in aggregates if row["additive_mv"] == 0.0), None)
    fiber_safe = bool(
        source_outcome == "all_bounded_CCO"
        and a0 is not None and a0["aggregate_outcome"] == "all_bounded_CCO"
        and margin_pass
    )
    return {
        "q": q,
        "low_fold_additive_mv": float(fold["additive_mv"]),
        "source_outcome": source_outcome,
        "matched_a0_outcome": None if a0 is None else a0["aggregate_outcome"],
        "coarse_exit_additive_mv": a_exit,
        "coarse_failure_additive_mv": a_fail,
        "tested_safe_additive_margin_mv": tested_margin,
        "failure_right_censored": a_fail is None,
        "fiber_safe_discovery": fiber_safe,
        "aggregates": aggregates,
    }


def _representative_trace_records(
    records: list[dict[str, Any]], fibers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    safe = [row for row in fibers if row["fiber_safe_discovery"]]
    target_q = min(safe, key=lambda row: abs(row["q"] - 0.84))["q"] if safe else 0.85
    fiber = next(row for row in fibers if row["q"] == target_q)
    target_a = fiber["coarse_exit_additive_mv"]
    selected = [
        row for row in records
        if row["q"] == target_q and row["phase"] == 0.0
        and (row["additive_mv"] == 0.0 or row["additive_mv"] == target_a)
    ]
    return selected


def _plot(
    figures: Path,
    folds: dict[float, dict],
    source_rows: list[dict[str, Any]],
    exit_rows: list[dict[str, Any]],
    fibers: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    cfg: dict,
) -> Path:
    plt.rcParams.update({"font.size": 8.2, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    qs = sorted(folds)

    ax = axes[0, 0]
    ax.plot(qs, [folds[q]["additive_mv"] for q in qs], "o-", color="#762A83", label="low-root fold")
    fail_q = float(cfg["r0a"]["autonomous_failure_q"])
    fail_a = float(cfg["r0a"]["autonomous_failure_additive_mv"])
    ax.scatter([fail_q], [fail_a], marker="x", s=55, color="black", label="autonomous failure")
    ax.set(xlabel="effective inhibitory q", ylabel="additive A (mV)", title="A  Failure is aligned to a 2D q–A fold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    period = []
    peak = []
    for q in qs:
        selected = [row for row in source_rows if row["q"] == q]
        period.append(float(np.median([row["core_period_ms"] for row in selected])))
        peak.append(float(np.max([row["core_peak_hz"] for row in selected])))
    ax.plot(qs, period, "o-", color="#2166AC", label="core period")
    ax.set(xlabel="q", ylabel="period (ms)", title="B  A=0 established CCO remains bounded")
    ax2 = ax.twinx()
    ax2.plot(qs, peak, "s--", color="#B2182B", label="core peak")
    ax2.set_ylabel("peak rE (Hz)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7)

    ax = axes[0, 2]
    palette = {
        "all_bounded_CCO": "#1B7837",
        "all_LLL": "#2166AC",
        "any_failure": "#B2182B",
        "any_ceiling": "#E08214",
        "phase_mixed_or_unresolved": "0.5",
    }
    for fiber in fibers:
        for row in fiber["aggregates"]:
            if row["additive_mv"] == 0.0:
                continue
            ax.scatter(
                row["q"], row["offset_from_low_fold_mv"], s=28,
                color=palette.get(row["aggregate_outcome"], "0.5"),
                edgecolor="white", linewidth=0.35,
            )
    for label in ("all_bounded_CCO", "all_LLL", "any_failure", "phase_mixed_or_unresolved"):
        ax.scatter([], [], s=28, color=palette[label], label=label)
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel="q", ylabel=r"A - A$_{SN}$(q) (mV)", title="C  Instantaneous A steps all reach LLL")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 0]
    safe_q = [row["q"] for row in fibers if row["fiber_safe_discovery"]]
    ax.plot(qs, [folds[q]["additive_mv"] for q in qs], color="#762A83", lw=1.0, label="A_SN")
    ax.plot(qs, [next(row for row in fibers if row["q"] == q)["coarse_exit_additive_mv"] for q in qs], "s-", color="#1B7837", label="coarse A_exit")
    observed_fail = [row for row in fibers if row["coarse_failure_additive_mv"] is not None]
    if observed_fail:
        ax.scatter([row["q"] for row in observed_fail], [row["coarse_failure_additive_mv"] for row in observed_fail], marker="x", color="#B2182B", label="first A_fail")
    ax.scatter(safe_q, [next(row for row in fibers if row["q"] == q)["coarse_exit_additive_mv"] for q in safe_q], s=65, facecolors="none", edgecolors="#1B7837", label="margin-safe node")
    ax.set(xlabel="q", ylabel="A (mV)", title="D  Coarse safe exit fibers")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 1]
    for record, color, label in zip(trace_records, ("#B2182B", "#2166AC"), ("matched A=0", "instantaneous fold step")):
        time_s = record["time_ms"] * 1.0e-3
        ax.plot(time_s, 1000.0 * record["rE_khz"][:, 0], color=color, lw=0.9, label=f"{label}, A={record['additive_mv']:.3f}")
        ax.plot(time_s, 1000.0 * record["rE_khz"][:, 1], color=PATCH_COLORS[1], lw=0.55, alpha=0.7)
    if trace_records:
        ax.set_title(f"E  Representative event-locked fork at q={trace_records[0]['q']:.4f}")
    else:
        ax.set_title("E  No representative safe fiber")
    ax.set(xlabel="time after fork (s)", ylabel="rE (Hz)")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 2]
    ax.axis("off")
    safe_nodes = [row["q"] for row in fibers if row["fiber_safe_discovery"]]
    lines = [
        "F  R0a mechanistic verdict", "",
        f"q nodes tested: {len(qs)}; phases: 4; dt: 0.125 ms",
        f"bounded A=0 source nodes: {sum(row['source_outcome']=='all_bounded_CCO' for row in fibers)}/{len(fibers)}",
        f"margin-safe coarse fibers: {len(safe_nodes)}/{len(fibers)}",
        f"safe q nodes: {safe_nodes}", "",
        "R0a is discovery only.",
        "R0b must add midpoint q nodes, half dt,",
        "smooth occupancy-gated M ramp, threshold",
        "refinement, and recovery forks.", "",
        "No q_res dynamics or E-to-E changes were run.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=7.8)
    fig.suptitle("Inhibitory reserve requires a two-dimensional q–A safe corridor", fontsize=12.5, fontweight="bold")
    stem = figures / "mz_inhibitory_reserve_corridor_r0a"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    parameters, low_parameters = _model(cfg)
    geometry = cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, low_root = _low_template(transfer, low_parameters)
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{cfg['r0a']['cycle_trace_key']}_state"], dtype=float)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    q_axis = [float(value) for value in cfg["r0a"]["q_axis"]]
    phases = [float(value) for value in cfg["r0a"]["relative_phase_fractions"]]
    if len(set(q_axis)) != len(q_axis) or len(set(phases)) != len(phases):
        raise ValueError("q and phase axes must be unique")
    dt = float(cfg["integration"]["discovery_dt_ms"])
    folds = _folds(cfg, prepared, parameters, transfer)

    source_meta = [(q, phase) for q in q_axis for phase in phases]
    source_states = np.asarray([
        _cycle_initial(low, cycle, phase, q, reduction, parameters)
        for q, phase in source_meta
    ])
    source_result = integrate_frozen_patch_batch(
        source_states, prepared, transfer, dt_ms=dt,
        duration_ms=float(cfg["integration"]["source_prelude_ms"]),
        save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
        section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
        rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
    )
    source_rows = []
    checkpoints: dict[tuple[float, float], np.ndarray] = {}
    for index, (q, phase) in enumerate(source_meta):
        row = {"q": q, "phase": phase, **_pattern_summary(source_result, index, cfg, prepared, transfer)}
        source_rows.append(row)
        if row["outcome"] == "bounded_CCO":
            checkpoint, checkpoint_time = _checkpoint(
                source_result, index, int(cfg["r0a"]["source_min_returns_each_region"])
            )
            checkpoints[(q, phase)] = checkpoint
            row["checkpoint_time_ms"] = checkpoint_time
        else:
            row["checkpoint_time_ms"] = None

    exit_rows: list[dict[str, Any]] = []
    phase0_traces: list[dict[str, Any]] = []
    q_per_batch = int(cfg["resource_contract"]["q_nodes_per_exit_batch"])
    for start in range(0, len(q_axis), q_per_batch):
        batch_q = q_axis[start:start + q_per_batch]
        metadata = []
        states = []
        for q in batch_q:
            if not all((q, phase) in checkpoints for phase in phases):
                continue
            additive_axis = [0.0] + [
                float(folds[q]["additive_mv"]) + float(offset)
                for offset in cfg["r0a"]["additive_offsets_from_low_fold_mv"]
            ]
            additive_axis = sorted(set(round(value, 12) for value in additive_axis))
            for phase in phases:
                for additive in additive_axis:
                    metadata.append({"q": q, "phase": phase, "additive_mv": additive})
                    states.append(_set_additive(checkpoints[(q, phase)], additive, parameters))
        if not states:
            continue
        if len(states) > int(cfg["resource_contract"]["maximum_vectorized_forks"]):
            raise RuntimeError("exit batch exceeds the registered fork limit")
        n_samples = int(round(float(cfg["integration"]["exit_post_ms"]) / float(cfg["integration"]["save_dt_ms"]))) + 1
        if _trace_bytes(n_samples, len(states)) > int(cfg["resource_contract"]["max_trace_bytes_per_batch"]):
            raise MemoryError("registered exit batch exceeds trace memory budget")
        result = integrate_frozen_patch_batch(
            np.asarray(states), prepared, transfer, dt_ms=dt,
            duration_ms=float(cfg["integration"]["exit_post_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        for index, meta in enumerate(metadata):
            exit_rows.append({**meta, **_pattern_summary(result, index, cfg, prepared, transfer)})
            if meta["phase"] == 0.0:
                phase0_traces.append({
                    **meta,
                    "time_ms": np.asarray(result["time_ms"], dtype=np.float32),
                    "rE_khz": np.asarray(result["rE_khz"][:, index, :], dtype=np.float32),
                })

    fibers = [
        _fiber_summary(
            q, folds[q], source_rows, exit_rows,
            float(cfg["r0a"]["additive_margin_mv"]),
        )
        for q in q_axis
    ]
    representative = _representative_trace_records(phase0_traces, fibers)
    intervals = safe_q_intervals(
        q_axis, [row["fiber_safe_discovery"] for row in fibers],
        maximum_spacing=max(np.diff(sorted(q_axis))),
    )
    status = (
        "R0A_2D_STEP_FIBER_DISCOVERY_SUPPORTED_R0B_RAMP_CONFIRM_REQUIRED"
        if any(row["fiber_safe_discovery"] for row in fibers)
        else "R0A_NO_MARGIN_SAFE_EXIT_FIBER_CLOSE_RESERVE_ROUTE"
    )
    figure = _plot(figures, folds, source_rows, exit_rows, fibers, representative, cfg)
    _save_csv(output / "r0a_source_cco.csv", source_rows)
    _save_csv(output / "r0a_event_locked_exit.csv", exit_rows)
    _save_csv(
        output / "r0a_q_fibers.csv",
        [{key: value for key, value in row.items() if key != "aggregates"} for row in fibers],
    )
    if representative:
        np.savez_compressed(
            output / "r0a_representative_traces.npz",
            q=np.asarray([row["q"] for row in representative]),
            additive_mv=np.asarray([row["additive_mv"] for row in representative]),
            time_ms=np.asarray([row["time_ms"] for row in representative]),
            rE_khz=np.asarray([row["rE_khz"] for row in representative]),
        )
    summary = {
        "status": status,
        "scientific_layer": "R0a_frozen_2D_q_A_corridor_discovery_not_reserve_dynamics",
        "decision": (
            "run_midpoint_half_dt_smooth_M_ramp_threshold_and_recovery_R0b_only"
            if status.startswith("R0A_2D") else "close_inhibitory_reserve_before_autonomous_test"
        ),
        "gates": {
            "all_source_q_nodes_are_bounded_CCO": all(row["source_outcome"] == "all_bounded_CCO" for row in fibers),
            "at_least_one_margin_safe_exit_fiber": any(row["fiber_safe_discovery"] for row in fibers),
            "autonomous_failure_is_near_low_root_fold": abs(
                float(cfg["r0a"]["autonomous_failure_additive_mv"])
                - float(np.interp(float(cfg["r0a"]["autonomous_failure_q"]), sorted(folds), [folds[q]["additive_mv"] for q in sorted(folds)]))
            ) <= 0.01,
            "r0a_is_base_dt_discovery_only": True,
        },
        "low_root_folds": {str(q): folds[q] for q in q_axis},
        "fibers": fibers,
        "coarse_safe_intervals": intervals,
        "input_sha256": hashes,
        "interictal_root": low_root,
        "geometry": {
            "patch_names": list(reduction.patch_names),
            "patch_cells": list(reduction.patch_cells),
            "patch_weights": reduction.kernels.weights().tolist(),
        },
        "claim_boundary": [
            "R0a freezes effective q and A; it does not integrate D_I or q_res dynamics",
            "base dt and a coarse A axis are discovery only; R0b half-dt threshold refinement and recovery remain required",
            "instantaneous A steps can bypass the simultaneous slow q-A passage seen in the autonomous failure; fixed-q smooth M ramps are a required R0b gate",
            "bath q is fixed for parity with the accepted regional oracle, so bath-low is not emergent containment",
            "no E-E weight, kernel, delay, conductance, relay, or dynamic threshold was changed",
        ],
        "config": cfg,
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "source_csv": str((output / "r0a_source_cco.csv").relative_to(ROOT)),
            "exit_csv": str((output / "r0a_event_locked_exit.csv").relative_to(ROOT)),
            "fiber_csv": str((output / "r0a_q_fibers.csv").relative_to(ROOT)),
        },
    }
    (output / "r0a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_inhibitory_reserve_corridor_r0a.png\n\n"
        "这张图把原先的一维 Z-floor 假设改成二维 `(q,A)` frozen geometry。A 显示 autonomous failure 与 low-root fold 的对齐；B 验证 A=0 established CCO；C–D 给出四 phase 的瞬时 event-locked A-step fiber；E 展示一个代表性 matched-cycle 与 fold step；F 锁定 R0a 只解锁 R0b。\n\n"
        "所有 q 的 step exit 都几乎从 `A_SN(q)` 开始，这说明瞬时跳变会绕过原 autonomous arm 的 simultaneous slow q-A passage。当前图不含 fixed-q smooth M ramp、reserve dynamics、half-dt 或 recovery/retrigger；bath q 被固定用于 oracle parity，不能把 bath-low 写成 emergent containment。\n\n"
        "**关注点**：R0b 必须证明原 225-ms occupancy-gated M 在固定 q 下平滑穿越 fold 仍安全，再确认连续 q strip 的双 dt 与 recovery；step fiber 单独不能解锁 q_res。\n",
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
        "decision": summary["decision"],
        "coarse_safe_intervals": summary["coarse_safe_intervals"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
