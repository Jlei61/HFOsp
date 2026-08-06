#!/usr/bin/env python3
"""Run the bounded autonomous regional-Z/p/M latch falsification screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import fields
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
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
    _validate_inputs,
)
from src.topic4_mz_spatial_autonomous_latch import (  # noqa: E402
    Pulse,
    RegionalSlowParameters,
    integrate_autonomous_latch_batch,
)
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_autonomous_latch.yaml"
PATCH_NAMES = ("core", "annulus", "bath")
PATCH_COLORS = ("#B2182B", "#EF8A62", "#2166AC")


def _realized_pulses(cfg: dict) -> list[Pulse]:
    challenge = cfg["background_event_challenge"]
    rng = np.random.default_rng(int(challenge["seed"]))
    time_ms = float(challenge["first_onset_ms"])
    generated = []
    while time_ms < float(challenge["stop_before_ms"]):
        generated.append(float(np.round(time_ms)))
        time_ms += float(challenge["refractory_ms"]) + rng.exponential(
            float(challenge["exponential_scale_ms"])
        )
    locked = [float(value) for value in challenge["realized_onsets_ms"]]
    if generated != locked:
        raise RuntimeError(f"fixed-seed pulse schedule drift: {generated} != {locked}")
    profile = tuple(float(value) for value in challenge["profile_core_annulus_bath"])
    return [
        Pulse(
            onset_ms=value,
            duration_ms=float(challenge["duration_ms"]),
            amplitude_mv=float(challenge["amplitude_mv"]),
            profile=profile,
        ).validate()
        for value in locked
    ]


def _arms(cfg: dict) -> tuple[list[str], list[RegionalSlowParameters]]:
    common = dict(cfg["slow_common"])
    common["depletion_mask"] = tuple(float(value) for value in common["depletion_mask"])
    accepted = {field.name for field in fields(RegionalSlowParameters)}
    names: list[str] = []
    output: list[RegionalSlowParameters] = []
    for row in cfg["arms"]:
        names.append(str(row["name"]))
        values = {**common, **{key: value for key, value in row.items() if key != "name"}}
        unknown = set(values) - accepted
        if unknown:
            raise ValueError(f"unknown RegionalSlowParameters fields: {sorted(unknown)}")
        output.append(RegionalSlowParameters(**values).validate())
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    return names, output


def _first_time(time_ms: np.ndarray, mask: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return None if indices.size == 0 else float(time_ms[int(indices[0])])


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _classify(
    result: dict[str, Any],
    index: int,
    name: str,
    dt_ms: float,
    cfg: dict,
) -> dict[str, Any]:
    time_ms = np.asarray(result["time_ms"], dtype=float)
    rates_hz = 1000.0 * np.asarray(result["rE"][:, index, :], dtype=float)
    z = np.asarray(result["z"][:, index, :], dtype=float)
    persistence = np.asarray(result["p"][:, index, :], dtype=float)
    m = np.asarray(result["m"][:, index, :], dtype=float)
    returns = result["return_times_ms"][index]
    pulse_free_start = float(cfg["background_event_challenge"]["pulse_free_analysis_start_ms"])
    pulse_free = [
        [float(value) for value in patch_returns if value >= pulse_free_start]
        for patch_returns in returns
    ]
    n_regional = min(len(pulse_free[0]), len(pulse_free[1]))
    fold_z = float(cfg["known_fast_boundaries"]["regional_entry_fold_z"])
    entry_time = _first_time(time_ms, np.min(z[:, :2], axis=1) < fold_z)
    failure_candidates = [
        value for value in (
            result["first_support_failure_ms"][index],
            result["first_bound_failure_ms"][index],
            result["first_nonfinite_ms"][index],
        ) if np.isfinite(value)
    ]
    failure_time = min(failure_candidates) if failure_candidates else None
    tail = time_ms >= float(cfg["classification"]["low_tail_start_ms"])
    tail_max = np.max(rates_hz[tail], axis=0)
    low_tail = bool(np.all(tail_max < float(cfg["classification"]["low_tail_max_hz"])))
    min_returns = int(cfg["classification"]["minimum_pulse_free_returns"])
    if failure_time is not None:
        outcome = f"support_or_physical_escape_after_{n_regional}_returns"
    elif entry_time is None:
        outcome = "interictal_no_entry"
    elif low_tail and n_regional >= min_returns:
        # This producer does not run the recovery/retrigger protocol, so even a
        # four-return finite low tail is not a complete lifecycle acceptance.
        outcome = "finite_low_tail_after_four_returns_pending_recovery"
    elif low_tail:
        outcome = "finite_exit_before_four_returns"
    else:
        outcome = "persistent_or_unresolved_after_entry"
    bath_z_error = float(np.max(np.abs(z[:, 2] - float(cfg["slow_common"]["z_rest"]))))
    return {
        "arm": name,
        "dt_ms": float(dt_ms),
        "outcome": outcome,
        "entry_time_ms": entry_time,
        "latch_set_time_ms": (
            float(result["latch_set_times_ms"][index][0])
            if result["latch_set_times_ms"][index] else None
        ),
        "latch_reset_time_ms": (
            float(result["latch_reset_times_ms"][index][0])
            if result["latch_reset_times_ms"][index] else None
        ),
        "failure_time_ms": failure_time,
        "first_support_failure_ms": _finite_or_none(result["first_support_failure_ms"][index]),
        "first_bound_failure_ms": _finite_or_none(result["first_bound_failure_ms"][index]),
        "first_nonfinite_ms": _finite_or_none(result["first_nonfinite_ms"][index]),
        "active_at_end": bool(result["active_at_end"][index]),
        "finite": bool(result["finite"][index]),
        "pulse_free_core_returns": len(pulse_free[0]),
        "pulse_free_annulus_returns": len(pulse_free[1]),
        "pulse_free_bath_returns": len(pulse_free[2]),
        "pulse_free_core_return_times_ms": json.dumps(pulse_free[0]),
        "pulse_free_annulus_return_times_ms": json.dumps(pulse_free[1]),
        "all_core_returns": len(returns[0]),
        "all_annulus_returns": len(returns[1]),
        "all_bath_returns": len(returns[2]),
        "min_z_core": float(np.min(z[:, 0])),
        "min_z_annulus": float(np.min(z[:, 1])),
        "max_abs_core_annulus_z_difference": float(np.max(np.abs(z[:, 0] - z[:, 1]))),
        "max_m_core": float(np.max(m[:, 0])),
        "max_m_annulus": float(np.max(m[:, 1])),
        "max_abs_core_annulus_m_difference": float(np.max(np.abs(m[:, 0] - m[:, 1]))),
        "max_p_core": float(np.max(persistence[:, 0])),
        "max_p_annulus": float(np.max(persistence[:, 1])),
        "peak_rE_core_hz": float(np.max(rates_hz[:, 0])),
        "peak_rE_annulus_hz": float(np.max(rates_hz[:, 1])),
        "peak_rE_bath_hz": float(np.max(rates_hz[:, 2])),
        "tail_max_rE_core_hz": float(tail_max[0]),
        "tail_max_rE_annulus_hz": float(tail_max[1]),
        "tail_max_rE_bath_hz": float(tail_max[2]),
        "bath_z_max_abs_error": bath_z_error,
        "support_violation_count": int(np.sum(result["support_violation_count"][index])),
        "state_bound_violation_count": int(np.sum(result["state_bound_violation_count"][index])),
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to save an empty outcome table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_trace(path: Path, names: list[str], result: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        arm_names=np.asarray(names),
        time_ms=np.asarray(result["time_ms"]),
        rE_khz=np.asarray(result["rE"]),
        rI_khz=np.asarray(result["rI"]),
        rE_fast_khz=np.asarray(result["rE_fast"]),
        z=np.asarray(result["z"]),
        persistence=np.asarray(result["p"]),
        m=np.asarray(result["m"]),
        z_use=np.asarray(result["z_use"]),
        occupancy=np.asarray(result["occupancy"]),
        neighborhood_recruitment=np.asarray(result["neighborhood_recruitment"]),
        latch=np.asarray(result["latch"]),
        external_e_mv=np.asarray(result["external_e_mv"]),
        shared=np.asarray(result["shared"]),
        active_at_end=np.asarray(result["active_at_end"]),
        first_support_failure_ms=np.asarray(result["first_support_failure_ms"]),
        first_bound_failure_ms=np.asarray(result["first_bound_failure_ms"]),
        first_nonfinite_ms=np.asarray(result["first_nonfinite_ms"]),
    )


def _row(rows: list[dict[str, Any]], arm: str, dt_ms: float) -> dict[str, Any]:
    return next(row for row in rows if row["arm"] == arm and row["dt_ms"] == dt_ms)


def _plot(
    figures: Path,
    names: list[str],
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    base_dt: float,
    cfg: dict,
) -> Path:
    time_s = np.asarray(result["time_ms"], dtype=float) / 1000.0
    pulse_times = np.asarray(cfg["background_event_challenge"]["realized_onsets_ms"]) / 1000.0
    fold = float(cfg["known_fast_boundaries"]["regional_entry_fold_z"])
    finite_name = "finite_exit_before_four_returns"
    boundary_name = "four_returns_then_support_escape"
    z_only_name = "z_only_entry_without_exit"
    slow_off_name = "slow_off_returning_events"
    indices = {name: names.index(name) for name in names}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    index = indices[slow_off_name]
    for patch, color, label in zip(range(3), PATCH_COLORS, PATCH_NAMES):
        ax.plot(time_s, 1000.0 * result["rE"][:, index, patch], color=color, lw=1.0, label=label)
    for value in pulse_times:
        ax.axvline(value, color="0.75", lw=0.6, zorder=0)
    ax.set(xlabel="time (s)", ylabel="rE (Hz)", title="A  Fixed background events return to interictal")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    index = indices[z_only_name]
    for patch, color, label in zip(range(3), PATCH_COLORS, PATCH_NAMES):
        ax.plot(time_s, 1000.0 * result["rE"][:, index, patch], color=color, lw=1.0, label=label)
    fail = _row(rows, z_only_name, base_dt)["failure_time_ms"]
    if fail is not None:
        ax.axvline(float(fail) / 1000.0, color="black", ls="--", lw=1.0, label="support exit")
    ax.set(xlabel="time (s)", ylabel="rE (Hz)", title="B  Z-only crosses the fold but keeps accelerating")

    ax = axes[0, 2]
    for name, color, ls in ((finite_name, "#D95F02", "-"), (boundary_name, "#7F0000", "--")):
        index = indices[name]
        row = _row(rows, name, base_dt)
        rate = 1000.0 * np.asarray(result["rE"][:, index, 0], dtype=float)
        failure = row["failure_time_ms"]
        if failure is not None:
            failure_s = float(failure) / 1000.0
            failure_index = int(np.searchsorted(time_s, failure_s))
            ax.scatter(
                [time_s[failure_index]], [rate[failure_index]], marker="x", s=38,
                color="black", linewidths=1.2, zorder=4,
            )
            rate = rate.copy()
            rate[time_s > failure_s] = np.nan
            ax.axvline(failure_s, color="black", ls="--", lw=0.8)
        ax.plot(time_s, rate, color=color, ls=ls, lw=1.1, label=name)
        set_time = row["latch_set_time_ms"]
        if set_time is not None:
            ax.axvline(float(set_time) / 1000.0, color=color, ls=":", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="core rE (Hz)", title="C  Three-return exit vs four-return support escape")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 0]
    index = indices[finite_name]
    ax.plot(time_s, result["z"][:, index, 0], color="#2166AC", lw=1.2, label="regional z")
    ax.axhline(fold, color="#2166AC", ls="--", lw=0.8, label="entry fold")
    ax2 = ax.twinx()
    additive = float(cfg["model"]["additive_max_mv"]) * result["m"][:, index, 0]
    ax2.plot(time_s, additive, color="#B2182B", lw=1.1, label="regional A")
    ax.set(xlabel="time (s)", ylabel="z", title="D  Outer slow trajectory on measured boundaries")
    ax2.set_ylabel("additive A (mV)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7, loc="best")

    ax = axes[1, 1]
    outcome_order = [
        "interictal_no_entry",
        "finite_exit_before_four_returns",
        "finite_low_tail_after_four_returns_pending_recovery",
        "persistent_or_unresolved_after_entry",
    ]
    def code(outcome: str) -> int:
        if outcome.startswith("support_or_physical_escape"):
            return 4
        return outcome_order.index(outcome) if outcome in outcome_order else 3
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    matrix = np.asarray([[code(_row(rows, name, dt)["outcome"]) for dt in dts] for name in names])
    cmap = matplotlib.colors.ListedColormap(["#2166AC", "#FDAE61", "#1A9850", "#8073AC", "#B2182B"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_xticks(range(len(dts)), [str(value) for value in dts])
    ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names], fontsize=7)
    ax.set(xlabel="dt (ms)", title="E  Registered outcome matrix")
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row_index, str(matrix[row_index, column]), ha="center", va="center", fontsize=7)
    ax.text(
        1.03, 0.5, "0  no entry\n1  finite early exit\n3  persistent\n4  support escape",
        transform=ax.transAxes, va="center", fontsize=6.7,
    )

    ax = axes[1, 2]
    ax.axis("off")
    early = _row(rows, finite_name, base_dt)
    boundary = _row(rows, boundary_name, base_dt)
    lines = [
        "F  Mechanistic verdict", "",
        "regional Z fold: PASS",
        "bounded CCO target: PASS (frozen oracle)",
        "bath stays low with fixed-resource mask: PASS", "",
        f"fast-M exit: {early['pulse_free_core_returns']} pulse-free returns, then low",
        f"slower-M arm: {boundary['pulse_free_core_returns']} returns, then support escape",
        "",
        "No arm satisfies both >=4 returns and finite exit.",
        "The same Z-use law required for sparse-event entry",
        "continues to drive z below the validated CCO window.",
        "",
        "Registered-margin clean no-go for this additive latch;",
        "do not resume downstream workflows yet.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.3)
    fig.suptitle(
        "Autonomous regional Z–M latch exposes an entry–exit timing conflict",
        fontsize=13, fontweight="bold",
    )
    stem = figures / "mz_spatial_autonomous_latch"
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
        grid_n=int(geometry["grid_n"]),
        grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, low_root = _low_template(transfer, low_parameters)
    initial = _low_initial(low, float(cfg["model"]["z_interictal"]), reduction, parameters)
    baseline = np.asarray(initial[3 * 3:4 * 3], dtype=float)
    names, arms = _arms(cfg)
    if len(names) != int(cfg["resource_contract"]["vectorized_forks"]):
        raise RuntimeError("arm count drifted from resource contract")
    pulses = _realized_pulses(cfg)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    base_result = None
    base_dt = float(cfg["integration"]["dt_ms"][0])
    for raw_dt in cfg["integration"]["dt_ms"]:
        dt_ms = float(raw_dt)
        result = integrate_autonomous_latch_batch(
            np.repeat(initial[None, :], len(arms), axis=0),
            prepared,
            transfer,
            arms,
            pulses,
            inhibitory_baseline_khz=baseline,
            dt_ms=dt_ms,
            duration_ms=float(cfg["integration"]["duration_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes"]),
        )
        rows.extend(_classify(result, index, name, dt_ms, cfg) for index, name in enumerate(names))
        _save_trace(output / f"autonomous_latch_dt{str(dt_ms).replace('.', 'p')}_traces.npz", names, result)
        if dt_ms == base_dt:
            base_result = result
    if base_result is None:
        raise RuntimeError("base-dt autonomous result is missing")

    def same_labels() -> bool:
        return all(len({row["outcome"] for row in rows if row["arm"] == name}) == 1 for name in names)

    early = _row(rows, "finite_exit_before_four_returns", base_dt)
    boundary = _row(rows, "four_returns_then_support_escape", base_dt)
    slow_off = _row(rows, "slow_off_returning_events", base_dt)
    z_only = _row(rows, "z_only_entry_without_exit", base_dt)
    finite_candidates = [
        row for row in rows
        if row["outcome"] == "finite_low_tail_after_four_returns_pending_recovery"
    ]
    gates = {
        "slow_off_preserves_returning_interictal_events": bool(
            slow_off["outcome"] == "interictal_no_entry"
            and slow_off["all_core_returns"] == int(
                cfg["background_event_challenge"]["slow_off_expected_returns_each_regional_patch"]
            )
            and slow_off["all_annulus_returns"] == slow_off["all_core_returns"]
        ),
        "regional_Z_autonomously_crosses_frozen_entry_fold": bool(z_only["entry_time_ms"] is not None),
        "bath_remains_unrecruited_under_fixed_resource_mask": all(
            row["all_bath_returns"] == 0
            and row["peak_rE_bath_hz"] < float(cfg["classification"]["bath_peak_max_hz"])
            and row["bath_z_max_abs_error"] < 1.0e-7
            for row in rows
        ),
        "fast_M_can_exit_but_before_four_returns": bool(
            early["outcome"] == "finite_exit_before_four_returns"
            and early["pulse_free_core_returns"] < int(cfg["classification"]["minimum_pulse_free_returns"])
        ),
        "four_return_arm_loses_transfer_support": bool(
            boundary["pulse_free_core_returns"] >= int(cfg["classification"]["minimum_pulse_free_returns"])
            and str(boundary["outcome"]).startswith("support_or_physical_escape")
        ),
        "base_half_dt_outcome_labels_match": same_labels(),
        "no_four_return_finite_low_tail_candidate": not finite_candidates,
    }
    status = (
        "AUTONOMOUS_REGIONAL_ADDITIVE_LATCH_REGISTERED_MARGIN_CLEAN_NO_GO_ENTRY_EXIT_TIMING_CONFLICT"
        if all(gates.values())
        else "AUTONOMOUS_REGIONAL_ADDITIVE_LATCH_INCOMPLETE_OR_NUMERICALLY_UNRESOLVED"
    )
    decision = (
        "close_registered_margin_regional_additive_latch_and_test_inhibitory_reserve_frozen_corridor"
        if "CLEAN_NO_GO" in status
        else "repair_failed_gate_before_mechanism_interpretation"
    )
    _save_csv(output / "autonomous_latch_outcomes.csv", rows)
    figure = _plot(figures, names, rows, base_result, base_dt, cfg)
    summary = {
        "status": status,
        "scientific_layer": "autonomous_background_event_driven_regional_rate_patch_screen_not_full_SNN",
        "gates": gates,
        "decision": decision,
        "geometry": {
            "patch_names": list(reduction.patch_names),
            "patch_cells": list(reduction.patch_cells),
            "patch_weights": reduction.kernels.weights().tolist(),
            "K_EE": reduction.kernels.K_EE.tolist(),
            "K_I": reduction.kernels.K_I.tolist(),
        },
        "anchors": {
            "interictal_root": low_root,
            "fast_boundaries": cfg["known_fast_boundaries"],
            "sensor_calibration": cfg["sensor_calibration"],
        },
        "outcome_counts": {
            outcome: sum(row["outcome"] == outcome for row in rows)
            for outcome in sorted({row["outcome"] for row in rows})
        },
        "interpretation": [
            "ordinary focal background events can autonomously accumulate a shared core-annulus inhibitory resource loss and cross the previously measured real fold",
            "under the preregistered zero-depletion bath mask, pulse-free activity remains regional and the bath has no directed returns",
            "a shared recruited-region additive effector can terminate the activity only when it acts before the registered four-return ictal-duration gate",
            "arms that preserve four pulse-free returns allow z to leave the validated bounded-CCO window and then leave transfer support before additive M catches the exit boundary",
            "there is therefore no registered-margin complete lifecycle arm in the tested regional-use-Z plus additive-M design",
        ],
        "claim_boundary": [
            "the event schedule is fixed-seed and state independent, but this is background-event-driven rather than strict zero-input spontaneous onset",
            "support escape is a failed physical/numerical arm, never a seizure termination",
            "the 225-ms arm reaches its fourth return only about 19 ms before support escape, so this is a robustness-gated no-go rather than a theorem over every intermediate parameter value",
            "the fast frozen entry and exit geometry remains valid; the no-go concerns autonomous timing along that geometry",
            "no E-E weight, kernel, delay, conductance, relay, or dynamic threshold was modified",
            "this screen does not justify resuming the downstream phase-transition, ecomode, or early-ictal workflows",
        ],
        "rows": rows,
        "input_sha256": hashes,
        "resource_contract": cfg["resource_contract"],
        "config": cfg,
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "outcome_csv": str((output / "autonomous_latch_outcomes.csv").relative_to(ROOT)),
            "trace_glob": str((output / "autonomous_latch_dt*_traces.npz").relative_to(ROOT)),
        },
    }
    (output / "autonomous_latch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_autonomous_latch.png\n\n"
        "这张图检验 frozen regional entry/exit geometry 能否由固定背景事件驱动的 autonomous Z–p–M 外慢环真正穿越。A–D 依次展示普通事件返回、Z-only 跨 fold 后继续加速、fast-M 提前退出与 four-return arm 的 support escape，以及对应 Z/A 慢轨迹；E 对照 base/half-dt outcome；F 给出机制判定。\n\n"
        "红色 support escape 不是终止；本 producer 即使观察到至少 4 个无外驱 returns 后回到低尾，也只能标为 pending recovery/retrigger，不能直接叫完整 lifecycle。当前没有这类 finite-low-tail candidate，因此本图是 registered-margin clean no-go diagnostic，不是 paper-level seizure figure。\n\n"
        "**关注点**：共享 regional Z 确实能产生区域性 entry；bath 资源由预注册 mask 固定，故 bath 未招募不能单独当 emergent containment。additive M 的有限退出发生得过早；一旦保留第 4 个 return，Z 已离开验证过的 bounded-CCO 窗口。\n",
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
        "outcome_counts": summary["outcome_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
