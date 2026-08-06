#!/usr/bin/env python3
"""Confirm the reserve-compatible q-A corridor with smooth fixed-q M ramps."""

from __future__ import annotations

import argparse
import csv
import itertools
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
    _low_initial,
    _low_template,
    _model,
    _pattern_summary,
    _set_additive,
    _set_recovered_parameters,
    _validate_inputs,
)
from src.topic4_mz_inhibitory_reserve import (  # noqa: E402
    interval_passes_gate,
    safe_q_intervals,
)
from src.topic4_mz_spatial_autonomous_latch import (  # noqa: E402
    RegionalSlowParameters,
    integrate_autonomous_latch_batch,
)
from src.topic4_mz_spatial_entry_exit import solve_regional_additive_fold  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import integrate_frozen_patch_batch  # noqa: E402
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_corridor_r0b.yaml"


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


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a producer table while restoring scalar types used by the gates."""

    def scalar(value: str) -> Any:
        if value == "":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            return float(value)
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _solve_folds(
    q_axis: list[float], cfg: dict, prepared: Any, parameters: Any, transfer: Any,
) -> dict[float, dict[str, Any]]:
    screen = cfg["r0b"]
    output = {}
    for q in q_axis:
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


def _frozen_view(result: dict[str, Any]) -> dict[str, Any]:
    """Expose an autonomous-ramp result to the registered frozen classifier."""

    return {
        "time_ms": result["time_ms"],
        "rE_khz": result["rE"],
        "rI_khz": result["rI"],
        "rE_fast_khz": result["rE_fast"],
        "final_state": result["final_state"],
        "finite": result["finite"],
        "support_violation_count": result["support_violation_count"],
        "state_bound_violation_count": result["state_bound_violation_count"],
        "return_times_ms": result["return_times_ms"],
        "return_states": result["return_states"],
    }


def _ramp_parameters(cfg: dict) -> RegionalSlowParameters:
    screen = cfg["r0b"]
    return RegionalSlowParameters(
        z_rest=float(cfg["model"]["z_interictal"]),
        tau_z_recovery_ms=20000.0,
        tau_z_depletion_ms=7500.0,
        tau_p_ms=750.0,
        occupancy_threshold_khz=float(screen["ramp_occupancy_threshold_khz"]),
        occupancy_width_khz=float(screen["ramp_occupancy_width_khz"]),
        persistence_on=0.115,
        persistence_off=0.030,
        recruitment_on=0.60,
        low_reset_threshold_khz=0.005,
        z_safe=0.885,
        tau_m_up_ms=float(screen["ramp_tau_m_up_ms"]),
        tau_m_down_ms=float(screen["ramp_tau_m_down_ms"]),
        depletion_mask=(1.0, 1.0, 0.0),
        pool_core_annulus_resource=True,
        pool_core_annulus_effector=True,
        enable_z=False,
        enable_m=True,
    ).validate()


def _labels_match(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> bool:
    groups: dict[tuple, set[str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, set()).add(str(row["outcome"]))
    return all(len(values) == 1 for values in groups.values())


def _cartesian_complete(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    axes: tuple[list[float], ...],
) -> bool:
    """Require the exact registered Cartesian product, without duplicates."""

    expected = set(itertools.product(*axes))
    observed = [tuple(float(row[key]) for key in keys) for row in rows]
    return bool(
        len(observed) == len(expected)
        and len(set(observed)) == len(observed)
        and set(observed) == expected
    )


def _all_nonempty(rows: list[dict[str, Any]], predicate: Any) -> bool:
    """A fail-closed replacement for ``all`` on filtered result rows."""

    return bool(rows) and all(bool(predicate(row)) for row in rows)


def _no_failclosed_violation(row: dict[str, Any]) -> bool:
    for patch in ("core", "annulus", "bath"):
        if int(row.get(f"{patch}_support_violations", 0)) != 0:
            return False
        if int(row.get(f"{patch}_bound_violations", 0)) != 0:
            return False
    for key in ("first_support_failure_ms", "first_nonfinite_ms"):
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            return False
    return True


def _build_r0b_gates(
    source_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    ramp_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    cfg: dict,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Build every R0b gate from observed rows and fail closed on omissions."""

    screen = cfg["r0b"]
    formal = cfg["formal_r0_gate"]
    confirm_q = [float(value) for value in screen["confirm_q_axis"]]
    stress_q = [float(value) for value in screen["ramp_stress_q_axis"]]
    all_q = sorted(set(confirm_q + stress_q))
    phases = [float(value) for value in screen["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    offsets = [float(value) for value in screen["threshold_offsets_from_low_fold_mv"]]

    below_candidates = [value for value in offsets if value < 0.0]
    above_candidates = [value for value in offsets if value > 0.0]
    below = max(below_candidates) if below_candidates else None
    above = min(above_candidates) if above_candidates else None
    margin_candidates = [
        value for value in offsets
        if value >= float(formal["minimum_additive_margin_mv"]) - 1.0e-12
    ]
    margin_offset = min(margin_candidates) if margin_candidates else None
    bracket_width = None if below is None or above is None else float(above - below)

    completeness = {
        "source": _cartesian_complete(
            source_rows, ("q", "phase", "dt_ms"), (all_q, phases, dts)
        ),
        "step": _cartesian_complete(
            step_rows,
            ("q", "phase", "dt_ms", "offset_from_low_fold_mv"),
            (confirm_q, phases, dts, offsets),
        ),
        "ramp": _cartesian_complete(
            ramp_rows, ("q", "phase", "dt_ms"), (all_q, phases, dts)
        ),
        "recovery": _cartesian_complete(
            recovery_rows,
            ("source_q", "phase", "dt_ms"),
            (confirm_q, phases, dts),
        ),
    }

    def select(rows: list[dict[str, Any]], **values: float) -> list[dict[str, Any]]:
        return [
            row for row in rows
            if all(
                np.isclose(float(row[key]), value, atol=1.0e-12, rtol=0.0)
                for key, value in values.items()
            )
        ]

    below_rows = [] if below is None else select(
        step_rows, offset_from_low_fold_mv=below
    )
    above_rows = [] if above is None else select(
        step_rows, offset_from_low_fold_mv=above
    )
    margin_rows = [] if margin_offset is None else select(
        step_rows, offset_from_low_fold_mv=margin_offset
    )
    confirm_ramps = [row for row in ramp_rows if float(row["q"]) in confirm_q]
    formal_rows = (
        [row for row in source_rows if float(row["q"]) in confirm_q]
        + step_rows
        + confirm_ramps
        + recovery_rows
    )

    safe_by_q: dict[float, bool] = {}
    for q in confirm_q:
        source_q = select(source_rows, q=q)
        below_q = [] if below is None else select(
            step_rows, q=q, offset_from_low_fold_mv=below
        )
        above_q = [] if above is None else select(
            step_rows, q=q, offset_from_low_fold_mv=above
        )
        margin_q = [] if margin_offset is None else select(
            step_rows, q=q, offset_from_low_fold_mv=margin_offset
        )
        ramp_q = select(ramp_rows, q=q)
        recovery_q = select(recovery_rows, source_q=q)
        q_rows = source_q + below_q + above_q + margin_q + ramp_q + recovery_q
        safe_by_q[q] = bool(
            _all_nonempty(source_q, lambda row: row["outcome"] == "bounded_CCO")
            and _all_nonempty(below_q, lambda row: row["outcome"] == "bounded_CCO")
            and _all_nonempty(above_q, lambda row: row["outcome"] == "LLL")
            and _all_nonempty(margin_q, lambda row: row["outcome"] == "LLL")
            and _all_nonempty(ramp_q, lambda row: row["outcome"] == "LLL")
            and _all_nonempty(
                ramp_q,
                lambda row: float(row["max_additive_mv"])
                >= float(row["low_fold_additive_mv"]),
            )
            and _all_nonempty(
                ramp_q,
                lambda row: float(row["max_abs_fixed_q_error"]) < 1.0e-7,
            )
            and _all_nonempty(recovery_q, lambda row: row["outcome"] == "LLL")
            and _all_nonempty(q_rows, _no_failclosed_violation)
        )

    safe_intervals = safe_q_intervals(
        confirm_q,
        [safe_by_q[q] for q in confirm_q],
        maximum_spacing=float(formal["maximum_q_spacing"]),
    )
    strip_pass = any(
        interval_passes_gate(
            interval,
            minimum_width=float(formal["minimum_q_width"]),
            minimum_nodes=int(formal["minimum_q_nodes"]),
        )
        for interval in safe_intervals
    )
    gates = {
        "tables_form_complete_cartesian_products": all(completeness.values()),
        "source_CCO_all_q_phase_dt": _all_nonempty(
            source_rows, lambda row: row["outcome"] == "bounded_CCO"
        ),
        "instantaneous_bracket_width_within_gate": bool(
            bracket_width is not None
            and bracket_width
            <= float(formal["threshold_bracket_max_mv"]) + 1.0e-12
        ),
        "step_below_fold_remains_CCO": _all_nonempty(
            below_rows, lambda row: row["outcome"] == "bounded_CCO"
        ),
        "step_above_fold_reaches_LLL": _all_nonempty(
            above_rows, lambda row: row["outcome"] == "LLL"
        ),
        "step_registered_margin_reaches_LLL": _all_nonempty(
            margin_rows, lambda row: row["outcome"] == "LLL"
        ),
        "smooth_M_ramp_fixed_q_reaches_LLL": _all_nonempty(
            confirm_ramps, lambda row: row["outcome"] == "LLL"
        ),
        "smooth_M_ramp_crosses_low_fold": _all_nonempty(
            confirm_ramps,
            lambda row: float(row["max_additive_mv"])
            >= float(row["low_fold_additive_mv"]),
        ),
        "formal_rows_have_zero_failclosed_violations": _all_nonempty(
            formal_rows, _no_failclosed_violation
        ),
        "effective_q_is_exactly_frozen_during_ramp": _all_nonempty(
            ramp_rows,
            lambda row: float(row["max_abs_fixed_q_error"]) < 1.0e-7,
        ),
        "parameter_restoration_returns_same_LLL_basin": _all_nonempty(
            recovery_rows, lambda row: row["outcome"] == "LLL"
        ),
        "base_half_dt_labels_match": bool(
            all(completeness.values())
            and _labels_match(step_rows, ("q", "phase", "offset_from_low_fold_mv"))
            and _labels_match(ramp_rows, ("q", "phase"))
            and _labels_match(recovery_rows, ("source_q", "phase"))
        ),
        "continuous_safe_q_strip_from_outcomes_meets_gate": bool(strip_pass),
    }
    diagnostics = {
        "expected_observed_rows": {
            "source": {
                "expected": len(all_q) * len(phases) * len(dts),
                "observed": len(source_rows),
            },
            "step": {
                "expected": len(confirm_q) * len(phases) * len(dts) * len(offsets),
                "observed": len(step_rows),
            },
            "ramp": {
                "expected": len(all_q) * len(phases) * len(dts),
                "observed": len(ramp_rows),
            },
            "recovery": {
                "expected": len(confirm_q) * len(phases) * len(dts),
                "observed": len(recovery_rows),
            },
        },
        "cartesian_complete": completeness,
        "bracket_offsets_mv": {
            "below": below,
            "above": above,
            "width": bracket_width,
        },
        "registered_margin_offset_mv": margin_offset,
        "safe_by_q": {str(q): safe_by_q[q] for q in confirm_q},
        "safe_q_intervals": safe_intervals,
    }
    return gates, diagnostics


def _plot(
    figures: Path,
    folds: dict[float, dict[str, Any]],
    step_rows: list[dict[str, Any]],
    ramp_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    representative: dict[str, np.ndarray],
    gates: dict[str, bool],
    cfg: dict,
) -> Path:
    plt.rcParams.update({"font.size": 8.1, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    confirm_q = [float(value) for value in cfg["r0b"]["confirm_q_axis"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]

    ax = axes[0, 0]
    qs = sorted(folds)
    ax.plot(qs, [folds[q]["additive_mv"] for q in qs], "o-", color="#762A83", label="A_SN(q)")
    for offset, color, label in ((-0.005, "#1B7837", "CCO side"), (0.005, "#2166AC", "LLL side")):
        ax.plot(confirm_q, [folds[q]["additive_mv"] + offset for q in confirm_q], "--", color=color, lw=1.0, label=label)
    ax.set(xlabel="fixed q", ylabel="A (mV)", title="A  Dual-dt instantaneous fold bracket")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    outcome_code = {
        "bounded_CCO": 0,
        "LLL": 1,
        "O_unresolved": 2,
        "physical_or_numerical_failure": 3,
        "ceiling_or_nonclosed": 4,
        "bounded_ceiling_CCO": 4,
    }
    offsets = [float(value) for value in cfg["r0b"]["threshold_offsets_from_low_fold_mv"]]
    matrix = np.full((len(confirm_q), len(offsets) * len(dts)), np.nan)
    for row_index, q in enumerate(confirm_q):
        for dt_index, dt in enumerate(dts):
            for offset_index, offset in enumerate(offsets):
                selected = [
                    row for row in step_rows
                    if row["q"] == q and row["dt_ms"] == dt
                    and row["offset_from_low_fold_mv"] == offset
                ]
                codes = {outcome_code.get(str(row["outcome"]), 2) for row in selected}
                matrix[row_index, dt_index * len(offsets) + offset_index] = max(codes) if codes else np.nan
    cmap = matplotlib.colors.ListedColormap(["#1B7837", "#2166AC", "0.55", "#B2182B", "#E08214"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_yticks(range(len(confirm_q)), [f"{q:.4f}" for q in confirm_q])
    labels = [f"{dt}\n{offset:+.3f}" for dt in dts for offset in offsets]
    ax.set_xticks(range(len(labels)), labels, fontsize=6.1)
    ax.set(xlabel="dt (ms) / A-A_SN (mV)", ylabel="q", title="B  Four-phase fail-closed outcome map")

    ax = axes[0, 2]
    time_s = representative["time_ms"] * 1.0e-3
    ax.plot(time_s, 1000.0 * representative["rE_khz"][:, 0], color="#B2182B", lw=0.9, label="core rE")
    ax.plot(time_s, 1000.0 * representative["rE_khz"][:, 1], color="#EF8A62", lw=0.75, label="annulus rE")
    ax.set(xlabel="time after established-cycle fork (s)", ylabel="rE (Hz)", title="C  Fixed-q occupancy-gated M ramp")
    ax2 = ax.twinx()
    ax2.plot(time_s, representative["additive_mv"], color="#2166AC", lw=1.1, label="A(t)")
    ax2.axhline(float(representative["fold_additive_mv"]), color="#762A83", ls="--", lw=0.8, label="A_SN")
    ax2.set_ylabel("A (mV)")
    crossing = np.flatnonzero(representative["additive_mv"] >= float(representative["fold_additive_mv"]))
    if crossing.size:
        crossing_ms = float(representative["time_ms"][int(crossing[0])])
        ax.axvline(crossing_ms * 1.0e-3, color="#762A83", ls=":", lw=0.8)
        ax.text(
            crossing_ms * 1.0e-3 + 0.01, 0.94,
            f"fold crossed at {crossing_ms:.0f} ms",
            transform=ax.get_xaxis_transform(), fontsize=6.5, va="top",
        )
    ax.set_xlim(0.0, 0.5)
    lines = [
        line for line in ax.get_lines() + ax2.get_lines()
        if not line.get_label().startswith("_")
    ]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=6.5)

    ax = axes[1, 0]
    code = {"LLL": 0, "bounded_CCO": 1, "O_unresolved": 2, "physical_or_numerical_failure": 3, "ceiling_or_nonclosed": 4}
    ramp_q = sorted({float(row["q"]) for row in ramp_rows})
    ramp_matrix = np.full((len(ramp_q), len(dts)), np.nan)
    annotations = np.empty((len(ramp_q), len(dts)), dtype=object)
    for i, q in enumerate(ramp_q):
        for j, dt in enumerate(dts):
            selected = [row for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt]
            values = {code.get(str(row["outcome"]), 2) for row in selected}
            ramp_matrix[i, j] = max(values) if values else np.nan
            annotations[i, j] = f"Amax={min(row['max_additive_mv'] for row in selected):.3f}" if selected else ""
    ax.imshow(ramp_matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_xticks(range(len(dts)), [str(value) for value in dts])
    ax.set_yticks(range(len(ramp_q)), [f"{q:.4f}" for q in ramp_q])
    for i in range(len(ramp_q)):
        for j in range(len(dts)):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=5.9)
    ax.set(xlabel="dt (ms)", ylabel="fixed q", title="D  Smooth-ramp outcome (0=LLL, 3=failure)")

    ax = axes[1, 1]
    for dt, marker in zip(dts, ("o", "s")):
        selected = [row for row in recovery_rows if row["dt_ms"] == dt]
        x = [row["source_q"] for row in selected]
        y = [max(row["core_tail_mean_hz"], row["annulus_tail_mean_hz"], row["bath_tail_mean_hz"]) for row in selected]
        ax.scatter(x, y, s=20, marker=marker, label=f"dt={dt}")
    ax.axhline(float(cfg["classification"]["low_tail_max_hz"]), color="0.5", ls="--", lw=0.8)
    ax.set(xlabel="source q before restoration", ylabel="max patch tail mean (Hz)", title="E  q=.90, A=0 restoration returns to LLL")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["F  R0b verdict", ""] + [
        f"{name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()
    ] + [
        "", "If every gate passes:",
        "safe fixed-q ramp geometry is established;",
        "only R1 q_res/nullcline mapping is unlocked.",
        "No autonomous reserve lifecycle is claimed.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=6.9)
    fig.suptitle("A fixed-q inhibitory reserve corridor supports smooth additive exit", fontsize=12.5, fontweight="bold")
    stem = figures / "mz_inhibitory_reserve_corridor_r0b"
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
    low_initial = _low_initial(low, float(cfg["model"]["z_interictal"]), reduction, parameters)
    inhibitory_baseline = np.asarray(low_initial[9:12], dtype=float)
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{cfg['r0b']['cycle_trace_key']}_state"], dtype=float)

    confirm_q = [float(value) for value in cfg["r0b"]["confirm_q_axis"]]
    stress_q = [float(value) for value in cfg["r0b"]["ramp_stress_q_axis"]]
    all_q = sorted(set(confirm_q + stress_q))
    phases = [float(value) for value in cfg["r0b"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    if len(confirm_q) != len(set(confirm_q)) or len(stress_q) != len(set(stress_q)):
        raise ValueError("registered q axes must be unique")
    folds = _solve_folds(all_q, cfg, prepared, parameters, transfer)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    ramp_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] | None = None
    ramp_arm = _ramp_parameters(cfg)

    for dt in dts:
        source_meta = [(q, phase) for q in all_q for phase in phases]
        source_states = np.asarray([
            _cycle_initial(low, cycle, phase, q, reduction, parameters)
            for q, phase in source_meta
        ])
        source = integrate_frozen_patch_batch(
            source_states, prepared, transfer, dt_ms=dt,
            duration_ms=float(cfg["integration"]["source_prelude_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        checkpoints: dict[tuple[float, float], np.ndarray] = {}
        for index, (q, phase) in enumerate(source_meta):
            row = {"q": q, "phase": phase, "dt_ms": dt, **_pattern_summary(source, index, cfg, prepared, transfer)}
            source_rows.append(row)
            if row["outcome"] != "bounded_CCO":
                continue
            checkpoint, checkpoint_time = _checkpoint(
                source, index, int(cfg["r0b"]["source_min_returns_each_region"])
            )
            checkpoints[(q, phase)] = checkpoint
            row["checkpoint_time_ms"] = checkpoint_time
        if len(checkpoints) != len(source_meta):
            continue

        step_meta = []
        step_states = []
        for q in confirm_q:
            for phase in phases:
                for offset_raw in cfg["r0b"]["threshold_offsets_from_low_fold_mv"]:
                    offset = float(offset_raw)
                    additive = float(folds[q]["additive_mv"]) + offset
                    if additive < 0.0:
                        raise RuntimeError("registered threshold offset produced negative A")
                    step_meta.append({"q": q, "phase": phase, "dt_ms": dt, "offset_from_low_fold_mv": offset, "additive_mv": additive})
                    step_states.append(_set_additive(checkpoints[(q, phase)], additive, parameters))
        if len(step_states) > int(cfg["resource_contract"]["maximum_vectorized_forks"]):
            raise RuntimeError("threshold confirm exceeds the fork limit")
        step_result = integrate_frozen_patch_batch(
            np.asarray(step_states), prepared, transfer, dt_ms=dt,
            duration_ms=float(cfg["integration"]["step_post_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        for index, meta in enumerate(step_meta):
            step_rows.append({**meta, **_pattern_summary(step_result, index, cfg, prepared, transfer)})

        ramp_meta = [{"q": q, "phase": phase, "dt_ms": dt} for q in all_q for phase in phases]
        ramp_initial = np.asarray([checkpoints[(row["q"], row["phase"])] for row in ramp_meta])
        ramp_result = integrate_autonomous_latch_batch(
            ramp_initial, prepared, transfer, [ramp_arm] * len(ramp_meta), [],
            inhibitory_baseline_khz=inhibitory_baseline,
            dt_ms=dt, duration_ms=float(cfg["integration"]["ramp_post_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes_per_batch"]),
            initial_latch_state=np.tile(np.asarray([[True, True, False]]), (len(ramp_meta), 1)),
        )
        ramp_view = _frozen_view(ramp_result)
        for index, meta in enumerate(ramp_meta):
            pattern = _pattern_summary(ramp_view, index, cfg, prepared, transfer)
            additive_trace = float(cfg["model"]["additive_max_mv"]) * np.asarray(ramp_result["m"][:, index, 0], dtype=float)
            z_trace = np.asarray(ramp_result["z"][:, index, :2], dtype=float)
            ramp_rows.append({
                **meta, **pattern,
                "max_additive_mv": float(np.max(additive_trace)),
                "final_additive_mv": float(additive_trace[-1]),
                "low_fold_additive_mv": float(folds[meta["q"]]["additive_mv"]),
                "max_abs_fixed_q_error": float(np.max(np.abs(z_trace - meta["q"]))),
                "first_support_failure_ms": None if not np.isfinite(ramp_result["first_support_failure_ms"][index]) else float(ramp_result["first_support_failure_ms"][index]),
                "first_nonfinite_ms": None if not np.isfinite(ramp_result["first_nonfinite_ms"][index]) else float(ramp_result["first_nonfinite_ms"][index]),
            })
            if dt == dts[0] and meta["q"] == 0.84 and meta["phase"] == 0.0:
                representative = {
                    "time_ms": np.asarray(ramp_result["time_ms"], dtype=np.float32),
                    "rE_khz": np.asarray(ramp_result["rE"][:, index, :], dtype=np.float32),
                    "additive_mv": np.asarray(additive_trace, dtype=np.float32),
                    "fold_additive_mv": np.asarray(float(folds[meta["q"]]["additive_mv"])),
                    "q": np.asarray(meta["q"]),
                }

        recover_indices = [index for index, meta in enumerate(ramp_meta) if meta["q"] in confirm_q]
        recovered_states = np.asarray([
            _set_recovered_parameters(ramp_result["final_state"][index])
            for index in recover_indices
        ])
        recovered = integrate_frozen_patch_batch(
            recovered_states, prepared, transfer, dt_ms=dt,
            duration_ms=float(cfg["integration"]["recovery_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        for recovered_index, source_index in enumerate(recover_indices):
            meta = ramp_meta[source_index]
            recovery_rows.append({
                "source_q": meta["q"], "phase": meta["phase"], "dt_ms": dt,
                "restored_q": 0.90, "restored_additive_mv": 0.0,
                **_pattern_summary(recovered, recovered_index, cfg, prepared, transfer),
            })

    if representative is None:
        raise RuntimeError("registered representative ramp trace is missing")
    gates, gate_diagnostics = _build_r0b_gates(
        source_rows, step_rows, ramp_rows, recovery_rows, cfg
    )
    status = (
        "R0B_RESERVE_COMPATIBLE_2D_CORRIDOR_SUPPORTED_R1_MAPPING_UNLOCKED"
        if all(gates.values())
        else "R0B_RESERVE_CORRIDOR_NO_GO_OR_NUMERICALLY_UNRESOLVED"
    )
    figure = _plot(figures, folds, step_rows, ramp_rows, recovery_rows, representative, gates, cfg)
    _save_csv(output / "r0b_source_cco.csv", source_rows)
    _save_csv(output / "r0b_step_threshold.csv", step_rows)
    _save_csv(output / "r0b_smooth_ramp.csv", ramp_rows)
    _save_csv(output / "r0b_recovery.csv", recovery_rows)
    np.savez_compressed(output / "r0b_representative_ramp.npz", **representative)
    summary = {
        "status": status,
        "scientific_layer": "R0b_frozen_q_step_and_smooth_M_ramp_confirm_not_reserve_dynamics",
        "decision": (
            "start_R1_cycle_use_and_q_res_mapping_only"
            if status.startswith("R0B_RESERVE_COMPATIBLE")
            else "close_or_repair_failed_R0b_gate_before_reserve_dynamics"
        ),
        "gates": gates,
        "gate_diagnostics": gate_diagnostics,
        "confirm_q_axis": confirm_q,
        "ramp_stress_q_axis": stress_q,
        "low_root_folds": {str(q): folds[q] for q in all_q},
        "ramp_summary": [
            {
                "q": q,
                "dt_ms": dt,
                "outcomes": sorted({row["outcome"] for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt}),
                "minimum_max_additive_mv": min(row["max_additive_mv"] for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt),
                "maximum_max_additive_mv": max(row["max_additive_mv"] for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt),
                "minimum_postfork_core_returns": min(row["core_returns"] for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt),
                "maximum_bath_peak_hz": max(row["bath_peak_hz"] for row in ramp_rows if row["q"] == q and row["dt_ms"] == dt),
            }
            for q in all_q for dt in dts
        ],
        "input_sha256": hashes,
        "interictal_root": low_root,
        "geometry": {
            "patch_names": list(reduction.patch_names),
            "patch_cells": list(reduction.patch_cells),
            "patch_weights": reduction.kernels.weights().tolist(),
        },
        "claim_boundary": [
            "R0b fixes q during the smooth M ramp; it does not integrate q_res or D_I dynamics",
            "the ramp reuses the registered 225-ms joint-occupancy M law after an established-cycle checkpoint",
            "parameter restoration is a basin oracle, not early/late retrigger validation",
            "bath q remains fixed for P3 oracle parity and is not emergent containment",
            "no E-E weight, kernel, delay, conductance, relay, or dynamic threshold was changed",
        ],
        "config": cfg,
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "source_csv": str((output / "r0b_source_cco.csv").relative_to(ROOT)),
            "step_csv": str((output / "r0b_step_threshold.csv").relative_to(ROOT)),
            "ramp_csv": str((output / "r0b_smooth_ramp.csv").relative_to(ROOT)),
            "recovery_csv": str((output / "r0b_recovery.csv").relative_to(ROOT)),
            "representative_trace": str((output / "r0b_representative_ramp.npz").relative_to(ROOT)),
        },
    }
    (output / "r0b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_inhibitory_reserve_corridor_r0b.png\n\n"
        "这张图确认 R0a 的瞬时 step fiber 是否能提升为 reserve-compatible smooth path。A–B 检验 low-root fold 两侧的双 dt/四 phase bracket；C–D 用原 225-ms、joint-occupancy-gated M 在固定 q 下平滑穿越 fold；E 检查恢复到 q=.90,A=0 后是否仍回同一 LLL basin；F 列出全部硬门。\n\n"
        "R0b 即使通过也只证明固定-q safe corridor；它还没有加入 D_I/q_res 动力学、背景事件 entry、reset 或 retrigger。bath q 固定用于与 frozen oracle 对齐，不是 emergent containment。\n\n"
        "**关注点**：承重结果是 smooth M ramp，而不是瞬时 A step。只有 confirm q strip 在四 phase、base/half dt 均无 support/bound/nonfinite failure 地回 LLL，才解锁 R1 的 q-nullcline 映射。\n",
        encoding="utf-8",
    )
    return summary


def refresh_existing(config_path: Path) -> dict[str, Any]:
    """Re-aggregate accepted tables after gate/plot code changes, without rerunning ODEs."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / cfg["result_root"]
    summary_path = output / "r0b_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_rows = _load_csv(output / "r0b_source_cco.csv")
    step_rows = _load_csv(output / "r0b_step_threshold.csv")
    ramp_rows = _load_csv(output / "r0b_smooth_ramp.csv")
    recovery_rows = _load_csv(output / "r0b_recovery.csv")
    with np.load(output / "r0b_representative_ramp.npz", allow_pickle=False) as payload:
        representative = {key: np.asarray(payload[key]) for key in payload.files}
    folds = {
        float(q): values for q, values in summary["low_root_folds"].items()
    }
    gates, diagnostics = _build_r0b_gates(
        source_rows, step_rows, ramp_rows, recovery_rows, cfg
    )
    status = (
        "R0B_RESERVE_COMPATIBLE_2D_CORRIDOR_SUPPORTED_R1_MAPPING_UNLOCKED"
        if all(gates.values())
        else "R0B_RESERVE_CORRIDOR_NO_GO_OR_NUMERICALLY_UNRESOLVED"
    )
    figure = _plot(
        output / "figures", folds, step_rows, ramp_rows, recovery_rows,
        representative, gates, cfg,
    )
    summary.update({
        "status": status,
        "decision": (
            "start_R1_cycle_use_and_q_res_mapping_only"
            if status.startswith("R0B_RESERVE_COMPATIBLE")
            else "close_or_repair_failed_R0b_gate_before_reserve_dynamics"
        ),
        "gates": gates,
        "gate_diagnostics": diagnostics,
        "config": cfg,
    })
    sentinel_path = output / "r0b_lower_boundary_sentinel.json"
    if sentinel_path.exists():
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        summary["lower_boundary_sentinel"] = {
            "status": sentinel["status"],
            "highest_confirmed_failing_anchor_q": sentinel[
                "highest_confirmed_failing_q"
            ],
            "lowest_confirmed_safe_anchor_q": sentinel[
                "lowest_confirmed_safe_q"
            ],
            "unresolved_source_q": sentinel["unresolved_source_q"],
            "anchor_bracket_width": sentinel["boundary_bracket_width"],
            "claim": (
                "confirmed anchors only; not a resolved monotone dynamical boundary"
            ),
        }
        summary["artifacts"]["lower_boundary_sentinel"] = str(
            sentinel_path.relative_to(ROOT)
        )
    summary["artifacts"]["figure"] = str(figure.relative_to(ROOT))
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--refresh-existing", action="store_true",
        help="rebuild gates and figure from existing accepted tables",
    )
    args = parser.parse_args()
    summary = (
        refresh_existing(args.config.resolve())
        if args.refresh_existing
        else run(args.config.resolve())
    )
    print(json.dumps({
        "status": summary["status"],
        "gates": summary["gates"],
        "decision": summary["decision"],
        "ramp_summary": summary["ramp_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
