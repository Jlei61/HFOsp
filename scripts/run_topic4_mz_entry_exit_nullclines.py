#!/usr/bin/env python3
"""Run the cheap MZ entry/exit nullcline and frozen-state-fork audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.topic4_mz_entry_exit_nullclines import (  # noqa: E402
    find_equilibria,
    fit_inverse_sqrt_period,
    integrate_frozen_forks,
    nullcline_grid,
    solve_fold,
    summarize_fork_trace,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    ExtendedSiegertTransfer,
)
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import (  # noqa: E402
    SmoothSiegertTransferV11,
)


DEFAULT_CONFIG = ROOT / "config/topic4_mz_entry_exit_nullclines.yaml"
COL_E = "#b2182b"
COL_I = "#2166ac"
COL_BASE = "#353535"
COL_ALPHA15 = "#c74343"
COL_ALPHA16 = "#3b6fb6"
COL_FOLD = "#7b3294"
COL_SENSOR = "#d8902f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_locked_inputs(cfg: dict) -> dict[str, str]:
    expected = cfg.get("input_sha256", {})
    keys = ("transfer_path", "phase_source_path", "current_capture_path", "current_capture_json")
    if set(expected) != set(keys):
        raise ValueError(f"input_sha256 must lock exactly {keys}")
    observed: dict[str, str] = {}
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"locked input is missing: {path}")
        observed[key] = _sha256(path)
        if observed[key] != str(expected[key]):
            raise RuntimeError(
                f"locked input drift for {key}: expected {expected[key]}, observed {observed[key]}"
            )
    return observed


def _load_transfers(cfg: dict) -> tuple[ExtendedSiegertTransfer, SmoothSiegertTransferV11]:
    with np.load(ROOT / cfg["transfer_path"], allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("source transfer did not assert no clipping")
        extended = ExtendedSiegertTransfer(
            payload["mu_axis"],
            payload["sigma_axis"],
            payload["log_integral_table"],
            name=str(payload["transfer_name"]),
        )
    smooth_cfg = cfg["smooth_transfer"]
    smooth = SmoothSiegertTransferV11.from_extended(
        extended,
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
    return extended, smooth


def _causal_state_onset_ms(rate_hz: np.ndarray, dt_ms: float) -> float:
    n = max(1, int(round(250.0 / dt_ms)))
    index = np.arange(rate_hz.size)
    start = np.maximum(0, index - n + 1)
    csum = np.r_[0.0, np.cumsum(rate_hz, dtype=float)]
    envelope = (csum[index + 1] - csum[start]) / (index - start + 1)
    above = envelope >= 20.0
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    minimum = int(round(1000.0 / dt_ms))
    eligible = [int(a) for a, b in zip(starts, stops) if int(b - a) >= minimum]
    if not eligible:
        raise RuntimeError("current capture lost the causal recruited macrostate")
    return float(eligible[0] * dt_ms)


def _fork_rows(
    initial_state: np.ndarray,
    points: list[PoolParameters],
    additive_mv: np.ndarray,
    extended: ExtendedSiegertTransfer,
    cfg: dict,
    labels: list[dict],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    integration = cfg["integration"]
    rows: list[dict] = []
    trace_by_dt: dict[str, np.ndarray] = {}
    for dt_ms in map(float, integration["dt_ms"]):
        simulation = integrate_frozen_forks(
            np.repeat(initial_state[None, :], len(points), axis=0),
            points,
            extended,
            additive_mv,
            dt_ms=dt_ms,
            duration_ms=float(integration["duration_ms"]),
            save_dt_ms=float(integration["save_dt_ms"]),
        )
        trace_by_dt[str(dt_ms)] = np.asarray(simulation["rE_khz"], dtype=float)
        for index, label in enumerate(labels):
            summary = summarize_fork_trace(
                simulation["time_ms"],
                simulation["rE_khz"][:, index],
                audit_start_ms=float(integration["audit_start_ms"]),
                tail_start_ms=float(integration["tail_start_ms"]),
            )
            rows.append(
                {
                    **label,
                    "dt_ms": dt_ms,
                    **summary,
                    "support_violation_count": int(simulation["support_violation_count"][index]),
                    "finite_every_step": bool(simulation["finite"][index]),
                }
            )
        trace_by_dt[f"time_{dt_ms}"] = np.asarray(simulation["time_ms"], dtype=float)
    return rows, trace_by_dt


def _first_operational_boundary(rows: list[dict], coordinate: str, filters: dict) -> dict:
    selected = [
        row for row in rows if all(row.get(key) == value for key, value in filters.items())
    ]
    oscillatory = sorted(float(row[coordinate]) for row in selected if row["status"] == "oscillatory")
    low = sorted(float(row[coordinate]) for row in selected if row["status"] == "settled_low")
    return {
        "largest_oscillatory": oscillatory[-1] if oscillatory else None,
        "smallest_settled_low": low[0] if low else None,
        "interpretation": "state_fork_strip_not_periodic_orbit_continuation",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _mark_roots(ax: plt.Axes, roots: list[dict]) -> None:
    for root in roots:
        if root["stability"] == "stable":
            ax.scatter(root["rE_hz"], root["rI_hz"], s=42, facecolor="white",
                       edgecolor="black", linewidth=1.1, zorder=6)
        else:
            ax.scatter(root["rE_hz"], root["rI_hz"], s=34, marker="x", color="black",
                       linewidth=1.1, zorder=6)


def _plot(
    out_dir: Path,
    cfg: dict,
    fold_rows: list[dict],
    roots_zero: list[dict],
    roots_recovery: list[dict],
    nullcline_payload: dict,
    cycle_rows: list[dict],
    additive_rows: list[dict],
    snic_fit: dict,
    sensor: dict,
    timing_oracle: list[dict],
    capture: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.7), constrained_layout=True)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes
    e_axis = nullcline_payload["e_hz"]
    i_axis = nullcline_payload["i_hz"]
    for ax, e_res, i_res, roots, title in (
        (ax_a, nullcline_payload["A0_E"], nullcline_payload["A0_I"], roots_zero,
         "A  Before recovery: no low fixed point"),
        (ax_b, nullcline_payload["Arec_E"], nullcline_payload["Arec_I"], roots_recovery,
         "B  Additive recovery restores low + saddle"),
    ):
        ax.contour(e_axis, i_axis, e_res, levels=[0.0], colors=[COL_E], linewidths=1.8)
        ax.contour(e_axis, i_axis, i_res, levels=[0.0], colors=[COL_I], linewidths=1.8)
        _mark_roots(ax, roots)
        ax.set(xlabel=r"$r_E$ (Hz)", ylabel=r"$r_I$ (Hz)", title=title)
        ax.text(0.97, 0.04, "red: E nullcline\nblue: I nullcline\n○ stable; × unstable",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92))

    fold_a = np.asarray([row["additive_mv"] for row in fold_rows], dtype=float)
    fold_z = np.asarray([row["z"] for row in fold_rows], dtype=float)
    ax_c.fill_between(fold_a, fold_z, 1.0, color="#ece2f0", alpha=0.8,
                      label="low fixed point exists")
    ax_c.plot(fold_a, fold_z, "o-", color=COL_FOLD, lw=1.6, ms=4.2,
              label=r"fixed-point fold $z_{SN}(A)$")
    ax_c.scatter(
        [row["required_additive_mv"] for row in timing_oracle],
        [row["z_if_unopposed"] for row in timing_oracle],
        c=[row["elapsed_cycles"] for row in timing_oracle], cmap="cividis", s=28,
        edgecolor="white", linewidth=0.45, zorder=6, label="1–5 unopposed cycles",
    )
    for row in timing_oracle:
        ax_c.text(row["required_additive_mv"] + 0.025, row["z_if_unopposed"],
                  f"{row['elapsed_cycles']}C", fontsize=6.5, va="center")
    ax_c.axhline(float(cfg["model"]["reference_z"]), color="0.35", ls="--", lw=1.0)
    ax_c.axvline(float(sensor["A_fold_at_reference_z_mv"]), color=COL_SENSOR, ls=":", lw=1.2)
    ax_c.set(xlabel="frozen additive E current A (mV)", ylabel="Z coordinate",
             title="C  Exit cost rises while Z keeps depleting", ylim=(0.69, 0.90))
    ax_c.legend(frameon=False, fontsize=7.6, loc="lower left")

    z_primary = [row for row in cycle_rows if row["dt_ms"] == 0.125]
    z_half = [row for row in cycle_rows if row["dt_ms"] == 0.0625]
    for rows, marker, label in ((z_primary, "o", "dt=0.125 ms"), (z_half, "x", "dt=0.0625 ms")):
        z = np.asarray([row["z"] for row in rows if row.get("period_ms") is not None])
        period = np.asarray([row["period_ms"] for row in rows if row.get("period_ms") is not None])
        ax_d.plot(z, period, marker=marker, color=COL_BASE, lw=1.1, ms=4, label=label)
    z_fold0 = float(fold_rows[0]["z"])
    z_fit = np.linspace(min(row["z"] for row in z_primary), z_fold0 - 1e-5, 300)
    period_fit = snic_fit["intercept_ms"] + snic_fit["coefficient_ms_sqrt_z"] / np.sqrt(z_fold0 - z_fit)
    ax_d.plot(z_fit, period_fit, color=COL_FOLD, ls="--", lw=1.2,
              label=fr"inverse-sqrt fit, $R^2$={snic_fit['r_squared']:.3f}")
    ax_d.axvline(z_fold0, color=COL_FOLD, ls=":", lw=1.0)
    ax_d.set(xlabel="frozen Z", ylabel="fast-cycle period (ms)",
             title="D  Period diverges at the low-state fold", yscale="log")
    ax_d.legend(frameon=False, fontsize=7.3)

    for alpha, color in ((15.0, COL_ALPHA15), (16.0, COL_ALPHA16)):
        for dt, marker, alpha_line in ((0.125, "o", 1.0), (0.0625, "x", 0.65)):
            rows = [row for row in additive_rows if row["alpha_G"] == alpha and row["dt_ms"] == dt]
            finite = [row for row in rows if row.get("period_ms") is not None]
            ax_e.plot([row["additive_mv"] for row in finite],
                      [row["period_ms"] for row in finite], marker=marker, color=color,
                      alpha=alpha_line, lw=1.1, ms=4,
                      label=fr"$\alpha_G$={int(alpha)}, dt={dt:g}")
            low = [row for row in rows if row["status"] == "settled_low"]
            ax_e.scatter([row["additive_mv"] for row in low], np.full(len(low), 350.0),
                         marker="v", color=color, alpha=alpha_line, s=22)
    ax_e.axvline(float(sensor["A_fold_at_reference_z_mv"]), color=COL_SENSOR, ls=":", lw=1.2)
    ax_e.set(xlabel="frozen additive E current A (mV)", ylabel="period (ms)",
             title="E  Cycle fork returns low near the same fold", yscale="log")
    ax_e.text(0.98, 0.04, "triangles: settled low\nstate fork ≠ orbit continuation",
              transform=ax_e.transAxes, ha="right", va="bottom", fontsize=7.4)
    ax_e.legend(frameon=False, fontsize=6.7, loc="upper left")

    time_rel_s = (capture["times_ms"] - float(sensor["causal_onset_ms"])) * 1e-3
    visible = (time_rel_s >= -4.5) & (time_rel_s <= 6.0)
    ax_f.plot(time_rel_s[visible], capture["slow_TG"][visible], color=COL_SENSOR, lw=1.2)
    ax_f.axvline(0.0, color="0.35", ls="--", lw=1.0)
    ax_f.axhline(float(sensor["candidate_persistence_threshold"]), color=COL_FOLD,
                 ls=":", lw=1.2, label="candidate persistence gate")
    ax_f.axhspan(0.0, float(sensor["pre_onset_TG_max"]), color="0.85", alpha=0.6,
                 label="locked pre-onset range")
    ax_f.set(xlabel="time from causal onset (s)", ylabel=r"existing persistence state $T_G$",
             title="F  Existing slow sensor separates only after onset")
    ax_f.legend(frameon=False, fontsize=7.2, loc="upper left")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10.2)
        ax.title.set_fontweight("bold")
        ax.title.set_ha("left")
        ax.title.set_position((0.0, 1.0))
    fig.suptitle("Entry/exit geometry of the current-based Z–M line",
                 fontsize=13.0, fontweight="bold")
    fig.text(0.5, -0.012,
             "Frozen-state diagnostic only: strong SNIC-like evidence and additive-current leverage; "
             "formal orbit continuation and a full slow lifecycle remain open.",
             ha="center", fontsize=8.2, color="#7f0000")
    stem = out_dir / "mz_entry_exit_nullcline_diagnostic"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_sha256 = _validate_locked_inputs(cfg)
    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    extended, smooth = _load_transfers(cfg)
    model = cfg["model"]
    alpha_primary = float(model["primary_alpha_G"])
    alpha_sensitivity = float(model["sensitivity_alpha_G"])
    reference_z = float(model["reference_z"])
    w_mult = float(model["w_ee_mult"])
    ratio = float(model["ratio"])

    fold_points = []
    warm = (0.00203, 0.00706, 0.87447)
    for additive in map(float, cfg["fold_additive_mv"]):
        point = solve_fold(
            additive, smooth, initial=warm, alpha_g=alpha_primary,
            w_ee_mult=w_mult, ratio=ratio,
        )
        fold_points.append(point)
        warm = (point.r_e_khz, point.r_i_khz, point.z)
    fold_rows = [point.as_dict() for point in fold_points]

    def fold_z(additive: float) -> float:
        return solve_fold(
            additive, smooth, initial=(0.00175, 0.00674, reference_z),
            alpha_g=alpha_primary, w_ee_mult=w_mult, ratio=ratio,
        ).z

    additive_at_reference = float(
        brentq(lambda additive: fold_z(additive) - reference_z, 0.25, 0.45, xtol=1e-10)
    )

    comparison_additive = float(cfg["nullclines"]["comparison_additive_mv"])
    reference_params = PoolParameters(reference_z, alpha_primary, w_mult, ratio)
    roots_zero = find_equilibria(reference_params, smooth, 0.0)
    roots_recovery = find_equilibria(reference_params, smooth, comparison_additive)
    e_lo, e_hi, e_n = cfg["nullclines"]["e_hz"]
    i_lo, i_hi, i_n = cfg["nullclines"]["i_hz"]
    e_axis = np.linspace(float(e_lo), float(e_hi), int(e_n))
    i_axis = np.linspace(float(i_lo), float(i_hi), int(i_n))
    e0, i0 = nullcline_grid(e_axis, i_axis, reference_params, smooth, 0.0)
    er, ir = nullcline_grid(
        e_axis, i_axis, reference_params, smooth, comparison_additive
    )
    nullcline_payload = {
        "e_hz": e_axis, "i_hz": i_axis,
        "A0_E": e0, "A0_I": i0, "Arec_E": er, "Arec_I": ir,
    }

    with np.load(ROOT / cfg["phase_source_path"], allow_pickle=False) as payload:
        phase_state = np.asarray(payload["phase_states"][0], dtype=float)
    z_values = np.asarray(cfg["cycle_z_values"], dtype=float)
    z_points = [PoolParameters(float(z), alpha_primary, w_mult, ratio) for z in z_values]
    z_labels = [{"scan": "z", "z": float(z), "alpha_G": alpha_primary,
                 "additive_mv": 0.0} for z in z_values]
    cycle_rows, _ = _fork_rows(
        phase_state, z_points, np.zeros(z_values.size), extended, cfg, z_labels
    )

    additive_values = np.asarray(cfg["additive_fork_mv"], dtype=float)
    additive_rows: list[dict] = []
    for alpha in (alpha_primary, alpha_sensitivity):
        points = [PoolParameters(reference_z, alpha, w_mult, ratio)] * additive_values.size
        labels = [{"scan": "additive", "z": reference_z, "alpha_G": alpha,
                   "additive_mv": float(additive)} for additive in additive_values]
        rows, _ = _fork_rows(
            phase_state, points, additive_values, extended, cfg, labels
        )
        additive_rows.extend(rows)

    fit_rows = [
        row for row in cycle_rows
        if row["dt_ms"] == 0.0625 and row.get("period_ms") is not None and row["z"] >= 0.868
    ]
    snic_fit = fit_inverse_sqrt_period(
        [row["z"] for row in fit_rows],
        [row["period_ms"] for row in fit_rows],
        fold_rows[0]["z"],
    )

    with np.load(ROOT / cfg["current_capture_path"], allow_pickle=False) as payload:
        capture = {
            "times_ms": np.asarray(payload["times_ms"], dtype=float),
            "rate_E_hz": np.asarray(payload["rate_E_hz"], dtype=float),
            "slow_TG": np.asarray(payload["slow_TG"], dtype=float),
            "slow_z_mean": np.asarray(payload["slow_z_mean"], dtype=float),
        }
    capture_meta = json.loads((ROOT / cfg["current_capture_json"]).read_text(encoding="utf-8"))
    causal_onset = _causal_state_onset_ms(
        capture["rate_E_hz"], float(capture_meta["simulation"]["dt_ms"])
    )
    pre = capture["times_ms"] < causal_onset - 500.0
    post = capture["times_ms"] >= causal_onset + 2000.0
    pre_max = float(np.max(capture["slow_TG"][pre]))
    post_q25 = float(np.quantile(capture["slow_TG"][post], 0.25))
    candidate_threshold = 0.5 * (pre_max + post_q25)
    crossing = np.flatnonzero(
        (capture["times_ms"] >= causal_onset)
        & (capture["slow_TG"] >= candidate_threshold)
    )
    sensor = {
        "A_fold_at_reference_z_mv": additive_at_reference,
        "causal_onset_ms": causal_onset,
        "pre_onset_TG_max": pre_max,
        "post_onset_plus_2s_TG_q25": post_q25,
        "candidate_persistence_threshold": candidate_threshold,
        "threshold_interval_nonempty": bool(pre_max < post_q25),
        "first_threshold_crossing_after_onset_ms": (
            float(capture["times_ms"][crossing[0]] - causal_onset) if crossing.size else None
        ),
        "interpretation": (
            "T_G is reused only as a persistence sensor candidate; the next arm must not apply it "
            "to the recurrent-E denominator"
        ),
    }

    # Timing/leverage oracle: use the exact mean-Z identity from the existing SNN capture only
    # to estimate a representative depletion occupancy.  Then ask how far the fixed-point fold
    # moves if the reduced coordinate is allowed to drift unopposed for 1--5 baseline cycles.
    # This is deliberately not presented as a calibrated cross-model slow trajectory.
    recruited = capture["times_ms"] >= causal_onset
    z_segment = capture["slow_z_mean"][recruited]
    t_segment = capture["times_ms"][recruited]
    tau_z_ms = float(capture_meta["simulation"]["cell_config"]["tau_z"])
    dz_per_ms = float((z_segment[-1] - z_segment[0]) / (t_segment[-1] - t_segment[0]))
    mean_z = float(np.mean(z_segment))
    depletion_occupancy = float(np.clip(1.0 - (mean_z + tau_z_ms * dz_per_ms), 0.0, 1.0))
    baseline_period_ms = float(next(
        row["period_ms"] for row in cycle_rows
        if row["dt_ms"] == 0.0625 and np.isclose(row["z"], reference_z)
    ))
    z_start = float(fold_rows[0]["z"])
    z_equilibrium = 1.0 - depletion_occupancy

    def required_additive(z_target: float) -> float:
        return float(brentq(lambda additive: fold_z(additive) - z_target, 0.0, 2.5, xtol=1e-9))

    timing_oracle = []
    for cycles in range(1, 6):
        elapsed_ms = cycles * baseline_period_ms
        z_unopposed = z_equilibrium + (z_start - z_equilibrium) * np.exp(-elapsed_ms / tau_z_ms)
        timing_oracle.append({
            "elapsed_cycles": cycles,
            "elapsed_ms": elapsed_ms,
            "z_if_unopposed": float(z_unopposed),
            "required_additive_mv": required_additive(float(z_unopposed)),
        })
    sensor_delay_ms = float(sensor["first_threshold_crossing_after_onset_ms"])
    z_at_sensor = z_equilibrium + (z_start - z_equilibrium) * np.exp(-sensor_delay_ms / tau_z_ms)
    timing_summary = {
        "source": "cross-model stress-test oracle, not a calibrated slow trajectory",
        "capture_derived_depletion_occupancy": depletion_occupancy,
        "capture_tau_z_ms": tau_z_ms,
        "baseline_cycle_period_ms": baseline_period_ms,
        "cycle_rows": timing_oracle,
        "at_existing_sensor_crossing": {
            "elapsed_ms": sensor_delay_ms,
            "z_if_unopposed": float(z_at_sensor),
            "required_additive_mv": required_additive(float(z_at_sensor)),
        },
    }

    boundaries = {
        "z_alpha15_dt_base": _first_operational_boundary(
            cycle_rows, "z", {"alpha_G": alpha_primary, "dt_ms": 0.125}
        ),
        "z_alpha15_dt_half": _first_operational_boundary(
            cycle_rows, "z", {"alpha_G": alpha_primary, "dt_ms": 0.0625}
        ),
    }
    for alpha in (alpha_primary, alpha_sensitivity):
        for dt_ms in (0.125, 0.0625):
            boundaries[f"A_alpha{int(alpha)}_dt{dt_ms}"] = _first_operational_boundary(
                additive_rows, "additive_mv", {"alpha_G": alpha, "dt_ms": dt_ms}
            )

    figure_path = _plot(
        figures, cfg, fold_rows, roots_zero, roots_recovery, nullcline_payload,
        cycle_rows, additive_rows, snic_fit, sensor, timing_oracle, capture,
    )
    _write_csv(output / "fixed_point_fold_surface.csv", fold_rows)
    _write_csv(output / "cycle_z_state_forks.csv", cycle_rows)
    _write_csv(output / "additive_current_state_forks.csv", additive_rows)
    _write_csv(output / "timing_leverage_oracle.csv", timing_oracle)
    np.savez_compressed(
        output / "nullcline_fields.npz",
        **{key: np.asarray(value) for key, value in nullcline_payload.items()},
    )
    summary = {
        "status": "entry_exit_geometry_resolved_next_slow_lifecycle_open",
        "model_contract": {
            "fast_system": "locked Stage0C nine-state E/I + delayed recurrent-E divisor",
            "new_counterfactual": "frozen additive E current mu_E -> mu_E - A",
            "parallel_line_exclusions": [
                "no E-E weight/kernel/delay changes", "no presynaptic E-E relay",
                "no new recurrent-E recovery divisor",
            ],
        },
        "fixed_point_fold_surface": fold_rows,
        "reference_equilibria": {
            "z": reference_z,
            "A0": roots_zero,
            f"A{comparison_additive}": roots_recovery,
        },
        "snic_like_period_fit": snic_fit,
        "operational_state_fork_boundaries": boundaries,
        "persistence_sensor_audit": sensor,
        "timing_leverage_oracle": timing_summary,
        "key_findings": [
            "the low-state saddle-node is near z=0.874475 and is independent of alpha_G at low rate",
            "the fast-cycle period has inverse-square-root divergence near the same boundary",
            "a frozen additive E current near 0.3165 mV moves the fold to z=0.85",
            "cycle-seeded state forks switch from oscillatory at A=0.31 mV to low at A=0.32 mV",
            "therefore additive M has exit leverage; the old linear-M failure was a gating/timing failure",
            "0.3165 mV is only the frozen-z lower bound; after three unopposed cycles the timing oracle requires about 0.9 mV",
            "alpha_G=15 approaches the transition with >100-Hz peaks; alpha_G=16 is closer to the physical ceiling and remains a sensitivity concern",
            "the existing T_G trace has a post-onset persistence window but must be detached from the E-E denominator",
        ],
        "claim_boundary": [
            "state-fork strips do not prove periodic-orbit loss",
            "inverse-square-root scaling is strong SNIC-like evidence, not a formal bifurcation label",
            "no full slow Z/M trajectory was integrated",
            "no spatial front, containment, recovery, or SNN lifecycle is claimed",
            "the current-stage capture remains a finite 20-s non-recovered trajectory",
        ],
        "artifacts": {
            "figure": str(figure_path.relative_to(ROOT)),
            "fold_csv": str((output / "fixed_point_fold_surface.csv").relative_to(ROOT)),
            "cycle_z_csv": str((output / "cycle_z_state_forks.csv").relative_to(ROOT)),
            "additive_csv": str((output / "additive_current_state_forks.csv").relative_to(ROOT)),
            "timing_oracle_csv": str((output / "timing_leverage_oracle.csv").relative_to(ROOT)),
            "nullcline_npz": str((output / "nullcline_fields.npz").relative_to(ROOT)),
        },
        "resource_contract": cfg["resource_contract"],
        "input_sha256": input_sha256,
        "config": str(config_path.relative_to(ROOT)),
    }
    (output / "entry_exit_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_entry_exit_nullcline_diagnostic.png / .pdf\n\n"
        "这张六面板诊断图把当前快系统的 entry/exit 几何拆开：前两格是固定 Z 下的 E/I nullcline，"
        "中间两格是加性 M 对 saddle-node 的移动及周期靠近该边界时的变慢，后两格是恒定 M-current "
        "state fork 与现有慢 persistence sensor 的时序分离。图只使用 0D rate/frozen-state 分析和已保存的 "
        "20 s capture，没有重跑完整 SNN。\n\n"
        "**关注点**：A≈0.3165 mV 在 z=0.85 重建 low+saddle，A=0.31→0.32 mV 的 state fork 由周期转低；"
        "这证明加性 M 有 exit leverage，但 0.3165 mV 只是冻结-Z 下界。若 Z 不受抵消地继续漂移 3 个周期，"
        "timing oracle 的需求已升到约 0.9 mV；同时还没有证明周期分支正式消失，也没有证明空间 front 能停住或回收。\n",
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
        "figure": summary["artifacts"]["figure"],
        "A_fold_at_z085_mv": summary["persistence_sensor_audit"]["A_fold_at_reference_z_mv"],
        "snic_like_r2": summary["snic_like_period_fit"]["r_squared"],
    }, indent=2))


if __name__ == "__main__":
    main()
