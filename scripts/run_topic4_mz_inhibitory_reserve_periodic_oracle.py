#!/usr/bin/env python3
"""Exact periodic-q hold oracle on hash-locked frozen CCO sensor windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_periodic_oracle.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(q_hold: float, phase: float, dt_ms: float) -> tuple[float, float, float]:
    return (round(float(q_hold), 10), round(float(phase), 10), round(float(dt_ms), 10))


def _expected_keys(cfg: dict[str, Any]) -> set[tuple[float, float, float]]:
    oracle = cfg["oracle"]
    return {
        _key(q, phase, dt)
        for q in oracle["q_hold_axis"]
        for phase in oracle["relative_phase_fractions"]
        for dt in oracle["dt_ms"]
    }


def _validate_hash_locked_inputs(
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    path_keys = (
        "mapping_summary_path",
        "cycle_sensor_replay_path",
        "cycle_measurements_path",
    )
    locks = cfg.get("input_sha256", {})
    if set(locks) != set(path_keys):
        raise ValueError(f"input_sha256 must lock exactly {path_keys}")
    observed: dict[str, str] = {}
    for name in path_keys:
        path = ROOT / str(cfg[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256(path)
        if observed[name] != str(locks[name]):
            raise RuntimeError(f"locked periodic-oracle input drift for {name}: {observed[name]}")

    summary_path = ROOT / str(cfg["mapping_summary_path"])
    mapping = json.loads(summary_path.read_text(encoding="utf-8"))
    status = str(mapping.get("status", ""))
    if "NO_GO" not in status:
        raise RuntimeError("mapping provenance no longer carries the locked entry-ordering no-go")
    if mapping.get("decision") != "preserve_no_go_and_require_noncyclic_R1a_periodic_oracle":
        raise RuntimeError("mapping provenance no longer requests the periodic oracle")
    return observed, mapping


def extract_piecewise_constant_window(
    time_ms: np.ndarray,
    use: np.ndarray,
    start_ms: float,
    stop_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract exact partial ZOH intervals on ``[start_ms, stop_ms)``."""

    time = np.asarray(time_ms, dtype=float)
    sensor = np.asarray(use, dtype=float)
    if time.ndim != 1 or sensor.shape != time.shape or time.size < 2:
        raise ValueError("time and use must be aligned non-empty vectors")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")
    if not np.all(np.isfinite(sensor)) or np.any(sensor < 0.0):
        raise ValueError("use must be finite and non-negative")
    start = float(start_ms)
    stop = float(stop_ms)
    if not np.isfinite(start) or not np.isfinite(stop) or not time[0] <= start < stop <= time[-1]:
        raise ValueError("exact window endpoints must lie inside the replay trace")

    interior = time[(time > start) & (time < stop)]
    boundaries = np.r_[start, interior, stop]
    durations = np.diff(boundaries)
    indices = np.searchsorted(time, boundaries[:-1], side="right") - 1
    if np.any(indices < 0) or np.any(durations <= 0.0):
        raise RuntimeError("invalid exact ZOH window partition")
    return durations, sensor[indices], boundaries


def exact_constant_use_step(
    q_initial: float,
    duration_ms: float,
    use: float,
    *,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
) -> tuple[float, float, float, float]:
    """Return exact q endpoint, time integral, affine slope, and intercept."""

    values = (
        q_initial,
        duration_ms,
        use,
        q_rest,
        q_reserve,
        tau_recovery_ms,
        tau_depletion_ms,
    )
    if not all(np.isfinite(values)):
        raise ValueError("constant-use update inputs must be finite")
    if duration_ms <= 0.0 or use < 0.0:
        raise ValueError("constant-use duration must be positive and use non-negative")
    if not 0.0 < q_reserve < q_rest <= 1.0:
        raise ValueError("q bounds must satisfy 0<q_reserve<q_rest<=1")
    if tau_recovery_ms <= 0.0 or tau_depletion_ms <= 0.0:
        raise ValueError("q time constants must be positive")

    decay = 1.0 / tau_recovery_ms + use / tau_depletion_ms
    drive = q_rest / tau_recovery_ms + q_reserve * use / tau_depletion_ms
    q_infinity = drive / decay
    alpha = float(np.exp(-decay * duration_ms))
    beta = q_infinity * (1.0 - alpha)
    q_final = alpha * q_initial + beta
    integral = (
        q_infinity * duration_ms
        + (q_initial - q_infinity) * (1.0 - alpha) / decay
    )
    return float(q_final), float(integral), alpha, float(beta)


def exact_periodic_hold(
    durations_ms: np.ndarray,
    use: np.ndarray,
    *,
    q_rest: float,
    q_reserve: float,
    tau_recovery_ms: float,
    tau_depletion_ms: float,
    integrated_returns: int,
    initial_q: float,
    convergence_tolerance: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    """Iterate the exact affine window map and evaluate its periodic fixed orbit."""

    durations = np.asarray(durations_ms, dtype=float)
    sensor = np.asarray(use, dtype=float)
    if durations.ndim != 1 or sensor.shape != durations.shape or durations.size == 0:
        raise ValueError("periodic durations and use must be aligned vectors")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("periodic durations must be finite and positive")
    if not np.all(np.isfinite(sensor)) or np.any(sensor < 0.0):
        raise ValueError("periodic use must be finite and non-negative")
    if integrated_returns <= 0 or maximum_iterations <= 0 or convergence_tolerance <= 0.0:
        raise ValueError("periodic convergence contract is invalid")

    map_alpha = 1.0
    map_beta = 0.0
    for duration, value in zip(durations, sensor):
        _, _, alpha, beta = exact_constant_use_step(
            0.0,
            float(duration),
            float(value),
            q_rest=q_rest,
            q_reserve=q_reserve,
            tau_recovery_ms=tau_recovery_ms,
            tau_depletion_ms=tau_depletion_ms,
        )
        map_alpha = alpha * map_alpha
        map_beta = alpha * map_beta + beta
    if not 0.0 <= map_alpha < 1.0:
        raise RuntimeError("periodic affine map is not contractive")

    q_strobe = float(initial_q)
    convergence_error = np.inf
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        next_q = map_alpha * q_strobe + map_beta
        convergence_error = abs(next_q - q_strobe)
        q_strobe = float(next_q)
        if convergence_error <= convergence_tolerance:
            break
    else:
        raise RuntimeError("stroboscopic q map did not converge")

    exact_fixed_point = map_beta / (1.0 - map_alpha)
    fixed_point_error = abs(q_strobe - exact_fixed_point)
    q_trace = [float(q_strobe)]
    elapsed = [0.0]
    integral = 0.0
    q_value = float(q_strobe)
    for duration, value in zip(durations, sensor):
        q_value, segment_integral, _, _ = exact_constant_use_step(
            q_value,
            float(duration),
            float(value),
            q_rest=q_rest,
            q_reserve=q_reserve,
            tau_recovery_ms=tau_recovery_ms,
            tau_depletion_ms=tau_depletion_ms,
        )
        integral += segment_integral
        q_trace.append(float(q_value))
        elapsed.append(elapsed[-1] + float(duration))

    closure_error = abs(q_trace[-1] - q_trace[0])
    window_rho = float(map_alpha)
    return {
        "q_min": float(np.min(q_trace)),
        "q_max": float(np.max(q_trace)),
        "q_mean": float(integral / np.sum(durations)),
        "window_rho": window_rho,
        "per_period_rho": float(window_rho ** (1.0 / float(integrated_returns))),
        "stroboscopic_iterations": int(iterations),
        "stroboscopic_convergence_error": float(convergence_error),
        "fixed_point_error": float(fixed_point_error),
        "window_closure_error": float(closure_error),
        "window_duration_ms": float(np.sum(durations)),
        "window_interval_count": int(durations.size),
        "q_stroboscopic": float(q_strobe),
        "trace_time_ms": np.asarray(elapsed, dtype=float),
        "trace_q": np.asarray(q_trace, dtype=float),
    }


def _load_measurement_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "q_hold",
        "phase",
        "dt_ms",
        "cycle_window_start_ms",
        "cycle_window_stop_ms",
        "integrated_returns",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("cycle CSV is missing exact-window columns")
        rows = []
        for raw in reader:
            rows.append({
                "q_hold": float(raw["q_hold"]),
                "phase": float(raw["phase"]),
                "dt_ms": float(raw["dt_ms"]),
                "cycle_window_start_ms": float(raw["cycle_window_start_ms"]),
                "cycle_window_stop_ms": float(raw["cycle_window_stop_ms"]),
                "integrated_returns": int(raw["integrated_returns"]),
            })
    return rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _status(gates_pass: bool) -> str:
    prefix = (
        "EXACT_PERIODIC_Q_HOLD_ORACLE_SUPPORTED"
        if gates_pass
        else "EXACT_PERIODIC_Q_HOLD_ORACLE_NO_GO_OR_UNRESOLVED"
    )
    return f"{prefix}_BUT_ENTRY_ORDERING_NO_GO_PERSISTS"


def _plot(
    figures: Path,
    rows: list[dict[str, Any]],
    traces: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]],
    gates: dict[str, bool],
    cfg: dict[str, Any],
) -> Path:
    plt.rcParams.update({"font.size": 8.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    thresholds = cfg["formal_gate"]
    q_axis = [float(value) for value in cfg["oracle"]["q_hold_axis"]]
    colors = {0.125: "#2166AC", 0.0625: "#B2182B"}

    ax = axes[0, 0]
    for dt in map(float, cfg["oracle"]["dt_ms"]):
        selected = [row for row in rows if row["dt_ms"] == dt]
        ax.scatter(
            [row["q_hold"] for row in selected],
            [row["q_min"] for row in selected],
            s=21, marker="v", color=colors[dt], alpha=0.75,
            label=f"q min, dt={dt}",
        )
        ax.scatter(
            [row["q_hold"] for row in selected],
            [row["q_max"] for row in selected],
            s=21, marker="^", facecolors="none", edgecolors=colors[dt], alpha=0.75,
            label=f"q max, dt={dt}",
        )
    ax.axhline(float(thresholds["minimum_q"]), color="0.4", ls="--", lw=0.8)
    ax.axhline(float(thresholds["maximum_q"]), color="0.4", ls="--", lw=0.8)
    ax.set(xlabel="target q_hold", ylabel="periodic q extrema", title="A  Exact periodic q range")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)

    ax = axes[0, 1]
    for dt in map(float, cfg["oracle"]["dt_ms"]):
        selected = [row for row in rows if row["dt_ms"] == dt]
        ax.scatter(
            [row["q_hold"] for row in selected],
            [row["q_mean_minus_hold"] for row in selected],
            s=24, color=colors[dt], alpha=0.75, label=f"dt={dt}",
        )
    tolerance = float(thresholds["maximum_abs_mean_minus_hold"])
    ax.axhline(tolerance, color="0.4", ls="--", lw=0.8)
    ax.axhline(-tolerance, color="0.4", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.7)
    ax.set(xlabel="target q_hold", ylabel="period mean - hold", title="B  Exact mean tracks mapped hold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    for dt in map(float, cfg["oracle"]["dt_ms"]):
        selected = [row for row in rows if row["dt_ms"] == dt]
        ax.scatter(
            [row["q_hold"] for row in selected],
            [row["per_period_rho"] for row in selected],
            s=24, color=colors[dt], alpha=0.75, label=f"dt={dt}",
        )
    ax.axhline(float(thresholds["maximum_per_period_rho"]), color="0.4", ls="--", lw=0.8)
    ax.set(xlabel="target q_hold", ylabel="rho per return period", title="C  Stroboscopic contraction")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    preferred = q_axis[len(q_axis) // 2]
    for dt in map(float, cfg["oracle"]["dt_ms"]):
        key = _key(preferred, 0.0, dt)
        time, q = traces[key]
        ax.plot(time * 1.0e-3, q, color=colors[dt], lw=1.0, label=f"dt={dt}")
    ax.axhline(preferred, color="0.4", ls="--", lw=0.8, label="target hold")
    ax.set(xlabel="time within exact 8-return window (s)", ylabel="periodic q", title="D  Representative exact periodic orbit")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.scatter(
        [row["stroboscopic_iterations"] for row in rows],
        [row["stroboscopic_convergence_error"] for row in rows],
        c=[row["q_hold"] for row in rows], cmap="viridis", s=27,
    )
    ax.set_yscale("log")
    ax.set(xlabel="window-map iterations", ylabel="final |q[n+1]-q[n]|", title="E  Stroboscopic convergence error")

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["F  Periodic-q oracle verdict", ""] + [
        f"{name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()
    ] + [
        "",
        "Even if all hold gates pass:",
        "BUT_ENTRY_ORDERING_NO_GO_PERSISTS",
        "fixed sensor + scalar q is not autonomous.",
        "No E-to-E or bath-containment claim.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=6.8)

    fig.suptitle(
        "Exact periodic-q hold is tested without relaxing the entry-ordering no-go",
        fontsize=12.2,
        fontweight="bold",
    )
    stem = figures / "mz_inhibitory_reserve_periodic_oracle"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed_hashes, mapping_summary = _validate_hash_locked_inputs(cfg)
    expected = _expected_keys(cfg)
    if len(expected) != int(cfg["formal_gate"]["expected_combinations"]):
        raise ValueError("configured periodic Cartesian product is not the locked size")

    measurement_rows = _load_measurement_rows(ROOT / str(cfg["cycle_measurements_path"]))
    csv_keys = [_key(row["q_hold"], row["phase"], row["dt_ms"]) for row in measurement_rows]
    if len(csv_keys) != len(set(csv_keys)):
        raise RuntimeError("cycle CSV contains duplicate q/phase/dt rows")
    if set(csv_keys) != expected:
        raise RuntimeError("cycle CSV is not the complete locked 24-combination product")

    returns_expected = int(cfg["oracle"]["integrated_returns_per_window"])
    if any(row["integrated_returns"] != returns_expected for row in measurement_rows):
        raise RuntimeError("cycle CSV contains a non-8-return exact window")

    mappings = {
        round(float(row["q_hold"]), 10): row
        for row in mapping_summary.get("mappings", [])
        if row.get("mapping_status") == "root_found"
    }
    required_q = {round(float(value), 10) for value in cfg["oracle"]["q_hold_axis"]}
    if set(mappings) != required_q:
        raise RuntimeError("mapping summary does not contain exactly one root for each q_hold")

    replay_path = ROOT / str(cfg["cycle_sensor_replay_path"])
    with np.load(replay_path, allow_pickle=False) as payload:
        required_arrays = {"time_ms", "use", "q_hold", "phase", "dt_ms"}
        if set(payload.files) != required_arrays:
            raise RuntimeError("cycle sensor replay schema drift")
        time_ms = np.asarray(payload["time_ms"], dtype=float)
        use_matrix = np.asarray(payload["use"], dtype=float)
        replay_keys = [
            _key(q, phase, dt)
            for q, phase, dt in zip(payload["q_hold"], payload["phase"], payload["dt_ms"])
        ]
    if len(replay_keys) != len(set(replay_keys)) or set(replay_keys) != expected:
        raise RuntimeError("cycle sensor NPZ is not the complete unique 24-combination product")
    if use_matrix.shape != (len(replay_keys), time_ms.size):
        raise RuntimeError("cycle sensor replay use matrix is misaligned")
    replay_index = {key: index for index, key in enumerate(replay_keys)}

    oracle_cfg = cfg["oracle"]
    rows: list[dict[str, Any]] = []
    traces: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for metadata in measurement_rows:
        key = _key(metadata["q_hold"], metadata["phase"], metadata["dt_ms"])
        durations, sensor, boundaries = extract_piecewise_constant_window(
            time_ms,
            use_matrix[replay_index[key]],
            metadata["cycle_window_start_ms"],
            metadata["cycle_window_stop_ms"],
        )
        mapped = mappings[round(metadata["q_hold"], 10)]
        result = exact_periodic_hold(
            durations,
            sensor,
            q_rest=float(oracle_cfg["q_rest"]),
            q_reserve=float(mapped["q_reserve"]),
            tau_recovery_ms=float(oracle_cfg["tau_recovery_ms"]),
            tau_depletion_ms=float(mapped["tau_depletion_ms"]),
            integrated_returns=metadata["integrated_returns"],
            initial_q=float(oracle_cfg["initial_q"]),
            convergence_tolerance=float(oracle_cfg["stroboscopic_tolerance"]),
            maximum_iterations=int(oracle_cfg["maximum_stroboscopic_iterations"]),
        )
        traces[key] = (result.pop("trace_time_ms"), result.pop("trace_q"))
        rows.append({
            **metadata,
            "q_reserve": float(mapped["q_reserve"]),
            "tau_depletion_ms": float(mapped["tau_depletion_ms"]),
            **result,
            "q_mean_minus_hold": float(result["q_mean"] - metadata["q_hold"]),
        })

    thresholds = cfg["formal_gate"]
    key_complete = (
        len(rows) == int(thresholds["expected_combinations"])
        and len({_key(row["q_hold"], row["phase"], row["dt_ms"]) for row in rows})
        == int(thresholds["expected_combinations"])
    )
    gates = {
        "hash_locked_inputs_match": True,
        "complete_24_q_phase_dt_combinations": bool(key_complete),
        "all_periodic_q_min_above_gate": all(
            row["q_min"] >= float(thresholds["minimum_q"]) for row in rows
        ),
        "all_periodic_q_max_below_gate": all(
            row["q_max"] <= float(thresholds["maximum_q"]) for row in rows
        ),
        "all_periodic_means_match_q_hold": all(
            abs(row["q_mean_minus_hold"])
            <= float(thresholds["maximum_abs_mean_minus_hold"])
            for row in rows
        ),
        "all_per_period_rho_below_gate": all(
            row["per_period_rho"] < float(thresholds["maximum_per_period_rho"])
            for row in rows
        ),
    }
    status = _status(all(gates.values()))
    output = ROOT / str(cfg["result_root"])
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure = _plot(figures, rows, traces, gates, cfg)
    csv_path = output / "periodic_q_hold_oracle.csv"
    _save_csv(csv_path, rows)

    representative_keys = [
        key for key in sorted(traces)
        if key[0] == round(float(oracle_cfg["q_hold_axis"][1]), 10)
        and key[1] == 0.0
    ]
    representative_lengths = np.asarray(
        [traces[key][0].size for key in representative_keys], dtype=np.int64
    )
    maximum_points = int(np.max(representative_lengths))
    representative_time = np.full(
        (len(representative_keys), maximum_points), np.nan, dtype=np.float64
    )
    representative_q = np.full_like(representative_time, np.nan)
    for index, key in enumerate(representative_keys):
        length = int(representative_lengths[index])
        representative_time[index, :length] = traces[key][0]
        representative_q[index, :length] = traces[key][1]
    np.savez_compressed(
        output / "periodic_q_representative_traces.npz",
        key=np.asarray(representative_keys, dtype=float),
        length=representative_lengths,
        time_ms=representative_time,
        q=representative_q,
    )
    aggregates = {
        str(q): {
            "q_min": min(row["q_min"] for row in rows if row["q_hold"] == q),
            "q_max": max(row["q_max"] for row in rows if row["q_hold"] == q),
            "maximum_abs_mean_minus_hold": max(
                abs(row["q_mean_minus_hold"]) for row in rows if row["q_hold"] == q
            ),
            "maximum_per_period_rho": max(
                row["per_period_rho"] for row in rows if row["q_hold"] == q
            ),
            "maximum_convergence_error": max(
                row["stroboscopic_convergence_error"] for row in rows if row["q_hold"] == q
            ),
        }
        for q in map(float, oracle_cfg["q_hold_axis"])
    }
    summary = {
        "status": status,
        "scientific_layer": "exact_scalar_periodic_q_hold_on_frozen_CCO_sensor_not_autonomous",
        "decision": "preserve_entry_ordering_no_go_and_stop_before_autonomous_lifecycle",
        "gates": gates,
        "gate_thresholds": thresholds,
        "combination_count": len(rows),
        "aggregates_by_q_hold": aggregates,
        "input_sha256": observed_hashes,
        "mapping_provenance_status": mapping_summary["status"],
        "claim_boundary": [
            "U(t) is a hash-locked frozen CCO sensor replay and does not respond to q(t)",
            "each q update is exact for piecewise-constant saved U over exact CSV return endpoints",
            "rho is the exact affine 8-return-window slope raised to the one-eighth power",
            "passing hold stability cannot repair the locked early-entry ordering conflict",
            "BUT_ENTRY_ORDERING_NO_GO_PERSISTS and autonomous testing remains locked",
            "no E-E, conductance, relay, dynamic threshold, or dynamic bath containment was added",
        ],
        "config": cfg,
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
            "representative_traces": str(
                (output / "periodic_q_representative_traces.npz").relative_to(ROOT)
            ),
        },
    }
    (output / "periodic_q_hold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_inhibitory_reserve_periodic_oracle.png\n\n"
        "这张图用 mapping 阶段 hash-locked 的 frozen CCO `U(t)`，在 CSV 的 exact 8-return 起止窗口内做分段常值解析 q 更新。A–C 检查 q 极值、周期均值与每-return contraction；D–E 展示代表性周期轨迹和 stroboscopic 收敛；F 锁定即使 hold gate 全通过，已有 entry-ordering no-go 仍然有效。\n\n"
        "这里的 sensor 不随 q 反馈，bath mask 与 E→E 均未改变，因此不能把 periodic hold 写成 autonomous seizure lifecycle 或空间 containment。\n\n"
        "**关注点**：先看 24 个 q×phase×dt 组合是否全部满足 `.8325<=q<=.850`、均值误差与 `rho<.9`；无论结果如何，都不能据此解锁 autonomous run。\n",
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
        "aggregates_by_q_hold": summary["aggregates_by_q_hold"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
