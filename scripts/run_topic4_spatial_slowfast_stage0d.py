#!/usr/bin/env python3
"""Run the locked prospective Stage-0D local-basin replication."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0d")

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
    _classifier,
    _run_resolution,
    _save_simulation,
    _sha256,
    _write_csv,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters, build_state_forks  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer  # noqa: E402
from src.topic4_spatial_slowfast_stage0d import (  # noqa: E402
    CENTRE,
    HISTORIES,
    MANHATTAN_NEIGHBOURS,
    PHASES,
    audited_single_resolution_status,
    build_local_battery,
    integrate_full_state_trace,
    point_metric_compatibility,
    select_phase_states,
    summarize_parameter_point,
    temporal_amplitude_status,
)


DEFAULT_CONFIG = ROOT / "config/topic4_spatial_slowfast_stage0d.yaml"
SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0d-local-basin-replication-design.md"
MODULE = ROOT / "src/topic4_spatial_slowfast_stage0d.py"
RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0d.py"
TRANSFER_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c_transfer.py"
LOCKED_POINTS = tuple((z, alpha) for z in (0.84, 0.85, 0.86) for alpha in (15.0, 16.0, 17.0))


def _validate_config(cfg: dict) -> None:
    points = tuple(
        (float(z), float(alpha))
        for z in cfg["parameter_grid"]["z"]
        for alpha in cfg["parameter_grid"]["alpha_G"]
    )
    if points != LOCKED_POINTS:
        raise ValueError("Stage0D parameter grid drifted")
    centre = cfg["centre"]
    if (float(centre["z"]), float(centre["alpha_G"]), str(centre["initial_label"])) != (
        0.85,
        16.0,
        "root_0_plus",
    ):
        raise ValueError("Stage0D discovery centre drifted")
    phase = cfg["phase_source"]
    expected_phase = (0.125, 12000.0, 40, 7200.0, 20.0, 10.0, 300.0)
    observed_phase = tuple(
        float(phase[key]) if key != "save_stride" else int(phase[key])
        for key in (
            "dt_ms",
            "duration_ms",
            "save_stride",
            "tail_start_ms",
            "peak_height_hz",
            "peak_prominence_hz",
            "peak_min_distance_ms",
        )
    )
    if observed_phase != expected_phase or tuple(float(value) for value in phase["phases"]) != PHASES:
        raise ValueError("Stage0D phase-source contract drifted")
    if tuple(cfg["battery"]["histories"]) != HISTORIES or float(cfg["battery"]["perturbation_fraction"]) != 0.03:
        raise ValueError("Stage0D battery drifted")
    expected_runs = {
        "screen": (0.25, 6000.0, 20),
        "confirm": (0.125, 24000.0, 40),
        "dt_half": (0.0625, 24000.0, 80),
    }
    for name, expected in expected_runs.items():
        row = cfg[name]
        if (float(row["dt_ms"]), float(row["duration_ms"]), int(row["save_stride"])) != expected:
            raise ValueError(f"Stage0D {name} contract drifted")
    acceptance = cfg["acceptance"]
    expected_acceptance = {
        "exact_max_abs_error_hz": 0.25,
        "exact_p99_relative_error": 0.02,
        "rate_abs_hz": 1.0,
        "rate_relative": 0.10,
        "frequency_abs_hz": 0.25,
        "frequency_relative": 0.10,
        "amplitude_abs_hz": 5.0,
        "amplitude_relative": 0.10,
        "minimum_off_orbit_histories": 2,
        "minimum_perturbation_families": 2,
        "minimum_phase_ids": 2,
    }
    if acceptance != expected_acceptance:
        raise ValueError("Stage0D acceptance gates drifted")
    if int(cfg["resource_contract"]["blas_threads"]) != 1 or float(cfg["resource_contract"]["max_memory_gib"]) != 4.0:
        raise ValueError("Stage0D resource contract drifted")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("Stage0D scope expansion is forbidden")


def _verify_locked_inputs(cfg: dict) -> dict[str, dict[str, str | bool]]:
    v1_root = ROOT / cfg["v1_1_root"]
    paths = {
        "root_continuation": ROOT / cfg["root_continuation"],
        "extra_fine_transfer": v1_root / "extended_transfer_extra_fine.npz",
        "v1_1_confirm_trace": v1_root / "state_fork_confirm_extra_fine_traces.npz",
        "transfer_module": TRANSFER_MODULE,
    }
    output: dict[str, dict[str, str | bool]] = {}
    for name, path in paths.items():
        observed = _sha256(path)
        expected = str(cfg["locked_hashes"][name])
        output[name] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "pass": observed == expected,
        }
    if not all(bool(row["pass"]) for row in output.values()):
        raise RuntimeError("locked Stage0D upstream provenance mismatch")
    return output


def _load_transfer(path: Path) -> ExtendedSiegertTransfer:
    with np.load(path, allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("Stage0D transfer provenance does not assert no clipping")
        return ExtendedSiegertTransfer(
            payload["mu_axis"],
            payload["sigma_axis"],
            payload["log_integral_table"],
            name="extra_fine",
        )


def _load_centre_root_plus(cfg: dict) -> tuple[dict, np.ndarray, PoolParameters]:
    rows = json.loads((ROOT / cfg["root_continuation"]).read_text(encoding="utf-8"))
    points = [
        row
        for row in rows
        if np.isclose(float(row["z"]), CENTRE[0]) and np.isclose(float(row["alpha_G"]), CENTRE[1])
    ]
    if len(points) != 1:
        raise RuntimeError("centre root artifact is not unique")
    metadata, states, params = build_state_forks(points)
    matches = [index for index, row in enumerate(metadata) if row["initial_label"] == "root_0_plus"]
    if len(matches) != 1:
        raise RuntimeError("centre root_0_plus state is not unique")
    index = matches[0]
    return metadata[index], states[index], params[index]


def _exact_by_local_fork(audit: dict) -> dict[int, dict]:
    return {int(row["fork_index"]): row for row in audit.get("per_fork", [])}


def _source_phase(
    cfg: dict,
    transfer: ExtendedSiegertTransfer,
    thresholds,
    output: Path,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    meta, state, params = _load_centre_root_plus(cfg)
    phase_cfg = cfg["phase_source"]
    standard, rows, exact = _run_resolution(
        [meta], [state], [params], transfer, phase_cfg, thresholds, "phase_source"
    )
    full = integrate_full_state_trace(
        state,
        params,
        transfer,
        dt_ms=float(phase_cfg["dt_ms"]),
        duration_ms=float(phase_cfg["duration_ms"]),
        save_stride=int(phase_cfg["save_stride"]),
    )
    shared = {
        "rE_khz": full["state"][:, 0],
        "rI_khz": full["state"][:, 1],
        "rE_fast_khz": full["state"][:, 6],
        "mu_G": full["state"][:, 7],
        "S_G": full["state"][:, 8],
    }
    parity = {key: float(np.max(np.abs(shared[key] - standard[key][:, 0]))) for key in shared}
    rate_ok = max(parity["rE_khz"], parity["rI_khz"], parity["rE_fast_khz"]) <= float(
        phase_cfg["parity_rate_atol_khz"]
    )
    pool_ok = max(parity["mu_G"], parity["S_G"]) <= float(phase_cfg["parity_pool_atol"])

    v1_path = ROOT / cfg["v1_1_root"] / "state_fork_confirm_extra_fine_traces.npz"
    with np.load(v1_path, allow_pickle=False) as prior:
        prior_parity = {
            key: float(np.max(np.abs(np.asarray(standard[key]) - np.asarray(prior[key]))))
            for key in ("rE_khz", "rI_khz", "rE_fast_khz", "mu_G", "S_G")
        }
    source_exact = _exact_by_local_fork(exact).get(0, {"pass": False})
    source_status = audited_single_resolution_status(rows[0], exact_error_pass=bool(source_exact.get("pass", False)))
    source_valid = bool(
        rate_ok
        and pool_ok
        and max(prior_parity.values()) <= 1e-7
        and source_status == "candidate_survives"
        and rows[0]["classification"] == "bounded_oscillatory_candidate"
    )
    phase_rows: list[dict] = []
    phase_states = np.empty((0, 9))
    phase_error = None
    if source_valid:
        try:
            phase_states, phase_rows = select_phase_states(
                full["time_ms"],
                full["state"],
                tail_start_ms=float(phase_cfg["tail_start_ms"]),
                peak_height_hz=float(phase_cfg["peak_height_hz"]),
                peak_prominence_hz=float(phase_cfg["peak_prominence_hz"]),
                peak_min_distance_ms=float(phase_cfg["peak_min_distance_ms"]),
                phases=phase_cfg["phases"],
            )
        except ValueError as exc:
            source_valid = False
            phase_error = str(exc)
    source_summary = {
        "valid": source_valid,
        "source_metadata": meta,
        "classification": rows[0],
        "direct_exact": source_exact,
        "source_status": source_status,
        "full_trace_vs_authoritative_max_abs": parity,
        "preserved_v1_1_vs_authoritative_max_abs": prior_parity,
        "phase_selection": phase_rows,
        "phase_error": phase_error,
    }
    np.savez_compressed(
        output / "phase_source_traces.npz",
        time_ms=full["time_ms"],
        state=full["state"].astype(np.float32),
        phase_states=phase_states.astype(np.float32),
        **{key: value for key, value in standard.items() if key in {"rE_khz", "rI_khz", "rE_fast_khz", "mu_G", "S_G"}},
    )
    _atomic_json(output / "phase_source.json", source_summary)
    return phase_states, source_summary, standard


def _finalize_rows(
    metadata: list[dict],
    screen_rows: list[dict],
    screen_exact: dict,
    confirm_indices: list[int],
    confirm_rows: list[dict],
    confirm_exact: dict,
    dt_indices: list[int],
    dt_rows: list[dict],
    dt_exact: dict,
) -> list[dict]:
    screen_exact_lookup = _exact_by_local_fork(screen_exact)
    confirm_lookup = {original: confirm_rows[local] for local, original in enumerate(confirm_indices)}
    confirm_exact_lookup = {
        original: _exact_by_local_fork(confirm_exact).get(local, {"pass": False})
        for local, original in enumerate(confirm_indices)
    }
    dt_lookup = {original: dt_rows[local] for local, original in enumerate(dt_indices)}
    dt_exact_lookup = {
        original: _exact_by_local_fork(dt_exact).get(local, {"pass": False})
        for local, original in enumerate(dt_indices)
    }
    final: list[dict] = []
    for index, (meta, screen) in enumerate(zip(metadata, screen_rows)):
        screen_status = audited_single_resolution_status(
            screen, exact_error_pass=bool(screen_exact_lookup.get(index, {}).get("pass", False))
        )
        if screen_status == "candidate_survives" and screen["classification"] != "bounded_oscillatory_candidate":
            screen_status = "numerical_unresolved"
        confirm = confirm_lookup.get(index)
        confirm_status = None
        if screen_status == "candidate_survives":
            if confirm is None:
                confirm_status = "numerical_unresolved"
            else:
                confirm_status = audited_single_resolution_status(
                    confirm, exact_error_pass=bool(confirm_exact_lookup.get(index, {}).get("pass", False))
                )
                if confirm_status == "candidate_survives" and confirm["classification"] != "bounded_oscillatory_candidate":
                    confirm_status = "numerical_unresolved"
        status = screen_status if screen_status != "candidate_survives" else str(confirm_status)
        refined = dt_lookup.get(index)
        if status == "candidate_survives":
            if refined is None or confirm is None:
                status = "numerical_unresolved"
            else:
                status = temporal_amplitude_status(
                    confirm,
                    refined,
                    exact_error_pass=bool(dt_exact_lookup.get(index, {}).get("pass", False)),
                )
                if status == "candidate_survives" and refined["classification"] != "bounded_oscillatory_candidate":
                    status = "numerical_unresolved"
        row = {
            "fork_index": index,
            **meta,
            "screen_status": screen_status,
            "screen_classification": screen["classification"],
            "screen_tail_mean_hz": screen.get("tail_mean_hz"),
            "screen_frequency_hz": screen.get("dominant_frequency_hz"),
            "screen_direct_exact_pass": bool(screen_exact_lookup.get(index, {}).get("pass", False)),
            "confirm_status": confirm_status,
            "confirm_classification": None if confirm is None else confirm["classification"],
            "confirm_tail_mean_hz": None if confirm is None else confirm.get("tail_mean_hz"),
            "confirm_frequency_hz": None if confirm is None else confirm.get("dominant_frequency_hz"),
            "confirm_amplitude_hz": None
            if confirm is None
            else float(confirm["tail_peak_hz"]) - float(confirm["tail_trough_hz"]),
            "confirm_direct_exact_pass": None
            if confirm is None
            else bool(confirm_exact_lookup.get(index, {}).get("pass", False)),
            "dt_half_classification": None if refined is None else refined["classification"],
            "dt_half_tail_mean_hz": None if refined is None else refined.get("tail_mean_hz"),
            "dt_half_frequency_hz": None if refined is None else refined.get("dominant_frequency_hz"),
            "dt_half_amplitude_hz": None
            if refined is None
            else float(refined["tail_peak_hz"]) - float(refined["tail_trough_hz"]),
            "dt_half_direct_exact_pass": None
            if refined is None
            else bool(dt_exact_lookup.get(index, {}).get("pass", False)),
            "final_status": status,
        }
        if confirm is not None and refined is not None:
            row.update(
                {
                    "dt_half_rate_abs_difference_hz": abs(
                        float(confirm["tail_mean_hz"]) - float(refined["tail_mean_hz"])
                    ),
                    "dt_half_frequency_abs_difference_hz": abs(
                        float(confirm["dominant_frequency_hz"]) - float(refined["dominant_frequency_hz"])
                    ),
                    "dt_half_amplitude_abs_difference_hz": abs(
                        (float(confirm["tail_peak_hz"]) - float(confirm["tail_trough_hz"]))
                        - (float(refined["tail_peak_hz"]) - float(refined["tail_trough_hz"]))
                    ),
                }
            )
        final.append(row)
    return final


def _plot(
    output: Path,
    source_standard: dict[str, np.ndarray],
    source_summary: dict,
    point_rows: list[dict],
    metadata: list[dict],
    dt_indices: list[int],
    dt_sim: dict[str, np.ndarray],
    verdict: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    time_s = source_standard["time_ms"] / 1000.0
    ax.plot(time_s, 1000.0 * source_standard["rE_khz"][:, 0], color="#3b528b", lw=1.0)
    for row in source_summary["phase_selection"]:
        ax.axvline(float(row["time_ms"]) / 1000.0, color="#e76f51", lw=0.8, alpha=0.8)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="A  Locked phase source")

    ax = axes[0, 1]
    if dt_indices:
        lookup = {original: local for local, original in enumerate(dt_indices)}
        colors = {"fast": "#21918c", "pool": "#440154", "anchor": "0.55"}
        shown: set[str] = set()
        for original, meta in enumerate(metadata):
            if original not in lookup or not (np.isclose(meta["z"], 0.85) and np.isclose(meta["alpha_G"], 16.0)):
                continue
            family = str(meta["perturbation_family"])
            label = family if family not in shown else None
            shown.add(family)
            ax.plot(
                dt_sim["time_ms"] / 1000.0,
                1000.0 * dt_sim["rE_khz"][:, lookup[original]],
                color=colors[family],
                lw=0.65,
                alpha=0.60,
                label=label,
            )
        if shown:
            ax.legend(frameon=False, fontsize=8)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="B  Centre dt/2 candidate histories")

    ax = axes[1, 0]
    grid = np.full((3, 3), np.nan)
    for row in point_rows:
        zi = (0.84, 0.85, 0.86).index(round(float(row["z"]), 2))
        ai = (15.0, 16.0, 17.0).index(float(row["alpha_G"]))
        grid[ai, zi] = float(row["n_off_orbit_survivors"])
    image = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=16, aspect="auto")
    for ai in range(3):
        for zi in range(3):
            ax.text(zi, ai, f"{int(grid[ai, zi])}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(3), ("0.84", "0.85", "0.86"))
    ax.set_yticks(range(3), ("15", "16", "17"))
    ax.set(xlabel="z", ylabel=r"$\alpha_G$", title="C  Off-orbit final survivors")
    fig.colorbar(image, ax=ax, shrink=0.78)

    ax = axes[1, 1]
    for row in point_rows:
        if row["mean_frequency_hz"] is None:
            continue
        marker = "*" if row["is_centre"] else ("o" if row["is_manhattan_neighbour"] else "s")
        color = "#2a9d8f" if row["open_local_basin_support"] else "#b7b7b7"
        ax.scatter(row["mean_frequency_hz"], row["mean_amplitude_hz"], marker=marker, s=70, color=color)
        ax.annotate(f"{row['z']:.2f},{row['alpha_G']:.0f}", (row["mean_frequency_hz"], row["mean_amplitude_hz"]), fontsize=7)
    ax.set(xlabel="frequency (Hz)", ylabel="peak-to-trough amplitude (Hz)", title="D  Same-object replication")
    fig.suptitle(verdict.replace("_", " "), fontsize=11)
    fig.savefig(figures / "stage0d_local_basin_replication.png", dpi=220)
    fig.savefig(figures / "stage0d_local_basin_replication.pdf")
    plt.close(fig)
    _atomic_text(
        figures / "README.md",
        "### stage0d_local_basin_replication.png\n\n"
        "这张图显示锁定 phase source、中心点通过数值细化的候选轨迹、3×3 邻域中 off-orbit survivor 数，以及各点频率与振幅是否收敛。"
        "灰色点不满足 open-basin 门，绿色点满足；中心和 Manhattan 邻点分别用星号和圆点表示。\n\n"
        "**关注点**：只有中心有跨 fast/pool、跨相位的 off-orbit 收敛，并且至少一个 Manhattan 邻点复制同一对象，才算 Stage0D 通过。\n",
    )


def _write_phase_failure(output: Path, cfg: dict, source_summary: dict, provenance: dict, start: float) -> dict:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    summary = {
        "schema_version": "topic4_stage0d_local_basin.v1",
        "verdict": "STAGE0D_PHASE_SOURCE_INVALID",
        "phase_source": source_summary,
        "n_screen_histories": 0,
        "stage1_opened": False,
        "provenance": provenance,
        "resource_usage": {
            "wall_seconds": time.perf_counter() - start,
            "max_rss_gib": rss,
            "within_memory_contract": rss < 4.0,
        },
        "scientific_boundary_cn": "phase source 未通过；未执行邻域复核，Stage1/space 保持关闭。",
    }
    _atomic_json(output / "stage0d_local_basin_summary.json", summary)
    _atomic_text(output / "STATUS.md", "# Stage 0D 状态\n\n- 结论：`STAGE0D_PHASE_SOURCE_INVALID`\n- Stage1/space：关闭\n")
    return summary


def run(config_path: Path) -> tuple[dict, Path]:
    start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    input_provenance = _verify_locked_inputs(cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    transfer = _load_transfer(ROOT / cfg["v1_1_root"] / "extended_transfer_extra_fine.npz")
    thresholds = _classifier(cfg)
    phase_states, phase_summary, source_standard = _source_phase(cfg, transfer, thresholds, output)
    base_provenance = {
        "locked_inputs": input_provenance,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "spec": str(SPEC.resolve()),
        "spec_sha256": _sha256(SPEC),
        "module": str(MODULE.resolve()),
        "module_sha256": _sha256(MODULE),
        "runner": str(RUNNER.resolve()),
        "runner_sha256": _sha256(RUNNER),
    }
    if not phase_summary["valid"]:
        return _write_phase_failure(output, cfg, phase_summary, base_provenance, start), output

    metadata, states, params = build_local_battery(
        phase_states, LOCKED_POINTS, perturbation_fraction=float(cfg["battery"]["perturbation_fraction"])
    )
    if len(metadata) != 180:
        raise RuntimeError("Stage0D battery must contain exactly 180 histories")

    screen_sim, screen_rows, screen_exact = _run_resolution(
        metadata, states, params, transfer, cfg["screen"], thresholds, "screen"
    )
    _save_simulation(output / "screen_traces.npz", screen_sim)
    screen_exact_lookup = _exact_by_local_fork(screen_exact)
    screen_survivors = []
    for index, row in enumerate(screen_rows):
        status = audited_single_resolution_status(
            row, exact_error_pass=bool(screen_exact_lookup.get(index, {}).get("pass", False))
        )
        if status == "candidate_survives" and row["classification"] == "bounded_oscillatory_candidate":
            screen_survivors.append(index)

    confirm_sim: dict[str, np.ndarray] = {"time_ms": np.asarray([]), "rE_khz": np.empty((0, 0))}
    confirm_rows: list[dict] = []
    confirm_exact: dict = {"pass": None, "per_fork": []}
    if screen_survivors:
        confirm_sim, confirm_rows, confirm_exact = _run_resolution(
            [metadata[index] for index in screen_survivors],
            states[screen_survivors],
            [params[index] for index in screen_survivors],
            transfer,
            cfg["confirm"],
            thresholds,
            "confirm",
        )
        _save_simulation(output / "confirm_traces.npz", confirm_sim)
    confirm_exact_lookup = _exact_by_local_fork(confirm_exact)
    confirm_survivor_local = []
    for local, row in enumerate(confirm_rows):
        status = audited_single_resolution_status(
            row, exact_error_pass=bool(confirm_exact_lookup.get(local, {}).get("pass", False))
        )
        if status == "candidate_survives" and row["classification"] == "bounded_oscillatory_candidate":
            confirm_survivor_local.append(local)
    dt_indices = [screen_survivors[local] for local in confirm_survivor_local]

    dt_sim: dict[str, np.ndarray] = {"time_ms": np.asarray([]), "rE_khz": np.empty((0, 0))}
    dt_rows: list[dict] = []
    dt_exact: dict = {"pass": None, "per_fork": []}
    if dt_indices:
        dt_sim, dt_rows, dt_exact = _run_resolution(
            [metadata[index] for index in dt_indices],
            states[dt_indices],
            [params[index] for index in dt_indices],
            transfer,
            cfg["dt_half"],
            thresholds,
            "dt_half",
        )
        _save_simulation(output / "dt_half_traces.npz", dt_sim)

    final_rows = _finalize_rows(
        metadata,
        screen_rows,
        screen_exact,
        screen_survivors,
        confirm_rows,
        confirm_exact,
        dt_indices,
        dt_rows,
        dt_exact,
    )
    point_rows = [summarize_parameter_point(final_rows, z, alpha) for z, alpha in LOCKED_POINTS]
    centre = next(row for row in point_rows if row["is_centre"])
    compatible_neighbours = [
        row
        for row in point_rows
        if row["is_manhattan_neighbour"]
        and row["open_local_basin_support"]
        and centre["open_local_basin_support"]
        and point_metric_compatibility(centre, row)
    ]
    counts = dict(Counter(str(row["final_status"]) for row in final_rows))
    if centre["open_local_basin_support"] and compatible_neighbours:
        verdict = "STAGE0D_REPLICATED_OPEN_BASIN_AND_LOCAL_PARAMETER_SUPPORT"
        reason_cn = "中心点存在跨 fast/pool、跨相位的 open basin，且至少一个 Manhattan 邻点复制同一对象。"
    elif centre["open_local_basin_support"]:
        verdict = "STAGE0D_CENTER_BASIN_ONLY_NO_NEIGHBOR_REPLICATION"
        reason_cn = "中心点通过 open-basin 门，但没有预定 Manhattan 邻点复制同一对象。"
    elif counts.get("numerical_unresolved", 0):
        verdict = "STAGE0D_NO_REPLICATION_WITH_UNRESOLVED_TRAJECTORIES"
        reason_cn = "中心 open basin 未复制，且仍有长瞬态或数值/分类未决轨迹。"
    else:
        verdict = "STAGE0D_CLEAN_NO_LOCAL_BASIN_REPLICATION"
        reason_cn = "中心 open basin 未复制，且全部历史已裁决为 low 或 >100 Hz。"

    elapsed = time.perf_counter() - start
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    summary = {
        "schema_version": "topic4_stage0d_local_basin.v1",
        "verdict": verdict,
        "reason_cn": reason_cn,
        "phase_source_valid": True,
        "n_parameter_points": 9,
        "n_histories_per_point": 20,
        "n_screen_histories": len(metadata),
        "n_screen_survivors": len(screen_survivors),
        "n_confirm_survivors": len(dt_indices),
        "final_status_counts": counts,
        "centre_open_local_basin_support": centre["open_local_basin_support"],
        "n_compatible_manhattan_neighbours": len(compatible_neighbours),
        "compatible_manhattan_neighbours": [
            {"z": row["z"], "alpha_G": row["alpha_G"]} for row in compatible_neighbours
        ],
        "parameter_point_outcomes": point_rows,
        "numerical_audits": {
            "phase_source_direct_exact": phase_summary["direct_exact"],
            "screen_direct_exact": screen_exact,
            "confirm_direct_exact": confirm_exact,
            "dt_half_direct_exact": dt_exact,
        },
        "integration_contract": {key: cfg[key] for key in ("phase_source", "screen", "confirm", "dt_half")},
        "battery_contract": cfg["battery"],
        "provenance": base_provenance,
        "resource_usage": {
            "wall_seconds": elapsed,
            "max_rss_gib": rss,
            "max_memory_gib_contract": 4.0,
            "within_memory_contract": rss < 4.0,
            "execution": "single_process_blas_threads_1",
        },
        "stage1_opened": False,
        "scientific_boundary_cn": (
            "Stage0D 只复核已发现 dynamic-divisor 快态的局部 basin 与参数邻域；"
            "不含 slow lifecycle 或空间耦合，Stage1/space 在本轮保持关闭。"
        ),
    }
    if rss >= 4.0:
        summary["verdict"] = "STAGE0D_ENGINEERING_OR_PROVENANCE_FAIL"
        summary["reason_cn"] = "峰值内存超过 4 GiB 合同。"
        summary["centre_open_local_basin_support"] = False

    _atomic_json(output / "fork_outcomes.json", final_rows)
    _write_csv(output / "fork_outcomes.csv", final_rows)
    _atomic_json(output / "parameter_point_outcomes.json", point_rows)
    _write_csv(output / "parameter_point_outcomes.csv", point_rows)
    _atomic_json(output / "stage0d_local_basin_summary.json", summary)
    _plot(output, source_standard, phase_summary, point_rows, metadata, dt_indices, dt_sim, summary["verdict"])
    _atomic_text(
        output / "STATUS.md",
        "# Stage 0D local-basin replication 状态\n\n"
        f"- 结论：`{summary['verdict']}`\n"
        f"- final counts：`{counts}`\n"
        f"- screen / confirm survivors：{len(screen_survivors)} / {len(dt_indices)}\n"
        f"- centre open basin：{centre['open_local_basin_support']}\n"
        f"- compatible Manhattan neighbours：{len(compatible_neighbours)}\n"
        f"- wall / peak RSS：{elapsed:.2f} s / {rss:.3f} GiB\n"
        f"- 解释：{summary['reason_cn']}\n\n"
        "本轮没有 slow/spatial 方程或 Stage1 仿真；Stage1/space 保持关闭。\n",
    )
    return summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked Stage0D replication")
    summary, output = run(args.config)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": summary["verdict"],
                "final_status_counts": summary.get("final_status_counts", {}),
                "wall_seconds": summary["resource_usage"]["wall_seconds"],
                "max_rss_gib": summary["resource_usage"]["max_rss_gib"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
