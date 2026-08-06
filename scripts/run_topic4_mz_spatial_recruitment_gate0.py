#!/usr/bin/env python3
"""Audit whether spatial recruitment is the missing state in Stage-B timing.

This is an artifact replay.  It does not rerun the SNN or integrate the new
P-patch model.  The saved global mean and local peak of the exact SNN
recruitment sensor are factorised, then checked against independent movie and
axial spatial summaries.
"""

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
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
SNN_ENGINE = ROOT / "src" / "snn_engine"
if str(SNN_ENGINE) not in os.sys.path:
    os.sys.path.insert(0, str(SNN_ENGINE))

from mz_divisive_pool import slow_gate_drive  # noqa: E402
from slow_field import psi_recruit  # noqa: E402
from src.topic4_mz_persistence_feasibility import (  # noqa: E402
    causal_sustained_onset_ms,
    integrate_lowpass,
)
from src.topic4_mz_spatial_recruitment_gate0 import (  # noqa: E402
    causal_frame_end_times,
    effective_extent,
    first_crossing_ms,
    frame_average_trace,
    participation_ratio,
)


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_recruitment_gate0.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    keys = ("capture_path", "capture_json", "persistence_config_path", "persistence_summary_path")
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _primary_threshold(upstream_cfg: dict, tau_p_ms: float) -> float:
    for key, value in upstream_cfg["persistence"]["threshold_by_tau"].items():
        if np.isclose(float(key), tau_p_ms):
            return float(value)
    raise KeyError(f"missing upstream threshold for tau={tau_p_ms}")


def _gate_drive(values: np.ndarray, cell_cfg: dict) -> np.ndarray:
    return np.asarray([
        slow_gate_drive(
            value,
            A0=float(cell_cfg["AG0_TG"]),
            A50=float(cell_cfg["AG50_TG"]),
            exponent=float(cell_cfg["n_TG"]),
        )
        for value in values
    ])


def _blocked_spearman(
    time_ms: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    onset_ms: float,
    block_ms: float,
    minimum_frames: int,
) -> list[float]:
    post = time_ms >= onset_ms
    group = np.floor((time_ms - onset_ms) / block_ms).astype(int)
    correlations: list[float] = []
    for block in np.unique(group[post]):
        selected = post & (group == block)
        if np.count_nonzero(selected) < minimum_frames:
            continue
        if np.ptp(left[selected]) == 0.0 or np.ptp(right[selected]) == 0.0:
            continue
        coefficient = float(spearmanr(left[selected], right[selected]).statistic)
        if np.isfinite(coefficient):
            correlations.append(coefficient)
    return correlations


def _spatial_validation(
    capture: dict[str, np.ndarray],
    area_sensor: np.ndarray,
    peak_sensor: np.ndarray,
    *,
    dt_ms: float,
    onset_ms: float,
    cfg: dict,
) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    movie_pr = participation_ratio(
        capture["movie_active_fraction"], valid_mask=capture["movie_occupancy"] > 0
    )
    axial_pr = participation_ratio(
        capture["axial_active_fraction"], valid_mask=capture["axial_occupancy"] > 0
    )
    modalities = {
        "movie_24x24": (
            capture["movie_times_ms"], float(cfg["spatial_validation"]["movie_frame_ms"]), movie_pr
        ),
        "axial_48bin": (
            capture["axial_times_ms"], float(cfg["spatial_validation"]["axial_frame_ms"]), axial_pr
        ),
    }
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    block_ms = float(cfg["spatial_validation"]["descriptive_block_ms"])
    minimum_frames = int(cfg["spatial_validation"]["minimum_frames_per_block"])
    for name, (times, duration, pr) in modalities.items():
        area_frame = frame_average_trace(
            area_sensor,
            dt_ms=dt_ms,
            frame_starts_ms=times,
            frame_duration_ms=duration,
        )
        peak_frame = frame_average_trace(
            peak_sensor,
            dt_ms=dt_ms,
            frame_starts_ms=times,
            frame_duration_ms=duration,
        )
        extent_frame = effective_extent(area_frame, peak_frame)
        # Saved timestamps are frame starts.  Spatial summaries become causal
        # only at the end of their [start, start+duration) acquisition window.
        available_times = causal_frame_end_times(times, frame_duration_ms=duration)
        post = available_times >= onset_ms
        coefficient = float(spearmanr(extent_frame[post], pr[post]).statistic)
        blocked = _blocked_spearman(
            available_times,
            extent_frame,
            pr,
            onset_ms=onset_ms,
            block_ms=block_ms,
            minimum_frames=minimum_frames,
        )
        summaries[name] = {
            "post_onset_spearman": coefficient,
            "n_post_onset_frames": int(np.count_nonzero(post)),
            "block_ms": block_ms,
            "n_eligible_blocks": len(blocked),
            "blocked_spearman_median": float(np.median(blocked)) if blocked else None,
            "blocked_spearman_q25": float(np.quantile(blocked, 0.25)) if blocked else None,
            "blocked_spearman_q75": float(np.quantile(blocked, 0.75)) if blocked else None,
            "inference_contract": "descriptive_only_temporally_dependent_frames_no_iid_p_value",
        }
        arrays[f"{name}_frame_start_ms"] = np.asarray(times, np.float32)
        arrays[f"{name}_available_time_ms"] = np.asarray(available_times, np.float32)
        arrays[f"{name}_area_sensor_frame_mean"] = np.asarray(area_frame, np.float32)
        arrays[f"{name}_peak_sensor_frame_mean"] = np.asarray(peak_frame, np.float32)
        arrays[f"{name}_extent_frame_mean"] = np.asarray(extent_frame, np.float32)
        arrays[f"{name}_participation_ratio"] = np.asarray(pr, np.float32)
        for start, available, rho, participation in zip(times, available_times, extent_frame, pr):
            rows.append({
                "modality": name,
                "frame_start_ms": float(start),
                "causal_available_time_ms": float(available),
                "time_from_onset_ms": float(available - onset_ms),
                "effective_extent_frame_mean": float(rho),
                "participation_ratio": float(participation),
                "post_onset": bool(available >= onset_ms),
            })
    return summaries, rows, arrays


def _longest_true_run(values: np.ndarray) -> int:
    sequence = np.asarray(values, dtype=bool)
    if sequence.ndim != 1:
        raise ValueError("run detector requires a 1D boolean array")
    edges = np.diff(np.r_[False, sequence, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return int(np.max(stops - starts)) if starts.size else 0


def _operational_sentinel(
    capture: dict[str, np.ndarray],
    *,
    area_sensor: np.ndarray,
    peak_sensor: np.ndarray,
    dt_ms: float,
    onset_ms: float,
    cfg: dict,
) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    """Compare onset-seed and recruited windows at matched local intensity."""

    contract = cfg["operational_sentinel"]
    starts = capture["movie_times_ms"]
    duration = float(cfg["spatial_validation"]["movie_frame_ms"])
    available = causal_frame_end_times(starts, frame_duration_ms=duration)
    raw_frame = frame_average_trace(
        capture["slow_rEfast_max"], dt_ms=dt_ms,
        frame_starts_ms=starts, frame_duration_ms=duration,
    )
    peak_frame = frame_average_trace(
        peak_sensor, dt_ms=dt_ms,
        frame_starts_ms=starts, frame_duration_ms=duration,
    )
    area_frame = frame_average_trace(
        area_sensor, dt_ms=dt_ms,
        frame_starts_ms=starts, frame_duration_ms=duration,
    )
    extent_frame = effective_extent(area_frame, peak_frame)
    movie_pr = participation_ratio(
        capture["movie_active_fraction"], valid_mask=capture["movie_occupancy"] > 0
    )
    valid_cells = (capture["movie_occupancy"] > 0).reshape(-1)
    flat_movie = capture["movie_active_fraction"].reshape(capture["movie_active_fraction"].shape[0], -1)
    coverage = np.mean(
        flat_movie[:, valid_cells] >= float(contract["coverage_sensitivity_threshold"]), axis=1
    )
    floor = float(contract["amplitude_floor_psi"])
    seed_bounds = [onset_ms + float(value) for value in contract["seed_window_from_onset_ms"]]
    late_bounds = [onset_ms + float(value) for value in contract["recruited_window_from_onset_ms"]]
    seed = (available >= seed_bounds[0]) & (available < seed_bounds[1]) & (peak_frame >= floor)
    late = (available >= late_bounds[0]) & (available < late_bounds[1]) & (peak_frame >= floor)
    if np.count_nonzero(seed) < 2 or np.count_nonzero(late) < 2:
        raise RuntimeError("operational sentinel windows contain too few amplitude-qualified frames")

    def _median(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.median(values[mask]))

    raw_seed, raw_late = _median(raw_frame, seed), _median(raw_frame, late)
    extent_seed, extent_late = _median(extent_frame, seed), _median(extent_frame, late)
    pr_seed, pr_late = _median(movie_pr, seed), _median(movie_pr, late)
    coverage_seed, coverage_late = _median(coverage, seed), _median(coverage, late)
    raw_relative_change = (raw_late - raw_seed) / raw_seed
    extent_absolute_change = extent_late - extent_seed
    extent_relative_change = extent_absolute_change / extent_seed
    pr_absolute_change = pr_late - pr_seed
    pr_relative_change = pr_absolute_change / pr_seed

    cost = np.abs(peak_frame[seed, None] - peak_frame[late][None, :])
    seed_assignment, late_assignment = linear_sum_assignment(cost)
    matched = cost[seed_assignment, late_assignment] <= float(contract["intensity_match_caliper_psi"])
    seed_indices = np.flatnonzero(seed)[seed_assignment[matched]]
    late_indices = np.flatnonzero(late)[late_assignment[matched]]
    matched_extent_change = float(np.median(extent_frame[late_indices] - extent_frame[seed_indices]))
    matched_pr_change = float(np.median(movie_pr[late_indices] - movie_pr[seed_indices]))

    block_ms = float(contract["persistence_block_ms"])
    n_blocks_float = (late_bounds[1] - late_bounds[0]) / block_ms
    n_blocks = int(round(n_blocks_float))
    if not np.isclose(n_blocks_float, n_blocks):
        raise ValueError("recruited sentinel window must contain whole persistence blocks")
    block_rows: list[dict] = []
    directional: list[bool] = []
    for block in range(n_blocks):
        left = late_bounds[0] + block * block_ms
        right = left + block_ms
        selected = (available >= left) & (available < right) & (peak_frame >= floor)
        if not np.any(selected):
            raise RuntimeError(f"sentinel block {block} contains no amplitude-qualified frames")
        extent_change = _median(extent_frame, selected) - extent_seed
        pr_change = _median(movie_pr, selected) - pr_seed
        agrees = extent_change > 0.0 and pr_change > 0.0
        directional.append(agrees)
        block_rows.append({
            "block": block,
            "start_from_onset_ms": float(left - onset_ms),
            "stop_from_onset_ms": float(right - onset_ms),
            "n_frames": int(np.count_nonzero(selected)),
            "extent_change_from_seed": extent_change,
            "movie_pr_change_from_seed": pr_change,
            "directional_increase_both": agrees,
        })
    longest_directional_run = _longest_true_run(np.asarray(directional))
    components = {
        "local_raw_intensity_stable": (
            abs(raw_relative_change) <= float(contract["maximum_local_raw_relative_change"])
        ),
        "extent_window_shift": (
            extent_absolute_change >= float(contract["minimum_extent_absolute_increase"])
            and extent_relative_change >= float(contract["minimum_extent_relative_increase"])
        ),
        "movie_pr_window_shift": (
            pr_absolute_change >= float(contract["minimum_movie_pr_absolute_increase"])
            and pr_relative_change >= float(contract["minimum_movie_pr_relative_increase"])
        ),
        "persistent_direction_across_blocks": (
            longest_directional_run >= int(contract["minimum_consecutive_directional_blocks"])
        ),
        "enough_intensity_matched_pairs": (
            seed_indices.size >= int(contract["minimum_intensity_matched_pairs"])
        ),
        "matched_extent_shift": (
            matched_extent_change >= float(contract["minimum_matched_extent_increase"])
        ),
        "matched_movie_pr_shift": (
            matched_pr_change >= float(contract["minimum_matched_movie_pr_increase"])
        ),
    }
    summary = {
        "status": "conditional_single_seed_operational_support" if all(components.values()) else "single_seed_operational_support_failed",
        "formal_multiseed_gate_can_pass": False,
        "formal_blocker": "only seed 1 stores the required spatial sensor history",
        "causal_frame_timing": "movie frame [start,start+25ms) is available only at frame end",
        "n_seed_frames": int(np.count_nonzero(seed)),
        "n_recruited_frames": int(np.count_nonzero(late)),
        "local_raw_intensity": {
            "seed_median": raw_seed,
            "recruited_median": raw_late,
            "relative_change": raw_relative_change,
        },
        "effective_extent": {
            "seed_median": extent_seed,
            "recruited_median": extent_late,
            "absolute_change": extent_absolute_change,
            "relative_change": extent_relative_change,
        },
        "movie_area_participation_ratio": {
            "seed_median": pr_seed,
            "recruited_median": pr_late,
            "absolute_change": pr_absolute_change,
            "relative_change": pr_relative_change,
        },
        "movie_coverage_sensitivity": {
            "threshold": float(contract["coverage_sensitivity_threshold"]),
            "seed_median": coverage_seed,
            "recruited_median": coverage_late,
            "absolute_change": coverage_late - coverage_seed,
        },
        "intensity_matched": {
            "caliper_psi": float(contract["intensity_match_caliper_psi"]),
            "n_pairs": int(seed_indices.size),
            "median_extent_change": matched_extent_change,
            "median_movie_pr_change": matched_pr_change,
        },
        "persistence_blocks": {
            "block_ms": block_ms,
            "longest_consecutive_directional_run": longest_directional_run,
            "rows": block_rows,
        },
        "decision_components": components,
    }
    arrays = {
        "movie_available_time_ms": np.asarray(available, np.float32),
        "movie_raw_peak_frame_mean": np.asarray(raw_frame, np.float32),
        "movie_psi_peak_frame_mean": np.asarray(peak_frame, np.float32),
        "movie_area_sensor_frame_mean": np.asarray(area_frame, np.float32),
        "movie_effective_extent": np.asarray(extent_frame, np.float32),
        "movie_participation_ratio": np.asarray(movie_pr, np.float32),
        "movie_coverage_sensitivity": np.asarray(coverage, np.float32),
        "movie_seed_window_mask": np.asarray(seed, bool),
        "movie_recruited_window_mask": np.asarray(late, bool),
        "matched_seed_frame_indices": np.asarray(seed_indices, np.int32),
        "matched_recruited_frame_indices": np.asarray(late_indices, np.int32),
    }
    return summary, block_rows, arrays


def _plot(
    figures: Path,
    *,
    time_ms: np.ndarray,
    causal_envelope: np.ndarray,
    onset_ms: float,
    peak_sensor: np.ndarray,
    area_sensor: np.ndarray,
    extent: np.ndarray,
    local_p: np.ndarray,
    area_p: np.ndarray,
    threshold: float,
    spatial_arrays: dict[str, np.ndarray],
    spatial_summary: dict,
    stride: int,
) -> Path:
    colors = {
        "rate": "#252525",
        "peak": "#762a83",
        "area": "#2166ac",
        "extent": "#1b7837",
        "local_p": "#d6604d",
        "movie": "#1b9e77",
        "axial": "#7570b3",
    }
    relative_s = (time_ms - onset_ms) / 1000.0
    keep = slice(None, None, max(1, stride))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    ax_a.plot(relative_s[keep], causal_envelope[keep], color=colors["rate"], lw=1.0)
    ax_a.axhline(20.0, color="#b2182b", ls="--", lw=1.0, label="20 Hz onset gate")
    ax_a.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax_a.set(xlabel="time from causal onset (s)", ylabel="trailing 250-ms E rate (Hz)",
             title="A  Returning IEDs and sustained recruitment")
    ax_a.legend(frameon=False, fontsize=8)

    ax_b.plot(relative_s[keep], peak_sensor[keep], color=colors["peak"], lw=0.9,
              label=r"local peak $\max_x\Psi$")
    ax_b.plot(relative_s[keep], area_sensor[keep], color=colors["area"], lw=1.0,
              label=r"area mean $A_G=\langle\Psi\rangle$")
    ax_b.plot(relative_s[keep], extent[keep], color=colors["extent"], lw=1.0,
              label=r"effective extent $\rho_{eff}=A_G/\max\Psi$")
    ax_b.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax_b.axvspan(0.0, 0.25, color="0.75", alpha=0.16, lw=0)
    ax_b.axvspan(1.0, 3.0, color="#a6dba0", alpha=0.16, lw=0)
    ax_b.set(xlabel="time from causal onset (s)", ylabel="bounded sensor value",
             title="B  Exact intensity–extent factorisation", ylim=(-0.02, 1.02))
    ax_b.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper center")

    ax_c.plot(relative_s[keep], local_p[keep], color=colors["local_p"], lw=1.0,
              label="local-intensity-only p")
    ax_c.plot(relative_s[keep], area_p[keep], color=colors["area"], lw=1.4,
              label="area-weighted p (saved SNN gate)")
    ax_c.axhline(threshold, color="#b2182b", ls="--", lw=1.0, label="locked p threshold")
    ax_c.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax_c.set(xlabel="time from causal onset (s)", ylabel="persistence p",
             title="C  Space, not local intensity, delays the gate", ylim=(-0.02, 1.02))
    ax_c.legend(frameon=False, fontsize=7.6)

    for name, label, color in (
        ("movie_24x24", "24×24 movie PR", colors["movie"]),
        ("axial_48bin", "48-bin axial PR", colors["axial"]),
    ):
        times = spatial_arrays[f"{name}_available_time_ms"]
        x = (times - onset_ms) / 1000.0
        rho = spatial_arrays[f"{name}_extent_frame_mean"]
        pr = spatial_arrays[f"{name}_participation_ratio"]
        post_view = x >= -2.0
        ax_d.plot(x[post_view], pr[post_view], color=color, lw=0.8, alpha=0.75, label=label)
        if name == "axial_48bin":
            ax_d.plot(x[post_view], rho[post_view], color=colors["extent"], lw=1.2,
                      label=r"frame-mean $\rho_{eff}$")
    ax_d.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax_d.axvspan(0.0, 0.25, color="0.75", alpha=0.16, lw=0)
    ax_d.axvspan(1.0, 3.0, color="#a6dba0", alpha=0.16, lw=0)
    correlation_text = ", ".join(
        f"{name.split('_')[0]} r={values['post_onset_spearman']:.2f}"
        for name, values in spatial_summary.items()
    )
    ax_d.text(0.02, 0.97, correlation_text, transform=ax_d.transAxes, va="top", fontsize=8)
    ax_d.set(xlabel="time from causal onset (s)", ylabel="effective spatial support",
             title="D  2D movie validation; axial span sensitivity", ylim=(-0.02, 1.02))
    ax_d.legend(frameon=False, fontsize=7.5, ncol=2)

    fig.suptitle("Gate 0 sentinel: spatial recruitment is a missing state, not a new E–E mechanism",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.012,
        "Seed 1 operational decomposition only; the formal multiseed gate remains open. Gray: onset seed window; green: recruited window.",
        ha="center", fontsize=8.0, color="#7f0000",
    )
    stem = figures / "mz_spatial_recruitment_gate0"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed_hashes = _validate_inputs(cfg)
    capture_meta = json.loads((ROOT / cfg["capture_json"]).read_text(encoding="utf-8"))
    upstream_cfg = yaml.safe_load((ROOT / cfg["persistence_config_path"]).read_text(encoding="utf-8"))
    upstream_summary = json.loads(
        (ROOT / cfg["persistence_summary_path"]).read_text(encoding="utf-8")
    )
    expected = cfg["expected_capture_contract"]
    if capture_meta.get("schema_version") != str(expected["schema_version"]):
        raise RuntimeError("capture schema drifted")
    simulation_meta = capture_meta["simulation"]
    if int(simulation_meta["seed"]) != int(expected["seed"]):
        raise RuntimeError("capture seed drifted")
    if not np.isclose(float(simulation_meta["dt_ms"]), float(expected["dt_ms"])):
        raise RuntimeError("capture dt drifted")
    if not np.isclose(float(simulation_meta["T_ms"]), float(expected["duration_ms"])):
        raise RuntimeError("capture duration drifted")
    required_keys = (
        "times_ms", "rate_E_hz", "slow_AG", "slow_UTG", "slow_TG", "slow_rEfast_max",
        "movie_active_fraction", "movie_times_ms", "movie_occupancy",
        "axial_active_fraction", "axial_times_ms", "axial_occupancy",
    )
    with np.load(ROOT / cfg["capture_path"], allow_pickle=False) as payload:
        missing = sorted(set(required_keys) - set(payload.files))
        if missing:
            raise KeyError(f"capture is missing Gate-0 arrays: {missing}")
        capture = {key: np.asarray(payload[key], dtype=float) for key in required_keys}

    dt_ms = float(capture_meta["simulation"]["dt_ms"])
    time_ms = np.arange(capture["rate_E_hz"].size, dtype=float) * dt_ms
    # The saved clock is float32 and loses sub-microsecond precision near 20 s.
    # Integration dt remains the JSON value; the clock is only checked within
    # one float32 ULP at the locked duration and is never differentiated.
    saved_clock_atol = float(np.spacing(np.float32(expected["duration_ms"])))
    if not np.allclose(capture["times_ms"], time_ms, atol=saved_clock_atol, rtol=0.0):
        raise RuntimeError("saved times do not match metadata dt")
    onset_cfg = cfg["causal_onset"]
    onset_ms, causal_envelope = causal_sustained_onset_ms(
        capture["rate_E_hz"],
        dt_ms=dt_ms,
        envelope_ms=float(onset_cfg["envelope_ms"]),
        threshold_hz=float(onset_cfg["threshold_hz"]),
        minimum_duration_ms=float(onset_cfg["minimum_duration_ms"]),
    )
    if not np.isclose(onset_ms, float(onset_cfg["expected_onset_ms"]), atol=dt_ms):
        raise RuntimeError(f"causal onset drifted: {onset_ms}")
    if not np.isclose(onset_ms, float(upstream_summary["causal_onset_ms"]), atol=dt_ms):
        raise RuntimeError("Gate-0 onset no longer matches Stage B")

    cell_cfg = capture_meta["simulation"]["cell_config"]
    if not np.isclose(float(cell_cfg["p_pool"]), float(expected["p_pool"])):
        raise RuntimeError("AG is not an arithmetic spatial mean when p_pool != 1")
    peak_sensor = psi_recruit(
        capture["slow_rEfast_max"],
        float(cell_cfg["r0_psi"]),
        float(cell_cfg["r50_psi"]),
        float(cell_cfg["n_psi"]),
    )
    extent = effective_extent(
        capture["slow_AG"],
        peak_sensor,
        zero_tolerance=float(cfg["extent"]["zero_tolerance"]),
        bound_tolerance=float(cfg["extent"]["bound_tolerance"]),
    )
    factorisation_error = float(np.max(np.abs(capture["slow_AG"] - peak_sensor * extent)))
    area_drive = _gate_drive(capture["slow_AG"], cell_cfg)
    local_drive = _gate_drive(peak_sensor, cell_cfg)
    drive_error = float(np.max(np.abs(area_drive - capture["slow_UTG"])))

    tau_p_ms = float(cfg["persistence"]["tau_p_ms"])
    threshold = float(cfg["persistence"]["threshold"])
    upstream_threshold = _primary_threshold(upstream_cfg, tau_p_ms)
    if not np.isclose(threshold, upstream_threshold, atol=1.0e-12):
        raise RuntimeError("Gate-0 persistence threshold no longer matches Stage B")
    area_p = integrate_lowpass(area_drive, dt_ms=dt_ms, tau_ms=tau_p_ms)
    local_p = integrate_lowpass(local_drive, dt_ms=dt_ms, tau_ms=tau_p_ms)
    persistence_error = float(np.max(np.abs(area_p - capture["slow_TG"])))
    area_crossing_ms = first_crossing_ms(area_p, threshold=threshold, dt_ms=dt_ms)
    local_crossing_ms = first_crossing_ms(local_p, threshold=threshold, dt_ms=dt_ms)

    onset_index = int(round(onset_ms / dt_ms))
    established_start_ms = onset_ms + float(cfg["persistence"]["established_delay_ms"])
    established = time_ms >= established_start_ms
    pre = time_ms < onset_ms
    pre_extent_max = float(np.max(extent[pre]))
    established_extent_q = float(np.quantile(
        extent[established], float(cfg["persistence"]["established_quantile"])
    ))
    instantaneous_separation = pre_extent_max < established_extent_q

    spatial_summary, spatial_rows, spatial_arrays = _spatial_validation(
        capture, capture["slow_AG"], peak_sensor,
        dt_ms=dt_ms, onset_ms=onset_ms, cfg=cfg
    )
    sentinel_summary, sentinel_block_rows, sentinel_arrays = _operational_sentinel(
        capture,
        area_sensor=capture["slow_AG"],
        peak_sensor=peak_sensor,
        dt_ms=dt_ms,
        onset_ms=onset_ms,
        cfg=cfg,
    )
    decision = cfg["decision_contract"]
    reconstruction_ok = (
        factorisation_error <= float(decision["reconstruction_atol"])
        and drive_error <= float(decision["reconstruction_atol"])
        and persistence_error <= float(decision["persistence_parity_atol"])
    )
    timing_disambiguation = (
        local_crossing_ms is not None
        and area_crossing_ms is not None
        and local_crossing_ms < onset_ms < area_crossing_ms
    )
    movie_concordance = (
        spatial_summary["movie_24x24"]["post_onset_spearman"]
        >= float(decision["minimum_movie_descriptive_spearman"])
    )
    sentinel_supported = sentinel_summary["status"] == "conditional_single_seed_operational_support"
    extent_supported = reconstruction_ok and timing_disambiguation and movie_concordance and sentinel_supported
    if extent_supported and instantaneous_separation:
        status = "single_seed_operational_spatial_decomposition_supported_formal_gate_open"
        next_step = "proceed_to_P1_parity_then_P2_pilot_formal_multiseed_gate_remains_open"
    elif extent_supported:
        status = "single_seed_operational_spatial_decomposition_supported_persistence_AND_latch_required"
        next_step = "proceed_to_P1_parity_then_P2_AND_latch_without_empirical_rho_threshold_lock"
    else:
        status = "spatial_extent_missing_state_not_supported_stop_before_P_patch_dynamics"
        next_step = "stop_and_revise_spatial_sensor_hypothesis"

    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    traces = {
        "time_ms": time_ms.astype(np.float32),
        "causal_rate_envelope_hz": causal_envelope.astype(np.float32),
        "local_peak_sensor": np.asarray(peak_sensor, np.float32),
        "area_mean_sensor": np.asarray(capture["slow_AG"], np.float32),
        "effective_extent": np.asarray(extent, np.float32),
        "local_intensity_drive": np.asarray(local_drive, np.float32),
        "area_weighted_drive": np.asarray(area_drive, np.float32),
        "local_intensity_persistence": np.asarray(local_p, np.float32),
        "area_weighted_persistence": np.asarray(area_p, np.float32),
        **spatial_arrays,
        **sentinel_arrays,
    }
    np.savez_compressed(output / "spatial_recruitment_gate0_traces.npz", **traces)
    _write_csv(output / "spatial_validation_frames.csv", spatial_rows)
    _write_csv(output / "operational_sentinel_blocks.csv", sentinel_block_rows)
    metric_rows = [
        {"metric": "causal_onset_ms", "value": onset_ms},
        {"metric": "local_intensity_p_crossing_ms", "value": local_crossing_ms},
        {"metric": "area_weighted_p_crossing_ms", "value": area_crossing_ms},
        {"metric": "area_weighted_p_delay_after_onset_ms", "value": area_crossing_ms - onset_ms},
        {"metric": "pre_onset_effective_extent_max", "value": pre_extent_max},
        {"metric": "established_effective_extent_q25", "value": established_extent_q},
        {"metric": "instantaneous_extent_separation", "value": instantaneous_separation},
        {"metric": "factorisation_max_abs_error", "value": factorisation_error},
        {"metric": "UTG_reconstruction_max_abs_error", "value": drive_error},
        {"metric": "saved_TG_parity_max_abs_error", "value": persistence_error},
        {
            "metric": "sentinel_local_raw_relative_change",
            "value": sentinel_summary["local_raw_intensity"]["relative_change"],
        },
        {
            "metric": "sentinel_effective_extent_absolute_change",
            "value": sentinel_summary["effective_extent"]["absolute_change"],
        },
        {
            "metric": "sentinel_movie_pr_absolute_change",
            "value": sentinel_summary["movie_area_participation_ratio"]["absolute_change"],
        },
        {
            "metric": "sentinel_intensity_matched_extent_change",
            "value": sentinel_summary["intensity_matched"]["median_extent_change"],
        },
        {
            "metric": "sentinel_intensity_matched_movie_pr_change",
            "value": sentinel_summary["intensity_matched"]["median_movie_pr_change"],
        },
    ]
    for modality, values in spatial_summary.items():
        metric_rows.extend([
            {"metric": f"{modality}_post_onset_spearman", "value": values["post_onset_spearman"]},
            {"metric": f"{modality}_blocked_spearman_median", "value": values["blocked_spearman_median"]},
        ])
    _write_csv(output / "gate0_metrics.csv", metric_rows)
    figure = _plot(
        figures,
        time_ms=time_ms,
        causal_envelope=causal_envelope,
        onset_ms=onset_ms,
        peak_sensor=peak_sensor,
        area_sensor=capture["slow_AG"],
        extent=extent,
        local_p=local_p,
        area_p=area_p,
        threshold=threshold,
        spatial_arrays=spatial_arrays,
        spatial_summary=spatial_summary,
        stride=int(cfg["plot"]["trace_stride"]),
    )
    summary = {
        "status": status,
        "scientific_layer": "single_seed_operational_spatial_sentinel_not_formal_gate_or_lifecycle",
        "next_step": next_step,
        "causal_onset_ms": onset_ms,
        "causal_onset_confirmed_online_ms": (
            onset_ms + float(cfg["causal_onset"]["minimum_duration_ms"])
        ),
        "sensor_factorisation": {
            "identity": "AG = max(Psi(rE_fast)) * rho_eff",
            "effective_extent_definition": "rho_eff = spatial_mean(Psi)/spatial_max(Psi)",
            "factorisation_max_abs_error": factorisation_error,
            "UTG_reconstruction_max_abs_error": drive_error,
            "saved_TG_parity_max_abs_error": persistence_error,
            "p_pool": float(cell_cfg["p_pool"]),
            "interpretation": "amplitude_normalised_soft_extent_not_geometric_area_fraction",
        },
        "timing_disambiguation": {
            "tau_p_ms": tau_p_ms,
            "locked_threshold": threshold,
            "local_intensity_p_first_crossing_ms": local_crossing_ms,
            "local_intensity_p_crossing_from_onset_ms": local_crossing_ms - onset_ms,
            "area_weighted_p_first_crossing_ms": area_crossing_ms,
            "area_weighted_p_crossing_from_onset_ms": area_crossing_ms - onset_ms,
            "local_intensity_alone_false_positive_before_macro_onset": local_crossing_ms < onset_ms,
        },
        "instantaneous_extent_separation": {
            "pre_onset_max": pre_extent_max,
            "established_q25": established_extent_q,
            "established_start_ms": established_start_ms,
            "premax_below_established_q25": instantaneous_separation,
            "interpretation": (
                "single instantaneous rho threshold is not identifiable; retain local persistence AND spatial latch"
                if not instantaneous_separation else
                "single-seed pilot interval only; primary-seed replication still required"
            ),
        },
        "independent_spatial_validation": spatial_summary,
        "operational_single_seed_sentinel": sentinel_summary,
        "decision_components": {
            "exact_reconstruction_and_saved_state_parity": reconstruction_ok,
            "local_vs_area_timing_disambiguated": timing_disambiguation,
            "movie_spatial_readout_concordant": movie_concordance,
            "axial_projection_is_sensitivity_only": True,
            "single_seed_operational_sentinel_supported": sentinel_supported,
            "effective_extent_missing_state_operationally_supported_single_seed": extent_supported,
            "formal_multiseed_gate_pass": False,
            "instantaneous_rho_threshold_lock_ready": False,
        },
        "claim_boundary": [
            "rho_eff is derived from AG and the sensor peak; only movie/axial participation are independent checks",
            "movie and axial frames are temporally dependent, so correlations are descriptive and no iid p-value is used",
            "only seed 1 has the required spatial sensor history; no multiseed rho threshold is locked",
            "axial participation measures longitudinal span and is a sensitivity readout, not the primary 2D extent gate",
            "movie timestamps are frame starts; all causal comparisons use frame-end availability times",
            "the capture lacks core-specific numerator fields, so exact core occupancy is not recoverable",
            "the capture was generated under the old divisive feedback and is a recorded-history stress bracket",
            "no autonomous rho dynamics, core-surround recruitment, termination, containment, or recovery is claimed",
        ],
        "input_sha256": observed_hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "metrics_csv": str((output / "gate0_metrics.csv").relative_to(ROOT)),
            "spatial_frames_csv": str((output / "spatial_validation_frames.csv").relative_to(ROOT)),
            "sentinel_blocks_csv": str((output / "operational_sentinel_blocks.csv").relative_to(ROOT)),
            "traces_npz": str((output / "spatial_recruitment_gate0_traces.npz").relative_to(ROOT)),
        },
        "config": str(config_path.relative_to(ROOT)),
    }
    (output / "spatial_recruitment_gate0_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_recruitment_gate0.png / .pdf\n\n"
        "这张四面板图用已有 SNN capture 检验 Stage B 缺失的状态是否真的是空间招募。"
        "A 显示 returning IED 到持续高活动的因果 onset；B 把同一个 SNN recruitment sensor "
        "严格分成局部峰值、全场均值和 effective extent；C 比较只看局部峰值与保留空间均值时 p 的开门时间；"
        "D 以独立保存的 24×24 movie participation 做二维主验证，并把 48-bin axial participation 仅作为轴向 span sensitivity。\n\n"
        "**关注点**：局部峰值传感器在宏观 onset 前很久已经误开门，而 area-weighted p 到 onset 后约 2.76 s 才开门。"
        "这支持加入空间 recruitment coordinate，但 pre-onset IED 的瞬时 extent 与 established state 仍重叠，"
        "所以不能从这个单 seed capture 锁一个 memoryless rho threshold；下一步必须保留 local persistence AND spatial latch。"
        "movie/axial 时间均按帧末作为因果可用时刻，且本图只是一条 seed-1 operational sentinel，不是正式 multiseed Gate 0。\n",
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
        "next_step": summary["next_step"],
        "timing_disambiguation": summary["timing_disambiguation"],
        "instantaneous_extent_separation": summary["instantaneous_extent_separation"],
        "independent_spatial_validation": summary["independent_spatial_validation"],
        "figure": summary["artifacts"]["figure"],
    }, indent=2))


if __name__ == "__main__":
    main()
