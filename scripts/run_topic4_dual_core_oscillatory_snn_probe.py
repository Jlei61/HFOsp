#!/usr/bin/env python3
"""Exploratory full-SNN validation of a fast-inhibition oscillatory candidate.

This runner keeps the frozen E1146 dual-core graph, node field, Z/M equations
and spatial OU process.  It changes only ``tau_d_GABA`` and records the state
evidence needed to decide whether a paper-panel trajectory is oscillatory or a
tonic plateau.  The output is development evidence, not a frozen rev21 result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate  # noqa: E402
from src.topic4_dual_core_oscillation_phase import (  # noqa: E402
    classify_coarse_trajectory,
    contact_oscillation_assay,
)
from src.topic4_zm_ictal_transition import make_external_drive, make_slow  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/oscillatory_snn_probe")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _spatial_frames(spikes: np.ndarray, positions: np.ndarray, *, dt_ms: float,
                    sheet_mm: float, frame_ms: float = 20.0,
                    bin_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    frame_steps = int(round(frame_ms / dt_ms))
    grid = int(round(sheet_mm / bin_mm))
    n_frames = spikes.shape[0] // frame_steps
    ix = np.clip((positions[:, 0] / bin_mm).astype(int), 0, grid - 1)
    iy = np.clip((positions[:, 1] / bin_mm).astype(int), 0, grid - 1)
    spatial_index = iy * grid + ix
    frames = np.empty((n_frames, grid, grid), np.float32)
    for frame in range(n_frames):
        counts = np.sum(
            spikes[frame * frame_steps:(frame + 1) * frame_steps], axis=0)
        frames[frame] = np.bincount(
            spatial_index, weights=counts, minlength=grid * grid,
        ).reshape(grid, grid)
    times = (np.arange(n_frames, dtype=np.float32) + 0.5) * frame_ms
    return frames, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--candidate-id", default="rev21_si_0p7_sm_0p5")
    parser.add_argument("--topology-seed", type=int, default=2542)
    parser.add_argument("--dynamics-seed", type=int, default=2641)
    parser.add_argument("--tau-d-gaba-ms", type=float, default=8.0)
    parser.add_argument("--duration-ms", type=float, default=6500.0)
    parser.add_argument("--post-runaway-record-ms", type=float, default=1500.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()

    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    substrate, transition_path, ou_config, manifest_path = _substrate(
        config_path, artifact_root, int(args.topology_seed))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [row for row in manifest["candidates"]
               if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise RuntimeError("candidate must occur exactly once in rev21 manifest")
    candidate = matches[0]
    if candidate["slow_variables"].get("mode") == "off":
        raise RuntimeError("oscillatory transition probe requires active Z/M")
    if not 0.0 < args.tau_d_gaba_ms < 18.0:
        raise ValueError("probe expects a positive GABA decay below the 18-ms baseline")

    substrate.params.tau_d_GABA = float(args.tau_d_gaba_ms)
    substrate.params.T = float(args.duration_ms)
    substrate.net["rng"] = np.random.default_rng(int(args.dynamics_seed))
    slow = make_slow(
        substrate, candidate["slow_variables"], trace_weights_E=substrate.h_e)
    centers = np.asarray(candidate["node_field"]["centers_mm"], float)
    distance = np.linalg.norm(
        substrate.positions_e[:, None, :] - centers[None, :, :], axis=2)
    in_core = np.asarray(substrate.h_e >= 0.5, bool)
    core_a = in_core & (distance[:, 0] <= distance[:, 1])
    core_b = in_core & ~core_a
    surround = ~in_core
    masks = (core_a, core_b, surround)
    if any(np.sum(mask) == 0 for mask in masks):
        raise RuntimeError("dual-core regional mask is empty")
    slow.enable_region_traces({
        "core_a": core_a, "core_b": core_b, "surround": surround})
    drive = make_external_drive(substrate, ou_config, int(args.dynamics_seed))
    recorder = LFPRecorder(
        substrate.params, substrate.net["pos"], substrate.net["labels"],
        sites=substrate.contact_xy)
    simulation = config["search"]["simulation"]
    print(json.dumps({
        "status": "SNN_STARTED", "candidate": args.candidate_id,
        "tau_d_GABA_ms": args.tau_d_gaba_ms,
        "topology_seed": args.topology_seed,
        "dynamics_seed": args.dynamics_seed,
        "duration_ms": args.duration_ms,
    }), flush=True)
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        lfp_recorder=recorder, early_stop_runaway=True, verbose=True,
        es_thresh_hz=float(simulation["es_thresh_hz"]),
        es_dur_ms=float(simulation["es_dur_ms"]),
        post_runaway_record_ms=float(args.post_runaway_record_ms),
        external_e_rate_drive=drive)

    dt_ms = float(substrate.params.dt)
    spikes = np.asarray(result["E_spk_bool"], bool)
    rate_raw = np.asarray(result["rate_E"], float)
    rate_smooth = gaussian_filter1d(rate_raw, sigma=2.0 / dt_ms)
    tail_ms = min(1000.0, spikes.shape[0] * dt_ms)
    tail_steps = int(round(tail_ms / dt_ms))
    per_neuron_hz = (
        np.sum(spikes[-tail_steps:], axis=0) / (tail_ms / 1000.0))
    regional = [float(np.mean(per_neuron_hz[mask])) for mask in masks]
    state = classify_coarse_trajectory(
        rate_smooth, dt_ms=dt_ms, regional_tail_rate_hz=regional,
        tail_ms=tail_ms, window_ms=250.0)
    state["boundary"] = (
        "full 40,000-cell SNN state assay on a 2-ms-smoothed population-rate "
        "trace; model-state label only, not a clinical seizure definition")
    time_ms = np.asarray(result["times"], float)
    onset_value = result["runaway_early_stop_ms"]
    onset_ms = None if onset_value is None else float(onset_value)
    baseline_window = (time_ms >= 500.0) & (time_ms < 1000.0)
    if onset_ms is None:
        early_window = time_ms >= max(0.0, float(time_ms[-1]) - 1000.0)
    else:
        early_window = ((time_ms >= onset_ms + 100.0)
                        & (time_ms < min(time_ms[-1], onset_ms + 1100.0)))
    terminal_window = time_ms >= max(0.0, float(time_ms[-1]) - 1000.0)
    early_contact_state = contact_oscillation_assay(
        np.asarray(result["lfp_trace"])[baseline_window],
        np.asarray(result["lfp_trace"])[early_window],
        dt_ms=dt_ms)
    early_contact_state["window_role"] = (
        "post_operational_runaway" if onset_ms is not None
        else "last_1000_ms_without_operational_runaway")
    terminal_contact_state = contact_oscillation_assay(
        np.asarray(result["lfp_trace"])[baseline_window],
        np.asarray(result["lfp_trace"])[terminal_window],
        dt_ms=dt_ms)
    terminal_contact_state["window_role"] = "terminal_1000_ms"

    frames, frame_times = _spatial_frames(
        spikes, substrate.positions_e, dt_ms=dt_ms,
        sheet_mm=float(substrate.params.L))
    sample_index = np.arange(0, substrate.n_e, 8, dtype=np.int32)
    sample_step, sample_column = np.nonzero(spikes[:, sample_index])
    sampled_neuron = sample_index[sample_column]
    slow_arrays = slow.trace_arrays()
    weighted = slow.weighted_trace_arrays()
    regional_slow = slow.region_trace_arrays()
    stem = (
        f"{args.candidate_id}_t{args.topology_seed}_d{args.dynamics_seed}_"
        f"tauGABA{args.tau_d_gaba_ms:g}".replace(".", "p"))
    output_root = args.output_root.resolve()
    npz_path = output_root / f"{stem}.npz"
    json_path = output_root / f"{stem}.json"
    arrays = {
        "time_ms": np.asarray(result["times"], np.float32),
        "population_rate_E_hz_raw": np.asarray(rate_raw, np.float32),
        "population_rate_E_hz_smooth_2ms": np.asarray(rate_smooth, np.float32),
        "virtual_seeg": np.asarray(result["lfp_trace"], np.float32),
        "contact_names": np.asarray(substrate.contact_names, dtype="U16"),
        "contact_shaft_ids": np.asarray(substrate.shaft_ids, dtype="U16"),
        "contact_xy_mm": np.asarray(substrate.contact_xy, np.float32),
        "spatial_spike_count_20ms_1mm": frames,
        "spatial_frame_time_ms": frame_times,
        "sampled_spike_time_ms": np.asarray(sample_step * dt_ms, np.float32),
        "sampled_spike_neuron_index": np.asarray(sampled_neuron, np.int32),
        "sampled_neuron_index": sample_index,
        "sampled_neuron_xy_mm": np.asarray(
            substrate.positions_e[sample_index], np.float32),
    }
    arrays.update({f"slow_{key}": np.asarray(value, np.float32)
                   for key, value in slow_arrays.items()})
    if weighted is not None:
        arrays.update({f"slow_weighted_{key}": np.asarray(value, np.float32)
                       for key, value in weighted.items()})
    if regional_slow is not None:
        arrays.update({f"slow_region_{key}": np.asarray(value, np.float32)
                       for key, value in regional_slow.items()})
    _atomic_npz(npz_path, **arrays)
    payload = {
        "status": "DUAL_CORE_OSCILLATORY_SNN_PROBE_COMPLETE",
        "scientific_role": "development_only_fast_inhibition_sensitivity",
        "candidate_id": args.candidate_id,
        "topology_seed": int(args.topology_seed),
        "dynamics_seed": int(args.dynamics_seed),
        "tau_d_GABA_ms": float(args.tau_d_gaba_ms),
        "baseline_tau_d_GABA_ms": 18.0,
        "simulated_ms": float(len(rate_raw) * dt_ms),
        "operational_runaway_onset_ms": result["runaway_early_stop_ms"],
        "post_runaway_recorded_ms": float(result["post_runaway_recorded_ms"]),
        "state_assay": state,
        "contact_oscillation_assay": terminal_contact_state,
        "contact_oscillation_assay_by_role": {
            "early_recruitment": early_contact_state,
            "terminal_high_state": terminal_contact_state,
        },
        "regional_tail_rate_hz": {
            "core_a": regional[0], "core_b": regional[1],
            "surround": regional[2]},
        "slow_summary": slow.summary(),
        "external_drive": result["external_e_rate_drive"],
        "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        "sources": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "transition_config": {
                "path": str(transition_path), "sha256": _sha256(transition_path)},
            "candidate_manifest": {
                "path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "claim_boundary": (
            "One exploratory 40,000-cell SNN sensitivity run. The frozen graph, "
            "dual-core field, Z/M equations and spatial OU process are retained, "
            "but tau_d_GABA is retuned; this is not rev21 confirmation or a "
            "patient mechanism claim."),
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(json_path, payload)
    print(json.dumps({
        "status": payload["status"], "state": state["state"],
        "frequency_hz": state["whole_tail"]["dominant_hz"],
        "modulation_depth": state["whole_tail"]["modulation_depth"],
        "mean_rate_hz": state["whole_tail"]["mean_rate_hz"],
        "onset_ms": result["runaway_early_stop_ms"],
        "output": str(json_path), "wall_seconds": payload["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
