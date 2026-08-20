#!/usr/bin/env python3
"""Replay one frozen Z/M work point and emit the complete Figure 5 state."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.replay_topic4_zm_fig5_frames import (  # noqa: E402
    _activity_frames, _repo_relative_output, _select_display_event,
)
from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive,
)
from src.topic4_zm_slow_vars import ZMTracedSlowVars  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-canary", required=True,
                        help="Prefix of the selected canary JSON/NPZ")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frame-dt-ms", type=float, default=5.0)
    parser.add_argument("--activity-window-ms", type=float, default=10.0)
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--low-activity-ms", type=float, default=1000.0)
    parser.add_argument("--pre-onset-offset-ms", type=float, default=500.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--workpoint-root", required=True)
    args = parser.parse_args()

    reference_prefix = Path(args.reference_canary).resolve()
    reference_meta = json.loads(reference_prefix.with_suffix(".json").read_text())
    with np.load(reference_prefix.with_suffix(".npz"), allow_pickle=False) as handle:
        reference = {key: handle[key] for key in handle.files}
    if int(reference_meta["seed"]) != int(args.seed):
        raise RuntimeError("reference canary and requested seed differ")
    parameters = reference_meta["parameters"]
    morphology = reference_meta["runaway_morphology"]
    scientific_onset_ms = float(morphology["scientific_onset_ms"])

    config = load_round_config(args.config)
    config["simulation"] = dict(config["simulation"])
    config["simulation"]["duration_ms"] = 10000.0
    candidate = config["arms"]["Joint"]
    output_root = ROOT / config["output_root"]
    started = time.time()
    substrate = build_substrate(
        config, candidate, int(args.seed),
        cache_dir=str(output_root / "network_cache"),
        ee_dose=float(parameters["E_to_E_dose"]),
        etoi_dose=float(parameters["E_to_I_dose"]),
    )
    dt = float(substrate.engine["dt"])
    slow = ZMTracedSlowVars(
        substrate.n_e + substrate.n_i,
        substrate.params.V_th,
        MZSlowVarsConfig(
            use_z=True, use_m=True,
            I_th_EI=float(parameters["I_th_EI"]),
            tau_z=float(parameters["tau_z"]),
            tau_adp=float(parameters["tau_adp"]),
            eta_m=float(parameters["eta_m"]),
            trace_stride_steps=int(config["zm"]["trace_stride_steps"]),
        ),
        NE=substrate.n_e,
        core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        trace_weights_E=substrate.h_e,
    )
    frame_stride = int(round(float(args.frame_dt_ms) / dt))
    slow.enable_field_frames(frame_stride)
    drive = make_external_drive(substrate, config["spatial_ou"], int(args.seed))

    checkpoint_steps = {
        "low_activity": int(round(float(args.low_activity_ms) / dt)),
        "pre_ictal": int(round(
            (scientific_onset_ms - float(args.pre_onset_offset_ms)) / dt)),
    }
    if checkpoint_steps["pre_ictal"] <= checkpoint_steps["low_activity"]:
        raise RuntimeError("pre-onset checkpoint does not follow the low-activity state")
    captured = {}

    from kick_probe import simulate_kick
    from lfp import LFPRecorder

    recorder = LFPRecorder(
        substrate.params, substrate.net["pos"], substrate.net["labels"],
        sites=substrate.contact_xy,
    )
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        lfp_recorder=recorder, early_stop_runaway=True,
        es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
        es_dur_ms=float(config["simulation"]["es_dur_ms"]),
        post_runaway_record_ms=2000.0,
        external_e_rate_drive=drive,
        checkpoint_steps=list(checkpoint_steps.values()),
        checkpoint_sink=lambda step, state: captured.setdefault(step, state),
    )
    operational_onset_ms = result["runaway_early_stop_ms"]
    verification = {
        "reference_json": str(reference_prefix.with_suffix(".json")),
        "reference_npz": str(reference_prefix.with_suffix(".npz")),
        "reference_json_sha256": _sha256(reference_prefix.with_suffix(".json")),
        "reference_npz_sha256": _sha256(reference_prefix.with_suffix(".npz")),
        "operational_onset_identical": bool(
            operational_onset_ms == reference_meta["operational_onset_ms"]),
        "rate_E_identical": bool(np.array_equal(
            np.asarray(result["rate_E"], np.float32), reference["rate_E_hz"])),
        "lfp_identical": bool(np.array_equal(
            np.asarray(result["lfp_trace"], np.float32), reference["lfp_trace"])),
    }
    verification["all_match"] = all(
        verification[key] for key in (
            "operational_onset_identical", "rate_E_identical", "lfp_identical")
    )
    if not verification["all_match"]:
        raise RuntimeError(f"workpoint replay diverged from canary: {verification}")
    missing_checkpoints = sorted(
        set(checkpoint_steps.values()).difference(captured)
    )
    if missing_checkpoints:
        raise RuntimeError(f"replay missed checkpoints {missing_checkpoints}")

    spikes = np.asarray(result["E_spk_bool"], bool)
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(spikes, dt, cmrun.BIN_MS)
    from src.sef_hfo_events import detect_events
    from src.sef_hfo_snn_adapter import snn_event_envelope

    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, substrate.positions_e, substrate.montage, dt)
    detected = detect_events(
        active, active_dt, event_on_frac=substrate.detector_threshold)
    onsets, ranks, event_rows = [], [], []
    for index, event in enumerate(detected):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, substrate.montage,
            substrate.valid_contacts, (event["t_on"], event["t_off"]),
            0.1, 0.5,
        )
        onsets.append(onset)
        ranks.append(rank)
        event_rows.append({
            "event_index": int(index),
            "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "returned": bool(event["returned"]),
            "before_onset": bool(float(event["t_off"]) < scientific_onset_ms),
        })
    onsets = np.asarray(onsets, float).reshape((-1, len(substrate.contact_names)))
    ranks = np.asarray(ranks, float).reshape((-1, len(substrate.contact_names)))
    sample_index = _select_display_event(
        [row["t_on_ms"] for row in event_rows],
        [row["t_off_ms"] for row in event_rows],
        [row["returned"] for row in event_rows],
        [row["before_onset"] for row in event_rows],
        onsets, scientific_onset_ms,
    )
    sample_t_on = event_rows[sample_index]["t_on_ms"]
    sample_t_off = event_rows[sample_index]["t_off_ms"]
    sample_lo = max(0, int(round(sample_t_on / dt)))
    sample_hi = min(spikes.shape[0], int(round(sample_t_off / dt)) + 1)
    sample_spikes = spikes[sample_lo:sample_hi]
    sample_active = np.any(sample_spikes, axis=0)
    sample_first = np.full(substrate.n_e, np.nan, np.float32)
    sample_first[sample_active] = (
        sample_lo + np.argmax(sample_spikes[:, sample_active], axis=0)
    ) * dt

    energy_lo = int(round(scientific_onset_ms / dt))
    energy_hi = min(spikes.shape[0], int(round((scientific_onset_ms + 100.0) / dt)))
    energy_window_s = max((energy_hi - energy_lo) * dt * 1e-3, 1e-9)
    early_energy = np.square(
        spikes[energy_lo:energy_hi].sum(axis=0) / energy_window_s
    ).astype(np.float32)

    frames = slow.field_frames()
    frame_steps = np.asarray(frames["call_index"], int)
    frame_steps = frame_steps[frame_steps < spikes.shape[0]]
    activity, occupancy = _activity_frames(
        spikes, substrate.positions_e, frame_steps,
        int(round(float(args.activity_window_ms) / dt)),
        int(args.grid_n), float(substrate.engine["L"]),
    )
    weighted = slow.weighted_trace_arrays()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        frame_time_ms=(frame_steps * dt).astype(np.float32),
        net_slow_field=frames["net_slow_current"][:len(frame_steps)],
        activity_spike_counts=activity,
        activity_cell_occupancy=occupancy,
        positions_E=substrate.positions_e.astype(np.float32),
        h=substrate.h_e.astype(np.float32),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        lfp_trace=np.asarray(result["lfp_trace"], np.float32),
        lfp_dt_ms=np.asarray(dt, float),
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        contact_xy_mm=substrate.contact_xy,
        shaft_ids=substrate.shaft_ids,
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        rate_E_hz=np.asarray(result["rate_E"], np.float32),
        full_field_time_ms=reference["full_field_time_ms"],
        active_neuron_fraction_20ms=reference["active_neuron_fraction_20ms"],
        recruited_spatial_fraction_1mm=reference["recruited_spatial_fraction_1mm"],
        zm_h_weighted_time_ms=weighted["time_ms"],
        zm_h_weighted_z=weighted["z_weighted_mean"],
        zm_h_weighted_m=weighted["m_weighted_mean"],
        zm_h_weighted_net=weighted["net_slow_current_weighted_mean"],
        sample_event_index=np.asarray(sample_index, np.int32),
        sample_event_t_on_ms=np.asarray(sample_t_on, float),
        sample_event_t_off_ms=np.asarray(sample_t_off, float),
        sample_contact_onsets_ms=np.asarray(onsets[sample_index], np.float32),
        sample_contact_ranks=np.asarray(ranks[sample_index], np.float32),
        sample_first_spike_ms=sample_first,
        early_activity_energy=early_energy,
        early_activity_energy_window_ms=np.asarray(100.0, float),
        axis_source_xy=substrate.axis_source_xy,
        axis_sink_xy=substrate.axis_sink_xy,
        axis_unit=substrate.axis_unit,
    )

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_records = {}
    for label, step in checkpoint_steps.items():
        path = checkpoint_dir / f"joint_04_control_seed_{args.seed}_{label}.npz"
        checkpoint_records[label] = {
            "path": str(path),
            "step": int(step),
            "absolute_time_ms": float(captured[step]["absolute_time_ms"]),
            "sha256": ckpt.save(captured[step], path),
        }

    workpoint_root = Path(args.workpoint_root).resolve()
    worker_path = workpoint_root / "workers" / f"joint_04_control_seed_{args.seed}.npz"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(worker_path, positions_E=substrate.positions_e.astype(np.float32))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    meta = {
        "status": "ZM_FIG5_WORKPOINT_REPLAY_COMPLETE",
        "candidate_id": candidate,
        "seed": int(args.seed),
        "workpoint_parameters": parameters,
        "model_ictal_onset_ms": float(operational_onset_ms),
        "morphology_onset_ms": scientific_onset_ms,
        "runaway_morphology": morphology,
        "frame_dt_ms": float(args.frame_dt_ms),
        "activity_window_ms": float(args.activity_window_ms),
        "grid_n": int(args.grid_n),
        "sample_event_selection": {
            "rule": ("latest returned pre-transition event with at least 8 readable "
                     "contacts; independent of direction, KMeans and appearance"),
            "index": int(sample_index),
            "t_on_ms": sample_t_on,
            "t_off_ms": sample_t_off,
            "n_readable_contacts": int(np.isfinite(onsets[sample_index]).sum()),
        },
        "verification_against_reference_run": verification,
        "checkpoints": checkpoint_records,
        "frames_do_not_consume_random_numbers": True,
        "git_head": head,
        "wall_seconds": time.time() - started,
        "npz": _repo_relative_output(out),
    }
    atomic_write_json(meta, str(out.with_suffix(".json")))
    print(json.dumps({
        "verified": verification["all_match"],
        "onset_ms": scientific_onset_ms,
        "n_frames": int(len(frame_steps)),
        "checkpoints": checkpoint_records,
        "wall_s": round(time.time() - started, 1),
    }))


if __name__ == "__main__":
    main()
