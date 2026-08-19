#!/usr/bin/env python3
"""Visualisation replay of one frozen Joint run, for the base Figure 5.

Reproduces the archived trajectory exactly and additionally records what the
three-panel figure needs and the worker never saved: the per-neuron net slow
current every few milliseconds, and per-frame 2-D E activity.

Nothing here consumes a random number, and the replay is verified against the
archived contact envelope, active fraction and onset before its frames are
written -- a figure built from a re-randomised look-alike would be showing a
trajectory nobody analysed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive)
from src.topic4_zm_slow_vars import ZMTracedSlowVars  # noqa: E402


def _activity_frames(spikes, positions, frame_steps, window_steps, grid_n, sheet_l):
    """Fraction of E neurons active in each grid cell over a short window."""
    ix = np.clip((positions[:, 0] / sheet_l * grid_n).astype(int), 0, grid_n - 1)
    iy = np.clip((positions[:, 1] / sheet_l * grid_n).astype(int), 0, grid_n - 1)
    flat = ix * grid_n + iy
    counts = np.bincount(flat, minlength=grid_n * grid_n).astype(float)
    counts[counts == 0] = np.nan
    frames = np.empty((len(frame_steps), grid_n, grid_n), np.float32)
    for index, step in enumerate(frame_steps):
        lo = max(0, int(step) - window_steps)
        hi = min(spikes.shape[0], int(step) + 1)
        active = spikes[lo:hi].any(axis=0)
        binned = np.bincount(flat[active], minlength=grid_n * grid_n).astype(float)
        frames[index] = (binned / counts).reshape(grid_n, grid_n)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frame-dt-ms", type=float, default=5.0)
    parser.add_argument("--activity-window-ms", type=float, default=10.0)
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_round_config(args.config)
    output_root = ROOT / config["output_root"]
    candidate = config["arms"]["Joint"]
    started = time.time()

    substrate = build_substrate(config, candidate, args.seed,
                                cache_dir=str(output_root / "network_cache"))
    engine, dt = substrate.engine, float(substrate.engine["dt"])
    simulation = config["simulation"]
    zm = config["zm"]
    stride = int(round(float(args.frame_dt_ms) / dt))

    slow = ZMTracedSlowVars(
        substrate.n_e + substrate.n_i, substrate.params.V_th,
        MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=float(zm["I_th_EI"]),
                         tau_z=float(zm["tau_z"]), tau_adp=float(zm["tau_adp"]),
                         eta_m=float(zm["eta_m"]),
                         trace_stride_steps=int(zm["trace_stride_steps"])),
        NE=substrate.n_e, core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        trace_weights_E=substrate.h_e)
    slow.enable_field_frames(stride)
    drive = make_external_drive(substrate, config["spatial_ou"], args.seed)

    from kick_probe import simulate_kick
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        es_thresh_hz=float(simulation["es_thresh_hz"]),
        es_dur_ms=float(simulation["es_dur_ms"]),
        post_runaway_record_ms=float(simulation["post_runaway_record_ms"]),
        external_e_rate_drive=drive)
    onset_ms = result["runaway_early_stop_ms"]
    spikes = np.asarray(result["E_spk_bool"], bool)

    # ---- verify against the archived run BEFORE writing anything ----
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(spikes, dt, cmrun.BIN_MS)
    from src.sef_hfo_snn_adapter import snn_event_envelope
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, substrate.positions_e, substrate.montage, dt)
    archived_json = output_root / "workers" / f"{candidate}_seed_{args.seed}.json"
    archived_npz = output_root / "workers" / f"{candidate}_seed_{args.seed}.npz"
    verification = {"archived_run": str(archived_json.relative_to(ROOT))}
    archived = json.loads(archived_json.read_text())
    with np.load(archived_npz, allow_pickle=False) as handle:
        n = min(handle["contact_envelope"].shape[1], envelope.shape[1])
        verification["contact_envelope_identical"] = bool(np.array_equal(
            np.asarray(envelope, np.float32)[:, :n],
            np.asarray(handle["contact_envelope"])[:, :n]))
        m = min(len(handle["active_fraction"]), len(active))
        verification["active_fraction_identical"] = bool(np.array_equal(
            np.asarray(active, np.float32)[:m],
            np.asarray(handle["active_fraction"])[:m]))
    verification["onset_identical"] = bool(
        onset_ms == archived["run"]["model_ictal_onset_ms"])
    verification["archived_onset_ms"] = archived["run"]["model_ictal_onset_ms"]
    verification["replay_onset_ms"] = onset_ms
    verification["all_match"] = all(
        verification[k] for k in ("contact_envelope_identical",
                                  "active_fraction_identical", "onset_identical"))
    if not verification["all_match"]:
        raise SystemExit(f"replay diverged from the archived run: {verification}")

    frames = slow.field_frames()
    frame_steps = np.asarray(frames["call_index"], int)
    frame_steps = frame_steps[frame_steps < spikes.shape[0]]
    net_field = frames["net_slow_current"][:len(frame_steps)]
    activity = _activity_frames(
        spikes, substrate.positions_e, frame_steps,
        int(round(float(args.activity_window_ms) / dt)),
        int(args.grid_n), float(engine["L"]))
    weighted = slow.weighted_trace_arrays()

    out = Path(args.out or output_root / "fig5_replay" / f"{candidate}_seed_{args.seed}_frames.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        frame_time_ms=(frame_steps * dt).astype(np.float32),
        net_slow_field=net_field,
        activity_grid=activity,
        positions_E=substrate.positions_e.astype(np.float32),
        h=substrate.h_e.astype(np.float32),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        contact_xy_mm=substrate.contact_xy, shaft_ids=substrate.shaft_ids,
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        rate_E_hz=np.asarray(result["rate_E"], np.float32),
        zm_h_weighted_time_ms=weighted["time_ms"],
        zm_h_weighted_z=weighted["z_weighted_mean"],
        zm_h_weighted_m=weighted["m_weighted_mean"],
        zm_h_weighted_net=weighted["net_slow_current_weighted_mean"],
        axis_source_xy=substrate.axis_source_xy, axis_sink_xy=substrate.axis_sink_xy,
        axis_unit=substrate.axis_unit)
    atomic_write_json({
        "status": "ZM_ITX_FIG5_REPLAY_COMPLETE",
        "candidate_id": candidate, "seed": int(args.seed),
        "model_ictal_onset_ms": onset_ms,
        "frame_dt_ms": float(args.frame_dt_ms), "n_frames": int(len(frame_steps)),
        "activity_window_ms": float(args.activity_window_ms),
        "grid_n": int(args.grid_n),
        "verification_against_archived_run": verification,
        "frames_do_not_consume_random_numbers": True,
        "wall_seconds": time.time() - started,
        "npz": str(out.relative_to(ROOT)),
    }, str(out.with_suffix(".json")))
    print(json.dumps({"onset_ms": onset_ms, "n_frames": int(len(frame_steps)),
                      "verified": verification["all_match"],
                      "wall_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
