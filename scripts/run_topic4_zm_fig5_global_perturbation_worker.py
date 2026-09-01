#!/usr/bin/env python3
"""Probe frozen low/runaway states at one chunk of global random sites."""
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

from scripts.prepare_topic4_zm_fig5_state_contrast import _apply_workpoint  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from scripts.run_topic4_zm_perturbation_worker import _continue  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_fig5 import stratified_random_sites  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402
from src.topic4_zm_perturbation import (  # noqa: E402
    in_window_ignition,
    response_metrics,
    select_packet,
)


def _probe_sites(substrate, config, state, sites, indices, *, dose_cells, window_ms):
    dt_ms = float(substrate.engine["dt"])
    sham, _ = _continue(substrate, config, state, duration_ms=float(window_ms))
    cmrun = substrate.extras["cmrun"]
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
    rows = []
    early_fields = []
    full_fields = []
    for site_index in indices:
        xy = np.asarray(sites[int(site_index)], float)
        packet = select_packet(
            substrate.positions_e, xy, n_cells=int(dose_cells),
            radius_mm=float(config["perturbation"]["packet_radius_mm"]),
        )
        probe, _ = _continue(
            substrate, config, state, duration_ms=float(window_ms), packet=packet)
        probe_active, _ = cmrun.active_fraction(
            np.asarray(probe["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
        metrics = response_metrics(
            probe, sham, dt_ms=dt_ms, positions_e=substrate.positions_e,
            packet_mask=packet, packet_xy=xy,
            envelope_probe=np.zeros((15, 1)),
            envelope_sham=np.zeros((15, 1)), envelope_dt_ms=2.0,
            inject_step=0,
            split_ms=float(config["perturbation"]["response_split_ms"]),
            window_ms=float(window_ms),
        )
        regime = in_window_ignition(
            probe_active, sham_active, active_dt_ms=float(active_dt),
            detector_threshold=substrate.detector_threshold, inject_ms=0.0,
            window_ms=float(window_ms),
            probe_rate_hz=np.asarray(probe["rate_E"], float), dt_ms=dt_ms,
            es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
            es_dur_ms=float(config["simulation"]["es_dur_ms"]),
        )
        rows.append({
            "site_index": int(site_index),
            "site_xy_mm": [float(xy[0]), float(xy[1])],
            "dose_cells": int(dose_cells),
            "susceptibility": float(metrics["susceptibility"]),
            "excess_spikes_early": float(metrics["excess_spikes_early"]),
            "excess_spikes_late": float(metrics["excess_spikes_late"]),
            "r90_mm": float(metrics["r90_mm"]),
            **regime,
        })
        early_fields.append(metrics["excess_per_neuron_early"])
        full_fields.append(metrics["excess_per_neuron"])
    return rows, np.asarray(early_fields, np.float32), np.asarray(full_fields, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--mode-snapshot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-label", choices=("low_activity", "runaway"), required=True)
    parser.add_argument("--site-indices", type=int, nargs="+", required=True)
    parser.add_argument("--n-side", type=int, default=4)
    parser.add_argument("--site-seed", type=int, default=20260820)
    parser.add_argument("--sheet-size-mm", type=float, default=20.0)
    parser.add_argument("--margin-mm", type=float, default=1.2)
    parser.add_argument("--dose-cells", type=int, default=16)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()

    replay_path = Path(args.replay).resolve()
    replay_meta = json.loads(replay_path.with_suffix(".json").read_text())
    snapshot_path = Path(args.mode_snapshot).resolve()
    snapshot_meta = json.loads(snapshot_path.with_suffix(".json").read_text())
    parameters = replay_meta["workpoint_parameters"]
    config = _apply_workpoint(load_round_config(args.config), parameters)
    seed = int(replay_meta["seed"])
    if int(snapshot_meta["seed"]) != seed:
        raise RuntimeError("mode snapshot and replay use different seeds")
    substrate = build_substrate(
        config, replay_meta["candidate_id"], seed,
        cache_dir=str(ROOT / config["output_root"] / "network_cache"),
        ee_dose=float(parameters["E_to_E_dose"]),
        etoi_dose=float(parameters["E_to_I_dose"]),
    )
    state = ckpt.load(args.checkpoint)
    selected_time = float(snapshot_meta["selected"]["time_ms"])
    expected_time = 1000.0 if args.state_label == "low_activity" else selected_time
    if not np.isclose(float(state["absolute_time_ms"]), expected_time, atol=1e-8):
        raise RuntimeError(
            f"{args.state_label} checkpoint time {state['absolute_time_ms']} != {expected_time}")

    sites = stratified_random_sites(
        n_side=int(args.n_side), extent_mm=(0.0, float(args.sheet_size_mm)),
        margin_mm=float(args.margin_mm), seed=int(args.site_seed))
    indices = sorted(set(int(index) for index in args.site_indices))
    if not indices or indices[0] < 0 or indices[-1] >= len(sites):
        raise ValueError("site indices fall outside the frozen site set")
    rows, early, full = _probe_sites(
        substrate, config, state, sites, indices,
        dose_cells=int(args.dose_cells), window_ms=float(args.window_ms))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        site_index=np.asarray(indices, np.int16),
        site_xy_mm=np.asarray(sites[indices], np.float32),
        excess_per_neuron_early=early,
        excess_per_neuron_full=full,
        susceptibility=np.asarray([row["susceptibility"] for row in rows], np.float32),
        excess_spikes_early=np.asarray(
            [row["excess_spikes_early"] for row in rows], np.float32),
        e1_evaluable=np.asarray([row["e1_evaluable"] for row in rows], bool),
    )
    summary = {
        "status": "ZM_FIG5_GLOBAL_PERTURBATION_CHUNK_COMPLETE",
        "state_label": args.state_label,
        "seed": seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_time_ms": float(state["absolute_time_ms"]),
        "mode_snapshot": str(snapshot_path),
        "site_contract": {
            "kind": "one uniform random point per square stratum",
            "n_side": int(args.n_side),
            "n_total": len(sites),
            "seed": int(args.site_seed),
            "sheet_extent_mm": [0.0, float(args.sheet_size_mm)],
            "edge_margin_mm": float(args.margin_mm),
        },
        "site_indices": indices,
        "dose_cells": int(args.dose_cells),
        "window_ms": float(args.window_ms),
        "rows": rows,
        "wall_seconds": time.time() - started,
        "npz": str(out),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
