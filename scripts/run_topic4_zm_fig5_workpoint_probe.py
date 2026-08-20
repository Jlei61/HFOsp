#!/usr/bin/env python3
"""Run one paired Figure 5 probe at a frozen Z/M work point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from scripts.run_topic4_zm_perturbation_worker import _continue  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402
from src.topic4_zm_perturbation import (  # noqa: E402
    frozen_sites, in_window_ignition, response_metrics, select_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay-meta", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, choices=("low_activity", "pre_ictal"))
    parser.add_argument("--dose-cells", type=int, default=64)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    replay_meta = json.loads(Path(args.replay_meta).read_text())
    parameters = replay_meta["workpoint_parameters"]
    seed = int(replay_meta["seed"])
    config = load_round_config(args.config)
    config["zm"] = {
        **config["zm"],
        "I_th_EI": float(parameters["I_th_EI"]),
        "tau_z": float(parameters["tau_z"]),
        "tau_adp": float(parameters["tau_adp"]),
        "eta_m": float(parameters["eta_m"]),
    }
    candidate = config["arms"]["Joint"]
    output_root = ROOT / config["output_root"]
    started = time.time()
    substrate = build_substrate(
        config, candidate, seed,
        cache_dir=str(output_root / "network_cache"),
        ee_dose=float(parameters["E_to_E_dose"]),
        etoi_dose=float(parameters["E_to_I_dose"]),
    )
    state = ckpt.load(args.checkpoint)
    sites = [site for site in frozen_sites(substrate, config, kind="representative")
             if site["site_id"] == "source"]
    if len(sites) != 1:
        raise RuntimeError(f"expected one frozen source site, found {len(sites)}")
    site = sites[0]
    packet = select_packet(
        substrate.positions_e, site["xy_mm"], n_cells=int(args.dose_cells),
        radius_mm=float(config["perturbation"]["packet_radius_mm"]),
    )
    window_ms = float(config["perturbation"]["response_window_ms"])
    dt = float(substrate.engine["dt"])
    accumulate_steps = int(round(100.0 / dt))
    sham, sham_slow = _continue(
        substrate, config, state, duration_ms=window_ms,
        accumulate_steps=accumulate_steps,
    )
    probe, _ = _continue(
        substrate, config, state, duration_ms=window_ms, packet=packet,
    )
    cmrun = substrate.extras["cmrun"]
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    probe_active, _ = cmrun.active_fraction(
        np.asarray(probe["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    metrics = response_metrics(
        probe, sham, dt_ms=dt, positions_e=substrate.positions_e,
        packet_mask=packet, packet_xy=site["xy_mm"],
        envelope_probe=np.zeros((15, 1)), envelope_sham=np.zeros((15, 1)),
        envelope_dt_ms=2.0, inject_step=0,
        split_ms=float(config["perturbation"]["response_split_ms"]),
        window_ms=window_ms,
    )
    regime = in_window_ignition(
        probe_active, sham_active, active_dt_ms=float(active_dt),
        detector_threshold=substrate.detector_threshold,
        inject_ms=0.0, window_ms=window_ms,
        probe_rate_hz=np.asarray(probe["rate_E"], float), dt_ms=dt,
        es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
        es_dur_ms=float(config["simulation"]["es_dur_ms"]),
    )
    field = sham_slow.field_accumulator_result()
    out = Path(args.out_prefix).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out.with_suffix(".npz"),
        site_id=np.asarray(["source"], dtype="U12"),
        site_xy_mm=np.asarray([site["xy_mm"]], float),
        susceptibility=np.asarray([metrics["susceptibility"]], np.float32),
        e1_evaluable=np.asarray([regime["e1_evaluable"]], bool),
        probe_attributable_event_200ms=np.asarray(
            [regime["probe_attributable_event_200ms"]], bool),
        reached_model_ictal_200ms=np.asarray(
            [regime["reached_model_ictal_200ms"]], bool),
        excess_per_neuron=np.asarray([metrics["excess_per_neuron"]], np.float32),
        excess_per_neuron_early=np.asarray(
            [metrics["excess_per_neuron_early"]], np.float32),
        slow_field_D=field["disinhibition_D"].astype(np.float32),
        slow_field_A=field["adaptation_A"].astype(np.float32),
        slow_field_net=field["net_slow_current"].astype(np.float32),
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "status": "ZM_FIG5_WORKPOINT_PROBE_COMPLETE",
        "candidate_id": candidate,
        "seed": seed,
        "label": args.label,
        "dose_cells": int(args.dose_cells),
        "site": site,
        "workpoint_parameters": parameters,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_absolute_time_ms": float(state["absolute_time_ms"]),
        "metrics": {key: value for key, value in metrics.items()
                    if not isinstance(value, np.ndarray)},
        "regime": regime,
        "git_head": head,
        "wall_seconds": time.time() - started,
    }
    atomic_write_json(payload, str(out.with_suffix(".json")))
    print(json.dumps({
        "label": args.label,
        "e1_evaluable": bool(regime["e1_evaluable"]),
        "probe_event": bool(regime["probe_attributable_event_200ms"]),
        "wall_s": round(time.time() - started, 1),
    }))


if __name__ == "__main__":
    main()
