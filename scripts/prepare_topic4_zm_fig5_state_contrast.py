#!/usr/bin/env python3
"""Prepare an exact low-activity versus early-runaway Figure 5 contrast.

The selected Fig5 trajectory has work-point-specific Z/M and edge doses that are
not the defaults in the round config.  This producer reads those values from the
verified replay metadata, resumes its frozen checkpoints, and refuses output if
the continuation rate diverges from the replay.  It then runs the same weak
source-site probe at the low-activity and post-onset states and records both the
signed probe-minus-sham response and the slow-state/current decomposition.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def _continuation_slice(offset_ms, duration_ms, dt_ms):
    start = int(round(float(offset_ms) / float(dt_ms)))
    n_steps = int(round(float(duration_ms) / float(dt_ms)))
    return slice(start, start + n_steps)


def _apply_workpoint(config, parameters):
    config = dict(config)
    config["zm"] = dict(config["zm"])
    for key in ("I_th_EI", "tau_z", "tau_adp", "eta_m"):
        config["zm"][key] = float(parameters[key])
    return config


def _source_site(substrate, config):
    sites = frozen_sites(substrate, config, kind="representative")
    matches = [row for row in sites if row["site_id"] == "source"]
    if len(matches) != 1:
        raise RuntimeError("frozen representative geometry has no unique source site")
    return matches[0]


def _slow_snapshot(state, n_e, eta_m):
    slow = state.get("slow")
    if slow is None:
        raise RuntimeError("Figure 5 state contrast requires a Z/M checkpoint")
    z = np.asarray(slow["z"], float)[:n_e]
    m = np.asarray(slow["m"], float)[:n_e]
    return 1.0 - z, float(eta_m) * m


def _state_probe(substrate, config, state, site, doses, window_ms):
    dt = float(substrate.engine["dt"])
    accumulate_steps = int(round(100.0 / dt))
    sham, sham_slow = _continue(
        substrate, config, state, duration_ms=window_ms,
        accumulate_steps=accumulate_steps,
    )
    field = sham_slow.field_accumulator_result()
    if field is None or int(field["n_steps"]) != accumulate_steps:
        raise RuntimeError("slow-current accumulator did not cover the frozen 100 ms window")
    cmrun = substrate.extras["cmrun"]
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    rows = []
    early_fields = []
    full_fields = []
    for dose in doses:
        packet = select_packet(
            substrate.positions_e, site["xy_mm"], n_cells=int(dose),
            radius_mm=float(config["perturbation"]["packet_radius_mm"]),
        )
        probe, _ = _continue(
            substrate, config, state, duration_ms=window_ms, packet=packet)
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
            detector_threshold=substrate.detector_threshold, inject_ms=0.0,
            window_ms=window_ms, probe_rate_hz=np.asarray(probe["rate_E"], float),
            dt_ms=dt, es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
            es_dur_ms=float(config["simulation"]["es_dur_ms"]),
        )
        rows.append({
            "dose_cells": int(dose),
            "susceptibility": float(metrics["susceptibility"]),
            "excess_spikes_early": float(metrics["excess_spikes_early"]),
            "excess_spikes_late": float(metrics["excess_spikes_late"]),
            "r90_mm": float(metrics["r90_mm"]),
            **regime,
        })
        early_fields.append(metrics["excess_per_neuron_early"])
        full_fields.append(metrics["excess_per_neuron"])
    return {
        "rows": rows,
        "early_fields": np.asarray(early_fields, np.float32),
        "full_fields": np.asarray(full_fields, np.float32),
        "slow_D": np.asarray(field["disinhibition_D"], np.float32),
        "slow_A": np.asarray(field["adaptation_A"], np.float32),
        "slow_net": np.asarray(field["net_slow_current"], np.float32),
    }


def _select_dose(low_rows, *, max_abs_early_excess=50.0):
    eligible = [
        row for row in low_rows
        if bool(row["e1_evaluable"])
        and abs(float(row["excess_spikes_early"])) <= float(max_abs_early_excess)
    ]
    if not eligible:
        raise RuntimeError("no near-zero baseline subevent probe dose exists in the scanned ladder")
    return int(max(eligible, key=lambda row: int(row["dose_cells"]))["dose_cells"])


def _reuse_low_contrast(path, replay_meta, doses):
    """Reuse only the frozen low-activity branches from an exact prior contrast."""
    path = Path(path).resolve()
    meta = json.loads(path.with_suffix(".json").read_text())
    if not bool(meta.get("continuation_rate_exact")):
        raise RuntimeError("reused low contrast is not tied to an exact replay")
    if int(meta["seed"]) != int(replay_meta["seed"]):
        raise RuntimeError("reused low contrast has a different seed")
    if meta["workpoint_parameters"] != replay_meta["workpoint_parameters"]:
        raise RuntimeError("reused low contrast has a different work point")
    with np.load(path, allow_pickle=False) as handle:
        block = {key: handle[key] for key in handle.files}
    old_doses = np.asarray(block["dose_cells"], int)
    indices = []
    for dose in doses:
        where = np.flatnonzero(old_doses == int(dose))
        if where.size != 1:
            raise RuntimeError(f"reused low contrast has no unique dose {dose}")
        indices.append(int(where[0]))
    rows = {int(row["dose_cells"]): row for row in meta["low_probe_scan"]}
    return {
        "rows": [rows[int(dose)] for dose in doses],
        "early_fields": np.asarray(block["low_response_early"])[indices],
        "full_fields": np.asarray(block["low_response_full"])[indices],
        "slow_D": np.asarray(block["low_slow_D"]),
        "slow_A": np.asarray(block["low_slow_A"]),
        "slow_net": np.asarray(block["low_slow_net"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--low-checkpoint", required=True)
    parser.add_argument("--pre-checkpoint", required=True)
    parser.add_argument("--post-offset-ms", type=float, default=120.0)
    parser.add_argument("--dose-ladder", type=int, nargs="+", default=(8, 16, 32, 64))
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--max-baseline-early-excess", type=float, default=50.0)
    parser.add_argument("--reuse-low-contrast")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    replay_path = Path(args.replay).resolve()
    replay_meta = json.loads(replay_path.with_suffix(".json").read_text())
    with np.load(replay_path, allow_pickle=False) as handle:
        replay = {key: handle[key] for key in handle.files}
    parameters = replay_meta["workpoint_parameters"]
    config = _apply_workpoint(load_round_config(args.config), parameters)
    seed = int(replay_meta["seed"])
    candidate = replay_meta["candidate_id"]
    substrate = build_substrate(
        config, candidate, seed,
        cache_dir=str(ROOT / config["output_root"] / "network_cache"),
        ee_dose=float(parameters["E_to_E_dose"]),
        etoi_dose=float(parameters["E_to_I_dose"]),
    )
    dt = float(substrate.engine["dt"])
    low_state = ckpt.load(args.low_checkpoint)
    pre_state = ckpt.load(args.pre_checkpoint)
    onset_ms = float(replay_meta["morphology_onset_ms"])
    post_ms = onset_ms + float(args.post_offset_ms)
    target_step = int(round(post_ms / dt))
    duration_ms = post_ms - float(pre_state["absolute_time_ms"]) + dt
    captured = {}
    continuation, _ = _continue(
        substrate, config, pre_state, duration_ms=duration_ms,
        checkpoint_steps=[target_step],
        checkpoint_sink=lambda step, state: captured.setdefault(step, state),
    )
    if target_step not in captured:
        raise RuntimeError("continuation missed the post-onset checkpoint")
    replay_slice = _continuation_slice(
        pre_state["absolute_time_ms"], duration_ms, dt)
    exact_rate = np.array_equal(
        np.asarray(continuation["rate_E"], np.float32),
        np.asarray(replay["rate_E_hz"], np.float32)[replay_slice],
    )
    if not exact_rate:
        raise RuntimeError("post-onset continuation diverged from the verified replay")
    post_state = captured[target_step]

    site = _source_site(substrate, config)
    doses = sorted(set(int(value) for value in args.dose_ladder))
    if any(value <= 0 for value in doses):
        raise ValueError("probe doses must be positive")
    if args.reuse_low_contrast:
        low = _reuse_low_contrast(args.reuse_low_contrast, replay_meta, doses)
    else:
        low = _state_probe(
            substrate, config, low_state, site, doses, float(args.window_ms))
    post = _state_probe(substrate, config, post_state, site, doses, float(args.window_ms))
    selected_dose = _select_dose(
        low["rows"], max_abs_early_excess=float(args.max_baseline_early_excess))
    eta_m = float(parameters["eta_m"])
    low_disinh, low_adapt = _slow_snapshot(low_state, substrate.n_e, eta_m)
    post_disinh, post_adapt = _slow_snapshot(post_state, substrate.n_e, eta_m)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    post_checkpoint = out.with_name(out.stem + "-post-checkpoint.npz")
    post_sha = ckpt.save(post_state, post_checkpoint)
    _atomic_npz(
        out,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h=np.asarray(substrate.h_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        site_xy_mm=np.asarray(site["xy_mm"], np.float32),
        dose_cells=np.asarray(doses, np.int32),
        selected_dose_cells=np.asarray(selected_dose, np.int32),
        low_response_early=low["early_fields"],
        post_response_early=post["early_fields"],
        low_response_full=low["full_fields"],
        post_response_full=post["full_fields"],
        low_disinhibition_state=np.asarray(low_disinh, np.float32),
        post_disinhibition_state=np.asarray(post_disinh, np.float32),
        low_adaptation_state=np.asarray(low_adapt, np.float32),
        post_adaptation_state=np.asarray(post_adapt, np.float32),
        low_slow_D=low["slow_D"], low_slow_A=low["slow_A"],
        low_slow_net=low["slow_net"],
        post_slow_D=post["slow_D"], post_slow_A=post["slow_A"],
        post_slow_net=post["slow_net"],
    )
    weights = np.asarray(substrate.h_e, float)
    weights /= weights.sum()
    summary = {
        "status": "ZM_FIG5_STATE_CONTRAST_COMPLETE",
        "seed": seed,
        "workpoint_parameters": parameters,
        "source_site_xy_mm": [float(value) for value in site["xy_mm"]],
        "low_time_ms": float(low_state["absolute_time_ms"]),
        "post_time_ms": float(post_state["absolute_time_ms"]),
        "post_offset_from_scientific_onset_ms": float(args.post_offset_ms),
        "continuation_rate_exact": bool(exact_rate),
        "reused_low_contrast": (None if not args.reuse_low_contrast else
                                str(Path(args.reuse_low_contrast).resolve().relative_to(ROOT))),
        "selected_dose_cells": selected_dose,
        "selection_rule": ("largest scanned dose with no attributable baseline event, "
                           "no model-ictal crossing, and absolute 0-50 ms baseline "
                           f"excess <= {float(args.max_baseline_early_excess):g} spikes"),
        "low_probe_scan": low["rows"],
        "post_probe_scan": post["rows"],
        "h_weighted_state": {
            "low_1_minus_z": float(np.dot(weights, low_disinh)),
            "post_1_minus_z": float(np.dot(weights, post_disinh)),
            "low_eta_m_m": float(np.dot(weights, low_adapt)),
            "post_eta_m_m": float(np.dot(weights, post_adapt)),
            "low_current_D": float(np.dot(weights, low["slow_D"])),
            "post_current_D": float(np.dot(weights, post["slow_D"])),
            "low_current_A": float(np.dot(weights, low["slow_A"])),
            "post_current_A": float(np.dot(weights, post["slow_A"])),
        },
        "post_checkpoint": str(post_checkpoint.relative_to(ROOT)),
        "post_checkpoint_sha256": post_sha,
        "npz": str(out.relative_to(ROOT)),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
