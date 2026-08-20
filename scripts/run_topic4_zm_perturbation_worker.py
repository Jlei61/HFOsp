#!/usr/bin/env python3
"""Paired sham/probe perturbation from a checkpoint.

The sham is computed ONCE per (network, checkpoint, splice) and reused for every
site: it is identical across sites by construction, and computing it per site
would double the cost of the largest phase in the round.

Every site carries its own in-window regime flags, grid sites included. Freezing
the dose on baseline checkpoints guarantees nothing at the pre-ictal checkpoint,
which is exactly where excitability is hypothesised to be higher; without the
flags, a pre-ictal probe that ignites would have its escape-dominated spike count
recorded as a larger finite response. An igniting site is marked not-evaluable
and handed to the ignition endpoint -- it is never deleted, because deleting
igniting sites would strip the most excitable locations out of the map.
"""
from __future__ import annotations

import argparse
import copy
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

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz, _runtime_provenance)
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_d4 import D4_ELEMENTS  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive, make_slow)
from src.topic4_zm_perturbation import (  # noqa: E402
    SPLICE_MODES, frozen_sites, ignition_metrics, in_window_ignition,
    response_metrics, select_packet, splice_checkpoint)


def _resume_with_accumulator(state, *, n_e, n_steps):
    """Copy a checkpoint and enable a fresh read-only slow-current recorder."""
    resume = copy.deepcopy(state)
    if resume.get("slow") is None:
        raise ValueError("a slow-current accumulator needs a slow checkpoint")
    resume["slow"]["acc_n"] = int(n_steps)
    resume["slow"]["acc_seen"] = 0
    resume["slow"]["acc_D"] = np.zeros(int(n_e), dtype=float)
    resume["slow"]["acc_A"] = np.zeros(int(n_e), dtype=float)
    return resume


def _continue(substrate, config, state, *, duration_ms, packet=None,
              accumulate_steps=0, early_stop=False, checkpoint_steps=None,
              checkpoint_sink=None):
    """One continuation from a checkpoint. `packet=None` is the sham."""
    from kick_probe import simulate_kick
    from params import Params

    engine, dt = substrate.engine, float(substrate.engine["dt"])
    offset = float(state["absolute_time_ms"])
    params = Params(g=engine["g"], L=engine["L"], density=engine["density"],
                    T=float(duration_ms), dt=dt,
                    nu_ext_ratio=substrate.params.nu_ext_ratio,
                    seed=int(substrate.params.seed))
    slow = make_slow(substrate, config["zm"], trace_weights_E=substrate.h_e)
    drive = make_external_drive(substrate, config["spatial_ou"],
                                int(substrate.params.seed))
    resume_state = copy.deepcopy(state)
    if accumulate_steps and slow is not None:
        # restore_slow runs after the live object is built, so enabling only on
        # `slow` is silently undone by an ordinary checkpoint's acc_n=0.
        resume_state = _resume_with_accumulator(
            state, n_e=substrate.n_e, n_steps=int(accumulate_steps))
    kwargs = {}
    if packet is not None:
        full = np.zeros(substrate.n_e + substrate.n_i, bool)
        full[:substrate.n_e] = packet
        kwargs = {"forced_spike_mask": full, "forced_spike_ms": offset}
    result = simulate_kick(
        params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        early_stop_runaway=bool(early_stop),
        es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
        es_dur_ms=float(config["simulation"]["es_dur_ms"]),
        external_e_rate_drive=drive,
        resume_state=resume_state, time_offset_ms=offset,
        checkpoint_steps=checkpoint_steps, checkpoint_sink=checkpoint_sink,
        **kwargs)
    return result, slow


def _json_safe(value):
    """Recursively convert numpy types for json.dump.

    atomic_write_json uses a plain json.dump with no default=, and the state and
    recruitment blocks carry nested dicts containing ndarrays. Filtering only the
    TOP level let one through and killed a 92-minute run at its final write --
    after the simulation, so the whole run was lost. Sanitise recursively.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--label", required=True,
                        choices=("low_activity", "early_transition",
                                 "pre_ictal", "sensitivity"))
    parser.add_argument("--splice", default="native", choices=("native",) + SPLICE_MODES)
    parser.add_argument("--sites", required=True, choices=("grid", "representative"))
    parser.add_argument("--dose-cells", required=True, type=int)
    parser.add_argument("--measure-onset-advance", action="store_true")
    parser.add_argument("--onset-cap-ms", type=float, default=20000.0)
    parser.add_argument("--sham-onset-ms", type=float)
    parser.add_argument("--field-transform", default="none",
                        choices=("none",) + D4_ELEMENTS)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allow-uncommitted-config", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    if args.measure_onset_advance and args.sites == "grid":
        parser.error("the long onset-advance arm is representative-sites only; "
                     "grid jobs stay cheap and carry E1 plus the in-window flags")
    if args.splice != "native":
        if args.sites != "representative":
            parser.error("counterfactual splices run at representative sites only")
        if not args.baseline_checkpoint:
            parser.error("a splice needs --baseline-checkpoint")

    config = load_round_config(args.config)
    provenance = _runtime_provenance(args.expected_commit)
    config_path = Path(args.config).resolve()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{args.expected_commit}:{config_path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL)
        committed_sha = hashlib.sha256(committed).hexdigest()
    except (subprocess.CalledProcessError, ValueError):
        committed_sha = None
    provenance.update(config_sha256=config_sha,
                      config_sha256_at_expected_commit=committed_sha,
                      config_matches_expected_commit=config_sha == committed_sha)
    if not args.allow_uncommitted_config and committed_sha != config_sha:
        raise RuntimeError("config differs from the launcher commit")
    if provenance["runtime_modules_dirty"]:
        raise RuntimeError("runtime modules are dirty")

    started = time.time()
    output_root = ROOT / config["output_root"]
    transform = None if args.field_transform == "none" else args.field_transform
    substrate = build_substrate(config, args.candidate_id, args.seed,
                                cache_dir=str(args.cache_dir
                                              or output_root / "network_cache"),
                                field_transform=transform)
    perturbation = config["perturbation"]
    dt = float(substrate.engine["dt"])
    window_ms = float(perturbation["response_window_ms"])
    accumulate_steps = int(round(100.0 / dt))

    host = ckpt.load(args.checkpoint)
    off_manifold = False
    splice_mode = args.splice
    if splice_mode != "native":
        baseline_state = ckpt.load(args.baseline_checkpoint)
        host = splice_checkpoint(host, baseline_state, mode=splice_mode)
        off_manifold = bool(host["off_manifold"])

    sites = frozen_sites(substrate, config, kind=args.sites)
    cmrun = substrate.extras["cmrun"]

    # ---- one sham for every site at this (checkpoint, splice) ----
    sham, sham_slow = _continue(substrate, config, host, duration_ms=window_ms,
                                accumulate_steps=accumulate_steps)
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    field = (sham_slow.field_accumulator_result() if sham_slow is not None else None)
    sham_long, sham_onset = None, args.sham_onset_ms
    if args.measure_onset_advance and sham_onset is None:
        remaining = float(args.onset_cap_ms) - float(host["absolute_time_ms"])
        sham_long, _ = _continue(substrate, config, host, duration_ms=remaining,
                                 early_stop=True)
        sham_onset = sham_long["runaway_early_stop_ms"]

    rows, excess_fields, early_excess_fields, dropped = [], [], [], []
    for site in sites:
        try:
            packet = select_packet(substrate.positions_e, site["xy_mm"],
                                   n_cells=int(args.dose_cells),
                                   radius_mm=float(perturbation["packet_radius_mm"]))
        except ValueError as exc:
            dropped.append({"site_id": site["site_id"], "reason": str(exc)})
            continue
        probe, _ = _continue(substrate, config, host, duration_ms=window_ms,
                             packet=packet)
        probe_active, _ = cmrun.active_fraction(
            np.asarray(probe["E_spk_bool"], bool), dt, cmrun.BIN_MS)
        metrics = response_metrics(
            probe, sham, dt_ms=dt, positions_e=substrate.positions_e,
            packet_mask=packet, packet_xy=site["xy_mm"],
            envelope_probe=np.zeros((15, 1)), envelope_sham=np.zeros((15, 1)),
            envelope_dt_ms=2.0, inject_step=0,
            split_ms=float(perturbation["response_split_ms"]), window_ms=window_ms)
        regime = in_window_ignition(
            probe_active, sham_active, active_dt_ms=float(active_dt),
            detector_threshold=substrate.detector_threshold, inject_ms=0.0,
            window_ms=window_ms, probe_rate_hz=np.asarray(probe["rate_E"], float),
            dt_ms=dt, es_thresh_hz=float(config["simulation"]["es_thresh_hz"]),
            es_dur_ms=float(config["simulation"]["es_dur_ms"]))
        row = {"site_id": site["site_id"],
               "site_xy_mm": [float(v) for v in site["xy_mm"]],
               "off_manifold": off_manifold, "splice_mode": splice_mode,
               **{k: v for k, v in metrics.items()
                  if k not in {"excess_per_neuron", "excess_per_neuron_early"}},
               **regime}
        if args.measure_onset_advance:
            remaining = float(args.onset_cap_ms) - float(host["absolute_time_ms"])
            probe_long, _ = _continue(substrate, config, host, duration_ms=remaining,
                                      packet=packet, early_stop=True)
            row.update(ignition_metrics(
                probe_active, sham_active, active_dt_ms=float(active_dt),
                detector_threshold=substrate.detector_threshold, inject_ms=0.0,
                window_ms=window_ms,
                probe_onset_ms=probe_long["runaway_early_stop_ms"],
                sham_onset_ms=sham_onset))
        rows.append(row)
        excess_fields.append(metrics["excess_per_neuron"])
        early_excess_fields.append(metrics["excess_per_neuron_early"])

    stem = Path(args.checkpoint).stem + f"_{args.sites}"
    if splice_mode != "native":
        stem += f"_{splice_mode}"
    out_json = Path(args.out_json or output_root / "perturbation" / f"{stem}.json")
    out_npz = Path(args.out_npz or output_root / "perturbation" / f"{stem}.npz")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    evaluable = [r for r in rows if r["e1_evaluable"]]
    atomic_write_json(_json_safe({
        "status": "ZM_ITX_PERTURBATION_COMPLETE",
        "candidate_id": args.candidate_id, "seed": int(args.seed),
        "label": args.label, "splice_mode": splice_mode,
        "off_manifold": off_manifold, "sites": args.sites,
        "dose_cells": int(args.dose_cells),
        "checkpoint": str(Path(args.checkpoint).relative_to(ROOT)),
        "checkpoint_absolute_time_ms": float(host["absolute_time_ms"]),
        "n_sites": len(rows), "n_e1_evaluable": len(evaluable),
        "n_dropped_sites": len(dropped), "dropped_sites": dropped,
        "ignition_fraction": (float(sum(not r["e1_evaluable"] for r in rows) / len(rows))
                              if rows else None),
        "sham_onset_ms": sham_onset,
        "rows": rows,
        "wall_seconds": time.time() - started,
        "provenance": provenance,
    }), str(out_json))

    arrays = {
        "site_id": np.asarray([r["site_id"] for r in rows], dtype="U12"),
        "site_xy_mm": np.asarray([r["site_xy_mm"] for r in rows], float).reshape(-1, 2),
        "susceptibility": np.asarray([r["susceptibility"] for r in rows], np.float32),
        "excess_spikes_early": np.asarray([r["excess_spikes_early"] for r in rows], np.float32),
        "excess_spikes_late": np.asarray([r["excess_spikes_late"] for r in rows], np.float32),
        "r90_mm": np.asarray([r["r90_mm"] for r in rows], np.float32),
        "e1_evaluable": np.asarray([r["e1_evaluable"] for r in rows], bool),
        "probe_attributable_event_200ms": np.asarray(
            [r["probe_attributable_event_200ms"] for r in rows], bool),
        "reached_model_ictal_200ms": np.asarray(
            [r["reached_model_ictal_200ms"] for r in rows], bool),
        "excess_per_neuron": (np.asarray(excess_fields, np.float32)
                              if excess_fields else np.zeros((0, substrate.n_e), np.float32)),
        "excess_per_neuron_early": (
            np.asarray(early_excess_fields, np.float32)
            if early_excess_fields else np.zeros((0, substrate.n_e), np.float32)),
    }
    if args.measure_onset_advance and rows:
        arrays["onset_advance_ms"] = np.asarray(
            [r.get("onset_advance_ms", np.nan) for r in rows], np.float32)
        arrays["onset_censored"] = np.asarray(
            [r.get("onset_censored", True) for r in rows], bool)
    if field is not None:
        arrays["slow_field_D"] = field["disinhibition_D"].astype(np.float32)
        arrays["slow_field_A"] = field["adaptation_A"].astype(np.float32)
        arrays["slow_field_net"] = field["net_slow_current"].astype(np.float32)
    _atomic_npz(out_npz, **arrays)
    print(json.dumps({"stem": stem, "n_sites": len(rows),
                      "n_e1_evaluable": len(evaluable),
                      "ignition_fraction": (round(sum(not r["e1_evaluable"] for r in rows)
                                                  / len(rows), 3) if rows else None),
                      "wall_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
