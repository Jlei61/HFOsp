#!/usr/bin/env python3
"""One primary Z/M ictal-transition run on one frozen network.

Two passes, because the onset-relative checkpoints are defined against an onset
that is not known in advance:
  pass 1  whole trajectory, emits only the 2000 ms baseline checkpoint, records onset
  pass 2  resumes from it and STOPS at onset-500 ms, emitting the two onset-relative
          checkpoints. Its overlap with pass 1 is asserted bit-identical -- Gate B
          applied in production.
Pass 2 runs for the Joint arm only; the latency arms need no onset-relative
checkpoints and stop after pass 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz, _runtime_provenance)
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_zm_slow_vars import ZMTracedSlowVars as MZSlowVars  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_d4 import D4_ELEMENTS, d4_matrix  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive, make_slow)
from src.topic4_zm_recruitment import (  # noqa: E402
    axial_lag, bin_baseline, bin_rate_traces, local_recruitment, spatial_bins)
from src.topic4_zm_state_characterization import (  # noqa: E402
    band_proxy, characterize_state, interictal_reference)

BASELINE_MS = 2000.0


def _montage_images(substrate, elements):
    """Pre-frozen transformed montages, sampled inside the SAME run.

    The 20 s per-neuron spike array is 6.4 GB and is never written, so the
    observation control cannot be reconstructed offline from the original
    envelope alone. Sampling the in-memory spikes through each montage costs one
    extra envelope per element and no simulation.
    """
    centre = float(substrate.engine["L"]) / 2.0
    images = {"identity": substrate.montage}
    for element in elements:
        if element == "identity":
            continue
        matrix = d4_matrix(element)
        xy = (matrix @ (substrate.contact_xy - centre).T).T + centre
        images[element] = VirtualMontage(
            xy, [f"{name}_{element}" for name in substrate.contact_names],
            provenance=f"observation_control_{element}")
    return images


def _audit_montages(substrate, images):
    """Every transformed montage must be locally readable, or its contacts are
    excluded and counted. A rotated contact can land where no E neuron is within
    the summation radius; silently keeping it would put a zero trace into the
    observation control."""
    cmrun = substrate.extras["cmrun"]
    audit = {}
    for name, montage in images.items():
        valid = np.asarray(cmrun.valid_mask(
            montage, substrate.positions_e, substrate.engine["L"],
            substrate.params.Rr), bool)
        audit[name] = {"n_contacts": int(valid.size),
                       "n_valid": int(valid.sum()),
                       "n_excluded": int((~valid).sum()),
                       "excluded_contacts": [str(montage.names[i])
                                             for i in np.flatnonzero(~valid)]}
    return audit


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
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--zm-mode", choices=("z_plus_m", "off"), default="z_plus_m")
    parser.add_argument("--field-transform", default="none",
                        choices=("none",) + D4_ELEMENTS)
    parser.add_argument("--emit-onset-checkpoints", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--allow-uncommitted-config", action="store_true",
                        help="development escape hatch; formal runs must not use it")
    args = parser.parse_args()

    config = load_round_config(args.config)
    if config["scientific_role"] != "development_only_data_driven_zm_interictal_to_ictal_transition":
        raise RuntimeError("scientific role changed")
    arms = config["arms"]
    arm_name = next((k for k, v in arms.items() if v == args.candidate_id), None)
    if arm_name is None:
        raise RuntimeError(f"{args.candidate_id!r} is outside the frozen arm set")
    if args.emit_onset_checkpoints and arm_name not in config["phases"][
            "onset_relative_checkpoints_for_arms"]:
        parser.error(f"onset-relative checkpoints are not defined for arm {arm_name}")

    provenance = _runtime_provenance(args.expected_commit)
    # _runtime_provenance only covers LOADED PYTHON MODULES. The config is an
    # input that changes numbers just as surely, so its identity is recorded and
    # checked against the launcher commit here.
    config_path = Path(args.config).resolve()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{args.expected_commit}:{config_path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL)
        committed_sha = hashlib.sha256(committed).hexdigest()
    except (subprocess.CalledProcessError, ValueError):
        committed_sha = None
    provenance["config_path"] = str(config_path.relative_to(ROOT))
    provenance["config_sha256"] = config_sha
    provenance["config_sha256_at_expected_commit"] = committed_sha
    provenance["config_matches_expected_commit"] = (config_sha == committed_sha)
    if not args.allow_uncommitted_config and committed_sha != config_sha:
        raise RuntimeError(
            "config differs from the launcher commit -- commit it or pass "
            "--allow-uncommitted-config for a development run")
    if provenance["runtime_modules_dirty"]:
        raise RuntimeError("runtime modules are dirty")
    if not provenance["runtime_modules_match_expected_commit"]:
        raise RuntimeError("runtime modules differ from the launcher commit")

    output_root = ROOT / config["output_root"]
    transform = None if args.field_transform == "none" else args.field_transform
    stem = f"{args.candidate_id}_seed_{args.seed}"
    if args.zm_mode == "off":
        stem += "_zmoff"
    if transform:
        stem += f"_ctl_{transform}"
    out_json = Path(args.out_json or output_root / "workers" / f"{stem}.json")
    out_npz = Path(args.out_npz or output_root / "workers" / f"{stem}.npz")
    ckpt_dir = Path(args.checkpoint_dir or output_root / "checkpoints")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.cache_dir or output_root / "network_cache"))

    started = time.time()
    substrate = build_substrate(config, args.candidate_id, args.seed,
                                cache_dir=cache_dir, field_transform=transform)
    engine, dt = substrate.engine, float(substrate.engine["dt"])
    simulation = config["simulation"]
    zm_cfg = config["zm"] if args.zm_mode == "z_plus_m" else {"mode": "off"}

    from kick_probe import simulate_kick
    from src.sef_hfo_events import detect_events
    from src.sef_hfo_snn_adapter import snn_event_envelope

    slow = make_slow(substrate, zm_cfg, trace_weights_E=substrate.h_e)
    drive = make_external_drive(substrate, config["spatial_ou"], args.seed)
    baseline_step = int(round(BASELINE_MS / dt))
    captured = {}
    substrate.net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        es_thresh_hz=float(simulation["es_thresh_hz"]),
        es_dur_ms=float(simulation["es_dur_ms"]),
        post_runaway_record_ms=float(simulation["post_runaway_record_ms"]),
        external_e_rate_drive=drive,
        checkpoint_steps=[baseline_step],
        checkpoint_sink=lambda step, state: captured.setdefault(step, state))
    pass1_wall = time.time() - started
    onset_ms = result["runaway_early_stop_ms"]
    spikes = np.asarray(result["E_spk_bool"], bool)

    # ---- readout on the frozen montage plus every observation-control image ----
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(spikes, dt, cmrun.BIN_MS)
    detected = detect_events(active, active_dt, event_on_frac=substrate.detector_threshold)
    # _bin_and_smooth is montage-INDEPENDENT and is the expensive half (it walks
    # a 6.4 GB bool array); sample_envelopes is the cheap per-montage half. Doing
    # the binning once instead of eight times is what keeps the observation
    # control free rather than doubling every run.
    from src.sef_hfo_observation import sample_envelopes
    from src.sef_hfo_snn_adapter import _bin_and_smooth
    binned_rate, envelope_dt = _bin_and_smooth(spikes, dt, 2.0, 5.0)
    images = _montage_images(
        substrate, config["observation_control"]["montage_transforms"])
    montage_audit = _audit_montages(substrate, images)
    envelopes = {name: np.asarray(sample_envelopes(binned_rate, substrate.positions_e,
                                                   montage, 0.25), np.float32)
                 for name, montage in images.items()}
    del binned_rate

    readout = {"participation_margin_fraction": 0.1, "timing_fraction": 0.5}
    onset_rows, rank_rows, event_rows = [], [], []
    for index, event in enumerate(detected):
        onset, rank = _contact_onsets(
            envelopes["identity"], envelope_dt, substrate.montage,
            substrate.valid_contacts, (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"], readout["timing_fraction"])
        onset_rows.append(onset); rank_rows.append(rank)
        event_rows.append({
            "event_index": int(index), "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]), "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": float(event["peak_ext"]),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.isfinite(onset).sum()),
            "before_onset": bool(onset_ms is None or event["t_off"] < onset_ms)})
    onsets = np.asarray(onset_rows, float).reshape((-1, len(substrate.contact_names)))
    ranks = np.asarray(rank_rows, float).reshape((-1, len(substrate.contact_names)))

    # ---- h-weighted Z/M trajectory (the core is 3.53 % of the E population, so
    # the unweighted population mean would mostly report background) ----
    mz_trace = (slow.trace_arrays() if slow is not None
                else {key: np.empty(0, np.float32) for key in MZSlowVars.TRACE_NAMES})
    mz_weighted = (slow.weighted_trace_arrays() if slow is not None else None)
    if mz_weighted is None:
        mz_weighted = {key: np.empty(0, np.float32)
                       for key in MZSlowVars.WEIGHTED_TRACE_NAMES}

    # ---- state characterization and local recruitment, recomputed here ----
    rate_hz = np.asarray(result["rate_E"], float)
    state_block, recruitment_block = {}, {}
    if onset_ms is not None:
        rec = config["recruitment"]
        ref_window = tuple(float(v) for v in rec["reference_window_ms"])
        reference = interictal_reference(rate_hz, dt_ms=dt, window_ms=ref_window)
        tail = (float(onset_ms), float(onset_ms) + float(simulation["post_runaway_record_ms"]))
        state_block = characterize_state(
            rate_hz, dt_ms=dt, window_ms=tail,
            silence_threshold_hz=max(1e-9, 0.05 * reference["percentile_95_hz"]))
        # NOT an interictal comparator: with tau_z = 5000 ms this window is
        # already inside the buildup, as the spec itself concedes. Named for
        # what it is. A true interictal comparator has to come from a matched
        # Z/M-off run or from a checkpoint verified against the Z/M-off support.
        matched = (float(onset_ms) - 1000.0 - float(simulation["post_runaway_record_ms"]),
                   float(onset_ms) - 1000.0)
        state_block["interictal_reference"] = reference
        state_block["pretransition_reference_window_ms"] = list(matched)
        state_block["pretransition_reference_is_not_interictal"] = True
        state_block["pretransition_reference"] = characterize_state(
            rate_hz, dt_ms=dt, window_ms=matched,
            silence_threshold_hz=max(1e-9, 0.05 * reference["percentile_95_hz"]))
        lo = int(round(tail[0] / dt)); hi = min(len(rate_hz), int(round(tail[1] / dt)))
        if hi - lo >= 8:
            state_block["band_proxy_ictal"] = band_proxy(rate_hz[lo:hi], dt_ms=dt)
        mlo = int(round(matched[0] / dt)); mhi = int(round(matched[1] / dt))
        if mhi - mlo >= 8:
            state_block["band_proxy_pretransition"] = band_proxy(rate_hz[mlo:mhi], dt_ms=dt)

        bins = spatial_bins(substrate.positions_e, bin_mm=float(rec["bin_mm"]),
                            sheet_l_mm=float(engine["L"]))
        traces = bin_rate_traces(spikes, bins["bin_index"], bins["n_bins"],
                                 dt_ms=dt, kernel_ms=float(rec["rate_kernel_ms"]))
        thresholds = bin_baseline(traces, dt_ms=dt, window_ms=ref_window,
                                  quantile=float(rec["bin_baseline_quantile"]))
        lo_ms, hi_ms = rec["search_window_relative_to_onset_ms"]
        start = max(0, int(round((float(onset_ms) + float(lo_ms)) / dt)))
        span = int(round((float(hi_ms) - float(lo_ms)) / dt))
        recruitment = local_recruitment(
            traces, thresholds, dt_ms=dt, search_window_steps=span,
            minimum_persistence_ms=float(rec["minimum_persistence_ms"]),
            search_start_step=start)
        lag = axial_lag(recruitment["recruitment_step"], bins["bin_xy_mm"], dt_ms=dt,
                        axis_unit=substrate.axis_unit,
                        origin_xy=substrate.axis_source_xy)
        recruitment_block = {**{k: v for k, v in recruitment.items()
                                if k != "recruitment_step"}, **lag,
                             "reference_window_ms": list(ref_window),
                             "search_window_ms": [float(lo_ms), float(hi_ms)]}
        recruitment_block["_step"] = recruitment["recruitment_step"]
        recruitment_block["_bin_xy"] = bins["bin_xy_mm"]

    # ---- pass 2: onset-relative checkpoints, Joint arm only ----
    pass2 = {"ran": False}
    if args.emit_onset_checkpoints and onset_ms is not None:
        limits = config["checkpoints"]
        if float(onset_ms) >= float(limits["minimum_onset_for_perturbation_ms"]):
            wanted = [int(round((float(onset_ms) - float(limits["pre_ictal_offset_ms"])) / dt))]
            if float(onset_ms) >= float(limits["minimum_onset_for_sensitivity_ms"]):
                wanted.insert(0, int(round(
                    (float(onset_ms) - float(limits["sensitivity_offset_ms"])) / dt)))
            # +dt: the loop runs [0, nsteps), so a segment of exactly
            # (stop_ms - BASELINE_MS) never reaches the step AT stop_ms -- which
            # is precisely the pre-ictal checkpoint. Every pre-ictal checkpoint
            # in the first canary batch was lost to this one missing step.
            stop_ms = float(onset_ms) - float(limits["pre_ictal_offset_ms"]) + dt
            from params import Params
            tail_params = Params(g=engine["g"], L=engine["L"], density=engine["density"],
                                 T=stop_ms - BASELINE_MS, dt=dt,
                                 nu_ext_ratio=substrate.params.nu_ext_ratio,
                                 seed=int(args.seed))
            slow2 = make_slow(substrate, zm_cfg, trace_weights_E=substrate.h_e)
            drive2 = make_external_drive(substrate, config["spatial_ou"], args.seed)
            captured2 = {}
            tail = simulate_kick(
                tail_params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
                V_th_per_neuron=substrate.vtheta, slow=slow2,
                early_stop_runaway=False, external_e_rate_drive=drive2,
                resume_state=captured[baseline_step], time_offset_ms=BASELINE_MS,
                checkpoint_steps=wanted,
                checkpoint_sink=lambda step, state: captured2.setdefault(step, state))
            overlap = len(tail["rate_E"])
            identical = bool(np.array_equal(
                np.asarray(tail["rate_E"]),
                np.asarray(result["rate_E"])[baseline_step:baseline_step + overlap]))
            if not identical:
                raise RuntimeError(
                    "pass 2 diverged from pass 1 over the overlap -- the checkpoint "
                    "is incomplete; stop the round rather than trusting it")
            missing = sorted(set(int(s) for s in wanted) - set(captured2))
            if missing:
                raise RuntimeError(
                    f"requested checkpoints were never reached: {missing}; "
                    f"segment covered absolute steps "
                    f"[{baseline_step}, {baseline_step + overlap - 1}]")
            captured.update(captured2)
            pass2 = {"ran": True, "overlap_steps": int(overlap),
                     "overlap_bit_identical": True,
                     "checkpoint_steps": sorted(int(s) for s in wanted),
                     "all_requested_checkpoints_captured": True}

    # Verified across all three canary seeds: the median 20 ms-EMA rate over
    # [1500, 2000] ms (37 / 64 / 50 Hz) exceeds the q95 of forty non-overlapping
    # 500 ms windows from the SAME-SEED Z/M-off run (30 / 30 / 31 Hz). The 2 s
    # point is already elevated on the Joint arm, so it is named for what it is.
    labels = {baseline_step: "early_transition"}
    if pass2["ran"]:
        for step in pass2["checkpoint_steps"]:
            offset_ms = float(onset_ms) - step * dt
            labels[step] = ("pre_ictal" if abs(offset_ms - 500.0) < 1e-6
                            else "sensitivity")
    checkpoint_records = {}
    for step, state in captured.items():
        label = labels.get(step)
        if label is None:
            continue
        path = ckpt_dir / f"{stem}_{label}.npz"
        checkpoint_records[label] = {
            "path": str(path.relative_to(ROOT)), "step": int(step),
            "absolute_time_ms": float(state["absolute_time_ms"]),
            "sha256": ckpt.save(state, path)}

    payload = {
        "status": "ZM_ITX_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "arm": arm_name, "candidate_id": args.candidate_id, "seed": int(args.seed),
        "zm_mode": args.zm_mode, "field_transform": args.field_transform,
        "run": {
            "model_ictal_onset_ms": onset_ms,
            "onset_label_is_operational": True,
            "post_runaway_recorded_ms": result["post_runaway_recorded_ms"],
            "n_detected_events": len(event_rows),
            "n_returned_events": int(sum(r["returned"] for r in event_rows)),
            "n_returned_events_before_onset": int(sum(
                r["returned"] and r["before_onset"] for r in event_rows)),
            "peak_active_fraction": float(np.max(active)) if len(active) else None,
            "fraction_time_above_common_detector": float(
                np.mean(active > substrate.detector_threshold)) if len(active) else None,
        },
        "pass2": pass2,
        "checkpoints": checkpoint_records,
        "state_characterization": state_block,
        "recruitment": {k: v for k, v in recruitment_block.items()
                        if not k.startswith("_")},
        "observation_control_montage_audit": montage_audit,
        "network": {"n_E": substrate.n_e, "n_I": substrate.n_i,
                    **substrate.network_cache},
        "simulation": {**simulation, "wall_seconds_pass1": pass1_wall,
                       "wall_seconds_total": time.time() - started},
        "events": event_rows,
        "provenance": provenance,
    }
    arrays = dict(
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        shaft_ids=substrate.shaft_ids, contact_xy_mm=substrate.contact_xy,
        onsets=onsets.astype(np.float32), ranks=ranks.astype(np.float32),
        event_t_on_ms=np.asarray([r["t_on_ms"] for r in event_rows], np.float32),
        event_t_off_ms=np.asarray([r["t_off_ms"] for r in event_rows], np.float32),
        event_returned=np.asarray([r["returned"] for r in event_rows], bool),
        event_before_onset=np.asarray([r["before_onset"] for r in event_rows], bool),
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        rate_E_hz=np.asarray(result["rate_E"], np.float32),
        rate_I_hz=np.asarray(result["rate_I"], np.float32),
        positions_E=substrate.positions_e.astype(np.float32),
        h=substrate.h_e.astype(np.float32),
        h_I_for_edge=substrate.h_i.astype(np.float32),
        delta_vtheta=substrate.delta_vtheta.astype(np.float32),
        edge_coefficients=substrate.edge_coefficients.astype(np.float64),
        ee_out_gain=substrate.ee_out_gain.astype(np.float32),
        etoi_out_gain=substrate.etoi_out_gain.astype(np.float32),
    )
    for name, env in envelopes.items():
        arrays[f"contact_envelope_ctl_{name}" if name != "identity"
               else "contact_envelope"] = env
    for key in MZSlowVars.TRACE_NAMES:
        arrays[f"mz_{key}"] = np.asarray(mz_trace[key], np.float32)
    for key in MZSlowVars.WEIGHTED_TRACE_NAMES:
        arrays[f"mz_h_weighted_{key}"] = np.asarray(mz_weighted[key], np.float32)
    if recruitment_block:
        arrays["recruitment_step"] = np.asarray(recruitment_block["_step"], np.float32)
        arrays["recruitment_bin_xy_mm"] = np.asarray(recruitment_block["_bin_xy"], np.float32)
    _atomic_npz(out_npz, **arrays)          # arrays first: a json failure must
    atomic_write_json(_json_safe(payload), str(out_json))   # never cost the run
    print(json.dumps({"stem": stem, "onset_ms": onset_ms,
                      "n_returned": payload["run"]["n_returned_events"],
                      "checkpoints": sorted(checkpoint_records),
                      "wall_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
