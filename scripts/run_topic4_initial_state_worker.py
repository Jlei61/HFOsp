#!/usr/bin/env python3
"""One initial-state arm on one frozen graph and one dynamics seed.

Physical model = the mainline multidimensional worker's frozen candidate; the
only interventions are the explicit t=0 membrane voltage (B0/B1/B2) and the
read-only per-step observer. Output = raw acquisition from 0 s (native field,
contact envelope, state trace, external-input digests) + the frozen repaired
observer + the frozen classifier.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src import topic4_initial_state_runtime as rt  # noqa: E402
from src.topic4_initial_state import (  # noqa: E402
    ARMS, RunObserver, array_sha256, build_initial_voltage, core_membership,
    event_times_ms, initial_voltage_summary, validate_initial_voltage,
    window_mode_counts,
)
from src.topic4_node_dualmode import sheet_activity_movie  # noqa: E402
from src.topic4_observation_repaired import observe  # noqa: E402
from src.topic4_zm_ictal_transition import make_external_drive  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402

SCIENTIFIC_ROLE = "development_only_initial_state_conditioned_propagation_v1"


def load_frozen(design, frozen_path):
    frozen = rt.read(frozen_path)
    if frozen["design_sha256"] != rt.sha(rt.DESIGN_PATH):
        raise RuntimeError("frozen manifest was written for a different design file")
    rt.verify_source_snapshot(frozen)
    arrays = np.load(frozen["membership"]["membership_file"])
    if rt.sha(frozen["membership"]["membership_file"]) != frozen["membership"]["membership_file_sha256"]:
        raise RuntimeError("frozen membership arrays changed")
    membership = {"S_A": np.asarray(arrays["S_A"], np.int64), "S_B": np.asarray(arrays["S_B"], np.int64),
                  "vtheta": np.asarray(arrays["vtheta"], np.float64), "h": np.asarray(arrays["h"], np.float64),
                  "positions_E": np.asarray(arrays["positions_E"], np.float64)}
    return frozen, membership


def run_unit(*, design, frozen, membership, substrate, candidate, transition, parameter_audit,
             network, evaluator, objective, contract, stage, topology_seed, dynamics_seed, arm,
             duration_ms, out_json, out_npz, qualification_tag=None, provenance=None):
    """Simulate + observe one unit on an already-built substrate. Returns payload."""
    started = time.time()
    n_e, n_i = substrate.n_e, substrate.n_i
    n_total = n_e + n_i
    init = design["initialization"]
    v_reset = float(substrate.params.V_reset)
    if v_reset != float(init["default_V_mV"]):
        raise RuntimeError("engine V_reset differs from the design")

    # ---- M6: membership recomputed from the runtime full-precision arrays ----
    recomputed = core_membership(substrate.positions_e, substrate.h_e, candidate["node_field"]["centers_mm"])
    if (not np.array_equal(recomputed["S_A"], membership["S_A"])
            or not np.array_equal(recomputed["S_B"], membership["S_B"])):
        raise RuntimeError("runtime membership differs from the frozen membership")
    if not np.array_equal(np.asarray(substrate.vtheta, np.float64), membership["vtheta"]):
        raise RuntimeError("runtime thresholds differ from the frozen thresholds")
    if not np.array_equal(np.asarray(substrate.h_e, np.float64), membership["h"]):
        raise RuntimeError("runtime h field differs from the frozen field")

    # ---- V1-V3: arm voltage rebuilt and checked against the frozen file ----
    voltage = build_initial_voltage(n_total, v_reset, membership, float(init["increment_mV"]), arm)
    frozen_arm = frozen["arms"][arm]
    frozen_voltage = np.load(frozen_arm["file"])
    if rt.sha(frozen_arm["file"]) != frozen_arm["file_sha256"] or not np.array_equal(voltage, frozen_voltage):
        raise RuntimeError("rebuilt initial voltage differs from the frozen array")
    margin_audit = validate_initial_voltage(voltage, substrate.vtheta, v_reset,
                                            float(init["required_threshold_margin_mV"]))
    voltage_summary = initial_voltage_summary(voltage, v_reset)
    voltage_for_engine = voltage.copy()

    # ---- observer groups (X10) ----
    in_core = np.zeros(n_e, bool)
    in_core[membership["S_A"]] = True
    in_core[membership["S_B"]] = True
    groups = {"coreA": membership["S_A"], "coreB": membership["S_B"],
              "surroundE": np.flatnonzero(~in_core), "I": np.arange(n_e, n_total)}
    dt = float(substrate.engine["dt"])
    substrate.params.T = float(duration_ms)
    n_steps = int(round(float(duration_ms) / dt))
    observer = RunObserver(dt_ms=dt, n_e=n_e, n_total=n_total, groups=groups, n_steps=n_steps,
                           segment_ms=1000.0, trace_ms=1.0)

    # ---- simulation: identical RNG seeding to the mainline worker ----
    substrate.net["rng"] = np.random.default_rng(int(dynamics_seed))
    drive = make_external_drive(substrate, transition["spatial_ou"], int(dynamics_seed))
    simulation = design["simulation"]
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        external_e_rate_drive=drive,
        initial_voltage=voltage_for_engine, step_observer=observer,
    )
    if not np.array_equal(voltage_for_engine, voltage):
        raise RuntimeError("engine modified the caller's initial voltage array")
    if not np.array_equal(result["initial_V"], voltage):
        raise RuntimeError("engine did not apply the requested initial voltage")
    observed = observer.finish()
    spikes = np.asarray(result["E_spk_bool"], bool)
    actual_duration_ms = float(spikes.shape[0] * dt)
    sim_wall = time.time() - started

    # ---- raw acquisition from 0 s (O1) ----
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(spikes, dt, cmrun.BIN_MS)
    movie = sheet_activity_movie(spikes, substrate.positions_e, dt_ms=dt, frame_ms=2.0,
                                 bin_mm=1.0, sheet_mm=float(substrate.engine["L"]))
    envelope, envelope_dt, _ = snn_event_envelope(spikes, substrate.positions_e, substrate.montage, dt)
    del spikes, result["E_spk_bool"]
    rate_e = np.asarray(result["rate_E"], np.float32)

    # ---- frozen observer (burn-in 500 ms inside the contract) + frozen classifier ----
    observation = observe(envelope, float(envelope_dt), contract)
    centroid = np.asarray(observation["centroid_ms"], np.float64).reshape(-1, len(contract["contact_names"]))
    labels, state, distance_modes = rt.classify_with_both_modes(evaluator, centroid) if len(centroid) else (
        np.zeros(0, int), np.zeros(0, int), np.zeros((0, evaluator.k)))
    readable = np.isfinite(centroid).sum(1) >= 2 if len(centroid) else np.zeros(0, bool)
    phi = np.full((len(centroid), objective.target_global.shape[0]), np.nan, np.float32)
    if readable.any():
        phi[readable] = objective.embedding(centroid[readable]).astype(np.float32)
    times, primary = event_times_ms(observation)
    windows = {"primary": design["statistics"]["primary_window_ms"]}
    for index, window in enumerate(design["statistics"]["secondary_windows_ms"]):
        windows[f"secondary_{index}"] = window
    window_counts = {name: window_mode_counts(times, labels, primary, window, evaluator.k)
                     for name, window in windows.items()}
    event_rows = []
    for index, event in enumerate(observation["events"]):
        row = {
            "detected_index": int(index),
            "window_ms": [float(v) for v in event["window_ms"]],
            "qualifying_interval_ms": [float(v) for v in event["qualifying_interval_ms"]],
            "event_time_ms": float(times[index]),
            "primary_eligible": bool(event["primary_eligible"]),
            "primary_exclusion_reasons": list(event["primary_exclusion_reasons"]),
            "prolonged": bool(event["prolonged"]),
            "n_unique_contacts": int(event["n_unique_contacts"]),
            "n_finite_centroids": int(np.isfinite(centroid[index]).sum()),
            "multiple_local_bursts_contact_fraction": float(event["multiple_local_bursts_contact_fraction"]),
            "median_mass_outside_detection": float(event["median_mass_outside_detection"]),
            "mode": int(labels[index]), "support_state": int(state[index]),
            "distance_mode0": float(distance_modes[index, 0]),
            "distance_mode1": float(distance_modes[index, 1]),
            "in_primary_window": bool(windows["primary"][0] <= times[index] < windows["primary"][1]),
        }
        event_rows.append(row)
    trace = observed["trace"]
    # 20 ms-EMA runaway flag already applied by the engine; keep the raw verdict.
    physical_status = ("RUNAWAY_EARLY_STOP" if result.get("runaway_early_stop_ms") is not None
                       else "COMPLETE_NO_RUNAWAY_BY_EXISTING_GATE")
    identity = rt.static_identity(substrate, parameter_audit)
    payload = {
        "status": "INITIAL_STATE_WORKER_COMPLETE",
        "execution_status": "COMPLETE",
        "physical_status": physical_status,
        "scientific_role": SCIENTIFIC_ROLE,
        "qualification_tag": qualification_tag,
        "stage": stage, "candidate_id": candidate["candidate_id"],
        "topology_seed": int(topology_seed), "dynamics_seed": int(dynamics_seed),
        "arm": arm, "initial_state_id": frozen_arm["initial_state_id"],
        "initial_voltage": {**voltage_summary, "threshold_margin": margin_audit,
                            "perturbed_core_index": frozen_arm["perturbed_core_index"],
                            "engine_applied_sha256": array_sha256(result["initial_V"])},
        "membership": {"K": int(len(membership["S_A"])),
                       "S_A_sha256": array_sha256(membership["S_A"]),
                       "S_B_sha256": array_sha256(membership["S_B"]),
                       "runtime_recomputed_equal": True},
        "static_array_identity": identity,
        "network_cache_source": network,
        "candidate_canonical_json_sha256": design["candidate_canonical_json_sha256"],
        "multidimensional_parameter_audit": parameter_audit,
        "seed_contract": substrate.extras["seed_contract"],
        "simulation": {
            "duration_ms": float(duration_ms), "actual_duration_ms": actual_duration_ms,
            "n_steps": int(n_steps), "dt_ms": dt,
            "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
            "post_runaway_recorded_ms": result.get("post_runaway_recorded_ms"),
            "simulation_wall_seconds": float(sim_wall),
            "wall_seconds": None,
            "slow_Z_M": "off", "kick": False, "forced_spikes": False,
            "external_drive": "global OU + spatial OU + Poisson (frozen mainline law)",
        },
        "external_input_digest": {
            "segments": observed["segments"],
            "segment_ms": observed["segment_ms"], "trace_ms": observed["trace_ms"],
            "n_observed_steps": observed["n_observed_steps"],
            "maximum_poisson_count": observed["maximum_poisson_count"],
            "definition": observed["digest_definition"],
            "independence_note": ("global OU and Poisson counts share the master "
                                  "net['rng'] stream; the spatial OU field is seeded by "
                                  "dynamics_seed + offset; equality across arms is an "
                                  "innovation-replay statement, not an independence claim"),
        },
        "spatial_ou_trace": {k: v for k, v in drive.trace_arrays().items() if k == "time_ms"} | {
            "n_updates": int(len(drive.trace_arrays()["time_ms"]))},
        "observation": {
            "contract_sha256": design["sources"]["observation_contract"]["sha256"],
            "n_detected_windows": int(observation["n_groups"]),
            "n_primary_events": int(observation["n_primary_events"]),
            "primary_event_indices": [int(i) for i in observation["primary_event_indices"]],
            "boundary_or_low_window_support": observation["boundary_or_low_window_support"],
            "required_unique_contacts": int(observation["required_unique_contacts"]),
            "burnin_ms": float(contract["burnin_ms"]),
            "n_unreadable_detected": int(np.sum(labels < 0)),
            "n_primary_unclassifiable": int(sum(1 for i in primary if labels[i] < 0)),
        },
        "window_counts": window_counts,
        "events": event_rows,
        "evaluator_sha256": design["sources"]["evaluator"]["sha256"],
        "objective_sha256": design["sources"]["objective"]["sha256"],
        "state_trace_fields": sorted(trace.keys()),
        "arrays": {"path": str(out_npz), "sha256": None},
        "provenance": provenance or {},
    }
    rt.atomic_npz(
        out_npz,
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float64),
        shaft_ids=np.asarray(substrate.shaft_ids, dtype="U8"),
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        rate_E_hz_per_step=rate_e,
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        sheet_activity_counts=movie["activity_counts"],
        sheet_activity_frame_ms=np.asarray(movie["frame_ms"], float),
        sheet_bin_mm=np.asarray(movie["bin_mm"], float),
        centroid_ms=np.asarray(centroid, np.float32),
        recruitment_ms=np.asarray(observation["recruitment_ms"], np.float32),
        primary_event_indices=np.asarray(observation["primary_event_indices"], np.int32),
        windows_ms=np.asarray(observation["windows_ms"], np.float32).reshape(-1, 2),
        event_time_ms=np.asarray(times, np.float32),
        event_mode=np.asarray(labels, np.int8),
        event_support_state=np.asarray(state, np.int8),
        event_distance_modes=np.asarray(distance_modes, np.float32),
        event_phi=phi,
        initial_V=np.asarray(result["initial_V"], np.float64),
        S_A=np.asarray(membership["S_A"], np.int32), S_B=np.asarray(membership["S_B"], np.int32),
        ext_segment_sums=observed["ext_segment_sums"],
        delta_segment_sums=observed["delta_segment_sums"],
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h=np.asarray(substrate.h_e, np.float32),
        vtheta=np.asarray(substrate.vtheta, np.float32),
        topology_seed=np.asarray(int(topology_seed), np.int64),
        dynamics_seed=np.asarray(int(dynamics_seed), np.int64),
        **{f"trace_{key}": value for key, value in trace.items()},
        **{f"spatial_ou_{key}": value for key, value in drive.trace_arrays().items()},
    )
    payload["arrays"]["sha256"] = rt.sha(out_npz)
    payload["simulation"]["wall_seconds"] = float(time.time() - started)
    payload["peak_rss_gib"] = rt.peak_rss_gib()
    rt.write(out_json, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "replication"), required=True)
    parser.add_argument("--dynamics-seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float, default=None,
                        help="qualification only; formal units use the design duration")
    parser.add_argument("--qualification-tag", default=None)
    args = parser.parse_args()

    design = rt.load_design(args.design)
    frozen, membership = load_frozen(design, args.frozen_manifest)
    stage = design["stages"][args.stage]
    if args.dynamics_seed not in stage["dynamics_seeds"] and args.qualification_tag is None:
        parser.error("dynamics seed is outside the frozen stage")
    duration = float(design["simulation"]["duration_ms"])
    if args.duration_ms is not None:
        if args.qualification_tag is None:
            parser.error("--duration-ms requires --qualification-tag")
        duration = float(args.duration_ms)
    topology_seed = int(stage["topology_seed"])
    substrate, candidate, transition, execution, parameter_audit, network = rt.build_frozen_substrate(
        design, topology_seed, args.dynamics_seed, frozen_manifest=frozen)
    evaluator = rt.load_evaluator(design)
    objective = rt.load_objective(design)
    contract = rt.load_observation_contract(design)
    provenance = {
        "git_commit": rt.git_commit(), "git_dirty_paths": rt.git_dirty_paths(),
        "frozen_manifest_sha256": rt.sha(args.frozen_manifest),
        "source_hashes_verified": True,
        "loaded_source_sha256": rt.loaded_source_hashes(),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "hostname": platform.node(),
        "systemd_unit": os.environ.get("ISCP_SYSTEMD_UNIT"),
    }
    run_unit(design=design, frozen=frozen, membership=membership, substrate=substrate,
             candidate=candidate, transition=transition, parameter_audit=parameter_audit,
             network=network, evaluator=evaluator, objective=objective, contract=contract,
             stage=args.stage, topology_seed=topology_seed, dynamics_seed=args.dynamics_seed,
             arm=args.arm, duration_ms=duration, out_json=args.out_json, out_npz=args.out_npz,
             qualification_tag=args.qualification_tag, provenance=provenance)
    print({"status": "INITIAL_STATE_WORKER_COMPLETE", "stage": args.stage,
           "dynamics_seed": args.dynamics_seed, "arm": args.arm, "output": str(args.out_json)})


if __name__ == "__main__":
    main()
