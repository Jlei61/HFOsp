#!/usr/bin/env python3
"""One fixed FCXR-LC6A natural trajectory on a frozen E->I graph."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5v2_natural_prefix as PREFIX  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle, ExactInputHasher, SparseSpikeBinaryWriter, SparseSpikeStream,
)
from src.topic4_fcxr_lc6_surround import (  # noqa: E402
    EToIGraph, extract_e_to_i, graph_sha256, replace_e_to_i_in_net,
)
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    NaturalCurrentObserver, cell_spatial_bins, coarse_field_mean, linear_slope,
    local_saturation_readout, observation_decision, per_second_cell_rates,
    spatial_map_persistence, spatial_rate_maps,
)


U2 = PREFIX.U2
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"
GRAPH_IDS = ("C0", "C1", "Q1", "Q2", "Q3")
TRACE_ATTRS = {
    "D_mean": "trace_z_mean",
    "H_mean": "trace_h_lc2_mean",
    "H_source_mean": "trace_gA_raw_lc2_mean",
    "gErec_mean": "trace_gErec_mean",
    "clip_frac": "trace_conductance_clip_frac",
}
MECHANISM_FILES = (
    Path(__file__).resolve(),
    Path(PREFIX.__file__).resolve(),
    Path(U2.__file__).resolve(),
    ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_statefork.py",
    ROOT / "src/topic4_fcxr_lc6_surround.py",
    ROOT / "src/topic4_fcxr_lc6_trajectory.py",
    ROOT / "src/snn_engine/mz_slow_vars.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in MECHANISM_FILES}


def _rotate_recovery_checkpoint(work, state):
    """Keep two recoverable exact states without publishing a partial result bundle."""

    work = Path(work)
    new = work / "rolling_checkpoint.new.npz"
    current = work / "rolling_checkpoint.current.npz"
    previous = work / "rolling_checkpoint.previous.npz"
    U2.save_loop_state(str(new), state)
    if current.exists():
        os.replace(current, previous)
    os.replace(new, current)


def _pin_checkpoint(work, state, *, name, onset_ms, target_ms, actual_ms):
    path = Path(work) / f"checkpoint_{name}.npz"
    if path.exists():
        return None
    U2.save_loop_state(str(path), state)
    return {
        "name": name,
        "file": path.name,
        "onset_ms": float(onset_ms),
        "target_ms": float(target_ms),
        "actual_ms": float(actual_ms),
        "timing_error_ms": float(actual_ms) - float(target_ms),
        "state_hash": state_hash(state),
    }


def _load_graph(path):
    with np.load(path, allow_pickle=False) as z:
        graph = EToIGraph(
            np.asarray(z["sources"], np.int32), np.asarray(z["weights"]),
            np.asarray(z["delay_steps"], np.int32),
        )
        expected = str(z["graph_sha256"][0])
        metadata = json.loads(str(z["metadata_json"][0]))
    if graph_sha256(graph) != expected or metadata.get("graph_sha256") != expected:
        raise RuntimeError("graph artifact hash mismatch")
    return graph, metadata


def _validate_manifest(path, condition):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_patient_axis_surround":
        raise RuntimeError("wrong LC6A execution manifest")
    if condition not in GRAPH_IDS:
        raise ValueError(f"condition must be one of {GRAPH_IDS}")
    for relative, expected in payload["blessed_engine_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"blessed engine hash mismatch: {relative}")
    source = ROOT / payload["lc5_continuation"]["source_summary"]
    if _sha(source) != payload["lc5_continuation"]["source_summary_sha256"]:
        raise RuntimeError("locked source config summary hash mismatch")
    return path, payload, source


def _load_c0_ied_reference(condition):
    if condition == "C0":
        return None
    summary = OUT / "trajectories/C0/summary.json"
    if not summary.is_file():
        raise RuntimeError("C0 natural trajectory must finish before non-C0 arms")
    return int(json.loads(summary.read_text())["n_returning_pre_onset"])


def _fresh_config(summary, ne):
    cfg = dict(summary["config_scalar"])
    cfg.update(
        use_pump=False,
        pump_sensor_only=False,
        pump_Imax=0.0,
        pump_p0_E=np.zeros(int(ne), dtype=float),
        pump_u_init_E=np.zeros(int(ne), dtype=float),
        use_m=False,
        use_x=True,
        x_relay_frozen_E=np.ones(int(ne), dtype=float),
    )
    return cfg


def _fresh_system(summary, graph, graph_expected_hash, condition, *, force_replacement=False):
    S = U2.PP.build_substrate(U2.CONNECTION_SEED)
    base = extract_e_to_i(S["net"], S["NE"], S["NI"])
    if condition == "C0" and not force_replacement:
        if graph_sha256(base) != graph_expected_hash:
            raise RuntimeError("C0 graph artifact is not exact substrate parity")
    else:
        S["net"] = replace_e_to_i_in_net(
            S["net"], graph, ne=S["NE"], ni=S["NI"],
        )
        rebuilt = extract_e_to_i(S["net"], S["NE"], S["NI"])
        if graph_sha256(rebuilt) != graph_expected_hash:
            raise RuntimeError("runtime E-to-I graph differs from frozen artifact")
    U2.install_registered_noise_rng(S["net"])
    cfg = _fresh_config(summary, S["NE"])
    slow = MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
        core_mask_E=U2.OLD_SLOW.build_core_masks(S),
    )
    S["net"]["rng"] = np.random.default_rng(U2.NOISE_SEED)
    return S, slow, cfg


def _combine_streams(streams):
    chunk_steps = int(round(U2.CHUNK_MS / U2.DT_MS))
    return SparseSpikeStream(
        np.concatenate([stream.steps + index * chunk_steps for index, stream in enumerate(streams)]),
        np.concatenate([stream.cells for stream in streams]),
        len(streams) * chunk_steps,
        streams[0].n_cells,
    )


def _rate_from_stream(stream):
    counts = np.bincount(stream.steps, minlength=stream.n_steps).astype(float)
    return counts / stream.n_cells / U2.DT_MS * 1000.0


def _prefix_stream(stream, end_ms):
    n_steps = int(round(float(end_ms) / U2.DT_MS))
    keep = stream.steps < n_steps
    return SparseSpikeStream(
        stream.steps[keep], stream.cells[keep], n_steps, stream.n_cells,
    )


def _c0_control_parity(condition, stream, rate):
    if condition != "C0":
        return {"required": False}
    reference_ms = min(float(stream.n_steps * U2.DT_MS), float(PREFIX.RUN_MS))
    reference_stream, reference_rate = PREFIX._reference_prefix(reference_ms)
    observed_stream = _prefix_stream(stream, reference_ms)
    stride = int(round(U2.TRACE_DT_MS / U2.DT_MS))
    observed_rate = np.asarray(rate[:observed_stream.n_steps:stride], np.float32)
    max_abs = float(np.max(np.abs(observed_rate - reference_rate)))
    record = {
        "required": True,
        "reference_ms": reference_ms,
        "spike_sha256_expected": reference_stream.sha256,
        "spike_sha256_observed": observed_stream.sha256,
        "spike_exact": bool(observed_stream.sha256 == reference_stream.sha256),
        "rate_max_abs_diff_hz": max_abs,
    }
    if not (record["spike_exact"] and max_abs == 0.0):
        raise RuntimeError("C0 natural trajectory does not reproduce the locked pump-off prefix")
    return record


def _trace_chunk(slow, starts, stride):
    result = {}
    for output, attribute in TRACE_ATTRS.items():
        values = np.asarray(getattr(slow, attribute)[starts[attribute]:], float)
        result[output] = values[::stride].astype(np.float32)
    result["D_mean"] = 1.0 - result["D_mean"]
    return result


def _event_count_before(events, onset_ms):
    if onset_ms is None:
        return len([event for event in events if event["returned"]])
    return len([
        event for event in events
        if event["returned"] and float(event["t_on"]) < float(onset_ms)
    ])


def _validate_confirmation(lock_path, *, parent_condition, output_condition, graph_path):
    if lock_path is None:
        raise RuntimeError("noncanonical graph run requires a confirmation lock")
    lock_path = Path(lock_path).resolve()
    payload = json.loads(lock_path.read_text())
    if payload.get("status") != "LOCKED" or payload.get("authorized") is not True:
        raise RuntimeError("LC6A graph-realization confirmation is not authorized")
    if payload.get("parent_condition") != parent_condition:
        raise RuntimeError("confirmation parent condition mismatch")
    if payload.get("output_condition") != output_condition:
        raise RuntimeError("confirmation output condition mismatch")
    if Path(payload["graph_artifact"]).resolve() != Path(graph_path).resolve():
        raise RuntimeError("confirmation graph path mismatch")
    if _sha(graph_path) != payload["graph_artifact_sha256"]:
        raise RuntimeError("confirmation graph artifact drift")
    return lock_path, payload


def run(
    condition, manifest_path, *, graph_path_override=None,
    output_condition=None, confirmation_lock=None,
):
    manifest_path, manifest, source_summary = _validate_manifest(manifest_path, condition)
    manifest_hash = _sha(manifest_path)
    source_hashes = _source_hashes()
    output_condition = condition if output_condition is None else str(output_condition)
    graph_path = (
        OUT / f"graphs/{condition}.npz"
        if graph_path_override is None else Path(graph_path_override).resolve()
    )
    confirmation = None
    if graph_path_override is not None:
        _lock_path, confirmation = _validate_confirmation(
            confirmation_lock, parent_condition=condition,
            output_condition=output_condition, graph_path=graph_path,
        )
    graph, graph_metadata = _load_graph(graph_path)
    if (
        graph_metadata.get("graph_legality", "PASS") != "PASS"
        and (condition != "C0" or graph_path_override is not None)
    ):
        raise RuntimeError(f"GRAPH_LEGALITY_FAILED_{condition}")
    summary_cfg = json.loads(source_summary.read_text())
    c0_ied_to_onset = _load_c0_ied_reference(output_condition)
    observation = manifest["observation"]
    resources = U2.GEO._meminfo()
    if resources["mem_available_gib"] < 96.0:
        raise RuntimeError("LC6A natural trajectory requires at least 96 GiB MemAvailable")
    baseline_swap = float(resources["swap_used_mib"])
    arm = OUT / f"trajectories/{output_condition}"
    work = OUT / f"trajectories/.{output_condition}.work"
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    if work.exists():
        raise RuntimeError(f"stale work directory requires inspection: {work}")
    work.mkdir(parents=True)
    started = time.time()
    S, slow, cfg = _fresh_system(
        summary_cfg, graph, graph_sha256(graph), condition,
        force_replacement=graph_path_override is not None,
    )
    state = U2.PM._seed_template(S, slow)
    initial_state_hash = state_hash(state)
    stride = int(round(U2.TRACE_DT_MS / U2.DT_MS))
    chunk_steps = int(round(U2.CHUNK_MS / U2.DT_MS))
    p = dataclasses.replace(S["p"], T=float(observation["hard_cap_ms"]), dt=U2.DT_MS)
    input_hasher = ExactInputHasher()
    current_observer = NaturalCurrentObserver(
        dt_ms=U2.DT_MS, sample_dt_ms=U2.TRACE_DT_MS,
    )
    streams, trace_parts = [], {}
    input_hashes_by_chunk = []
    pinned_checkpoints = {}
    spatial_bins, spatial_occupancy = cell_spatial_bins(
        S["posE"], sheet_size_mm=S["L"], n_bins_axis=32,
    )
    d_maps, h_maps = [], []
    adjudication = None
    stop = None
    max_chunks = int(round(float(observation["hard_cap_ms"]) / U2.CHUNK_MS))
    for chunk in range(max_chunks):
        if _sha(manifest_path) != manifest_hash:
            raise RuntimeError("execution manifest drifted during LC6A natural trajectory")
        if _source_hashes() != source_hashes:
            raise RuntimeError("mechanism source drifted during LC6A natural trajectory")
        starts = {attribute: len(getattr(state.slow, attribute)) for attribute in TRACE_ATTRS.values()}
        binary = work / f"chunk_{chunk:02d}.bin"
        writer = SparseSpikeBinaryWriter(
            binary, step_origin=state.t, n_steps=chunk_steps, n_cells=S["NE"],
        )
        out = run_fcxr_loop(
            p, S["net"], start=state, n_steps=chunk_steps, capture_final=True,
            store_spikes=False, spike_sink=writer, input_sink=input_hasher,
            membrane_term_sink=current_observer.sample,
            v_th_per_neuron=S["vth"],
        )
        state = out["checkpoint"]
        stream = writer.finalize(work / f"chunk_{chunk:02d}_spikes.npz")
        binary.unlink(missing_ok=True)
        streams.append(stream)
        trace = _trace_chunk(state.slow, starts, stride)
        for key, value in trace.items():
            trace_parts.setdefault(key, []).append(value)
        d_maps.append(coarse_field_mean(1.0 - state.slow.z[:S["NE"]], spatial_bins, spatial_occupancy))
        h_maps.append(coarse_field_mean(state.slow.h_lc2_E, spatial_bins, spatial_occupancy))
        full = _combine_streams(streams)
        rate = _rate_from_stream(full)
        adjudication = PREFIX._adjudicate(full, rate)
        onset_ms = adjudication.get("onset_ms")
        n_pre = _event_count_before(adjudication["events"], onset_ms)
        chunk_rate = stream.steps.size / stream.n_cells / (U2.CHUNK_MS / 1000.0)
        stop = observation_decision(
            total_ms=full.n_steps * U2.DT_MS,
            onset_ms=onset_ms,
            n_returning_ied=n_pre,
            c0_ied_to_onset=c0_ied_to_onset,
            saturated_contiguous_1s=chunk_rate >= float(U2.SAT_CEILING_HZ),
            base_end_ms=observation["base_end_ms"],
            post_onset_ms=observation["post_onset_ms"],
            hard_cap_ms=observation["hard_cap_ms"],
            ied_multiplier=observation["entry_blocked_ied_multiplier"],
        )
        row = U2._resource_row(
            f"LC6A_{output_condition}_CHUNK", baseline_swap, chunk=chunk + 1,
            completed_total_ms=full.n_steps * U2.DT_MS,
            wall_s=time.time() - started,
        )
        input_hashes_by_chunk.append(input_hasher.sha256)
        completed_ms = full.n_steps * U2.DT_MS
        if onset_ms is not None:
            for name, target_ms in (
                ("onset_detected", float(onset_ms)),
                ("onset_plus_1s", float(onset_ms) + 1000.0),
                ("onset_plus_2s", float(onset_ms) + 2000.0),
                ("onset_plus_4s", float(onset_ms) + 4000.0),
                ("onset_plus_6s", float(onset_ms) + 6000.0),
                ("onset_plus_8s", float(onset_ms) + 8000.0),
                ("onset_plus_12s", float(onset_ms) + 12000.0),
            ):
                if name not in pinned_checkpoints and completed_ms >= target_ms:
                    record = _pin_checkpoint(
                        work, state, name=name, onset_ms=onset_ms,
                        target_ms=target_ms, actual_ms=completed_ms,
                    )
                    if record is not None:
                        pinned_checkpoints[name] = record
        _write_json(work / "progress.json", {
            "status": "RUNNING", "condition": condition, "completed_chunks": chunk + 1,
            "completed_total_ms": full.n_steps * U2.DT_MS, "onset_ms": onset_ms,
            "n_returning_pre_onset": n_pre, "observation": stop,
            "resource_action": row["action"], "state_hash": state_hash(state),
            "external_input_sha256": input_hasher.sha256,
            "pinned_checkpoints": pinned_checkpoints,
        })
        if (chunk + 1) % 5 == 0 or not stop["continue"]:
            _rotate_recovery_checkpoint(work, state)
        if row["action"] == "TERMINATE_AFTER_CHECKPOINT":
            raise RuntimeError("RESOURCE_STOP_AFTER_CHECKPOINT")
        if not stop["continue"]:
            break

    full = _combine_streams(streams)
    rate = _rate_from_stream(full)
    control_parity = _c0_control_parity(output_condition, full, rate)
    adjudication = PREFIX._adjudicate(full, rate)
    traces = {key: np.concatenate(parts) for key, parts in trace_parts.items()}
    current_traces = current_observer.arrays()
    rate_maps = spatial_rate_maps(
        full.steps, full.cells, spatial_bins, spatial_occupancy,
        n_steps=full.n_steps, dt_ms=U2.DT_MS, window_ms=1000.0,
    )
    cell_rates = per_second_cell_rates(
        full.steps, full.cells, n_steps=full.n_steps, n_cells=S["NE"], dt_ms=U2.DT_MS,
    )
    refractory_ceiling = 1000.0 / float(S["p"].tau_ref_E)
    saturation = local_saturation_readout(
        cell_rates, refractory_ceiling_hz=refractory_ceiling,
    )
    reports = adjudication["reports"]
    per_second_rate = np.asarray([row["mean_hz"] for row in reports], float)
    onset_ms = adjudication.get("onset_ms")
    n_pre = _event_count_before(adjudication["events"], onset_ms)
    if stop["reason"] == "REGISTERED_SATURATION_1S":
        outcome = "ESCALATING_SATURATION"
    elif onset_ms is None:
        outcome = "NO_GLOBAL_ONSET"
    elif adjudication.get("offset_ms") is not None:
        outcome = "AUTONOMOUS_OFFSET_OBSERVED"
    else:
        outcome = "HIGH_STATE_OBSERVED_FOR_PHENOTYPE_MAP"
    tail = min(2, len(per_second_rate))
    summary = {
        "status": "COMPLETE", "condition": output_condition,
        "parent_condition": condition, "outcome": outcome,
        "graph_sha256": graph_sha256(graph), "graph_artifact": str(graph_path),
        "graph_construction_q": graph_metadata["construction_q"],
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "runtime_semantics": "fresh_t0_ZH_dynamic_U0_M0_X1_no_kick_no_step",
        "T_ms": full.n_steps * U2.DT_MS,
        "onset_ms": onset_ms, "offset_ms": adjudication.get("offset_ms"),
        "observation_terminal": stop,
        "n_events": len(adjudication["events"]),
        "n_returning": len(adjudication["returned"]),
        "n_returning_pre_onset": n_pre,
        "control_parity": control_parity,
        "per_second_mean_rate_hz": per_second_rate.tolist(),
        "late_rate_slope_hz_per_s": linear_slope(per_second_rate[-tail:], dt_s=1.0),
        "late_D_slope_per_s": linear_slope(traces["D_mean"][-tail * int(1000/U2.TRACE_DT_MS):], dt_s=U2.TRACE_DT_MS/1000),
        "late_H_slope_per_s": linear_slope(traces["H_mean"][-tail * int(1000/U2.TRACE_DT_MS):], dt_s=U2.TRACE_DT_MS/1000),
        "local_saturation": saturation,
        "local_rate_q95_peak_hz": float(np.nanmax(np.nanquantile(rate_maps, .95, axis=1))),
        "local_rate_q99_peak_hz": float(np.nanmax(np.nanquantile(rate_maps, .99, axis=1))),
        "spatial_map_persistence": spatial_map_persistence(rate_maps),
        "current_decomposition": {
            "sample_dt_ms": float(U2.TRACE_DT_MS),
            "F_E_mean_peak": float(np.max(current_traces["F_E_mean"])),
            "F_E_mean_late": float(np.mean(current_traces["F_E_mean"][-100:])),
            "F_I_mean_peak": float(np.max(current_traces["F_I_mean"])),
            "F_I_mean_late": float(np.mean(current_traces["F_I_mean"][-100:])),
            "I_syn_signed_mean_late": float(
                np.mean(current_traces["I_syn_signed_mean"][-100:])
            ),
        },
        "D_start_end": [float(traces["D_mean"][0]), float(traces["D_mean"][-1])],
        "H_start_end": [float(traces["H_mean"][0]), float(traces["H_mean"][-1])],
        "clip_frac_max": float(np.max(traces["clip_frac"])),
        "external_input_sha256": input_hasher.sha256,
        "external_input_sha256_by_chunk": input_hashes_by_chunk,
        "spike_sha256": full.sha256,
        "initial_state_hash": initial_state_hash,
        "final_state_hash": state_hash(state),
        "source_sha256": source_hashes,
        "graph_realization_confirmation": confirmation,
        "pinned_checkpoints": pinned_checkpoints,
        "config_scalar": {
            key: value for key, value in cfg.items()
            if np.isscalar(value) and not isinstance(value, (bytes, bytearray))
        },
        "wall_s": time.time() - started,
    }
    with AtomicStageBundle(arm) as bundle:
        _write_json(bundle.path("summary.json"), summary)
        _npz_atomic(
            bundle.path("spikes.npz"), steps=full.steps, cells=full.cells.astype(np.int32),
            n_steps=np.asarray([full.n_steps], np.int64),
            n_cells=np.asarray([full.n_cells], np.int64), sha256=np.asarray([full.sha256]),
        )
        _npz_atomic(
            bundle.path("traces.npz"), rate_dt_ms=np.asarray([U2.TRACE_DT_MS], np.float32),
            rate_E=rate[::stride].astype(np.float32), **traces, **current_traces,
        )
        _npz_atomic(
            bundle.path("spatial_readouts.npz"),
            rate_maps_1s=rate_maps.astype(np.float32),
            D_maps_1s=np.asarray(d_maps, np.float32), H_maps_1s=np.asarray(h_maps, np.float32),
            cell_bins=spatial_bins, occupancy=spatial_occupancy,
            n_bins_axis=np.asarray([32], np.int32),
            positions_E=np.asarray(S["posE"], np.float32),
            patient_axis_unit=np.asarray(S["axis_unit"], np.float32),
            source_xy=np.asarray(S["src_xy"], np.float32),
            sink_xy=np.asarray(S["snk_xy"], np.float32),
            sheet_size_mm=np.asarray([S["L"]], np.float32),
        )
        U2.save_loop_state(str(bundle.path("final_state.npz")), state)
        pinned_required = []
        for record in pinned_checkpoints.values():
            shutil.copy2(work / record["file"], bundle.path(record["file"]))
            pinned_required.append(record["file"])
        bundle.commit(required=[
            "summary.json", "spikes.npz", "traces.npz", "spatial_readouts.npz",
            "final_state.npz", *pinned_required,
        ])
    shutil.rmtree(work, ignore_errors=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=GRAPH_IDS, required=True)
    parser.add_argument("--graph-artifact", type=Path)
    parser.add_argument("--output-condition")
    parser.add_argument("--confirmation-lock", type=Path)
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A natural trajectory requires --confirm-run")
    if args.graph_artifact is None and any(
        value is not None for value in (args.output_condition, args.confirmation_lock)
    ):
        parser.error("confirmation options require --graph-artifact")
    if args.graph_artifact is not None and not (args.output_condition and args.confirmation_lock):
        parser.error("confirmation graph requires --output-condition and --confirmation-lock")
    output_condition = args.condition if args.output_condition is None else args.output_condition
    if not output_condition.replace("_", "").isalnum():
        parser.error("output condition must be alphanumeric/underscore")
    trajectory_root = OUT / "trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    lock_path = trajectory_root / f".{output_condition}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC6A {output_condition} is already running") from exc
        running = trajectory_root / f"RUNNING_{output_condition}.json"
        failed = trajectory_root / f"FAILED_{output_condition}.json"
        done = trajectory_root / f"DONE_{output_condition}.json"
        _write_json(running, {"status": "RUNNING", "condition": output_condition, "pid": os.getpid()})
        try:
            result = run(
                args.condition, args.execution_manifest,
                graph_path_override=args.graph_artifact,
                output_condition=output_condition,
                confirmation_lock=args.confirmation_lock,
            )
            _write_json(done, {
                "status": "DONE", "condition": output_condition,
                "outcome": result["outcome"], "summary": str(OUT / f"trajectories/{output_condition}/summary.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {
                "status": "FAILED", "condition": args.condition,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
