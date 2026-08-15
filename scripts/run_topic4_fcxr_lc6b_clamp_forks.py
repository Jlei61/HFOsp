#!/usr/bin/env python3
"""FCXR-LC6B: continue one C0 snapshot with D and/or H held fixed, and read the fast subsystem.

One arm = one (source snapshot, clamp) pair resumed from an exact LC6A checkpoint.  The four arms of
a snapshot share the same full fast state and, because the checkpoint restores the generator and the
per-step draw pattern does not depend on D or H, the same future external input -- which the runner
proves by hash rather than by argument.
"""

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

import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from mz_slow_vars import lc2_h_gate  # noqa: E402
from src.topic4_fcxr_lc3 import _jsonable as _state_jsonable  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle, ExactInputHasher, SparseSpikeBinaryWriter,
)
from src.topic4_fcxr_lc6_gain import active_area_mm2, binned_global_rate  # noqa: E402
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    NaturalCurrentObserver, cell_spatial_bins, per_second_cell_rates, spatial_rate_maps,
)
from src.topic4_fcxr_lc6b_clamp import (  # noqa: E402
    ARMS, apply_slow_clamp, cell_rate_distribution, classify_clamp_window,
)


MANIFEST = ROOT / "config/topic4_fcxr_lc6b_frozen_slow_atlas.json"
LC6A_MANIFEST = ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json"
LOCAL_LOCK = NAT.OUT / "local_classifier_manifest_addendum.json"
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas"
FORK_ROOT = OUT / "forks"
SUMMARY = OUT / "clamp_fork_summary.json"
RESOURCE_LOG = OUT / "resource_log.jsonl"
MECHANISM_FILES = (
    Path(__file__).resolve(),
    Path(NAT.__file__).resolve(),
    ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_statefork.py",
    ROOT / "src/topic4_fcxr_lc6b_clamp.py",
    ROOT / "src/topic4_fcxr_lc6_gain.py",
    ROOT / "src/topic4_fcxr_lc6_trajectory.py",
    ROOT / "src/snn_engine/mz_slow_vars.py",
)
TRACE_ATTRS = {
    "D_mean": "trace_z_mean",
    "H_mean": "trace_h_lc2_mean",
    "H_max": "trace_h_lc2_max",
    "H_source_mean": "trace_gA_raw_lc2_mean",
    "gH_mean": "trace_gH_lc2_mean",
    "gErec_mean": "trace_gErec_mean",
    "gI_mean": "trace_gI_mean",
    "clip_frac": "trace_conductance_clip_frac",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in MECHANISM_FILES}


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest():
    payload = json.loads(MANIFEST.read_text())
    if payload.get("experiment_id") != "fcxr_lc6b_frozen_slow_atlas":
        raise RuntimeError("wrong LC6B execution manifest")
    for relative, expected in payload["blessed_engine_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"blessed engine hash mismatch: {relative}")
    if _sha(LC6A_MANIFEST) != payload["upstream"]["lc6a_manifest_sha256"]:
        raise RuntimeError("upstream LC6A manifest drift")
    return payload


def _resource_row(stage, **extra):
    info = NAT.U2.GEO._meminfo()
    row = {
        "stage": stage, "mem_available_gib": info["mem_available_gib"],
        "swap_used_mib": info["swap_used_mib"], **extra,
    }
    RESOURCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RESOURCE_LOG.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _system():
    """The C0 substrate exactly as LC6A built it, plus a template state for exact loading."""

    _path, _payload, source_summary = NAT._validate_manifest(LC6A_MANIFEST, "C0")
    graph, metadata = NAT._load_graph(NAT.OUT / "graphs/C0.npz")
    S, slow, cfg = NAT._fresh_system(
        json.loads(source_summary.read_text()), graph, NAT.graph_sha256(graph), "C0",
    )
    template = NAT.U2.PM._seed_template(S, slow)
    return S, template, cfg, NAT.graph_sha256(graph), metadata


def _checkpoint_record(state, *, snapshot, arm, path, graph_sha, config_sha, onset_ms,
                       input_sha, input_steps):
    """The LC6B checkpoint metadata contract: a file name never carries the time again."""

    t_ms = float(state.t) * NAT.U2.DT_MS
    return {
        "file": Path(path).name,
        "snapshot_time_ms": t_ms,
        "onset_time_ms": float(onset_ms),
        "relative_to_onset_ms": t_ms - float(onset_ms),
        "t_steps": int(state.t),
        "full_state_sha256": _sha(path),
        "state_hash": state_hash(state),
        "external_input_counter": int(input_steps),
        # The generator state IS the external-input state: the noise stream is drawn from it and
        # nothing else.  ``_jsonable`` is the same canonicalisation the state file itself uses, so
        # this digest is comparable across arms and across saved checkpoints.
        "external_input_state": _digest(_state_jsonable(state.rng_state)),
        "external_input_sha256": input_sha,
        "graph_sha256": graph_sha,
        "config_sha256": config_sha,
        "source_snapshot": snapshot,
        "arm": arm,
    }


def _trace_chunk(slow, starts, stride):
    result = {}
    for output, attribute in TRACE_ATTRS.items():
        values = np.asarray(getattr(slow, attribute)[starts[attribute]:], float)
        result[output] = values[::stride].astype(np.float32)
    result["D_mean"] = 1.0 - result["D_mean"]          # trace_z_mean holds z; D = 1 - z
    return result


def _h_gate_readout(slow, cfg):
    gate = lc2_h_gate(slow.h_lc2_E, theta=cfg["theta_h_lc2"], k=cfg["k_h_lc2"])
    return {
        "h_mean": float(slow.h_lc2_E.mean()), "h_max": float(slow.h_lc2_E.max()),
        "gate_mean": float(gate.mean()),
        "gate_fraction_above_half": float(np.mean(gate > 0.5)),
        "gate_fraction_above_0p99": float(np.mean(gate > 0.99)),
        "gH_mean": float(cfg["rho_h_lc2"] * gate.mean()),
        "gH_ceiling": float(cfg["rho_h_lc2"]),
    }


def run_arm(snapshot, arm, *, continuation_ms=None, tag=None):
    manifest = _manifest()
    manifest_sha = _sha(MANIFEST)
    source_hashes = _source_hashes()
    snap = manifest["source_snapshots"][snapshot]
    clamp_d, clamp_h = ARMS[arm]
    observation = manifest["observation"]
    total_ms = float(observation["continuation_ms"] if continuation_ms is None else continuation_ms)
    chunk_ms = min(float(observation["chunk_ms"]), total_ms)
    bin_ms = float(observation["rate_bin_ms"])

    required_gib = float(manifest["resource"]["required_mem_available_gib"])
    if NAT.U2.GEO._meminfo()["mem_available_gib"] < required_gib:
        raise RuntimeError(f"LC6B requires at least {required_gib} GiB MemAvailable")

    name = tag or f"{snapshot}_{arm}"
    bundle_dir = FORK_ROOT / name
    if bundle_dir.is_dir():
        return json.loads((bundle_dir / "summary.json").read_text())
    work = FORK_ROOT / f".{name}.work"
    if work.exists():
        raise RuntimeError(f"stale work directory requires inspection: {work}")
    work.mkdir(parents=True)
    started = time.time()

    checkpoint_path = ROOT / snap["checkpoint"]
    if _sha(checkpoint_path) != snap["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint drift")
    S, template, cfg, graph_sha, graph_meta = _system()
    if graph_sha != manifest["upstream"]["graph_sha256"]:
        raise RuntimeError("runtime C0 graph differs from the registered artifact")
    state = NAT.U2.load_into(str(checkpoint_path), template)
    if state_hash(state) != snap["state_hash"]:
        raise RuntimeError("loaded source checkpoint state hash mismatch")
    if int(state.t) != int(snap["t_steps"]):
        raise RuntimeError("loaded source checkpoint step counter mismatch")

    child, clamp_record = apply_slow_clamp(state, clamp_d=clamp_d, clamp_h=clamp_h)
    start_gate = _h_gate_readout(child.slow, cfg)
    start_d = 1.0 - np.asarray(child.slow.z[: S["NE"]], float)

    stride = int(round(NAT.U2.TRACE_DT_MS / NAT.U2.DT_MS))
    chunk_steps = int(round(chunk_ms / NAT.U2.DT_MS))
    n_chunks = int(round(total_ms / chunk_ms))
    p = dataclasses.replace(S["p"], T=total_ms, dt=NAT.U2.DT_MS)
    hasher = ExactInputHasher()
    observer = NaturalCurrentObserver(dt_ms=NAT.U2.DT_MS, sample_dt_ms=NAT.U2.TRACE_DT_MS)
    numerical_fail, fail_detail = False, None
    steps, cells, trace_parts, gate_rows, resource_rows = [], [], {}, [], []
    completed_steps = 0

    try:
        for chunk in range(n_chunks):
            if _source_hashes() != source_hashes:
                raise RuntimeError("mechanism source drifted during an LC6B arm")
            starts = {attr: len(getattr(child.slow, attr)) for attr in TRACE_ATTRS.values()}
            binary = work / f"chunk_{chunk:02d}.bin"
            writer = SparseSpikeBinaryWriter(
                binary, step_origin=child.t, n_steps=chunk_steps, n_cells=S["NE"],
            )
            try:
                out = run_fcxr_loop(
                    p, S["net"], start=child, n_steps=chunk_steps, capture_final=True,
                    store_spikes=False, spike_sink=writer, input_sink=hasher,
                    membrane_term_sink=observer.sample, v_th_per_neuron=S["vth"],
                )
                stream = writer.finalize(work / f"chunk_{chunk:02d}_spikes.npz")
            finally:
                writer.close()
                binary.unlink(missing_ok=True)
            child = out["checkpoint"]
            steps.append(np.asarray(stream.steps, np.int64) + chunk * chunk_steps)
            cells.append(np.asarray(stream.cells, np.int32))
            completed_steps += chunk_steps
            for key, value in _trace_chunk(child.slow, starts, stride).items():
                trace_parts.setdefault(key, []).append(value)
            gate_rows.append({"chunk_end_ms": (chunk + 1) * chunk_ms,
                              **_h_gate_readout(child.slow, cfg)})
            NAT.U2.save_loop_state(str(work / "rolling_checkpoint.npz"), child)
            resource_rows.append(_resource_row(
                f"LC6B_{name}_CHUNK", chunk=chunk + 1,
                completed_ms=completed_steps * NAT.U2.DT_MS, wall_s=time.time() - started,
            ))
            NAT._write_json(work / "progress.json", {
                "status": "RUNNING", "arm": name, "completed_chunks": chunk + 1,
                "completed_ms": completed_steps * NAT.U2.DT_MS,
                "state_hash": state_hash(child), "external_input_sha256": hasher.sha256,
            })
    except FloatingPointError as exc:
        numerical_fail, fail_detail = True, f"{type(exc).__name__}: {exc}"

    completed_ms = completed_steps * NAT.U2.DT_MS
    all_steps = np.concatenate(steps) if steps else np.zeros(0, np.int64)
    all_cells = np.concatenate(cells) if cells else np.zeros(0, np.int32)
    rate_bins = binned_global_rate(
        all_steps, n_steps=completed_steps, n_cells=S["NE"], dt_ms=NAT.U2.DT_MS, bin_ms=bin_ms,
    ) if completed_steps else np.zeros(0)
    cell_rates = per_second_cell_rates(
        all_steps, all_cells, n_steps=completed_steps, n_cells=S["NE"], dt_ms=NAT.U2.DT_MS,
    ) if completed_steps else np.zeros((0, S["NE"]))
    bins, occupancy = cell_spatial_bins(
        S["posE"], sheet_size_mm=S["L"], n_bins_axis=int(observation["spatial_bins_per_axis"]),
    )
    maps = spatial_rate_maps(
        all_steps, all_cells, bins, occupancy, n_steps=completed_steps,
        dt_ms=NAT.U2.DT_MS, window_ms=float(observation["spatial_window_ms"]),
    ) if completed_steps else np.zeros((0, occupancy.size))
    local_thresholds = json.loads(LOCAL_LOCK.read_text())["thresholds"]
    area = active_area_mm2(
        maps, occupancy, rate_threshold_hz=local_thresholds["rate_threshold_hz"],
        sheet_size_mm=S["L"],
    ) if maps.size else np.zeros(0)
    traces = {key: np.concatenate(parts) for key, parts in trace_parts.items()}
    currents = observer.arrays()

    verdict = classify_clamp_window(
        rate_bins_hz=rate_bins, cell_rates_hz=cell_rates,
        completed_ms=completed_ms, registered_ms=total_ms,
        numerical_fail=numerical_fail, bin_ms=bin_ms,
        **{key: manifest["classifier"]["thresholds"][key] for key in (
            "global_saturation_hz", "refractory_ceiling_hz",
            "near_refractory_fraction_gate", "interictal_roll_hi_hz",
            "drift_ci_gate_per_s", "silence_bin_fraction_gate", "tail_s")},
    )
    config_sha = _digest({"manifest_sha256": manifest_sha, "clamp": clamp_record["clamp_config_sha256"]})
    final_state_path = work / "final_state.npz"
    NAT.U2.save_loop_state(str(final_state_path), child)
    final_record = _checkpoint_record(
        child, snapshot=snapshot, arm=arm, path=final_state_path, graph_sha=graph_sha,
        config_sha=config_sha, onset_ms=snap["onset_time_ms"],
        input_sha=hasher.sha256, input_steps=completed_steps,
    )
    end_d = 1.0 - np.asarray(child.slow.z[: S["NE"]], float)
    summary = {
        "status": "COMPLETE", "arm_id": name, "source_snapshot": snapshot, "arm": arm,
        "clamp": clamp_record, "numerical_fail": numerical_fail, "numerical_detail": fail_detail,
        "manifest": str(MANIFEST), "manifest_sha256": manifest_sha, "config_sha256": config_sha,
        "graph_sha256": graph_sha, "graph_construction_q": graph_meta["construction_q"],
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": snap["checkpoint_sha256"],
        "start_state_hash": snap["state_hash"], "start_t_steps": snap["t_steps"],
        "snapshot_time_ms": snap["snapshot_time_ms"], "onset_time_ms": snap["onset_time_ms"],
        "relative_to_onset_ms": snap["relative_to_onset_ms"],
        "state_hash_after_clamp": clamp_record["state_hash_after_clamp"],
        "registered_continuation_ms": total_ms, "completed_ms": completed_ms,
        "rate_bin_ms": bin_ms, "trace_dt_ms": NAT.U2.TRACE_DT_MS,
        "external_input_sha256": hasher.sha256,
        "spike_count": int(all_steps.size),
        "verdict": verdict,
        "cell_rate_distribution": cell_rate_distribution(
            cell_rates, refractory_ceiling_hz=manifest["classifier"]["thresholds"]["refractory_ceiling_hz"],
        ) if cell_rates.size else None,
        "active_area_mm2": area.tolist(),
        "max_active_area_mm2": float(area.max()) if area.size else None,
        "sheet_area_mm2": float(S["L"]) ** 2,
        "local_rate_q95_peak_hz": float(np.nanmax(np.nanquantile(maps, .95, axis=1))) if maps.size else None,
        "local_rate_q99_peak_hz": float(np.nanmax(np.nanquantile(maps, .99, axis=1))) if maps.size else None,
        "local_rate_threshold_hz": float(local_thresholds["rate_threshold_hz"]),
        "h_gate": {"at_fork": start_gate, "per_chunk": gate_rows},
        "D_field": {
            "at_fork": {"mean": float(start_d.mean()), "min": float(start_d.min()),
                        "max": float(start_d.max()), "median": float(np.median(start_d))},
            "at_end": {"mean": float(end_d.mean()), "min": float(end_d.min()),
                       "max": float(end_d.max()), "median": float(np.median(end_d))},
        },
        "current_decomposition": {
            "sample_dt_ms": float(NAT.U2.TRACE_DT_MS),
            "F_E_mean_peak": float(np.max(currents["F_E_mean"])) if currents["F_E_mean"].size else None,
            "F_E_mean_late": float(np.mean(currents["F_E_mean"][-100:])) if currents["F_E_mean"].size else None,
            "F_I_mean_peak": float(np.max(currents["F_I_mean"])) if currents["F_I_mean"].size else None,
            "F_I_mean_late": float(np.mean(currents["F_I_mean"][-100:])) if currents["F_I_mean"].size else None,
            "I_syn_signed_mean_late": float(np.mean(currents["I_syn_signed_mean"][-100:]))
            if currents["I_syn_signed_mean"].size else None,
        },
        "trace_endpoints": {
            key: [float(values[0]), float(values[-1])] for key, values in traces.items() if values.size
        },
        "clip_frac_max": float(np.max(traces["clip_frac"])) if traces.get("clip_frac", np.zeros(0)).size else 0.0,
        "final_checkpoint": final_record,
        "termination_tested": False, "lifecycle_tested": False,
        "perturbation_return_tested": False,
        "source_sha256": source_hashes,
        "resource_rows": resource_rows,
        "wall_s": time.time() - started,
    }
    with AtomicStageBundle(bundle_dir) as bundle:
        NAT._write_json(bundle.path("summary.json"), summary)
        NAT._npz_atomic(
            bundle.path("spikes.npz"), steps=all_steps, cells=all_cells,
            n_steps=np.asarray([completed_steps], np.int64),
            n_cells=np.asarray([S["NE"]], np.int64),
        )
        NAT._npz_atomic(
            bundle.path("traces.npz"), rate_bin_ms=np.asarray([bin_ms], np.float32),
            rate_bins_hz=rate_bins.astype(np.float32),
            active_area_mm2=area.astype(np.float32),
            per_second_mean_hz=np.asarray(verdict["per_second_mean_hz"], np.float32),
            **traces, **currents,
        )
        shutil.copy2(final_state_path, bundle.path("final_state.npz"))
        bundle.commit(required=["summary.json", "spikes.npz", "traces.npz", "final_state.npz"])
    shutil.rmtree(work, ignore_errors=True)
    return summary


def _decision(rows):
    """Spec section 10: the first-round A/B/C branch, derived only from the DH_CLAMP outcomes."""

    by_key = {(row["source_snapshot"], row["arm"]): row["verdict"]["label"] for row in rows}
    dh = {snap: by_key.get((snap, "DH_CLAMP")) for snap in sorted({r["source_snapshot"] for r in rows})}
    bounded = {"BOUNDED_STATIONARY", "BOUNDED_OSCILLATORY"}
    low = {"LOW_STATE", "SILENCE", "AFTER_DISCHARGE"}
    if any(label in bounded for label in dh.values()):
        branch, action = "B", "ENTER_CONDITIONAL_NATURAL_PATH_ATLAS"
    elif all(label in low for label in dh.values() if label is not None) and dh:
        branch, action = "C", "ADD_AN_EARLIER_ADJACENT_SNAPSHOT_ONSET_PLUS_1S"
    else:
        branch, action = "A", "WRITE_H_EFF_AND_H_CAP_SPEC_PLAN_ONLY_AND_STOP_FOR_REVIEW"
    driver = {}
    for snap in dh:
        labels = {arm: by_key.get((snap, arm)) for arm in ARMS}
        if labels.get("DH_CLAMP") in bounded:
            h_ok = labels.get("H_CLAMP") in bounded
            d_ok = labels.get("D_CLAMP") in bounded
            driver[snap] = ("EITHER_SLOW_VARIABLE_ALONE_CROSSES_THE_CARRIER" if h_ok and d_ok
                            else "CONTINUED_H_RECRUITMENT_IS_THE_MAIN_DRIVER" if h_ok
                            else "CONTINUED_D_DEPLETION_IS_THE_MAIN_DRIVER" if d_ok
                            else "D_AND_H_SLOW_FLOW_ACT_TOGETHER")
        else:
            driver[snap] = None
    return {
        "branch": branch, "next_authorized_action": action,
        "dh_clamp_labels": dh, "labels": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_key.items())},
        "driver_attribution": driver,
        "natural_path_atlas_authorized": branch == "B",
    }


def finalize():
    manifest = _manifest()
    rows = []
    for snapshot in manifest["source_snapshots"]:
        for arm in ARMS:
            path = FORK_ROOT / f"{snapshot}_{arm}" / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"LC6B arm incomplete: {path}")
            rows.append(json.loads(path.read_text()))
    paired = {}
    for snapshot in manifest["source_snapshots"]:
        hashes = {
            row["arm"]: row["external_input_sha256"]
            for row in rows if row["source_snapshot"] == snapshot
        }
        paired[snapshot] = {
            "external_input_sha256": hashes,
            "all_arms_share_future_input": len(set(hashes.values())) == 1,
            "state_hashes_distinct": len({
                row["final_checkpoint"]["state_hash"]
                for row in rows if row["source_snapshot"] == snapshot
            }) == len(hashes),
        }
        if not paired[snapshot]["all_arms_share_future_input"]:
            raise RuntimeError(f"G1: arms of {snapshot} do not share a future external input")
    payload = {
        "status": "COMPLETE", "experiment_id": manifest["experiment_id"],
        "manifest_sha256": _sha(MANIFEST),
        "n_arms": len(rows), "rows": rows, "paired_input_check": paired,
        "decision": _decision(rows),
        "termination_tested": False, "lifecycle_tested": False,
        "perturbation_return_tested": False,
        "claim_boundary": (
            "Every label describes one clamped continuation of ONE canonical-seed C0 trajectory at "
            "one of two snapshots.  A bounded label states that the branch persisted across the "
            "registered window; it does not state that a weak perturbation returns to the same "
            "envelope, and it does not test termination or lifecycle."
        ),
        "source_sha256": _source_hashes(),
    }
    NAT._write_json(SUMMARY, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "finalize"))
    parser.add_argument("--snapshot")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--continuation-ms", type=float)
    parser.add_argument("--tag")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B clamp forks require --confirm-run")
    if args.stage == "run" and not (args.snapshot and args.arm):
        parser.error("run requires --snapshot and --arm")
    OUT.mkdir(parents=True, exist_ok=True)
    FORK_ROOT.mkdir(parents=True, exist_ok=True)
    label = "FINALIZE" if args.stage == "finalize" else (args.tag or f"{args.snapshot}_{args.arm}")
    with (OUT / f".{label}.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC6B {label} is already running") from exc
        running = OUT / f"RUNNING_{label}.json"
        failed = OUT / f"FAILED_{label}.json"
        done = OUT / f"DONE_{label}.json"
        NAT._write_json(running, {"status": "RUNNING", "stage": args.stage, "pid": os.getpid()})
        try:
            if args.stage == "run":
                result = run_arm(args.snapshot, args.arm,
                                 continuation_ms=args.continuation_ms, tag=args.tag)
                note = {"label": result["verdict"]["label"], "reason": result["verdict"]["reason"]}
            else:
                result = finalize()
                note = result["decision"]
            NAT._write_json(done, {"status": "DONE", "stage": args.stage, "label": label, **note})
            failed.unlink(missing_ok=True)
            print(json.dumps(NAT._jsonable(note), indent=2, sort_keys=True))
        except BaseException as exc:
            NAT._write_json(failed, {"status": "FAILED", "label": label,
                                     "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
