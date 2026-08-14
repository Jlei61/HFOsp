#!/usr/bin/env python3
"""Lock and run the descriptive paired LC6A baseline functional probes."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import clone_loop_state, run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc3_perturb import run_fcxr_perturbation  # noqa: E402
from src.topic4_fcxr_lc5 import AtomicStageBundle, ExactInputHasher  # noqa: E402
from src.topic4_fcxr_lc6_functional import (  # noqa: E402
    FunctionalResponseRecorder, array_sha256, local_patch_pattern,
    locked_patch_centers, paired_response,
)
from src.topic4_fcxr_lc6_surround import (  # noqa: E402
    extract_e_to_i, graph_sha256, replace_e_to_i_in_net,
)


OUT = NAT.OUT
PRELOCK = ROOT / "config/topic4_fcxr_lc6a_functional_probe_prelock.json"
LOCK = OUT / "functional_probe_lock.json"
LOCK_STATE = OUT / "functional_probe_lock_state.npz"
AUTHORIZATION = OUT / "lc5_to_lc6a_authorization.json"
GRAPH_IDS = NAT.GRAPH_IDS
MECHANISM_FILES = (
    Path(__file__).resolve(), PRELOCK,
    ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_perturb.py",
    ROOT / "src/topic4_fcxr_lc6_functional.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in MECHANISM_FILES}


def _write_json(path, payload):
    NAT._write_json(path, payload)


def _require_authorization():
    if not AUTHORIZATION.is_file():
        raise RuntimeError("LC5 terminal-negative authorization is required for LC6A 40k dynamics")
    payload = json.loads(AUTHORIZATION.read_text())
    if payload.get("authorize_lc6a_40k_dynamics") is not True:
        raise RuntimeError("LC5 result does not authorize LC6A 40k dynamics")
    source = Path(payload["lc5_summary"])
    if not source.is_file() or _sha(source) != payload["lc5_summary_sha256"]:
        raise RuntimeError("LC5 authorization source hash mismatch")
    return payload


def _validate_prelock():
    payload = json.loads(PRELOCK.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_functional_probe":
        raise RuntimeError("wrong functional-probe prelock")
    return payload


def _probe_system(condition, manifest_path):
    manifest_path, manifest, source_summary = NAT._validate_manifest(manifest_path, condition)
    summary = json.loads(source_summary.read_text())
    graph, metadata = NAT._load_graph(OUT / f"graphs/{condition}.npz")
    if metadata.get("graph_legality", "PASS") != "PASS":
        raise RuntimeError(f"GRAPH_LEGALITY_FAILED_{condition}")
    S = NAT.U2.PP.build_substrate(NAT.U2.CONNECTION_SEED)
    base = extract_e_to_i(S["net"], S["NE"], S["NI"])
    if condition == "C0":
        if graph_sha256(base) != graph_sha256(graph):
            raise RuntimeError("C0 graph is not exact substrate parity")
    else:
        S["net"] = replace_e_to_i_in_net(S["net"], graph, ne=S["NE"], ni=S["NI"])
    NAT.U2.install_registered_noise_rng(S["net"])
    cfg = NAT._fresh_config(summary, S["NE"])
    cfg.update(
        use_z=False,
        z_frozen_E=np.ones(S["NE"], dtype=float),
        use_h_lc2=True,
        h_lc2_init_E=np.zeros(S["NE"], dtype=float),
        rho_h_lc2=0.0,
        use_pump=False,
        pump_Imax=0.0,
        use_m=False,
        use_x=True,
        x_relay_frozen_E=np.ones(S["NE"], dtype=float),
    )
    slow = MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
        core_mask_E=NAT.U2.OLD_SLOW.build_core_masks(S),
    )
    S["net"]["rng"] = np.random.default_rng(NAT.U2.NOISE_SEED)
    state = NAT.U2.PM._seed_template(S, slow)
    return S, state, graph, metadata, manifest


def _active_fraction_1ms(raster, dt_ms):
    raster = np.asarray(raster, bool)
    steps = int(round(1.0 / float(dt_ms)))
    usable = raster.shape[0] // steps * steps
    return raster[:usable].reshape(-1, steps, raster.shape[1]).any(axis=1).mean(axis=1)


def _find_c0_quiet_state(S, state, timing, event_bar):
    window_ms = float(timing["quiet_search_window_ms"])
    steps = int(round(window_ms / NAT.U2.DT_MS))
    minimum = float(timing["minimum_c0_burnin_ms"])
    maximum = float(timing["maximum_c0_burnin_ms"])
    p = dataclasses.replace(S["p"], T=maximum, dt=NAT.U2.DT_MS)
    rows = []
    while state.t * NAT.U2.DT_MS < maximum:
        start = clone_loop_state(state)
        out = run_fcxr_loop(
            p, S["net"], start=state, n_steps=steps, capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )
        state = out["checkpoint"]
        af = _active_fraction_1ms(out["E_spk_bool"], NAT.U2.DT_MS)
        row = {
            "start_ms": start.t * NAT.U2.DT_MS,
            "end_ms": state.t * NAT.U2.DT_MS,
            "max_active_fraction_1ms": float(np.max(af)),
            "mean_rate_hz": float(np.mean(out["rate_E"])),
        }
        rows.append(row)
        if row["start_ms"] >= minimum and row["max_active_fraction_1ms"] < float(event_bar):
            return S, start, rows
    raise RuntimeError("no event-free C0 baseline window was found under the frozen search rule")


def _advance_to_ms(S, state, target_ms):
    n_steps = int(round(float(target_ms) / NAT.U2.DT_MS)) - int(state.t)
    if n_steps < 0:
        raise RuntimeError("probe start time precedes fresh state")
    if n_steps == 0:
        return state
    p = dataclasses.replace(S["p"], T=float(target_ms), dt=NAT.U2.DT_MS)
    return run_fcxr_loop(
        p, S["net"], start=state, n_steps=n_steps, capture_final=True,
        store_spikes=False, v_th_per_neuron=S["vth"],
    )["checkpoint"]


def _make_recorder(S, center, prelock):
    geometry = prelock["geometry"]
    windows = prelock["timing"]["registered_windows_ms"]
    edges = [windows[0][0], *[row[1] for row in windows]]
    axis_range = geometry["axis_range_mm"]
    axis_edges = np.arange(
        float(axis_range[0]), float(axis_range[1]) + 1e-12,
        float(geometry["axis_bin_width_mm"]),
    )
    return FunctionalResponseRecorder(
        S["posE"], patch_center=center, axis_unit=S["axis_unit"],
        dt_ms=NAT.U2.DT_MS, window_edges_ms=edges, axis_edges_mm=axis_edges,
        transverse_half_width_mm=geometry["transverse_half_width_mm"],
        sheet_size_mm=S["L"], n_map_bins_axis=32,
    )


def _run_one(S, start, center, amplitude, prelock):
    timing = prelock["timing"]
    response_ms = float(timing["response_ms"])
    pulse_ms = float(timing["pulse_ms"])
    recorder = _make_recorder(S, center, prelock)
    hasher = ExactInputHasher()
    pattern = local_patch_pattern(
        S["posE"], center, radius_mm=prelock["geometry"]["patch_radius_mm"],
    )
    p = dataclasses.replace(S["p"], T=response_ms, dt=NAT.U2.DT_MS)
    out = run_fcxr_perturbation(
        p, S["net"], start=clone_loop_state(start),
        n_steps=int(round(response_ms / NAT.U2.DT_MS)),
        current_pattern=pattern, amplitude=float(amplitude),
        pulse_steps=int(round(pulse_ms / NAT.U2.DT_MS)),
        capture_final=True, store_spikes=False, v_th_per_neuron=S["vth"],
        input_sink=hasher,
        membrane_term_sink=recorder.sample_membrane,
        spike_sink=recorder.sample_spikes,
    )
    return {
        "readout": recorder.finalize(),
        "input_sha256": hasher.sha256,
        "final_state_hash": state_hash(out["checkpoint"]),
        "pulse_accounting": out["pulse_accounting"],
    }


def _pair(S, start, center, amplitude, prelock):
    sham = _run_one(S, start, center, 0.0, prelock)
    probe = _run_one(S, start, center, amplitude, prelock)
    if sham["input_sha256"] != probe["input_sha256"]:
        raise RuntimeError("paired functional arms did not share exact external input")
    paired = paired_response(sham["readout"], probe["readout"])
    return sham, probe, paired


def lock_amplitude(manifest_path):
    _require_authorization()
    prelock = _validate_prelock()
    if LOCK.is_file() and LOCK_STATE.is_file():
        return json.loads(LOCK.read_text())
    S, state, graph, _meta, _manifest = _probe_system("C0", manifest_path)
    baseline = NAT.U2._baseline()
    S, state, quiet_rows = _find_c0_quiet_state(
        S, state, prelock["timing"], baseline["event_bar"],
    )
    centers = locked_patch_centers(
        S, patch_radius_mm=prelock["geometry"]["patch_radius_mm"],
        core_radius_mm=PP.CORE_R,
    )
    slow = clone_loop_state(state).slow
    drive, _g_rel, _g_rev = slow.membrane_terms(
        state.I_E, state.I_I, S["net"]["labels"], I_E_rec=state.I_E_rec,
    )
    i_ref = float(np.quantile(np.abs(np.asarray(drive[:S["NE"]], float)), .95))
    if not np.isfinite(i_ref) or i_ref <= 0.0:
        raise RuntimeError("C0 functional amplitude reference is invalid")
    candidates = []
    selected = None
    for fraction in prelock["amplitude_lock"]["candidate_fractions"]:
        amplitude = float(fraction) * i_ref
        sham, probe, paired = _pair(S, state, centers["neutral_axis"], amplitude, prelock)
        subthreshold = paired["max_active_fraction_1ms_probe"] < float(baseline["event_bar"])
        candidates.append({
            "fraction": float(fraction), "amplitude": amplitude,
            "subthreshold": bool(subthreshold),
            "max_active_fraction_1ms_sham": paired["max_active_fraction_1ms_sham"],
            "max_active_fraction_1ms_probe": paired["max_active_fraction_1ms_probe"],
            "excess_spikes": paired["excess_spikes"],
            "external_input_sha256": sham["input_sha256"],
        })
        if subthreshold:
            selected = candidates[-1]
    if selected is None:
        raise RuntimeError("no frozen functional amplitude candidate remained subthreshold")
    NAT.U2.save_loop_state(str(LOCK_STATE), state)
    payload = {
        "status": "LOCKED",
        "stage": "LC6A_FUNCTIONAL_AMPLITUDE_LOCK",
        "prelock": str(PRELOCK), "prelock_sha256": _sha(PRELOCK),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha(manifest_path),
        "C0_graph_sha256": graph_sha256(graph),
        "q_trajectory_outcome_read": False,
        "selected_start_ms": state.t * NAT.U2.DT_MS,
        "selected_state_hash": state_hash(state),
        "selected_state_artifact": str(LOCK_STATE),
        "selected_state_artifact_sha256": _sha(LOCK_STATE),
        "quiet_search": quiet_rows,
        "patch_centers": centers,
        "amplitude_reference": {"definition": prelock["amplitude_lock"]["reference"], "value": i_ref},
        "event_bar": baseline["event_bar"],
        "candidates": candidates,
        "selected": selected,
        "source_sha256": _source_hashes(),
    }
    _write_json(LOCK, payload)
    return payload


def run_condition(condition, manifest_path):
    _require_authorization()
    prelock = _validate_prelock()
    if not LOCK.is_file():
        raise RuntimeError("functional amplitude lock must run first")
    lock = json.loads(LOCK.read_text())
    if lock["prelock_sha256"] != _sha(PRELOCK):
        raise RuntimeError("functional prelock drifted after amplitude selection")
    source_hashes = _source_hashes()
    arm = OUT / f"functional_probes/{condition}"
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    S, state, graph, metadata, _manifest = _probe_system(condition, manifest_path)
    state = _advance_to_ms(S, state, lock["selected_start_ms"])
    locations = prelock["probe_matrix"][condition]
    summaries, arrays = {}, {}
    for location in locations:
        if _source_hashes() != source_hashes:
            raise RuntimeError("functional-probe source drifted during execution")
        center = lock["patch_centers"][location]
        sham, probe, paired = _pair(S, state, center, lock["selected"]["amplitude"], prelock)
        summaries[location] = {
            "external_input_exact": sham["input_sha256"] == probe["input_sha256"],
            "external_input_sha256": sham["input_sha256"],
            "pulse_accounting": probe["pulse_accounting"],
            "max_active_fraction_1ms_sham": paired["max_active_fraction_1ms_sham"],
            "max_active_fraction_1ms_probe": paired["max_active_fraction_1ms_probe"],
            "excess_spikes": paired["excess_spikes"],
            "window_zero_crossings": paired["window_zero_crossings"],
            "latency_ms": paired["latency_ms"],
            "sham_final_state_hash": sham["final_state_hash"],
            "probe_final_state_hash": probe["final_state_hash"],
        }
        for key in (
            "delta_components", "delta_axis_components", "delta_axis_rate_hz",
            "delta_map_components", "delta_map_rate_hz", "delta_axis_signed_1ms",
        ):
            arrays[f"{location}__{key}"] = np.asarray(paired[key])
        arrays[f"{location}__axis_edges_mm"] = probe["readout"]["axis_edges_mm"]
        arrays[f"{location}__window_edges_ms"] = probe["readout"]["window_edges_ms"]
    summary = {
        "status": "COMPLETE", "condition": condition,
        "scientific_role": "descriptive_functional_geometry_not_trajectory_gate",
        "graph_sha256": graph_sha256(graph),
        "graph_construction_q": metadata["construction_q"],
        "manifest_sha256": _sha(manifest_path),
        "prelock_sha256": _sha(PRELOCK), "amplitude_lock_sha256": _sha(LOCK),
        "start_ms": lock["selected_start_ms"], "start_state_hash": state_hash(state),
        "locations": summaries,
        "arrays_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        "zero_crossing_is_a_gate": False,
        "source_sha256": source_hashes,
    }
    with AtomicStageBundle(arm) as bundle:
        _write_json(bundle.path("summary.json"), summary)
        NAT._npz_atomic(bundle.path("responses.npz"), **arrays)
        bundle.commit(required=["summary.json", "responses.npz"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lock", "run"))
    parser.add_argument("--condition", choices=GRAPH_IDS)
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A functional probe requires --confirm-run")
    if args.stage == "run" and args.condition is None:
        parser.error("run stage requires --condition")
    lock_name = ".functional_probe_lock.lock" if args.stage == "lock" else f".functional_probe_{args.condition}.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / lock_name).open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("requested functional-probe stage is already running") from exc
        label = "LOCK" if args.stage == "lock" else str(args.condition)
        sentinel_root = OUT if args.stage == "lock" else OUT / "functional_probes"
        sentinel_root.mkdir(parents=True, exist_ok=True)
        running = sentinel_root / f"RUNNING_LC6A_FUNCTIONAL_{label}.json"
        failed = sentinel_root / f"FAILED_LC6A_FUNCTIONAL_{label}.json"
        done = sentinel_root / f"DONE_LC6A_FUNCTIONAL_{label}.json"
        _write_json(running, {"status": "RUNNING", "pid": os.getpid(), "stage": args.stage, "condition": args.condition})
        try:
            if args.stage == "lock":
                result = lock_amplitude(args.execution_manifest)
            else:
                result = run_condition(args.condition, args.execution_manifest)
            _write_json(done, {
                "status": "DONE", "stage": args.stage, "condition": args.condition,
                "result": str(LOCK if args.stage == "lock" else OUT / f"functional_probes/{args.condition}/summary.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(NAT._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {
                "status": "FAILED", "stage": args.stage, "condition": args.condition,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
