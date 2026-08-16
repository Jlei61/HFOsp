#!/usr/bin/env python3
"""FCXR-LC6B round 3: cross D and H at the boundary to find out which one opens the bursting regime.

Round 2 located the entry between onset and onset+1 s, but D and the H gate move TOGETHER there
(D 0.0656 -> 0.0870, gate 0 -> 0.2696), so it could not say which one is responsible.  Denser
sampling along the natural path cannot fix that -- the two always move together on the path.  The
only way to separate them is to build slow fields that take D from one time and H from another.

Four cells, two shared initialisations each, one shared input stream:

    D11_H11   D from onset,      H from onset          (equals round 2's field_t11s)
    D12_H11   D from onset+1 s,  H from onset
    D11_H12   D from onset,      H from onset+1 s
    D12_H12   D from onset+1 s,  H from onset+1 s      (equals round 2's field_t12s)

The two diagonal cells are, by construction, round 2 points recomputed by a different code path, so
they are free correctness checks: their spikes must be bitwise identical to the atlas points.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
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

import run_topic4_fcxr_lc6b_atlas as ATLAS  # noqa: E402
import run_topic4_fcxr_lc6b_clamp_forks as CF  # noqa: E402
import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import AtomicStageBundle, SparseSpikeBinaryWriter  # noqa: E402
from src.topic4_fcxr_lc6_gain import active_area_mm2, binned_global_rate  # noqa: E402
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    NaturalCurrentObserver, cell_spatial_bins, per_second_cell_rates, spatial_rate_maps,
)
from src.topic4_fcxr_lc6b_clamp import (  # noqa: E402
    apply_slow_clamp, cell_rate_distribution, classify_clamp_window,
)


OUT = CF.OUT / "dh_cross"
#: (cell name) -> (field the z/D comes from, field the h/H comes from)
CELLS = {
    "D11_H11": ("field_t11s", "field_t11s"),
    "D12_H11": ("field_t12s", "field_t11s"),
    "D11_H12": ("field_t11s", "field_t12s"),
    "D12_H12": ("field_t12s", "field_t12s"),
}
#: Diagonal cells reproduce an atlas point by a different code path; identity is asserted at finalize.
DIAGONAL_EQUIVALENT = {"D11_H11": "field_t11s", "D12_H12": "field_t12s"}
INITS = ("locked_low", "locked_high")
OBSERVE_MS = ATLAS.OBSERVE_MS
MECHANISM_FILES = (
    Path(__file__).resolve(), Path(ATLAS.__file__).resolve(), Path(CF.__file__).resolve(),
    Path(NAT.__file__).resolve(), ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_statefork.py", ROOT / "src/topic4_fcxr_lc6b_clamp.py",
    ROOT / "src/topic4_fcxr_lc6_gain.py", ROOT / "src/topic4_fcxr_lc6_trajectory.py",
    ROOT / "src/snn_engine/mz_slow_vars.py",
)


def _source_hashes():
    return {str(p.relative_to(ROOT)): CF._sha(p) for p in MECHANISM_FILES}


def run_cell(cell, initialisation, *, observe_ms=OBSERVE_MS):
    manifest = CF._manifest()
    source_hashes = _source_hashes()
    index = ATLAS._fields_index()
    d_field_name, h_field_name = CELLS[cell]

    required = float(manifest["resource"]["required_mem_available_gib"])
    if NAT.U2.GEO._meminfo()["mem_available_gib"] < required:
        raise RuntimeError(f"LC6B cross requires at least {required} GiB MemAvailable")

    name = f"{cell}__{initialisation}"
    bundle_dir = OUT / name
    if bundle_dir.is_dir():
        return json.loads((bundle_dir / "summary.json").read_text())
    work = OUT / f".{name}.work"
    if work.exists():
        raise RuntimeError(f"stale work directory requires inspection: {work}")
    work.mkdir(parents=True)
    started = time.time()

    S, template, cfg, graph_sha, graph_meta = CF._system()
    ne = int(S["NE"])
    rng_state, xi, stream_record = ATLAS._canonical_stream(template)

    def _field(field_name):
        record = index["fields"][field_name]
        state = NAT.U2.load_into(str(ATLAS.FIELDS_DIR / record["file"]), template)
        if state_hash(state) != record["state_hash"]:
            raise RuntimeError(f"{field_name} state hash mismatch")
        return state, record

    d_state, d_record = _field(d_field_name)
    z_field = np.asarray(d_state.slow.z[:ne], float).copy()
    h_state, h_record = _field(h_field_name)
    h_field = np.asarray(h_state.slow.h_lc2_E, float).copy()

    origin = str(ATLAS.LOW_STATE if initialisation == "locked_low" else ATLAS.HIGH_STATE)
    state = NAT.U2.load_into(origin, template)
    fast_origin_hash = state_hash(state)
    state.slow.z[:ne] = z_field
    state.slow.h_lc2_E[:] = h_field
    state.rng_state = copy.deepcopy(rng_state)
    state.xi = xi
    child, clamp_record = apply_slow_clamp(state, clamp_d=True, clamp_h=True)
    if clamp_record["frozen_field_sha256"]["z"] != ATLAS._digest_array(z_field):
        raise RuntimeError("pinned z is not the requested D field")
    if clamp_record["frozen_field_sha256"]["h_lc2_E"] != ATLAS._digest_array(h_field):
        raise RuntimeError("pinned H is not the requested H field")

    bin_ms = float(manifest["observation"]["rate_bin_ms"])
    chunk_steps = int(round(1000.0 / NAT.U2.DT_MS))
    p = dataclasses.replace(S["p"], T=observe_ms, dt=NAT.U2.DT_MS)
    stream_hash = ATLAS.StreamHasher()
    observer = NaturalCurrentObserver(dt_ms=NAT.U2.DT_MS, sample_dt_ms=NAT.U2.TRACE_DT_MS)
    steps, cells_out, completed = [], [], 0
    numerical_fail, detail = False, None
    try:
        for chunk in range(int(round(observe_ms / 1000.0))):
            if _source_hashes() != source_hashes:
                raise RuntimeError("mechanism source drifted during an LC6B cross cell")
            writer = SparseSpikeBinaryWriter(
                work / f"c{chunk:02d}.bin", step_origin=child.t, n_steps=chunk_steps, n_cells=ne)
            try:
                out = run_fcxr_loop(
                    p, S["net"], start=child, n_steps=chunk_steps, capture_final=True,
                    store_spikes=False, spike_sink=writer, input_sink=stream_hash,
                    membrane_term_sink=observer.sample, v_th_per_neuron=S["vth"])
                sparse = writer.finalize(work / f"c{chunk:02d}_spikes.npz")
            finally:
                writer.close()
                (work / f"c{chunk:02d}.bin").unlink(missing_ok=True)
            child = out["checkpoint"]
            steps.append(np.asarray(sparse.steps, np.int64) + chunk * chunk_steps)
            cells_out.append(np.asarray(sparse.cells, np.int32))
            completed += chunk_steps
    except FloatingPointError as exc:
        numerical_fail, detail = True, f"{type(exc).__name__}: {exc}"

    all_steps = np.concatenate(steps) if steps else np.zeros(0, np.int64)
    all_cells = np.concatenate(cells_out) if cells_out else np.zeros(0, np.int32)
    completed_ms = completed * NAT.U2.DT_MS
    rate_bins = binned_global_rate(all_steps, n_steps=completed, n_cells=ne,
                                  dt_ms=NAT.U2.DT_MS, bin_ms=bin_ms) if completed else np.zeros(0)
    cell_rates = per_second_cell_rates(all_steps, all_cells, n_steps=completed, n_cells=ne,
                                       dt_ms=NAT.U2.DT_MS) if completed else np.zeros((0, ne))
    bins, occupancy = cell_spatial_bins(S["posE"], sheet_size_mm=S["L"], n_bins_axis=32)
    maps = spatial_rate_maps(all_steps, all_cells, bins, occupancy, n_steps=completed,
                             dt_ms=NAT.U2.DT_MS, window_ms=100.0) if completed else np.zeros((0, occupancy.size))
    local = json.loads(CF.LOCAL_LOCK.read_text())["thresholds"]
    area = active_area_mm2(maps, occupancy, rate_threshold_hz=local["rate_threshold_hz"],
                          sheet_size_mm=S["L"]) if maps.size else np.zeros(0)
    th = manifest["classifier"]["thresholds"]
    verdict = classify_clamp_window(
        rate_bins_hz=rate_bins, cell_rates_hz=cell_rates, completed_ms=completed_ms,
        registered_ms=observe_ms, numerical_fail=numerical_fail, bin_ms=bin_ms,
        **{k: th[k] for k in ("global_saturation_hz", "refractory_ceiling_hz",
                              "near_refractory_fraction_gate", "interictal_roll_hi_hz",
                              "drift_ci_gate_per_s", "silence_bin_fraction_gate", "tail_s")})
    summary = {
        "status": "COMPLETE", "cell_id": name, "cell": cell, "initialisation": initialisation,
        "d_field": d_field_name, "h_field": h_field_name,
        "d_field_D_mean": d_record["D_mean"], "h_field_h_gate_mean": h_record["h_gate_mean"],
        "d_field_relative_to_onset_ms": d_record["relative_to_onset_ms"],
        "h_field_relative_to_onset_ms": h_record["relative_to_onset_ms"],
        "initialisation_origin": origin, "fast_origin_state_hash": fast_origin_hash,
        "clamp": clamp_record, "numerical_fail": numerical_fail, "numerical_detail": detail,
        "observe_ms": observe_ms, "completed_ms": completed_ms,
        "canonical_stream_source": stream_record["name"],
        "future_input_content_sha256": stream_hash.sha256,
        "verdict": verdict,
        "cell_rate_distribution": cell_rate_distribution(
            cell_rates, refractory_ceiling_hz=th["refractory_ceiling_hz"]) if cell_rates.size else None,
        "active_area_mm2": area.tolist(),
        "median_active_area_mm2": float(np.median(area)) if area.size else None,
        "sheet_area_mm2": float(S["L"]) ** 2,
        "graph_sha256": graph_sha, "graph_construction_q": graph_meta["construction_q"],
        "diagonal_equivalent_atlas_field": DIAGONAL_EQUIVALENT.get(cell),
        "termination_tested": False, "lifecycle_tested": False,
        "perturbation_return_tested": False,
        "source_sha256": source_hashes, "wall_s": time.time() - started,
    }
    with AtomicStageBundle(bundle_dir) as bundle:
        NAT._write_json(bundle.path("summary.json"), summary)
        NAT._npz_atomic(bundle.path("spikes.npz"), steps=all_steps, cells=all_cells,
                        n_steps=np.asarray([completed], np.int64), n_cells=np.asarray([ne], np.int64))
        NAT._npz_atomic(bundle.path("traces.npz"), rate_bin_ms=np.asarray([bin_ms], np.float32),
                        rate_bins_hz=rate_bins.astype(np.float32),
                        active_area_mm2=area.astype(np.float32),
                        per_second_mean_hz=np.asarray(verdict["per_second_mean_hz"], np.float32),
                        **observer.arrays())
        NAT.U2.save_loop_state(str(bundle.path("final_state.npz")), child)
        bundle.commit(required=["summary.json", "spikes.npz", "traces.npz", "final_state.npz"])
    shutil.rmtree(work, ignore_errors=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run",))
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    parser.add_argument("--init", choices=INITS, required=True)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B D-H cross requires --confirm-run")
    OUT.mkdir(parents=True, exist_ok=True)
    label = f"{args.cell}__{args.init}"
    with (CF.OUT / f".cross_{label}.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC6B cross {label} is already running") from exc
        running = CF.OUT / f"RUNNING_CROSS_{label}.json"
        NAT._write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            result = run_cell(args.cell, args.init)
            NAT._write_json(CF.OUT / f"DONE_CROSS_{label}.json", {
                "status": "DONE", "cell": result["cell_id"],
                "label": result["verdict"]["label"], "reason": result["verdict"]["reason"]})
            (CF.OUT / f"FAILED_CROSS_{label}.json").unlink(missing_ok=True)
            print(json.dumps({"cell": result["cell_id"], "label": result["verdict"]["label"],
                              "per_second_mean_hz": result["verdict"]["per_second_mean_hz"]},
                             indent=2))
        except BaseException as exc:
            NAT._write_json(CF.OUT / f"FAILED_CROSS_{label}.json",
                            {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
