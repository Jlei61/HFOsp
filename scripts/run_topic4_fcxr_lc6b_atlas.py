#!/usr/bin/env python3
"""FCXR-LC6B natural-path atlas: walk the trajectory's own (D, H) fields, three initialisations each.

Round 1 showed that pinning D at 13 s or 15 s replaces whole-sheet saturation with a wide-field
re-ignition train.  This asks WHERE along the natural path that continuation exists, and whether the
tissue's final state depends on where it started from within the same pinned field.

Four locks carried from the 2026-08-16 review:
  1. every run observes at least 10 s; a 6 s label is never a verdict here
  2. every run shares one canonical future-input stream, so a difference between initialisations can
     only come from the initialisation
  3. 20 ms population rate AND active area are recorded, never just the one-second mean
  4. if the low and high initialisations settle on different labels the pair is reported as a
     bistability CANDIDATE, pending a perturbation-return test and a second noise stream

Lock 2 needs care: ``ExactInputHasher`` folds the absolute step index into its digest, and the three
initialisations legitimately carry different step counters, so their digests differ even when the
draws are identical.  The stream digest below hashes only the OU value and the per-cell Poisson draw
-- the input content itself -- which is what the lock is actually about.
"""
from __future__ import annotations

import argparse
import copy
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
import run_topic4_fcxr_lc6b_clamp_forks as CF  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle, ExactInputHasher, SparseSpikeBinaryWriter,
)
from src.topic4_fcxr_lc6_gain import active_area_mm2, binned_global_rate  # noqa: E402
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    NaturalCurrentObserver, cell_spatial_bins, per_second_cell_rates, spatial_rate_maps,
)
from src.topic4_fcxr_lc6b_clamp import (  # noqa: E402
    apply_slow_clamp, cell_rate_distribution, classify_clamp_window,
)


OUT = CF.OUT / "atlas"
FIELDS_DIR = CF.OUT / "path_fields"
LOW_STATE = NAT.OUT / "functional_probe_lock_state.npz"
HIGH_STATE = NAT.OUT / "trajectories/C0/final_state.npz"
CANONICAL_STREAM_FIELD = "field_t13s"          # registered: same snapshot round 1 forked from
OBSERVE_MS = 10000.0
INITIALISATIONS = ("path_native", "locked_low", "locked_high")
MECHANISM_FILES = (
    Path(__file__).resolve(), Path(CF.__file__).resolve(), Path(NAT.__file__).resolve(),
    ROOT / "src/topic4_fcxr_lc3.py", ROOT / "src/topic4_fcxr_lc3_statefork.py",
    ROOT / "src/topic4_fcxr_lc6b_clamp.py", ROOT / "src/topic4_fcxr_lc6_gain.py",
    ROOT / "src/topic4_fcxr_lc6_trajectory.py", ROOT / "src/snn_engine/mz_slow_vars.py",
)


def _source_hashes():
    return {str(p.relative_to(ROOT)): CF._sha(p) for p in MECHANISM_FILES}


class StreamHasher:
    """Digest of the input CONTENT only: the OU value and the per-cell Poisson draw, no step index."""

    def __init__(self):
        self._h = hashlib.sha256()
        self.n_steps = 0

    def __call__(self, absolute_step, xi, external_counts):
        self._h.update(np.asarray([float(xi)], np.float64).tobytes())
        self._h.update(np.ascontiguousarray(np.asarray(external_counts, np.float64)).tobytes())
        self.n_steps += 1

    @property
    def sha256(self):
        return self._h.hexdigest()


def _fields_index():
    index = json.loads((FIELDS_DIR / "index.json").read_text())
    for record in index["fields"].values():
        if CF._sha(FIELDS_DIR / record["file"]) != record["sha256"]:
            raise RuntimeError(f"path-field artifact drift: {record['file']}")
    return index


def _canonical_stream(template):
    """The registered future-input stream: the generator state and OU value carried by field_t13s."""

    index = _fields_index()
    record = index["fields"][CANONICAL_STREAM_FIELD]
    state = NAT.U2.load_into(str(FIELDS_DIR / record["file"]), template)
    if state_hash(state) != record["state_hash"]:
        raise RuntimeError("canonical stream source state hash mismatch")
    return state.rng_state, float(state.xi), record


def run_point(field_name, initialisation, *, observe_ms=OBSERVE_MS):
    manifest = CF._manifest()
    source_hashes = _source_hashes()
    index = _fields_index()
    if field_name not in index["fields"]:
        raise ValueError(f"unknown field {field_name}")
    if initialisation not in INITIALISATIONS:
        raise ValueError(f"unknown initialisation {initialisation}")
    field = index["fields"][field_name]

    required = float(manifest["resource"]["required_mem_available_gib"])
    if NAT.U2.GEO._meminfo()["mem_available_gib"] < required:
        raise RuntimeError(f"LC6B atlas requires at least {required} GiB MemAvailable")

    name = f"{field_name}__{initialisation}"
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
    rng_state, xi, stream_record = _canonical_stream(template)

    # 1. the slow field this atlas point is about
    field_state = NAT.U2.load_into(str(FIELDS_DIR / field["file"]), template)
    if state_hash(field_state) != field["state_hash"]:
        raise RuntimeError("field state hash mismatch")
    z_field = np.asarray(field_state.slow.z[:ne], float).copy()
    h_field = np.asarray(field_state.slow.h_lc2_E, float).copy()

    # 2. the fast state it is applied to
    if initialisation == "path_native":
        state, origin = field_state, str(FIELDS_DIR / field["file"])
    else:
        origin = str(LOW_STATE if initialisation == "locked_low" else HIGH_STATE)
        state = NAT.U2.load_into(origin, template)
    fast_origin_hash = state_hash(state)

    # 3. install the field, then the canonical stream, then pin both
    state.slow.z[:ne] = z_field
    state.slow.h_lc2_E[:] = h_field
    # Lock 2.  The noise path draws exactly one OU normal and one per-cell Poisson vector per step and
    # reads no neural state, so installing the same generator state AND the same OU value makes the
    # whole future input bit-identical across initialisations.  xi alone would not be enough: it sets
    # the Poisson rate, so a different xi changes the draw even from an identical generator.
    state.rng_state = copy.deepcopy(rng_state)
    state.xi = xi
    child, clamp_record = apply_slow_clamp(state, clamp_d=True, clamp_h=True)
    if clamp_record["frozen_field_sha256"]["z"] != _digest_array(z_field):
        raise RuntimeError("pinned z is not the atlas point's field")
    if clamp_record["frozen_field_sha256"]["h_lc2_E"] != _digest_array(h_field):
        raise RuntimeError("pinned H is not the atlas point's field")

    # 4. observe
    bin_ms = float(manifest["observation"]["rate_bin_ms"])
    chunk_steps = int(round(1000.0 / NAT.U2.DT_MS))
    n_chunks = int(round(observe_ms / 1000.0))
    p = dataclasses.replace(S["p"], T=observe_ms, dt=NAT.U2.DT_MS)
    stream_hash, step_hash = StreamHasher(), ExactInputHasher()
    observer = NaturalCurrentObserver(dt_ms=NAT.U2.DT_MS, sample_dt_ms=NAT.U2.TRACE_DT_MS)

    def _sink(step, xi_now, ext):
        stream_hash(step, xi_now, ext)
        step_hash(step, xi_now, ext)

    steps, cells = [], []
    completed = 0
    numerical_fail, detail = False, None
    try:
        for chunk in range(n_chunks):
            if _source_hashes() != source_hashes:
                raise RuntimeError("mechanism source drifted during an LC6B atlas point")
            writer = SparseSpikeBinaryWriter(
                work / f"c{chunk:02d}.bin", step_origin=child.t, n_steps=chunk_steps, n_cells=ne)
            try:
                out = run_fcxr_loop(
                    p, S["net"], start=child, n_steps=chunk_steps, capture_final=True,
                    store_spikes=False, spike_sink=writer, input_sink=_sink,
                    membrane_term_sink=observer.sample, v_th_per_neuron=S["vth"])
                sparse = writer.finalize(work / f"c{chunk:02d}_spikes.npz")
            finally:
                writer.close()
                (work / f"c{chunk:02d}.bin").unlink(missing_ok=True)
            child = out["checkpoint"]
            steps.append(np.asarray(sparse.steps, np.int64) + chunk * chunk_steps)
            cells.append(np.asarray(sparse.cells, np.int32))
            completed += chunk_steps
            NAT._write_json(work / "progress.json", {
                "status": "RUNNING", "point": name, "completed_ms": completed * NAT.U2.DT_MS})
    except FloatingPointError as exc:
        numerical_fail, detail = True, f"{type(exc).__name__}: {exc}"

    all_steps = np.concatenate(steps) if steps else np.zeros(0, np.int64)
    all_cells = np.concatenate(cells) if cells else np.zeros(0, np.int32)
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
        "status": "COMPLETE", "point_id": name, "field": field_name,
        "initialisation": initialisation, "initialisation_origin": origin,
        "fast_origin_state_hash": fast_origin_hash,
        "field_snapshot_time_ms": field["snapshot_time_ms"],
        "field_relative_to_onset_ms": field["relative_to_onset_ms"],
        "field_D_mean": field["D_mean"], "field_H_mean": field["H_mean"],
        "field_h_gate_mean": field["h_gate_mean"],
        "clamp": clamp_record, "numerical_fail": numerical_fail, "numerical_detail": detail,
        "observe_ms": observe_ms, "completed_ms": completed_ms,
        "canonical_stream_source": stream_record["name"],
        "future_input_content_sha256": stream_hash.sha256,
        "future_input_with_step_index_sha256": step_hash.sha256,
        "verdict": verdict,
        "cell_rate_distribution": cell_rate_distribution(
            cell_rates, refractory_ceiling_hz=th["refractory_ceiling_hz"]) if cell_rates.size else None,
        "active_area_mm2": area.tolist(),
        "median_active_area_mm2": float(np.median(area)) if area.size else None,
        "max_active_area_mm2": float(area.max()) if area.size else None,
        "sheet_area_mm2": float(S["L"]) ** 2,
        "graph_sha256": graph_sha, "graph_construction_q": graph_meta["construction_q"],
        "manifest_sha256": CF._sha(CF.MANIFEST),
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


def _digest_array(values):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, float)).tobytes()).hexdigest()


def finalize():
    index = _fields_index()
    fields = sorted(index["fields"], key=lambda k: index["fields"][k]["snapshot_time_ms"])
    rows, missing = {}, []
    for field in fields:
        for init in INITIALISATIONS:
            path = OUT / f"{field}__{init}" / "summary.json"
            if not path.is_file():
                missing.append(f"{field}__{init}")
                continue
            rows[(field, init)] = json.loads(path.read_text())
    if missing:
        raise RuntimeError(f"atlas incomplete: {missing}")

    digests = {key: row["future_input_content_sha256"] for key, row in rows.items()}
    shared = len(set(digests.values())) == 1
    if not shared:
        raise RuntimeError("lock 2: atlas points do not share one future-input stream")

    per_field = {}
    for field in fields:
        labels = {init: rows[(field, init)]["verdict"]["label"] for init in INITIALISATIONS}
        low, high = labels["locked_low"], labels["locked_high"]
        per_field[field] = {
            "snapshot_time_ms": index["fields"][field]["snapshot_time_ms"],
            "relative_to_onset_ms": index["fields"][field]["relative_to_onset_ms"],
            "D_mean": index["fields"][field]["D_mean"],
            "H_mean": index["fields"][field]["H_mean"],
            "h_gate_mean": index["fields"][field]["h_gate_mean"],
            "labels": labels,
            "plateau_hz": {init: rows[(field, init)]["verdict"]["per_second_mean_hz"][-1]
                           for init in INITIALISATIONS},
            "median_active_area_mm2": {init: rows[(field, init)]["median_active_area_mm2"]
                                       for init in INITIALISATIONS},
            # Lock 4: a split between the two locked initialisations is a CANDIDATE only.  Confirming
            # it needs a perturbation-return test and a second noise stream, neither of which is run
            # here, so the word "bistability" never appears without "candidate".
            "initialisation_split": low != high,
            "verdict": ("BISTABILITY_CANDIDATE_PENDING_PERTURBATION_AND_SECOND_STREAM"
                        if low != high else "SINGLE_OUTCOME_FROM_BOTH_LOCKED_INITIALISATIONS"),
        }
    # Cross-check against round 1, in two different senses.
    #
    # field_t13s is the IDENTITY check: the canonical stream is taken from that field, so its
    # path_native point receives exactly the input round 1 received and must reproduce it second by
    # second.  A mismatch means the regenerated path fields are not the trajectory round 1 forked
    # from, and publishing would build an atlas on a different trajectory -- so it raises.
    #
    # field_t15s cannot be an identity check and must not be asserted as one: it is handed
    # field_t13s's generator state, while round 1 used the 15 s state's own.  That makes it something
    # more useful instead -- the round-1 S4 arm re-run under a DIFFERENT noise stream -- so it is
    # reported as a replication readout, which is a partial answer to the review's request for a
    # second noise stream.
    identity = {"field_t13s": "S2_DH_CLAMP_EXT"}
    replication = {"field_t15s": "S4_DH_CLAMP_EXT"}
    cross = {}

    def _round1(arm_id):
        path = CF.FORK_ROOT / arm_id / "summary.json"
        return json.loads(path.read_text()) if path.is_file() else None

    for field, arm_id in identity.items():
        parent = _round1(arm_id)
        if field not in fields or parent is None:
            continue
        want = [round(x, 6) for x in parent["verdict"]["per_second_mean_hz"]]
        got = [round(x, 6) for x in rows[(field, "path_native")]["verdict"]["per_second_mean_hz"]]
        n = min(len(want), len(got))
        cross[field] = {"kind": "identity", "round1_arm": arm_id, "n_compared_seconds": n,
                        "identical": want[:n] == got[:n],
                        "atlas_per_second_hz": got[:n], "round1_per_second_hz": want[:n]}
        if not cross[field]["identical"]:
            raise RuntimeError(
                f"{field}__path_native does not reproduce round 1 arm {arm_id}; the regenerated "
                "path fields are not the trajectory round 1 forked from")

    for field, arm_id in replication.items():
        parent = _round1(arm_id)
        if field not in fields or parent is None:
            continue
        atlas_row = rows[(field, "path_native")]
        want_rate = float(parent["verdict"]["per_second_mean_hz"][-1])
        got_rate = float(atlas_row["verdict"]["per_second_mean_hz"][-1])
        scale = max(abs(want_rate), abs(got_rate), 1e-9)
        cross[field] = {
            "kind": "different_noise_stream_replication",
            "round1_arm": arm_id,
            "note": ("this point runs the canonical stream taken from field_t13s, not the 15 s "
                     "state's own, so it is round 1's S4 arm under a different noise stream; "
                     "identity is not expected and is not asserted"),
            "round1_label": parent["verdict"]["label"],
            "atlas_label": atlas_row["verdict"]["label"],
            "round1_final_second_hz": want_rate, "atlas_final_second_hz": got_rate,
            "final_second_relative_difference": abs(want_rate - got_rate) / scale,
            "round1_median_active_area_mm2": None,
            "atlas_median_active_area_mm2": atlas_row["median_active_area_mm2"],
            "same_regime": (parent["verdict"]["bounded_candidate"]
                            == atlas_row["verdict"]["bounded_candidate"]),
        }

    payload = {
        "status": "COMPLETE", "n_points": len(rows),
        "round1_cross_check": cross,
        "fields_in_time_order": fields,
        "initialisations": list(INITIALISATIONS),
        "observe_ms": OBSERVE_MS,
        "lock_1_min_observation_ms": OBSERVE_MS,
        "lock_2_all_points_share_one_future_input_stream": shared,
        "lock_2_stream_content_sha256": next(iter(digests.values())),
        "lock_3_recorded": ["20 ms population rate", "100 ms active area",
                            "per-cell rate distribution"],
        "lock_4_split_handling": "reported as BISTABILITY_CANDIDATE, never as demonstrated bistability",
        "per_field": per_field,
        "rows": {f"{k[0]}__{k[1]}": v for k, v in rows.items()},
        "termination_tested": False, "lifecycle_tested": False,
        "perturbation_return_tested": False,
        "claim_boundary": (
            "One canonical-seed C0 graph, one noise stream, one trajectory.  Each label describes a "
            "10 s continuation under a slow field pinned at the value that trajectory reached; a "
            "bounded label does not state that a weak perturbation returns, and the bounded state is "
            "a wide-field re-ignition train, not a demonstrated seizure carrier."),
        "source_sha256": _source_hashes(),
    }
    NAT._write_json(CF.OUT / "natural_path_atlas.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "finalize"))
    parser.add_argument("--field")
    parser.add_argument("--init", choices=INITIALISATIONS)
    parser.add_argument("--observe-ms", type=float, default=OBSERVE_MS)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B atlas requires --confirm-run")
    if args.stage == "run" and not (args.field and args.init):
        parser.error("run requires --field and --init")
    OUT.mkdir(parents=True, exist_ok=True)
    label = "FINALIZE" if args.stage == "finalize" else f"{args.field}__{args.init}"
    with (CF.OUT / f".atlas_{label}.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC6B atlas {label} is already running") from exc
        running = CF.OUT / f"RUNNING_ATLAS_{label}.json"
        NAT._write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            if args.stage == "run":
                result = run_point(args.field, args.init, observe_ms=args.observe_ms)
                note = {"point": result["point_id"], "label": result["verdict"]["label"],
                        "reason": result["verdict"]["reason"]}
            else:
                result = finalize()
                note = {"n_points": result["n_points"],
                        "per_field": {k: v["verdict"] for k, v in result["per_field"].items()}}
            NAT._write_json(CF.OUT / f"DONE_ATLAS_{label}.json", {"status": "DONE", **note})
            (CF.OUT / f"FAILED_ATLAS_{label}.json").unlink(missing_ok=True)
            print(json.dumps(NAT._jsonable(note), indent=2, sort_keys=True))
        except BaseException as exc:
            NAT._write_json(CF.OUT / f"FAILED_ATLAS_{label}.json",
                            {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
