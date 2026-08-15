#!/usr/bin/env python3
"""Regenerate the six natural (D, H) fields the LC6B atlas walks along.

LC6A pinned exact states only at 12 / 13 / 15 / 17 / 18 s, but the atlas needs onset-1 s through
onset+4 s, i.e. 10 through 15 s.  The missing times cannot be recovered from the saved coarse field
maps -- those are 32x32 bin means, not the per-cell fields a clamp needs -- so the canonical C0 run is
reproduced from t=0 and an exact state is pinned every second from 10 s.

The reproduction is checked, not assumed: every completed second's population rate must equal the
value the original C0 trajectory recorded for that second.
"""
from __future__ import annotations

import argparse
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

import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
import run_topic4_fcxr_lc6b_clamp_forks as CF  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import ExactInputHasher, SparseSpikeBinaryWriter  # noqa: E402


OUT = CF.OUT / "path_fields"
PIN_SECONDS = (10, 11, 12, 13, 14, 15)
END_S = 15


def run():
    manifest = CF._manifest()
    onset_ms = float(manifest["source_snapshots"]["S2"]["onset_time_ms"])
    reference = json.loads(
        (NAT.OUT / "trajectories/C0/summary.json").read_text())["per_second_mean_rate_hz"]
    if OUT.is_dir():
        return json.loads((OUT / "index.json").read_text())
    work = CF.FORK_ROOT.parent / ".path_fields.work"
    if work.exists():
        raise RuntimeError(f"stale work directory requires inspection: {work}")
    work.mkdir(parents=True)
    started = time.time()

    _path, _payload, source_summary = NAT._validate_manifest(CF.LC6A_MANIFEST, "C0")
    graph, _meta = NAT._load_graph(NAT.OUT / "graphs/C0.npz")
    graph_sha = NAT.graph_sha256(graph)
    S, slow, cfg = NAT._fresh_system(
        json.loads(source_summary.read_text()), graph, graph_sha, "C0")
    p = dataclasses.replace(S["p"], T=float(END_S) * 1000.0, dt=NAT.U2.DT_MS)
    # t=0 must be materialised exactly the way the natural runner does it: a fresh run consumes two
    # registered recorder-selection draws before its first membrane step, and _seed_template would
    # instead advance a full step, making this a shifted 1..N continuation of a different trajectory.
    state = NAT._fresh_initial_state(S, slow, p)
    hasher = ExactInputHasher()
    chunk_steps = int(round(1000.0 / NAT.U2.DT_MS))
    rows, pinned = [], {}
    for second in range(END_S):
        writer = SparseSpikeBinaryWriter(
            work / f"s{second:02d}.bin", step_origin=state.t, n_steps=chunk_steps, n_cells=S["NE"])
        try:
            out = run_fcxr_loop(
                p, S["net"], start=state, n_steps=chunk_steps, capture_final=True,
                store_spikes=False, spike_sink=writer, input_sink=hasher,
                v_th_per_neuron=S["vth"])
            stream = writer.finalize(work / f"s{second:02d}_spikes.npz")
        finally:
            writer.close()
            (work / f"s{second:02d}.bin").unlink(missing_ok=True)
        state = out["checkpoint"]
        rate = stream.steps.size / S["NE"]
        expected = float(reference[second])
        if abs(rate - expected) > 1e-6:
            raise RuntimeError(
                f"C0 reproduction diverged at second {second}: {rate} != {expected}")
        (work / f"s{second:02d}_spikes.npz").unlink(missing_ok=True)
        rows.append({"second": second, "mean_rate_hz": rate, "expected_hz": expected})
        end_s = second + 1
        if end_s in PIN_SECONDS:
            name = f"field_t{end_s:02d}s"
            path = work / f"{name}.npz"
            NAT.U2.save_loop_state(str(path), state)
            ne = int(state.slow.NE)
            gate = CF.lc2_h_gate(state.slow.h_lc2_E, theta=cfg["theta_h_lc2"], k=cfg["k_h_lc2"])
            pinned[name] = {
                "name": name, "file": path.name,
                "snapshot_time_ms": float(state.t) * NAT.U2.DT_MS,
                "onset_time_ms": onset_ms,
                "relative_to_onset_ms": float(state.t) * NAT.U2.DT_MS - onset_ms,
                "t_steps": int(state.t), "state_hash": state_hash(state),
                "external_input_counter": int(state.t),
                "external_input_state": CF._digest(CF._state_jsonable(state.rng_state)),
                "external_input_sha256": hasher.sha256,
                "graph_sha256": graph_sha,
                "config_sha256": CF._digest({"manifest_sha256": CF._sha(CF.MANIFEST)}),
                "D_mean": float(np.mean(1.0 - np.asarray(state.slow.z[:ne], float))),
                "H_mean": float(state.slow.h_lc2_E.mean()),
                "h_gate_mean": float(gate.mean()),
                "preceding_1s_global_rate_hz": rate,
            }
    index = {
        "status": "COMPLETE", "stage": "LC6B_PATH_FIELDS",
        "graph_sha256": graph_sha, "onset_time_ms": onset_ms,
        "reproduction_of_c0_exact_per_second": rows,
        "fields": pinned, "wall_s": time.time() - started,
        "source_sha256": CF._source_hashes(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for record in pinned.values():
        shutil.move(str(work / record["file"]), str(OUT / record["file"]))
        record["sha256"] = CF._sha(OUT / record["file"])
    NAT._write_json(OUT / "index.json", index)
    shutil.rmtree(work, ignore_errors=True)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B path fields require --confirm-run")
    CF.OUT.mkdir(parents=True, exist_ok=True)
    with (CF.OUT / ".path_fields.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("LC6B path fields are already running") from exc
        running = CF.OUT / "RUNNING_PATH_FIELDS.json"
        NAT._write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            index = run()
            NAT._write_json(CF.OUT / "DONE_PATH_FIELDS.json", {
                "status": "DONE", "n_fields": len(index["fields"]),
                "fields": sorted(index["fields"]),
            })
            print(json.dumps({k: {kk: v[kk] for kk in
                                  ("snapshot_time_ms", "relative_to_onset_ms", "D_mean",
                                   "h_gate_mean", "preceding_1s_global_rate_hz")}
                              for k, v in index["fields"].items()}, indent=2, sort_keys=True))
        except BaseException as exc:
            NAT._write_json(CF.OUT / "FAILED_PATH_FIELDS.json",
                            {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
