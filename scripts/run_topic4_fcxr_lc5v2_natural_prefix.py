#!/usr/bin/env python3
"""Continuous-from-baseline LC5v2 prefix adjudication.

Unlike the U2 onset fork, this runner never switches an accumulated actuator on at onset.  The
per-cell load starts at zero and both sensing and membrane current are online from the first step.
The Gamma=0 arm is an exact transaction control against the accepted U1 trajectory.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5 as LC5  # noqa: E402
import run_topic4_fcxr_lc5v2_u2 as U2  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle,
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    SparseSpikeStream,
    json_sanitize,
    load_sparse_spike_stream,
)


RUN_MS = 18000.0
N_CHUNKS = int(round(RUN_MS / U2.CHUNK_MS))
MECHANISM_FILES = U2.MECHANISM_FILES + (Path(__file__),)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tag(gamma):
    milli = int(round(float(gamma) * 1000.0))
    if not np.isclose(milli / 1000.0, float(gamma), rtol=0.0, atol=1e-12):
        raise ValueError("natural-prefix Gamma must resolve exactly in milli-Gamma units")
    return f"u3_prefix_tau8_gamma_milli{milli:03d}"


def _paths(gamma):
    tag = _tag(gamma)
    return U2.OUT / tag, U2.OUT / f".{tag}.work"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_sanitize(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _stage_lock(gamma):
    U2.OUT.mkdir(parents=True, exist_ok=True)
    f = (U2.OUT / f".{_tag(gamma)}.lock").open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        f.close()
        raise SystemExit(f"{_tag(gamma)} is already running") from exc
    return f


def _fresh_system(p0, imax, a_load):
    candidate = LC5._load_candidate()
    S = U2.PP.build_substrate(U2.CONNECTION_SEED)
    U2.install_registered_noise_rng(S["net"])
    cfg_dict = LC5._config(S, candidate)
    cfg_dict.update(
        use_pump=True,
        pump_sensor_only=False,
        pump_a_load=float(a_load),
        pump_tau_ms=U2.TAU_MS,
        pump_Imax=float(imax),
        pump_h=3,
        pump_excess_mode="rectified_excess",
        pump_p0_E=np.asarray(p0, float).copy(),
        pump_u_init_E=np.zeros(int(S["NE"]), dtype=float),
        pump_record_calibration=False,
    )
    slow = MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg_dict), NE=S["NE"],
        core_mask_E=U2.OLD_SLOW.build_core_masks(S),
    )
    S["net"]["rng"] = np.random.default_rng(U2.NOISE_SEED)
    return S, slow, cfg_dict


def _reference_prefix():
    stream = load_sparse_spike_stream(U2.SOURCE / "u1_sparse_spikes.npz")
    n_steps = int(round(RUN_MS / U2.DT_MS))
    keep = stream.steps < n_steps
    short = SparseSpikeStream(stream.steps[keep], stream.cells[keep], n_steps, stream.n_cells)
    with np.load(U2.SOURCE / "u1_capture_traces.npz", allow_pickle=False) as z:
        rate = np.asarray(z["rate_E"], np.float32)[: int(round(RUN_MS / 10.0))]
    return short, rate


def _outcome(lifecycle, regimes, saturated):
    bout = LC5.LC4.first_ictal_bout(regimes, float(U2._baseline()["band"]["win_ms"]))
    if bout is None:
        return "NO_NATURAL_ONSET", None, None
    win = float(U2._baseline()["band"]["win_ms"])
    onset = bout[0] * win
    offset = (bout[1] + 1) * win
    terminated = bout[1] + 1 < len(regimes) and regimes[bout[1] + 1] != "ICTAL"
    if not terminated:
        return ("ESCALATING_SATURATION" if saturated else "SUSTAINED_HIGH_NO_OFFSET"), onset, None
    duration = offset - onset
    tail = RUN_MS - offset
    if 1000.0 <= duration <= 5000.0 and tail >= 2000.0 and lifecycle["label"] != "RAPID_RELAPSE":
        return "FINITE_EXCURSION_CANDIDATE", onset, offset
    return "OFFSET_OUTSIDE_TARGET", onset, offset


def stage_prefix(gamma):
    arm, work = _paths(gamma)
    tag = _tag(gamma)
    sentinel = tag.upper()
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    lock = _stage_lock(gamma)
    running = U2.OUT / f"RUNNING_{sentinel}.json"
    try:
        prelock, p0, _unused_u, imax = U2._load_contract(gamma)
        if work.exists():
            raise SystemExit(f"stale work directory requires inspection: {work}")
        resources = U2.GEO._meminfo()
        if resources["mem_available_gib"] < 128.0:
            raise SystemExit("natural prefix requires at least 128 GiB MemAvailable")
        baseline_swap = float(resources["swap_used_mib"])
        work.mkdir(parents=True)
        started = time.time()
        _write_json(running, {
            "status": "RUNNING", "pid": os.getpid(), "started": U2.GEO._now(),
            "gamma": float(gamma), "Imax": float(imax), "T_ms": RUN_MS,
            "semantics": "fresh_t0_u0_always_online_no_step",
        })
        (U2.OUT / f"{tag}.pid").write_text(f"{os.getpid()}\n")
        U2._resource_row(f"{sentinel}_PREFLIGHT", baseline_swap)

        S, slow, cfg_dict = _fresh_system(p0, imax, prelock["a_load"])
        stride = int(round(U2.TRACE_DT_MS / U2.DT_MS))
        force_scale = float(slow.cfg.E_E - slow.cfg.v_match)
        slow.recurrent_drive_observer = RecurrentDriveBlockObserver(
            S["NE"], sample_every=stride,
            steps_per_block=int(round(1000.0 / U2.DT_MS)), force_scale=force_scale,
        )
        input_hasher = ExactInputHasher()
        p = dataclasses.replace(S["p"], T=RUN_MS, dt=U2.DT_MS)
        chunk_steps = int(round(U2.CHUNK_MS / U2.DT_MS))
        state = None
        rate_parts, trace_parts, streams = [], {}, []

        for chunk in range(N_CHUNKS):
            active_slow = slow if state is None else state.slow
            attrs = (
                "trace_z_mean", "trace_h_lc2_mean", "trace_gA_raw_lc2_mean",
                "trace_gErec_mean", "trace_u_mean", "trace_u_max",
                "trace_phi_pump_mean", "trace_pump_excess_mean", "trace_pump_excess_max",
                "trace_conductance_clip_frac",
            )
            starts = {name: len(getattr(active_slow, name)) for name in attrs}
            binary = work / f"chunk_{chunk:02d}.bin"
            writer = SparseSpikeBinaryWriter(
                binary, step_origin=0 if state is None else state.t,
                n_steps=chunk_steps, n_cells=S["NE"],
            )
            kwargs = dict(
                n_steps=chunk_steps, capture_final=True, store_spikes=False,
                spike_sink=writer, input_sink=input_hasher, v_th_per_neuron=S["vth"],
            )
            run = run_fcxr_loop(p, S["net"], slow=slow, **kwargs) if state is None else run_fcxr_loop(
                p, S["net"], start=state, **kwargs
            )
            state = run["checkpoint"]
            stream = writer.finalize(work / f"chunk_{chunk:02d}_spikes.npz")
            binary.unlink(missing_ok=True)
            streams.append(stream)
            rate_parts.append(np.asarray(run["rate_E"], float))
            sliced = U2._trace_slice(state.slow, starts, stride)
            for key, value in sliced.items():
                trace_parts.setdefault(key, []).append(value)
            _npz_atomic(
                work / f"chunk_{chunk:02d}_summary.npz",
                rate_dt_ms=np.asarray([U2.TRACE_DT_MS], np.float32),
                rate_E=np.asarray(run["rate_E"], float)[::stride].astype(np.float32), **sliced,
            )
            U2.save_loop_state(str(work / "rolling_checkpoint.npz"), state)
            row = U2._resource_row(
                f"{sentinel}_CHUNK", baseline_swap, chunk=chunk + 1,
                completed_ms=(chunk + 1) * U2.CHUNK_MS, wall_s=time.time() - started,
            )
            _write_json(work / "progress.json", {
                "status": "RUNNING", "completed_chunks": chunk + 1,
                "completed_ms": (chunk + 1) * U2.CHUNK_MS,
                "state_hash": state_hash(state), "resource_action": row["action"],
            })
            if row["action"] == "TERMINATE_AFTER_CHECKPOINT":
                raise RuntimeError("RESOURCE_STOP_AFTER_CHECKPOINT")

        steps = np.concatenate([s.steps + k * chunk_steps for k, s in enumerate(streams)])
        cells = np.concatenate([s.cells for s in streams])
        stream = SparseSpikeStream(steps, cells, N_CHUNKS * chunk_steps, S["NE"])
        rate = np.concatenate(rate_parts)
        traces = {key: np.concatenate(parts) for key, parts in trace_parts.items()}
        reports = U2._window_reports(stream, n_seconds=N_CHUNKS)
        saturated = bool(any(x["mean_sat_ceiling_ratio"] >= 1.0 for x in reports))
        baseline = U2._baseline()
        af, af_dt = stream.active_fraction(
            dt_ms=U2.DT_MS, bin_ms=float(baseline["af_bin_ms"])
        )
        events = LC5.CM.detect_events(
            af, af_dt, event_on_frac=float(baseline["frozen_event_bar"])
        )
        returned = [e for e in events if e["returned"]]
        windows = LC5.build_windows(
            rate, U2.DT_MS, af, af_dt, float(baseline["band"]["roll_hi"]), returned,
            float(baseline["band"]["win_ms"]),
            event_lookback_ms=float(baseline["band"]["event_lookback_ms"]),
            finite=bool(np.all(np.isfinite(rate))),
        )
        lifecycle = LC5.classify_lifecycle(windows, baseline["band"], runaway=saturated)
        regimes = LC5._smooth_isolated(lifecycle["regimes"])
        outcome, onset_ms, offset_ms = _outcome(lifecycle, regimes, saturated)

        ref_stream, ref_rate = _reference_prefix()
        rate10 = rate[::stride].astype(np.float32)
        control = {
            "required": bool(float(gamma) == 0.0),
            "spike_sha256_expected": ref_stream.sha256,
            "spike_sha256_observed": stream.sha256,
            "spike_exact": bool(stream.sha256 == ref_stream.sha256),
            "rate_max_abs_diff_hz": float(np.max(np.abs(rate10 - ref_rate))),
        }
        if float(gamma) == 0.0 and not (control["spike_exact"] and control["rate_max_abs_diff_hz"] == 0.0):
            outcome = "NATURAL_PREFIX_CONTROL_MISMATCH"

        summary = {
            "status": "COMPLETE", "arm": tag, "outcome": outcome,
            "runtime_semantics": "fresh_t0_u0_always_online_no_step",
            "gamma_nominal_dose": float(gamma), "Imax": float(imax),
            "tau_ms": U2.TAU_MS, "a_load": float(prelock["a_load"]), "h": 3,
            "T_ms": RUN_MS, "onset_ms": onset_ms, "offset_ms": offset_ms,
            "lifecycle": lifecycle, "n_events": len(events),
            "n_returning": len(returned), "control_parity": control,
            "external_input_sha256": input_hasher.sha256,
            "spike_sha256": stream.sha256, "window_reports": reports,
            "per_second_mean_rate_hz": [float(x["mean_hz"]) for x in reports],
            "mean_rate_hz": float(np.mean(rate)),
            "end_rate_hz": float(np.mean(rate[-int(round(1000.0 / U2.DT_MS)):])),
            "pump_current_peak_mean": float(np.max(traces["pump_current_mean"])),
            "D_start_end": [float(traces["D_mean"][0]), float(traces["D_mean"][-1])],
            "H_start_end": [float(traces["H_mean"][0]), float(traces["H_mean"][-1])],
            "u_start_end": [float(traces["u_mean"][0]), float(traces["u_mean"][-1])],
            "clip_frac_max": float(np.max(traces["clip_frac"])),
            "final_state_hash": state_hash(state),
            "mechanism_hashes": {str(path): _sha(path) for path in MECHANISM_FILES},
            "config_scalar": {
                k: v for k, v in cfg_dict.items()
                if np.isscalar(v) and not isinstance(v, (bytes, bytearray))
            },
            "wall_s": time.time() - started, "finished": U2.GEO._now(),
        }
        with AtomicStageBundle(arm) as bundle:
            _write_json(bundle.path("summary.json"), summary)
            _npz_atomic(
                bundle.path("traces.npz"), rate_dt_ms=np.asarray([U2.TRACE_DT_MS], np.float32),
                rate_E=rate10, af=af.astype(np.float32), af_dt_ms=np.asarray([af_dt], np.float32),
                **traces,
            )
            _npz_atomic(
                bundle.path("spikes.npz"), steps=stream.steps, cells=stream.cells.astype(np.int32),
                n_steps=np.asarray([stream.n_steps], np.int64),
                n_cells=np.asarray([stream.n_cells], np.int64), sha256=np.asarray([stream.sha256]),
            )
            U2.save_loop_state(str(bundle.path("final_state.npz")), state)
            bundle.commit(required=["summary.json", "traces.npz", "spikes.npz", "final_state.npz"])
        _write_json(U2.OUT / f"DONE_{sentinel}.json", {
            "status": "DONE", "bundle": str(arm), "outcome": outcome,
            "finished": U2.GEO._now(),
        })
        running.unlink(missing_ok=True)
        shutil.rmtree(work)
        return summary
    except BaseException as exc:
        _write_json(U2.OUT / f"FAILED_{sentinel}.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "work_dir_retained": str(work), "finished": U2.GEO._now(),
        })
        running.unlink(missing_ok=True)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, required=True, choices=(0.0, 0.001, 0.003))
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("a 40k natural-prefix arm requires --confirm-run")
    print(json.dumps(json_sanitize(stage_prefix(args.gamma)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
