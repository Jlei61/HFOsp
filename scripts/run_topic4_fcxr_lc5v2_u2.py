#!/usr/bin/env python3
"""Run the minimal LC5v2 onset-fork experiment.

The archived no-pump trajectory is the exact control.  The only new 40k trajectory initially
authorised is tau=8 s, Gamma=0.25.  It starts from the same natural-onset exact state, carries the
offline-replayed per-cell load field, and leaves Z/H dynamic.  Results are written in one-second
transactional chunks so a teardown cannot turn a partial long run into a published arm.
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

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_phase_map as PM  # noqa: E402
import run_topic4_fcxr_lc5 as LC5  # noqa: E402
import run_topic4_mz_slowvars as OLD_SLOW  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from params import Params  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_statefork import load_into, save_loop_state  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle,
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    SparseSpikeStream,
    json_sanitize,
    load_sparse_spike_stream,
    refractory_ceiling_report,
)
from src.topic4_fcxr_lc5_finite_episode import (  # noqa: E402
    calibrate_episode_dose,
    classify_u2_excursion,
    estimate_shrunken_p0,
    replay_finite_load,
)


DT_MS = 0.05
RUN_MS = 7000.0
CHUNK_MS = 1000.0
TRACE_DT_MS = 10.0
CONNECTION_SEED = 1
NOISE_SEED = 401
TAU_MS = 8000.0
SAT_CEILING_HZ = float(PP.SAT_CEILING_FRAC) * (1000.0 / float(Params().tau_ref_E))

LC5_OLD = ROOT / "results/topic4_sef_hfo/fcxr_lc5_episode_pump"
SOURCE = LC5_OLD / "u1_capture"
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode"
CAL = OUT / "finite_calibration"
CONTROL = OUT / "u2a_control"
EXACT_AUDIT = OUT / "exact_load_audit"
MECHANISM_FILES = (
    ROOT / "src/snn_engine/mz_slow_vars.py",
    ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_statefork.py",
    ROOT / "src/topic4_fcxr_lc5_finite_episode.py",
    Path(__file__),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_sanitize(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _arm_tag(gamma):
    return f"u2a_tau8_gamma{int(round(float(gamma) * 100)):03d}"


def _arm_paths(gamma):
    tag = _arm_tag(gamma)
    return OUT / tag, OUT / f".{tag}.work"


def _load_contract(gamma=0.25, *, prefer_exact=True):
    prelock = json.loads((CAL / "candidate_prelock.json").read_text())
    if prelock.get("status") != "LC5V2_CANDIDATE_PRELOCK":
        raise SystemExit("LC5v2 candidate prelock is absent or invalid")
    fields_path = Path(prelock["calibration_arrays"])
    p0_key, u_key = prelock["p0_key"], prelock["u_onset_key"]
    exact_used = False
    if prefer_exact and EXACT_AUDIT.is_dir():
        audit = json.loads((EXACT_AUDIT / "summary.json").read_text())
        if audit.get("status") == "EXACT_LOAD_AUDIT_PASS":
            fields_path = EXACT_AUDIT / "fields.npz"
            p0_key, u_key = "p0_exact", "u_onset_exact"
            exact_used = True
    with np.load(fields_path, allow_pickle=False) as z:
        p0 = np.asarray(z[p0_key], float)
        u_onset = np.asarray(z[u_key], float)
    key = str(float(gamma))
    if exact_used:
        imax = float(audit["dose_exact"]["Imax_by_gamma"][key])
    else:
        imax = float(prelock["Imax_by_gamma"][key])
    prelock = dict(prelock, runtime_fields=str(fields_path), exact_fields_used=exact_used)
    return prelock, p0, u_onset, imax


def _baseline():
    return json.loads(Path(E01.ARTIFACTS["lc1_baseline"]).read_text())


def _window_reports(stream, *, n_seconds=7):
    reports = []
    steps_per_second = int(round(1000.0 / DT_MS))
    for sec in range(int(n_seconds)):
        rates = stream.per_cell_rate_hz(
            lo_step=sec * steps_per_second,
            hi_step=(sec + 1) * steps_per_second,
            dt_ms=DT_MS,
        )
        report = refractory_ceiling_report(
            rates,
            tau_ref_ms=float(Params().tau_ref_E),
            sat_ceiling_hz=SAT_CEILING_HZ,
        )
        report["window_ms"] = [sec * 1000.0, (sec + 1) * 1000.0]
        reports.append(report)
    return reports


def stage_control():
    """Publish the already-recorded 11--18 s no-U segment as the exact U2 control."""

    if CONTROL.is_dir():
        return json.loads((CONTROL / "summary.json").read_text())
    summary = json.loads((SOURCE / "u1_capture_summary.json").read_text())
    onset_ms = float(summary["onset_ms"])
    full = load_sparse_spike_stream(SOURCE / "u1_sparse_spikes.npz")
    lo = int(round(onset_ms / DT_MS))
    hi = int(round((onset_ms + RUN_MS) / DT_MS))
    left = int(np.searchsorted(full.steps, lo, side="left"))
    right = int(np.searchsorted(full.steps, hi, side="left"))
    stream = SparseSpikeStream(
        full.steps[left:right] - lo,
        full.cells[left:right].copy(),
        hi - lo,
        full.n_cells,
    )
    traces = np.load(SOURCE / "u1_capture_traces.npz")
    source_dt = float(traces["rate_dt_ms"][0])
    i0, i1 = int(round(onset_ms / source_dt)), int(round((onset_ms + RUN_MS) / source_dt))
    rate = np.asarray(traces["rate_E"][i0:i1], float)
    reports = _window_reports(stream)
    label = classify_u2_excursion(
        rate,
        dt_ms=source_dt,
        interictal_upper_hz=float(_baseline()["band"]["roll_hi"]),
        saturated=bool(any(r["mean_sat_ceiling_ratio"] >= 1.0 for r in reports)),
    )
    # The seven-second fork ends before the registered crossing, but the same uninterrupted source
    # is already adjudicated as escalating saturation at 18--22 s.  Keep the local label and the
    # known continuation separate instead of forcing one into the other.
    source_adjudication = json.loads((LC5_OLD / "u1_carrier_adjudication.json").read_text())
    payload = {
        "status": "COMPLETE",
        "arm": "u2a_control_reused_exact_source",
        "new_40k_compute": False,
        "source_interval_ms": [onset_ms, onset_ms + RUN_MS],
        "within_7s_label": label,
        "known_uninterrupted_continuation": source_adjudication["source_type"],
        "window_reports": reports,
        "spike_sha256": stream.sha256,
        "mean_rate_hz": float(np.mean(rate)),
        "end_rate_hz": float(np.mean(rate[-int(round(1000.0 / source_dt)):])),
    }
    with AtomicStageBundle(CONTROL) as bundle:
        _write_json(bundle.path("summary.json"), payload)
        _npz_atomic(
            bundle.path("traces.npz"),
            rate_dt_ms=np.asarray([source_dt], np.float32),
            rate_E=rate.astype(np.float32),
            steps=stream.steps,
            cells=stream.cells.astype(np.int32),
        )
        bundle.commit(required=["summary.json", "traces.npz"])
    return payload


def stage_exact_load_audit():
    """Audit the 1-ms calibration field against the exact 0.05-ms load equation at onset.

    This is a targeted validation unlocked by the first strong U2 separation.  It does not tune a
    parameter: ``a_load``, tau, windows, h and Gamma remain the finite-calibration values.
    """

    if EXACT_AUDIT.is_dir():
        return json.loads((EXACT_AUDIT / "summary.json").read_text())
    prelock, p0_coarse, u_coarse, _ = _load_contract(0.25, prefer_exact=False)
    full = load_sparse_spike_stream(SOURCE / "u1_sparse_spikes.npz")
    stop_step = int(round(15000.0 / DT_MS))
    right = int(np.searchsorted(full.steps, stop_step, side="left"))
    stream = SparseSpikeStream(
        full.steps[:right].copy(), full.cells[:right].copy(), stop_step, full.n_cells
    )
    blocks = {
        "base0": (int(7000.0 / DT_MS), int(8000.0 / DT_MS)),
        "base1": (int(8000.0 / DT_MS), int(9000.0 / DT_MS)),
        "base2": (int(9000.0 / DT_MS), int(10000.0 / DT_MS)),
        "base3": (int(10000.0 / DT_MS), int(11000.0 / DT_MS)),
        "early": (int(12000.0 / DT_MS), int(14000.0 / DT_MS)),
    }
    snapshots = {int(11000.0 / DT_MS) - 1: "onset"}
    first = replay_finite_load(
        stream, dt_ms=DT_MS, tau_ms=TAU_MS, a_load=float(prelock["a_load"]),
        blocks=blocks, snapshot_steps=snapshots,
    )
    baseline_rate = np.load(CAL / "u_fields_tau3_8_15.npz")["baseline_rate_hz"]
    phi_blocks = np.vstack([first.block_phi_mean[f"base{k}"] for k in range(4)])
    p0_fit = estimate_shrunken_p0(phi_blocks, baseline_rate)
    p0_exact = np.asarray(p0_fit.pop("p0"), float)
    second = replay_finite_load(
        stream, dt_ms=DT_MS, tau_ms=TAU_MS, a_load=float(prelock["a_load"]),
        blocks={"early": blocks["early"]}, p0=p0_exact, excess_block="early",
    )
    force = np.load(CAL / "u_fields_tau3_8_15.npz")["recurrent_force_integral_ms"]
    dose = calibrate_episode_dose(
        unit_excess_integral_ms=second.excess_integral_ms,
        recurrent_force_integral_ms=force,
        gammas=(0.10, 0.25, 0.40),
    )
    u_exact = first.snapshots["onset"]
    du, dp0 = np.abs(u_exact - u_coarse), np.abs(p0_exact - p0_coarse)
    coarse_imax = float(prelock["Imax_by_gamma"]["0.25"])
    exact_imax = float(dose["Imax_by_gamma"]["0.25"])
    passes = bool(
        np.quantile(du, 0.99) < 1e-3
        and np.quantile(dp0, 0.99) < 1e-3
        and abs(exact_imax / coarse_imax - 1.0) < 0.01
    )
    payload = {
        "status": "EXACT_LOAD_AUDIT_PASS" if passes else "EXACT_LOAD_AUDIT_FAIL",
        "scientific_scope": "clock-alignment audit only; no lifecycle claim",
        "dt_exact_ms": DT_MS,
        "tau_ms": TAU_MS,
        "a_load": float(prelock["a_load"]),
        "u_onset_abs_error": {
            "mean": float(du.mean()), "q99": float(np.quantile(du, 0.99)),
            "max": float(du.max()),
        },
        "p0_abs_error": {
            "mean": float(dp0.mean()), "q99": float(np.quantile(dp0, 0.99)),
            "max": float(dp0.max()),
        },
        "Imax_gamma025": {"coarse": coarse_imax, "exact": exact_imax},
        "dose_exact": dose,
        "p0_fit_exact": p0_fit,
    }
    with AtomicStageBundle(EXACT_AUDIT) as bundle:
        _write_json(bundle.path("summary.json"), payload)
        _npz_atomic(
            bundle.path("fields.npz"), p0_exact=p0_exact.astype(np.float32),
            u_onset_exact=u_exact.astype(np.float32),
            unit_excess_integral_ms_exact=second.excess_integral_ms.astype(np.float32),
        )
        bundle.commit(required=["summary.json", "fields.npz"])
    if not passes:
        raise RuntimeError("exact load audit failed; do not launch another U2 arm")
    return payload


def _stage_lock(gamma):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = _arm_tag(gamma)
    f = (OUT / f".{tag}.lock").open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        f.close()
        raise SystemExit(f"LC5v2 {tag} is already running") from exc
    return f


def _resource_row(stage, baseline_swap, **extra):
    row = dict(stage=stage, epoch=time.time(), **GEO._meminfo(), **extra)
    row["swap_delta_mib"] = float(row["swap_used_mib"] - baseline_swap)
    row["action"] = (
        "TERMINATE_AFTER_CHECKPOINT" if row["swap_delta_mib"] >= 512.0
        else "NO_NEW_SUBMISSION" if row["swap_delta_mib"] >= 256.0
        else "CONTINUE"
    )
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(json_sanitize(row), sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    return row


def _build_start(p0, u_onset, imax, *, a_load):
    candidate = LC5._load_candidate()
    S = PP.build_substrate(CONNECTION_SEED)
    install_registered_noise_rng(S["net"])
    cfg_dict = LC5._config(S, candidate)
    cfg_dict.update(
        use_pump=True,
        pump_sensor_only=False,
        pump_a_load=float(a_load),
        pump_tau_ms=TAU_MS,
        pump_Imax=float(imax),
        pump_h=3,
        pump_excess_mode="rectified_excess",
        pump_p0_E=p0.copy(),
        pump_u_init_E=np.zeros(int(S["NE"]), dtype=float),
        pump_record_calibration=False,
    )
    slow = MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg_dict), NE=S["NE"],
        core_mask_E=OLD_SLOW.build_core_masks(S),
    )
    template = PM._seed_template(S, slow)
    start = load_into(SOURCE / "states/onset.npz", template)
    start.slow.u_pump_E[:] = u_onset
    return S, start, cfg_dict


def _trace_slice(slow, starts, stride):
    names = {
        "D_mean": "trace_z_mean",
        "H_mean": "trace_h_lc2_mean",
        "H_source_mean": "trace_gA_raw_lc2_mean",
        "gErec_mean": "trace_gErec_mean",
        "u_mean": "trace_u_mean",
        "u_max": "trace_u_max",
        "pump_phi_mean": "trace_phi_pump_mean",
        "pump_current_mean": "trace_pump_excess_mean",
        "pump_current_max": "trace_pump_excess_max",
        "clip_frac": "trace_conductance_clip_frac",
    }
    out = {}
    for key, attr in names.items():
        values = np.asarray(getattr(slow, attr)[starts[attr]:], float)
        if key == "D_mean":
            values = 1.0 - values
        out[key] = values[::stride].astype(np.float32)
    return out


def stage_gamma(gamma):
    arm, work = _arm_paths(gamma)
    tag = _arm_tag(gamma)
    sentinel_tag = tag.upper()
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    lock_file = _stage_lock(gamma)
    try:
        prelock, p0, u_onset, imax = _load_contract(gamma)
        if work.exists():
            raise SystemExit(f"stale LC5v2 work directory requires inspection: {work}")
        work.mkdir(parents=True)
        resources = GEO._meminfo()
        if resources["mem_available_gib"] < 128.0:
            raise SystemExit("LC5v2 U2 requires at least 128 GiB MemAvailable")
        baseline_swap = float(resources["swap_used_mib"])
        started = time.time()
        running_path = OUT / f"RUNNING_{sentinel_tag}.json"
        _write_json(running_path, {
            "status": "RUNNING", "pid": os.getpid(), "started": GEO._now(),
            "tau_ms": TAU_MS, "gamma": gamma, "Imax": imax,
            "work_dir": str(work),
        })
        (OUT / f"{tag}.pid").write_text(f"{os.getpid()}\n")
        _resource_row(f"{sentinel_tag}_PREFLIGHT", baseline_swap)

        S, state, cfg_dict = _build_start(
            p0, u_onset, imax, a_load=float(prelock["a_load"])
        )
        source_state_hash = state_hash(state)
        stride = int(round(TRACE_DT_MS / DT_MS))
        force_scale = float(state.slow.cfg.E_E - state.slow.cfg.v_match)
        state.slow.recurrent_drive_observer = RecurrentDriveBlockObserver(
            S["NE"], sample_every=stride,
            steps_per_block=int(round(1000.0 / DT_MS)), force_scale=force_scale,
        )
        input_hasher = ExactInputHasher()
        rate_parts, trace_parts, streams = [], {}, []
        p = dataclasses.replace(S["p"], T=RUN_MS, dt=DT_MS)
        chunk_steps = int(round(CHUNK_MS / DT_MS))
        n_chunks = int(round(RUN_MS / CHUNK_MS))

        for chunk in range(n_chunks):
            trace_attrs = (
                "trace_z_mean", "trace_h_lc2_mean", "trace_gA_raw_lc2_mean",
                "trace_gErec_mean", "trace_u_mean", "trace_u_max",
                "trace_phi_pump_mean", "trace_pump_excess_mean", "trace_pump_excess_max",
                "trace_conductance_clip_frac",
            )
            trace_starts = {name: len(getattr(state.slow, name)) for name in trace_attrs}
            binary = work / f"chunk_{chunk:02d}.bin"
            writer = SparseSpikeBinaryWriter(
                binary, step_origin=state.t, n_steps=chunk_steps, n_cells=S["NE"]
            )
            run = run_fcxr_loop(
                p, S["net"], start=state, n_steps=chunk_steps, capture_final=True,
                store_spikes=False, spike_sink=writer, input_sink=input_hasher,
                v_th_per_neuron=S["vth"],
            )
            state = run["checkpoint"]
            stream = writer.finalize(work / f"chunk_{chunk:02d}_spikes.npz")
            binary.unlink(missing_ok=True)
            streams.append(stream)
            rate_part = np.asarray(run["rate_E"], float)
            rate_parts.append(rate_part)
            sliced = _trace_slice(state.slow, trace_starts, stride)
            for key, value in sliced.items():
                trace_parts.setdefault(key, []).append(value)
            _npz_atomic(
                work / f"chunk_{chunk:02d}_summary.npz",
                rate_dt_ms=np.asarray([TRACE_DT_MS], np.float32),
                rate_E=rate_part[::stride].astype(np.float32),
                **sliced,
            )
            save_loop_state(str(work / "rolling_checkpoint.npz"), state)
            row = _resource_row(
                f"{sentinel_tag}_CHUNK", baseline_swap, chunk=chunk + 1,
                completed_ms=(chunk + 1) * CHUNK_MS, wall_s=time.time() - started,
            )
            _write_json(work / "progress.json", {
                "status": "RUNNING", "completed_chunks": chunk + 1,
                "completed_ms": (chunk + 1) * CHUNK_MS,
                "state_hash": state_hash(state), "resource_action": row["action"],
            })
            if row["action"] == "TERMINATE_AFTER_CHECKPOINT":
                raise RuntimeError("RESOURCE_STOP_AFTER_CHECKPOINT")

        steps = np.concatenate([
            s.steps + k * chunk_steps for k, s in enumerate(streams)
        ]).astype(np.int64)
        cells = np.concatenate([s.cells for s in streams]).astype(np.int64)
        stream = SparseSpikeStream(steps, cells, n_chunks * chunk_steps, S["NE"])
        rate_full = np.concatenate(rate_parts)
        window_reports = _window_reports(stream)
        saturated = bool(any(r["mean_sat_ceiling_ratio"] >= 1.0 for r in window_reports))
        label = classify_u2_excursion(
            rate_full, dt_ms=DT_MS,
            interictal_upper_hz=float(_baseline()["band"]["roll_hi"]),
            saturated=saturated,
        )
        drive = state.slow.recurrent_drive_observer.arrays()
        traces = {key: np.concatenate(parts) for key, parts in trace_parts.items()}
        recurrent_force_mean = traces["gErec_mean"] * force_scale
        achieved_mean_ratio = np.divide(
            traces["pump_current_mean"], recurrent_force_mean,
            out=np.full_like(recurrent_force_mean, np.nan), where=recurrent_force_mean > 0,
        )
        summary = {
            "status": "COMPLETE", "arm": tag, "label": label,
            "tau_ms": TAU_MS, "gamma_nominal_dose": gamma, "Imax": imax,
            "a_load": float(prelock["a_load"]), "h": 3,
            "source_exact_state": str(SOURCE / "states/onset.npz"),
            "source_with_attached_u_state_hash": source_state_hash,
            "final_state_hash": state_hash(state),
            "external_input_sha256": input_hasher.sha256,
            "n_external_input_steps": input_hasher.n_steps,
            "spike_sha256": stream.sha256,
            "window_reports": window_reports,
            "mean_rate_hz": float(np.mean(rate_full)),
            "end_rate_hz": float(np.mean(rate_full[-int(round(1000.0 / DT_MS)):])),
            "pump_current_peak_mean": float(np.max(traces["pump_current_mean"])),
            "achieved_population_mean_ratio_median": float(np.nanmedian(achieved_mean_ratio)),
            "achieved_population_mean_ratio_peak": float(np.nanmax(achieved_mean_ratio)),
            "achieved_ratio_note": (
                "diagnostic population-mean instantaneous ratio; nominal Gamma remains the locked "
                "median-per-cell finite-window dose ratio"
            ),
            "D_start_end": [float(traces["D_mean"][0]), float(traces["D_mean"][-1])],
            "H_start_end": [float(traces["H_mean"][0]), float(traces["H_mean"][-1])],
            "u_start_end": [float(traces["u_mean"][0]), float(traces["u_mean"][-1])],
            "clip_frac_max": float(np.max(traces["clip_frac"])),
            "wall_s": time.time() - started,
            "mechanism_hashes": {str(p): _sha(p) for p in MECHANISM_FILES},
            "config_scalar": {
                k: v for k, v in cfg_dict.items()
                if np.isscalar(v) and not isinstance(v, (bytes, bytearray))
            },
            "finished": GEO._now(),
        }
        with AtomicStageBundle(arm) as bundle:
            _write_json(bundle.path("summary.json"), summary)
            _npz_atomic(
                bundle.path("traces.npz"),
                rate_dt_ms=np.asarray([TRACE_DT_MS], np.float32),
                rate_E=rate_full[::stride].astype(np.float32),
                recurrent_force_mean=recurrent_force_mean.astype(np.float32),
                achieved_population_mean_ratio=achieved_mean_ratio.astype(np.float32),
                recurrent_block_index=drive["block_index"],
                recurrent_effective_force_mean=drive["effective_force_mean"],
                **traces,
            )
            _npz_atomic(
                bundle.path("spikes.npz"), steps=stream.steps,
                cells=stream.cells.astype(np.int32),
                n_steps=np.asarray([stream.n_steps], np.int64),
                n_cells=np.asarray([stream.n_cells], np.int64),
                sha256=np.asarray([stream.sha256]),
            )
            save_loop_state(str(bundle.path("final_state.npz")), state)
            bundle.commit(required=["summary.json", "traces.npz", "spikes.npz", "final_state.npz"])
        _write_json(OUT / f"DONE_{sentinel_tag}.json", {
            "status": "DONE", "bundle": str(arm), "label": label, "finished": GEO._now(),
        })
        running_path.unlink(missing_ok=True)
        shutil.rmtree(work)
        return summary
    except BaseException as exc:
        _write_json(OUT / f"FAILED_{sentinel_tag}.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "work_dir_retained": str(work), "finished": GEO._now(),
        })
        (OUT / f"RUNNING_{sentinel_tag}.json").unlink(missing_ok=True)
        raise
    finally:
        lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("control", "exact-load-audit", "gamma010", "gamma025"),
        required=True,
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if args.stage.startswith("gamma") and not args.confirm_run:
        raise SystemExit("a 40k U2 arm requires --confirm-run")
    if args.stage == "control":
        result = stage_control()
    elif args.stage == "exact-load-audit":
        result = stage_exact_load_audit()
    else:
        result = stage_gamma(0.10 if args.stage == "gamma010" else 0.25)
    print(json.dumps(json_sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
