#!/usr/bin/env python3
"""FCXR-LC5 U0 audit and U1a canonical no-actuator capture.

This runner deliberately stops before formal U calibration.  U1a records the fresh spike/input
history and exact SNN landmarks from which U1b must be derived; it never constructs a provisional
load and therefore cannot circularly choose its own scale.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-lc5")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_phaseplane as PP  # noqa: E402
import run_sef_hfo_snn_cm_spontaneous_readout as CM  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc4_lifecycle as LC4  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_statefork import save_loop_state  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    json_sanitize,
)
from src.topic4_mz_fcxr_lifecycle import (  # noqa: E402
    _smooth_isolated,
    build_windows,
    classify_lifecycle,
)
from src.topic4_mz_fcxr_pump import VirtualSeegComponentObserver  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5_episode_pump"
U1_FINAL = OUT / "u1_capture"
U1_WORK = OUT / ".u1_capture.work"
DT_MS = 0.05
T_MS = 22000.0
CHUNK_MS = 1000.0
TRACE_SAMPLE_MS = 1.0
SNAPSHOT_MS = 250.0
CONNECTION_SEED = 1
NOISE_SEED = 401
LC4F = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4f_x_depth_closure"
PERCELL = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/percell_separation/per_cell.npz"
SPEC = ROOT / "docs/superpowers/specs/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump.md"
MECHANISM_FILES = (
    "src/snn_engine/mz_slow_vars.py",
    "src/topic4_mz_fcxr_pump.py",
    "src/topic4_fcxr_lc3.py",
    "src/topic4_fcxr_lc3_statefork.py",
    "src/topic4_fcxr_lc5.py",
    "scripts/run_topic4_fcxr_lc5.py",
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_resource(stage, **extra):
    row = dict(stage=stage, epoch=time.time(), **GEO._meminfo(), **extra)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(json_sanitize(row), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return row


def _git_head():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _stage_lock(name):
    OUT.mkdir(parents=True, exist_ok=True)
    f = (OUT / f".{name}.lock").open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        f.close()
        raise SystemExit(f"LC5 stage {name} is already running") from exc
    return f


def _load_candidate():
    path = LC4F / "candidate_lock.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "X0_PASS":
        raise SystemExit("LC5 requires the accepted LC4f candidate lock")
    return payload["candidate"]


def _baseline_path():
    path = Path(E01.ARTIFACTS["lc1_baseline"])
    if not path.is_file():
        raise SystemExit(f"missing frozen LC1 baseline: {path}")
    return path


def stage_audit():
    E01.FCXR._assert_engine_blessed()
    candidate_path = LC4F / "candidate_lock.json"
    screen_path = LC4F / "x_depth_screen.json"
    required = (candidate_path, screen_path, PERCELL, _baseline_path(), SPEC, PLAN)
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"LC5 U0 missing required artifacts: {missing}")
    with np.load(PERCELL, allow_pickle=False) as z:
        percell_keys = sorted(z.files)
    payload = {
        "status": "U0_PASS",
        "scientific_scope": "instrument and lineage audit only; no pump efficacy claim",
        "git_head": _git_head(),
        "artifacts": {str(Path(p)): _sha(p) for p in required},
        "percell_keys": percell_keys,
        "mechanism_module_hashes": {p: _sha(ROOT / p) for p in MECHANISM_FILES},
        "blessed_engine_hashes": json.loads((ROOT / "results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json").read_text()),
        "canonical_substrate": {
            "Z": "dynamic", "H": "dynamic", "X": "fixed_1_from_t0",
            "M": "off_from_t0", "U_actuator": "off", "kick": False, "reset": False,
            "connection_seed": CONNECTION_SEED, "noise_seed": NOISE_SEED,
        },
        "candidate_provenance_only": _load_candidate(),
        "created": GEO._now(),
    }
    _write_json(OUT / "u0_lineage_audit.json", payload)
    return payload


def _assert_audit_current():
    path = OUT / "u0_lineage_audit.json"
    if not path.is_file():
        raise SystemExit("LC5 U1 requires U0 audit")
    audit = json.loads(path.read_text())
    if audit.get("status") != "U0_PASS":
        raise SystemExit("LC5 U0 audit is not PASS")
    if audit.get("git_head") != _git_head():
        raise SystemExit("LC5 code commit drifted after U0 audit")
    for rel, expected in audit.get("mechanism_module_hashes", {}).items():
        if _sha(ROOT / rel) != expected:
            raise SystemExit(f"LC5 mechanism source drift after U0 audit: {rel}")
    for name, expected in audit.get("artifacts", {}).items():
        if _sha(name) != expected:
            raise SystemExit(f"LC5 source artifact drift after U0 audit: {name}")
    return audit


def _config(S, candidate):
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    cfg.update(
        use_m=False,
        theta_h_lc2=float(candidate["theta_h_lc2"]),
        x_relay_frozen_E=np.ones(int(S["NE"]), dtype=float),
        use_pump=False,
    )
    return cfg


def _config_provenance(cfg):
    clean = {}
    for key, value in cfg.items():
        if isinstance(value, np.ndarray):
            clean[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        else:
            clean[key] = value
    raw = json.dumps(json_sanitize(clean), sort_keys=True, separators=(",", ":")).encode()
    return clean, hashlib.sha256(raw).hexdigest()


def _save_npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _copy_state(seconds_dir, target_dir, t_ms, label):
    src = seconds_dir / f"t{int(round(t_ms))}.npz"
    if not src.is_file():
        raise FileNotFoundError(f"missing exact second state {src}")
    dst = target_dir / f"{label}.npz"
    shutil.copy2(src, dst)
    return dst


def stage_capture():
    if U1_FINAL.is_dir():
        return json.loads((U1_FINAL / "u1_capture_summary.json").read_text())
    _assert_audit_current()
    preflight = _append_resource("U1_PREFLIGHT")
    if preflight["mem_available_gib"] < 128.0:
        raise SystemExit(f"LC5 U1 MemAvailable {preflight['mem_available_gib']:.1f} GiB < 128")
    if U1_WORK.exists():
        raise SystemExit(f"stale/incomplete U1 work directory requires audit: {U1_WORK}")
    U1_WORK.mkdir(parents=True)
    seconds_dir = U1_WORK / "states_by_second"
    states_dir = U1_WORK / "states"
    seconds_dir.mkdir(); states_dir.mkdir()
    _write_json(OUT / "U1_RUNNING.json", {
        "status": "RUNNING", "pid": os.getpid(), "started": GEO._now(),
        "work_dir": str(U1_WORK), "T_ms": T_MS,
    })
    (OUT / "u1_capture.pid").write_text(f"{os.getpid()}\n")

    t_wall = time.time()
    try:
        candidate = _load_candidate()
        S = PP.build_substrate(CONNECTION_SEED)
        install_registered_noise_rng(S["net"])
        cfg_dict = _config(S, candidate)
        cfg = MZSlowVarsConfig(**cfg_dict)
        snapshot_steps = {
            int(round(t / DT_MS)): f"t{int(t)}"
            for t in np.arange(0.0, T_MS + SNAPSHOT_MS, SNAPSHOT_MS)
        }
        slow = MZSlowVars(
            S["N"], 18.0, cfg, NE=S["NE"], core_mask_E=OLD.build_core_masks(S),
            snapshot_steps=snapshot_steps,
        )
        contacts = np.asarray(S["reg"]["montage_sheet"].contacts, float)
        recorder = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
        sample_every = int(round(TRACE_SAMPLE_MS / DT_MS))
        slow.seeg_observer = VirtualSeegComponentObserver(recorder, cfg, sample_every=sample_every)
        slow.recurrent_drive_observer = RecurrentDriveBlockObserver(
            S["NE"], sample_every=sample_every,
            steps_per_block=int(round(1000.0 / DT_MS)),
            force_scale=float(cfg.E_E - cfg.v_match),
        )
        S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
        p = dataclasses.replace(S["p"], T=T_MS, dt=DT_MS)
        n_steps = int(round(T_MS / DT_MS))
        chunk_steps = int(round(CHUNK_MS / DT_MS))
        spike_writer = SparseSpikeBinaryWriter(
            U1_WORK / "u1_sparse_spikes.bin", step_origin=0, n_steps=n_steps, n_cells=S["NE"]
        )
        input_hasher = ExactInputHasher()
        rate_e_parts, rate_i_parts = [], []
        checkpoint = None
        for chunk, start in enumerate(range(0, n_steps, chunk_steps), start=1):
            take = min(chunk_steps, n_steps - start)
            kwargs = dict(
                n_steps=take, capture_final=True, store_spikes=False,
                spike_sink=spike_writer, input_sink=input_hasher,
                v_th_per_neuron=S["vth"],
            )
            if checkpoint is None:
                out = run_fcxr_loop(p, S["net"], slow=slow, **kwargs)
            else:
                out = run_fcxr_loop(p, S["net"], start=checkpoint, **kwargs)
            checkpoint = out["checkpoint"]
            rate_e_parts.append(np.asarray(out["rate_E"], np.float32))
            rate_i_parts.append(np.asarray(out["rate_I"], np.float32))
            state_path = seconds_dir / f"t{int(round(checkpoint.t * DT_MS))}.npz"
            save_loop_state(str(state_path), checkpoint)
            _write_json(U1_WORK / "rolling_checkpoint.json", {
                "state": str(state_path), "state_hash": state_hash(checkpoint),
                "completed_ms": checkpoint.t * DT_MS, "chunk": chunk,
            })
            _append_resource("U1_CHUNK", chunk=chunk, completed_ms=checkpoint.t * DT_MS)
        stream = spike_writer.finalize(U1_WORK / "u1_sparse_spikes.npz")
        rate_e = np.concatenate(rate_e_parts)
        rate_i = np.concatenate(rate_i_parts)

        baseline = json.loads(_baseline_path().read_text())
        af, af_dt = stream.active_fraction(dt_ms=DT_MS, bin_ms=float(baseline["af_bin_ms"]))
        events = CM.detect_events(af, af_dt, event_on_frac=float(baseline["frozen_event_bar"]))
        ret = [e for e in events if e["returned"]]
        windows = build_windows(
            rate_e, DT_MS, af, af_dt, float(baseline["band"]["roll_hi"]), ret,
            float(baseline["band"]["win_ms"]),
            event_lookback_ms=float(baseline["band"]["event_lookback_ms"]),
            finite=bool(np.all(np.isfinite(rate_e))),
        )
        lifecycle = classify_lifecycle(windows, baseline["band"])
        regimes = _smooth_isolated(lifecycle["regimes"])
        bout = LC4.first_ictal_bout(regimes, float(baseline["band"]["win_ms"]))
        onset_ms = None if bout is None else float(bout[0] * baseline["band"]["win_ms"])
        if onset_ms is None:
            _write_json(OUT / "U1_ENTRY_NOT_REPRODUCED.json", {
                "status": "STOP", "verdict": "U1_ENTRY_NOT_REPRODUCED",
                "lifecycle": lifecycle, "events": len(events), "finished": GEO._now(),
            })
            raise RuntimeError("U1 canonical trajectory did not enter within 22 s")
        if onset_ms < 4000.0 or onset_ms + 4000.0 > T_MS:
            raise RuntimeError("U1_REFERENCE_WINDOW_UNAVAILABLE")

        pre_lo, pre_hi = int(round((onset_ms - 4000.0) / DT_MS)), int(round(onset_ms / DT_MS))
        high_lo, high_hi = int(round((onset_ms + 1000.0) / DT_MS)), int(round((onset_ms + 4000.0) / DT_MS))
        baseline_rates = stream.per_cell_rate_hz(lo_step=pre_lo, hi_step=pre_hi, dt_ms=DT_MS)
        high_rates = stream.per_cell_rate_hz(lo_step=high_lo, hi_step=high_hi, dt_ms=DT_MS)
        eligible = high_rates > 0.0
        if not eligible.any():
            raise RuntimeError("U1 high-reference support is empty")
        r_hi_ref = float(np.median(high_rates[eligible]))

        slow_f = checkpoint.slow
        table = snapshot_table(slow_f.snapshots, DT_MS, GEO._region_masks(S))
        r_base = float(np.median(rate_e[:int(round(onset_ms / DT_MS))]))
        ledger = build_event_ledger(
            events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
            rate_hz=rate_e, dt_ms=DT_MS, r_base_hz=r_base, table=table,
            onset_ms=onset_ms, offset_ms=None, total_ms=T_MS,
        )
        _write_json(U1_WORK / "u1_event_ledger.json", ledger)

        _copy_state(seconds_dir, states_dir, onset_ms - 1000.0, "pre_onset")
        _copy_state(seconds_dir, states_dir, onset_ms, "onset")
        _copy_state(seconds_dir, states_dir, onset_ms + 1000.0, "onset_plus_1s")
        _copy_state(seconds_dir, states_dir, onset_ms + 4000.0, "onset_plus_4s")
        _copy_state(seconds_dir, states_dir, T_MS, "late")

        drive = slow_f.recurrent_drive_observer.arrays()
        seeg = slow_f.seeg_observer.stack()
        stride = max(1, int(round(10.0 / DT_MS)))
        _save_npz_atomic(
            U1_WORK / "u1_capture_traces.npz",
            rate_dt_ms=np.asarray([10.0], np.float32),
            rate_E=rate_e[::stride], rate_I=rate_i[::stride],
            af=af.astype(np.float32), af_dt_ms=np.asarray([af_dt], np.float32),
            slow_trace_dt_ms=np.asarray([10.0], np.float32),
            D_mean=(1.0 - np.asarray(slow_f.trace_z_mean)[::stride]).astype(np.float32),
            H_mean=np.asarray(slow_f.trace_h_lc2_mean)[::stride].astype(np.float32),
            gErec_raw_mean=np.asarray(slow_f.trace_gA_raw_lc2_mean)[::stride].astype(np.float32),
            gErec_eff_mean=np.asarray(slow_f.trace_gErec_mean)[::stride].astype(np.float32),
            seeg_dt_ms=np.asarray([TRACE_SAMPLE_MS], np.float32),
            **{f"seeg_{k}": np.asarray(v, np.float32) for k, v in seeg.items()},
        )
        _save_npz_atomic(
            U1_WORK / "u1_rate_fields.npz",
            baseline_rate_hz=baseline_rates.astype(np.float32),
            high_rate_hz=high_rates.astype(np.float32),
            eligible_E_hi=eligible,
            recurrent_block_index=drive["block_index"],
            recurrent_raw_conductance_mean=drive["raw_conductance_mean"],
            recurrent_effective_force_mean=drive["effective_force_mean"],
        )
        cfg_clean, cfg_hash = _config_provenance(cfg_dict)
        noise = {
            "noise_seed": NOISE_SEED,
            "external_input_sha256": input_hasher.sha256,
            "hashed_steps": input_hasher.n_steps,
            "definition": "sha256 over absolute step, OU xi and complete per-cell external Poisson draw",
        }
        _write_json(U1_WORK / "u1_noise_provenance.json", noise)
        summary = {
            "status": "U1_CAPTURE_COMPLETE",
            "scientific_scope": "canonical no-U-actuator source capture; no pump efficacy claim",
            "canonical_substrate": {
                "Z": "dynamic", "H": "dynamic", "X": "fixed_1_from_t0",
                "M": "off_from_t0", "U_actuator": "off", "kick": False,
                "reset": False, "parameter_step": False,
            },
            "connection_seed": CONNECTION_SEED, "noise_seed": NOISE_SEED,
            "T_ms": T_MS, "dt_ms": DT_MS, "onset_ms": onset_ms,
            "n_events": len(events),
            "n_returning_before_onset": int(sum(e["returned"] and e["t_off"] < onset_ms for e in events)),
            "lifecycle": lifecycle,
            "baseline_window_ms": [onset_ms - 4000.0, onset_ms],
            "high_reference_window_ms": [onset_ms + 1000.0, onset_ms + 4000.0],
            "E_hi_count": int(eligible.sum()), "E_hi_fraction": float(eligible.mean()),
            "r_hi_ref_hz": r_hi_ref,
            "spike_stream_sha256": stream.sha256, "n_E_spikes": int(stream.steps.size),
            "external_input_sha256": input_hasher.sha256,
            "config": cfg_clean, "config_sha256": cfg_hash,
            "raw_exact_state_hashes": {
                p.stem: json.loads(str(np.load(p, allow_pickle=False)["meta"][0]))["state_hash"]
                for p in sorted(states_dir.glob("*.npz"))
            },
            "numerical": {
                "finite": bool(np.all(np.isfinite(rate_e))),
                "clip_frac_max": float(np.max(slow_f.trace_conductance_clip_frac)),
                "refractory_ceiling_not_adjudicated_here": True,
            },
            "wall_s": time.time() - t_wall,
            "peak_rss_gib": GEO._meminfo()["self_peak_rss_gib"],
            "finished": GEO._now(),
        }
        _write_json(U1_WORK / "u1_capture_summary.json", summary)
        _write_json(U1_WORK / "U1_DONE.json", {
            "status": "DONE", "onset_ms": onset_ms, "r_hi_ref_hz": r_hi_ref,
            "spike_stream_sha256": stream.sha256, "finished": GEO._now(),
        })
        required = (
            "u1_capture_traces.npz", "u1_capture_summary.json", "u1_event_ledger.json",
            "u1_sparse_spikes.npz", "u1_rate_fields.npz", "u1_noise_provenance.json",
            "states/pre_onset.npz", "states/onset.npz", "states/onset_plus_1s.npz",
            "states/onset_plus_4s.npz", "states/late.npz", "U1_DONE.json",
        )
        missing = [r for r in required if not (U1_WORK / r).is_file()]
        if missing:
            raise RuntimeError(f"U1 artifact transaction incomplete: {missing}")
        shutil.rmtree(seconds_dir)
        (U1_WORK / "u1_sparse_spikes.bin").unlink(missing_ok=True)
        os.replace(U1_WORK, U1_FINAL)
        _write_json(OUT / "U1_DONE.json", {
            "status": "DONE", "bundle": str(U1_FINAL), "onset_ms": onset_ms,
            "r_hi_ref_hz": r_hi_ref, "finished": GEO._now(),
        })
        (OUT / "U1_RUNNING.json").unlink(missing_ok=True)
        _append_resource("U1_DONE", wall_s=summary["wall_s"])
        return summary
    except BaseException as exc:
        _write_json(OUT / "FAILED_U1.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "work_dir_retained": str(U1_WORK), "finished": GEO._now(),
        })
        (OUT / "U1_RUNNING.json").unlink(missing_ok=True)
        _append_resource("U1_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("audit", "capture"), required=True)
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    lock = _stage_lock(args.stage)
    try:
        if args.stage == "audit":
            out = stage_audit()
        else:
            if not args.confirm_run:
                raise SystemExit("LC5 U1 40k capture requires --confirm-run")
            out = stage_capture()
        print(json.dumps(json_sanitize({
            "status": out.get("status"), "onset_ms": out.get("onset_ms"),
            "r_hi_ref_hz": out.get("r_hi_ref_hz"),
        }), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
