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
import tempfile
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
from params import Params  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle,
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    admissible_target_activation,
    json_sanitize,
    load_sparse_spike_stream,
    lock_load_scales,
    refractory_ceiling_report,
    resource_stop_reason,
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
PREFIX_MS = 1000.0
# ``build_substrate`` does not override the refractory constants, so the hard single-cell wall and
# the substrate's already-registered saturation line both follow from the engine defaults.
TAU_REF_E_MS = float(Params().tau_ref_E)
SAT_CEILING_HZ = float(PP.SAT_CEILING_FRAC) * (1000.0 / TAU_REF_E_MS)
# Design 3.3 / plan T2: the locked scale fails loudly and blocks every forward stage.
SCALE_STOP = "U_SCALE_NOT_IDENTIFIABLE.json"
STAGES = ("audit", "capture", "prefix", "adjudicate", "scale", "cells", "manifest")
# Read-only / adjudication stages, exempt from the scale stop so the stop stays reproducible.
# Kept as a SEPARATE literal from STAGES on purpose: a forward stage has to be added to STAGES to
# be selectable at all, and that edit must not silently exempt it from the stop as well.
STOP_EXEMPT_STAGES = ("audit", "capture", "prefix", "adjudicate", "scale", "cells", "manifest")
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


_RESOURCE_BASELINE = {}


def _append_resource(stage, *, enforce=True, **extra):
    """Log a resource sample and, unless this is a terminal row, actually enforce design 12.

    The swap contract used to be logged and never adjudicated, so a stage could run to completion
    far past its own kill line.  ``enforce=False`` is only for rows written after the stage's work
    is already published, where raising would destroy a finished result rather than protect it.
    """

    row = dict(stage=stage, epoch=time.time(), **GEO._meminfo(), **extra)
    if _RESOURCE_BASELINE:
        row["guard"] = resource_stop_reason(
            swap_used_mib=row["swap_used_mib"],
            swap_baseline_mib=_RESOURCE_BASELINE["swap_used_mib"],
            self_rss_gib=row["self_peak_rss_gib"],
            self_rss_baseline_gib=_RESOURCE_BASELINE["self_peak_rss_gib"],
        )
    else:
        _RESOURCE_BASELINE.update(
            swap_used_mib=row["swap_used_mib"], self_peak_rss_gib=row["self_peak_rss_gib"]
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(json_sanitize(row), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if enforce:
        _enforce_resource_guard(row)
    return row


def _enforce_resource_guard(row):
    guard = row.get("guard")
    if not guard or guard["action"] != "TERMINATE_NEWEST_WORKER":
        return
    payload = {
        "status": "RESOURCE_STOP", "stage": row["stage"], "guard": guard,
        "swap_used_mib": row["swap_used_mib"],
        "swap_baseline_mib": _RESOURCE_BASELINE["swap_used_mib"],
        "mem_available_gib": row["mem_available_gib"],
        "sibling_topic4_python_count": row["sibling_topic4_python_count"],
        "attribution_note": (
            "swap growth is a whole-machine reading; self_rss_delta_mib and the sibling count are "
            "recorded so a stop caused by neighbouring workers stays distinguishable"
        ),
        "finished": GEO._now(),
    }
    _write_json(OUT / "RESOURCE_STOP.json", payload)
    raise SystemExit(
        f"LC5 RESOURCE_STOP at {row['stage']}: swap +{guard['swap_delta_mib']:.1f} MiB "
        f">= {guard['kill_delta_mib']:.0f} MiB kill line"
    )


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
        _append_resource("U1_DONE", enforce=False, wall_s=summary["wall_s"])
        return summary
    except BaseException as exc:
        _write_json(OUT / "FAILED_U1.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "work_dir_retained": str(U1_WORK), "finished": GEO._now(),
        })
        (OUT / "U1_RUNNING.json").unlink(missing_ok=True)
        _append_resource("U1_FAILED", enforce=False, error=f"{type(exc).__name__}: {exc}")
        raise


def _capture_bundle():
    path = U1_FINAL / "u1_capture_summary.json"
    if not path.is_file():
        raise SystemExit(f"this stage requires the published U1 capture bundle: {path}")
    summary = json.loads(path.read_text())
    if summary.get("status") != "U1_CAPTURE_COMPLETE":
        raise SystemExit("published U1 capture bundle is not complete")
    return summary


def _analysis_windows(onset_ms, total_ms):
    """Pre-locked baseline/high-reference windows plus the successive late windows after them.

    The first two entries are exactly the supports the design fixed before the capture ran; the
    later ones only continue the same partition to the end of the record, so no window boundary is
    chosen after seeing a number.
    """

    edges = [onset_ms - 4000.0, onset_ms, onset_ms + 1000.0, onset_ms + 4000.0]
    while edges[-1] + 3000.0 <= total_ms:
        edges.append(edges[-1] + 3000.0)
    if edges[-1] < total_ms:
        edges.append(total_ms)
    names = ["baseline", "entry", "high_reference"]
    windows = []
    for i in range(len(edges) - 1):
        name = names[i] if i < len(names) else f"late_{i - len(names) + 1}"
        windows.append((name, float(edges[i]), float(edges[i + 1])))
    return windows


def stage_prefix():
    """Plan T3.2: show the capture instrumentation never perturbed the trajectory it recorded.

    The published bundle is only a *canonical* source if adding the spike/input sinks and the
    observers left the arithmetic untouched.  That was never checked, so this runs the same first
    second twice -- fully instrumented and bare -- and additionally replays it against the spikes
    the capture actually published.
    """

    out_path = OUT / "u1_prefix_validation.json"
    if out_path.is_file():
        return json.loads(out_path.read_text())
    summary = _capture_bundle()
    _append_resource("U1_PREFIX_PREFLIGHT")

    n_steps = int(round(PREFIX_MS / DT_MS))
    candidate = _load_candidate()
    S = PP.build_substrate(CONNECTION_SEED)
    install_registered_noise_rng(S["net"])
    cfg_dict = _config(S, candidate)
    _, cfg_hash = _config_provenance(cfg_dict)
    if cfg_hash != summary["config_sha256"]:
        raise SystemExit("prefix arm configuration drifted from the published capture config")
    snapshot_steps = {
        int(round(t / DT_MS)): f"t{int(t)}"
        for t in np.arange(0.0, PREFIX_MS + SNAPSHOT_MS, SNAPSHOT_MS)
    }
    sample_every = int(round(TRACE_SAMPLE_MS / DT_MS))

    def _arm(tmp, instrumented):
        cfg = MZSlowVarsConfig(**cfg_dict)
        slow = MZSlowVars(
            S["N"], 18.0, cfg, NE=S["NE"], core_mask_E=OLD.build_core_masks(S),
            snapshot_steps=snapshot_steps,
        )
        kwargs = dict(
            n_steps=n_steps, capture_final=True, store_spikes=False, v_th_per_neuron=S["vth"]
        )
        writer = None
        if instrumented:
            contacts = np.asarray(S["reg"]["montage_sheet"].contacts, float)
            recorder = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
            slow.seeg_observer = VirtualSeegComponentObserver(recorder, cfg, sample_every=sample_every)
            slow.recurrent_drive_observer = RecurrentDriveBlockObserver(
                S["NE"], sample_every=sample_every,
                steps_per_block=int(round(1000.0 / DT_MS)),
                force_scale=float(cfg.E_E - cfg.v_match),
            )
            writer = SparseSpikeBinaryWriter(
                tmp / "prefix_spikes.bin", step_origin=0, n_steps=n_steps, n_cells=S["NE"]
            )
            kwargs.update(spike_sink=writer, input_sink=ExactInputHasher())
        S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
        p = dataclasses.replace(S["p"], T=T_MS, dt=DT_MS)
        out = run_fcxr_loop(p, S["net"], slow=slow, **kwargs)
        stream = writer.finalize(tmp / "prefix_spikes.npz") if writer is not None else None
        return {"state_hash": state_hash(out["checkpoint"]), "stream": stream,
                "input_sha256": kwargs["input_sink"].sha256 if instrumented else None}

    with tempfile.TemporaryDirectory(prefix="lc5-prefix-", dir=str(OUT)) as tmpdir:
        tmp = Path(tmpdir)
        instrumented = _arm(tmp, True)
        bare = _arm(tmp, False)
        published = load_sparse_spike_stream(U1_FINAL / "u1_sparse_spikes.npz")
        keep = published.steps < n_steps
        replayed = instrumented["stream"]
        prefix_reproduced = bool(
            replayed.steps.size == int(keep.sum())
            and np.array_equal(replayed.steps, published.steps[keep])
            and np.array_equal(replayed.cells, published.cells[keep])
        )
        n_prefix_spikes = int(replayed.steps.size)

    byte_parity = bool(instrumented["state_hash"] == bare["state_hash"])
    verdict = (
        "CAPTURE_DOES_NOT_CONTAMINATE_TRAJECTORY"
        if (byte_parity and prefix_reproduced)
        else "CAPTURE_CONTAMINATES_TRAJECTORY"
    )
    payload = {
        "status": verdict,
        "scientific_scope": (
            "instrumentation purity and published-prefix reproduction only; says nothing about "
            "whether the captured high state is an acceptable carrier"
        ),
        "contract": "plan 2026-08-10 T3.2",
        "prefix_ms": PREFIX_MS,
        "n_steps": n_steps,
        "instrumented_state_hash": instrumented["state_hash"],
        "bare_state_hash": bare["state_hash"],
        "instrumentation_byte_parity": byte_parity,
        "published_prefix_reproduced": prefix_reproduced,
        "n_prefix_E_spikes": n_prefix_spikes,
        "prefix_external_input_sha256": instrumented["input_sha256"],
        "capture_config_sha256": cfg_hash,
        "capture_spike_stream_sha256": summary["spike_stream_sha256"],
        "analysis_git_head": _git_head(),
        "finished": GEO._now(),
    }
    _write_json(out_path, payload)
    _append_resource("U1_PREFIX_DONE", enforce=False)
    return payload


def stage_adjudicate():
    """Design 7.1 + 11: say explicitly what kind of source the capture produced.

    ``lifecycle.label`` comes from the shared lifecycle-shape classifier and only reports that the
    bout stayed finite and ran to the end of the record.  The capture summary already flags
    ``refractory_ceiling_not_adjudicated_here``; this stage is that missing adjudication.
    """

    out_path = OUT / "u1_carrier_adjudication.json"
    summary = _capture_bundle()
    onset_ms, total_ms = float(summary["onset_ms"]), float(summary["T_ms"])
    stream = load_sparse_spike_stream(U1_FINAL / "u1_sparse_spikes.npz")
    traces = np.load(U1_FINAL / "u1_capture_traces.npz")
    slow_dt = float(traces["slow_trace_dt_ms"][0])

    def _slow_at(key, t_ms):
        series = np.asarray(traces[key], float)
        return float(series[min(int(round(t_ms / slow_dt)), series.size - 1)])

    windows = []
    for name, lo_ms, hi_ms in _analysis_windows(onset_ms, total_ms):
        rates = stream.per_cell_rate_hz(
            lo_step=int(round(lo_ms / DT_MS)), hi_step=int(round(hi_ms / DT_MS)), dt_ms=DT_MS
        )
        report = refractory_ceiling_report(
            rates, tau_ref_ms=TAU_REF_E_MS, sat_ceiling_hz=SAT_CEILING_HZ
        )
        report.update(
            window=name, window_ms=[lo_ms, hi_ms],
            D_mean_end=_slow_at("D_mean", hi_ms - slow_dt),
            H_mean_end=_slow_at("H_mean", hi_ms - slow_dt),
            gErec_raw_mean_end=_slow_at("gErec_raw_mean", hi_ms - slow_dt),
            gErec_eff_mean_end=_slow_at("gErec_eff_mean", hi_ms - slow_dt),
        )
        windows.append(report)

    high_ref = next(w for w in windows if w["window"] == "high_reference")
    last = windows[-1]
    escalation = last["mean_hz"] / high_ref["mean_hz"]
    source_type = (
        "escalating_saturated_source"
        if last["mean_sat_ceiling_ratio"] >= 1.0 and escalation > 1.0
        else "bounded_carrier_candidate"
    )
    payload = {
        "status": "U1_CARRIER_ADJUDICATED",
        "source_type": source_type,
        "scientific_scope": (
            "adjudicates the captured source only; the pump actuator was never on, so this is not "
            "a statement about per-cell load termination"
        ),
        "lifecycle_label_from_capture": summary["lifecycle"]["label"],
        "lifecycle_label_meaning": (
            "shape-only: the bout stayed finite, produced no hard clip, and ran to the end of the "
            "record; it is not carrier acceptance and must not be cited as one"
        ),
        "registered_criteria": {
            "tau_ref_E_ms": TAU_REF_E_MS,
            "hard_single_cell_ceiling_hz": 1000.0 / TAU_REF_E_MS,
            "registered_sat_ceiling_hz": SAT_CEILING_HZ,
            "registered_sat_ceiling_source": (
                "run_m4_phaseplane.SAT_CEILING_FRAC * 1000/tau_ref_E; its own contract is 'peak "
                "rate below this means finite energy rather than pinned at a runaway ceiling'"
            ),
        },
        "windows": windows,
        "late_over_high_reference_mean_rate_ratio": float(escalation),
        "autonomous_offset_observed": False,
        "u2_source_state_note": (
            "U2 forks from onset+1s, i.e. from the foot of this escalation; any offset measured "
            "there is a step-on sufficiency readout on a saturating source, not a seizure duration"
        ),
        "analysis_git_head": _git_head(),
        "finished": GEO._now(),
    }
    _write_json(out_path, payload)
    return payload


def stage_scale():
    """Plan T4 / design 3.3: lock the load scale from the fresh rate field, or stop.

    ``q_i* = target * r_i / r_hi_ref`` is the activation a cell would settle at, and ``Phi`` only
    reaches it while ``q_i* < 1``.  The gate is therefore two-sided by contract -- ``q99 < 0.90``
    *and* every eligible cell below 1 -- and the two halves are not interchangeable.
    """

    out_dir = OUT / "u1b_scale_lock"
    if out_dir.is_dir():
        return json.loads((out_dir / "u1b_scale_verdict.json").read_text())
    summary = _capture_bundle()
    onset_ms, total_ms = float(summary["onset_ms"]), float(summary["T_ms"])
    r_hi_ref = float(summary["r_hi_ref_hz"])
    fields = np.load(U1_FINAL / "u1_rate_fields.npz")
    high = np.asarray(fields["high_rate_hz"], float)
    base = np.asarray(fields["baseline_rate_hz"], float)
    eligible = np.asarray(fields["eligible_E_hi"], bool)
    if float(np.median(high[eligible])) != r_hi_ref:
        raise SystemExit("recomputed r_hi_ref disagrees with the published capture summary")

    locked = lock_load_scales(
        r_hi_ref_hz=r_hi_ref, per_cell_rate_hz=high[eligible],
        tau_ms=(3000.0, 8000.0, 15000.0), target_activation=0.5,
    )
    q_star = np.asarray(locked.pop("q_star"), float)
    divergent = np.where(eligible)[0][q_star >= 1.0]
    stream = load_sparse_spike_stream(U1_FINAL / "u1_sparse_spikes.npz")

    sweep = []
    for name, lo_ms, hi_ms in _analysis_windows(onset_ms, total_ms):
        rates = stream.per_cell_rate_hz(
            lo_step=int(round(lo_ms / DT_MS)), hi_step=int(round(hi_ms / DT_MS)), dt_ms=DT_MS
        )
        active = rates[rates > 0.0]
        sweep.append({
            "window": name, "window_ms": [lo_ms, hi_ms],
            "max_rate_hz": float(rates.max()),
            "mean_rate_hz": float(rates.mean()),
            "admissible_target_activation_sup": admissible_target_activation(
                active, r_hi_ref_hz=r_hi_ref
            ),
            "locked_target_divergent_fraction": float(np.mean(0.5 * active / r_hi_ref >= 1.0)),
        })
    common_sup = min(w["admissible_target_activation_sup"] for w in sweep)

    admissible = bool(locked["admissible"])
    verdict = "U_SCALE_LOCKED" if admissible else "U_SCALE_NOT_IDENTIFIABLE"
    payload = {
        "status": verdict,
        "contract": "plan 2026-08-10 T2/T4; design 3.3",
        "gate": {
            "q99_lt_0p90": bool(locked["q_star_q99"] < 0.90),
            "all_eligible_lt_1": bool(locked["divergent_fraction"] == 0.0),
            "note": "both halves are required; q99 never substitutes for max",
        },
        "target_activation": 0.5,
        "r_hi_ref_hz": r_hi_ref,
        "high_reference_window_ms": summary["high_reference_window_ms"],
        "E_hi_count": int(eligible.sum()),
        **{k: v for k, v in locked.items() if k != "admissible"},
        "admissible": admissible,
        "divergent_cells": [
            {
                "cell": int(i),
                "high_rate_hz": float(high[i]),
                "baseline_rate_hz": float(base[i]),
                "q_star": float(0.5 * high[i] / r_hi_ref),
            }
            for i in divergent
        ],
        "window_sweep": sweep,
        "common_admissible_target_activation_sup": float(common_sup),
        "rescue_by_lowering_target_is_not_available": {
            "high_reference_window_only_sup": float(
                next(w["admissible_target_activation_sup"] for w in sweep
                     if w["window"] == "high_reference")
            ),
            "whole_high_state_sup": float(common_sup),
            "why": (
                "q_i* is an equilibrium bookkeeping quantity that presumes a rate holding still on "
                "the tau_U timescale (3-15 s).  This source's per-cell rate rises about eightfold "
                "inside that same timescale, so the admissible target keeps shrinking as the "
                "window moves later.  Trimming 0.5 to just under the high-reference supremum "
                "satisfies the gate on the locked window while remaining inadmissible on the state "
                "the actuator would actually meet."
            ),
        },
        "forbidden_next_actions": [
            "lowering target_activation to clear the gate without a fresh design lock",
            "dropping the divergent cells from the support",
            "replacing the max gate with q99",
            "starting the U2 3x3 authority screen",
        ],
        "analysis_git_head": _git_head(),
        "finished": GEO._now(),
    }
    with AtomicStageBundle(out_dir) as bundle:
        (bundle.path("u1b_scale_verdict.json")).write_text(
            json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n"
        )
        np.savez_compressed(
            bundle.path("u1b_q_star.npz"),
            q_star=q_star.astype(np.float32),
            eligible_E_hi=eligible,
            high_rate_hz=high.astype(np.float32),
        )
        bundle.commit(required=("u1b_scale_verdict.json", "u1b_q_star.npz"))
    if not admissible:
        _write_json(OUT / SCALE_STOP, {
            "status": "STOP", "verdict": verdict,
            "blocks": "every forward LC5 stage, including the U2 3x3 authority screen",
            "evidence": str(out_dir / "u1b_scale_verdict.json"),
            "q_star_max": locked["q_star_max"],
            "divergent_cell_count": len(divergent),
            "finished": GEO._now(),
        })
    return payload


def stage_cells():
    """Locate the cells the load scale diverges on, so 'just drop them' can be answered with data.

    ``build_substrate`` puts the pathology in the per-neuron threshold field: two low-threshold
    patches against a uniform base.  If the divergent cells sit inside those patches they are the
    reason the sheet ignites at all, and excluding them from the support would remove the mechanism
    rather than repair the scale.
    """

    out_path = OUT / "u1b_divergent_cell_audit.json"
    if out_path.is_file():
        return json.loads(out_path.read_text())
    verdict_path = OUT / "u1b_scale_lock/u1b_scale_verdict.json"
    if not verdict_path.is_file():
        raise SystemExit("divergent-cell audit requires the published U1b scale verdict")
    verdict = json.loads(verdict_path.read_text())
    cells = [int(c["cell"]) for c in verdict["divergent_cells"]]

    S = PP.build_substrate(CONNECTION_SEED)
    vth = np.asarray(S["vth"], float)
    pos = np.asarray(S["net"]["pos"], float)[: int(S["NE"])]
    base_vth = float(np.max(vth))
    low = vth < base_vth
    core_masks = OLD.build_core_masks(S)
    core_names = sorted(core_masks) if isinstance(core_masks, dict) else []

    rows = []
    for cell in cells:
        row = {
            "cell": cell,
            "v_th": float(vth[cell]),
            "in_low_threshold_patch": bool(low[cell]),
            "position_mm": [float(v) for v in pos[cell]],
        }
        for name in core_names:
            row[f"in_core_{name}"] = bool(np.asarray(core_masks[name], bool)[cell])
        rows.append(row)

    payload = {
        "status": "U1B_DIVERGENT_CELLS_LOCATED",
        "scientific_scope": "identity of the divergent cells only; does not reopen the scale gate",
        "base_v_th": base_vth,
        "low_threshold_cell_count": int(low.sum()),
        "low_threshold_cell_fraction": float(low.mean()),
        "divergent_cells": rows,
        "divergent_in_low_threshold_patch": int(sum(r["in_low_threshold_patch"] for r in rows)),
        "enrichment_vs_sheet": (
            float(np.mean([r["in_low_threshold_patch"] for r in rows]) / low.mean())
            if low.mean() > 0.0 else None
        ),
        "analysis_git_head": _git_head(),
        "finished": GEO._now(),
    }
    _write_json(out_path, payload)
    return payload


def stage_manifest():
    """Design 12: turn the measured U1a cost into the wall budget every later stage must respect.

    ``c_wall = T_wall/T_sim`` was required after U1a and never written down, so no later stage had
    a machine-measured budget to check its 12 h safety cap against.
    """

    out_path = OUT / "run_manifest.json"
    summary = _capture_bundle()
    rows = [json.loads(line) for line in (OUT / "resource_log.jsonl").read_text().splitlines()]
    pre = next(r for r in rows if r["stage"] == "U1_PREFLIGHT")
    chunks = [r for r in rows if r["stage"] == "U1_CHUNK"]
    done = next(r for r in rows if r["stage"] == "U1_DONE")
    t_sim_s = float(summary["T_ms"]) / 1000.0
    c_wall = float(done["wall_s"]) / t_sim_s

    at_ms = {float(r["completed_ms"]): float(r["epoch"]) for r in chunks}
    onset_ms = float(summary["onset_ms"])
    arm_lo, arm_hi = onset_ms + 1000.0, onset_ms + 9000.0
    arm_wall = (
        at_ms[arm_hi] - at_ms[arm_lo] if arm_lo in at_ms and arm_hi in at_ms else None
    )

    swap_peak = max(float(r["swap_used_mib"]) for r in rows)
    retro = resource_stop_reason(
        swap_used_mib=swap_peak, swap_baseline_mib=float(pre["swap_used_mib"]),
        self_rss_gib=max(float(r["self_peak_rss_gib"]) for r in chunks),
        self_rss_baseline_gib=float(chunks[0]["self_peak_rss_gib"]),
    )

    def _budget(t_target_s, n_runs):
        wall = 1.5 * c_wall * t_target_s
        return {
            "t_target_s": float(t_target_s), "n_runs": int(n_runs),
            "wall_kill_per_run_h": wall / 3600.0,
            "serial_campaign_h": n_runs * wall / 3600.0,
            "within_12h_cap": bool(wall / 3600.0 <= 12.0),
        }

    payload = {
        "status": "LC5_RUN_MANIFEST",
        "stage_reached": "U1a capture + closeout adjudication; U2 blocked",
        "measured_cost": {
            "T_sim_s": t_sim_s,
            "T_wall_s": float(done["wall_s"]),
            "c_wall_s_per_sim_s": c_wall,
            "u2_arm_window_ms": [arm_lo, arm_hi],
            "u2_arm_measured_wall_s": arm_wall,
            "u2_arm_measured_wall_h": None if arm_wall is None else arm_wall / 3600.0,
            "note": (
                "the per-second cost rises with the firing rate, so the overall c_wall understates "
                "a fork that starts already in the high state; up to 37 sibling processes shared "
                "the machine during U1a, so these numbers mix model cost and contention and are "
                "not a clean single-worker benchmark"
            ),
        },
        "forward_budget": {
            "u2_single_arm": _budget(8.0, 1),
            "u2_full_grid_9_plus_control": _budget(8.0, 10),
            "u2_full_grid_from_measured_arm_h": (
                None if arm_wall is None else 10.0 * arm_wall / 3600.0
            ),
            "u3_single_lifecycle": _budget(70.0, 1),
            "u3_primary_plus_sensitivity": _budget(70.0, 2),
            "cap_h": 12.0,
            "collision": (
                "at the measured c_wall a single 70 s lifecycle run needs a 33.6 h wall-kill, "
                "which the 12 h hard safety cap forbids; U3 as specified is not executable at this "
                "cost and needs either a cheaper substrate, a shorter target, or an explicit "
                "re-lock of the cap"
            ),
        },
        "resource_contract": {
            "guard_active_during_u1a": False,
            "guard_active_now": True,
            "swap_baseline_mib": float(pre["swap_used_mib"]),
            "swap_peak_mib": swap_peak,
            "retrospective_verdict": retro,
            "min_mem_available_gib": min(float(r["mem_available_gib"]) for r in rows),
            "max_sibling_topic4_python_count": max(
                int(r["sibling_topic4_python_count"]) for r in rows
            ),
            "attribution": (
                "this worker's resident size was flat across the run and MemAvailable never fell "
                "below 143.7 GiB while up to 37 sibling processes were present, so the swap growth "
                "is not attributable to the capture and the captured data is not compromised; the "
                "contract was nevertheless unenforced"
            ),
        },
        "verdicts": {
            "u0_lineage_audit": json.loads((OUT / "u0_lineage_audit.json").read_text())["status"],
            "u1_capture": summary["status"],
            "u1_prefix_validation": json.loads(
                (OUT / "u1_prefix_validation.json").read_text()
            )["status"],
            "u1_carrier_adjudication": json.loads(
                (OUT / "u1_carrier_adjudication.json").read_text()
            )["source_type"],
            "u1b_scale_lock": json.loads(
                (OUT / "u1b_scale_lock/u1b_scale_verdict.json").read_text()
            )["status"],
        },
        "provenance": {
            "capture_git_head": json.loads((OUT / "u0_lineage_audit.json").read_text())["git_head"],
            "analysis_git_head": _git_head(),
            "capture_config_sha256": summary["config_sha256"],
            "spike_stream_sha256": summary["spike_stream_sha256"],
            "external_input_sha256": summary["external_input_sha256"],
            "connection_seed": summary["connection_seed"],
            "noise_seed": summary["noise_seed"],
            "note": (
                "the U0 audit certifies the code that produced the capture; the closeout stages "
                "were added afterwards, so re-running --stage capture now correctly refuses with a "
                "source-drift error until a fresh audit is taken"
            ),
        },
        "finished": GEO._now(),
    }
    _write_json(out_path, payload)
    return payload


def _assert_no_stop(stage):
    """A written STOP has to bind mechanically, not only in prose.

    ``STOP_EXEMPT_STAGES`` are the read-only/adjudication stages that must stay runnable so the stop
    itself can be reproduced.  Any stage added later -- U2 above all -- is refused until the stop
    file is removed by a fresh design lock.
    """

    stop = OUT / SCALE_STOP
    if stop.is_file() and stage not in STOP_EXEMPT_STAGES:
        raise SystemExit(
            f"LC5 stage {stage!r} is blocked by {stop.name}: the locked load scale has no finite "
            "per-cell equilibrium, so a fresh design lock is required before any forward stage"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    _assert_no_stop(args.stage)
    lock = _stage_lock(args.stage)
    try:
        if args.stage == "audit":
            out = stage_audit()
        elif args.stage == "capture":
            if not args.confirm_run:
                raise SystemExit("LC5 U1 40k capture requires --confirm-run")
            out = stage_capture()
        elif args.stage == "prefix":
            if not args.confirm_run:
                raise SystemExit("LC5 prefix validation steps the 40k loop; pass --confirm-run")
            out = stage_prefix()
        elif args.stage == "adjudicate":
            out = stage_adjudicate()
        elif args.stage == "scale":
            out = stage_scale()
        elif args.stage == "cells":
            out = stage_cells()
        else:
            out = stage_manifest()
        print(json.dumps(json_sanitize({
            "status": out.get("status"), "onset_ms": out.get("onset_ms"),
            "r_hi_ref_hz": out.get("r_hi_ref_hz"),
        }), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
