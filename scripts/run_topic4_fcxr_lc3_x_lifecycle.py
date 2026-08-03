#!/usr/bin/env python3
"""FCXR-LC3 E6 conditional X calibration and E7 <=6 no-kick lifecycle runs."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_recon as RECON  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import (  # noqa: E402
    clone_loop_state,
    replace_frozen_fields,
    run_fcxr_loop,
)
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    H1_POINT_ID,
    load_prepared_checkpoint,
)
from src.topic4_fcxr_lc3_xcal import (  # noqa: E402
    choose_calibration_family,
    lifecycle_candidate_gate,
    multivariate_statistical_return,
    relay_x_inf,
    return_brackets,
    select_x_candidates,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402


OUT = E01.OUT
LOCK = os.path.join(OUT, "x_lifecycle_execution_lock.json")
XCAL = os.path.join(OUT, "x_calibration.json")
LIFECYCLE_MANIFEST = os.path.join(OUT, "lifecycle_manifest.json")
LIFECYCLE_VERDICT = os.path.join(OUT, "lifecycle_verdict.json")
CELL_DIR = os.path.join(OUT, "x_calibration_cells")
LIFE_DIR = os.path.join(OUT, "lifecycle_cells")
DT = E01.DT
NOISES = (401, 405, 406)
T_LOW_MS = 8000.0
T_HIGH_MS = 4000.0
T1_MS = 32000.0
T_CAP_MS = 45000.0
SNAP_MS = 250.0
SOURCES = (
    "src/topic4_fcxr_lc3.py",
    "src/topic4_fcxr_lc3_geometry.py",
    "src/topic4_fcxr_lc3_recon.py",
    "src/topic4_fcxr_lc3_xcal.py",
    "src/snn_engine/mz_slow_vars.py",
    "scripts/run_topic4_fcxr_lc3_x_lifecycle.py",
    "scripts/run_topic4_fcxr_lc3_x_lifecycle_autopilot.sh",
    "docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md",
    "docs/superpowers/plans/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability.md",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(
        mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
        swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
        self_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    )


def _wait_submission(swap_baseline_mib):
    while True:
        mem = _meminfo()
        if (mem["swap_used_mib"] - float(swap_baseline_mib) < 256.0
                and mem["mem_available_gib"] >= 96.0):
            return
        time.sleep(30.0)


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".{name}.lock"), "a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"X/lifecycle stage already running: {name}") from exc
        yield


def _primary_d_fields():
    dlock = _load(os.path.join(OUT, "d_field_lock.json"))
    rec = dlock["families"]["seed1_q75"]
    with np.load(rec["output_path"]) as z:
        labels = [str(v) for v in z["labels"]]
        fields = {label: z["D_fields"][i].astype(float) for i, label in enumerate(labels)}
    return fields


def _boundary_adjudication():
    geometry_path = os.path.join(OUT, "geometry_map.json")
    if not os.path.isfile(geometry_path):
        return dict(status="X_CALIBRATION_NOT_IDENTIFIABLE",
                    reason="geometry_map_missing_or_unresolved", brackets=[])
    geometry = _load(geometry_path)
    dmeans = {"D_healthy": 0.0, **_load(os.path.join(OUT, "d_field_targets.json"))["target_means_D"]}
    brackets = return_brackets(geometry["rows"], dmeans)
    if not brackets:
        return dict(status="X_CALIBRATION_NOT_IDENTIFIABLE",
                    reason="no_same_D_high_start_return_survival_bracket", brackets=[])
    target = float(dmeans["D50"])
    selected = min(brackets, key=lambda row: (abs(row["mean_D"] - target), row["d_label"]))
    return dict(status="BOUNDARY_IDENTIFIED", brackets=brackets, selected_boundary=selected)


def cmd_lock(_args):
    required = [
        os.path.join(RECON.OUT, "aggregate.json"),
        os.path.join(OUT, "SPATIAL_PROBE_DONE.json"),
        os.path.join(OUT, "d_field_lock.json"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise SystemExit("missing E6 prerequisites: " + ", ".join(missing))
    initial = _load(os.path.join(OUT, "execution_lock.json"))
    adjudication = _boundary_adjudication()
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-x-lifecycle-lock-1.0",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip(),
        adjudication=adjudication,
        inputs={path: _sha(path) for path in required},
        sources={rel: _sha(os.path.join(ROOT, rel)) for rel in SOURCES},
        engine_hashes=initial["engine_hashes"], resource_at_lock=_meminfo(),
        maximum_calibration_grid="3x3", maximum_lifecycle_runs=6,
        lifecycle_no_kick=True, lifecycle_no_reset=True,
        lifecycle_no_parameter_step=True, locked_at=_now(),
    )
    _write_json(LOCK, payload)
    print(json.dumps(dict(status="LOCKED", adjudication=adjudication), indent=2))


def _assert_lock():
    lock = _load(LOCK)
    if lock.get("status") != "LOCKED":
        raise SystemExit("invalid X/lifecycle lock")
    for path, expected in lock["inputs"].items():
        if _sha(path) != expected:
            raise SystemExit(f"X/lifecycle input drift: {path}")
    for rel, expected in lock["sources"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"X/lifecycle source drift: {rel}")
    for rel, expected in lock["engine_hashes"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"engine drift: {rel}")
    return lock


def _candidate_cfg(base_cfg, candidate):
    cfg = base_cfg
    cfg.use_x = True
    cfg.x_relay_frozen_E = None
    cfg.x_min = float(candidate.get("x_min", 0.1))
    cfg.y_gate = float(candidate["y_gate"])
    cfg.K_y = float(candidate["K_y"])
    cfg.hill_n = int(candidate.get("hill_n", 4))
    cfg.tau_y = float(candidate.get("tau_y", 120.0))
    cfg.tau_x_down = float(candidate["tau_x_down"])
    cfg.tau_x_up = float(candidate.get("tau_x_up", 5000.0))
    return cfg


def _activate_x(state, candidate):
    child = clone_loop_state(state)
    _candidate_cfg(child.slow.cfg, candidate)
    child.slow.x_relay[:] = 1.0
    child.slow.ee_relay_send[:] = 1.0
    child.slow.y[:] = 0.0
    child.slow.trace_x_relay_mean = []
    child.slow.trace_x_relay_min = []
    child.slow.trace_y_mean = []
    child.slow.trace_y_max = []
    return child


def _calibration_states(boundary):
    low_pkl, low_js = GEO._prep_paths(H1_POINT_ID, "low")
    high_pkl, high_js = GEO._prep_paths(H1_POINT_ID, "high")
    low_meta, high_meta = _load(low_js), _load(high_js)
    if (low_meta.get("status") != "ACCEPTED_CANONICAL_STATE"
            or high_meta.get("status") != "ACCEPTED_CANONICAL_STATE"):
        raise RuntimeError("accepted H1 low/high states are required for X calibration")
    low = load_prepared_checkpoint(
        low_pkl, expected_file_sha256=low_meta["checkpoint"]["file_sha256"])["state"]
    high = load_prepared_checkpoint(
        high_pkl, expected_file_sha256=high_meta["checkpoint"]["file_sha256"])["state"]
    field = _primary_d_fields()[boundary["d_label"]]
    high = replace_frozen_fields(high, d_field=field, x_field=np.ones(high.slow.NE))
    return low, high


def _calibration_run(S, low_base, high_base, candidate, boundary):
    low = _activate_x(low_base, candidate)
    high = _activate_x(high_base, candidate)
    p_low = dataclasses.replace(S["p"], T=T_LOW_MS, dt=DT)
    p_high = dataclasses.replace(S["p"], T=T_HIGH_MS, dt=DT)
    low_out = run_fcxr_loop(
        p_low, S["net"], start=low, n_steps=int(round(T_LOW_MS / DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    high_out = run_fcxr_loop(
        p_high, S["net"], start=high, n_steps=int(round(T_HIGH_MS / DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = _load(E01.ARTIFACTS["lc1_baseline"])
    point = GEO._point(H1_POINT_ID)
    low_cls = GEO._tail_observables(
        low_out, S, point, tail_ms=2000.0, analysis_start_ms=T_LOW_MS - 2000.0)
    high_cls = GEO._tail_observables(
        high_out, S, point, tail_ms=1000.0, analysis_start_ms=T_HIGH_MS - 1000.0)
    events, _af, _bin, _floor, _rate = OLD._events_from_res(
        low_out, DT, event_bar=float(baseline["frozen_event_bar"]))
    returning = [event for event in events if event.get("returned", False)]
    low_slow = low_out["checkpoint"].slow
    high_slow = high_out["checkpoint"].slow
    low_x = np.asarray(low_slow.trace_x_relay_mean[-low_out["n_steps"]:], float)
    high_x = np.asarray(high_slow.trace_x_relay_mean[-high_out["n_steps"]:], float)
    crossing = np.flatnonzero(high_x <= float(boundary["a_off_midpoint"]))
    crossing_ms = float(crossing[0] * DT) if crossing.size else None
    all_clips = list(low_slow.trace_conductance_clip_frac[-low_out["n_steps"]:]) \
        + list(high_slow.trace_conductance_clip_frac[-high_out["n_steps"]:])
    safe = bool(
        np.all(np.isfinite(low_out["rate_E"])) and np.all(np.isfinite(high_out["rate_E"]))
        and max(all_clips or [0.0]) == 0.0)
    burn = int(round(2000.0 / DT))
    row = dict(
        candidate_id=candidate["candidate_id"], config=candidate,
        numerical_safe=safe, low_label=low_cls["label"], high_label=high_cls["label"],
        n_low_returning_events=len(returning),
        ied_mean_a_x=float(np.mean(low_x[burn:])) if low_x.size > burn else float(np.mean(low_x)),
        high_min_a_x=float(np.min(high_x)), crossing_time_ms=crossing_ms,
        high_returned_to_low=high_cls["label"] == "INTERICTAL_WORKPOINT",
        final_mean_y=float(np.mean(high_slow.y)),
        inferred_final_mean_x_inf=float(np.mean(relay_x_inf(
            high_slow.y, y_gate=candidate["y_gate"], K_y=candidate["K_y"],
            hill_n=candidate.get("hill_n", 4), x_min=candidate.get("x_min", 0.1)))),
        low_max_rate_hz=float(np.max(low_out["rate_E"])),
        high_end_rate_hz=float(np.mean(high_out["rate_E"][-int(round(1000.0 / DT)):])),
    )
    del low_out, high_out
    gc.collect()
    return row


def _candidate_grid(family, sensor):
    base_gate = float(sensor["y_gate_q999"])
    if family == "HILL_MIDPOINT_AND_RISE_TIME":
        return [dict(
            candidate_id=f"gain{gain:g}_td{int(tau)}",
            effective_sensor_gain=float(gain),
            gain_implementation="K_y_eff=K_y_reference/gain",
            y_gate=base_gate, K_y=5.0 / float(gain),
            tau_x_down=float(tau), tau_x_up=5000.0,
            x_min=0.1, tau_y=120.0, hill_n=4,
        ) for gain in (0.5, 1.0, 2.0) for tau in (250.0, 500.0, 1000.0)]
    if family == "BOUNDARY_ALREADY_REACHED_NO_RECALIBRATION_NEEDED":
        return [dict(
            candidate_id="current_unretuned", effective_sensor_gain=1.0,
            gain_implementation="current_config", y_gate=base_gate, K_y=5.0,
            tau_x_down=500.0, tau_x_up=5000.0,
            x_min=0.1, tau_y=120.0, hill_n=4,
        )]
    return []


def cmd_calibrate(args):
    if not args.confirm_run:
        raise SystemExit("40k X calibration requires --confirm-run")
    lock = _assert_lock()
    adjudication = lock["adjudication"]
    if adjudication["status"] != "BOUNDARY_IDENTIFIED":
        payload = dict(**adjudication, selected_candidates=[], completed=_now())
        _write_json(XCAL, payload)
        print(json.dumps(payload, indent=2)); return
    if _meminfo()["mem_available_gib"] < 128.0:
        raise SystemExit("X calibration requires 128 GiB MemAvailable")
    boundary = adjudication["selected_boundary"]
    S = PP.build_substrate(1)
    low, high = _calibration_states(boundary)
    sensor = _load(E01.ARTIFACTS["lc1_sensor"])
    current = _candidate_grid("BOUNDARY_ALREADY_REACHED_NO_RECALIBRATION_NEEDED", sensor)[0]
    probe = _calibration_run(S, low, high, current, boundary)
    family = choose_calibration_family(
        observed_x=probe["high_min_a_x"],
        inferred_x_inf=probe["inferred_final_mean_x_inf"],
        a_return_max=boundary["a_return_max"], a_survive_min=boundary["a_survive_min"])
    grid = _candidate_grid(family, sensor)
    if not grid:
        payload = dict(
            status="X_CALIBRATION_NOT_IDENTIFIABLE",
            reason="requested_sensor_gain_plus_midpoint_is_not_identifiable_in_current_X_equation",
            boundary=boundary, current_probe=probe, routed_family=family,
            selected_candidates=[], completed=_now())
        _write_json(XCAL, payload)
        print(json.dumps(payload, indent=2)); return
    os.makedirs(CELL_DIR, exist_ok=True)
    rows = []
    with _stage_lock("x_calibration"):
        swap0 = _meminfo()["swap_used_mib"]
        for candidate in grid:
            path = os.path.join(CELL_DIR, f"{candidate['candidate_id']}.json")
            done = path.replace(".json", ".DONE.json")
            if os.path.isfile(path) and os.path.isfile(done) \
                    and _load(done).get("output_sha256") == _sha(path):
                rows.append(_load(path)); continue
            _wait_submission(swap0)
            row = _calibration_run(S, low, high, candidate, boundary)
            _write_json(path, row)
            _write_json(done, dict(status="DONE", output_sha256=_sha(path), finished=_now()))
            rows.append(row)
            print(f"[xcal] {candidate['candidate_id']} xIED={row['ied_mean_a_x']:.3f} "
                  f"cross={row['crossing_time_ms']} return={row['high_returned_to_low']}", flush=True)
    selected = select_x_candidates(rows)
    payload = dict(
        status="X_CANDIDATES_SELECTED" if selected else "X_CALIBRATION_BOUNDED_NEGATIVE",
        boundary=boundary, current_probe=probe, routed_family=family,
        grid=grid, rows=rows,
        selected_candidates=[row["config"] for row in selected],
        selection_gate=("numerical safe; low state remains interictal with >=3 returning IEDs; "
                        "IED mean aX>0.9; cross empirical a_off in 1-3s; high state returns "
                        "to interictal tail"),
        completed=_now(),
    )
    _write_json(XCAL, payload)
    print(json.dumps(dict(status=payload["status"], routed_family=family,
                          selected=[c["candidate_id"] for c in payload["selected_candidates"]]), indent=2))


def cmd_lifecycle_manifest(_args):
    _assert_lock()
    xcal = _load(XCAL)
    candidates = xcal.get("selected_candidates", [])[:2]
    rows = []
    for candidate in candidates:
        for noise in NOISES:
            row_id = f"{candidate['candidate_id']}__noise{noise}"
            rows.append(dict(
                index=len(rows), row_id=row_id, candidate=candidate,
                connection_seed=1, noise_seed=noise, no_kick=True, no_reset=True,
                no_parameter_step=True, output_path=os.path.join(LIFE_DIR, f"{row_id}.json"),
                done_path=os.path.join(LIFE_DIR, f"{row_id}.DONE.json"),
            ))
    payload = dict(
        status="LOCKED" if rows else "NO_CALIBRATED_X_CANDIDATE",
        n_rows=len(rows), rows=rows, x_calibration_sha256=_sha(XCAL),
        maximum_rows=6, created=_now())
    _write_json(LIFECYCLE_MANIFEST, payload)
    print(json.dumps(dict(status=payload["status"], n_rows=len(rows)), indent=2))


def _event_features(events, spikes, S, *, lo_ms, hi_ms):
    selected = [event for event in events
                if float(event["t_on"]) >= lo_ms and float(event["t_off"]) <= hi_ms
                and event.get("returned", False)]
    masks = GEO._region_masks(S)
    rows = []
    for event in selected:
        i0 = max(0, int(round(float(event["t_on"]) / DT)))
        i1 = min(spikes.shape[0], int(round(float(event["t_off"]) / DT)) + 1)
        seg = spikes[i0:i1]
        active = seg.any(axis=0)
        if not active.any():
            continue
        pos = np.asarray(S["posE"])[active]
        center = pos.mean(axis=0)
        compact = float(np.sqrt(np.mean(np.sum((pos - center) ** 2, axis=1))))
        def first(mask):
            sub = seg[:, mask]
            return float(np.argmax(sub.any(axis=1)) * DT) if sub.any() else np.inf
        fa, fb = first(masks["core_A"]), first(masks["core_B"])
        polarity = "A" if fa < fb else ("B" if fb < fa else "tie")
        rows.append(dict(
            t_on=float(event["t_on"]), duration_ms=float(event["dur_ms"]),
            participation=float(active.mean()), compactness_mm=compact, polarity=polarity))
    duration_s = max((hi_ms - lo_ms) * 1e-3, 1e-9)
    onsets = np.asarray([row["t_on"] for row in rows], float)
    iei = np.diff(onsets) if onsets.size >= 2 else np.asarray([])
    def median(key):
        return float(np.median([row[key] for row in rows])) if rows else None
    return dict(
        n_events=len(rows), event_rate_hz=len(rows) / duration_s,
        median_iei_ms=float(np.median(iei)) if iei.size else None,
        median_duration_ms=median("duration_ms"),
        median_participation=median("participation"),
        median_compactness_mm=median("compactness_mm"),
        fraction_A=(sum(row["polarity"] == "A" for row in rows) / len(rows) if rows else None),
        rows=rows,
    )


def _run_lifecycle_row(row):
    if os.path.isfile(row["output_path"]) and os.path.isfile(row["done_path"]):
        if _load(row["done_path"]).get("output_sha256") == _sha(row["output_path"]):
            return _load(row["output_path"])
    before = _meminfo()
    if before["mem_available_gib"] < 128.0:
        raise RuntimeError("lifecycle row requires 128 GiB MemAvailable")
    S = PP.build_substrate(1)
    point = GEO._point(H1_POINT_ID)
    cfg = E01._dynamic_cfg(point)
    candidate = row["candidate"]
    cfg.update(
        K_y=float(candidate["K_y"]), y_gate=float(candidate["y_gate"]),
        tau_x_down=float(candidate["tau_x_down"]),
        tau_x_up=float(candidate.get("tau_x_up", 5000.0)),
        x_min=float(candidate.get("x_min", 0.1)),
    )
    snapshot_steps = {int(round(t / DT)): f"t{int(t)}"
                      for t in np.arange(0.0, T_CAP_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
        core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(int(row["noise_seed"]))
    baseline = _load(E01.ARTIFACTS["lc1_baseline"])
    t0 = time.time()
    p1 = dataclasses.replace(S["p"], T=T1_MS, dt=DT)
    first = run_fcxr_loop(
        p1, S["net"], slow=slow, n_steps=int(round(T1_MS / DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    wins1, _num1, _ = LC1R._reduce_run_windows(
        first, first["checkpoint"].slow, S, DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle1 = classify_lifecycle(wins1, baseline["band"])
    extend = lifecycle1.get("bout") is not None
    if extend:
        extra = T_CAP_MS - T1_MS
        second = run_fcxr_loop(
            dataclasses.replace(S["p"], T=extra, dt=DT), S["net"],
            start=first["checkpoint"], n_steps=int(round(extra / DT)),
            capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
        rate_e = np.concatenate([first["rate_E"], second["rate_E"]])
        rate_i = np.concatenate([first["rate_I"], second["rate_I"]])
        spikes = np.concatenate([first["E_spk_bool"], second["E_spk_bool"]], axis=0)
        final, total_ms = second, T_CAP_MS
    else:
        rate_e, rate_i, spikes = first["rate_E"], first["rate_I"], first["E_spk_bool"]
        final, total_ms = first, T1_MS
    res = dict(rate_E=rate_e, rate_I=rate_i, E_spk_bool=spikes)
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, final["checkpoint"].slow, S, DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, _af, _bin, _floor, _rate = OLD._events_from_res(
        res, DT, event_bar=float(baseline["frozen_event_bar"]))
    bout = lifecycle.get("bout")
    onset_ms = float(bout[0] * baseline["band"]["win_ms"]) if bout is not None else None
    offset_ms = float((bout[1] + 1) * baseline["band"]["win_ms"]) if bout is not None else None
    high_ms = (offset_ms - onset_ms) if bout is not None else None
    pre = (_event_features(events, spikes, S, lo_ms=onset_ms - 8000.0, hi_ms=onset_ms)
           if onset_ms is not None and onset_ms >= 8000.0 else None)
    post = (_event_features(events, spikes, S, lo_ms=total_ms - 8000.0, hi_ms=total_ms)
            if offset_ms is not None and total_ms - offset_ms >= 8000.0 else None)
    rest = multivariate_statistical_return(pre, post)
    xtrace = np.asarray(final["checkpoint"].slow.trace_x_relay_mean, float)
    x_peak_ms = float(np.argmin(xtrace) * DT) if xtrace.size else None
    x_after = bool(onset_ms is not None and x_peak_ms is not None and x_peak_ms >= onset_ms)
    early_post_rate = (float(np.mean(rate_e[int(round(offset_ms / DT)):
                                                 int(round((offset_ms + 1000.0) / DT))]))
                       if offset_ms is not None and offset_ms + 1000.0 <= total_ms else None)
    pre_rate = pre["event_rate_hz"] if pre is not None else None
    postictal = bool(early_post_rate is not None and pre_rate is not None
                     and early_post_rate <= max(float(baseline["band"]["roll_hi"]), 0.5 * pre_rate))
    ceiling = 0.0
    if bout is not None:
        i0 = int(round(onset_ms / DT)); i1 = min(spikes.shape[0], int(round(offset_ms / DT)))
        if i1 > i0:
            hz = spikes[i0:i1].sum(axis=0) / ((i1 - i0) * DT * 1e-3)
            ceiling = float(np.mean(hz >= 0.8 * 1000.0 / S["p"].tau_ref_E))
    candidate_gate = lifecycle_candidate_gate(
        lifecycle_label=lifecycle.get("label"), onset_ms=onset_ms,
        high_duration_ms=high_ms, x_activates_after_onset=x_after,
        postictal_suppression=postictal, statistical_return=rest,
        numerical_unsafe=numerical["numerical_unsafe"],
        refractory_ceiling_fraction=ceiling)
    slow_final = final["checkpoint"].slow
    snap_times = {label: float(snap["step"] * DT)
                  for label, snap in slow_final.snapshots.items()
                  if float(snap["step"] * DT) <= total_ms}
    selected = RECON.nearest_snapshot_labels(
        snap_times, RECON.select_landmark_times(
            lifecycle, win_ms=float(baseline["band"]["win_ms"]), total_ms=total_ms))
    names = list(selected)
    snaps = [slow_final.snapshots[selected[name]["snapshot_label"]] for name in names]
    npz_path = row["output_path"].replace(".json", "_fields.npz")
    stride = max(1, int(round(10.0 / DT)))
    _write_npz(
        npz_path, rate_dt_ms=np.asarray([10.0]), rate_E=rate_e[::stride],
        landmark_names=np.asarray(names),
        landmark_times_ms=np.asarray([selected[name]["snapshot_time_ms"] for name in names]),
        D_fields=np.stack([1.0 - np.asarray(snap["z_E"]) for snap in snaps]),
        X_fields=np.stack([np.asarray(snap["x_E"]) for snap in snaps]),
        H_fields=np.stack([np.asarray(snap["h_E"]) for snap in snaps]),
    )
    record = dict(
        status="COMPLETE", row_id=row["row_id"], candidate=row["candidate"],
        connection_seed=1, noise_seed=row["noise_seed"], T_ms=total_ms,
        no_kick=True, no_reset=True, no_parameter_step=True,
        lifecycle=lifecycle, onset_ms=onset_ms, offset_ms=offset_ms,
        high_duration_ms=high_ms, pre_statistics=pre, post_statistics=post,
        statistical_return=rest, postictal_suppression=postictal,
        early_post_rate_hz=early_post_rate, x_peak_depletion_ms=x_peak_ms,
        x_activates_after_onset=x_after, x_min=float(np.min(xtrace)) if xtrace.size else None,
        numerical=numerical, refractory_ceiling_fraction=ceiling,
        lifecycle_candidate=candidate_gate["pass_"], candidate_gate=candidate_gate,
        field_landmarks=selected, fields_path=npz_path, fields_sha256=_sha(npz_path),
        wall_s=time.time() - t0, resources=dict(start=before, end=_meminfo()),
        finished=_now(),
    )
    _write_json(row["output_path"], record)
    _write_json(row["done_path"], dict(status="DONE", output_sha256=_sha(row["output_path"]),
                                       finished=_now()))
    del first, final, res, spikes, S
    gc.collect()
    return record


def cmd_lifecycle(args):
    if not args.confirm_run:
        raise SystemExit("40k lifecycle requires --confirm-run")
    _assert_lock()
    manifest = _load(LIFECYCLE_MANIFEST)
    if manifest["status"] == "NO_CALIBRATED_X_CANDIDATE":
        payload = dict(
            status="LIFECYCLE_NOT_RUN_NO_CALIBRATED_X_CANDIDATE",
            n_rows=0, candidates=[], completed=_now())
        _write_json(LIFECYCLE_VERDICT, payload)
        print(json.dumps(payload, indent=2)); return
    if _sha(XCAL) != manifest["x_calibration_sha256"] or len(manifest["rows"]) > 6:
        raise SystemExit("lifecycle manifest drift or row cap exceeded")
    rows = []
    with _stage_lock("lifecycle"):
        swap0 = _meminfo()["swap_used_mib"]
        for row in manifest["rows"]:
            _wait_submission(swap0)
            rows.append(_run_lifecycle_row(row))
            print(f"[lifecycle] {row['row_id']} candidate={rows[-1]['lifecycle_candidate']} "
                  f"label={rows[-1]['lifecycle']['label']}", flush=True)
    candidates = [row["row_id"] for row in rows if row["lifecycle_candidate"]]
    payload = dict(
        status=("TEMPORAL_LIFECYCLE_CANDIDATE" if candidates
                else "NO_TEMPORAL_LIFECYCLE_CANDIDATE_IN_LOCKED_MATRIX"),
        n_rows=len(rows), candidate_rows=candidates,
        label_counts={label: sum(row["lifecycle"]["label"] == label for row in rows)
                      for label in sorted({row["lifecycle"]["label"] for row in rows})},
        claim_boundary=("candidate only; morphology M/K/A/ELR absent; no patient-like seizure claim"),
        completed=_now())
    _write_json(LIFECYCLE_VERDICT, payload)
    print(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--confirm-run", action="store_true")
    sub.add_parser("lifecycle-manifest")
    life = sub.add_parser("lifecycle")
    life.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock": cmd_lock(args)
    elif args.cmd == "calibrate": cmd_calibrate(args)
    elif args.cmd == "lifecycle-manifest": cmd_lifecycle_manifest(args)
    elif args.cmd == "lifecycle": cmd_lifecycle(args)


if __name__ == "__main__":
    main()
