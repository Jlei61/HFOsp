#!/usr/bin/env python3
"""FCXR-LC3 E1 prepared states and field-preserving frozen geometry.

This runner is intentionally separate from the E0/E1 D-field runner so an
already detached replay cannot be invalidated by development of the next stage.
Every 40k command is explicit and guarded by ``--confirm-run``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import copy
import dataclasses
import fcntl
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import replace_frozen_fields, run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    EXTENDED_MS,
    EXTENDED_TAIL_MS,
    H1_POINT_ID,
    H6_POINT_ID,
    PRIMARY_D_LABELS,
    SCREEN_MS,
    SCREEN_TAIL_MS,
    build_geometry_manifest_rows,
    classify_geometry_tail,
    compact_checkpoint_diagnostics,
    configured_state_hash,
    extension_required,
    load_prepared_checkpoint,
    save_prepared_checkpoint,
    validate_geometry_manifest,
)


OUT = E01.OUT
DT = E01.DT
GEOMETRY_LOCK = os.path.join(OUT, "geometry_execution_lock.json")
D_FIELD_SPATIAL_AUDIT = os.path.join(OUT, "d_field_spatial_audit.json")
PREP_DIR = os.path.join(OUT, "prepared_states")
CELL_DIR = os.path.join(OUT, "geometry_cells")
MANIFEST = os.path.join(OUT, "geometry_manifest.json")
BASELINE = E01.ARTIFACTS["lc1_baseline"]
PREP_MS = {
    (H1_POINT_ID, "low"): 10000.0,
    (H1_POINT_ID, "high"): 5000.0,
    (H6_POINT_ID, "high"): 8.0 * 632.4555320336759,
}
GEOMETRY_SOURCES = (
    "src/topic4_fcxr_lc3.py",
    "src/topic4_fcxr_lc3_geometry.py",
    "src/snn_engine/mz_slow_vars.py",
    "scripts/run_topic4_fcxr_lc3.py",
    "scripts/run_topic4_fcxr_lc3_geometry.py",
    "scripts/run_topic4_fcxr_lc3_geometry_autopilot.sh",
    "docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md",
    "docs/superpowers/plans/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability.md",
)
POINT_IDS = {"H1": H1_POINT_ID, "H6": H6_POINT_ID}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    sibling = 0
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        for line in ps.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or int(parts[0]) == os.getpid():
                continue
            cmd = parts[1]
            if ("python" in cmd and "scripts/run_topic4" in cmd
                    and "fcxr_lc3" not in cmd and "pytest" not in cmd):
                sibling += 1
    except Exception:
        sibling = -1
    return dict(
        mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
        swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
        self_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
        sibling_topic4_python_count=sibling,
    )


def _resource_log(stage, **extra):
    row = dict(t=_now(), stage=stage, **_meminfo(), **extra)
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


@contextmanager
def _stage_lock(name):
    path = os.path.join(OUT, f".{name}.lock")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC3 geometry stage already running: {name}") from exc
        yield


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _assert_clean_tracked_tree():
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if out.strip():
        raise SystemExit("tracked worktree changes must be committed before geometry lock")


def cmd_lock(_args):
    _assert_clean_tracked_tree()
    dlock_path = os.path.join(OUT, "d_field_lock.json")
    if not os.path.isfile(dlock_path) or _load_json(dlock_path).get("status") != "PASS":
        raise SystemExit("complete PASS d_field_lock.json is required")
    if (not os.path.isfile(D_FIELD_SPATIAL_AUDIT)
            or _load_json(D_FIELD_SPATIAL_AUDIT).get("status") != "PASS"):
        raise SystemExit("PASS d_field_spatial_audit.json is required")
    e0 = _load_json(os.path.join(OUT, "prepared_state_contract.json"))
    if e0.get("status") != "PASS":
        raise SystemExit("E0 exact-state contract is not PASS")
    initial_lock = _load_json(os.path.join(OUT, "execution_lock.json"))
    for rel in GEOMETRY_SOURCES:
        if not os.path.isfile(os.path.join(ROOT, rel)):
            raise SystemExit(f"missing geometry source: {rel}")
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-geometry-lock-1.0", git_head=_git_head(),
        initial_execution_lock_sha256=_sha(os.path.join(OUT, "execution_lock.json")),
        initial_execution_git_head=initial_lock["git_head"],
        e0_sha256=_sha(os.path.join(OUT, "prepared_state_contract.json")),
        d_field_lock_sha256=_sha(dlock_path),
        d_field_spatial_audit_sha256=_sha(D_FIELD_SPATIAL_AUDIT),
        sources={rel: _sha(os.path.join(ROOT, rel)) for rel in GEOMETRY_SOURCES},
        engine_hashes=initial_lock["engine_hashes"], resource_at_lock=_meminfo(),
        locked_at=_now(), scientific_negatives_do_not_abort_map=True,
    )
    _write_json(GEOMETRY_LOCK, payload)
    print(json.dumps(dict(status="LOCKED", git_head=payload["git_head"],
                          n_sources=len(payload["sources"])), indent=2))


def _assert_lock():
    if not os.path.isfile(GEOMETRY_LOCK):
        raise SystemExit("missing geometry_execution_lock.json")
    lock = _load_json(GEOMETRY_LOCK)
    if lock.get("status") != "LOCKED":
        raise SystemExit("geometry execution lock is not active")
    if _sha(os.path.join(OUT, "execution_lock.json")) != lock["initial_execution_lock_sha256"]:
        raise SystemExit("initial execution lock drift")
    if _sha(os.path.join(OUT, "prepared_state_contract.json")) != lock["e0_sha256"]:
        raise SystemExit("E0 artifact drift")
    if _sha(os.path.join(OUT, "d_field_lock.json")) != lock["d_field_lock_sha256"]:
        raise SystemExit("D-field lock drift")
    if _sha(D_FIELD_SPATIAL_AUDIT) != lock["d_field_spatial_audit_sha256"]:
        raise SystemExit("D-field spatial audit drift")
    for rel, expected in lock["sources"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"geometry source drift: {rel}")
    for rel, expected in lock["engine_hashes"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"blessed engine drift: {rel}")
    return lock


def _point(point_id):
    manifest = _load_json(E01.ARTIFACTS["gx1_strip_manifest"])
    rows = [r for r in manifest["rows"]
            if r["point_id"] == point_id and r["arm"] == "healthy_low"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one point row for {point_id}, got {len(rows)}")
    return rows[0]


def _region_masks(S):
    pos = np.asarray(S["posE"], dtype=float)
    src = np.asarray(S["src_xy"], dtype=float)
    snk = np.asarray(S["snk_xy"], dtype=float)
    axis = np.asarray(S["axis_unit"], dtype=float)
    core_a = np.linalg.norm(pos - src, axis=1) <= PP.CORE_R
    core_b = np.linalg.norm(pos - snk, axis=1) <= PP.CORE_R
    rel = pos - src
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    either_core = core_a | core_b
    axial = (perp <= PP.CORE_R) & (~either_core)
    off_axis = ~either_core & ~axial
    if not all(mask.any() for mask in (core_a, core_b, axial, off_axis)):
        raise RuntimeError("one or more registered D-field regions are empty")
    return dict(core_A=core_a, core_B=core_b, axial=axial, off_axis=off_axis)


def cmd_field_audit(_args):
    """Add the region statistics required by spec §5.1 without rerunning a 40k trace."""

    dlock_path = os.path.join(OUT, "d_field_lock.json")
    if not os.path.isfile(dlock_path):
        raise SystemExit("complete d_field_lock.json is required")
    dlock = _load_json(dlock_path)
    if dlock.get("status") != "PASS":
        raise SystemExit("D-field lock is not PASS")
    families = {}
    for family, rec in dlock["families"].items():
        if _sha(rec["output_path"]) != rec["output_sha256"]:
            raise RuntimeError(f"{family}: D-field file hash drift")
        seed = 3 if family.startswith("seed3") else 1
        S = PP.build_substrate(seed)
        masks = _region_masks(S)
        with np.load(rec["output_path"], allow_pickle=False) as data:
            labels = [str(x) for x in data["labels"]]
            fields = np.asarray(data["D_fields"], dtype=float)
        source_rows = {row["label"]: row for row in rec["rows"]}
        rows = []
        for label, field in zip(labels, fields):
            if E01._arr_hash(field) != source_rows[label]["field_sha256"]:
                raise RuntimeError(f"{family}/{label}: field checksum drift")
            rows.append(dict(
                label=label, field_sha256=source_rows[label]["field_sha256"],
                mean_D=float(np.mean(field)), q05=float(np.quantile(field, 0.05)),
                q50=float(np.quantile(field, 0.50)), q95=float(np.quantile(field, 0.95)),
                core_A_mean=float(np.mean(field[masks["core_A"]])),
                core_B_mean=float(np.mean(field[masks["core_B"]])),
                axial_mean=float(np.mean(field[masks["axial"]])),
                off_axis_mean=float(np.mean(field[masks["off_axis"]])),
                spatial_l2=float(np.linalg.norm(field)),
                n_cells={name: int(mask.sum()) for name, mask in masks.items()},
            ))
        families[family] = dict(
            connection_seed=seed, source_path=rec["output_path"],
            source_sha256=rec["output_sha256"], rows=rows,
        )
        del S; gc.collect()
    payload = dict(
        status="PASS", schema="fcxr-lc3-d-field-spatial-audit-1.0",
        d_field_lock_sha256=_sha(dlock_path), band_half_width_mm=PP.CORE_R,
        region_contract="core A/B radius CORE_R; axial corridor width CORE_R; remainder off-axis",
        families=families, completed=_now(),
    )
    _write_json(D_FIELD_SPATIAL_AUDIT, payload)
    print(json.dumps(dict(status="PASS", families=list(families)), indent=2))


def _primary_fields():
    lock = _load_json(os.path.join(OUT, "d_field_lock.json"))
    fam = lock["families"]["seed1_q75"]
    with np.load(fam["output_path"], allow_pickle=False) as data:
        labels = [str(x) for x in data["labels"]]
        matrix = np.asarray(data["D_fields"], dtype=float)
    if labels != ["D10", "D30", "D50", "D70", "Dmax"]:
        raise RuntimeError(f"unexpected primary field labels: {labels}")
    ne = matrix.shape[1]
    fields = {label: matrix[i].copy() for i, label in enumerate(labels)}
    fields["D_healthy"] = np.zeros(ne, dtype=float)
    records = {
        "D_healthy": dict(field_sha256=E01._arr_hash(fields["D_healthy"]),
                          source_path=fam["output_path"], source_sha256=fam["output_sha256"],
                          kind="exact_all_zero_control"),
    }
    by_label = {row["label"]: row for row in fam["rows"]}
    for label in labels:
        records[label] = dict(field_sha256=E01._arr_hash(fields[label]),
                              source_path=fam["output_path"], source_sha256=fam["output_sha256"],
                              replay=by_label[label])
        if records[label]["field_sha256"] != by_label[label]["field_sha256"]:
            raise RuntimeError(f"{label}: primary field hash drift")
    return fields, records


def _frozen_cfg(point, d_field, a_x, *, h_init_scale):
    cfg = E01._dynamic_cfg(point)
    cfg.update(
        use_z=False, z_frozen_E=1.0 - np.asarray(d_field, dtype=float),
        use_x=True, x_relay_frozen_E=np.full(len(d_field), float(a_x)),
        h_lc2_init_E=np.full(len(d_field), float(h_init_scale) * float(point["theta"])),
    )
    return cfg


def _slow(S, cfg):
    return MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S))


def _tail_observables(out, S, point, *, tail_ms, analysis_start_ms):
    n_tail = max(1, int(round(float(tail_ms) / DT)))
    spikes = np.asarray(out["E_spk_bool"][-n_tail:], dtype=bool)
    slow = out["checkpoint"].slow
    h = np.asarray(slow.trace_h_lc2_mean[-out["n_steps"]:], dtype=float)
    h_tail = h[-n_tail:]
    tau_ratio = np.asarray(slow.trace_tau_eff_ratio_min[-out["n_steps"]:], dtype=float)
    clips = np.asarray(slow.trace_conductance_clip_frac[-out["n_steps"]:], dtype=float)
    return classify_geometry_tail(
        rate_hz=out["rate_E"], dt_ms=DT,
        baseline_roll_hi_hz=float(_load_json(BASELINE)["roll_hi_hz"]),
        analysis_start_ms=float(analysis_start_ms),
        per_cell_tail_spike_counts=spikes.sum(axis=0), tail_duration_ms=float(tail_ms),
        tau_ref_e_ms=float(S["p"].tau_ref_E), h_mean_trace=h_tail,
        theta_h=float(point["theta"]), finite=bool(np.all(np.isfinite(out["rate_E"]))),
        clip_frac_max=float(clips.max()) if clips.size else 0.0,
        tau_eff_min_ms=float(S["p"].tau_m_E * tau_ratio.min()) if tau_ratio.size else np.inf,
    )


def _prep_paths(point_id, state_kind):
    stem = f"{point_id}_{state_kind}"
    return os.path.join(PREP_DIR, f"{stem}.pkl"), os.path.join(PREP_DIR, f"{stem}.json")


def _inject_h6_low(lock):
    donor_pkl, donor_json = _prep_paths(H1_POINT_ID, "low")
    if not (os.path.isfile(donor_pkl) and os.path.isfile(donor_json)):
        raise SystemExit("H1 canonical low must be prepared before H6 low injection")
    donor_record = _load_json(donor_json)
    if donor_record.get("status") != "ACCEPTED_CANONICAL_STATE":
        raise SystemExit("H1 low donor is not accepted")
    payload = load_prepared_checkpoint(
        donor_pkl, expected_file_sha256=donor_record["checkpoint"]["file_sha256"])
    state = compact_checkpoint_diagnostics(payload["state"])
    point = _point(H6_POINT_ID)
    cfg = state.slow.cfg
    cfg.tau_h_lc2 = float(point["tau_ms"])
    cfg.theta_h_lc2 = float(point["theta"])
    cfg.k_h_lc2 = float(point["k"])
    cfg.rho_h_lc2 = float(point["rho"])
    cfg.h_lc2_init_E = np.zeros(state.slow.NE)
    state.slow.h_lc2_E[:] = 0.0
    state.slow._h_source_lc2_E[:] = 0.0
    pkl_path, json_path = _prep_paths(H6_POINT_ID, "low")
    checkpoint = save_prepared_checkpoint(
        pkl_path, state,
        metadata=dict(point_id=H6_POINT_ID, state_kind="low", method="H1-low-fast-state_H6-H-zero-injection",
                      donor_configured_state_hash=donor_record["checkpoint"]["configured_state_hash"],
                      source_lock_git_head=lock["git_head"]),
    )
    record = dict(
        status="ACCEPTED_SENTINEL_INJECTED_LOW_STATE", point_id=H6_POINT_ID,
        state_kind="low", point={k: point[k] for k in ("tau_ms", "theta", "k", "rho")},
        scientific_role="healthy-self-ignition sentinel; not an H6 low equilibrium",
        donor=donor_record["checkpoint"], checkpoint=checkpoint,
        source_lock_git_head=lock["git_head"], finished=_now(),
    )
    _write_json(json_path, record)
    return record


def cmd_prepare(args):
    if not args.confirm_run:
        raise SystemExit("40k preparation requires --confirm-run")
    lock = _assert_lock()
    point_id = POINT_IDS[args.point]
    with _stage_lock(f"prepare_{args.point}_{args.state}"):
        if point_id == H6_POINT_ID and args.state == "low":
            record = _inject_h6_low(lock)
            print(json.dumps(record, indent=2)); return
        before = _resource_log(f"PREP_{args.point}_{args.state}_START")
        if before["mem_available_gib"] < 128.0:
            raise SystemExit("prepared-state resource gate requires 128 GiB MemAvailable")
        fields, _records = _primary_fields()
        d_label = "D_healthy" if args.state == "low" else "D50"
        point = _point(point_id)
        h_scale = 0.0 if args.state == "low" else 2.0
        cfg = _frozen_cfg(point, fields[d_label], 1.0, h_init_scale=h_scale)
        T = float(PREP_MS[(point_id, args.state)])
        S = PP.build_substrate(1)
        p = dataclasses.replace(S["p"], T=T, dt=DT)
        S["net"]["rng"] = np.random.default_rng(401)
        t0 = time.time()
        out = run_fcxr_loop(
            p, S["net"], slow=_slow(S, cfg), n_steps=int(round(T / DT)),
            capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"],
        )
        tail_ms = 2000.0
        cls = _tail_observables(out, S, point, tail_ms=tail_ms,
                                analysis_start_ms=max(0.0, T - tail_ms))
        events, _af, _bin, _floor, _rate = OLD._events_from_res(
            out, DT, event_bar=float(_load_json(BASELINE)["frozen_event_bar"]))
        returning = [e for e in events if e.get("returned", False)]
        if args.state == "low":
            accepted = bool(cls["label"] == "INTERICTAL_WORKPOINT" and len(returning) >= 3)
        else:
            accepted = cls["label"] in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        pkl_path, json_path = _prep_paths(point_id, args.state)
        checkpoint = save_prepared_checkpoint(
            pkl_path if accepted else pkl_path.replace(".pkl", ".attempt.pkl"),
            out["checkpoint"], metadata=dict(point_id=point_id, state_kind=args.state,
                                             d_label=d_label, a_x=1.0,
                                             source_lock_git_head=lock["git_head"]),
        )
        after = _resource_log(f"PREP_{args.point}_{args.state}_DONE",
                              wall_s=time.time() - t0, accepted=accepted)
        record = dict(
            status="ACCEPTED_CANONICAL_STATE" if accepted else "PREPARED_STATE_UNRESOLVED",
            point_id=point_id, state_kind=args.state,
            point={k: point[k] for k in ("tau_ms", "theta", "k", "rho")},
            d_label=d_label, a_x=1.0, T_ms=T, tail_ms=tail_ms,
            classification=cls, n_returning_events=len(returning),
            returning_events=[dict(t_on=float(e["t_on"]), t_off=float(e["t_off"]),
                                   dur_ms=float(e["dur_ms"]), peak_ext=float(e["peak_ext"]))
                              for e in returning],
            checkpoint=checkpoint, resources=dict(start=before, end=after),
            source_lock_git_head=lock["git_head"], finished=_now(),
        )
        _write_json(json_path, record)
        del out, S; gc.collect()
        print(json.dumps(dict(status=record["status"], point_id=point_id,
                              state=args.state, classification=cls["label"],
                              n_returning=len(returning), checkpoint=checkpoint), indent=2))


def _prepared_records():
    records = {}
    for point in (H1_POINT_ID, H6_POINT_ID):
        for state in ("low", "high"):
            pkl, js = _prep_paths(point, state)
            if not (os.path.isfile(pkl) and os.path.isfile(js)):
                raise SystemExit(f"missing prepared state {point}/{state}")
            rec = _load_json(js)
            allowed = ("ACCEPTED_CANONICAL_STATE", "ACCEPTED_SENTINEL_INJECTED_LOW_STATE")
            if rec.get("status") not in allowed:
                raise SystemExit(f"prepared state unresolved: {point}/{state}")
            if rec["checkpoint"]["file_sha256"] != _sha(pkl):
                raise SystemExit(f"prepared checkpoint drift: {point}/{state}")
            records[(point, state)] = rec
    return records


def cmd_manifest(_args):
    lock = _assert_lock()
    fields, field_records = _primary_fields()
    if set(fields) != set(PRIMARY_D_LABELS):
        raise RuntimeError("primary field set is incomplete")
    prepared = _prepared_records()
    hashes = {(point, state): rec["checkpoint"]["configured_state_hash"]
              for (point, state), rec in prepared.items()}
    rows = build_geometry_manifest_rows(
        fields=field_records, prepared_state_hashes=hashes, output_root=OUT)
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-geometry-manifest-1.0",
        audit=validate_geometry_manifest(rows), rows=rows,
        prepared_state_files={f"{p}/{s}": prepared[(p, s)]["checkpoint"]
                              for p, s in prepared},
        geometry_lock_sha256=_sha(GEOMETRY_LOCK), source_lock_git_head=lock["git_head"],
        created=_now(),
    )
    _write_json(MANIFEST, payload)
    print(json.dumps(dict(status="LOCKED", **payload["audit"]), indent=2))


def _load_manifest():
    _assert_lock()
    if not os.path.isfile(MANIFEST):
        raise SystemExit("missing geometry_manifest.json")
    manifest = _load_json(MANIFEST)
    validate_geometry_manifest(manifest["rows"])
    if manifest.get("geometry_lock_sha256") != _sha(GEOMETRY_LOCK):
        raise SystemExit("geometry manifest was written under another lock")
    return manifest


_WORKER_SUBSTRATE = None
_WORKER_FIELDS = None
_WORKER_PREPARED = None


def _worker_context():
    global _WORKER_SUBSTRATE, _WORKER_FIELDS, _WORKER_PREPARED
    if _WORKER_SUBSTRATE is None:
        _WORKER_SUBSTRATE = PP.build_substrate(1)
        # Materialise flattened sparse scatter once per worker.
        from src.topic4_fcxr_lc3 import _constants
        _constants(_WORKER_SUBSTRATE["p"], _WORKER_SUBSTRATE["net"])
    if _WORKER_FIELDS is None:
        _WORKER_FIELDS, _ = _primary_fields()
    if _WORKER_PREPARED is None:
        _WORKER_PREPARED = _prepared_records()
    return _WORKER_SUBSTRATE, _WORKER_FIELDS, _WORKER_PREPARED


def _row_result_path(row):
    return row["output_path"]


def _run_row(row):
    prior_path = _row_result_path(row)
    if os.path.isfile(prior_path) and os.path.isfile(row["done_path"]):
        prior = _load_json(prior_path)
        if prior.get("row_id") == row["row_id"] and prior.get("status") == "COMPLETE":
            return prior
    S, fields, prepared = _worker_context()
    prep = prepared[(row["point_id"], row["state_kind"])]
    loaded = load_prepared_checkpoint(
        prep["checkpoint"]["path"], expected_file_sha256=prep["checkpoint"]["file_sha256"])
    if configured_state_hash(loaded["state"]) != row["prepared_state_hash"]:
        raise RuntimeError(f"{row['row_id']}: prepared state hash mismatch")
    child = replace_frozen_fields(
        loaded["state"], d_field=fields[row["d_label"]],
        x_field=np.full(S["NE"], float(row["a_x"])),
    )
    point = _point(row["point_id"])
    p = dataclasses.replace(S["p"], T=SCREEN_MS, dt=DT)
    t0 = time.time()
    first = run_fcxr_loop(
        p, S["net"], start=child, n_steps=int(round(SCREEN_MS / DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"],
    )
    cls1 = _tail_observables(first, S, point, tail_ms=SCREEN_TAIL_MS,
                             analysis_start_ms=0.0)
    extended = extension_required(state_kind=row["state_kind"], label=cls1["label"])
    final = first
    combined_rate = first["rate_E"]
    combined_spikes = first["E_spk_bool"]
    if extended:
        extra_ms = EXTENDED_MS - SCREEN_MS
        p2 = dataclasses.replace(S["p"], T=extra_ms, dt=DT)
        second = run_fcxr_loop(
            p2, S["net"], start=first["checkpoint"],
            n_steps=int(round(extra_ms / DT)), capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )
        combined_rate = np.concatenate([first["rate_E"], second["rate_E"]])
        combined_spikes = np.concatenate([first["E_spk_bool"], second["E_spk_bool"]], axis=0)
        # Repackage final 5 s observables while preserving the exact second checkpoint.
        final = dict(second)
        final["rate_E"] = combined_rate
        final["E_spk_bool"] = combined_spikes
        final["n_steps"] = combined_rate.size
        cls2 = _tail_observables(final, S, point, tail_ms=EXTENDED_TAIL_MS,
                                 analysis_start_ms=EXTENDED_MS - EXTENDED_TAIL_MS)
    else:
        cls2 = None
    resolved = cls2 or cls1
    stride = max(1, int(round(10.0 / DT)))
    slow = final["checkpoint"].slow
    current_steps = int(final["n_steps"])
    htrace = np.asarray(slow.trace_h_lc2_mean[-current_steps:], dtype=float)
    result = dict(
        status="COMPLETE", row_id=row["row_id"], point_id=row["point_id"],
        d_label=row["d_label"], d_field_sha256=row["d_field_sha256"],
        a_x=row["a_x"], state_kind=row["state_kind"], sentinel=row["sentinel"],
        prepared_state_hash=row["prepared_state_hash"], initial_screen=cls1,
        extended=extended, extended_classification=cls2,
        resolved_label=resolved["label"], total_ms=EXTENDED_MS if extended else SCREEN_MS,
        final_configured_state_hash=configured_state_hash(final["checkpoint"]),
        rate_trace_dt_ms=10.0, rate_trace=combined_rate[::stride].astype(float).tolist(),
        h_trace=htrace[::stride].astype(float).tolist(),
        max_rate_hz=float(np.max(combined_rate)), mean_rate_hz=float(np.mean(combined_rate)),
        wall_s=time.time() - t0, peak_rss_gib=_meminfo()["self_peak_rss_gib"],
        source_lock_git_head=_load_json(GEOMETRY_LOCK)["git_head"], finished=_now(),
    )
    _write_json(prior_path, result)
    _write_json(row["done_path"], dict(status="DONE", row_id=row["row_id"],
                                       output_sha256=_sha(prior_path), finished=_now()))
    del first, final, combined_spikes
    gc.collect()
    return result


def cmd_row(args):
    if not args.confirm_run:
        raise SystemExit("40k row requires --confirm-run")
    manifest = _load_manifest()
    rows = [row for row in manifest["rows"] if row["row_id"] == args.row_id]
    if len(rows) != 1:
        raise SystemExit(f"unknown row_id {args.row_id}")
    with _stage_lock(f"geometry_row_{args.row_id}"):
        out = _run_row(rows[0])
    print(json.dumps(dict(row_id=out["row_id"], label=out["resolved_label"],
                          extended=out["extended"], wall_s=out["wall_s"]), indent=2))


def _choose_workers(single_rss_gib, swap_baseline):
    mem = _meminfo()
    need = 96.0 + 2.0 * 1.35 * float(single_rss_gib)
    stable_swap = mem["swap_used_mib"] - float(swap_baseline) < 256.0
    return 2 if mem["mem_available_gib"] >= need and stable_swap else 1


def _aggregate(rows):
    h1 = [r for r in rows if not r["sentinel"]]
    h6 = [r for r in rows if r["sentinel"]]
    high_labels = {"FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT"}
    return dict(
        status="COMPLETE", n_rows=len(rows), n_h1=len(h1), n_h6=len(h6),
        label_counts=dict(sorted({label: sum(r["resolved_label"] == label for r in rows)
                                  for label in {r["resolved_label"] for r in rows}}.items())),
        h1_low_start_entry=[r["row_id"] for r in h1
                            if r["state_kind"] == "low" and r["resolved_label"] in high_labels],
        h1_high_start_survival=[r["row_id"] for r in h1
                               if r["state_kind"] == "high" and r["resolved_label"] in high_labels],
        h1_high_to_low_return=[r["row_id"] for r in h1
                              if r["state_kind"] == "high"
                              and r["resolved_label"] == "INTERICTAL_WORKPOINT"],
        probability_contours_authorized=False,
        note="single prepared microstate/noise gives empirical brackets only",
        rows=rows, completed=_now(),
    )


def cmd_map(args):
    if not args.confirm_run:
        raise SystemExit("40k geometry map requires --confirm-run")
    manifest = _load_manifest()
    with _stage_lock("geometry_map"):
        swap0 = _meminfo()["swap_used_mib"]
        _write_json(os.path.join(OUT, "GEOMETRY_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), n_rows=102,
                         swap_baseline_mib=swap0, started=_now()))
        # Registered smoke first; its result remains one of the canonical 102 rows.
        smoke_id = f"{H1_POINT_ID}_D_healthy_aX1p00_low"
        smoke_row = next(row for row in manifest["rows"] if row["row_id"] == smoke_id)
        smoke = _run_row(smoke_row)
        workers = _choose_workers(smoke["peak_rss_gib"], swap0)
        _resource_log("GEOMETRY_SMOKE_DONE", row_id=smoke_id,
                      single_rss_gib=smoke["peak_rss_gib"], workers=workers)
        pending_rows = [row for row in manifest["rows"] if row["row_id"] != smoke_id]
        results = [smoke]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            active = {}
            cursor = 0
            while cursor < len(pending_rows) or active:
                mem = _meminfo()
                swap_delta = mem["swap_used_mib"] - swap0
                allow_submit = swap_delta < 256.0 and mem["mem_available_gib"] >= 96.0
                while allow_submit and cursor < len(pending_rows) and len(active) < workers:
                    row = pending_rows[cursor]; cursor += 1
                    active[pool.submit(_run_row, row)] = row["row_id"]
                if not active:
                    raise RuntimeError("resource gate stopped submissions with no active worker")
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    row_id = active.pop(future)
                    out = future.result()
                    results.append(out)
                    print(f"[geometry] {len(results)}/102 {row_id} -> {out['resolved_label']}",
                          flush=True)
        if len(results) != 102:
            raise RuntimeError(f"geometry map incomplete: {len(results)}/102")
        by_id = {row["row_id"]: row for row in results}
        ordered = [by_id[row["row_id"]] for row in manifest["rows"]]
        aggregate = _aggregate(ordered)
        _write_json(os.path.join(OUT, "geometry_map.json"), aggregate)
        _write_json(os.path.join(OUT, "GEOMETRY_DONE.json"),
                    dict(status="DONE", n_rows=102,
                         geometry_map_sha256=_sha(os.path.join(OUT, "geometry_map.json")),
                         finished=_now()))
        running = os.path.join(OUT, "GEOMETRY_RUNNING.json")
        if os.path.exists(running):
            os.replace(running, os.path.join(OUT, "GEOMETRY_RUNNING.superseded.json"))
    print(json.dumps({k: aggregate[k] for k in ("status", "n_rows", "label_counts")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("field-audit")
    sub.add_parser("lock")
    prep = sub.add_parser("prepare")
    prep.add_argument("--point", choices=("H1", "H6"), required=True)
    prep.add_argument("--state", choices=("low", "high"), required=True)
    prep.add_argument("--confirm-run", action="store_true")
    sub.add_parser("manifest")
    row = sub.add_parser("row")
    row.add_argument("--row-id", required=True)
    row.add_argument("--confirm-run", action="store_true")
    mp = sub.add_parser("map")
    mp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "field-audit": cmd_field_audit(args)
    elif args.cmd == "lock": cmd_lock(args)
    elif args.cmd == "prepare": cmd_prepare(args)
    elif args.cmd == "manifest": cmd_manifest(args)
    elif args.cmd == "row": cmd_row(args)
    elif args.cmd == "map": cmd_map(args)


if __name__ == "__main__":
    main()
