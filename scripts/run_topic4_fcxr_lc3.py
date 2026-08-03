#!/usr/bin/env python3
"""FCXR-LC3 staged runner (currently E0 exact-state contract only).

No simulation runs on import.  The 40k E0 probe requires ``--confirm-run`` and
does not authorize any E1 scientific map row.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-lc3")

import argparse
import copy
import dataclasses
import fcntl
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
import run_topic4_mz_fcxr as FCXR  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import (  # noqa: E402
    clone_loop_state,
    replace_frozen_fields,
    run_fcxr_loop,
    state_hash,
)


OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc3_dx_spatial_instability")
GX1 = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "gx1_entry_offset_diagnostics")
LC2 = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "closed_loop_exploration")
LC1 = ("/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/"
       "topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure")
DT = 0.05
E0_T_MS = 120.0
E0_SPLIT_MS = 45.0
NOISE_SEED = 401


ARTIFACTS = {
    "gx1_verdict": os.path.join(GX1, "candidate_verdict.json"),
    "gx1_strip": os.path.join(GX1, "selectivity_strip.json"),
    "gx1_strip_manifest": os.path.join(GX1, "selectivity_strip_manifest.json"),
    "gx1_x_map": os.path.join(GX1, "x_authority_map.json"),
    "lc2_frozen_map": os.path.join(LC2, "frozen_fork_map.json"),
    "lc1_baseline": os.path.join(LC1, "baseline_contract_seed1.json"),
    "lc1_sensor": os.path.join(LC1, "sensor_separation_seed1_ty120.json"),
    "z_seed1_q75": os.path.join(
        LC1, "runs", "20260722T171901.346631Z_2583352_f56b721_zonly_seed1_q75_T24000",
        "zonly_traces.npz"),
    "z_seed1_q50": os.path.join(
        LC1, "runs", "20260722T175846.447885Z_2587821_f56b721_zonly_seed1_q50_T24000",
        "zonly_traces.npz"),
    "z_seed3_q75": os.path.join(
        LC1, "runs", "20260722T222048.076671Z_2587820_aac6ab1_zonly_seed3_q75_T24000",
        "zonly_traces.npz"),
    "zx_q75_seed1": os.path.join(
        LC1, "runs", "20260723T013724.630934Z_2614174_5c8b9fb_lifecycle_seed1_q75_xm0.1_td500_tu5000_T16000",
        "lifecycle_traces.npz"),
    "zx_q50_seed1": os.path.join(
        LC1, "runs", "20260723T020801.390740Z_2615846_5c8b9fb_lifecycle_seed1_q50_xm0.1_td1000_tu10000_T12000",
        "lifecycle_traces.npz"),
}

SOURCES = (
    "src/topic4_fcxr_lc3.py",
    "src/snn_engine/mz_slow_vars.py",
    "scripts/run_topic4_fcxr_lc3.py",
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


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=FCXR._jsonable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(
        mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
        swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
        self_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    )


def _resource_log(stage, **extra):
    row = dict(t=_now(), stage=stage, **_meminfo(), **extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


@contextmanager
def _stage_lock(stage):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".{stage}.lock")
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"LC3 stage already running: {stage}") from exc
        yield


def _assert_lock_current():
    path = os.path.join(OUT, "execution_lock.json")
    if not os.path.isfile(path):
        raise SystemExit("missing execution_lock.json; run lock first")
    lock = json.load(open(path))
    for name, rec in lock["artifacts"].items():
        if _sha(rec["path"]) != rec["sha256"]:
            raise SystemExit(f"artifact drift after lock: {name}")
    for rel, expected in lock["sources"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"source drift after lock: {rel}")
    for rel, expected in lock["engine_hashes"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"engine drift after lock: {rel}")
    return lock


def cmd_lock(_args):
    FCXR._assert_engine_blessed()
    missing = [p for p in ARTIFACTS.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit("missing LC3 input artifacts:\n" + "\n".join(missing))
    verdict = json.load(open(ARTIFACTS["gx1_verdict"]))
    if verdict.get("canonical_verdict") != "GX1_MECHANISM_MAP_ACCEPTED":
        raise SystemExit("GX1 mechanism map is not accepted")
    engine_versions = json.load(open(FCXR.ENGINE_VERSIONS))
    payload = dict(
        status="LOCKED", stage="E0", schema="fcxr-lc3-lock-1.0", git_head=_git_head(),
        gx1_canonical_verdict=verdict["canonical_verdict"],
        artifacts={name: dict(path=path, sha256=_sha(path)) for name, path in ARTIFACTS.items()},
        sources={rel: _sha(os.path.join(ROOT, rel)) for rel in SOURCES},
        engine_hashes=dict(engine_versions), resource_at_lock=_meminfo(), locked_at=_now(),
        hard_stops=["exact_state", "numerical_or_resource", "manifest_or_hash"],
        science_negatives_never_stop_reconnaissance=True,
    )
    _write_json(os.path.join(OUT, "execution_lock.json"), payload)
    print(json.dumps(dict(status="LOCKED", n_artifacts=len(ARTIFACTS),
                          n_sources=len(SOURCES), git_head=payload["git_head"]), indent=2))


def _h1_point():
    manifest = json.load(open(ARTIFACTS["gx1_strip_manifest"]))
    rows = [r for r in manifest["rows"]
            if r["point_id"] == "H1_ts1.25_r025" and r["arm"] == "healthy_low"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one nominal H1 row, got {len(rows)}")
    return rows[0]


def _dynamic_cfg(point):
    sensor = json.load(open(ARTIFACTS["lc1_sensor"]))
    cfg = FCXR._fc_cfg(
        1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=True,
        rec_sat_g=21.6,
    )
    cfg.update(
        use_h_lc2=True, tau_h_lc2=float(point["tau_ms"]),
        theta_h_lc2=float(point["theta"]), k_h_lc2=float(point["k"]),
        rho_h_lc2=float(point["rho"]),
        use_z=True, tau_z=5000.0, I_th_EI=95.19851312666987,
        use_x=True, x_min=0.1, tau_y=120.0, tau_x_down=500.0,
        tau_x_up=5000.0, K_y=5.0, y_gate=float(sensor["y_gate_q999"]), hill_n=4,
    )
    return cfg


def _slow(S, cfg):
    return MZSlowVars(
        S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
        core_mask_E=OLD.build_core_masks(S),
    )


def _arr_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        x = np.asarray(a)
        h.update(x.dtype.str.encode())
        h.update(np.asarray(x.shape, np.int64).tobytes())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def _alias_audit(parent):
    a, b = clone_loop_state(parent), clone_loop_state(parent)
    pairs = [
        (a.V, b.V), (a.ref, b.ref), (a.ring_sE, b.ring_sE), (a.ring_sI, b.ring_sI),
        (a.slow.z, b.slow.z), (a.slow.x_relay, b.slow.x_relay),
        (a.slow.h_lc2_E, b.slow.h_lc2_E),
    ]
    return bool(all(not np.shares_memory(x, y) for x, y in pairs))


def _field_replacement_audit(parent):
    ne = int(parent.slow.NE)
    d = np.linspace(0.0, 0.2, ne)
    x = np.linspace(0.6, 1.0, ne)
    d_child = replace_frozen_fields(parent, d_field=d)
    x_child = replace_frozen_fields(parent, x_field=x)
    return dict(
        parent_unchanged=state_hash(parent) == state_hash(clone_loop_state(parent)),
        d_exact=bool(np.array_equal(d_child.slow.z[:ne], 1.0 - d)),
        x_exact=bool(np.array_equal(x_child.slow.x_relay, x)
                     and np.array_equal(x_child.slow.ee_relay_send, x)),
        fast_state_unchanged_D=bool(np.array_equal(d_child.V, parent.V)
                                    and np.array_equal(d_child.ring_sE, parent.ring_sE)),
        fast_state_unchanged_X=bool(np.array_equal(x_child.V, parent.V)
                                    and np.array_equal(x_child.ring_sE, parent.ring_sE)),
    )


def cmd_e0(args):
    if not args.confirm_run:
        raise SystemExit("E0 40k simulation requires --confirm-run")
    with _stage_lock("e0_exact_fork"):
        lock = _assert_lock_current()
        before = _resource_log("E0_START", git_head=lock["git_head"])
        if before["mem_available_gib"] < 128.0:
            raise SystemExit(f"E0 resource gate: MemAvailable {before['mem_available_gib']:.1f} GiB <128")
        _write_json(os.path.join(OUT, "E0_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), started=_now(), resource=before))

        t_start = time.time()
        S = PP.build_substrate(1)
        point = _h1_point()
        cfg = _dynamic_cfg(point)
        p = dataclasses.replace(S["p"], T=E0_T_MS, dt=DT)
        n_total = int(round(E0_T_MS / DT))
        n_pre = int(round(E0_SPLIT_MS / DT))

        # Guarded reference.
        slow_ref = _slow(S, cfg)
        S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
        guarded = simulate_kick(
            p, S["net"], 0.0, slow=slow_ref, kick_center=list(S["src_xy"]),
            r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
            early_stop_runaway=False,
        )

        # Uninterrupted exact loop.
        slow_full = _slow(S, cfg)
        S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
        full = run_fcxr_loop(
            p, S["net"], slow=slow_full, n_steps=n_total, capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )

        # Split exact loop and resume from a complete checkpoint.
        slow_pre = _slow(S, cfg)
        S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
        pre = run_fcxr_loop(
            p, S["net"], slow=slow_pre, n_steps=n_pre, capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )
        child = clone_loop_state(pre["checkpoint"])
        tail = run_fcxr_loop(
            p, S["net"], start=child, n_steps=n_total - n_pre, capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )
        split_rate = np.concatenate([pre["rate_E"], tail["rate_E"]])
        split_spikes = np.concatenate([pre["E_spk_bool"], tail["E_spk_bool"]], axis=0)

        guarded_hash = _arr_hash(guarded["rate_E"], guarded["rate_I"], guarded["E_spk_bool"])
        full_hash = _arr_hash(full["rate_E"], full["rate_I"], full["E_spk_bool"])
        split_hash = _arr_hash(split_rate, split_spikes)
        field_audit = _field_replacement_audit(pre["checkpoint"])
        clauses = dict(
            guarded_vs_exact_rate=bool(np.array_equal(guarded["rate_E"], full["rate_E"])),
            guarded_vs_exact_i_rate=bool(np.array_equal(guarded["rate_I"], full["rate_I"])),
            guarded_vs_exact_raster=bool(np.array_equal(guarded["E_spk_bool"], full["E_spk_bool"])),
            split_vs_full_rate=bool(np.array_equal(split_rate, full["rate_E"])),
            split_vs_full_raster=bool(np.array_equal(split_spikes, full["E_spk_bool"])),
            split_vs_full_final_state=bool(state_hash(tail["checkpoint"]) == state_hash(full["checkpoint"])),
            child_forks_nonaliasing=_alias_audit(pre["checkpoint"]),
            field_replacement=bool(all(field_audit.values())),
            finite=bool(np.all(np.isfinite(full["rate_E"]))),
            zero_clip=bool(max(slow_ref.trace_conductance_clip_frac) == 0.0),
        )
        passed = bool(all(clauses.values()))
        after = _resource_log("E0_DONE", wall_s=time.time() - t_start, pass_contract=passed)
        payload = dict(
            status="PASS" if passed else "EXACT_FORK_BLOCKED", schema="fcxr-lc3-e0-1.0",
            point_id=point["point_id"], point={k: point[k] for k in (
                "tau_ms", "theta", "k", "rho", "rho_fraction")},
            z_provenance="LC1 q75", x_provenance="LC1 q75 x_min0.1 td500 tu5000",
            connection_seed=1, noise_seed=NOISE_SEED, dt_ms=DT, T_ms=E0_T_MS,
            split_ms=E0_SPLIT_MS, clauses=clauses, field_replacement_audit=field_audit,
            hashes=dict(guarded_output=guarded_hash, exact_output=full_hash,
                        split_output=split_hash, exact_final_state=state_hash(full["checkpoint"]),
                        split_final_state=state_hash(tail["checkpoint"])),
            resources=dict(start=before, end=after), source_lock_git_head=lock["git_head"],
            finished=_now(),
        )
        _write_json(os.path.join(OUT, "prepared_state_contract.json"), payload)
        if passed:
            _write_json(os.path.join(OUT, "E0_DONE.json"), payload)
            running = os.path.join(OUT, "E0_RUNNING.json")
            if os.path.exists(running):
                os.replace(running, os.path.join(OUT, "E0_RUNNING.superseded.json"))
        else:
            _write_json(os.path.join(OUT, "EXACT_FORK_BLOCKED.json"), payload)
            raise SystemExit("E0 exact-state contract failed")
        print(json.dumps(dict(status=payload["status"], clauses=clauses,
                              wall_s=after.get("wall_s"), peak_rss_gib=after["self_peak_rss_gib"]), indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    e0 = sub.add_parser("e0")
    e0.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock":
        cmd_lock(args)
    elif args.cmd == "e0":
        cmd_e0(args)


if __name__ == "__main__":
    main()
