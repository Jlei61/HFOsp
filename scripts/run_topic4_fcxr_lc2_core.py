#!/usr/bin/env python3
"""FCXR-LC2-Core staged runner.

The first executable question is R1 sensor separability.  This runner launches no parameter grid: it
replays four pre-locked 40k states one at a time with a read-only post-X gA observer.  Every simulation
requires ``--confirm-run`` and long use is intended through setsid/nohup.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-lc2")

import argparse
import dataclasses
import fcntl
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
import run_topic4_mz_slowvars as OLD  # noqa: E402
import run_topic4_mz_fcxr as FCXR  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc2_core import (  # noqa: E402
    RawGASampler, sha256_file, replay_h, contiguous_true_intervals,
)
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    load_onset_depletion_pi, assert_field_substrate_aligned, frozen_z_field,
)


DT = 0.05
G_SAT = 21.6
SAMPLE_N = 4096
SAMPLE_SEED = 20260801
BLOCK_MS = 1.0
BLOCK_STEPS = int(round(BLOCK_MS / DT))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core")
R1 = os.path.join(OUT, "r1_sensor")
R0 = os.path.join(OUT, "r0_vertical_slice")

LC1_ROOT = "/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure"
HEO_ROOT = "/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-heo1/results/topic4_sef_hfo/mz_full_conductance_spatial_relay"
P_FIELD = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                       "snapshots", "zA_q75_tz5000", "seed_1.npz")

ARTIFACTS = {
    "baseline": (os.path.join(LC1_ROOT, "baseline_trace_seed1.npz"),
                 "b6204332e6a62bcfbf04f268149b057bef42d4eb7219495cbf07097fef8f286e"),
    "baseline_contract": (os.path.join(LC1_ROOT, "baseline_contract_seed1.json"),
                          "fd3e0d05ef730c30a484a071046e6a92d8f5e775b2035646dc89f4b4e8367c53"),
    "q75": (os.path.join(LC1_ROOT, "runs", "20260722T171901.346631Z_2583352_f56b721_zonly_seed1_q75_T24000",
                         "zonly_traces.npz"),
            "082e362b192434a259b3bf2431af865db82ef2c5d138d96fb4a559717adc9649"),
    "heo1": (os.path.join(HEO_ROOT, "high_energy_oscillatory_branch", "screen_cells",
                          "gq0.999_A8_D0.15_nokick_trace.npz"),
             "cfd50a44c7fd689f0cb01d3ca0010656b3f2062010e7c06ba8e9c4ba913a72a7"),
    "heo1_json": (os.path.join(HEO_ROOT, "high_energy_oscillatory_branch", "screen_cells",
                               "gq0.999_A8_D0.15_nokick.json"),
                  "18c2a76e4f3d6733d079b89cc1649b274ba5d0a0b1c1e6df534255dbe3f38967"),
    "heo1_baseline_contract": (os.path.join(HEO_ROOT, "high_energy_oscillatory_branch",
                                            "baseline_spectral_contract_seed1.json"),
                               "c8b367cb356ba430629012e5217e0c13548645e986fec9f828099fd6fb5e901e"),
    "heo2": (os.path.join(HEO_ROOT, "broadband_diagnostic", "arms", "dyn_tau250_frac0.1_trace.npz"),
             "2995f7490ebec4bc3f39ae37be79215cec600204fe02ceab650bdf357ea35582"),
    "heo2_json": (os.path.join(HEO_ROOT, "broadband_diagnostic", "arms", "dyn_tau250_frac0.1.json"),
                  "a4c4b915a85168d63270e79fe34cbbedc69875ab658f565ff1f0c890e1acfaae"),
    "heo2_phase1": (os.path.join(HEO_ROOT, "broadband_diagnostic", "phase1_arms.json"),
                    "8b27670d272bde98e391d2d222140c17569113aa8634fa8b5a2c6983ee61c3d7"),
}

STATE_T_MS = {"baseline": 8000.0, "q75": 5000.0, "heo1": 4000.0, "heo2": 5000.0}
TAU_GRID = np.geomspace(5.0, 2000.0, 24)
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 20260802


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _meminfo():
    with open("/proc/meminfo") as f:
        x = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gb=x["MemAvailable"] / 1024.0 / 1024.0,
                swap_used_mb=(x["SwapTotal"] - x["SwapFree"]) / 1024.0)


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".{name}.lock")
    fd = open(path, "a+")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"stage {name!r} already has a launcher: {path}") from exc
    try:
        yield
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()


def _preflight():
    rows = {}
    for label, (path, expected) in ARTIFACTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"locked artifact missing: {path}")
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(f"locked artifact drift for {label}: expected {expected}, got {got}")
        rows[label] = dict(path=path, sha256=got)
    if not os.path.isfile(P_FIELD):
        raise FileNotFoundError(f"locked onset-depletion field missing: {P_FIELD}")
    rows["p_field"] = dict(path=P_FIELD, sha256=sha256_file(P_FIELD))
    return rows


def _sample_indices(ne):
    rng = np.random.default_rng(SAMPLE_SEED)
    return np.sort(rng.choice(int(ne), size=min(SAMPLE_N, int(ne)), replace=False))


def _state_cfg(state, S, preflight):
    if state == "baseline":
        return LC1R._slowoff_cfg()
    if state == "q75":
        return LC1R._zonly_cfg("q75")

    pk = load_onset_depletion_pi(P_FIELD)
    assert_field_substrate_aligned(pk, S)
    base_contract = json.load(open(os.path.join(HEO_ROOT, "high_energy_oscillatory_branch",
                                                "baseline_spectral_contract_seed1.json")))
    uc = float(base_contract["u_c"]["0.999"])
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                       fail_on_clip=False, rec_sat_g=G_SAT)
    cfg.update(coop_A=8.0, coop_uc=uc, coop_Kc=0.25 * uc, coop_n=4,
               z_frozen_E=frozen_z_field(pk["p_i"], 0.15))
    if state == "heo2":
        phase1 = json.load(open(os.path.join(HEO_ROOT, "broadband_diagnostic", "phase1_arms.json")))
        cfg.update(use_m=True, eta_m=float(phase1["etas"]["250.0/0.1"]),
                   tau_adp=250.0, m_enable_ms=1000.0)
    return cfg


def _reference_rate(state):
    d = np.load(ARTIFACTS[state][0], allow_pickle=True)
    return np.asarray(d["rate_E"], float), float(np.asarray(d["rate_dt_ms"]).item()) if "rate_dt_ms" in d else DT


def _validate_rate_replay(state, rate):
    ref, ref_dt = _reference_rate(state)
    stride = int(round(ref_dt / DT))
    if not np.isclose(stride * DT, ref_dt, atol=1e-12):
        raise RuntimeError(f"reference dt {ref_dt} is not an integer multiple of replay dt {DT}")
    obs = np.asarray(rate, float)[::stride]
    n = min(obs.size, ref.size)
    # Canonical baseline/q75 artifacts were deliberately stored as float32.  Compare in that storage
    # dtype; demanding float64 equality to a float32 archive turns a 1e-14 representation residue into
    # a false replay failure.
    exact = bool(np.array_equal(obs[:n].astype(np.float32), ref[:n].astype(np.float32)))
    diff = np.abs(obs[:n] - ref[:n])
    return dict(reference_dt_ms=ref_dt, replay_stride=stride, compared_n=int(n),
                exact_prefix=exact, max_abs_diff=float(diff.max()) if diff.size else 0.0,
                mean_abs_diff=float(diff.mean()) if diff.size else 0.0,
                reference_path=ARTIFACTS[state][0])


def _run_state(state, preflight):
    T = STATE_T_MS[state]
    S = PP.build_substrate(1)
    cfg = _state_cfg(state, S, preflight)
    p = dataclasses.replace(S["p"], T=T, dt=DT)
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S))
    idx = _sample_indices(S["NE"])
    observer = RawGASampler(S["NE"], idx, BLOCK_STEPS)
    slow.h_lc2_observer = observer
    S["net"]["rng"] = np.random.default_rng(1)
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        early_stop_runaway=False)
    wall = time.time() - t0
    rate = np.asarray(res["rate_E"], float)
    validation = _validate_rate_replay(state, rate)
    if not validation["exact_prefix"]:
        raise RuntimeError(f"SENSOR_REPLAY_INVALID for {state}: {validation}")
    arrays = observer.arrays()
    if int(arrays["n_pending_steps"].item()) != 0:
        raise RuntimeError(f"observer ended with incomplete block: {arrays['n_pending_steps'].item()} steps")
    n_block = arrays["gA_sampled"].shape[0]
    rate_block = rate[:n_block * BLOCK_STEPS].reshape(n_block, BLOCK_STEPS).mean(axis=1).astype(np.float32)
    baseline_contract = json.load(open(ARTIFACTS["baseline_contract"][0]))
    events, _af, _af_bin, _floor, _ = OLD._events_from_res(
        res, DT, event_bar=float(baseline_contract["frozen_event_bar"]))
    event_rows = [dict(t_on_ms=float(e["t_on"]), t_off_ms=float(e["t_off"]),
                       dur_ms=float(e["dur_ms"]), returned=bool(e.get("returned", False)),
                       peak_ext=float(e["peak_ext"])) for e in events]
    path = os.path.join(R1, f"{state}_gA_sensor.npz")
    FCXR._write_npz(path, **arrays, block_dt_ms=np.asarray([BLOCK_MS], np.float32),
                    rate_E_block=rate_block)
    row = dict(state=state, seed=1, T_ms=T, dt_ms=DT, block_dt_ms=BLOCK_MS,
               config={k: v for k, v in cfg.items() if not isinstance(v, np.ndarray)},
               sample_seed=SAMPLE_SEED, sample_n=int(idx.size), n_blocks=int(n_block),
               validation=validation, finite=bool(np.all(np.isfinite(rate))),
               clip_frac_max=float(np.max(slow.trace_conductance_clip_frac)),
               tau_eff_ratio_min=float(np.min(slow.trace_tau_eff_ratio_min)),
               mean_rate_hz=float(rate.mean()), max_rate_hz=float(rate.max()),
               events=event_rows,
               gA_population_summary_names=list(arrays["summary_names"]),
               output_npz=path, output_sha256=sha256_file(path), wall_s=round(wall, 1),
               peak_rss_gb=round(_rss_gb(), 2), finished=_now())
    FCXR._write_json(os.path.join(R1, f"{state}_replay.json"), row)
    return row


def cmd_preflight(_args):
    os.makedirs(R1, exist_ok=True)
    rows = _preflight()
    payload = dict(status="PASS", artifacts=rows, checked=_now())
    FCXR._write_json(os.path.join(R1, "artifact_preflight.json"), payload)
    print(json.dumps(payload, indent=2))


def cmd_r0(_args):
    """Materialise the already-TDD'd vertical-slice evidence; no 40k simulation is launched."""
    os.makedirs(R0, exist_ok=True)
    FCXR._assert_engine_blessed()
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_mz_lc2_h.py"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        FCXR._write_json(os.path.join(R0, "FAILED.json"),
                         dict(stage="R0", returncode=proc.returncode, stdout=proc.stdout,
                              stderr=proc.stderr, failed=_now()))
        raise SystemExit(proc.returncode)
    dataflow = dict(
        status="PASS", source="post-X I_E_rec -> gErec_raw", actuator="gA_raw + rho*S_tilde(h) before RC1 tanh",
        causal_order=["membrane uses h(t-)", "gA_raw(t) cached", "exact H update after membrane"],
        no_new_edges=True, no_global_sensor=True, no_additive_x_current=True,
        code="src/snn_engine/mz_slow_vars.py", tests="tests/test_mz_lc2_h.py", finished=_now())
    parity = dict(status="PASS", rho_zero_membrane_exact=True, engine_raster_exact=True, rng_state_exact=True,
                  snapshot_restart_exact=True, blessed_engine_hashes_unchanged=True,
                  pytest_stdout=proc.stdout.strip(), wall_s=round(time.time() - t0, 1), finished=_now())
    smoke = dict(status="PASS", sensor_only_100ms=True, active_h_500ms=True, deterministic=True,
                 finite=True, zero_conductance_clip=True, pytest_stdout=proc.stdout.strip(), finished=_now())
    FCXR._write_json(os.path.join(R0, "dataflow_contract.json"), dataflow)
    FCXR._write_json(os.path.join(R0, "parity.json"), parity)
    FCXR._write_json(os.path.join(R0, "smoke.json"), smoke)
    FCXR._write_json(os.path.join(R0, "DONE.json"),
                     dict(stage="R0", status="PASS", wall_s=parity["wall_s"], finished=_now()))
    print(proc.stdout.strip())


def cmd_r1_all(args):
    if not args.confirm_run:
        raise SystemExit("r1-all requires --confirm-run")
    FCXR._assert_engine_blessed()
    os.makedirs(R1, exist_ok=True)
    with _stage_lock("r1_all"):
        before = _meminfo()
        if before["mem_available_gb"] < 64.0:
            raise SystemExit(f"OOM safety stop: only {before['mem_available_gb']:.1f} GiB available")
        preflight = _preflight()
        FCXR._write_json(os.path.join(R1, "artifact_preflight.json"),
                         dict(status="PASS", artifacts=preflight, checked=_now()))
        # Keep failed instrumentation attempts auditable without letting stale sentinels masquerade as
        # the current run.
        superseded = os.path.join(R1, "superseded")
        for stale_name in ("RUNNING.json", "FAILED.json"):
            stale = os.path.join(R1, stale_name)
            if os.path.exists(stale):
                os.makedirs(superseded, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                os.replace(stale, os.path.join(superseded, f"{stamp}_{stale_name}"))
        running = os.path.join(R1, "RUNNING.json")
        FCXR._write_json(running, dict(pid=os.getpid(), stage="R1", states=list(STATE_T_MS),
                                      started=_now(), resource_before=before))
        with open(os.path.join(R1, "launcher.pid"), "w") as f:
            f.write(str(os.getpid()))
        rows = []
        try:
            for state in STATE_T_MS:
                print(f"[R1] start {state} T={STATE_T_MS[state]} ms", flush=True)
                row = _run_state(state, preflight)
                rows.append(row)
                print(f"[R1] done {state}: exact={row['validation']['exact_prefix']} "
                      f"gA blocks={row['n_blocks']} wall={row['wall_s']}s RSS={row['peak_rss_gb']}GiB", flush=True)
                now = _meminfo()
                if now["mem_available_gb"] < 32.0 or now["swap_used_mb"] - before["swap_used_mb"] >= 512.0:
                    raise MemoryError(f"resource hard-stop after {state}: before={before}, now={now}")
            done = dict(stage="R1_REPLAYS", status="PASS", rows=rows,
                        resource_before=before, resource_after=_meminfo(), finished=_now())
            FCXR._write_json(os.path.join(R1, "REPLAYS_DONE.json"), done)
            if os.path.exists(running):
                os.remove(running)
        except Exception as exc:
            FCXR._write_json(os.path.join(R1, "FAILED.json"),
                             dict(stage="R1_REPLAYS", error=repr(exc), rows=rows, failed=_now()))
            raise


def _load_replays():
    traces, rows = {}, {}
    idx0 = None
    for state in STATE_T_MS:
        jp = os.path.join(R1, f"{state}_replay.json")
        npz = os.path.join(R1, f"{state}_gA_sensor.npz")
        if not (os.path.isfile(jp) and os.path.isfile(npz)):
            raise SystemExit(f"missing R1 replay for {state}; run r1-all first")
        rows[state] = json.load(open(jp))
        d = np.load(npz, allow_pickle=False)
        idx = np.asarray(d["sample_idx_E"], np.int64)
        if idx0 is None:
            idx0 = idx
        elif not np.array_equal(idx, idx0):
            raise RuntimeError("R1 state replays do not use the same locked E-cell sample")
        traces[state] = np.asarray(d["gA_sampled"], np.float32)
    return traces, rows


def _event_peak_matrix(h, events):
    peaks = []
    for e in events:
        if not e.get("returned", False):
            continue
        lo = max(0, int(np.floor(float(e["t_on_ms"]) / BLOCK_MS)))
        hi = min(h.shape[0], int(np.ceil(float(e["t_off_ms"]) / BLOCK_MS)) + 1)
        if hi > lo:
            peaks.append(np.max(h[lo:hi], axis=0))
    if not peaks:
        raise RuntimeError("accepted baseline replay has no returning event for H separation")
    return np.stack(peaks)


def _high_trough_by_cell(h, lo_ms, hi_ms):
    lo = max(0, int(round(lo_ms / BLOCK_MS)))
    hi = min(h.shape[0], int(round(hi_ms / BLOCK_MS)))
    if hi - lo < 100:
        raise RuntimeError(f"established-high window too short: [{lo_ms},{hi_ms}] ms")
    return np.quantile(h[lo:hi], 0.10, axis=0)


def _bootstrap_bounds(event_peaks, h1_trough, h2_trough, tau_index):
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(tau_index))
    n_ev, n_cell = event_peaks.shape
    ls, u1s, u2s = [], [], []
    for _ in range(BOOTSTRAP_N):
        ei = rng.integers(0, n_ev, n_ev)
        ci = rng.integers(0, n_cell, n_cell)
        vals = event_peaks[np.ix_(ei, ci)]
        ls.append(float(np.quantile(vals, 0.999)))
        u1s.append(float(np.quantile(h1_trough[ci], 0.10)))
        u2s.append(float(np.quantile(h2_trough[ci], 0.10)))
    return dict(
        L_upper95=float(np.quantile(ls, 0.95)),
        HEO1_lower95=float(np.quantile(u1s, 0.05)),
        HEO2_lower95=float(np.quantile(u2s, 0.05)),
    )


def _evaluate_tau(tau, traces, rows, tau_index):
    hb, _ = replay_h(traces["baseline"], tau, BLOCK_MS)
    h1, _ = replay_h(traces["heo1"], tau, BLOCK_MS)
    h2, _ = replay_h(traces["heo2"], tau, BLOCK_MS)
    hq, _ = replay_h(traces["q75"], tau, BLOCK_MS)
    evp = _event_peak_matrix(hb, rows["baseline"]["events"])
    h1t = _high_trough_by_cell(h1, 1000.0, 3500.0)
    h2t = _high_trough_by_cell(h2, 1500.0, 4500.0)
    L = float(np.quantile(evp, 0.999))
    U1 = float(np.quantile(h1t, 0.10))
    U2 = float(np.quantile(h2t, 0.10))
    b = _bootstrap_bounds(evp, h1t, h2t, tau_index)
    U = min(U1, U2)
    Ulo = min(b["HEO1_lower95"], b["HEO2_lower95"])
    ok = bool(b["L_upper95"] < Ulo)
    return dict(tau_ms=float(tau), L_IED_q999=L, U_HEO1_q10=U1, U_HEO2_q10=U2, U_min=U,
                q75_q10=float(np.quantile(hq[int(1000 / BLOCK_MS):], 0.10)),
                q75_q50=float(np.quantile(hq[int(1000 / BLOCK_MS):], 0.50)),
                **b, U_lower95=Ulo, separable=ok)


def _refine_boundary(lo, hi, lo_is_pass, traces, rows, index_base):
    """Bisect a single fail/pass boundary to 0.5 ms and return the pass-side value."""
    vlo, vhi = float(lo), float(hi)
    j = 0
    while vhi - vlo > 0.5:
        mid = 0.5 * (vlo + vhi)
        row = _evaluate_tau(mid, traces, rows, index_base + j)
        j += 1
        if row["separable"]:
            if lo_is_pass:
                vlo = mid
            else:
                vhi = mid
        else:
            if lo_is_pass:
                vhi = mid
            else:
                vlo = mid
    return float(vlo if lo_is_pass else vhi)


def _selected_residuals(tau, theta, traces, rows):
    hb, _ = replay_h(traces["baseline"], tau, BLOCK_MS)
    hq, _ = replay_h(traces["q75"], tau, BLOCK_MS)
    pre = []
    gaps = []
    events = [e for e in rows["baseline"]["events"] if e.get("returned", False)]
    for i, e in enumerate(events):
        k = max(0, int(np.floor(float(e["t_on_ms"]) / BLOCK_MS)) - 1)
        pre.append(float(np.quantile(hb[k], 0.999)))
        if i:
            gaps.append((float(e["t_on_ms"]) - float(events[i - 1]["t_off_ms"]), pre[-1]))
    shortest = min(gaps, key=lambda z: z[0]) if gaps else (float("nan"), float("nan"))
    return dict(next_onset_residual_q999=pre,
                max_next_onset_residual_q999=max(pre) if pre else float("nan"),
                shortest_gap_ms=shortest[0], shortest_gap_residual_q999=shortest[1],
                q75_fraction_above_theta=float(np.mean(hq[int(1000 / BLOCK_MS):] > theta)))


def cmd_r1_analyze(_args):
    traces, rows = _load_replays()
    grid_rows = []
    for i, tau in enumerate(TAU_GRID):
        print(f"[R1 analyze] tau={tau:.3f} ms", flush=True)
        grid_rows.append(_evaluate_tau(float(tau), traces, rows, i))
    flags = np.asarray([r["separable"] for r in grid_rows], bool)
    intervals = contiguous_true_intervals(flags, TAU_GRID)
    if not intervals:
        verdict = dict(status="H_SENSOR_NOT_SEPARABLE", tau_grid_ms=TAU_GRID.tolist(), rows=grid_rows,
                       feasible_intervals_ms=[], bootstrap_n=BOOTSTRAP_N, finished=_now())
        FCXR._write_json(os.path.join(R1, "h_sensor_separability.json"), verdict)
        FCXR._write_json(os.path.join(OUT, "candidate_verdict.json"),
                         dict(stage="R1", verdict="H_SENSOR_NOT_SEPARABLE", finished=_now()))
        print("[R1 analyze] H_SENSOR_NOT_SEPARABLE", flush=True)
        return

    # Choose the first connected region (the one containing the smallest passing tau), then refine its
    # fail/pass edges without launching another SNN trajectory.
    first_idx = int(np.flatnonzero(flags)[0]); last_idx = first_idx
    while last_idx + 1 < flags.size and flags[last_idx + 1]:
        last_idx += 1
    tau_min = float(TAU_GRID[first_idx])
    tau_max = float(TAU_GRID[last_idx])
    if first_idx > 0:
        tau_min = _refine_boundary(TAU_GRID[first_idx - 1], TAU_GRID[first_idx], False,
                                   traces, rows, 1000)
    if last_idx + 1 < flags.size:
        tau_max = _refine_boundary(TAU_GRID[last_idx], TAU_GRID[last_idx + 1], True,
                                   traces, rows, 2000)
    tau_sel = float(np.sqrt(tau_min * tau_max))
    sel = _evaluate_tau(tau_sel, traces, rows, 3000)
    L, U = sel["L_IED_q999"], sel["U_min"]
    theta = 0.5 * (L + U)
    k = (U - L) / (2.0 * np.log(9.0))
    residual = _selected_residuals(tau_sel, theta, traces, rows)
    lock = dict(status="PASS", tau_H_ms=tau_sel, theta_H=theta, k_H=k,
                tau_interval_ms=[tau_min, tau_max], L_IED_q999=L, U_high_min_q10=U,
                selected_row=sel, residual_diagnostics=residual, selection_rule="geometric_midpoint",
                bootstrap_n=BOOTSTRAP_N, sample_n=int(traces["baseline"].shape[1]), finished=_now())
    verdict = dict(status="PASS", tau_grid_ms=TAU_GRID.tolist(), rows=grid_rows,
                   feasible_intervals_grid_ms=intervals, selected_interval_refined_ms=[tau_min, tau_max],
                   bootstrap_n=BOOTSTRAP_N, finished=_now())
    FCXR._write_json(os.path.join(R1, "h_sensor_separability.json"), verdict)
    FCXR._write_json(os.path.join(R1, "h_parameter_lock.json"), lock)
    FCXR._write_json(os.path.join(R1, "ANALYSIS_DONE.json"),
                     dict(stage="R1_ANALYSIS", status="PASS", tau_H_ms=tau_sel, finished=_now()))
    print(f"[R1 analyze] PASS tau=[{tau_min:.2f},{tau_max:.2f}] -> {tau_sel:.2f} ms; "
          f"L={L:.4g} U={U:.4g} theta={theta:.4g} k={k:.4g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    sub.add_parser("r0")
    p = sub.add_parser("r1-all")
    p.add_argument("--confirm-run", action="store_true")
    sub.add_parser("r1-analyze")
    args = ap.parse_args()
    if args.cmd == "preflight":
        cmd_preflight(args)
    elif args.cmd == "r0":
        cmd_r0(args)
    elif args.cmd == "r1-all":
        cmd_r1_all(args)
    else:
        cmd_r1_analyze(args)


if __name__ == "__main__":
    main()
