#!/usr/bin/env python3
"""FCXR-LC2 closed-loop exploration runner.

E0/E1 are read-only with respect to the SNN: they lock provenance and characterize the already accepted
R1 traces.  Later subcommands are added stage-by-stage behind the locked design contract.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_topic4_mz_fcxr as FCXR  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    load_onset_depletion_pi, assert_field_substrate_aligned, frozen_z_field,
)
from src.topic4_fcxr_lc2_core import (  # noqa: E402
    sha256_file, replay_h, sustained_latch_score, empirical_false_latch_threshold,
    pareto_mask, select_sensor_candidates,
)


BASE = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core")
R1 = os.path.join(BASE, "r1_sensor")
OUT = os.path.join(BASE, "closed_loop_exploration")
P_FIELD = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                       "snapshots", "zA_q75_tz5000", "seed_1.npz")
HEO2_TRACE = ("/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-heo1/results/"
              "topic4_sef_hfo/mz_full_conductance_spatial_relay/broadband_diagnostic/arms/"
              "dyn_tau250_frac0.1_trace.npz")

TAU_GRID = np.geomspace(20.0, 2000.0, 25)
FALSE_TARGETS = (0.0, 0.05, 0.10, 0.25)
LATCH_MS = 50
TRIM_MS = 50
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 20260802
K_RATIOS = (0.05, 0.10, 0.20)
RHO_FRACS = (0.10, 0.20, 0.35, 0.50, 0.70)
X_DEPLETION_LEVELS = (0.128, 0.214)
SCREEN_WORKER_CHOICES = (1, 2, 3, 4)
DT = 0.05
ENGINE_FILES = (
    "src/snn_engine/kick_probe.py", "src/snn_engine/params.py", "src/snn_engine/model.py",
    "src/snn_engine/connectivity.py", "src/snn_engine/connectivity_rot.py", "src/snn_engine/lfp.py",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _meminfo():
    with open("/proc/meminfo") as f:
        x = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=x["MemAvailable"] / 1024.0 / 1024.0,
                swap_used_mib=(x["SwapTotal"] - x["SwapFree"]) / 1024.0)


def cmd_lock(_args):
    os.makedirs(OUT, exist_ok=True)
    traces = {}
    for state in ("baseline", "q75", "heo1", "heo2"):
        path = os.path.join(R1, f"{state}_gA_sensor.npz")
        if not os.path.isfile(path):
            raise SystemExit(f"missing accepted R1 trace: {path}")
        traces[state] = dict(path=path, sha256=sha256_file(path))
    if not os.path.isfile(HEO2_TRACE) or not os.path.isfile(P_FIELD):
        raise SystemExit("locked HEO2 LFP or p-field artifact is missing")
    payload = dict(
        status="LOCKED", design_commit="8a4dad00", launcher_head=_git_head(), locked_at=_now(),
        traces=traces,
        heo2_lfp=dict(path=HEO2_TRACE, sha256=sha256_file(HEO2_TRACE)),
        p_field=dict(path=P_FIELD, sha256=sha256_file(P_FIELD)),
        tau_grid_ms=TAU_GRID.tolist(), false_latch_targets=list(FALSE_TARGETS), latch_ms=LATCH_MS,
        bout_trim_ms=TRIM_MS, bootstrap_n=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED,
        k_over_theta=list(K_RATIOS), rho_over_gsat=list(RHO_FRACS), g_sat=21.6,
        frozen_depletion=dict(healthy=0.0, susceptible=0.15),
        x_depletion_levels=list(X_DEPLETION_LEVELS),
        development_noise=401, replication_noise=402, connection_seed=1,
        locked_out=["M", "K", "A", "ELR", "new_EE_edges", "global_seizure_label", "kick"],
        engine_hashes={p: sha256_file(os.path.join(ROOT, p)) for p in ENGINE_FILES},
    )
    FCXR._write_json(os.path.join(OUT, "execution_lock.json"), payload)
    print(json.dumps(payload, indent=2))


def _load_r1():
    traces, rows = {}, {}
    idx = None
    for state in ("baseline", "q75", "heo1", "heo2"):
        d = np.load(os.path.join(R1, f"{state}_gA_sensor.npz"), allow_pickle=False)
        this = np.asarray(d["sample_idx_E"], np.int64)
        if idx is None:
            idx = this
        elif not np.array_equal(idx, this):
            raise RuntimeError("R1 traces have different sampled E cells")
        traces[state] = np.asarray(d["gA_sampled"], np.float32)
        rows[state] = _load_json(os.path.join(R1, f"{state}_replay.json"))
    return traces, rows, idx


def _event_indices(events, n_t, returned_only=False, trim=0, first_floor=None):
    out = []
    for e in events:
        if returned_only and not e.get("returned", False):
            continue
        lo = int(np.ceil(float(e["t_on_ms"]))) + int(trim)
        hi = int(np.floor(float(e["t_off_ms"]))) - int(trim)
        if first_floor is not None and not out:
            lo = max(lo, int(first_floor))
        lo, hi = max(0, lo), min(n_t, hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def _concat_indices(intervals):
    return np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in intervals])


def _rolling_mean(x, w):
    x = np.asarray(x, float)
    if x.size < w:
        return np.empty(0)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _gap_diagnostic(rows, baseline_rate_block):
    events = [e for e in rows["heo2"]["events"] if e.get("returned", False)]
    if len(events) < 2:
        raise RuntimeError("HEO2 needs two returned bouts for the locked gap diagnostic")
    gap = (int(np.ceil(events[0]["t_off_ms"])) + TRIM_MS,
           int(np.floor(events[1]["t_on_ms"])) - TRIM_MS)
    base_mask = np.ones(baseline_rate_block.size, bool)
    for e in rows["baseline"]["events"]:
        lo = max(0, int(e["t_on_ms"]) - 50); hi = min(base_mask.size, int(e["t_off_ms"]) + 51)
        base_mask[lo:hi] = False
    base_roll = _rolling_mean(baseline_rate_block, 50)
    valid = base_mask[24:24 + base_roll.size]
    rate_ref_q95 = float(np.quantile(base_roll[valid], 0.95))
    heo2_rate = np.load(os.path.join(R1, "heo2_gA_sensor.npz"), allow_pickle=False)["rate_E_block"]
    gap_rate = _rolling_mean(heo2_rate[gap[0]:gap[1]], 50)
    gap_rate_median = float(np.median(gap_rate))

    d = np.load(HEO2_TRACE, allow_pickle=False)
    lfp = np.asarray(d["lfp_trace"], float)
    dt = float(np.asarray(d["dt"]).item())
    block = int(round(20.0 / dt))

    def rms_window(lo_ms, hi_ms):
        lo = int(round(lo_ms / dt)); hi = int(round(hi_ms / dt))
        x = lfp[lo:hi]
        n = x.shape[0] // block
        x = x[:n * block].reshape(n, block, x.shape[1])
        return np.sqrt(np.mean(x * x, axis=1) + 1e-20)

    low = rms_window(0.0, 150.0)
    gap_rms = rms_window(*gap)
    low_med = np.median(low, axis=0); gap_med = np.median(gap_rms, axis=0)
    delta_db = 20.0 * np.log10((gap_med + 1e-12) / (low_med + 1e-12))
    within = np.abs(delta_db) <= 3.0
    rest_like = bool(gap_rate_median <= rate_ref_q95 and np.sum(within) >= 12)
    return dict(
        raw_gap_ms=[float(events[0]["t_off_ms"]), float(events[1]["t_on_ms"])],
        trimmed_gap_ms=list(gap), raw_gap_duration_ms=float(events[1]["t_on_ms"] - events[0]["t_off_ms"]),
        rate_ref_interevent_q95_hz=rate_ref_q95, gap_rate50_median_hz=gap_rate_median,
        lfp_delta_db_per_contact=delta_db.tolist(), contacts_within_3db=int(np.sum(within)),
        classification="rest_like" if rest_like else "silent_gap_unresolved",
    )


def _spatial_blocks(pos, L, n_side=4):
    xy = np.floor(np.clip(pos / float(L), 0.0, np.nextafter(1.0, 0.0)) * n_side).astype(int)
    return xy[:, 0] * n_side + xy[:, 1]


def _block_duty_ci(h, time_idx, cell_mask, block_id, theta, rng):
    cells = np.flatnonzero(cell_mask)
    if cells.size == 0 or time_idx.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    vals = []
    t0, t1 = int(time_idx.min()), int(time_idx.max()) + 1
    keep_t = np.zeros(h.shape[0], bool); keep_t[time_idx] = True
    for a in range(t0, t1, 50):
        ti = np.arange(a, min(a + 50, t1)); ti = ti[keep_t[ti]]
        if not ti.size:
            continue
        for b in np.unique(block_id[cells]):
            ci = cells[block_id[cells] == b]
            if ci.size:
                vals.append(float(np.mean(h[np.ix_(ti, ci)] > theta)))
    vals = np.asarray(vals, float)
    if not vals.size:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.empty(BOOTSTRAP_N)
    for k in range(BOOTSTRAP_N):
        boots[k] = np.mean(vals[rng.integers(0, vals.size, vals.size)])
    # The registered estimand gives each 50 ms x spatial block equal weight.  Returning the raw
    # cell-time mean here would mix estimands and can place the point estimate outside its own CI when
    # spatial blocks contain unequal sampled-cell counts.
    return float(np.mean(vals)), float(np.quantile(boots, .05)), float(np.quantile(boots, .95))


def _latch_scores(hb, events):
    a = np.quantile(hb, 0.99, axis=1)
    ev = [e for e in events if e.get("returned", False)]
    scores = []
    for i, e in enumerate(ev):
        lo = max(0, int(np.ceil(float(e["t_off_ms"]))))
        hi = int(np.floor(float(ev[i + 1]["t_on_ms"]))) if i + 1 < len(ev) else a.size
        if hi - lo >= LATCH_MS:
            scores.append(sustained_latch_score(a[lo:hi], LATCH_MS))
    return np.asarray(scores, float)


def cmd_r1_characterize(_args):
    lock_path = os.path.join(OUT, "execution_lock.json")
    if not os.path.isfile(lock_path):
        raise SystemExit("run the pre-outcome lock first")
    traces, rows, sample_idx = _load_r1()
    p = np.load(P_FIELD, allow_pickle=True)
    pos = np.asarray(p["pos_E"], float)[sample_idx]
    L = float(np.asarray(p["L"]).item())
    block_id = _spatial_blocks(pos, L)

    h2_intervals = _event_indices(rows["heo2"]["events"], traces["heo2"].shape[0],
                                  trim=TRIM_MS, first_floor=1500)
    h2_active_idx = _concat_indices(h2_intervals)
    h1_active_idx = np.arange(1000, min(3500, traces["heo1"].shape[0]), dtype=np.int64)
    h2_full_idx = np.arange(1500, min(4500, traces["heo2"].shape[0]), dtype=np.int64)
    support_bar = np.quantile(traces["baseline"], 0.99, axis=0)
    support_score = np.median(traces["heo2"][h2_active_idx], axis=0)
    support = support_score > support_bar
    support_fraction = float(np.mean(support))
    support_status = "PASS" if support_fraction >= 0.02 else "UNRESOLVED_SUPPORT"
    gap_diag = _gap_diagnostic(rows, np.load(os.path.join(R1, "baseline_gA_sensor.npz"), allow_pickle=False)["rate_E_block"])
    gap_a, gap_b = map(int, gap_diag["trimmed_gap_ms"])
    all_cells = np.ones(sample_idx.size, bool)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out_rows = []

    for tau in TAU_GRID:
        print(f"[E1] tau={tau:.3f} ms", flush=True)
        hb, _ = replay_h(traces["baseline"], tau, 1.0)
        h1, _ = replay_h(traces["heo1"], tau, 1.0)
        h2, _ = replay_h(traces["heo2"], tau, 1.0)
        scores = _latch_scores(hb, rows["baseline"]["events"])
        for target in FALSE_TARGETS:
            theta, observed = empirical_false_latch_threshold(scores, target)
            d1a = _block_duty_ci(h1, h1_active_idx, all_cells, block_id, theta, rng)
            d1s = _block_duty_ci(h1, h1_active_idx, support, block_id, theta, rng)
            d2a = _block_duty_ci(h2, h2_active_idx, all_cells, block_id, theta, rng)
            d2s = _block_duty_ci(h2, h2_active_idx, support, block_id, theta, rng)
            d2full = _block_duty_ci(h2, h2_full_idx, support, block_id, theta, rng)
            mask = support if np.any(support) else all_cells
            start = np.median(h2[gap_a:min(gap_a + 20, gap_b), mask])
            end = np.median(h2[max(gap_a, gap_b - 20):gap_b, mask])
            bridge = float(end / max(start, 1e-12))
            bridge_above = float(np.mean(h2[max(gap_a, gap_b - 20):gap_b, mask] > theta))
            out_rows.append(dict(
                tau_ms=float(tau), false_latch_target=float(target), theta=float(theta),
                false_latch_fraction=float(observed), n_ied_scores=int(scores.size),
                false_latch_resolution=float(1.0 / scores.size),
                heo1_all_duty=d1a[0], heo1_all_ci05=d1a[1], heo1_all_ci95=d1a[2],
                heo1_support_duty=d1s[0], heo1_support_ci05=d1s[1], heo1_support_ci95=d1s[2],
                heo2_active_all_duty=d2a[0], heo2_active_all_ci05=d2a[1], heo2_active_all_ci95=d2a[2],
                heo2_active_support_duty=d2s[0], heo2_active_support_ci05=d2s[1], heo2_active_support_ci95=d2s[2],
                heo2_full_support_duty=d2full[0], heo2_full_support_ci05=d2full[1], heo2_full_support_ci95=d2full[2],
                gap_persistence=bridge, gap_end_support_above_theta=bridge_above,
            ))

    pmask = pareto_mask(out_rows, ["false_latch_fraction"],
                        ["heo1_support_duty", "heo2_active_support_duty", "gap_persistence"])
    for r, flag in zip(out_rows, pmask):
        r["pareto"] = bool(flag)
    selected = select_sensor_candidates(out_rows, max_n=6)
    for i, row in enumerate(selected, 1):
        row["candidate_id"] = f"H{i}"
    summary = dict(
        status="SENSOR_CHARACTERIZATION_COMPLETED", finished=_now(),
        segmentation=dict(heo1_active_ms=[1000, 3500], heo2_active_intervals_ms=h2_intervals,
                          heo2_full_window_ms=[1500, 4500], gap=gap_diag),
        support=dict(status=support_status, n=int(np.sum(support)), sample_n=int(support.size),
                     fraction=support_fraction, occupied_spatial_blocks=int(np.unique(block_id[support]).size)),
        bootstrap=dict(kind="within-trajectory time-block x spatial-block", n=BOOTSTRAP_N,
                       not_cross_seed=True),
        rows_n=len(out_rows), pareto_n=int(np.sum(pmask)), selected_candidates=selected,
        interpretation=("R1 characterizes healthy reset, active-state sensitivity and long-gap stress "
                        "separately; it does not adjudicate closed-loop H basin geometry."),
    )
    os.makedirs(OUT, exist_ok=True)
    FCXR._write_json(os.path.join(OUT, "r1_resegmentation_summary.json"), summary)
    FCXR._write_npz(os.path.join(OUT, "r1_sensor_support_map.npz"), sample_idx_E=sample_idx,
                    pos_E=pos.astype(np.float32), spatial_block_id=block_id.astype(np.int16),
                    recruited_support=support, support_score=support_score.astype(np.float32),
                    support_bar=support_bar.astype(np.float32))
    csv_path = os.path.join(OUT, "r1_sensor_pareto.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)
    FCXR._write_json(os.path.join(OUT, "E1_DONE.json"),
                     dict(stage="E1", status=summary["status"], candidates=len(selected), finished=_now()))
    print(json.dumps(summary, indent=2))


def cmd_screen_manifest(_args):
    src = os.path.join(OUT, "r1_resegmentation_summary.json")
    if not os.path.isfile(src):
        raise SystemExit("E1 characterization is required before the screen manifest")
    candidates = _load_json(src)["selected_candidates"]
    rows = []
    for c in candidates:
        for kr in K_RATIOS:
            for rf in RHO_FRACS:
                cid = c["candidate_id"]
                rows.append(dict(
                    index=len(rows), run_id=f"{cid}_k{int(round(kr*100)):02d}_r{int(round(rf*100)):02d}",
                    candidate_id=cid, tau_ms=float(c["tau_ms"]), theta=float(c["theta"]),
                    false_latch_fraction=float(c["false_latch_fraction"]),
                    k_ratio=float(kr), k=float(kr * c["theta"]), rho_fraction=float(rf),
                    rho=float(rf * 21.6), D=0.15, h_init_scale=2.0, T_ms=1000.0,
                    connection_seed=1, noise_seed=401, x_mode="off", M=False, coop_A=0.0,
                ))
    payload = dict(status="LOCKED", stage="E3_SCREEN", code_head=_git_head(), rows=rows,
                   n_rows=len(rows), created=_now())
    FCXR._write_json(os.path.join(OUT, "h_loop_screen_manifest.json"), payload)
    print(json.dumps(dict(status=payload["status"], n_rows=len(rows), code_head=payload["code_head"]), indent=2))


_SCREEN_SUBSTRATE = None


def _screen_substrate():
    global _SCREEN_SUBSTRATE
    if _SCREEN_SUBSTRATE is None:
        _SCREEN_SUBSTRATE = PP.build_substrate(1)
    return _SCREEN_SUBSTRATE


def _run_screen_row(row):
    S = _screen_substrate()
    pk = load_onset_depletion_pi(P_FIELD)
    assert_field_substrate_aligned(pk, S)
    theta = float(row["theta"])
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                       fail_on_clip=False, rec_sat_g=21.6)
    cfg.update(
        coop_A=0.0, use_m=False, use_x=False,
        z_frozen_E=frozen_z_field(pk["p_i"], float(row["D"])),
        use_h_lc2=True, tau_h_lc2=float(row["tau_ms"]), theta_h_lc2=theta,
        k_h_lc2=float(row["k"]), rho_h_lc2=float(row["rho"]),
        h_lc2_init_E=np.full(S["NE"], float(row["h_init_scale"]) * theta),
    )
    p = dataclasses.replace(S["p"], T=float(row["T_ms"]), dt=DT)
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S))
    S["net"]["rng"] = np.random.default_rng(int(row["noise_seed"]))
    t0 = time.time()
    res = simulate_kick(
        p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
        t_kick=1e9, V_th_per_neuron=S["vth"], early_stop_runaway=True,
        es_thresh_hz=300.0, es_dur_ms=100.0)
    wall = time.time() - t0
    rate = np.asarray(res["rate_E"], float)
    n_tail = min(rate.size, int(round(250.0 / DT)))
    tail_rate = rate[-n_tail:]
    tail_bool = np.asarray(res["E_spk_bool"][-n_tail:], bool)
    per_cell_hz = tail_bool.sum(axis=0) / (n_tail * DT * 1e-3)
    ceiling_hz = 0.8 * (1000.0 / float(p.tau_ref_E))
    ceiling_frac = float(np.mean(per_cell_hz >= ceiling_hz))
    htrace = np.asarray(slow.trace_h_lc2_mean, float)
    n_h = min(htrace.size, n_tail)
    ht = htrace[-n_h:]
    x = np.arange(n_h, dtype=float) * DT
    slope_per_ms = float(np.polyfit(x, ht, 1)[0]) if n_h >= 2 else float("nan")
    slope_per_s = 1000.0 * slope_per_ms
    slope_floor = -0.05 * max(float(np.mean(ht)), theta)
    clip = float(np.max(slow.trace_conductance_clip_frac))
    finite = bool(np.all(np.isfinite(rate)) and np.all(np.isfinite(ht)))
    numerical = (not finite) or clip > 0.0 or res["runaway_early_stop_ms"] is not None
    if numerical:
        label = "numerical_failure"
    elif ceiling_frac >= 0.05:
        label = "saturated_tonic"
    elif float(np.mean(tail_rate)) >= 20.0 and slope_per_s >= slope_floor:
        label = "screen_survivor"
    elif float(row["tau_ms"]) > 500.0 and float(np.mean(tail_rate)) >= 20.0:
        label = "unresolved_1s"
    else:
        label = "decay_low"
    stride = max(1, int(round(10.0 / DT)))
    return dict(
        **row, label=label, finite=finite, runaway_early_stop_ms=res["runaway_early_stop_ms"],
        mean_rate_hz=float(np.mean(rate)), tail_rate_hz=float(np.mean(tail_rate)),
        tail_rate_sd_hz=float(np.std(tail_rate)), max_rate_hz=float(np.max(rate)),
        tail_h_mean=float(np.mean(ht)), tail_h_slope_per_s=slope_per_s,
        tail_h_slope_floor=slope_floor, tail_gH_mean=float(np.mean(slow.trace_gH_lc2_mean[-n_h:])),
        tail_gA_mean=float(np.mean(slow.trace_gA_raw_lc2_mean[-n_h:])),
        ceiling_hz=ceiling_hz, refractory_ceiling_fraction=ceiling_frac,
        clip_frac_max=clip, tau_eff_ratio_min=float(np.min(slow.trace_tau_eff_ratio_min)),
        wall_s=round(wall, 2), peak_rss_gib=round(_rss_gib(), 3),
        trace_dt_ms=10.0, rate_trace=rate[::stride].astype(float).tolist(),
        h_trace=htrace[::stride].astype(float).tolist(),
        gH_trace=np.asarray(slow.trace_gH_lc2_mean, float)[::stride].tolist(),
        finished=_now(),
    )


def _screen_cell_path(row):
    return os.path.join(OUT, "screen_cells", f"{row['run_id']}.json")


def _screen_worker(row):
    path = _screen_cell_path(row)
    if os.path.isfile(path):
        return _load_json(path)
    out = _run_screen_row(row)
    FCXR._write_json(path, out)
    return out


def _load_screen_manifest():
    path = os.path.join(OUT, "h_loop_screen_manifest.json")
    if not os.path.isfile(path):
        raise SystemExit("run screen-manifest first")
    return _load_json(path)


def cmd_screen_one(args):
    if not args.confirm_run:
        raise SystemExit("screen-one requires --confirm-run")
    FCXR._assert_engine_blessed()
    manifest = _load_screen_manifest()
    row = manifest["rows"][int(args.index)]
    os.makedirs(os.path.join(OUT, "screen_cells"), exist_ok=True)
    print(json.dumps(_screen_worker(row), indent=2))


def cmd_screen_all(args):
    if not args.confirm_run:
        raise SystemExit("screen-all requires --confirm-run")
    FCXR._assert_engine_blessed()
    manifest = _load_screen_manifest()
    rows = manifest["rows"]
    os.makedirs(os.path.join(OUT, "screen_cells"), exist_ok=True)
    before = _meminfo()
    if before["mem_available_gib"] < 96.0:
        raise SystemExit(f"OOM safety stop: MemAvailable={before['mem_available_gib']:.1f} GiB")
    max_by_mem = int(np.floor((before["mem_available_gib"] - 96.0) / 6.793))
    if int(args.workers) > max_by_mem:
        raise SystemExit(f"OOM safety stop: workers={args.workers} exceeds measured-RSS cap={max_by_mem}")
    running = os.path.join(OUT, "E3_RUNNING.json")
    FCXR._write_json(running, dict(stage="E3", pid=os.getpid(), workers=int(args.workers),
                                  n_rows=len(rows), resource_before=before, started=_now()))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            fut = {ex.submit(_screen_worker, row): row for row in rows}
            for j, f in enumerate(as_completed(fut), 1):
                out = f.result(); results.append(out)
                print(f"[E3] {j}/{len(rows)} {out['run_id']} -> {out['label']} "
                      f"tail={out['tail_rate_hz']:.1f}Hz RSS={out['peak_rss_gib']:.2f}GiB", flush=True)
                now = _meminfo()
                if now["swap_used_mib"] - before["swap_used_mib"] >= 512.0:
                    raise MemoryError(f"swap hard stop: before={before}, now={now}")
        results.sort(key=lambda r: int(r["index"]))
        counts = {k: sum(r["label"] == k for r in results) for k in
                  ("screen_survivor", "unresolved_1s", "decay_low", "saturated_tonic", "numerical_failure")}
        payload = dict(stage="E3", status="COMPLETE", n_rows=len(results), counts=counts,
                       rows=results, resource_before=before, resource_after=_meminfo(), finished=_now())
        FCXR._write_json(os.path.join(OUT, "h_loop_screen.json"), payload)
        FCXR._write_json(os.path.join(OUT, "E3_DONE.json"),
                         dict(stage="E3", status="COMPLETE", counts=counts, finished=_now()))
        if os.path.exists(running):
            os.remove(running)
    except Exception as exc:
        FCXR._write_json(os.path.join(OUT, "E3_FAILED.json"),
                         dict(stage="E3", error=repr(exc), traceback=traceback.format_exc(), failed=_now()))
        raise


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("r1-characterize")
    sub.add_parser("screen-manifest")
    one = sub.add_parser("screen-one")
    one.add_argument("--index", type=int, required=True)
    one.add_argument("--confirm-run", action="store_true")
    allp = sub.add_parser("screen-all")
    allp.add_argument("--workers", type=int, choices=SCREEN_WORKER_CHOICES, default=1)
    allp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock":
        cmd_lock(args)
    elif args.cmd == "r1-characterize":
        os.makedirs(OUT, exist_ok=True)
        running = os.path.join(OUT, "E1_RUNNING.json")
        FCXR._write_json(running, dict(stage="E1", pid=os.getpid(), started=_now()))
        try:
            cmd_r1_characterize(args)
            if os.path.exists(running):
                os.remove(running)
        except Exception as exc:
            FCXR._write_json(os.path.join(OUT, "E1_FAILED.json"),
                             dict(stage="E1", error=repr(exc), traceback=traceback.format_exc(),
                                  failed=_now()))
            raise
    elif args.cmd == "screen-manifest":
        cmd_screen_manifest(args)
    elif args.cmd == "screen-one":
        cmd_screen_one(args)
    else:
        cmd_screen_all(args)


if __name__ == "__main__":
    main()
