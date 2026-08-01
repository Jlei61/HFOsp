#!/usr/bin/env python3
"""FCXR-LC2 closed-loop exploration runner.

E0/E1 are read-only with respect to the SNN: they lock provenance and characterize the already accepted
R1 traces.  Later subcommands are added stage-by-stage behind the locked design contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_topic4_mz_fcxr as FCXR  # noqa: E402
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
    return float(np.mean(h[np.ix_(time_idx, cells)] > theta)), float(np.quantile(boots, .05)), float(np.quantile(boots, .95))


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
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    FCXR._write_json(os.path.join(OUT, "E1_DONE.json"),
                     dict(stage="E1", status=summary["status"], candidates=len(selected), finished=_now()))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("r1-characterize")
    args = ap.parse_args()
    if args.cmd == "lock":
        cmd_lock(args)
    else:
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


if __name__ == "__main__":
    main()
