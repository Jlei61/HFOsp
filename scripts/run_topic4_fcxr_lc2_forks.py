#!/usr/bin/env python3
"""FCXR-LC2 E4 frozen low/high/X fork runner.

This runner is intentionally separate from the E3 process that may still be executing.  It consumes
only a completed, immutable E3 aggregate and writes a new manifest before any fork is launched.
"""
from __future__ import annotations

import argparse
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

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_fcxr as FCXR  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
import run_topic4_fcxr_lc2_explore as E  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    assert_field_substrate_aligned, frozen_z_field, load_onset_depletion_pi,
)


OUT = E.OUT
P_FIELD = E.P_FIELD
DT = E.DT
ARMS = (
    ("A_low", 0.0, 0.0, 0.0),
    ("A_high", 0.0, 2.0, 0.0),
    ("B", 0.15, 0.0, 0.0),
    ("C", 0.15, 2.0, 0.0),
    ("D1", 0.15, 2.0, 0.128),
    ("D2", 0.15, 2.0, 0.214),
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _meminfo():
    with open("/proc/meminfo") as f:
        x = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(MemAvailable_GiB=x["MemAvailable"] / 1024.0 / 1024.0,
                SwapUsed_MiB=(x["SwapTotal"] - x["SwapFree"]) / 1024.0)


def local_gain_proxy(row):
    """Locked diagnostic ordering proxy: H contribution divided by raw recurrent drive."""
    return float(row["tail_gH_mean"]) / max(float(row["tail_gA_mean"]), 1e-12)


def select_finalists(rows, max_n=6):
    """Deterministic ordering plus one local grid neighbour when available."""
    survivors = [dict(r, tail_local_gain_proxy=local_gain_proxy(r))
                 for r in rows if r.get("label") == "screen_survivor"]
    survivors.sort(key=lambda r: (
        float(r["refractory_ceiling_fraction"]), float(r["rho"]),
        -float(r["tail_local_gain_proxy"]), float(r["false_latch_fraction"]),
        float(r["tau_ms"]), float(r["k_ratio"]), str(r["run_id"]),
    ))
    if not survivors:
        return []
    chosen = [survivors[0]]
    anchor = survivors[0]
    neighbours = [r for r in survivors[1:]
                  if r["candidate_id"] == anchor["candidate_id"] and (
                      (r["k_ratio"] == anchor["k_ratio"] and
                       abs(E.RHO_FRACS.index(r["rho_fraction"]) -
                           E.RHO_FRACS.index(anchor["rho_fraction"])) == 1)
                      or
                      (r["rho_fraction"] == anchor["rho_fraction"] and
                       abs(E.K_RATIOS.index(r["k_ratio"]) -
                           E.K_RATIOS.index(anchor["k_ratio"])) == 1)
                  )]
    if neighbours:
        chosen.append(neighbours[0])
    used = {r["run_id"] for r in chosen}
    chosen.extend(r for r in survivors if r["run_id"] not in used)
    return chosen[:int(max_n)]


def _duration_ms(tau_ms, arm):
    if arm.startswith("D"):
        return min(8000.0, max(3000.0, 4.0 * float(tau_ms)))
    return min(5000.0, max(2000.0, 5.0 * float(tau_ms)))


def cmd_manifest(_args):
    screen_path = os.path.join(OUT, "h_loop_screen.json")
    if not os.path.isfile(screen_path):
        raise SystemExit("completed h_loop_screen.json is required")
    screen = _load_json(screen_path)
    if screen.get("status") != "COMPLETE" or len(screen.get("rows", [])) != 90:
        raise SystemExit("E3 screen is incomplete")
    selected = select_finalists(screen["rows"], 6)
    rows = []
    for rank, cand in enumerate(selected, 1):
        for arm, D, h_scale, x_dep in ARMS:
            rows.append(dict(
                index=len(rows), finalist_rank=rank, candidate_id=cand["candidate_id"],
                candidate_run_id=cand["run_id"], arm=arm, tau_ms=float(cand["tau_ms"]),
                theta=float(cand["theta"]), k=float(cand["k"]), rho=float(cand["rho"]),
                k_ratio=float(cand["k_ratio"]), rho_fraction=float(cand["rho_fraction"]),
                D=float(D), h_init_scale=float(h_scale), x_depletion=float(x_dep),
                x_availability=float(1.0 - x_dep), T_ms=_duration_ms(cand["tau_ms"], arm),
                connection_seed=1, noise_seed=401, no_kick=True, M=False, coop_A=0.0,
            ))
    payload = dict(stage="E4_FROZEN_FORKS", status="LOCKED", code_head=_git_head(),
                   screen_code_head=_load_json(os.path.join(OUT, "h_loop_screen_manifest.json"))["code_head"],
                   selected_finalists=selected, n_finalists=len(selected), rows=rows,
                   n_rows=len(rows), created=_now())
    FCXR._write_json(os.path.join(OUT, "frozen_fork_manifest.json"), payload)
    print(json.dumps(dict(status="LOCKED", n_finalists=len(selected), n_rows=len(rows),
                          finalists=[r["run_id"] for r in selected], code_head=payload["code_head"]), indent=2))


_SUBSTRATE = None
_P_PACK = None


def _substrate_and_field():
    global _SUBSTRATE, _P_PACK
    if _SUBSTRATE is None:
        _SUBSTRATE = PP.build_substrate(1)
        _P_PACK = load_onset_depletion_pi(P_FIELD)
        assert_field_substrate_aligned(_P_PACK, _SUBSTRATE)
    return _SUBSTRATE, _P_PACK


def _region_masks(S):
    pos = np.asarray(S["posE"], float)
    src = np.asarray(S["src_xy"], float)
    axis = np.asarray(S["axis_unit"], float)
    core = np.linalg.norm(pos - src, axis=1) <= PP.CORE_R
    rel = pos - src
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    axis_band = (perp <= PP.CORE_R) & (~core)
    return core, axis_band, (~core & ~axis_band)


def _window_metrics(rate, spk, h, theta, tau_ref, dt, window_ms):
    n = min(rate.size, max(1, int(round(float(window_ms) / dt))))
    rw = np.asarray(rate[-n:], float)
    sw = np.asarray(spk[-n:], bool)
    hw = np.asarray(h[-n:], float)
    per_cell = sw.sum(axis=0) / (n * dt * 1e-3)
    ceiling_hz = 0.8 * 1000.0 / float(tau_ref)
    x = np.arange(hw.size, dtype=float) * dt
    slope = float(np.polyfit(x, hw, 1)[0] * 1000.0) if hw.size >= 2 else float("nan")
    rate_mean = float(np.mean(rw))
    high_occ = float(np.mean(rw >= 20.0))
    h_mean = float(np.mean(hw))
    return dict(
        window_ms=float(n * dt), rate_mean_hz=rate_mean, rate_sd_hz=float(np.std(rw)),
        rate_high_occupancy=high_occ, h_mean=h_mean, h_slope_per_s=slope,
        ceiling_fraction=float(np.mean(per_cell >= ceiling_hz)), ceiling_hz=ceiling_hz,
        high_like=bool(rate_mean >= 20.0 and high_occ >= 0.25 and h_mean >= float(theta)
                       and slope >= -0.05 * max(h_mean, float(theta))),
        low_like=bool(rate_mean < 20.0 and high_occ < 0.20 and h_mean < float(theta)),
    )


def _isi_cv_sample(spk, dt, n_sample=256):
    idx = np.linspace(0, spk.shape[1] - 1, min(n_sample, spk.shape[1]), dtype=int)
    cvs = []
    for j in idx:
        t = np.flatnonzero(spk[:, j]) * dt
        if t.size >= 4:
            isi = np.diff(t)
            m = float(np.mean(isi))
            if m > 0:
                cvs.append(float(np.std(isi) / m))
    return float(np.median(cvs)) if cvs else float("nan")


def _pairwise_corr(spk, dt, n_sample=64, bin_ms=10.0):
    idx = np.linspace(0, spk.shape[1] - 1, min(n_sample, spk.shape[1]), dtype=int)
    w = max(1, int(round(bin_ms / dt)))
    n = (spk.shape[0] // w) * w
    if n == 0 or idx.size < 2:
        return float("nan")
    b = spk[:n, idx].reshape(-1, w, idx.size).sum(axis=1).astype(float)
    good = np.std(b, axis=0) > 0
    if np.sum(good) < 2:
        return float("nan")
    c = np.corrcoef(b[:, good], rowvar=False)
    return float(np.mean(c[np.triu_indices_from(c, 1)]))


def _region_rate(spk, mask, dt):
    if not np.any(mask):
        return float("nan")
    return float(spk[:, mask].sum() / (spk.shape[0] * np.sum(mask) * dt * 1e-3))


def _fork_path(row):
    return os.path.join(OUT, "frozen_fork_cells",
                        f"{row['candidate_run_id']}__{row['arm']}__n{row['noise_seed']}.json")


def _run_row(row):
    S, pk = _substrate_and_field()
    theta = float(row["theta"])
    xf = np.full(S["NE"], float(row["x_availability"]), float)
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                       fail_on_clip=False, rec_sat_g=21.6, use_x=True)
    cfg.update(
        coop_A=0.0, use_m=False, z_frozen_E=frozen_z_field(pk["p_i"], float(row["D"])),
        x_relay_frozen_E=xf, use_h_lc2=True, tau_h_lc2=float(row["tau_ms"]),
        theta_h_lc2=theta, k_h_lc2=float(row["k"]), rho_h_lc2=float(row["rho"]),
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
    spk = np.asarray(res["E_spk_bool"], bool)
    h = np.asarray(slow.trace_h_lc2_mean, float)
    state = _window_metrics(rate, spk, h, theta, p.tau_ref_E, DT, 1000.0)
    required_low_ms = max(1000.0, 3.0 * float(row["tau_ms"]))
    post = _window_metrics(rate, spk, h, theta, p.tau_ref_E, DT, required_low_ms)
    core, axis, off = _region_masks(S)
    stride = max(1, int(round(10.0 / DT)))
    finite = bool(np.all(np.isfinite(rate)) and np.all(np.isfinite(h)))
    clip = float(np.max(slow.trace_conductance_clip_frac))
    numerical = bool((not finite) or clip > 0.0 or res["runaway_early_stop_ms"] is not None)
    gA = np.asarray(slow.trace_gA_raw_lc2_mean, float)
    if gA.size == rate.size and np.var(gA) > 0:
        gain_proxy = float(np.cov(gA, rate, ddof=0)[0, 1] / np.var(gA))
    else:
        gain_proxy = float("nan")
    return dict(
        **row, finite=finite, numerical_failure=numerical, clip_frac_max=clip,
        runaway_early_stop_ms=res["runaway_early_stop_ms"],
        tau_eff_ratio_min=float(np.min(slow.trace_tau_eff_ratio_min)), state_tail_1s=state,
        post_offset_required_ms=required_low_ms, state_required_low_window=post,
        mean_rate_hz=float(np.mean(rate)), max_rate_hz=float(np.max(rate)),
        h_end=float(h[-1]), h_max=float(np.max(slow.trace_h_lc2_max)),
        x_relay_mean=float(np.mean(slow.x_relay)), y_mean=float(np.mean(slow.y)),
        isi_cv_median_sample=_isi_cv_sample(spk, DT), pairwise_corr_sample=_pairwise_corr(spk, DT),
        local_input_output_gain_proxy=gain_proxy,
        regional_rate_hz=dict(core=_region_rate(spk, core, DT), axis=_region_rate(spk, axis, DT),
                              off_axis=_region_rate(spk, off, DT)),
        wall_s=round(wall, 2), peak_rss_gib=round(_rss_gib(), 3), trace_dt_ms=10.0,
        rate_trace=rate[::stride].tolist(), h_trace=h[::stride].tolist(),
        x_trace=np.asarray(slow.trace_x_relay_mean, float)[::stride].tolist(),
        gA_trace=gA[::stride].tolist(),
        gH_trace=np.asarray(slow.trace_gH_lc2_mean, float)[::stride].tolist(), finished=_now(),
    )


def _worker(row):
    path = _fork_path(row)
    if os.path.isfile(path):
        return _load_json(path)
    out = _run_row(row)
    FCXR._write_json(path, out)
    return out


def classify_candidate(arms):
    by = {r["arm"]: r for r in arms}
    safe = all(not r["numerical_failure"] for r in arms)
    low_ok = safe and all(by[a]["state_tail_1s"]["low_like"] for a in ("A_low", "A_high", "B"))
    high_ok = safe and by["C"]["state_tail_1s"]["high_like"]
    x_low = [a for a in ("D1", "D2") if by[a]["state_required_low_window"]["low_like"]]
    if not safe:
        label = "numerical_failure"
    elif not by["A_high"]["state_tail_1s"]["low_like"]:
        label = "healthy_false_high_basin"
    elif not by["B"]["state_tail_1s"]["low_like"]:
        label = "susceptible_low_basin_missing"
    elif not high_ok:
        label = "no_finite_high_basin"
    elif not x_low:
        label = "x_load_does_not_remove_high"
    else:
        label = "H_BASIN_CANDIDATE"
    return dict(label=label, safe=safe, matched_low_states=low_ok, finite_high_state=high_ok,
                x_return_arms=x_low)


def _aggregate(rows):
    candidates = []
    for cid in sorted({r["candidate_run_id"] for r in rows}):
        arms = [r for r in rows if r["candidate_run_id"] == cid]
        if len(arms) != len(ARMS):
            continue
        verdict = classify_candidate(arms)
        candidates.append(dict(candidate_run_id=cid, **verdict))
    return candidates


def cmd_one(args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    m = _load_json(os.path.join(OUT, "frozen_fork_manifest.json"))
    row = m["rows"][int(args.index)]
    os.makedirs(os.path.join(OUT, "frozen_fork_cells"), exist_ok=True)
    print(json.dumps(_worker(row), indent=2))


def cmd_all(args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    m = _load_json(os.path.join(OUT, "frozen_fork_manifest.json"))
    rows = m["rows"]
    os.makedirs(os.path.join(OUT, "frozen_fork_cells"), exist_ok=True)
    before = _meminfo()
    if before["MemAvailable_GiB"] < 96.0:
        raise SystemExit(f"OOM safety stop: {before}")
    running = os.path.join(OUT, "E4_RUNNING.json")
    FCXR._write_json(running, dict(stage="E4", pid=os.getpid(), workers=args.workers,
                                  resource_before=before, started=_now()))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            fut = {ex.submit(_worker, row): row for row in rows}
            for j, f in enumerate(as_completed(fut), 1):
                out = f.result()
                results.append(out)
                print(f"[E4] {j}/{len(rows)} {out['candidate_run_id']} {out['arm']} "
                      f"rate={out['state_tail_1s']['rate_mean_hz']:.1f} "
                      f"high={out['state_tail_1s']['high_like']} low={out['state_tail_1s']['low_like']} "
                      f"RSS={out['peak_rss_gib']:.2f}GiB", flush=True)
                now = _meminfo()
                if now["SwapUsed_MiB"] - before["SwapUsed_MiB"] >= 512.0:
                    raise MemoryError(f"swap hard stop: before={before}, now={now}")
        results.sort(key=lambda r: int(r["index"]))
        verdicts = _aggregate(results)
        payload = dict(stage="E4", status="COMPLETE", n_rows=len(results), rows=results,
                       candidate_verdicts=verdicts, resource_before=before,
                       resource_after=_meminfo(), finished=_now())
        FCXR._write_json(os.path.join(OUT, "frozen_fork_map.json"), payload)
        FCXR._write_json(os.path.join(OUT, "E4_DONE.json"),
                         dict(stage="E4", status="COMPLETE", candidate_verdicts=verdicts,
                              finished=_now()))
        if os.path.exists(running):
            os.remove(running)
    except Exception as exc:
        FCXR._write_json(os.path.join(OUT, "E4_FAILED.json"),
                         dict(stage="E4", error=repr(exc), traceback=traceback.format_exc(), failed=_now()))
        raise


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    one = sub.add_parser("one")
    one.add_argument("--index", type=int, required=True)
    one.add_argument("--confirm-run", action="store_true")
    allp = sub.add_parser("all")
    allp.add_argument("--workers", type=int, choices=(1, 2), default=1)
    allp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "manifest":
        cmd_manifest(args)
    elif args.cmd == "one":
        cmd_one(args)
    else:
        cmd_all(args)


if __name__ == "__main__":
    main()
