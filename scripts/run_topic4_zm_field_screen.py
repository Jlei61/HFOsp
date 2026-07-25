#!/usr/bin/env python
"""Reduced 2-D S_L(x)+S_G field screen: Phase 0 (mean-field gate) -> Phase A (write-once lock) -> Phase B
(FLOQUET FIRST for every level; nonlinear runs ONLY for candidate windows) -> adjudicate. Per-arm resume.
Spec 2026-07-24 rev3. Reduced rate field only. --confirm-run required."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.topic4_zm_field_meanfield import meanfield_continuation, simulate_meanfield, MFParams, detect_orbit
from src.topic4_zm_field_screen import (FieldParams, simulate_field, field_metrics, uniform_orbit,
                                        floquet_map, orbit_phasepoint_state, add_r_perturbation,
                                        resolve_w_frac, ARMS)
from src.topic4_zm_field_verdict import adjudicate_field_screen

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_field_screen")
RUNS, TRACES = os.path.join(OUT, "runs"), os.path.join(OUT, "traces")
SEEDS = (0, 1, 2, 3); DT = 0.25; N = 32; EPS = 1e-4; REC_MS = 5.0

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def _git(*a):
    return subprocess.check_output(["git", "-C", _ROOT, *a], text=True).strip()

def phase0():
    r = meanfield_continuation()
    if not r["has_orbit"]:
        print("[PHASE0] no contiguous orbit segment in the grid -> STOP (field NOT built)."); sys.exit(2)
    op, seg = r["operating_point"], r["segment"]
    # dt/2 classification stability of the operating point
    o2 = detect_orbit(simulate_meanfield(MFParams(op["W0"], op["alpha"], op["beta"], op["theta"], op["I0"]),
                                         dt=DT / 2), DT / 2)
    if not o2["oscillates"]:
        print("[PHASE0] operating point not stable under dt/2 -> STOP."); sys.exit(2)
    print(f"[PHASE0] op={op} segment={seg['I0_lo']}..{seg['I0_hi']} interior={seg['interior_I0s']}")
    return op, seg

def phaseA_lock(op, seg):
    path = os.path.join(OUT, "phaseA_lock.json")
    if os.path.exists(path):
        # WRITE-ONCE and FAIL-CLOSED: reuse an existing lock ONLY if it still describes the same contract.
        # Reusing a lock whose spec, operating point, grid or dt no longer match the current code silently
        # pairs old pre-registration with a new implementation (review finding).
        old = json.load(open(path))
        cur = dict(spec_sha=_sha(os.path.join(_ROOT, "docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md")),
                   operating_point=op, dt=DT, grid_n=N, seeds=list(SEEDS))
        mismatch = {k: (old.get(k), v) for k, v in cur.items() if old.get(k) != v}
        if mismatch:
            print(f"[PHASE A] REFUSING to reuse {path}: contract changed -> " +
                  "; ".join(f"{k}: locked={o!r} now={n!r}" for k, (o, n) in mismatch.items()))
            print("[PHASE A] delete the lock explicitly (and archive the old results) to re-lock.")
            sys.exit(3)
        print(f"[PHASE A] lock exists and its contract still matches; reusing {path}.")
        return old
    os.makedirs(OUT, exist_ok=True)
    interior = [float(x) for x in seg["interior_I0s"]]
    op_I0 = float(op["I0"])
    if len(interior) >= 5:
        idx = np.linspace(0, len(interior) - 1, 5).round().astype(int)
        levels = sorted({interior[i] for i in idx.tolist()})
        if not any(abs(x - op_I0) < 1e-9 for x in levels):
            # round-half-to-even can drop the very point Phase 0 validated at dt and dt/2; swap the
            # nearest level for the operating point so the best-validated point is actually measured
            j = min(range(len(levels)), key=lambda k: abs(levels[k] - op_I0))
            levels[j] = op_I0
            levels = sorted(set(levels))
    else:
        levels = interior
    p0 = FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=op["I0"], n=N)
    lock = dict(spec_sha=_sha(os.path.join(_ROOT, "docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md")),
                operating_point=op, segment=seg, I0_levels=levels, seeds=list(SEEDS), dt=DT, grid_n=N,
                w_frac_derived=resolve_w_frac(p0), eps_perturb=EPS,
                git_sha=_git("rev-parse", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(lock, open(path, "w"), indent=2)
    print(f"[PHASE A] wrote {path} levels={levels}")
    return lock

def _params(lock, I0, n=None):
    op = lock["operating_point"]
    return FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=I0,
                       n=n or lock["grid_n"])

def _run_path(level, arm, seed, tag, T, dt, n):
    """Cache key MUST encode every parameter that changes the trajectory, else a short smoke run silently
    resumes as if it were a full-length production run (review finding)."""
    return os.path.join(RUNS, f"{tag}_L{level}_{arm}_s{seed}_T{int(round(T))}_dt{dt:g}_n{n}.json")

def run_formation_arm(lock, I0, arm, seed, T, dt=None, n=None, tag="form"):
    """Aligned-orbit start + fixed 1e-4 zero-mean r-perturbation -> does a staggered state FORM?"""
    dt = dt or lock["dt"]; level = f"{I0:.4f}"
    p = _params(lock, I0, n)
    path = _run_path(level, arm, seed, tag, T, dt, p.n)
    if os.path.exists(path):
        return json.load(open(path))                 # resume
    orbit, per = uniform_orbit(p, dt)
    st = add_r_perturbation(orbit_phasepoint_state(p, orbit, len(orbit) // 3), lock["eps_perturb"], seed, p.n)
    out = simulate_field(p, arm, T=T, dt=dt, seed=seed, state_init=st, record_stride=int(round(REC_MS / dt)))
    m = field_metrics(out["r_trace"], REC_MS)
    os.makedirs(RUNS, exist_ok=True); os.makedirs(TRACES, exist_ok=True)
    tp = os.path.join(TRACES, f"{tag}_L{level}_{arm}_s{seed}_T{int(round(T))}_dt{dt:g}_n{p.n}.npz")
    np.savez_compressed(tp, r_trace=out["r_trace"][::4], t_ms=out["t_ms"][::4])   # downsampled for figures
    rec = dict(level=level, arm=arm, seed=seed, tag=tag, T=T, dt=dt, n=p.n, metrics=m, period_ms=per,
               trace=os.path.basename(tp), final_state={k: (v.tolist() if hasattr(v, "tolist") else v)
                                                        for k, v in out["final_state"].items()})
    json.dump(rec, open(path, "w"), indent=2)
    return rec

def run_phase_reset_arm(lock, I0, arm, seed, T, formed_rec):
    """NOT IMPLEMENTED (spec §8(i) criterion 5, phase-reset return). Reaching this requires a candidate
    window, which the current screen does not produce. It must reset r/muL/SL/muG/SG from an ALREADY-FORMED
    staggered state -- NOT re-run formation, which is what a previous draft silently did."""
    raise NotImplementedError(
        "phase-reset return (spec §8(i) criterion 5) is not automated; a GO verdict requires it")


def run_long_confirm(lock, I0, arm, seed):
    """NOT IMPLEMENTED (spec §8(i) criterion 6, 60 s confirmation)."""
    raise NotImplementedError("60 s confirmation (spec §8(i) criterion 6) is not automated")


def run_resolution_confirm(lock, I0, arm, seed):
    """NOT IMPLEMENTED (spec §8(i) criterion 6, n=64 resolution confirmation)."""
    raise NotImplementedError("n=64 confirmation (spec §8(i) criterion 6) is not automated")


def run_dt_confirm(lock, I0, arm, seed):
    """NOT IMPLEMENTED (spec §8(i) criterion 6, halved-timestep confirmation)."""
    raise NotImplementedError("dt/2 confirmation (spec §8(i) criterion 6) is not automated")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("refusing to run without --confirm-run")
    op, seg = phase0()
    lock = phaseA_lock(op, seg)
    T = 3000.0 if a.smoke else 30000.0

    # ---- Phase B.1: FLOQUET FIRST (cheap) ----
    fmap = {}
    for I0 in lock["I0_levels"]:
        p = _params(lock, I0); orbit, _ = uniform_orbit(p, lock["dt"])
        fmap[f"{I0:.4f}"] = {arm: floquet_map(p, arm, orbit, lock["dt"], m_max=4)["lam_max"] for arm in ARMS}
        print(f"[floquet I0={I0:.3f}] " + " ".join(f"{k}={v:+.4f}" for k, v in fmap[f'{I0:.4f}'].items()))
    json.dump(fmap, open(os.path.join(OUT, "floquet_map.json"), "w"), indent=2)
    targets = [I0 for I0 in lock["I0_levels"]
               if fmap[f"{I0:.4f}"]["dual_global"] < 0 and
               max(fmap[f"{I0:.4f}"]["dual_local"], fmap[f"{I0:.4f}"]["dual_mixed"]) > 0]
    print(f"[floquet] target-window levels: {targets}")
    if not targets and not a.smoke:
        print("[floquet] no target window -> writing taxonomy verdict WITHOUT the expensive nonlinear sweep.")

    # ---- Phase B.2: nonlinear ONLY for candidate levels (or all, in smoke) ----
    # candidates only; under --smoke run every level so the plumbing is exercised end-to-end
    run_levels = targets if targets else (lock["I0_levels"] if a.smoke else [])
    summary = dict(phaseA=lock, floquet=fmap, levels={})
    for I0 in run_levels:
        key = f"{I0:.4f}"; arms = {}
        for arm in ("dual_global", "dual_local", "dual_mixed"):
            recs = [run_formation_arm(lock, I0, arm, s, T=T) for s in lock["seeds"]]
            arms[arm] = dict(metrics=[r["metrics"] for r in recs], period_ms=recs[0]["period_ms"],
                             lambda_perp_max=fmap[key][arm])
            print(f"  [L{key} {arm}] R={np.median([r['metrics']['median_R_phase'] for r in recs]):.2f} "
                  f"occ={np.median([r['metrics']['occupancy'] for r in recs]):.2f}")
        summary["levels"][key] = dict(arms=arms)
    verdict = adjudicate_field_screen(summary, lock)
    summary["verdict"] = verdict
    json.dump(summary, open(os.path.join(OUT, "field_screen_summary.json"), "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"[VERDICT] {verdict['verdict']} | taxonomy={verdict['taxonomy']} | window={verdict['window']}")

if __name__ == "__main__":
    main()
