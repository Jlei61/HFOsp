"""Stage-4 v2 spontaneous big-focus working-point search.

Screens the Stage-4 `extended_patch` substrate (ONE large excitable disk, spontaneous / no kick)
with the M3A-v2.2 slow variables coupled -- q_I (slow, across-event depletion) + g_K (fast,
per-event fatigue brake, `eta_K>0`) -- looking for a working point where a TRAIN of discrete,
self-terminating events slowly depletes q_I until the sheet tips into runaway (`train_then_runaway`).

Two-stage CLI (cost control; each full-T runaway sim ~= 1 hr):
    python scripts/run_stage4_v2_workpoint_search.py                       # --stage fast (default)
    python scripts/run_stage4_v2_workpoint_search.py --stage confirm --survivor-json <screen_fast.json>

The fast stage (short T + early-abort) is the ONLY thing that runs without an explicit user go; it
writes survivors + a wall-clock estimate and EXITS. The confirm stage is a SEPARATE invocation.

This module owns the pure verdict classifier (Task 2) and the screener/CLI (Task 3). The SNN-heavy
imports (`plot_fig_m3a_v2_2_hG_runaway_transition_gif`, `run_sef_hfo_snn_cm_spontaneous_readout`) are
deferred into `run_one`/`main` so the classifier stays import-cheap and unit-testable without the engine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENG = ROOT / "src" / "snn_engine"
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "paper_figures"), str(ENG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def classify_workpoint(event_ons, runaway_ms, aborted_ms, T, *, min_train=3, early_ms=200.0):
    """Verdict for a spontaneous big-focus run. A working point = a TRAIN of >= min_train discrete
    events with a DELAYED runaway (not an immediate all-or-nothing burst). An abort counts AS a
    runaway (it only fires on the shared sustained-runaway criterion); `aborted_ms` is the DETECTION
    time (inflated ~100 ms over onset by the sustained-window rule), so a lone ignition event that
    itself becomes the runaway shows up as n_pre==1 -- that is still a burst, not a train."""
    eff = runaway_ms if runaway_ms is not None else aborted_ms
    end = eff if eff is not None else T
    n_pre = sum(1 for t in event_ons if t < (end - 20.0))
    if eff is None:
        return "train_no_runaway" if n_pre >= min_train else "silent"
    if n_pre >= min_train:
        return "train_then_runaway"
    if eff <= early_ms or n_pre <= 1:
        return "one_shot_burst"
    return "few_events_then_runaway"                 # 2 .. min_train-1 events then runaway (near-miss)


def is_working_point(verdict):
    return verdict == "train_then_runaway"


# ---- Task 3: screener + two-stage CLI (SNN-heavy; H/C imported lazily) ----

FAST_T = 900.0
FULL_T = 2500.0

# g_K discretization bet: eta_K>0 couples fatigue as a FAST per-event brake (short tau_K) so each
# spontaneous nucleation self-terminates, while q_I (slow tau_q) does the across-event buildup.
GRID = [dict(core_mean=cm, eta_K=ek, tau_K=tk)
        for cm in (16.5, 17.0) for ek in (0.3, 0.5, 0.8) for tk in (150.0, 400.0)]
BASE = dict(L=20.0, core_std=1.5, core_r=6.0, drive=0.6, k_q=0.25, tau_q=5000.0,
            sigma_q=1.5, k_K=1.5, sigma_K=0.5, seed=1)
OUT = ROOT / "results" / "topic4_sef_hfo" / "stage4_v2_workpoint"


def run_one(*, L, core_mean, core_std, core_r, drive, k_q, tau_q, sigma_q,
            eta_K, k_K, tau_K, sigma_K, T, seed, abort=True):
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False,
                           eta_K=eta_K, k_K=k_K, tau_K=tau_K, sigma_K=sigma_K,
                           k_q=k_q, tau_q=tau_q, sigma_q=sigma_q, q_min=0.05,
                           core_mean=core_mean, core_std=core_std, core_radius=core_r,
                           drive=drive, L=L, T=T, n_pulses=0, seed=seed)
    S = H._build(cfg)
    DT = float(S["p"].dt)
    assert abs(DT - C.DT) < 1e-12                          # timestep LOCKED (companion has no H.DT)
    t0 = time.time()
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], abort_on_runaway=abort)
    spk = res["E_spk_bool"]
    rate_hz = np.asarray(res["rate_E"], float)
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    event_ons = [float(e["t_on"]) for e in C.detect_events(af, bin_w, event_on_frac=bar)]
    runaway = H._first_sustained(H._smooth_rate(rate_hz, DT, 20.0), DT, 120.0, 100.0)   # shared criterion
    aborted = res.get("aborted_ms")
    eff = runaway if runaway is not None else aborted
    verdict = classify_workpoint(event_ons, runaway, aborted, cfg.T)
    return dict(L=L, core_mean=core_mean, core_r=core_r, drive=drive, k_q=k_q, tau_q=tau_q,
                eta_K=eta_K, k_K=k_K, tau_K=tau_K, seed=seed, T=T,
                n_events=len(event_ons),
                n_pre=sum(1 for t in event_ons if eff is None or t < eff - 20.0),
                runaway_ms=runaway, aborted_ms=aborted, effective_runaway_ms=eff,
                q_min_final=round(float(np.asarray(res["trace_qI_min"]).min()), 4),
                verdict=verdict, wall_s=round(time.time() - t0, 1))


def _stage_fast():
    rows = []
    for gk in GRID:
        r = run_one(**{**BASE, **gk, "T": FAST_T, "abort": True})
        rows.append(r); print("FAST " + json.dumps(r), flush=True)
    survivors = [r for r in rows if r["verdict"] in
                 ("train_then_runaway", "train_no_runaway", "few_events_then_runaway")
                 and (r["aborted_ms"] is None or r["aborted_ms"] > 300.0)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_fast.json").write_text(json.dumps({"base": BASE, "grid": GRID,
        "fast": rows, "survivors": survivors}, indent=2))
    est_hr = round(min(len(survivors), 4) * FULL_T / 900.0 * 0.33, 1)   # rough: ~1 hr per full-T run
    print(f"SURVIVORS {len(survivors)} / {len(rows)}", flush=True)
    print(f"CONFIRM_ESTIMATE up to {min(len(survivors),4)} runs ~ {est_hr} h "
          f"(run: --stage confirm --survivor-json {OUT/'screen_fast.json'})", flush=True)
    return 0 if survivors else 2


def _stage_confirm(survivor_json, max_confirm):
    data = json.loads(Path(survivor_json).read_text())
    survivors = data["survivors"][:max_confirm]
    rows = []
    for r in survivors:
        gk = dict(core_mean=r["core_mean"], eta_K=r["eta_K"], tau_K=r["tau_K"])
        c = run_one(**{**BASE, **gk, "T": FULL_T, "abort": True})
        rows.append(c); print("CONFIRM " + json.dumps(c), flush=True)
    working = [c for c in rows if is_working_point(c["verdict"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_confirm.json").write_text(json.dumps({"base": BASE,
        "confirm": rows, "working_points": working}, indent=2))
    print("WORKING_POINTS " + json.dumps(working), flush=True)
    return 0 if working else 2


def main():
    import argparse
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fast", "confirm"], default="fast")
    ap.add_argument("--survivor-json", default=str(OUT / "screen_fast.json"))
    ap.add_argument("--max-confirm", type=int, default=4)
    a = ap.parse_args()
    os.chdir(ROOT)
    C._engine_guard()
    return _stage_fast() if a.stage == "fast" else _stage_confirm(a.survivor_json, a.max_confirm)


if __name__ == "__main__":
    raise SystemExit(main())
