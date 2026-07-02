"""Feasibility pilot: Stage-4 big-focus SPONTANEOUS substrate + M3A-v2.2 q_I slow field.

Question (go/no-go before building the full figure): does the single large excitable disk
(`extended_patch`), driven ONLY by background noise (KICK_BOOST=0, no scheduled kicks), emit a
TRAIN of discrete interictal-like events that -- with the dynamic q_I inhibitory-resource field
depleting across those events -- builds up to a runaway? Or does it (a) one-shot burst immediately,
or (b) never run away in the window?

This reuses the spontaneous runner's extended_patch build (`C.build_lesion_vth`) and the engine's
`simulate_kick` slow-slot with a dynamic `SpatialSlowField` (the runner's own `--slow-var` path uses
a FROZEN field whose step() is a no-op, so it cannot accumulate across events). No figure/GIF -- it
prints one JSON line per screened config so we can pick a train->runaway regime.

Run:  python scripts/pilot_stage4_spontaneous_qI.py
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
for _p in (str(ROOT), str(ROOT / "scripts"), str(ENG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_sef_hfo_snn_cm_spontaneous_readout as C  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from params import Params  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402

DT = C.DT


def _smooth(rate, dt, win_ms=20.0):
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(np.asarray(rate, float), np.ones(n) / n, mode="same")


def _first_sustained(rate, dt, threshold_hz=120.0, dur_ms=100.0):
    above = np.asarray(rate) >= threshold_hz
    n = max(1, int(round(dur_ms / dt)))
    if above.size < n:
        return None
    c = np.convolve(above.astype(float), np.ones(n), mode="valid")
    idx = np.flatnonzero(c >= 0.80 * n)
    return None if idx.size == 0 else round(float(idx[0] * dt), 1)


def run_one(*, L, density, core_r, core_mean, core_std, drive, AR, T, seed,
            k_q, q_min, tau_q, sigma_q, use_gK, k_K, eta_K, sigma_K):
    theta_rad = 0.0
    axis_unit = np.array([1.0, 0.0])
    p = Params(g=3.6, L=L, density=density, T=T, dt=DT, nu_ext_ratio=drive, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    center = np.array([L / 2.0, L / 2.0])
    half = L / 2.0
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=AR, verbose=False)
    # ONE large isotropic excitable disk at the sheet centre = extended_patch (elongation=1.0). Built
    # directly (the runner's build_lesion_vth extended_patch branch passes elongation/axis_unit, which
    # the current sample_core_field signature does not accept -- a pre-existing runner/engine drift).
    is_E = np.zeros(NE + NI, bool)
    is_E[:NE] = True
    cf = sample_core_field(net["pos"], is_E, center, core_r, np.random.default_rng(seed + 7),
                           core_mean=core_mean, core_std=core_std, base_mean=18.0)
    vth = cf["vth"]
    core_mask = cf["core_mask"]
    posE = net["pos"][:NE]
    posI = net["pos"][NE:]
    scfg = SpatialSlowFieldConfig(use_qI=True, use_gK=use_gK, k_q=k_q, k_K=(k_K if use_gK else 0.0),
                                  sigma_q=sigma_q, sigma_K=sigma_K, eta_K=eta_K, q_min=q_min,
                                  q_init=1.0, tau_q=tau_q, tau_a=20.0)
    slow = SpatialSlowField(NE + NI, 18.0, posE, posI, L, cfg=scfg)
    net["rng"] = np.random.default_rng(seed)
    t0 = time.time()
    res = simulate_kick(p, net, 0.0, slow=slow, kick_center=list(center), r_kick=core_r,
                        t_kick=1e9, V_th_per_neuron=vth, lfp_recorder=None)
    spk = res["E_spk_bool"]
    rate = np.asarray(res["rate_E"], float)

    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max())
    bar = floor + C.CAL_FRAC * (peak - floor)
    events = C.detect_events(af, bin_w, event_on_frac=bar)
    rate_s = _smooth(rate, DT)
    runaway = _first_sustained(rate_s, DT)
    n_pre = sum(1 for e in events if runaway is None or e["t_on"] < runaway - 20.0)

    if runaway is None:
        verdict = "no_runaway"
    elif n_pre >= 2 and runaway > 200.0:
        verdict = "train_then_runaway"
    elif runaway <= 200.0 or n_pre == 0:
        verdict = "one_shot_burst"
    else:
        verdict = "few_events_then_runaway"

    return dict(
        core_mean=core_mean, k_q=k_q, drive=drive, L=L, core_r=core_r,
        n_events=len(events), n_pre_runaway=int(n_pre), runaway_ms=runaway,
        max_rate_hz=round(float(rate_s.max()), 1),
        q_mean_final=round(float(slow.q_I.mean()), 4), q_min_final=round(float(slow.q_I.min()), 4),
        tonic=round(float((af > bar).mean()), 4), verdict=verdict, wall_s=round(time.time() - t0, 1),
    )


def main():
    os.chdir(ROOT)
    C._engine_guard()
    grid = [dict(core_mean=cm, k_q=kq) for cm in (16.5, 17.0, 17.5) for kq in (0.18, 0.35)]
    base = dict(L=20.0, density=100.0, core_r=6.0, core_std=1.5, drive=0.6, AR=2.0, T=2500.0, seed=1,
                q_min=0.05, tau_q=5000.0, sigma_q=1.5, use_gK=False, k_K=1.0, eta_K=0.0, sigma_K=0.5)
    rows = []
    for g in grid:
        row = run_one(**{**base, **g})
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = ROOT / "results" / "topic4_sef_hfo" / "stage4_spontaneous_qI_pilot"
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen.json").write_text(json.dumps({"base": base, "rows": rows}, indent=2))
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print("SUMMARY " + json.dumps(verdicts), flush=True)
    print(f"wrote {out / 'screen.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
