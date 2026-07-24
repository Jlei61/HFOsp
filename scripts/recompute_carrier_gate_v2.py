#!/usr/bin/env python
"""Recompute the carrier gate with carrier_gate_v2 (faithful onset/baseline + B2/A7/A8 fixes) OFFLINE from
the saved seed-1 NPZ -- no SNN re-run. Reports v1-vs-v2 verdicts + a baseline-sensitivity sweep for the
observed occupancy (the review's key robustness check). Writes carrier_gate_v2_seed{seed}.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_carrier_gate_v2 import (  # noqa: E402
    compute_source_gate_v2, compute_observed_gate_v2, carrier_verdict_v2, OBS_BASELINE_MS)

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "topic4_sef_hfo", "zm_ictal_carrier_gate")


def _rd(d, k):
    return np.asarray(d[k])


def recompute(arm, seed=1):
    npz = np.load(os.path.join(DIR, f"{arm}_seed{seed}.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(DIR, f"{arm}_seed{seed}.json")))
    bin_ms = float(npz["rate_bin_ms"])
    runaway = meta.get("runaway_early_stop_ms")
    src = compute_source_gate_v2(_rd(npz, "core_rate"), _rd(npz, "active_frac"),
                                 _rd(npz, "kymo_axis"), _rd(npz, "kymo_t_ms"), bin_ms, runaway)
    obs = compute_observed_gate_v2(_rd(npz, "lfp"), float(npz["lfp_fs"]))
    label, detail = carrier_verdict_v2(src, obs)

    # baseline sensitivity for the observed occupancy (the review's check). Report the ACTIVE contacts
    # (top-4 by peak low-gamma dB) -- a quiet flat contact trivially has occupancy 1.0 and must not count.
    sweep = {}
    for bl in (150.0, 300.0, 500.0, 800.0):
        o = compute_observed_gate_v2(_rd(npz, "lfp"), float(npz["lfp_fs"]), baseline_ms=bl)
        peaks = np.array([c["peak_lowgamma_db"] for c in o["contacts"]])
        active = np.argsort(peaks)[-4:]                    # 4 most-enhanced (sink) contacts
        occ = [o["contacts"][i]["macro"]["occupancy"] for i in active]
        sweep[int(bl)] = dict(n_sustained=o["n_sustained_contacts"],
                              active_max_occupancy=round(float(max(occ)), 3),
                              active_median_occupancy=round(float(np.median(occ)), 3),
                              active_peak_db=round(float(peaks.max()), 1))
    return dict(
        arm=arm, v1_verdict=meta["ictal_carrier_verdict"], v2_verdict=label, v2_detail=detail,
        v1_source_onset_ms=meta["source_metrics"]["onset_ms"], v2_source_onset_ms=src["onset_ms"],
        v2_source_macro=dict(duration_ms=round(src["macro"]["duration_ms"], 1),
                             occupancy=round(src["macro"]["occupancy"], 3), sustained=src["macro"]["sustained"],
                             baseline=round(src["macro"]["baseline"], 3)),
        v2_obs=dict(n_sustained_contacts=obs["n_sustained_contacts"], highfreq_enhanced=obs["highfreq_enhanced"],
                    best_occupancy=round(obs["best_macro"]["occupancy"], 3),
                    contact_peak_lowgamma_db=obs["contact_peak_lowgamma_db"]),
        observed_baseline_sensitivity=sweep)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = {"note": "carrier_gate_v2 offline recompute (faithful onset/baseline + B2/A7/A8); v1 kept for history",
           "obs_baseline_ms_default": OBS_BASELINE_MS, "arms": {}}
    for arm in ("interictal_ctrl", "bare", "sg"):
        r = recompute(arm, seed)
        out["arms"][arm] = r
        s = r["observed_baseline_sensitivity"]
        print(f"[{arm}] v1={r['v1_verdict']} -> v2={r['v2_verdict']} | v1_onset={r['v1_source_onset_ms']} "
              f"v2_onset={r['v2_source_onset_ms']} src_sustained={r['v2_source_macro']['sustained']} "
              f"src_occ={r['v2_source_macro']['occupancy']} | obs n_sust={r['v2_obs']['n_sustained_contacts']} "
              f"best_occ={r['v2_obs']['best_occupancy']}")
        print(f"    obs baseline sweep (n_sustained / active-contact max occupancy): "
              + " ".join(f"{k}ms:{v['n_sustained']}/{v['active_max_occupancy']}" for k, v in s.items()))
    p = os.path.join(DIR, f"carrier_gate_v2_seed{seed}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()
