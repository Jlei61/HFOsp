"""Re-run the fit's best candidate on networks that had no say in choosing it.

This writes every field the artifact carries. An earlier version wrote only part
of them and the rest were patched into the JSON by an inline script that lived
nowhere, so re-running the producer would have silently deleted fields the
figure depends on.

The comparison floor is built to match the scorer's own structure. The scorer
compares a small model sample against the FULL patient training set, so the
floor must do the same: draw the same number of events from the patient's
held-out recordings and score them against that same full training set. Pitting
80 against an independent 80 instead adds sampling noise on the reference side
that the scorer never has, and inflates the floor from 0.18 to 0.25.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import OUT, STAGE2, _evaluate  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import (  # noqa: E402
    axial_map, distance, patient_events, signed_monotonicity)
from src.topic4_core_field_profile import split_by_block  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402

N_BOOTSTRAP = 10000
SPLIT_SEED, HELD_OUT_FRAC = 20260808, 0.3


def structure_matched_floor(model_n, p_train, p_test, seed=0, n_boot=N_BOOTSTRAP):
    """What the scorer reads when the model is replaced by the patient herself.

    Same asymmetry as the real comparison: `model_n` events on one side, the
    entire training set on the other.
    """
    rng = np.random.default_rng(seed)
    d = [distance(rng.choice(p_test, model_n, replace=False), p_train)
         for _ in range(n_boot)]
    return dict(model_n=int(model_n), n_bootstrap=int(n_boot),
                median=float(np.median(d)), p05=float(np.percentile(d, 5)),
                p95=float(np.percentile(d, 95)),
                structure="model_n events vs the full patient training set",
                note=("scoring the same model_n against an independent model_n "
                      "instead would add reference-side sampling noise the "
                      "scorer never has and inflate this floor"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--restart", type=int, default=0)
    ap.add_argument("--n-confirm", type=int, default=6)
    a = ap.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    ck_path = f"{OUT}/fit/checkpoint_K3_r{a.restart}.json"
    ck = json.load(open(ck_path))
    axial = axial_map()
    vals, blocks = patient_events(axial)
    tr, te = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    p_train, p_test = vals[tr], vals[te]

    hist = ck["history"]
    best = min(hist, key=lambda r: r["distance"])
    fit_seeds = {int(s) for r in hist for s in r["seeds"]}
    confirm = [s for s in range(501, 560) if s not in fit_seeds][:a.n_confirm]
    if set(confirm) & fit_seeds:
        raise SystemExit("confirmation seeds must be independent of the fit")

    print(f"fit used networks {sorted(fit_seeds)}")
    print(f"confirming on {confirm}\nbest candidate scored {best['distance']:.3f} "
          f"during the fit on {best['n_events']} events", flush=True)

    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(_evaluate, [(best["theta"], s, cfg,
                                    os.path.join(STAGE2, "network_cache"))
                                   for s in confirm])
    vv = [x for r in res if "error" not in r for e in r["events"]
          if (x := signed_monotonicity(e.get("ranks"), axial)) is not None]

    d_tr = distance(vv, p_train) if len(vv) >= 10 else None
    d_te = distance(vv, p_test) if len(vv) >= 10 else None
    floor = structure_matched_floor(len(vv), p_train, p_test)

    out = dict(
        best_theta=best["theta"], fit_value=best["distance"],
        fit_n_events=best["n_events"], confirm_seeds=confirm,
        confirm_n_events=len(vv), confirm_distance_train=d_tr,
        confirm_distance_heldout=d_te,
        n_errors=sum(1 for r in res if "error" in r),
        floor_structure_matched=floor,
        patient_train_vs_heldout_full=float(distance(p_train, p_test)),
        winners_curse=dict(
            fit_minus_confirmed=(None if d_tr is None
                                 else float(best["distance"] - d_tr)),
            retracted=("not a selection-bias estimate: the event count also "
                       "changed, from the fit's per-candidate count to this "
                       "one, and a total-variation distance between histograms "
                       "is biased upward at small samples, so selection and "
                       "sample size move together")),
        objective_actually_used=(
            "one-dimensional binned total variation over sign(slope)*r2, NOT "
            "the two-dimensional energy distance frozen in spec 9.3; the "
            "marginal is satisfiable by a single mid-array generator"),
        reference=ck.get("reference"), provenance=provenance())
    atomic_write_json(out, f"{OUT}/fit/confirmation_K3_r{a.restart}.json")

    print(f"\nconfirmed on {len(vv)} events from {len(confirm)} independent networks")
    print(f"  vs patient training recordings  {d_tr:.3f}")
    print(f"  vs patient held-out recordings  {d_te:.3f}")
    print(f"  structure-matched floor          {floor['median']:.3f} "
          f"[{floor['p05']:.3f}-{floor['p95']:.3f}]")
    print(f"  patient train vs held-out        {out['patient_train_vs_heldout_full']:.3f}")


if __name__ == "__main__":
    main()
