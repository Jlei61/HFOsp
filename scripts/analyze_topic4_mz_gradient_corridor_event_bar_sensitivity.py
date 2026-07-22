"""Event-bar threshold sensitivity for MZ gradient-corridor stimulation (reviewer P1).

The interictal-scale event bar (early z-undepleted window) was chosen AFTER seeing the "0 pre-stim
events" artifact, so its influence on the 4/4 dynamics admission must be checked. This recomputes the
per-subject-seed pre-stim recoverable-event count over a grid of (interictal window, CAL_FRAC) and reports
whether the >=3-event eligibility survives, WITHOUT re-running the network.

CAVEAT: the saved baseline active_fraction is downsampled (~3.1 ms bins vs the native 1 ms), so event
counts are APPROXIMATE and slightly under-resolved; this checks robustness of admission, not exact counts.
A definitive control is a slow-off baseline (native scale) — noted as future work.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sef_hfo_events import detect_events  # noqa: E402

RES = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_gradient_corridor_stimulation")
WINDOWS_MS = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
CAL_FRACS = (0.3, 0.5, 0.7)
MIN_EVENTS = 3


def _bar(af, binw, window_ms, cal_frac):
    nb0, nb1 = int(5.0 / binw), int(50.0 / binw)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 and af.size > nb1 else float(af.min())
    ew = int(window_ms / binw)
    early = af[:ew] if af.size > ew else af
    scale = float(early.max())
    if not np.isfinite(scale) or scale <= floor:
        scale = float(af.max())
    return floor + cal_frac * (scale - floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    args = ap.parse_args()
    per = os.path.join(args.res, "per_run")
    grid = {}
    subj_seed = []
    for jp in sorted(glob.glob(os.path.join(per, "*", "*", "baseline_no_stim.json"))):
        rec = json.load(open(jp))
        v = rec.get("baseline_verdict", {})
        if v.get("baseline_runaway_ms") is None:
            continue  # no runaway -> ineligible regardless of bar
        npz = jp.replace(".json", ".npz")
        if not os.path.isfile(npz):
            continue
        d = np.load(npz)
        af = d["active_frac"]
        T = rec["n_steps"] * rec["dt"]
        binw = T / len(af)
        stim_on = v["stim_on_ms"]
        subj_seed.append((rec["subject_id"], rec["seed"]))
        for wm in WINDOWS_MS:
            for cf in CAL_FRACS:
                bar = _bar(af, binw, wm, cf)
                ev = detect_events(af, binw, event_on_frac=bar)
                n_pre = sum(1 for e in ev if e["t_off"] <= stim_on and e["returned"])
                grid.setdefault((wm, cf), []).append((rec["subject_id"], rec["seed"], n_pre, n_pre >= MIN_EVENTS))
    # summarize
    print(f"event-bar sensitivity over {len(subj_seed)} subject-seed baselines (approx, downsampled af)")
    print(f"{'window_ms':>9} {'cal_frac':>8} {'n_admit':>8} {'min_pre':>7} {'median_pre':>10}")
    out = {"caveat": "downsampled active_fraction (~3.1ms bins); counts approximate; definitive control = slow-off baseline",
           "windows_ms": list(WINDOWS_MS), "cal_fracs": list(CAL_FRACS), "n_subject_seed": len(subj_seed),
           "grid": []}
    for (wm, cf), rows in sorted(grid.items()):
        pres = [r[2] for r in rows]
        n_admit = sum(1 for r in rows if r[3])
        print(f"{wm:>9.0f} {cf:>8.1f} {n_admit:>4}/{len(rows):<3} {min(pres):>7} {int(np.median(pres)):>10}")
        out["grid"].append(dict(window_ms=wm, cal_frac=cf, n_admitted=n_admit, n_total=len(rows),
                                min_pre=int(min(pres)), median_pre=float(np.median(pres))))
    all_admit = all(g["n_admitted"] == g["n_total"] for g in out["grid"])
    out["all_admitted_across_grid"] = bool(all_admit)
    with open(os.path.join(args.res, "event_bar_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAll subject-seed baselines admitted across the ENTIRE (window x cal_frac) grid: {all_admit}")
    print(f"-> {args.res}/event_bar_sensitivity.json")


if __name__ == "__main__":
    main()
