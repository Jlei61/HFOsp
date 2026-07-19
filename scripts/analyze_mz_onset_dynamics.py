#!/usr/bin/env python3
"""Full per-condition dynamics for the MZ z–m gap grid (review 2026-07-19 §6.4). Reports, per target
adaptation strength and seed, NOT just a phenotype label: onset latency, D_max & pre-onset D, adaptation
onset-time vs recruitment onset, event count/duration, peak rate, spontaneous retrigger, whether activity
returns to baseline. Then a cross-seed verdict on whether any strength shows the working-point signature:
  interictal events -> D approaches the z-only onset region -> recruitment -> bounded -> return -> responsive.
Consumes per_seed/traj_zA_q75_tz5000_A*_seed*.npz (+ slow-off). Pure post-processing; no simulation.
"""
import glob
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_onset_dynamics")
TRAJ = os.path.join(OUT, "per_seed")
REGIME = "zA_q75_tz5000"
# clean gap grid (T=20000, frozen event bar). Old strong-end fracs (0.025-0.20, T=15000, polluted bar)
# are excluded from the verdict so protocols are not mixed.
GAP_FRACS = {0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01}


def per_cell(z):
    t, D, a, r = z["t_ms"], z["D_allE"], z["a_allE"], z["rate_E_hz"]
    on, off = z["event_on_ms"], z["event_off_ms"]
    ra = float(z["runaway_ms"])
    onset = ra if np.isfinite(ra) else None
    if onset is not None:                                   # pre-onset D = D 500 ms before run-off
        d_pre = float(D[max(0, np.searchsorted(t, onset - 500.0))])
    else:                                                   # else mean D over the last 500 ms
        d_pre = float(np.mean(D[t >= t[-1] - 500.0]))
    a_max = float(a.max())
    a_onset = float(t[int(np.argmax(a > 0.1 * a_max))]) if a_max > 0 else None    # adaptation engages
    # D and realized adaptation AT the first sustained crossing (runaway_ms) = the true onset state (P1-2):
    # onset reference must be D_at_runaway, NOT the post-crossing D_max (which is ~100-140 ms later).
    ic = min(int(np.searchsorted(t, onset)), len(t) - 1) if onset is not None else None
    d_at_runaway = float(D[ic]) if ic is not None else None
    a_at_runaway = float(a[ic]) if ic is not None else None
    rec_onset = float(on[0]) if on.size else None                                 # first interictal event
    durs = (off - on) if on.size else np.array([])
    t_peakD = float(t[int(np.argmax(D))])
    retrig = int(np.sum(on > t_peakD + 200.0)) if on.size else 0                  # events after peak-D
    returned = bool(np.mean(r[t >= t[-1] - 1000.0]) < 1.0)                        # rate quiet in last 1 s
    d_recovered = bool(float(D[-1]) < 0.3 * float(D.max()) + 1e-9)                # D substantially back to baseline
    return dict(
        target_frac=round(float(z["A_frac"]), 5), seed=int(z["seed"]), eta_m=round(float(z["eta_m"]), 5),
        onset_ms=None if onset is None else round(onset, 0),
        D_max=round(float(D.max()), 4), D_pre_onset=round(d_pre, 4), D_end=round(float(D[-1]), 4),
        D_at_runaway=None if d_at_runaway is None else round(d_at_runaway, 4),
        a_at_runaway=None if a_at_runaway is None else round(a_at_runaway, 5),
        a_max_realized=round(a_max, 5), a_onset_ms=None if a_onset is None else round(a_onset, 0),
        recruit_onset_ms=None if rec_onset is None else round(rec_onset, 0),
        n_events=int(on.size), event_dur_med_ms=round(float(np.median(durs)), 1) if durs.size else None,
        rate_max_hz=round(float(r.max()), 1), spont_retrigger=retrig, returned=returned, d_recovered=d_recovered)


def main():
    cells = {}
    for f in sorted(glob.glob(os.path.join(TRAJ, f"traj_{REGIME}_A*_seed*.npz"))):
        if "_tau" in os.path.basename(f):
            continue                          # tau-sweep cells are a separate sensitivity, NOT the tau=2000 gap grid
        z = np.load(f)
        fr = round(float(z["A_frac"]), 5)
        if fr not in GAP_FRACS:
            continue
        cells[(fr, int(z["seed"]))] = per_cell(z)
    if not cells:
        print("no gap-grid trajectories found yet")
        return
    # onset-D reference = D at FIRST-crossing (runaway_ms) of a=0 cells, NOT post-crossing D_max (P1-2 fix)
    zonly = [c["D_at_runaway"] for (fr, s), c in cells.items() if fr == 0.0 and c["D_at_runaway"] is not None]
    D_onset_ref = float(np.mean(zonly)) if zonly else 0.087
    # run-off corridor + m-timing across ALL run-off cells (gap z-only + tau-sweep z+m runaways) — cross-grid
    xc, mt = [], []
    for f in sorted(glob.glob(os.path.join(TRAJ, f"traj_{REGIME}_A*_seed*.npz"))):
        zz = np.load(f)
        ra = float(zz["runaway_ms"])
        if not np.isfinite(ra):
            continue
        tt = zz["t_ms"]
        ii = min(int(np.searchsorted(tt, ra)), len(tt) - 1)
        xc.append(float(zz["D_allE"][ii]))
        if float(zz["a_allE"].max()) > 0:                       # m on (tau z+m runaways): a@crossing vs a_max
            mt.append((float(zz["a_allE"][ii]), float(zz["a_allE"].max())))
    fracs = sorted({fr for fr, _ in cells})
    seeds = sorted({s for _, s in cells})

    print(f"onset-D reference (D at first-crossing, a=0 cells) = {D_onset_ref:.4f}")
    if xc:
        print(f"run-off D corridor (all {len(xc)} runaways) = {np.mean(xc):.4f} ± {np.std(xc):.4f}"
              f"  — m determines whether the trajectory REACHES the boundary, not where it is")
    if mt:
        print(f"adaptation at crossing = {np.mean([m[0] for m in mt]):.5f} (weak) vs a_max = "
              f"{np.mean([m[1] for m in mt]):.5f} (post-onset) — feedback too late to contain")
    print()
    hdr = "target_frac  seed  eta_m    onset_ms  D_max   D_pre  a_max     n_ev  rate_max  retrig  returned"
    print(hdr)
    rows = []
    for fr in fracs:
        for s in seeds:
            c = cells.get((fr, s))
            if not c:
                continue
            rows.append(c)
            print("  %.4f    s%d   %.5f  %-8s  %.4f  %.4f  %.5f  %-4d  %-8s  %-6d  %s" % (
                fr, s, c["eta_m"], str(c["onset_ms"]), c["D_max"], c["D_pre_onset"], c["a_max_realized"],
                c["n_events"], str(c["rate_max_hz"]), c["spont_retrigger"], c["returned"]))

    # per-cell regime, then cross-seed majority. A TRUE onset-containment-recovery cycle needs D to
    # approach onset AND recover (d_recovered). "plateau" = D approaches onset but stays elevated (no recovery).
    def regime(c):
        if c["onset_ms"] is not None:
            return "runaway"
        if c["D_max"] >= 0.5 * D_onset_ref:
            return "bounded_elevated_cycle" if c["d_recovered"] else "bounded_elevated_plateau"
        return "prevention"

    print("\n--- cross-seed regime per target frac (onset-D ref=%.4f; bounded_elevated = D>=0.5*onset) ---" % D_onset_ref)
    verdict = {}
    for fr in fracs:
        cs = [cells[(fr, s)] for s in seeds if (fr, s) in cells]
        labs = [regime(c) for c in cs]
        maj = Counter(labs).most_common(1)[0][0] if labs else "n/a"
        dmx = float(np.mean([c["D_max"] for c in cs])) if cs else 0.0
        n_recov = sum(1 for c in cs if c["d_recovered"])
        verdict[fr] = dict(n_seeds=len(cs), regimes=labs, majority=maj, mean_D_max=round(dmx, 4), n_recovered=n_recov)
        print(f"  frac={fr:.4f}: {maj:26s} mean_D_max={dmx:.4f}  regimes={labs}  recovered={n_recov}/{len(cs)}")

    has_cycle = any(v["majority"] == "bounded_elevated_cycle" for v in verdict.values())
    has_plateau = any(v["majority"] == "bounded_elevated_plateau" for v in verdict.values())
    if has_cycle:
        conclusion = ("A cross-seed onset-containment-RECOVERY cycle exists (D approaches onset, bounded, recovers) "
                      "= working point.")
    elif has_plateau:
        conclusion = ("Graded prevention with a bounded sub-onset PLATEAU: weak adaptation stalls the z-driven "
                      "disinhibition ratchet in a bounded elevated band (D>=0.5*onset) that does NOT recover; "
                      "stronger adaptation stalls it lower. Richer than binary, but NOT an onset-containment-recovery "
                      "cycle (no discrete recruited event, no recovery). Minimal linear spike-adaptation PREVENTS/"
                      "STALLS onset; it does not contain-and-recover a seizure.")
    else:
        conclusion = "Binary prevention: adaptation prevents onset with no bounded elevated approach."
    print("\nCONCLUSION:", conclusion)
    json.dump(dict(D_onset_ref=D_onset_ref, cells=rows, verdict={str(k): v for k, v in verdict.items()},
                   conclusion=conclusion),
              open(os.path.join(OUT, "gap_dynamics_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "gap_dynamics_summary.json"))


if __name__ == "__main__":
    main()
