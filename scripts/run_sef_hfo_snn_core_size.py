"""SNN core-SIZE axis probe (autonomous exploration C, 2026-06-09, GUARDED).

Question: does a BIGGER pathology core (more low-threshold tail cells) self-ignite
at a HIGHER mean — i.e. does core size lower the ignition boundary? The tail
mechanism (confirmed by A's boundary + B's weak-recruitment finding) predicts yes,
and predicts it acts via the WIDE tail (so the radius effect should be stronger for
wide than narrow).

ADVISOR GUARD (core-size-up back-doors into the wide-everywhere bursting regime):
a bigger WIDE core holds more low-threshold tail cells, so the HEALTHY baseline
(wide, mean=18) core itself may self-ignite at large r — destroying the clean
reference (the failure that produced the false "matched reduces synchrony" once).
So: r capped at 0.6 (at L=3 that is ~6% of sheet area, still "small core"); and the
(r, wide, mean=18) self-ignition rate is the BUILT-IN GUARD — it must stay 0 for the
boundary-lowering reading to be clean. self-ignition is pre-kick, so this IS the
kick-OFF spontaneous-bursting check the advisor asked for.

Design: radius ∈ {0.3, 0.45, 0.6} × mean ∈ {18,17,16.5,16} (spec-locked range, no
gap-limit relaxation) × std {wide,narrow} × 6 connection seeds. Mid core, end kick
(kick irrelevant to PRE-kick ignition). Pre-kick ignition rate per cell.

Pre-registered reads:
  - GUARD: (r, wide, 18) ignition rate = 0 at every r (clean reference)   [else FLAG]
  - bigger r → ignition boundary at HIGHER mean (boundary lowers)         [tail predicts yes]
  - radius effect stronger for WIDE than NARROW                           [tail-specific]
OVERTURN/FLAG: if baseline (wide,18) ignites at large r, the finite-size bursting
regime is reached — report it loudly, don't fold a compromised reference into a
clean boundary story.

Reuses grid _build/_run_one/_montage/_provenance/_engine_guard + sample_core_field(radius).
Modes: --quick (L=1) ; default (L=3).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import run_sef_hfo_snn_hetero_grid as G            # noqa: E402
sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_core_field   # noqa: E402

OUT = G.OUT
RADII = [0.3, 0.45, 0.6]
MEANS = [18.0, 17.0, 16.5, 16.0]
STDS = {"wide": 1.5, "narrow": 0.5}


def _geom(L):
    ctr = np.array([L / 2, L / 2])
    u = np.array([np.cos(np.deg2rad(G.THETA)), np.sin(np.deg2rad(G.THETA))])
    return tuple(ctr), tuple(ctr + 0.6 * (L / 2) * u)


def _cross_mean(means_desc, rates):
    m = np.asarray(means_desc, float); r = np.asarray(rates, float)
    for i in range(len(m) - 1):
        if (r[i] - 0.5) * (r[i + 1] - 0.5) <= 0 and r[i] != r[i + 1]:
            f = (0.5 - r[i]) / (r[i + 1] - r[i])
            return float(m[i] + f * (m[i + 1] - m[i]))
    return float("nan")


def _field_seed(cs, m, sname, r):
    return int(cs * 100003 + int(round(m * 10)) * 7 + int(round(r * 100)) * 3
               + (1 if sname == "narrow" else 0))


def _probe(L, density, conn_seeds, out_tag=""):
    G._engine_guard()
    patch, kick = _geom(L)
    montage = G._montage(L)
    rows = []; t0 = time.time(); p_ref = None
    tot = len(RADII) * len(conn_seeds) * len(MEANS) * len(STDS)
    k = 0
    for cs in conn_seeds:
        p, net, nu_theta, NE = G._build(L, density, cs)
        p_ref = p
        is_E = net["labels"] == 0
        for r in RADII:
            for m in MEANS:
                for sname, sval in STDS.items():
                    f = sample_core_field(net["pos"], is_E, patch, r,
                                          np.random.default_rng(_field_seed(cs, m, sname, r)),
                                          core_mean=m, core_std=sval)
                    c = G._run_one(p, net, nu_theta, NE, kick, f["vth"], montage, f["core_mask"], cs)
                    k += 1
                    rows.append(dict(conn_seed=cs, radius=r, mean=m, std=sname,
                                     n_core=int(f["core_mask"].sum()),
                                     ignited=bool(c["prekick_ignited"]),
                                     latency=c["ignition_latency"]))
            print(f"[conn{cs} r{r}] " + " ".join(
                f"{m:.1f}:" + "".join("I" if [x for x in rows if x['conn_seed']==cs and x['radius']==r
                                              and x['mean']==m and x['std']==s][0]['ignited'] else "."
                                      for s in STDS) for m in MEANS)
                  + f"  ({time.time()-t0:.0f}s, {k}/{tot})", flush=True)

    agg = {}; boundary = {}
    for sname in STDS:
        agg[sname] = {}; boundary[sname] = {}
        for r in RADII:
            rates = []
            for m in MEANS:
                sub = [x for x in rows if x["radius"] == r and x["mean"] == m and x["std"] == sname]
                ig = float(np.mean([x["ignited"] for x in sub]))
                agg[sname][f"r{r}_m{m}"] = dict(ignition_rate=ig, n=len(sub),
                                                n_core=int(np.median([x["n_core"] for x in sub])))
                rates.append(ig)
            boundary[sname][f"r{r}"] = _cross_mean(sorted(MEANS, reverse=True),
                                                   [agg[sname][f"r{r}_m{m}"]["ignition_rate"]
                                                    for m in sorted(MEANS, reverse=True)])
    # GUARD: baseline (wide, mean=18) self-ignition rate per radius
    guard = {f"r{r}": float(np.mean([x["ignited"] for x in rows
                                     if x["radius"] == r and x["mean"] == 18.0 and x["std"] == "wide"]))
             for r in RADII}
    guard_clean = all(v == 0.0 for v in guard.values())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"core_size{out_tag}_metrics.json").write_text(json.dumps(dict(
        provenance=G._provenance(p_ref, conn_seeds[0]), radii=RADII, means=MEANS, stds=STDS,
        conn_seeds=list(conn_seeds), patch=list(patch), kick=list(kick),
        aggregate=agg, ignition_boundary=boundary,
        guard_baseline_wide18_ignition=guard, guard_clean=guard_clean,
        raw=rows), indent=2))
    print(f"\nGUARD baseline(wide,18) ignition by radius: {guard}  -> clean={guard_clean}")
    print(f"core sizes (median n_core) by radius: " + ", ".join(
        f"r{r}:{int(np.median([x['n_core'] for x in rows if x['radius']==r]))}" for r in RADII))
    print("ignition boundary (mean @0.5) — WIDE: " + ", ".join(
        f"r{r}:{boundary['wide'][f'r{r}']:.2f}" for r in RADII))
    print("ignition boundary (mean @0.5) — NARROW: " + ", ".join(
        f"r{r}:{boundary['narrow'][f'r{r}']:.2f}" for r in RADII))
    print(f"wrote core_size{out_tag}_metrics.json ({len(rows)} rows, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--conn", type=int, default=6)
    a = ap.parse_args()
    if a.quick:
        _probe(1.0, 4000.0, conn_seeds=[1, 2])
    else:
        _probe(3.0, 1800.0, conn_seeds=list(range(1, a.conn + 1)))


if __name__ == "__main__":
    main()
