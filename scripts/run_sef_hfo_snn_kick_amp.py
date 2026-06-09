"""SNN kick-amplitude robustness probe (next-step roadmap P1, 2026-06-09).

Question: is the matched (variance-only, mean held at 18) evoked-synchrony NULL
robust to a STRONGER stimulus? The standing result (sweep + mean-scan) found
matched = no consistent evoked effect at the locked kick (2·nu_theta). If a 3×
kick suddenly surfaces a matched evoked effect, that's a pivot signal.

Advisor-locked design:
  - VARY only the NON-igniting conditions across amplitude. Self-ignition is
    detected pre-kick ([20,150]ms, kick at 150ms) so mean_only/unmatched are
    BIT-IDENTICAL pre-kick at 1×/2×/3× — sweeping them would re-confirm a
    true-by-construction non-result. So the real test is matched + baseline.
  - matched + baseline (both stay evoked_clean, mean=18) across KICK_BOOST ∈
    {1,2,3}·nu_theta × seeds → matched d_core_paf vs amplitude.
  - ONE igniting cell (unmatched, mid) run once at each amplitude as a
    confirmatory PRE-KICK-INVARIANT sanity (ignition latency must be identical).
  - fixed mid core + end kick.

Pre-registered reads:
  - matched d_core_paf ≈ 0 at 1×/2×/3× (null robust)            [PASS = stays ~0]
  - baseline never self-ignites at stronger kick (rate stays 0)  [sanity]
  - whole-net evoked peak GROWS with amplitude                   [sanity: kick does something]
  - unmatched pre-kick ignition latency identical across amps    [sanity: pre-kick invariant]
OVERTURN: matched shows a real evoked effect at 3× that was absent at 2× → STOP + flag.

Reuses grid _build/_run_one(kick_boost_mult)/_montage/_provenance/_engine_guard.
Modes: --quick (L=1 smoke) ; default (L=3).
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
PATCH_R = G.PATCH_R
AMPS = [1.0, 2.0, 3.0]
BASE_MEAN, BASE_STD, NARROW = 18.0, 1.5, 0.5


def _geom(L):
    ctr = np.array([L / 2, L / 2])
    u = np.array([np.cos(np.deg2rad(G.THETA)), np.sin(np.deg2rad(G.THETA))])
    return tuple(ctr), tuple(ctr + 0.6 * (L / 2) * u)        # mid core, end kick


def _ci(v):
    v = np.asarray(v, float); n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(v.mean()) if n else float("nan"), sd=sd,
                sem=float(sd / np.sqrt(n)) if n > 1 else 0.0, n=int(n), vals=v.tolist())


def _probe(L, density, conn_seeds, fn_seeds, out_tag=""):
    G._engine_guard()
    patch, kick = _geom(L)
    montage = G._montage(L)
    rows = []; t0 = time.time(); p_ref = None
    for cs in conn_seeds:
        p, net, nu_theta, NE = G._build(L, density, cs)
        p_ref = p
        is_E = net["labels"] == 0
        for fn in fn_seeds:
            base = sample_core_field(net["pos"], is_E, patch, PATCH_R,
                                     np.random.default_rng(fn * 911 + 1),
                                     core_mean=BASE_MEAN, core_std=BASE_STD)
            matched = sample_core_field(net["pos"], is_E, patch, PATCH_R,
                                        np.random.default_rng(fn * 911 + 2),
                                        core_mean=BASE_MEAN, core_std=NARROW)
            cm = base["core_mask"]
            for amp in AMPS:
                b = G._run_one(p, net, nu_theta, NE, kick, base["vth"], montage, cm, fn,
                               kick_boost_mult=amp)
                m = G._run_one(p, net, nu_theta, NE, kick, matched["vth"], montage, cm, fn,
                               kick_boost_mult=amp)
                rows.append(dict(
                    conn_seed=cs, fn_seed=fn, amp=amp,
                    d_core_paf=float(m["core_paf"] - b["core_paf"]),
                    matched_ignited=bool(m["prekick_ignited"]),
                    base_ignited=bool(b["prekick_ignited"]),
                    base_whole_paf=float(b["peak_active_frac"]),
                    matched_evoked_clean=(not m["prekick_ignited"]) and (not b["prekick_ignited"])))
            print(f"[conn{cs} fn{fn}] " + "  ".join(
                f"{amp:.0f}x:d={[r for r in rows if r['conn_seed']==cs and r['fn_seed']==fn and r['amp']==amp][0]['d_core_paf']:+.3f}"
                for amp in AMPS) + f"  ({time.time()-t0:.0f}s)", flush=True)

    # PRE-KICK-INVARIANT sanity: one igniting cell (unmatched mid) at the 3 amps, one seed.
    cs0 = conn_seeds[0]
    p, net, nu_theta, NE = G._build(L, density, cs0)
    is_E = net["labels"] == 0
    unm = sample_core_field(net["pos"], is_E, patch, PATCH_R, np.random.default_rng(7),
                            core_mean=BASE_MEAN - 2.0, core_std=NARROW)  # (16, narrow) = igniting
    cm = unm["core_mask"]
    sanity = []
    for amp in AMPS:
        u = G._run_one(p, net, nu_theta, NE, kick, unm["vth"], montage, cm, 1, kick_boost_mult=amp)
        sanity.append(dict(amp=amp, ignited=bool(u["prekick_ignited"]),
                           ignition_latency=u["ignition_latency"]))
    lat = [s["ignition_latency"] for s in sanity]
    prekick_invariant = bool(len(set(round(x, 4) for x in lat)) == 1)

    agg = {}
    for amp in AMPS:
        sub = [r for r in rows if r["amp"] == amp]
        dpaf = np.array([r["d_core_paf"] for r in sub if r["matched_evoked_clean"]], float)
        agg[f"{amp:.0f}x"] = dict(
            matched_d_core_paf=(_ci(dpaf) if dpaf.size else None),
            base_ignition_rate=float(np.mean([r["base_ignited"] for r in sub])),
            matched_ignition_rate=float(np.mean([r["matched_ignited"] for r in sub])),
            base_whole_paf=_ci([r["base_whole_paf"] for r in sub]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"kick_amp{out_tag}_metrics.json").write_text(json.dumps(dict(
        provenance=G._provenance(p_ref, conn_seeds[0]), amps=AMPS,
        conn_seeds=list(conn_seeds), fn_seeds=list(fn_seeds), patch=list(patch), kick=list(kick),
        aggregate=agg, sanity_prekick=sanity, prekick_invariant=prekick_invariant,
        raw=rows), indent=2))
    print(f"\nmatched d_core_paf by amp: " + ", ".join(
        f"{a}: {agg[a]['matched_d_core_paf']['mean']:+.3f}±{agg[a]['matched_d_core_paf']['sem']:.3f}"
        if agg[a]['matched_d_core_paf'] else f"{a}: --" for a in agg))
    print(f"base ignition rate by amp: " + ", ".join(f"{a}:{agg[a]['base_ignition_rate']:.2f}" for a in agg))
    print(f"whole-net evoked peak by amp: " + ", ".join(
        f"{a}:{agg[a]['base_whole_paf']['mean']:.3f}" for a in agg))
    print(f"PRE-KICK-INVARIANT sanity (unmatched latency identical across amps): {prekick_invariant} "
          f"(latencies {lat})")
    print(f"wrote kick_amp{out_tag}_metrics.json ({len(rows)} rows, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--conn", type=int, default=6)
    ap.add_argument("--fn", type=int, default=2)
    a = ap.parse_args()
    if a.quick:
        _probe(1.0, 4000.0, conn_seeds=[1, 2], fn_seeds=[1])
    else:
        _probe(3.0, 1800.0, conn_seeds=list(range(1, a.conn + 1)),
               fn_seeds=list(range(1, a.fn + 1)))


if __name__ == "__main__":
    main()
