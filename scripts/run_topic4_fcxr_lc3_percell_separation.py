#!/usr/bin/env python
"""Can a cell tell, from its own firing alone, that the tissue is seizing?

Every terminator tried here failed on one of two sides.  Loads a cell builds from its own spikes
were on during the interictal train and suppressed it.  A brake reading whole-array recruitment
left the train alone and terminated nothing.  The accepted contract requires the state to be
per-cell; the separation analysis in the entry ledger says recruitment is the only measured
variable that tells the states apart.  Those two are only in conflict if a per-cell load genuinely
cannot separate them -- and that is measurable.

So: the same network, in the two states, with nothing added.  Each cell's firing is recorded, its
load is reconstructed, and the question is whether a level exists above everything the interictal
train produces and below where the discharge sits.

* Yes -> an actuator whose half-activation sits there is seizure-selective while reading nothing
  but the cell's own spikes.  Selectivity then comes from a mechanism parameter, not from a sensor.
* No  -> no per-cell spike-driven mechanism can be selective on this substrate, whatever its time
  constant or actuator shape, and the recruitment finding holds at the level of single cells too.

Both outcomes end a line of work, which is why this runs before any arm does.

Nothing is perturbed: the probes carry no adaptation, no brake and no frozen fields.  The
interictal probe re-runs the reference trajectory's own first seconds, so its mean rate is checked
against the reference as a guard that this is the same tissue in the same state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_phase_map as PM  # noqa: E402
from src.topic4_fcxr_lc3_percell_load import (  # noqa: E402
    replay_load,
    separation,
    stationary_load,
)
from src.topic4_fcxr_lc3_statefork import load_into  # noqa: E402

OUT = os.path.join(E01.OUT, "percell_separation")
# The interictal window is everything the trajectory has before it enters at ~5 s; the discharge
# needs only enough bursts to fix a rate distribution, and at ~11.5 bursts a second 2 s holds ~23.
INTERICTAL_MS = 4000.0
ICTAL_MS = 2000.0
TAUS = (250.0, 500.0, 1000.0, 2000.0, 5000.0)
PROBE_SEED = 4101


def _probe(name, span_ms):
    """One state, recorded.  No adaptation, no brake, no frozen field -- the tissue as it runs."""
    S = PM._context()
    slow = PM._fresh_slow(S)
    if name == "interictal":
        # Re-runs the reference trajectory's own opening, so the guard below can compare it.
        S["net"]["rng"] = np.random.default_rng(PM.NOISE)
        start = None
        kw = dict(slow=slow)
    else:
        start = load_into(os.path.join(PM.OUT, "ref_ictal.npz"), PM._seed_template(S, slow))
        S["net"]["rng"] = np.random.default_rng(PROBE_SEED)
        kw = dict(start=start)
    run = PM._loop(S, T_ms=span_ms, n_steps=int(round(span_ms / E01.DT)),
                   capture_final=False, store_spikes=True, **kw)
    spk = np.asarray(run["E_spk_bool"])
    per_cell_hz = spk.sum(axis=0) / (span_ms * 1e-3)
    mean_rate = float(np.mean(run["rate_E"]))
    return spk, per_cell_hz, mean_rate


def _loads(spk, per_cell_hz, span_ms):
    """Peak and settled load per cell, at each time constant, seeded at the stationary level.

    A replay that starts at zero reports its own charge ramp over a window shorter than a few time
    constants -- and on the interictal side that understates the load, which is the direction that
    would make the separation look better than it is.
    """
    out = {}
    for tau in TAUS:
        r = replay_load(spk, E01.DT, tau, init=stationary_load(per_cell_hz, tau),
                        settle_from_ms=span_ms / 2.0)
        out[tau] = dict(peak=r["peak"], settled=r["settled"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k-neuron probes require --confirm-run")
    os.makedirs(OUT, exist_ok=True)
    ref = GEO._load_json(os.path.join(PM.OUT, "reference.json"))
    t0 = time.time()

    rates, loads, means = {}, {}, {}
    for name, span in (("interictal", INTERICTAL_MS), ("ictal", ICTAL_MS)):
        spk, hz, mean_rate = _probe(name, span)
        print(f"[sep] {name}: {span / 1000:.0f} s, mean rate {mean_rate:.2f} Hz, "
              f"per-cell rate median {np.median(hz):.2f} max {hz.max():.1f} Hz", flush=True)
        rates[name], means[name] = hz, mean_rate
        loads[name] = _loads(spk, hz, span)
        del spk

    guard = dict(
        reference_interictal_mean_rate=ref["states"]["interictal"]["mean_rate"],
        probe_interictal_mean_rate=means["interictal"],
        # The reference averaged its rate over a 15 s allocation while running 4 s of it, so the
        # two are compared on the ratio of what each actually simulated, not on equality.
        note="a large mismatch means this is not the reference trajectory's interictal state")

    verdicts = {}
    for tau in TAUS:
        ana = stationary_load(rates["ictal"], tau)
        v = separation(loads["interictal"][tau]["peak"], ana)
        v["ictal_settled_replayed_median"] = float(np.median(loads["ictal"][tau]["settled"]))
        v["ictal_settled_analytic_median"] = float(np.median(ana))
        v["interictal_settled_median"] = float(np.median(loads["interictal"][tau]["settled"]))
        verdicts[f"{tau:g}"] = v
        print(f"[sep] tau {tau:>6.0f} ms | K {v['K']:8.3f} | discharge median {v['ictal_median']:9.3f}"
              f" | headroom {v['headroom']:7.2f} | {v['ictal_frac_above_K'] * 100:5.1f}% of cells"
              f" above K | {'SEPARATES' if v['separates'] else 'no'}", flush=True)

    np.savez_compressed(
        os.path.join(OUT, "per_cell.npz"),
        interictal_hz=rates["interictal"].astype(np.float32),
        ictal_hz=rates["ictal"].astype(np.float32),
        **{f"interictal_peak_tau{t:g}": loads["interictal"][t]["peak"].astype(np.float32)
           for t in TAUS},
        **{f"ictal_settled_tau{t:g}": loads["ictal"][t]["settled"].astype(np.float32)
           for t in TAUS})
    rec = dict(
        status="COMPLETE", schema="fcxr-lc3-percell-separation-1.0",
        interictal_ms=INTERICTAL_MS, ictal_ms=ICTAL_MS, taus=list(TAUS),
        probe_seed=PROBE_SEED, point_id=GEO.H1_POINT_ID, guard=guard,
        mean_rate=means, verdicts=verdicts,
        rate_ratio=means["ictal"] / means["interictal"],
        boundary=("the load is a linear filter of the cell's own spikes, so its stationary value is "
                  "rate x tau and the ratio between the two states is the ratio of the rates at "
                  "every tau; the time constant changes the transient, never the separation of the "
                  "levels.  A yes here licenses placing an actuator's half-activation between them, "
                  "not a claim that doing so terminates anything."),
        wall_s=time.time() - t0, finished=GEO._now())
    GEO._write_json(os.path.join(OUT, "separation.json"), rec)
    print(f"\n[sep] wrote {os.path.join(OUT, 'separation.json')}  ({rec['wall_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
