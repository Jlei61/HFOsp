#!/usr/bin/env python
"""An empirical state map on the SNN itself: disinhibition against relay load.

The trajectory already on disk is one curve through the slow variables.  A curve is not a map: it
says what happened once, not what would happen at a state the tissue did not visit.  This holds
the two slow fields at chosen values and asks the fast network, from two different starting points,
what it does there.

Two choices make it a map of *this* substrate rather than of a homogeneous sheet:

* **the fields keep their shape.**  Each grid point scales the amplitude of a field taken from a
  real trajectory, so the two cores, the patient's axis and the recruitment history survive.
  Replacing every cell's slow variable with one number would be a question about a different
  tissue.
* **each point is started twice**, from a real interictal state and from a real ictal state, with
  everything the engine carries forward -- membrane, refractory counters, all synaptic states,
  both delay rings, the OU variable and the generator.  A point that stays low from one and high
  from the other is bistable in the only sense a finite stochastic network can be.

Axes: ``D = 1 - z`` is disinhibition, the variable that accumulates into entry.  ``X = 1 - relay``
is the termination load: sustained firing depletes the presynaptic relay, which lowers recurrent
excitation, so it opposes the discharge rather than driving it.

**Boundary.** The local recurrent state ``h`` is not frozen -- there is no frozen path for it in
the engine, and its 522 ms time constant equilibrates several times over inside a probe, so it is
part of the fast response here rather than a coordinate.  Adaptation ``m`` is off, as it is at the
registered working point.  Neither is hidden: the map's coordinates are D and X and nothing else.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_regime import classify_regime  # noqa: E402
from src.topic4_fcxr_lc3_statefork import (  # noqa: E402
    load_into,
    save_loop_state,
    scaled_fields,
)

OUT = os.path.join(E01.OUT, "phase_map")
NOISE = 401
REF_MS = 15000.0        # enough to hold an interictal moment and a settled ictal one
T_INTERICTAL_MS = 4000.0
T_ICTAL_MS = 12000.0
# The plan's 3-8 s window, at its short end: 3 s still holds ~35 bursts at 11.7/s, and this
# machine is oversubscribed by a neighbour, so a probe costs about half an hour of wall time.
PROBE_MS = 3000.0
ALPHA_D = (0.6, 0.8, 1.0, 1.2, 1.5)
ALPHA_X = (0.0, 0.5, 1.0, 1.5, 2.0)
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9

_CTX = {}


def _context():
    if not _CTX:
        S = PP.build_substrate(1)
        install_registered_noise_rng(S)
        _CTX["S"] = S
    return _CTX["S"]


def _loop(S, **kw):
    """Every run in this script goes through here.

    ``run_fcxr_loop`` defaults ``v_th_per_neuron`` to None, which silently replaces the per-neuron
    thresholds with one uniform value -- and the pathology in this substrate *is* two patches of
    lowered threshold, so omitting it runs a homogeneous sheet that never ignites.  The first
    version of this script omitted it at all three call sites and produced a reference trajectory
    with no slow-variable development at all.  One helper, one place to get it right.
    """
    p = dataclasses.replace(S["p"], T=kw.pop("T_ms"), dt=E01.DT)
    return run_fcxr_loop(p, S["net"], v_th_per_neuron=S["vth"], **kw)


def _fresh_slow(S, cfg_updates=None):
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    if cfg_updates:
        cfg.update(cfg_updates)
    return MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S))


def stage_a():
    """One reference trajectory, with the two moments the map is started from written out."""
    S = _context()
    ne = int(S["NE"])
    paths = dict(interictal=os.path.join(OUT, "ref_interictal.npz"),
                 ictal=os.path.join(OUT, "ref_ictal.npz"))
    meta_path = os.path.join(OUT, "reference.json")
    if os.path.isfile(meta_path) and GEO._load_json(meta_path).get("status") == "COMPLETE":
        return GEO._load_json(meta_path)

    slow = _fresh_slow(S)
    S["net"]["rng"] = np.random.default_rng(NOISE)
    t0 = time.time()
    legs = [("interictal", T_INTERICTAL_MS), ("ictal", T_ICTAL_MS - T_INTERICTAL_MS)]
    state, out, ceiling = None, {}, None
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    for name, span_ms in legs:
        n_steps = int(round(span_ms / E01.DT))
        run = _loop(S, T_ms=REF_MS, slow=(slow if state is None else None),
                    start=state, n_steps=n_steps, capture_final=True,
                    store_spikes=(name == "interictal"))
        if name == "interictal":
            # The probes start already high, so they carry no pre-onset stretch of their own; the
            # level a trough must clear has to come from here or it cannot come from anywhere.
            _e, af0, _b, _f, _ = OLD._events_from_res(
                dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"]),
                E01.DT, event_bar=float(baseline["frozen_event_bar"]))
            ceiling = float(np.percentile(np.asarray(af0, float), 95))
        state = run["checkpoint"]
        out[name] = dict(path=paths[name], t_ms=(T_INTERICTAL_MS if name == "interictal"
                                                 else T_ICTAL_MS),
                         state_hash=save_loop_state(paths[name], state),
                         mean_rate=float(np.mean(run["rate_E"])),
                         d_mean=float(np.mean(1.0 - np.asarray(state.slow.z[:ne], float))),
                         relay_mean=float(np.mean(np.asarray(state.slow.x_relay, float))))
        print(f"[phase] reference {name} at {out[name]['t_ms'] / 1000:.0f} s: "
              f"D {out[name]['d_mean']:.4f}  relay {out[name]['relay_mean']:.3f}", flush=True)

    ictal = load_into(paths["ictal"], state)
    d_star = 1.0 - np.asarray(ictal.slow.z[:ne], float)
    x_star = np.asarray(ictal.slow.x_relay, float)
    np.savez_compressed(os.path.join(OUT, "reference_fields.npz"),
                        d_star=d_star.astype(np.float32), x_star=x_star.astype(np.float32))
    rec = dict(status="COMPLETE", ref_ms=REF_MS, noise_seed=NOISE,
               point_id=GEO.H1_POINT_ID, states=out,
               interictal_ceiling_af=ceiling,
               d_star_mean=float(d_star.mean()), x_star_mean=float(x_star.mean()),
               wall_s=time.time() - t0, finished=GEO._now())
    GEO._write_json(meta_path, rec)
    return rec


def _probe(spec):
    """One grid point from one starting state, with D and X held where the point says."""
    tag = f"aD{spec['alpha_d']:g}_aX{spec['alpha_x']:g}_{spec['ic']}"
    out_json = os.path.join(OUT, "points", f"{tag}.json")
    if os.path.isfile(out_json) and GEO._load_json(out_json).get("status") == "COMPLETE":
        return GEO._load_json(out_json)

    S = _context()
    ne = int(S["NE"])
    ref = np.load(os.path.join(OUT, "reference_fields.npz"))
    d_field, x_field = scaled_fields(ref["d_star"], ref["x_star"],
                                     spec["alpha_d"], spec["alpha_x"])
    z_field = 1.0 - d_field
    template = _fresh_slow(S, dict(use_z=False, z_frozen_E=z_field.copy(),
                                   x_relay_frozen_E=x_field.copy()))
    seed_state = load_into(os.path.join(OUT, f"ref_{spec['ic']}.npz"),
                           _seed_template(S, template))
    slow = seed_state.slow
    slow.cfg = dataclasses.replace(slow.cfg, use_z=False, z_frozen_E=z_field.copy(),
                                   x_relay_frozen_E=x_field.copy())
    slow.z[:ne] = z_field
    slow.x_relay[:] = x_field
    slow.ee_relay_send[:] = x_field

    S["net"]["rng"] = np.random.default_rng(NOISE + 1)
    t0 = time.time()
    run = _loop(S, T_ms=PROBE_MS, start=seed_state,
                n_steps=int(round(PROBE_MS / E01.DT)),
                capture_final=True, store_spikes=True)
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    _events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))
    rate = np.asarray(run["rate_E"], float)
    band_hi = float(baseline["band"].get("roll_hi_hz", np.percentile(rate, 50)))
    spk = np.asarray(run["E_spk_bool"])
    per_cell_hz = spk.sum(axis=0) / (PROBE_MS * 1e-3)
    ceiling_frac = float(np.mean(per_cell_hz >= 0.8 * (1000.0 / S["p"].tau_ref_E)))
    ref_meta = GEO._load_json(os.path.join(OUT, "reference.json"))
    reg = classify_regime(af=af, af_bin_ms=af_dt, rate_hz=rate, dt_ms=E01.DT,
                          baseline_roll_hi_hz=band_hi, onset_ms=0.0, offset_ms=PROBE_MS,
                          run_ms=PROBE_MS, terminated=False, recovered=False,
                          refractory_ceiling_fraction=ceiling_frac,
                          interictal_ceiling_af=ref_meta["interictal_ceiling_af"],
                          numerical_unsafe=not np.all(np.isfinite(rate)))
    rec = dict(status="COMPLETE", tag=tag, alpha_d=spec["alpha_d"], alpha_x=spec["alpha_x"],
               ic=spec["ic"], probe_ms=PROBE_MS,
               D_mean=float(np.mean(d_field)), X_load_mean=float(np.mean(1.0 - x_field)),
               regime=reg["regime"], carrier=reg["carrier"], reason=reg["reason"],
               trough_af=reg.get("trough_af"), epoch_modulation=reg.get("epoch_modulation"),
               interictal_ceiling_af=reg.get("interictal_ceiling_af"),
               workpoint_label=reg.get("workpoint_label"),
               mean_af=float(np.mean(af)), max_af=float(np.max(af)),
               refractory_ceiling_fraction=ceiling_frac,
               wall_s=time.time() - t0, finished=GEO._now())
    GEO._write_json(out_json, rec)
    del run, res, spk
    gc.collect()
    return rec


def _seed_template(S, slow):
    """A zero-length run just to obtain a structurally valid state to load onto."""
    S["net"]["rng"] = np.random.default_rng(NOISE)
    return _loop(S, T_ms=E01.DT, slow=slow, n_steps=1, capture_final=True,
                 store_spikes=False)["checkpoint"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("a", "b", "both"), default="both")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k-neuron state map requires --confirm-run")
    os.makedirs(os.path.join(OUT, "points"), exist_ok=True)

    if args.stage in ("a", "both"):
        ref = stage_a()
        print(f"[phase] reference ready: {json.dumps(ref['states'], indent=2)[:400]}", flush=True)
    if args.stage == "a":
        return

    specs = [dict(alpha_d=ad, alpha_x=ax, ic=ic)
             for ad in ALPHA_D for ax in ALPHA_X for ic in ("interictal", "ictal")]
    per = BASE_RSS_GIB + GIB_PER_SIM_SECOND * PROBE_MS / 1000.0
    avail = GEO._meminfo()["mem_available_gib"]
    workers = args.workers or max(1, min(len(specs), int((avail - 40.0) // per)))
    if workers * per + 40.0 > avail:
        raise SystemExit(f"{workers} workers need {workers * per + 40.0:.0f} GiB; "
                         f"{avail:.0f} available")
    print(f"[phase] {len(specs)} points, {workers} workers, {per:.0f} GiB each, "
          f"{avail:.0f} GiB available", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe, s): s for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"[phase] {r['tag']:>28}  {r['regime']:>18}  "
                  f"mean_af {r['mean_af']:.4f}  ({len(rows)}/{len(specs)})", flush=True)

    GEO._write_json(os.path.join(OUT, "phase_map.json"),
                    dict(status="COMPLETE", probe_ms=PROBE_MS, alpha_d=list(ALPHA_D),
                         alpha_x=list(ALPHA_X), rows=rows))
    GEO._write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=GEO._now()))


if __name__ == "__main__":
    main()
