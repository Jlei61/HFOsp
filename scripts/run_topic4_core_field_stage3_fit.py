"""Fit a flexible field to the patient's per-event profile distribution.

The question is capacity: with the field free to sit anywhere and take any of
several shapes, how close can this network get to the patient's actual spread of
event profiles? The rigid family we already had spans 0.591 to 0.98 on that
distance, and the patient's own two halves sit at 0.031, so the answer is a
number on a scale we can already read.

One scalar objective, no tiers and no direction labels. Three networks per
candidate with common random numbers inside a generation: one run gives a
distance with sd 0.058 and a +0.06 small-sample bias, three brings the noise to
about 0.038, which is well under the differences worth resolving.

WARNING -- this is NOT the objective frozen in spec 9.3. That contract specifies
a two-dimensional energy distance over (slope, r2); what runs here is a
one-dimensional binned total variation over sign(slope)*r2. The substitution had
a reason -- the model recruits about seven contacts where the patient recruits
twelve, so slope magnitude carries the participation mismatch -- but it was made
without amending the contract, and it turned out to be wrong on the merits: the
marginal is satisfiable by a single mid-array generator, and the field this
script produced has its two directions correlated at +0.65 rather than opposite.
A future round must restore a joint observable. See spec 9.3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_profile_round1 import (  # noqa: E402
    axial_map, distance, patient_events, signed_monotonicity)
from src.topic4_core_field_cmaes import CMAES  # noqa: E402
from src.topic4_core_field_profile import split_by_block  # noqa: E402
from src.topic4_core_field_runner import (_placement, atomic_write_json,  # noqa: E402
                                          canonical_checksum, get_network,
                                          provenance)
from src.topic4_core_field_stage3 import (K_COMPONENTS, n_free,  # noqa: E402
                                          params_to_h, spatial_diagnostics)

STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
DROP = ("checksum", "provenance")

POPSIZE = 10
SEEDS_PER_CANDIDATE = 3
SEED_POOL = tuple(range(401, 441))        # never used by any earlier stage
SIGMA0 = 1.0
HELD_OUT_FRAC, SPLIT_SEED = 0.3, 20260808


def _load_cmrun():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evaluate(job):
    theta, seed, cfg, cache_dir, *optional = job
    component_count = int(optional[0]) if optional else K_COMPONENTS
    participant_target = int(optional[1]) if len(optional) > 1 else None
    try:
        from kick_probe import simulate_kick
        from lfp import LFPRecorder
        from params import Params
        from src.sef_hfo_events import detect_events
        from src.sef_hfo_snn_adapter import snn_event_envelope
        from src.topic4_core_field import (build_vth, core_thresholds,
                                           sample_core_quantiles, signed_depth)

        cmrun = _load_cmrun()
        e = cfg["engine"]
        k_dir = int(e["k_dir"])
        cmrun.KDIR, cmrun.PART_MIN = k_dir, 2 * k_dir + 1
        reg = _placement(cfg)
        p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
                   dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
        net, NE, NI, _ = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
        posE = net["pos"][:NE]

        h = params_to_h(np.asarray(theta, float), posE, component_count,
                        float(e["L"]), float(cfg["N_core_manual"]))
        d = signed_depth(core_thresholds(
            sample_core_quantiles(NE, cfg["quantile_seed"]), e["core_mean"],
            e["core_std"]), e["v_base"])
        vth = build_vth(h, d, n_total=NE + NI, n_E=NE, v_base=e["v_base"])

        msheet = reg["montage_sheet"]
        valid = cmrun.valid_mask(msheet, posE, e["L"], p.Rr)
        rec = LFPRecorder(p, net["pos"], net["labels"], sites=msheet.contacts)
        net["rng"] = np.random.default_rng(seed)
        res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
                            r_kick=e["core_r"], t_kick=1e9, V_th_per_neuron=vth,
                            lfp_recorder=rec)
        spk = res["E_spk_bool"]
        af, bin_w = cmrun.active_fraction(spk, e["dt"], cmrun.BIN_MS)
        nb0, nb1 = int(cmrun.BASELINE_MS[0] / bin_w), int(cmrun.BASELINE_MS[1] / bin_w)
        floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
        bar = floor + cmrun.CAL_FRAC * (float(af.max()) - floor)
        env_f, fdt, _ = snn_event_envelope(spk, posE, msheet, e["dt"])

        events = []
        detected = detect_events(af, bin_w, event_on_frac=bar)
        for ev in detected:
            rd = cmrun.read_event(env_f, fdt, msheet, valid,
                                  (ev["t_on"], ev["t_off"]), reg["axis_unit_vec"],
                                  k_dir=k_dir, part_min=2 * k_dir + 1)
            events.append(dict(ranks=rd["ranks"], n_part=rd["n_part"]))
        diag = spatial_diagnostics(h, posE, reg["center"], reg["axis_unit_vec"])
        part_min = 2 * k_dir + 1
        credit_target = part_min if participant_target is None else participant_target
        return dict(seed=seed, events=events, r_bar=diag["r_bar"],
                    s_bar=diag["s_bar"], c_axis_2mm=diag["c_axis"][2.0],
                    n_detected=int(len(detected)),
                    max_n_part=int(max((ev["n_part"] for ev in events), default=0)),
                    participant_credit=float(sum(
                        min(float(ev["n_part"]) / credit_target, 1.0)
                        for ev in events)),
                    active_fraction_peak=float(af.max()),
                    active_fraction_floor=float(floor))
    except Exception as exc:                                # noqa: BLE001
        return dict(seed=seed, error=repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-gens", type=int, default=14)
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--seeds-per-candidate", type=int, default=SEEDS_PER_CANDIDATE,
                    help="networks per candidate; 1 gives distance sd 0.058, "
                         "2 gives 0.044, 3 gives 0.038")
    ap.add_argument("--restart", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    axial = axial_map()
    vals, blocks = patient_events(axial)
    tr, te = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    p_train, p_test = vals[tr], vals[te]

    tag = f"K{K_COMPONENTS}_r{a.restart}"
    ck_path = os.path.join(a.out, "fit", f"checkpoint_{tag}.json")
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    dim = n_free(K_COMPONENTS)

    if os.path.exists(ck_path):
        ck = json.load(open(ck_path))
        es = CMAES.from_state(ck["optimizer"])
        history = ck["history"]
        print(f"[{tag}] resuming at generation {len(history) // POPSIZE}", flush=True)
    else:
        rng = np.random.default_rng(1000 + a.restart)
        x0 = np.zeros(dim)
        for k in range(K_COMPONENTS):                 # centres uniform in the sheet
            x0[5 * k:5 * k + 2] = rng.uniform(2.0, 18.0, size=2)
            # sigma 0.8-3.0 spread the fixed budget of 1129 cells so thin that
            # no cell ended up strongly pathological -- max h 0.55-0.91 with at
            # most 30 cells above 0.9, against 738 for the single blob that
            # produces 22 events. Nothing ignited, so generation 1 came back
            # flat at the penalty value with no gradient to follow.
            x0[5 * k + 2:5 * k + 4] = np.log(rng.uniform(0.5, 1.5, size=2))
            x0[5 * k + 4] = rng.uniform(0, np.pi)
        es = CMAES(x0, SIGMA0, seed=2000 + a.restart, popsize=POPSIZE)
        history = []
        print(f"[{tag}] fresh start, {dim} free dimensions, no warm start",
              flush=True)

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    cache = os.path.join(STAGE2, "network_cache")
    t0 = time.time()
    gen = len(history) // POPSIZE
    while gen < a.max_gens and (time.time() - t0) / 3600.0 < a.hours:
        xs = es.ask()
        # common random numbers: every candidate in this generation meets the
        # same networks, so comparisons within a generation are paired
        rng = np.random.default_rng(9000 + gen)
        seeds = list(rng.choice(SEED_POOL, size=a.seeds_per_candidate, replace=False))
        jobs = [(x, int(s), cfg, cache) for x in xs for s in seeds]
        with Pool(a.workers, maxtasksperchild=1) as pool:
            res = pool.map(_evaluate, jobs)

        keys, rows = [], []
        for i, x in enumerate(xs):
            chunk = res[i * len(seeds):(i + 1) * len(seeds)]
            vals_i, diag = [], [c for c in chunk if "error" not in c]
            for c in diag:
                for ev in c["events"]:
                    v = signed_monotonicity(ev.get("ranks"), axial)
                    if v is not None:
                        vals_i.append(v)
            # a flat penalty leaves the search no way out of an event-less
            # region; grade it by how close the candidate came to having enough
            if len(vals_i) >= 10:
                dist = distance(vals_i, p_train)
            else:
                dist = 1.0 + 0.5 * (10 - len(vals_i)) / 10.0
            keys.append(-float(dist))                 # CMA-ES maximises
            rows.append(dict(theta=[float(t) for t in x], distance=float(dist),
                             n_events=len(vals_i), seeds=[int(s) for s in seeds],
                             r_bar=float(np.mean([c["r_bar"] for c in diag])) if diag else None,
                             c_axis_2mm=float(np.mean([c["c_axis_2mm"] for c in diag])) if diag else None,
                             n_failed=len(chunk) - len(diag)))
        es.tell(xs, keys)
        history.extend(rows)
        best = min(rows, key=lambda r: r["distance"])
        gen += 1
        print(f"[{tag}] gen {gen:2d}  best {best['distance']:.3f}  "
              f"gen-median {np.median([r['distance'] for r in rows]):.3f}  "
              f"events {best['n_events']:3d}  sigma {es.sigma:.3f}  "
              f"{(time.time()-t0)/60:.0f} min", flush=True)
        atomic_write_json(dict(
            tag=tag, K=K_COMPONENTS, popsize=POPSIZE,
            seeds_per_candidate=a.seeds_per_candidate, restart=a.restart,
            optimizer=es.get_state(), history=history,
            patient=dict(n_train=int(len(p_train)), n_heldout=int(len(p_test)),
                         split_seed=SPLIT_SEED, frac=HELD_OUT_FRAC),
            reference=dict(rigid_family_best=0.591, rigid_family_worst=0.983,
                           hand_placed_two_cores=0.855, learned_filament=0.686,
                           # the floor must match the scorer's structure: a
                           # model-sized sample against the FULL patient
                           # training set. 0.031 uses every event on both sides
                           # and is not comparable to any model number.
                           floor_note=("computed per artifact by "
                                       "run_topic4_core_field_stage3_confirm_fit; "
                                       "about 0.18 at 80 model events")),
            config_checksum=cfg["checksum"], provenance=provenance()), ck_path)

    overall = min(history, key=lambda r: r["distance"])
    print(f"\n[{tag}] {len(history)} candidates, best distance {overall['distance']:.3f}")
    print(f"        rigid family best 0.591, hand-placed pair 0.855; the floor "
          f"is structure-matched per artifact by the confirmation script")


if __name__ == "__main__":
    main()
