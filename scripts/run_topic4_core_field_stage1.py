"""Stage 1: 8 arms x 12 seeds paired probe, parallelised over NETWORK SEEDS.

One worker owns a seed: it builds or loads that seed's network once and runs all
eight arms on it. Dispatching (arm, seed) instead would have eight workers miss
the cache together, build the same network eight times and overwrite one file.

Refuses to launch if a shape comparison is vacuous -- that check costs
milliseconds and the run costs an hour.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (  # noqa: E402
    ARM_NAMES, arm_h, axis_coords, manual_mask, preflight_shape)
from src.topic4_core_field_runner import (  # noqa: E402
    _placement, atomic_write_json, canonical_checksum, get_network,
    provenance, run_arm_on_network)

OUT = "results/topic4_sef_hfo/data_driven_core_field"


def _load_cmrun():
    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _geom_and_mask(cfg, reg, posE):
    s, r = axis_coords(posE, reg["center"], reg["axis_unit_vec"])
    geom = dict(sep=float(np.linalg.norm(reg["sink_centroid"] - reg["source_centroid"])),
                s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                           float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                M=cfg["field"]["M"], sigma_perp=cfg["engine"]["core_r"],
                shift_mm=cfg["field"]["SHIFT_MM"])
    mask = manual_mask(posE, reg["source_centroid"], reg["sink_centroid"],
                       cfg["engine"]["core_r"])
    return s, r, geom, mask


def preflight(cfg):
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from connectivity import place_neurons
    from params import Params
    e = cfg["engine"]
    reg = _placement(cfg)
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=100.0, dt=e["dt"], seed=1)
    pos, _, NE, _ = place_neurons(p, np.random.default_rng(1))
    posE = pos[:NE]
    s, r, geom, mask = _geom_and_mask(cfg, reg, posE)
    target = float(cfg["N_core_manual"])
    h_by_arm = {a: arm_h(a, s, r, geom, target, manual_mask_E=mask)
                for a in ARM_NAMES if a != "manual_hard"}
    h_by_arm["manual_hard"] = mask.astype(float)
    return preflight_shape(h_by_arm, s, r, target)


def _seed_job(args):
    seed, cfg, cache_dir, out_dir = args
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from params import Params
    try:
        cmrun = _load_cmrun()
        k_dir = int(cfg["engine"]["k_dir"])
        cmrun.KDIR, cmrun.PART_MIN = k_dir, 2 * k_dir + 1
        reg = _placement(cfg)
        e = cfg["engine"]
        p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
                   dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
        net, NE, NI, hit = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
        done = []
        for arm in ARM_NAMES:
            path = os.path.join(out_dir, str(seed), f"{arm}.json")
            if os.path.exists(path):
                done.append(f"{arm}:cached")
                continue
            rec = run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun)
            atomic_write_json(rec, path)
            done.append(f"{arm}:{rec['n_events']}")
        return dict(seed=seed, network_cache_hit=hit, arms=done)
    except Exception as exc:
        import traceback
        return dict(seed=seed, error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="one worker per network seed")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    recomputed = canonical_checksum(cfg)
    if recomputed != cfg["checksum"]:
        print(f"[stage1] config checksum mismatch: stored={cfg['checksum'][:12]} "
              f"recomputed={recomputed[:12]} -- config was edited after Stage 0")
        return 1

    probe = os.path.join(a.out, "stage1_variance_probe")
    per_run = os.path.join(probe, "per_run")
    os.makedirs(probe, exist_ok=True)

    rep = preflight(cfg)
    json.dump(dict(preflight=rep, provenance=provenance()),
              open(os.path.join(probe, "preflight.json"), "w"), indent=2, default=str)
    if not rep["ok"]:
        bad = [k for k, v in rep["checks"].items() if not v["ok"]]
        print(f"[stage1] PREFLIGHT FAILED on {bad}; worst budget error "
              f"{rep['worst_budget_error']:.2e}")
        print("[stage1] refusing to launch 96 simulations on a vacuous comparison")
        return 1
    print("[stage1] preflight OK:",
          {k: round(v["observed"], 3) for k, v in rep["checks"].items()}, flush=True)

    todo = [(s, cfg, os.path.join(a.out, "network_cache"), per_run) for s in cfg["seeds"]]
    print(f"[stage1] {len(todo)} seeds x {len(ARM_NAMES)} arms, {a.workers} workers",
          flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_seed_job, todo), 1):
            print(f"[stage1] {i}/{len(todo)} seed {r['seed']} "
                  f"{r.get('error') or r['arms']}", flush=True)
            if r.get("traceback"):
                print(r["traceback"], flush=True)

    got = {(int(d), f[:-5]) for d in os.listdir(per_run)
           for f in os.listdir(os.path.join(per_run, d)) if f.endswith(".json")}
    want = {(s, arm) for s in cfg["seeds"] for arm in ARM_NAMES}
    missing = want - got
    print(f"[stage1] {len(got)}/{len(want)} runs present")
    if missing:
        print(f"[stage1] MISSING {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
