"""Leg A: sweep a fixed-budget heterogeneity across the sheet (spec section 9; plan Task 4).

No optimiser is involved, so nothing here can overfit one -- but picking the
best of ninety-eight cells at four seeds each is still a winner's curse, so the
map is reported descriptively and its high-scoring region is re-run on
independent seeds.

Every probe goes through the same budget projection, so all cells carry exactly
the same number of pathological cells: the map compares position and size, not
dose.
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
from src.topic4_core_field import project_to_budget  # noqa: E402
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import (_placement, atomic_write_json,  # noqa: E402
                                          canonical_checksum, get_network,
                                          provenance)
from src.topic4_core_field_scoring import (assignment_invariant_S,  # noqa: E402
                                           load_patient_templates,
                                           model_templates,
                                           recruited_per_direction, sim_matrix)
from src.topic4_core_field_stage3 import probe_q  # noqa: E402

STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"

# --- frozen sweep design (spec section 9.4; plan Task 4) --------------------
GRID_N = 7                       # 7 x 7 centres
GRID_LO, GRID_HI = 2.0, 18.0     # sheet interior, CENTER_MARGIN_MM from each edge
SIGMA_PRIMARY = 1.2              # the pre-registered main map
SIGMA_SENSITIVITY = 2.4          # reported separately; never max'd per cell
SWEEP_SEEDS = (201, 202, 203, 204)
CONFIRM_SEEDS = (301, 302, 303, 304, 305, 306)
TOP_FRAC, MIN_VALID = 0.10, 3


def _load_cmrun():
    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep_config(stage2_cfg):
    xs = np.linspace(GRID_LO, GRID_HI, GRID_N)
    return dict(
        stage="stage3_legA_sweep",
        subject=stage2_cfg["subject"], support=stage2_cfg["support"],
        part_min=stage2_cfg["part_min"], sources=stage2_cfg["sources"],
        missing_rules=stage2_cfg["missing_rules"],
        N_core_manual=stage2_cfg["N_core_manual"],
        quantile_seed=stage2_cfg["quantile_seed"],
        duration_ms=stage2_cfg["duration_ms"], engine=stage2_cfg["engine"],
        grid=dict(n=GRID_N, lo=GRID_LO, hi=GRID_HI,
                  centers=[[float(x), float(y)] for y in xs for x in xs]),
        sigmas=dict(primary=SIGMA_PRIMARY, sensitivity=SIGMA_SENSITIVITY),
        sweep_seeds=list(SWEEP_SEEDS), confirm_seeds=list(CONFIRM_SEEDS),
        region=dict(top_frac=TOP_FRAC, min_valid=MIN_VALID),
        stage2_checksum=stage2_cfg["checksum"],
        note=("all probes share one budget projection, so every cell has the "
              "same pathological-cell count; the map compares position and "
              "size, not dose"))


def _evaluate(job):
    cx, cy, sigma, seed, cfg, cache_dir = job
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

        h, _ = project_to_budget(probe_q(posE, (cx, cy), sigma),
                                 float(cfg["N_core_manual"]))
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
        axis_u = reg["axis_unit_vec"]

        events = []
        for ev in detect_events(af, bin_w, event_on_frac=bar):
            rd = cmrun.read_event(env_f, fdt, msheet, valid,
                                  (ev["t_on"], ev["t_off"]), axis_u,
                                  k_dir=k_dir, part_min=2 * k_dir + 1)
            events.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                               n_part=rd["n_part"], sign=rd["sign"],
                               readability=rd["readability"], ranks=rd["ranks"]))
        return dict(cx=cx, cy=cy, sigma=sigma, seed=seed, events=events)
    except Exception as exc:                                # noqa: BLE001
        return dict(cx=cx, cy=cy, sigma=sigma, seed=seed, error=repr(exc))


def _score(events, cfg, tgt, rule):
    part_min = cfg["part_min"]
    m = model_templates(events, cfg["support"], part_min=part_min)
    S = assignment_invariant_S(sim_matrix(m, tgt, cfg["support"], rule))
    fwd, rev = recruited_per_direction(events, cfg["support"], part_min)
    n_f = sum(1 for e in events if e.get("sign") is not None
              and e["sign"] > 0 and e.get("n_part", 0) >= part_min)
    n_r = sum(1 for e in events if e.get("sign") is not None
              and e["sign"] < 0 and e.get("n_part", 0) >= part_min)
    return dict(n_events=len(events), n_dir=m["n_dir"], n_forward=n_f,
                n_reverse=n_r, recruited_forward=fwd, recruited_reverse=rev,
                recruited_min=int(min(fwd, rev)),
                S_rank=None if not np.isfinite(S) else float(S))


def _cell_path(out, sigma, i, seed):
    d = os.path.join(out, "cells", f"sigma{sigma:g}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"c{i:03d}_s{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=None,
                    help="omit to run primary then sensitivity")
    ap.add_argument("--seeds", default=None, help="comma list; default = sweep seeds")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg_path = os.path.join(a.out, "config", "sweep_config.json")
    stage2_cfg = json.load(open(os.path.join(STAGE2, "config", "stage_config.json")))
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        if canonical_checksum({k: v for k, v in cfg.items() if k != "checksum"}) != cfg["checksum"]:
            raise SystemExit("sweep config checksum mismatch")
    else:
        cfg = sweep_config(stage2_cfg)
        cfg["checksum"] = canonical_checksum(cfg)
        cfg["provenance"] = provenance()
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        atomic_write_json(cfg, cfg_path)
        print(f"[config] frozen -> {cfg_path}  checksum={cfg['checksum']}")

    sigmas = [a.sigma] if a.sigma else [cfg["sigmas"]["primary"],
                                        cfg["sigmas"]["sensitivity"]]
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds
             else list(cfg["sweep_seeds"]))
    centers = cfg["grid"]["centers"]
    cache = os.path.join(STAGE2, "network_cache")

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    for sigma in sigmas:
        jobs = [(c[0], c[1], float(sigma), sd, cfg, cache)
                for i, c in enumerate(centers) for sd in seeds
                if not os.path.exists(_cell_path(a.out, sigma, i, sd))]
        done = len(centers) * len(seeds) - len(jobs)
        print(f"[sigma={sigma:g}] {len(jobs)} to run, {done} already on disk",
              flush=True)
        if not jobs:
            continue
        index = {(c[0], c[1]): i for i, c in enumerate(centers)}
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for n, r in enumerate(pool.imap_unordered(_evaluate, jobs), 1):
                i = index[(r["cx"], r["cy"])]
                atomic_write_json(r, _cell_path(a.out, sigma, i, r["seed"]))
                if n % 10 == 0 or n == len(jobs):
                    print(f"[sigma={sigma:g}] {n}/{len(jobs)}", flush=True)
    print("[sweep] done", flush=True)


if __name__ == "__main__":
    main()
