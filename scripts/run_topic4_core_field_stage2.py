"""Stage 2: optimise the axial pathology field against contact-rank match.

Unattended contract:
  - common random numbers WITHIN a generation (same networks, same noise seeds),
    so candidates differ only by their field
  - checkpoint after every generation; re-running resumes from it
  - hard wall-clock cap; on expiry it stops cleanly and writes its products
  - training seeds and held-out seeds are disjoint, and the optimiser never sees
    the held-out ones

Candidates are ranked lexicographically by (n_dir, S_rank): S_rank may never be
compared across direction tiers (spec 5.3).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import axis_coords, manual_mask  # noqa: E402
from src.topic4_core_field_cmaes import CMAES  # noqa: E402
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    _placement, atomic_write_json, canonical_checksum, get_network, provenance)
from src.topic4_core_field_scoring import (  # noqa: E402
    assignment_invariant_S, candidate_key, coverage_matched_axis_only,
    load_patient_templates, model_templates, sim_matrix)
from src.topic4_core_field_stage2 import N_PARAMS, params_to_h, uniform_theta  # noqa: E402

OUT = "results/topic4_sef_hfo/data_driven_core_field"
TRAIN_SEEDS_FULL = (1, 2, 3, 4)
HELDOUT_SEEDS = (9, 10, 11, 12)      # never seen by the optimiser
SIGMA0 = 1.0


def _load_cmrun():
    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _budget_tier(out_dir, forced=None):
    """Seeds per candidate and evaluation ceiling, from the Stage 1 report.

    The user's standing instruction: if Stage 1 shows the read-out cannot tell
    field shapes apart, still run, but on the reduced budget.
    """
    if forced:
        return dict(seeds_per_candidate=1, max_evals=150, tier=f"forced:{forced}")
    path = os.path.join(out_dir, "stage1_variance_probe", "stage1_report.json")
    if not os.path.exists(path):
        return dict(seeds_per_candidate=1, max_evals=150, tier="no_stage1_report")
    rep = json.load(open(path))
    if rep.get("integrity", {}).get("status") != "ok":
        return dict(seeds_per_candidate=1, max_evals=150, tier="stage1_integrity_failed")
    if not rep.get("recommendation", {}).get("shape_separates", False):
        return dict(seeds_per_candidate=1, max_evals=150, tier="shape_does_not_separate")
    conc = rep.get("concordance", {}).get("|".join(PRIMARY_KEY))
    conc = float(conc) if conc is not None else 0.0
    if conc >= 0.85:
        return dict(seeds_per_candidate=1, max_evals=400, tier=f"concordance={conc:.2f}")
    if conc >= 0.65:
        return dict(seeds_per_candidate=4, max_evals=150, tier=f"concordance={conc:.2f}")
    return dict(seeds_per_candidate=1, max_evals=150, tier=f"low_concordance={conc:.2f}")


def _evaluate(args):
    """One candidate on one seed. Returns the event table, or an error record."""
    theta, seed, cfg, cache_dir = args
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    try:
        from kick_probe import simulate_kick
        from lfp import LFPRecorder
        from params import Params
        from src.sef_hfo_events import detect_events
        from src.sef_hfo_snn_adapter import snn_event_envelope
        from src.topic4_core_field import (
            build_vth, core_thresholds, sample_core_quantiles, signed_depth)

        cmrun = _load_cmrun()
        e = cfg["engine"]
        k_dir = int(e["k_dir"])
        cmrun.KDIR, cmrun.PART_MIN = k_dir, 2 * k_dir + 1
        reg = _placement(cfg)
        p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
                   dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
        net, NE, NI, _ = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
        posE = net["pos"][:NE]
        s, r = axis_coords(posE, reg["center"], reg["axis_unit_vec"])
        geom = dict(sep=float(np.linalg.norm(reg["sink_centroid"] - reg["source_centroid"])),
                    s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                               float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                    M=cfg["field"]["M"], sigma_perp=e["core_r"],
                    shift_mm=cfg["field"]["SHIFT_MM"])
        h = params_to_h(np.asarray(theta, float), s, r, geom, float(cfg["N_core_manual"]))
        d = signed_depth(core_thresholds(
            sample_core_quantiles(NE, cfg["quantile_seed"]), e["core_mean"], e["core_std"]),
            e["v_base"])
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
        events = detect_events(af, bin_w, event_on_frac=bar)
        env_f, fdt, _ = snn_event_envelope(spk, posE, msheet, e["dt"])
        recs = []
        for ev in events:
            rd = cmrun.read_event(env_f, fdt, msheet, valid, (ev["t_on"], ev["t_off"]),
                                  reg["axis_unit_vec"], k_dir=k_dir, part_min=2 * k_dir + 1)
            recs.append(dict(n_part=int(rd["n_part"]), sign=rd["sign"], ranks=rd["ranks"]))
        return dict(seed=int(seed), events=recs, h=h.astype(np.float32).tolist(),
                    n_events=len(recs))
    except Exception as exc:
        import traceback
        return dict(seed=int(seed), error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-1200:])


def _score(evals, support, targets, proj, part_min):
    """Pool a candidate's seeds into one lexicographic key plus diagnostics."""
    keys, srs, covs, deltas = [], [], [], []
    for ev in evals:
        if "error" in ev:
            continue
        m = model_templates(ev["events"], support, part_min=part_min)
        S = assignment_invariant_S(
            sim_matrix(m, targets[PRIMARY_KEY[0]], support, PRIMARY_KEY[1]))
        keys.append(candidate_key(m["n_dir"], S))
        srs.append(S)
        covs.append(min(m["coverage_forward"], m["coverage_reverse"]))
        ao = coverage_matched_axis_only(m, proj, support=support)
        if ao is not None:
            s_ao = assignment_invariant_S(
                sim_matrix(ao, targets[PRIMARY_KEY[0]], support, PRIMARY_KEY[1]))
            if np.isfinite(S) and np.isfinite(s_ao):
                deltas.append(S - s_ao)
    if not keys:
        return (0, -np.inf), dict(n_dir=0, S_rank=float("nan"), coverage=0.0,
                                  vs_matched_axis_only=float("nan"), n_ok=0)
    n_dir = int(np.min([k[0] for k in keys]))       # a candidate is as good as its worst seed
    finite = [v for v in srs if np.isfinite(v)]
    S = float(np.mean(finite)) if finite else float("nan")
    return candidate_key(n_dir, S), dict(
        n_dir=n_dir, S_rank=S, coverage=float(np.mean(covs)) if covs else 0.0,
        vs_matched_axis_only=float(np.mean(deltas)) if deltas else float("nan"),
        n_ok=len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hours", type=float, default=8.0, help="hard wall-clock cap")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force-tier", default=None)
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.hours * 3600.0

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    if canonical_checksum(cfg) != cfg["checksum"]:
        print("[stage2] config checksum mismatch; refusing to run")
        return 1
    support = cfg["support"]
    targets = {s: load_patient_templates(cfg["subject"], s) for s in cfg["sources"]}
    part_min = cfg["part_min"]

    fd = np.load(os.path.join(
        "results/topic4_sef_hfo/field_swap_subject_snn",
        f"figdata_{cfg['subject']}_gradient_shared_corefrozen_cr1p5_s5_20260722.npz"),
        allow_pickle=True)
    reg0 = fd["reg"].item()
    u = np.asarray(reg0["axis_unit"], float); u = u / np.linalg.norm(u)
    proj = {str(n): float((np.asarray(fd["contacts"], float)[i]
                           - np.asarray(reg0["center"], float)) @ u)
            for i, n in enumerate([str(x) for x in fd["names"]])}

    tier = _budget_tier(a.out, a.force_tier)
    train_seeds = list(TRAIN_SEEDS_FULL[:tier["seeds_per_candidate"]])
    stage2 = os.path.join(a.out, "stage2_optimization")
    os.makedirs(stage2, exist_ok=True)
    ckpt_path = os.path.join(stage2, "checkpoint.json")
    M = cfg["field"]["M"]

    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        es = CMAES.from_state(ck["cmaes"])
        history = ck["history"]
        print(f"[stage2] resumed at generation {es.generation}, {len(history)} evals so far",
              flush=True)
    else:
        es = CMAES(uniform_theta(M), sigma0=SIGMA0, seed=20260806)
        history = []
        print(f"[stage2] fresh start; tier={tier}", flush=True)

    print(f"[stage2] dim={N_PARAMS(M)} popsize={es.popsize} train_seeds={train_seeds} "
          f"heldout={list(HELDOUT_SEEDS)} max_evals={tier['max_evals']} cap={a.hours}h",
          flush=True)

    cache_dir = os.path.join(a.out, "network_cache")
    stop = None
    with Pool(a.workers, maxtasksperchild=1) as pool:
        while True:
            if len(history) >= tier["max_evals"]:
                stop = "evaluation budget reached"; break
            if time.time() >= deadline:
                stop = "wall-clock cap reached"; break
            xs = es.ask()
            jobs = [(x, sd, cfg, cache_dir) for x in xs for sd in train_seeds]
            results = pool.map(_evaluate, jobs)
            keys, gen_rows = [], []
            for i, x in enumerate(xs):
                evals = results[i * len(train_seeds):(i + 1) * len(train_seeds)]
                k, diag = _score(evals, support, targets, proj, part_min)
                keys.append(k)
                row = dict(generation=es.generation, theta=list(map(float, x)), **diag)
                gen_rows.append(row); history.append(row)
            es.tell(xs, keys)
            best = max(gen_rows, key=lambda r: candidate_key(r["n_dir"], r["S_rank"]))
            print(f"[stage2] gen {es.generation:3d} evals={len(history):4d} "
                  f"best n_dir={best['n_dir']} S={best['S_rank']:.3f} "
                  f"vs_geom={best['vs_matched_axis_only']:+.3f} sigma={es.sigma:.3f} "
                  f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)
            atomic_write_json(dict(cmaes=es.get_state(), history=history, tier=tier,
                                   train_seeds=train_seeds,
                                   heldout_seeds=list(HELDOUT_SEEDS),
                                   config_checksum=cfg["checksum"],
                                   provenance=provenance()), ckpt_path)

    print(f"[stage2] stopped: {stop} after {len(history)} evals, "
          f"{(time.time()-t_start)/3600:.2f}h", flush=True)
    atomic_write_json(dict(stop_reason=stop, n_evals=len(history), tier=tier,
                           train_seeds=train_seeds, heldout_seeds=list(HELDOUT_SEEDS),
                           elapsed_hours=(time.time() - t_start) / 3600.0,
                           config_checksum=cfg["checksum"], provenance=provenance()),
                      os.path.join(stage2, "run_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
