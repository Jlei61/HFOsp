"""Stage 2 held-out assessment: the half the optimisation driver does not do.

Takes the best training candidates, re-runs them on networks the optimiser never
saw, builds the equivalent-optimum family, and applies the outcome taxonomy that
was frozen before Stage 2 started.

Baselines on the same held-out seeds already exist from Stage 1 and are reused
rather than re-simulated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json, canonical_checksum, provenance  # noqa: E402
from src.topic4_core_field_scoring import (  # noqa: E402
    assignment_invariant_S, candidate_key, coverage_matched_axis_only,
    load_patient_templates, model_templates, sim_matrix)
from src.topic4_core_field_stage2_outcome import (  # noqa: E402
    classify_stage2, equivalent_optimum_family)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
TOP_K = 6
BIDIR_MIN_EVENTS = 10        # spec 10.3
FWD_REV_MAX_CORR = -0.2      # spec 10.3: the two model templates must be distinguishable


def _axial_projection(subject):
    fd = np.load(os.path.join(
        RUN, f"figdata_{subject}_gradient_shared_corefrozen_cr1p5_s5_20260722.npz"),
        allow_pickle=True)
    reg = fd["reg"].item()
    u = np.asarray(reg["axis_unit"], float); u = u / np.linalg.norm(u)
    proj = (np.asarray(fd["contacts"], float)
            - np.asarray(reg["center"], float)[None, :]) @ u
    return {str(n): float(p) for n, p in zip([str(x) for x in fd["names"]], proj)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    if canonical_checksum(cfg) != cfg["checksum"]:
        raise SystemExit("config checksum mismatch")
    support, part_min = cfg["support"], cfg["part_min"]
    targets = {s: load_patient_templates(cfg["subject"], s) for s in cfg["sources"]}
    tgt, rule = targets[PRIMARY_KEY[0]], PRIMARY_KEY[1]
    proj = _axial_projection(cfg["subject"])

    s2 = os.path.join(a.out, "stage2_optimization")
    ck = json.load(open(os.path.join(s2, "checkpoint.json")))
    held = ck["heldout_seeds"]
    hist = ck["history"]

    ranked = sorted(hist, key=lambda r: candidate_key(r["n_dir"], r["S_rank"]), reverse=True)
    seen, top = set(), []
    for r in ranked:
        key = tuple(np.round(r["theta"], 6))
        if key in seen:
            continue
        seen.add(key); top.append(r)
        if len(top) >= a.top_k:
            break
    print(f"[heldout] {len(top)} distinct candidates x {len(held)} held-out seeds "
          f"= {len(top)*len(held)} sims", flush=True)

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from scripts.run_topic4_core_field_stage2 import _evaluate
    jobs = [(r["theta"], sd, cfg, os.path.join(a.out, "network_cache"))
            for r in top for sd in held]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(_evaluate, jobs)

    # ---- score the learned candidates on held-out --------------------------
    cand = []
    for i, r in enumerate(top):
        evs = res[i * len(held):(i + 1) * len(held)]
        per_seed, deltas, hfield, pooled = {}, [], None, []
        for ev in evs:
            if "error" in ev:
                continue
            hfield = np.asarray(ev["h"], float)
            m = model_templates(ev["events"], support, part_min=part_min)
            S = assignment_invariant_S(sim_matrix(m, tgt, support, rule))
            ao = coverage_matched_axis_only(m, proj, support=support)
            g = (assignment_invariant_S(sim_matrix(ao, tgt, support, rule))
                 if ao is not None else float("nan"))
            per_seed[ev["seed"]] = dict(n_dir=m["n_dir"], S=S, geom=g,
                                        cov=min(m["coverage_forward"], m["coverage_reverse"]))
            if np.isfinite(S) and np.isfinite(g):
                deltas.append(S - g)
            pooled.extend(ev["events"])
        Ss = [v["S"] for v in per_seed.values() if np.isfinite(v["S"])]
        cand.append(dict(
            rank=i, theta=r["theta"], train_S=r["S_rank"], train_n_dir=r["n_dir"],
            train_vs_geom=r["vs_matched_axis_only"], per_seed=per_seed,
            heldout_S=float(np.mean(Ss)) if Ss else float("nan"),
            heldout_vs_geom=float(np.mean(deltas)) if deltas else float("nan"),
            bidir_seeds=sum(1 for v in per_seed.values() if v["n_dir"] == 2),
            coverage=float(np.mean([v["cov"] for v in per_seed.values()])) if per_seed else 0.0,
            h=hfield.tolist() if hfield is not None else [],
            pooled_events=pooled))

    # ---- Stage 1 baselines on the SAME held-out seeds ----------------------
    def stage1_arm(arm):
        vals, covs = [], []
        for sd in held:
            rec = json.load(open(os.path.join(
                a.out, "stage1_variance_probe", "per_run", str(sd), f"{arm}.json")))
            m = model_templates(rec["events"], support, part_min=part_min)
            vals.append(assignment_invariant_S(sim_matrix(m, tgt, support, rule)))
            covs.append(min(m["coverage_forward"], m["coverage_reverse"]))
        return vals, float(np.mean(covs))

    base = {arm: stage1_arm(arm) for arm in
            ("manual_smooth", "manual_projected", "uniform_axial")}

    best = max(cand, key=lambda c: candidate_key(
        2 if c["bidir_seeds"] == len(held) else c["bidir_seeds"],
        c["heldout_S"] if np.isfinite(c["heldout_S"]) else -np.inf))

    def paired(learned_per_seed, other_vals):
        d = []
        for sd, ov in zip(held, other_vals):
            lv = learned_per_seed.get(sd, {}).get("S", float("nan"))
            if np.isfinite(lv) and np.isfinite(ov):
                d.append(lv - ov)
        return dict(mean=float(np.mean(d)) if d else float("nan"),
                    n_above=int(sum(x > 0 for x in d)), n=len(d))

    vs_geom = [v["S"] - v["geom"] for v in best["per_seed"].values()
               if np.isfinite(v["S"]) and np.isfinite(v["geom"])]

    # ---- bidirectional sufficiency, pooled over held-out (spec 10.3) -------
    pm = model_templates(best["pooled_events"], support, part_min=part_min)
    n_f = sum(1 for e in best["pooled_events"]
              if e.get("sign") is not None and e["sign"] > 0 and e.get("n_part", 0) >= part_min)
    n_r = sum(1 for e in best["pooled_events"]
              if e.get("sign") is not None and e["sign"] < 0 and e.get("n_part", 0) >= part_min)
    common = sorted(set(pm["forward"]) & set(pm["reverse"]))
    fr = (float(spearmanr([pm["forward"][n] for n in common],
                          [pm["reverse"][n] for n in common]).correlation)
          if len(common) >= 4 else float("nan"))
    M = sim_matrix(pm, tgt, support, rule)
    diag_ok = bool(np.isfinite(M).all() and
                   ((M[0, 0] > 0 and M[1, 1] > 0 and M[0, 1] < 0 and M[1, 0] < 0) or
                    (M[0, 1] > 0 and M[1, 0] > 0 and M[0, 0] < 0 and M[1, 1] < 0)))
    bidir = dict(passed=bool(n_f >= BIDIR_MIN_EVENTS and n_r >= BIDIR_MIN_EVENTS
                             and np.isfinite(fr) and fr <= FWD_REV_MAX_CORR and diag_ok),
                 n_forward=n_f, n_reverse=n_r, forward_reverse_corr=fr,
                 matrix_ok=diag_ok, min_events=BIDIR_MIN_EVENTS)

    scores = [c["heldout_S"] for c in cand]
    fields = [c["h"] for c in cand]
    sd_pair = float(np.std(vs_geom, ddof=1)) if len(vs_geom) > 1 else 0.05
    fam = equivalent_optimum_family(scores, fields, paired_sd=sd_pair)

    res_in = dict(
        integrity_ok=all(np.isfinite(c["heldout_S"]) for c in cand[:1]),
        train_delta=float(best["train_vs_geom"]) if np.isfinite(best["train_vs_geom"]) else 0.0,
        heldout_delta=float(np.mean(vs_geom)) if vs_geom else float("nan"),
        vs_axis_only=dict(mean=float(np.mean(vs_geom)) if vs_geom else float("nan"),
                          n_above=int(sum(x > 0 for x in vs_geom)), n=len(vs_geom)),
        vs_uniform=paired(best["per_seed"], base["uniform_axial"][0]),
        vs_manual_projected=paired(best["per_seed"], base["manual_projected"][0]),
        bidirectional_gate=bidir,
        family=dict(median_field_corr=fam["median_field_corr"], n_members=fam["n_members"]),
        restart_field_corr_median=float("nan"),   # single restart -- NOT measured
        coverage=dict(learned=best["coverage"], manual_smooth=base["manual_smooth"][1],
                      margin=0.10),
    )
    verdict = classify_stage2(res_in)

    out = dict(
        verdict=verdict, inputs=res_in, family=fam,
        caveats=[
            "single CMA-ES restart: the UNIDENTIFIABLE branch (cross-restart field "
            "stability) was NOT evaluated and its check passed vacuously",
            "training used ONE seed, so the per-generation best is a max over 10 noisy "
            "draws and is upward biased",
            "Stage 1's frozen shape criterion returned 'does not separate', so this "
            "optimisation ran on the reduced budget",
        ],
        candidates=[{k: v for k, v in c.items() if k not in ("h", "pooled_events")}
                    for c in cand],
        baselines={a_: dict(heldout_S=[float(x) for x in v[0]], coverage=v[1])
                   for a_, v in base.items()},
        held_out_seeds=held, top_k=len(cand),
        config_checksum=cfg["checksum"], provenance=provenance())
    atomic_write_json(out, os.path.join(s2, "stage2_heldout.json"))

    print(f"\n[heldout] VERDICT = {verdict['outcome']}")
    print(f"          {verdict['allowed_statement']}")
    print(f"\n  best candidate: held-out S={best['heldout_S']:.3f} "
          f"vs geometry {res_in['heldout_delta']:+.3f} "
          f"(above in {res_in['vs_axis_only']['n_above']}/{res_in['vs_axis_only']['n']}), "
          f"bidirectional {best['bidir_seeds']}/{len(held)}")
    print(f"  vs uniform corridor : {res_in['vs_uniform']['mean']:+.3f} "
          f"({res_in['vs_uniform']['n_above']}/{res_in['vs_uniform']['n']})")
    print(f"  vs manual projected : {res_in['vs_manual_projected']['mean']:+.3f} "
          f"({res_in['vs_manual_projected']['n_above']}/{res_in['vs_manual_projected']['n']})")
    print(f"  bidirectional gate  : {bidir['passed']} "
          f"(fwd={n_f} rev={n_r} need>={BIDIR_MIN_EVENTS}, fwd-rev corr={fr:.3f})")
    print(f"  equivalent-optimum family: {fam['n_members']} members, "
          f"median field corr {fam['median_field_corr']:.3f}")
    print("\n  CAVEATS:")
    for c in out["caveats"]:
        print(f"   - {c}")


if __name__ == "__main__":
    main()
