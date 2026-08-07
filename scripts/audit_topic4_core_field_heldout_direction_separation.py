"""Per-seed check of how distinguishable the learned field's two directions are.

The Stage 2 held-out gate pooled events from four different networks into one
template pair. Pooling across networks mixes their geometry, so a low
forward/reverse correlation there does not establish that the two directions
are alike on any single network. This re-derives the number per seed, and
reports the pooled value alongside for comparison.
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
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_core_field_scoring import (candidate_key,  # noqa: E402
                                           model_templates)

OUT = "results/topic4_sef_hfo/data_driven_core_field"


def _corr(m, part_min):
    common = sorted(set(m["forward"]) & set(m["reverse"]))
    if len(common) < 4:
        return float("nan"), len(common)
    return float(spearmanr([m["forward"][n] for n in common],
                           [m["reverse"][n] for n in common]).correlation), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(OUT, "config", "stage_config.json")))
    ck = json.load(open(os.path.join(OUT, "stage2_optimization", "checkpoint.json")))
    held = ck["heldout_seeds"]
    best = max(ck["history"], key=lambda x: candidate_key(x["n_dir"], x["S_rank"]))
    support, part_min = cfg["support"], cfg["part_min"]

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from scripts.run_topic4_core_field_stage2 import _evaluate
    jobs = [(best["theta"], sd, cfg, os.path.join(OUT, "network_cache")) for sd in held]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(_evaluate, jobs)

    rows, pooled = [], []
    for ev in res:
        if "error" in ev:
            rows.append(dict(seed=ev.get("seed"), error=ev["error"]))
            continue
        m = model_templates(ev["events"], support, part_min=part_min)
        c, n_common = _corr(m, part_min)
        nf = sum(1 for e in ev["events"] if e.get("sign") is not None
                 and e["sign"] > 0 and e.get("n_part", 0) >= part_min)
        nr = sum(1 for e in ev["events"] if e.get("sign") is not None
                 and e["sign"] < 0 and e.get("n_part", 0) >= part_min)
        rows.append(dict(seed=ev["seed"], n_forward=nf, n_reverse=nr,
                         forward_reverse_corr=c, n_common_contacts=n_common,
                         n_dir=m["n_dir"]))
        pooled.extend(ev["events"])

    pm = model_templates(pooled, support, part_min=part_min)
    pooled_c, pooled_n = _corr(pm, part_min)
    per_seed = [r["forward_reverse_corr"] for r in rows
                if np.isfinite(r.get("forward_reverse_corr", np.nan))]

    out = dict(
        held_out_seeds=held, theta=best["theta"], per_seed=rows,
        per_seed_median=float(np.median(per_seed)) if per_seed else float("nan"),
        pooled_across_networks=dict(forward_reverse_corr=pooled_c,
                                    n_common_contacts=pooled_n),
        note=("pooled concatenates events from four different networks into one "
              "template pair; per-seed values are the like-for-like comparison "
              "against a single-network hand-placed run"),
        config_checksum=cfg["checksum"], provenance=provenance())
    atomic_write_json(out, os.path.join(OUT, "stage2_optimization",
                                        "heldout_direction_separation.json"))

    print(f"{'seed':>5} {'fwd':>4} {'rev':>4} {'fwd-rev corr':>13} {'common':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['seed']!s:>5}  ERROR {r['error']}")
            continue
        print(f"{r['seed']:5d} {r['n_forward']:4d} {r['n_reverse']:4d} "
              f"{r['forward_reverse_corr']:13.3f} {r['n_common_contacts']:7d}")
    print(f"\nper-seed median      : {out['per_seed_median']:+.3f}")
    print(f"pooled across the 4  : {pooled_c:+.3f}  "
          f"({pooled_n} common contacts)")


if __name__ == "__main__":
    main()
