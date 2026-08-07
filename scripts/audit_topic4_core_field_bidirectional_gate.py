"""Evaluate spec section 10.3's four-clause bidirectional sufficiency gate exactly.

The gate is pre-registered and must be read off the FROZEN scoring support, not
off whatever channel set a figure happens to display after its participation
filter. This consumes the spontaneous learned-field pool and reports each clause
separately, plus the held-out-set enlargement that made the first clause
evaluable at all.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_core_field_scoring import (load_patient_templates,  # noqa: E402
                                           model_templates, sim_matrix)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
POOL_GLOB = "readout_epilepsiae_1146_learned_core_field_pool_s*.json"

MIN_PER_DIRECTION = 10      # spec 10.3 clause 1
MIN_BIDIR_SEED_FRAC = 2 / 3  # spec 10.3 clause 1
MAX_FWD_REV_CORR = -0.2     # spec 10.3 clause 2


def _corr(m, support):
    common = sorted(set(m["forward"]) & set(m["reverse"]) & set(support))
    if len(common) < 4:
        return float("nan"), common
    return float(spearmanr([m["forward"][n] for n in common],
                           [m["reverse"][n] for n in common]).correlation), common


def _diag_ok(M):
    """Clause 3: under the best assignment both diagonals positive, both off negative."""
    M = np.asarray(M, float)
    if not np.isfinite(M).all():
        return False, None
    ident = M[0, 0] > 0 and M[1, 1] > 0 and M[0, 1] < 0 and M[1, 0] < 0
    swap = M[0, 1] > 0 and M[1, 0] > 0 and M[0, 0] < 0 and M[1, 1] < 0
    return bool(ident or swap), ("identity" if ident else "swapped" if swap else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    support, part_min = cfg["support"], cfg["part_min"]
    tgt = load_patient_templates(cfg["subject"], PRIMARY_KEY[0])
    rule = PRIMARY_KEY[1]

    files = sorted(glob.glob(os.path.join(RUN, POOL_GLOB)))
    if not files:
        raise SystemExit(f"no pool runs matching {POOL_GLOB}")
    pooled, per_seed = [], []
    for f in files:
        r = json.load(open(f))
        sel = [e for e in r["events"] if e.get("sign") is not None
               and e.get("n_part", 0) >= part_min]
        nf = sum(1 for e in sel if e["sign"] > 0)
        nr = sum(1 for e in sel if e["sign"] < 0)
        per_seed.append(dict(seed=r["seed"], n_forward=nf, n_reverse=nr,
                             bidirectional=bool(nf > 0 and nr > 0)))
        pooled.extend(sel)

    n_f = sum(1 for e in pooled if e["sign"] > 0)
    n_r = sum(1 for e in pooled if e["sign"] < 0)
    bidir_seeds = sum(1 for s in per_seed if s["bidirectional"])
    frac = bidir_seeds / len(per_seed)
    c1 = bool(n_f >= MIN_PER_DIRECTION and n_r >= MIN_PER_DIRECTION
              and frac >= MIN_BIDIR_SEED_FRAC)

    m = model_templates(pooled, support, part_min=part_min)
    fr, common = _corr(m, support)
    c2 = bool(np.isfinite(fr) and fr <= MAX_FWD_REV_CORR)

    M = sim_matrix(m, tgt, support, rule)
    c3, assignment = _diag_ok(M)

    # Clause 4: the verdict must survive u_C -> -u_C (which swaps the two
    # direction labels, i.e. swaps the rows of M) and the TA/TB relabel (which
    # swaps the columns). Both are re-evaluated with the same predicate.
    c4 = bool(_diag_ok(np.asarray(M)[::-1, :])[0]
              and _diag_ok(np.asarray(M)[:, ::-1])[0]
              and _diag_ok(np.asarray(M)[::-1, ::-1])[0])

    res = dict(
        gate="spec 10.3 held-out bidirectional sufficiency",
        pool=dict(
            n_seeds=len(per_seed), duration_ms_per_seed=cfg["duration_ms"],
            seeds=[s["seed"] for s in per_seed],
            preregistered_heldout_seeds=[9, 10, 11, 12],
            enlargement_reason=(
                "the four pre-registered held-out seeds yielded 4 reverse events "
                "in total, below clause 1's pre-registered minimum of 10, so the "
                "clause could not be evaluated at all. The pool was enlarged to "
                "120 fresh seeds (21-140), disjoint from the training seed and "
                "from the original held-out set. The criterion was NOT changed; "
                "only the sample backing it. This enlargement was decided after "
                "seeing that the original set was too small, and must be reported "
                "as such."),
        ),
        clause_1_support=dict(passed=c1, n_forward=n_f, n_reverse=n_r,
                              minimum=MIN_PER_DIRECTION,
                              bidirectional_seeds=bidir_seeds,
                              bidirectional_fraction=frac,
                              minimum_fraction=MIN_BIDIR_SEED_FRAC),
        clause_2_distinguishable=dict(passed=c2, forward_reverse_corr=fr,
                                      threshold=MAX_FWD_REV_CORR,
                                      n_common_contacts=len(common),
                                      contacts=common,
                                      scored_on="frozen SUPPORT"),
        clause_3_distinct_targets=dict(passed=bool(c3), assignment=assignment,
                                       matrix_rows_model_fr_cols_data_fr=M.tolist()),
        clause_4_swap_invariant=dict(passed=c4,
                                     checked="row flip, column flip, both"),
        gate_passed=bool(c1 and c2 and c3 and c4),
        does_not_overturn=(
            "spec 8.1 short-circuits in order; rule 1 (SIMULATOR_OVERFIT) fires "
            "before rule 2 consults this gate. The Stage 2 verdict stands. This "
            "gate is reported alongside it, not in place of it."),
        support=support, part_min=part_min,
        config_checksum=cfg["checksum"], provenance=provenance())
    atomic_write_json(res, os.path.join(a.out, "bidirectional_gate_pooled.json"))

    print(f"pool: {len(per_seed)} seeds x {cfg['duration_ms']:.0f} ms")
    print(f"  clause 1 support           : {c1}  ({n_f} fwd / {n_r} rev, "
          f"bidirectional {bidir_seeds}/{len(per_seed)} = {frac:.1%})")
    print(f"  clause 2 distinguishable   : {c2}  (rho = {fr:+.3f} on "
          f"{len(common)} frozen-support contacts, need <= {MAX_FWD_REV_CORR})")
    print(f"  clause 3 distinct targets  : {c3}  (assignment = {assignment})")
    print(f"      {np.round(M, 3).tolist()}")
    print(f"  clause 4 swap invariant    : {c4}")
    print(f"\n  GATE = {res['gate_passed']}")
    print(f"  {res['does_not_overturn']}")


if __name__ == "__main__":
    main()
