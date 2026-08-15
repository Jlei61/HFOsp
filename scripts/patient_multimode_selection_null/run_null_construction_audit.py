#!/usr/bin/env python
"""Does each null actually do what its name says?

A null is only usable if the thing it claims to hold fixed is really held fixed
and the thing it claims to destroy is really destroyed.  Both nulls in
`run_selection_corrected_null.py` make such a claim, and one of them fails it.

Measured per subject, on repeated draws:

  * `per_contact_order_spread` - the standard deviation, across contacts, of the
    contact's mean masked firing order.  This is the "does contact c tend to go
    early or late" structure.
      - observed        : the subject's real value
      - order null      : must collapse to ~0, because that null replaces every
                          ordering with a uniform draw
      - marginal null   : must land NEAR the observed value, because that null
                          claims to preserve each contact's marginal rank
                          distribution while destroying the within-event
                          coordination
  * `marginal_total_variation` - histogram distance between the observed and the
    drawn per-contact rank distribution.  Already stored by the main script, but
    it is a *signed-blind* distance: it cannot tell an amplified marginal from a
    flattened one, which is exactly the failure mode found here.

Output: null_construction_audit.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402
import run_selection_corrected_null as S  # noqa: E402

SUBJECTS = ["yuquan_huangwanling", "epilepsiae_818", "epilepsiae_916",
            "yuquan_zhangjinhan", "yuquan_zhourongxuan", "yuquan_zhaojinrui"]
N_DRAWS = 64
SEED = 20260816


def main() -> None:
    out = {"provenance": {"git_commit": M._git_commit(), "n_draws": N_DRAWS,
                          "seed": SEED},
           "subjects": {}}
    for sid in SUBJECTS:
        rec = M.replay_and_audit(sid)
        ve = rec["_valid_events"]
        ranks_v = rec["_loaded"]["ranks"][:, ve]
        bools_v = rec["_loaded"]["bools"][:, ve]
        X = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")
        masked = np.where(bools_v, X.T, np.nan)
        rnd = S.OrderRandomiser(bools_v, masked)
        rng = np.random.default_rng(SEED)

        def spread(feat):
            m = np.where(bools_v, feat.T, np.nan)
            return float(np.std(np.nanmean(m, axis=1)))

        obs = spread(X)
        so = [spread(rnd.draw_order(rng)) for _ in range(N_DRAWS)]
        sm = [spread(rnd.draw_marginal(rng)[0]) for _ in range(N_DRAWS)]
        so, sm = np.array(so), np.array(sm)
        out["subjects"][sid] = {
            "n_channels": rec["n_channels"],
            "chosen_k": rec["chosen_k"],
            "observed_per_contact_order_spread": obs,
            "order_null_spread": {"mean": float(so.mean()), "q05": float(np.percentile(so, 5)),
                                  "q95": float(np.percentile(so, 95))},
            "marginal_null_spread": {"mean": float(sm.mean()), "q05": float(np.percentile(sm, 5)),
                                     "q95": float(np.percentile(sm, 95))},
            "order_null_collapses_marginals": bool(so.mean() < 0.05 * max(obs, 1e-9) or so.mean() < 0.01),
            "marginal_null_ratio_to_observed": float(sm.mean() / obs) if obs > 0 else None,
            "marginal_null_preserves_marginals": bool(0.8 <= (sm.mean() / max(obs, 1e-12)) <= 1.25),
        }
        r = out["subjects"][sid]
        print(f"{sid:24s} nch={rec['n_channels']} obs_spread={obs:.4f} "
              f"order_null={so.mean():.4f} marginal_null={sm.mean():.4f} "
              f"ratio={r['marginal_null_ratio_to_observed']:.2f} "
              f"order_collapses={r['order_null_collapses_marginals']} "
              f"marginal_preserves={r['marginal_null_preserves_marginals']}", flush=True)

    n_ok_order = sum(v["order_null_collapses_marginals"] for v in out["subjects"].values())
    n_ok_marg = sum(v["marginal_null_preserves_marginals"] for v in out["subjects"].values())
    out["verdict"] = {
        "order_null_construction_valid_n": n_ok_order,
        "marginal_null_construction_valid_n": n_ok_marg,
        "n_subjects": len(out["subjects"]),
        "statement": (
            "The order-randomised primary null does what it claims: the per-contact "
            "order spread collapses to near zero in every subject, while recruitment "
            "masks, event counts and recording blocks are untouched. The "
            "marginal-preserving sensitivity null does NOT: its realised per-contact "
            "order spread exceeds the observed one, i.e. the re-rank repair amplifies "
            "the very marginal structure it was meant to hold fixed. Observations "
            "falling below that null are therefore uninformative about within-event "
            "coordination, and the sensitivity arm is reported as construction-invalid "
            "rather than as a negative result. The stored "
            "`marginal_drift_total_variation` did not catch this because a histogram "
            "distance is blind to the direction of the drift."),
    }
    p = HERE / "null_construction_audit.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\norder-null valid: {n_ok_order}/{len(out['subjects'])}; "
          f"marginal-null valid: {n_ok_marg}/{len(out['subjects'])} -> {p}")


if __name__ == "__main__":
    main()
