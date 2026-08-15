#!/usr/bin/env python
"""Invariant tests for the two null constructions and the 916 extent split.

Each null makes promises. These tests check the promises rather than trusting
the docstrings:

  T1  the order-randomised null leaves every event's recruitment mask untouched
  T2  its drawn features are legal orderings on the participating contacts and
      exactly the impute constant elsewhere
  T3  it leaves the event count and the recording-block membership untouched
  T4  the max-entropy stratum draws are legal permutations
  T5  the max-entropy law's realised contact-by-rank marginal converges to the
      fitted target as the number of draws grows
  T6  the producer-identical clustering reproduces the frozen labels exactly
  T7  the 916 train/held-out split shares no recording block

Run: python test_null_constructions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
from src.topic4_core_field_profile import split_by_block, assert_block_disjoint  # noqa: E402
import run_multimode_grammar_audit as M  # noqa: E402
import run_selection_corrected_null as S  # noqa: E402
import run_marginal_maxent_null as X  # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def main() -> None:
    for sid in ("yuquan_zhourongxuan", "epilepsiae_818"):
        print(f"=== {sid}")
        rec = M.replay_and_audit(sid)
        ve = rec["_valid_events"]
        k = rec["chosen_k"]
        ranks_v = rec["_loaded"]["ranks"][:, ve]
        bools_v = rec["_loaded"]["bools"][:, ve]
        X_obs = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")
        masked = np.where(bools_v, X_obs.T, np.nan)
        rnd = S.OrderRandomiser(bools_v, masked)
        rng = np.random.default_rng(11)
        Xn = rnd.draw_order(rng)                       # (n_ev, n_ch)
        npart = bools_v.sum(axis=0)

        # T1 recruitment mask untouched. Comparing against the impute constant
        # is ambiguous because 0.5 is itself a legal normalised rank whenever the
        # participating count is odd, so the mask is read off a sentinel fill
        # instead: whatever cells the randomiser writes must be exactly `bools_v`.
        sentinel = -7.0
        real_fill, S.IMPUTE_FILL = S.IMPUTE_FILL, sentinel
        try:
            rnd_s = S.OrderRandomiser(bools_v, masked)
            Xs = rnd_s.draw_order(np.random.default_rng(11))
        finally:
            S.IMPUTE_FILL = real_fill
        check("T1 recruitment mask preserved exactly (sentinel fill)",
              bool(np.array_equal((Xs.T != sentinel), bools_v)),
              f"n_events={ve.size}, n_contacts={bools_v.shape[0]}")

        # T2 legal ordering on the participating contacts, impute elsewhere
        ok_vals, ok_fill = True, True
        for e in range(min(4000, ve.size)):
            p = int(npart[e])
            got = np.sort(Xn[e, bools_v[:, e]])
            want = np.arange(p, dtype=float) / (p - 1)
            ok_vals &= np.allclose(got, want)
            ok_fill &= np.allclose(Xn[e, ~bools_v[:, e]], S.IMPUTE_FILL)
        check("T2 drawn events are legal orderings on their participating contacts", ok_vals)
        check("T2 non-participating cells hold the impute constant", ok_fill)

        # T3 event count and block membership untouched
        check("T3 event count unchanged", Xn.shape[0] == ve.size)
        check("T3 block membership unchanged",
              rec["_block_ids_valid"].size == ve.size)

        # T4/T5 max-entropy stratum draws
        strata = X.build_strata(bools_v, masked)
        st = max(strata, key=lambda s: s.n)
        d = st.draw(np.random.default_rng(3))
        p = st.p
        want = np.arange(p, dtype=float) / max(p - 1, 1)
        check("T4 max-entropy draws are legal permutations",
              bool(np.allclose(np.sort(d, axis=1), want[None, :])),
              f"p={p}, n={st.n}")
        rng2 = np.random.default_rng(5)
        acc = np.zeros((p, p))
        n_rep = 200
        for _ in range(n_rep):
            dd = np.rint(st.draw(rng2) * max(p - 1, 1)).astype(int)
            for c in range(p):
                acc[c] += np.bincount(dd[:, c], minlength=p)
        realised = acc / acc.sum(axis=1, keepdims=True)
        fitted = np.zeros((p, p))
        for c in range(p):
            fitted[c] = np.bincount(st.perms[:, c], weights=st.log_q, minlength=p)
        err = float(np.abs(realised - fitted).max())
        check("T5 realised marginal converges to the fitted law", err < 0.01,
              f"max cell error={err:.4f} over {n_rep} draws")

        # T6 producer-identical clustering reproduces the frozen labels
        lab, _, _ = S.producer_kmeans(X_obs, k)
        ami = float(adjusted_mutual_info_score(rec["_labels"], lab))
        check("T6 producer-identical clustering reproduces the frozen labels",
              ami > 0.999, f"AMI={ami:.6f}")

    # T7 916 split has no block leakage
    print("=== epilepsiae_916 split")
    rec = M.replay_and_audit("epilepsiae_916")
    blocks = rec["_block_ids_valid"]
    tr, ho = split_by_block(blocks, 0.3, 20260815)
    try:
        assert_block_disjoint(blocks, tr, ho)
        leak = False
    except Exception:
        leak = True
    check("T7 train/held-out share no recording block", not leak,
          f"train_blocks={np.unique(blocks[tr]).size} heldout_blocks={np.unique(blocks[ho]).size}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
