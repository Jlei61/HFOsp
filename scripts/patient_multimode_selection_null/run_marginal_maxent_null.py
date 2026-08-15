#!/usr/bin/env python
"""A contact-marginal-preserving sensitivity null that actually preserves them.

Why this file exists
--------------------
`run_selection_corrected_null.py` ships a `marginal` sensitivity null built by
shuffling each contact's rank values across events and then re-ranking each event
so it is a valid ordering again.  `run_null_construction_audit.py` shows that
repair step does not preserve the marginals, it AMPLIFIES them: the realised
per-contact order spread comes out 1.17x to 2.49x the observed one, valid in only
1 of 6 subjects.  An observation falling below such a null says nothing, so that
arm is reported as construction-invalid rather than as a negative result.

The construction used here
--------------------------
Every valid event has p participating contacts with p <= 6 in these subjects, so
the p! <= 720 possible orderings can simply be ENUMERATED.  Within each
participation-set stratum:

  1. Read the observed contact-by-rank count matrix N (rows = contacts of the
     stratum, columns = rank slots).  Its row and column sums both equal the
     number of events, so N/n is doubly stochastic and IS the per-contact
     marginal rank distribution we must keep.
  2. Fit the maximum-entropy distribution over the p! orderings whose induced
     contact-by-rank marginal equals N/n.  That distribution has the log-linear
     form q(pi) proportional to prod_c A[c, pi(c)], and A is fitted by iterative
     proportional fitting over the enumerated orderings.
  3. Draw the stratum's events i.i.d. from q.

By construction this keeps every contact's marginal rank distribution (up to
multinomial sampling noise) and removes ALL higher-order structure, because the
events are independent draws from a product-form law.  That is exactly
"destroy the within-event cross-contact coordination, keep the contact
marginals", and unlike the shuffle-and-repair version it can be checked: the
realised marginal error is measured and reported for every draw.

Everything else is identical to the primary null: recruitment masks, event
counts, recording blocks, contact order and K are untouched, and each draw is
re-clustered with the producer-identical estimator at the frozen K.

SAFETY: reads only; never writes `src/` or `scripts/` (the concurrently running
formal cohort worker hashes its imported modules against commit 96618174).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import permutations
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

SEED = 20260816
IPF_ITERS = 4000
IPF_TOL = 1e-10


class MaxEntStratum:
    """Enumerated orderings of one participation set + the fitted max-ent law."""

    def __init__(self, contacts: np.ndarray, events: np.ndarray, ranks: np.ndarray):
        self.contacts = contacts
        self.events = events
        p = contacts.size
        self.p = p
        n = events.size
        self.n = n
        self.perms = np.array(list(permutations(range(p))), dtype=np.int8)  # (p!, p)
        # observed contact-by-rank counts
        counts = np.zeros((p, p))
        for j in range(p):
            counts[j] = np.bincount(ranks[:, j].astype(int), minlength=p)
        self.target = counts / float(n)                      # doubly stochastic
        self.log_q, self.marginal_error = self._fit()

    def _fit(self):
        """Log-space iterative proportional fitting of q(pi) ~ prod_c A[c, pi(c)].

        Log space, plus an explicit -inf for structural zeros of the target, is
        what keeps this stable: a contact that never occupies a given rank slot
        must give zero weight to every ordering that would put it there, and in
        linear space the product of six such small factors underflows to zero
        for the whole enumeration, which is how the first version produced a
        NaN probability vector on the six-contact subject.
        """
        p, perms = self.p, self.perms
        T = np.maximum(self.target, 0.0)
        idx_c = np.arange(p)[None, :].repeat(perms.shape[0], axis=0)
        with np.errstate(divide="ignore"):
            logA = np.where(T > 0, np.log(np.maximum(T, 1e-300)), -np.inf)
        best_logA, best_err = logA.copy(), np.inf

        def _q(logA_):
            lw = logA_[idx_c, perms].sum(axis=1)
            fin = np.isfinite(lw)
            if not fin.any():
                return None
            w = np.where(fin, np.exp(lw - lw[fin].max()), 0.0)
            tot = w.sum()
            if not np.isfinite(tot) or tot <= 0:
                return None
            return w / tot

        for _ in range(IPF_ITERS):
            q = _q(logA)
            if q is None:
                break
            marg = np.zeros((p, p))
            for c in range(p):
                marg[c] = np.bincount(perms[:, c], weights=q, minlength=p)
            err = float(np.abs(marg - T).max())
            if err < best_err:
                best_err, best_logA = err, logA.copy()
            if err < IPF_TOL:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                adj = np.where((marg > 0) & (T > 0),
                               np.log(np.maximum(T, 1e-300))
                               - np.log(np.maximum(marg, 1e-300)), 0.0)
            adj = np.clip(np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
            logA = logA + adj
            fin = np.isfinite(logA)
            if fin.any():
                logA = logA - logA[fin].max()

        q = _q(best_logA)
        self.fit_degenerate = q is None
        if q is None:                      # last resort: uniform on the support
            q = np.full(perms.shape[0], 1.0 / perms.shape[0])
            best_err = float("nan")
        q = np.nan_to_num(q, nan=0.0)
        if q.sum() <= 0:
            q = np.full(perms.shape[0], 1.0 / perms.shape[0])
            self.fit_degenerate = True
        q = q / q.sum()
        return q, float(best_err)

    def draw(self, rng) -> np.ndarray:
        """(n, p) normalised ranks drawn i.i.d. from the fitted law."""
        pick = rng.choice(self.log_q.size, size=self.n, p=self.log_q)
        return self.perms[pick].astype(float) / max(self.p - 1, 1)


def build_strata(bools_v, masked_v):
    codes = np.packbits(bools_v.T.astype(np.uint8), axis=1)
    _, inv = np.unique(codes, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    npart = bools_v.sum(axis=0).astype(int)
    strata = []
    for s in np.unique(inv):
        ev = np.flatnonzero(inv == s)
        p = int(npart[ev[0]])
        if p < 2:
            continue
        contacts = np.flatnonzero(bools_v[:, ev[0]])
        vals = masked_v[np.ix_(contacts, ev)].T                    # (n_s, p) in [0,1]
        ranks = np.rint(vals * (p - 1)) if p > 1 else np.zeros_like(vals)
        strata.append(MaxEntStratum(contacts, ev, ranks))
    return strata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--n-perm", type=int, default=512)
    ap.add_argument("--out-dir", default=str(HERE / "marginal_maxent_null"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rec = M.replay_and_audit(args.subject)
    k, ve = rec["chosen_k"], rec["_valid_events"]
    ranks_v = rec["_loaded"]["ranks"][:, ve]
    bools_v = rec["_loaded"]["bools"][:, ve]
    n_ch, n_ev = bools_v.shape
    X_obs = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")
    masked_v = np.where(bools_v, X_obs.T, np.nan)
    _, order_id = np.unique(np.round(X_obs, 9), axis=0, return_inverse=True)
    order_id = np.asarray(order_id).ravel()
    observed = S.ordering_stats(rec["_labels"], order_id, k)
    obs_spread = float(np.std(np.nanmean(np.where(bools_v, X_obs.T, np.nan), axis=1)))

    strata = build_strata(bools_v, masked_v)
    errs = np.array([st.marginal_error for st in strata], float)
    sizes = np.array([st.n for st in strata], float)
    fit_err = float(np.nanmax(errs))
    fit_err_weighted = float(np.nansum(errs * sizes) / sizes.sum())
    bad = errs > 0.05
    fit_bad_strata = int(bad.sum())
    fit_bad_event_fraction = float(sizes[bad].sum() / sizes.sum())
    n_degenerate = int(sum(getattr(st, "fit_degenerate", False) for st in strata))

    key = int.from_bytes(hashlib.sha256(args.subject.encode()).digest()[:4], "big") % 10_000
    rng = np.random.default_rng(SEED + key + 7)
    draws, spreads = [], []
    for _ in range(args.n_perm):
        Xn = np.full((n_ch, n_ev), S.IMPUTE_FILL)
        for st in strata:
            Xn[np.ix_(st.contacts, st.events)] = st.draw(rng).T
        Xn = Xn.T
        lab, _, _ = S.producer_kmeans(Xn, k)
        if np.unique(lab).size != k:
            continue
        _, oid = np.unique(np.round(Xn, 9), axis=0, return_inverse=True)
        draws.append(S.ordering_stats(lab, np.asarray(oid).ravel(), k))
        spreads.append(float(np.std(np.nanmean(np.where(bools_v, Xn.T, np.nan), axis=1))))

    conc = np.array([d["equal_mode_weighted_dominant_order_concentration"] for d in draws])
    effc = np.array([d["equal_mode_weighted_effective_ordering_count"] for d in draws])
    spreads = np.array(spreads)
    obs_c = observed["equal_mode_weighted_dominant_order_concentration"]
    ratio = float(spreads.mean() / obs_spread) if obs_spread > 0 else None
    payload = {
        "provenance": {
            "git_commit": M._git_commit(), "seed": SEED,
            "seed_derivation": "SEED + sha256(subject_id)[:4] % 10000 + 7",
            "n_perm_requested": args.n_perm,
            "construction": "enumerated p! orderings; max-entropy law matching the "
                            "observed contact-by-rank marginal; events drawn i.i.d.",
            "ipf_max_marginal_error": fit_err,
            "ipf_event_weighted_marginal_error": fit_err_weighted,
            "ipf_n_strata_error_gt_0p05": fit_bad_strata,
            "ipf_event_fraction_in_those_strata": fit_bad_event_fraction,
            "n_degenerate_strata_fallback_uniform": n_degenerate,
            "ipf_note": ("a per-cell marginal error is reported per stratum; the max is "
                         "dominated by small strata with structural zeros, so the "
                         "event-weighted mean and the event fraction sitting in strata "
                         "above 0.05 are the numbers to read"),
            "kmeans": "producer-identical `_kmeans_stability_for_k` at frozen k",
            "safety": "reads only; never writes src/ or scripts/",
        },
        "subject_id": args.subject, "chosen_k": k, "n_channels": n_ch,
        "n_valid_events": int(ve.size), "n_strata": len(strata),
        "observed": observed,
        "construction_check": {
            "observed_per_contact_order_spread": obs_spread,
            "null_per_contact_order_spread_mean": float(spreads.mean()),
            "null_over_observed_spread_ratio": ratio,
            "preserves_marginals": bool(ratio is not None and 0.8 <= ratio <= 1.25),
        },
        "n_valid_randomisations": len(draws),
        "concentration_null": S._mc(conc),
        "effective_ordering_count_null": S._mc(effc),
        "observed_minus_null_concentration": float(obs_c - conc.mean()) if conc.size else None,
        "observed_exceeds_null_q95_concentration": (
            bool(obs_c > np.percentile(conc, 95)) if conc.size else None),
        "empirical_p_concentration_ge": (
            float((np.sum(conc >= obs_c) + 1) / (conc.size + 1)) if conc.size else None),
        "draws_concentration": conc.tolist(),
        "draws_effective_ordering_count": effc.tolist(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out = out_dir / f"{args.subject}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    cc = payload["construction_check"]
    print(f"{args.subject}: K={k} strata={len(strata)} degen={n_degenerate} ipf_err_max={fit_err:.2e} ipf_err_wt={fit_err_weighted:.2e} bad_strata={fit_bad_strata}({fit_bad_event_fraction*100:.2f}% events) | "
          f"obs={obs_c:.4f} null={conc.mean():.4f} q95={np.percentile(conc,95):.4f} "
          f"delta={payload['observed_minus_null_concentration']:+.4f} "
          f"p={payload['empirical_p_concentration_ge']:.4f} | "
          f"spread obs={cc['observed_per_contact_order_spread']:.4f} "
          f"null={cc['null_per_contact_order_spread_mean']:.4f} "
          f"ratio={cc['null_over_observed_spread_ratio']:.2f} "
          f"valid={cc['preserves_marginals']} ({payload['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
